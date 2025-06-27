import datetime
import itertools
import os
import random
import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.nn as nn
from loguru import logger
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid

log_dir = Path("logs")
dataset_root = Path("dataset")

os.makedirs(log_dir, exist_ok=True)
log_dir = log_dir / "T-{}".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)
checkpoint_dir = log_dir / "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
pred_dir = log_dir / "preds"
os.makedirs(pred_dir, exist_ok=True)

logger.add(log_dir / "train.log", rotation="10 MB",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


###############################################
#               models.py (保留自代码段二)     #
###############################################
def weights_init_normal(m):
    """ 权重初始化函数 """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    """ 残差块 """

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """ CycleGAN 生成器 """

    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]  # 3 通道
        out_features = 64

        model = [
            nn.ReflectionPad2d(5),  # 这里为 1，则在HW两侧各 pad 1
            nn.Conv2d(channels, out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # 下采样 2 次
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 残差块
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # 上采样 2 次
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # 输出层
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(in_features, channels, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """ CycleGAN 判别器（PatchGAN） """

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            _layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                _layers.append(nn.InstanceNorm2d(out_filters))
            _layers.append(nn.LeakyReLU(0.2, inplace=True))
            return _layers

        layers = []
        layers += discriminator_block(channels, 64, normalize=False)
        layers += discriminator_block(64, 128)
        layers += discriminator_block(128, 256)
        layers += discriminator_block(256, 512)
        layers += [nn.ZeroPad2d((1, 0, 1, 0)),
                   nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


###############################################
#               utils.py (保留自代码段二)      #
###############################################
class ReplayBuffer:
    """ 生成器生成的假样本缓存，以防止判别器过拟合 """

    def __init__(self, max_size=50):
        assert max_size > 0, "ReplayBuffer 的 max_size 必须大于 0"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    """ 学习率调整策略 """

    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
                self.n_epochs - self.decay_start_epoch
        )


###############################################
#        下面开始修正的训练脚本 (参考段一)      #
###############################################
class PortraitDataset(Dataset):
    """
    简单示例：原本灰度图改为强制转成 RGB，以适配代码段二的三通道模型
    """

    def __init__(self, origin_dir, target_dir):
        """
        origin_dir: 域 A 的图片文件夹
        target_dir: 域 B 的图片文件夹
        """
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.image_list = os.listdir(origin_dir)

    def __len__(self):
        return len(self.image_list)

    """
        transform_ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])
    """

    @staticmethod
    def _transform(img_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        # image = np.repeat(image, 3, axis=0)  # 灰度图转 RGB
        image = image / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).float()

        return image

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        origin_path = self.origin_dir / self.image_list[idx]
        target_path = self.target_dir / self.image_list[idx]

        origin_img = self._transform(str(origin_path))
        target_img = self._transform(str(target_path))

        # CycleGAN 要求返回 dict, 分别是 {"A", "B"}
        return {"A": origin_img, "B": target_img}


def sample_images(val_dataloader, batches_done, G_AB, G_BA, device, out_dir="images"):
    """
    在验证集上采样并保存对比图，方便可视化
    """
    try:
        batch = next(iter(val_dataloader))
    except StopIteration:
        return

    G_AB.eval()
    G_BA.eval()

    real_A = batch["A"].to(device)
    real_B = batch["B"].to(device)

    # 生成结果
    with torch.no_grad():
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

    # 拼图与保存
    real_A_grid = make_grid(real_A, nrow=5, normalize=True)
    fake_B_grid = make_grid(fake_B, nrow=5, normalize=True)
    real_B_grid = make_grid(real_B, nrow=5, normalize=True)
    fake_A_grid = make_grid(fake_A, nrow=5, normalize=True)

    image_grid = torch.cat((real_A_grid, fake_B_grid, real_B_grid, fake_A_grid), 1)
    os.makedirs(out_dir, exist_ok=True)
    save_image(image_grid, pred_dir / f"{batches_done}.png", normalize=False)


def train_cycleGAN(dataset_name="CUFSF"):
    global dataset_root

    ############## 部分可调参数 ##############
    # 训练轮数、衰减起始轮次
    n_epochs = 10
    decay_epoch = 5
    batch_size = 2
    lr = 0.0002

    # CycleGAN 损失比重
    lambda_cyc = 10.0
    lambda_id = 5.0

    dataset_root = dataset_root / dataset_name
    train_A = os.path.join(dataset_root, "train/origin")
    train_B = os.path.join(dataset_root, "train/target")
    test_A = os.path.join(dataset_root, "test/origin")
    test_B = os.path.join(dataset_root, "test/target")

    ############## 准备数据集 ##############
    dataset_train = PortraitDataset(train_A, train_B)
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    # 若需要验证集，可自己拆分或使用另一对文件夹
    dataset_val = PortraitDataset(test_A, test_B)
    val_dataloader = DataLoader(dataset_val, batch_size=5, shuffle=True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############## 初始化模型 ##############
    input_shape = (1, 256, 256)  # 三通道模型
    G_AB = GeneratorResNet(input_shape, num_residual_blocks=9).to(device)
    G_BA = GeneratorResNet(input_shape, num_residual_blocks=9).to(device)
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)

    # 如果不加载预训练，就初始化
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    ############## 优化器 & 学习率调度 ##############
    optimizer_G = optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
    )
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
    )
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
    )

    ############## 损失函数 ##############
    criterion_GAN = nn.MSELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)

    ############## 生成器输出缓存 ##############
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    ############## 训练循环 ##############
    prev_time = time.time()
    for epoch in range(n_epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(device)  # 域 A
            real_B = batch["B"].to(device)  # 域 B

            # 真实/假的标签
            valid = torch.ones((real_A.size(0), *D_A.output_shape), device=device, requires_grad=False)
            fake_ = torch.zeros((real_A.size(0), *D_A.output_shape), device=device, requires_grad=False)

            ##########  训练生成器 G_AB 和 G_BA  ##########
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()

            # ---- Identity loss ----
            # 让 G_BA(A) ≈ A，  G_AB(B) ≈ B
            _ = G_BA(real_A)
            loss_id_A = criterion_identity(_, real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # ---- GAN loss ----
            fake_B = G_AB(real_A)  # A -> B
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)

            fake_A = G_BA(real_B)  # B -> A
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN_ = (loss_GAN_AB + loss_GAN_BA) / 2

            # ---- Cycle loss (A->B->A, B->A->B) ----
            recov_A = G_BA(fake_B)  # A->B->A
            loss_cycle_A = criterion_cycle(recov_A, real_A)

            recov_B = G_AB(fake_A)  # B->A->B
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle_ = (loss_cycle_A + loss_cycle_B) / 2

            # ---- 总生成器损失 ----
            loss_G = loss_GAN_ + lambda_cyc * loss_cycle_ + lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

            ##########  训练判别器 D_A  ##########
            optimizer_D_A.zero_grad()

            # 判别器应该判断真实的 A 为真
            loss_real_A = criterion_GAN(D_A(real_A), valid)

            # 判别器应该判断生成的 A'(fake_A) 为假
            # 从 buffer 里取假样本，可提升稳定性
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake_A = criterion_GAN(D_A(fake_A_.detach()), fake_)

            loss_D_A = (loss_real_A + loss_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            ##########  训练判别器 D_B  ##########
            optimizer_D_B.zero_grad()

            # 判别器应该判断真实的 B 为真
            loss_real_B = criterion_GAN(D_B(real_B), valid)

            # 判别器应该判断生成的 B'(fake_B) 为假
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake_B = criterion_GAN(D_B(fake_B_.detach()), fake_)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # 估计剩余时间
            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )
            prev_time = time.time()

            # 打印日志
            info = ("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G "
                    "loss: %f, adv: %f, cycle: %f, id: %f] ETA: %s") % (
                       epoch, n_epochs,
                       i, len(dataloader),
                       loss_D.item(),
                       loss_G.item(),
                       loss_GAN_.item(),
                       loss_cycle_.item(),
                       loss_identity.item(),
                       time_left,
                   )

            sys.stdout.write(info)
            logger.info(info)

            # 可视化
            if batches_done % 100 == 0:
                sample_images(val_dataloader, batches_done, G_AB, G_BA, device)

        # 学习率更新
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # 每轮结束保存一次模型

        torch.save(G_AB.state_dict(), checkpoint_dir / f"G_AB_{epoch}.pth")
        torch.save(G_BA.state_dict(), checkpoint_dir / f"G_BA_{epoch}.pth")
        torch.save(D_A.state_dict(), checkpoint_dir / f"D_A_{epoch}.pth")
        torch.save(D_B.state_dict(), checkpoint_dir / f"D_B_{epoch}.pth")

    logger.success("训练完成！")


if __name__ == "__main__":
    train_cycleGAN()
