import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch.optim as optim
from PIL import Image
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model.gan import *

dataset_sel = "CUFS"
dataset_root = Path(f"../dataset")
checkpoint_dir = Path("../checkpoints")
log_dir = Path("../logs")
dataset_dir = dataset_root / dataset_sel
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
# 配置日志输出
logger.add(log_dir / "Training-{}.log".format(int(time.time())), rotation="10 MB",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# 超参数设置
BATCH_SIZE = 32
EPOCHS = 200
LR = 0.0002
LAMBDA_L1 = 100  # L1损失的权重


# 自定义数据集
class PortraitDataset(Dataset):
    def __init__(self, origin_dir, target_dir):
        self.origin_dir = str(origin_dir)
        self.target_dir = str(target_dir)
        self.image_list = os.listdir(origin_dir)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((200, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 单通道灰度图
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        origin_path = os.path.join(self.origin_dir, self.image_list[idx])
        target_path = os.path.join(self.target_dir, self.image_list[idx])

        origin_img = Image.open(origin_path).convert('L')  # 转为灰度
        target_img = Image.open(target_path).convert('L')

        origin_tensor = self.transform(origin_img)
        target_tensor = self.transform(target_img)

        return origin_tensor, target_tensor




# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generatorAB = Generator().to(device)
generatorBA = Generator().to(device)
discriminator = Discriminator().to(device)

# 优化器
opt_g_ab = optim.Adam(generatorAB.parameters(), lr=LR, betas=(0.5, 0.999))
opt_g_ba = optim.Adam(generatorBA.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# 损失函数
criterion_gan = nn.BCELoss()
criterion_l1 = nn.L1Loss()

# 数据加载
dataset = PortraitDataset(dataset_dir / 'train' / 'origin', dataset_dir / 'train' / 'target')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train():
    # 训练循环
    for epoch in range(EPOCHS):
        for i, (origin, target) in enumerate(dataloader):
            origin = origin.to(device)
            target = target.to(device)

            # 训练判别器
            opt_d.zero_grad()

            # 真实样本
            real_output = discriminator(origin, target)
            real_loss = criterion_gan(real_output, torch.ones_like(real_output))

            # 生成样本
            fake_ab = generatorAB(origin)
            fake_output = discriminator(origin, fake_ab.detach())
            fake_loss = criterion_gan(fake_output, torch.zeros_like(fake_output))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()

            # 训练生成器
            opt_g_ab.zero_grad()
            opt_g_ba.zero_grad()

            # GAN损失
            fake_output = discriminator(origin, fake_ab)
            g_loss_gan = criterion_gan(fake_output, torch.ones_like(fake_output))
            # L1损失
            g_loss_l1 = criterion_l1(fake_ab, target) * LAMBDA_L1

            g_total_loss = g_loss_gan + g_loss_l1
            g_total_loss.backward()
            opt_g_ab.step()

            # 日志记录
            if (i + 1) == len(dataloader):
                name = str(epoch + 1).zfill(3)
                logger.info(
                    f"Epoch [{name}/{EPOCHS}] "
                    f"D_loss: {d_loss.item():.4f} G_loss: {g_total_loss.item():.4f} "
                    f"GAN: {g_loss_gan.item():.4f} L1: {g_loss_l1.item():.4f}"
                )

        # 保存模型
        if (epoch + 1) % 10 == 0:
            name = str(epoch + 1).zfill(3)
            torch.save(generatorAB.state_dict(), checkpoint_dir / f"G_{name}.pth")
            torch.save(discriminator.state_dict(), checkpoint_dir / f"D_{name}.pth")
            test(epoch + 1)

    logger.success("Training completed!")


# 效果可视化
def test(name=200):
    name = "G_" + str(name).zfill(3)
    generatorAB.load_state_dict(torch.load(checkpoint_dir / f"{name}.pth"))
    generatorAB.eval()

    # 预处理输入
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((200, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dsd = dataset_dir / 'test'
    for i, f in enumerate(os.listdir(dsd / 'origin')):
        path_to_image = dsd/ 'origin' / f
        img = Image.open(path_to_image)
        origin_img = transform(img).unsqueeze(0).to(device)

        # 生成
        with torch.no_grad():
            output = generatorAB(origin_img)
            output = output.squeeze(0).squeeze(0).cpu().numpy()
            output = (output + 1) / 2 * 255  # 反归一化
            output = output.astype(np.uint8)

        # 保存结果
        out_path = str(dsd / 'target' / f"{name}_{f}")
        cv2.imwrite(out_path, output)


# train()
test(200)
