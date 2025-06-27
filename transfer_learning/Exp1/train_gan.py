import datetime
import itertools
import random
import time

import cv2
import torch.optim as optim
from loguru import logger
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid

import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from pathlib import Path


dataset_root = Path("dataset")
log_dir = Path("logs")
log_dir = log_dir / "T-{}".format(int(time.time()))
checkpoint_dir = log_dir / "checkpoints"
pred_dir = log_dir / "preds"

model_config = {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'layer_idxs': [2, 5, 8, 11]}

H = 280
W = 280


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionDepthBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class FeatureFusionControlBlock(FeatureFusionBlock):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super.__init__(features, activation, deconv,
                       bn, expand, align_corners, size)
        self.copy_block = FeatureFusionBlock(
            features, activation, deconv, bn, expand, align_corners, size)

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeatureFusionDepthBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionDepthBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit_depth = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, stride=1,
                      padding=1, bias=True, groups=1),
            activation,
            nn.Conv2d(features, features, kernel_size=3,
                      stride=1, padding=1, bias=True, groups=1),
            activation,
            zero_module(
                nn.Conv2d(features, features, kernel_size=3,
                          stride=1, padding=1, bias=True, groups=1)
            )
        )
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, prompt_depth=None, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if prompt_depth is not None:
            prompt_depth = F.interpolate(
                prompt_depth, output.shape[2:], mode='bilinear', align_corners=False)
            res = self.resConfUnit_depth(prompt_depth)
            output = self.skip_add.add(output, res)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
class DPTHead(nn.Module):
    def __init__(self,
                 nclass,
                 in_channels,
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 output_act='sigmoid'):
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(
            features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        act_func = nn.Sigmoid() if output_act == 'sigmoid' else nn.Identity()

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass,
                          kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1,
                          stride=1, padding=0),
                act_func,
            )

    def forward(self, out_features, patch_h, patch_w, prompt_depth=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(
            layer_4_rn, size=layer_3_rn.shape[2:], prompt_depth=prompt_depth)
        path_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:], prompt_depth=prompt_depth)
        path_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:], prompt_depth=prompt_depth)
        path_1 = self.scratch.refinenet1(
            path_2, layer_1_rn, prompt_depth=prompt_depth)
        out = self.scratch.output_conv1(path_1)
        out_feat = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out_feat)
        return out

# logger.add(sys.stdout, level="INFO", colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>")


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


class GeneratorTransformer(nn.Module):
    """ CycleGAN 生成器 """

    def __init__(self, input_shape):
        super(GeneratorTransformer, self).__init__()
        channels = input_shape[0]  # 3 通道
        out_features = 64
        self.dino_v2 = torch.hub.load(
            f'F:\\WorkSpace\\py\\Experiment\\tl\\Exp1\\dinov2',
            'dinov2_vitb14',
            source='local',
            pretrained=False)
        self.register_buffer('_mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        dim = self.dino_v2.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(nclass=1,
                                  in_channels=dim,
                                  features=model_config['features'],
                                  out_channels=model_config['out_channels'],
                                  use_bn=False,
                                  use_clstoken=False,
                                  output_act="Sigmoid")

        self.patch_size = 14

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.dino_v2.get_intermediate_layers(
            (x - self._mean) / self._std, model_config['layer_idxs'],
            return_class_token=True)
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        depth = self.depth_head(features, patch_h, patch_w, None)
        return depth


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

    def __init__(self, origin_dir, target_dir=None):
        """
        origin_dir: 域 A 的图片文件夹
        target_dir: 域 B 的图片文件夹
        """
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.image_list = os.listdir(origin_dir)

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def _transform(img_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (W, H))
        image = np.expand_dims(image, axis=0)
        # image = np.repeat(image, 3, axis=0)  # 灰度图转 RGB
        image = image / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).float()

        return image

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:

        origin_path = self.origin_dir / self.image_list[idx]
        origin_img = self._transform(str(origin_path))

        if self.target_dir:
            target_path = self.target_dir / self.image_list[idx]
            target_img = self._transform(str(target_path))
            return {"A": origin_img, "B": target_img}
        else:
            return {"A": origin_img}


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
    batches_done = str(batches_done).zfill(6)
    save_image(image_grid, pred_dir / f"{batches_done}.png", normalize=False)


def train_cycleGAN(dataset_name="CUFSF", pretrained=None):
    if pretrained:
        pretrained = Path(pretrained)
    global dataset_root, log_dir, pred_dir, checkpoint_dir

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
    train_A = dataset_root / "train/origin"
    train_B = dataset_root / "train/target"
    test_A = dataset_root / "test/origin"
    test_B = dataset_root / "test/target"

    ############## 准备数据集 ##############
    dataset_train = PortraitDataset(train_A, train_B)
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    # 若需要验证集，可自己拆分或使用另一对文件夹
    dataset_val = PortraitDataset(test_A, test_B)
    val_dataloader = DataLoader(dataset_val, batch_size=5, shuffle=True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############## 初始化模型 ##############
    input_shape = (1, H, W)  # 三通道模型
    # G_AB = GeneratorResNet(input_shape, num_residual_blocks=9).to(device)
    # G_BA = GeneratorResNet(input_shape, num_residual_blocks=9).to(device)
    G_AB = GeneratorTransformer(input_shape).to(device)
    G_BA = GeneratorTransformer(input_shape).to(device)
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)
    first_in = 0
    if pretrained:
        log_dir = log_dir.parent / pretrained
        f_in_list = log_dir / "checkpoints"
        pred_dir = log_dir / "preds"
        checkpoint_dir = log_dir / "checkpoints"
        first_in = len(os.listdir(f_in_list)) // 4
        max_iter = first_in - 1
        G_AB.load_state_dict(torch.load(f_in_list / f"G_AB_{max_iter}.pth"))
        G_BA.load_state_dict(torch.load(f_in_list / f"G_BA_{max_iter}.pth"))
        D_A.load_state_dict(torch.load(f_in_list / f"D_A_{max_iter}.pth"))
        D_B.load_state_dict(torch.load(f_in_list / f"D_B_{max_iter}.pth"))
    else:
        # 如果不加载预训练，就初始化
        # G_AB.apply(weights_init_normal)
        # G_BA.apply(weights_init_normal)
        # D_A.apply(weights_init_normal)
        # D_B.apply(weights_init_normal)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

    logger.add(log_dir / "train.log", rotation="10 MB",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
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
        if epoch < first_in:
            continue
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
            info = ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G "
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




def psnr_ssim(path):
    # 读取拼接图像
    grid_image_path = Path(path)
    if not grid_image_path.exists():
        raise FileNotFoundError(f"未找到图片 {grid_image_path}")

    grid_img = cv2.imread(str(grid_image_path))

    # 获取图像尺寸
    h, w, _ = grid_img.shape

    # 去除网格Grid，网格为4行5列，每个图片padding为2
    real_A = grid_img[2:h // 4 - 2, 2:w // 5 - 2]
    fake_A = grid_img[h // 4 + 2:h // 2 - 2, 2:w // 5 - 2]
    real_B = grid_img[h // 2 + 2:3 * h // 4 - 2, 2:w // 5 - 2]
    fake_B = grid_img[3 * h // 4 + 2:h - 2, 2:w // 5 - 2]

    # 转换为灰度图用于 SSIM 计算
    real_B_gray = cv2.cvtColor(real_B, cv2.COLOR_BGR2GRAY)
    fake_B_gray = cv2.cvtColor(fake_B, cv2.COLOR_BGR2GRAY)
    real_A_gray = cv2.cvtColor(real_A, cv2.COLOR_BGR2GRAY)
    fake_A_gray = cv2.cvtColor(fake_A, cv2.COLOR_BGR2GRAY)

    # 计算 PSNR
    psnr_B = psnr(real_B, fake_B, data_range=255)
    psnr_A = psnr(real_A, fake_A, data_range=255)

    # 计算 SSIM
    ssim_B = ssim(real_B_gray, fake_B_gray, data_range=255)
    ssim_A = ssim(real_A_gray, fake_A_gray, data_range=255)

    # 记录结果
    results_df = pd.DataFrame({
        "Comparison": ["fake_B vs real_B", "fake_A vs real_A"],
        "PSNR": [psnr_B, psnr_A],
        "SSIM": [ssim_B, ssim_A]
    })

    # 打印结果
    print(results_df)

    # 可选：保存结果到 CSV 文件
    results_df.to_csv("psnr_ssim_results.csv", index=False)


if __name__ == "__main__":
    train_cycleGAN("FS2K")
