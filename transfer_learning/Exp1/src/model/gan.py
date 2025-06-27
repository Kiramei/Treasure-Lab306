# 生成器网络（U-Net结构）
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # 下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # 128x100
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),  # 64x50
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),  # 32x25
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),  # 16x13
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # 残差块（增强特征提取）
        self.res_blocks = nn.Sequential(
            *[ResBlock(512) for _ in range(6)]
        )

        # 上采样
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, output_padding=(1, 0)),  # 32x25
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),  # 64x50
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),  # 128x100
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 256x200
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        # 输出层
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)  # 64 @ 128x100
        d2 = self.down2(d1)  # 128 @ 64x50
        d3 = self.down3(d2)  # 256 @ 32x25
        d4 = self.down4(d3)  # 512 @ 16x13

        # 残差块处理
        bn = self.res_blocks(d4)

        # 上采样并拼接跳跃连接
        u1 = self.up1(bn)  # 256 @ 32x25
        u1 = torch.cat([u1, d3], 1)  # 512 @ 32x25
        u2 = self.up2(u1)  # 128 @ 64x50
        u2 = torch.cat([u2, d2], 1)  # 256 @ 64x50
        u3 = self.up3(u2)  # 64 @ 128x100
        u3 = torch.cat([u3, d1], 1)  # 128 @ 128x100
        u4 = self.up4(u3)  # 64 @ 256x200

        return self.final(u4)



# 判别器网络（PatchGAN）
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),  # 输入是origin+target拼接
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),  # 输出是16x20的patch
            nn.Sigmoid()
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)
