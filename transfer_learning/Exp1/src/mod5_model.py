import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from loguru import logger

# 设置日志
logger.add("training.log", rotation="10MB")


# 数据集定义
class FaceToSketchDataset(Dataset):
    def __init__(self, origin_dir, target_dir, transform=None):
        self.origin_files = sorted(os.listdir(origin_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.transform = transform
        assert len(self.origin_files) == len(self.target_files), "数据不匹配！"

    def __len__(self):
        return len(self.origin_files)

    def __getitem__(self, idx):
        origin_path = os.path.join(self.origin_dir, self.origin_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        origin_img = Image.open(origin_path).convert("L")
        target_img = Image.open(target_path).convert("L")

        if self.transform:
            origin_img = self.transform(origin_img)
            target_img = self.transform(target_img)

        return origin_img, target_img


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 80)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 目录路径
dataset_path = "../dataset/CUFSF/train"
train_dataset = FaceToSketchDataset(os.path.join(dataset_path, "origin"), os.path.join(dataset_path, "target"),
                                    transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 生成器（U-Net）
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UNet().to(device)
discriminator = Discriminator().to(device)
criterion_gan = nn.BCELoss()
criterion_l1 = nn.L1Loss()
optimizer_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

num_epochs = 100
logger.info("开始训练...")
for epoch in range(num_epochs):
    for i, (origin, target) in enumerate(train_loader):
        origin, target = origin.to(device), target.to(device)
        real_labels = torch.ones(origin.size(0), 1, 30, 38).to(device)
        fake_labels = torch.zeros(origin.size(0), 1, 30, 38).to(device)

        # 训练生成器
        optimizer_g.zero_grad()
        fake_target = generator(origin)
        d_output = discriminator(torch.cat((origin, fake_target), 1))
        loss_g = criterion_gan(d_output, real_labels) + 100 * criterion_l1(fake_target, target)
        loss_g.backward()
        optimizer_g.step()

        # 训练判别器
        optimizer_d.zero_grad()
        d_real = discriminator(torch.cat((origin, target), 1))
        d_fake = discriminator(torch.cat((origin, fake_target.detach()), 1))
        loss_d = (criterion_gan(d_real, real_labels) + criterion_gan(d_fake, fake_labels)) / 2
        loss_d.backward()
        optimizer_d.step()

    logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss_G: {loss_g.item():.4f}, Loss_D: {loss_d.item():.4f}")
logger.info("训练完成！")

# 保存模型
torch.save(generator.state_dict(), "face_to_sketch_gan.pth")
