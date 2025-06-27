import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 轻量配置
class LiteConfig:
    seed = 42
    image_size = (64, 80)  # 保持目标尺寸
    batch_size = 8  # 更小的批次
    lr = 2e-4  # 更低学习率
    epochs = 100  # 更少训练轮次
    latent_size = (16, 20)  # 潜在空间压缩
    num_timesteps = 500  # 减少扩散步数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./lite_model.pth"


config = LiteConfig()
logger.add("lite_train.log")


# 极简数据集
class LiteDataset(Dataset):
    def __init__(self):
        self.portraits = sorted(Path("../dataset/CUFSF/train/origin").glob("*.png"))
        self.sketches = sorted(Path("../dataset/CUFSF/train/target").glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return min(len(self.portraits), len(self.sketches))

    def __getitem__(self, idx):
        p = self.transform(Image.open(self.portraits[idx])).squeeze(0)
        s = self.transform(Image.open(self.sketches[idx])).squeeze(0)
        return p, s


# 微型条件UNet
class MicroConditionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 条件编码器
        self.cond_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # [B,16,64,80]
            nn.MaxPool2d(4),  # [B,16,16,20]
            nn.Conv2d(16, 32, 3, padding=1),  # [B,32,16,20]
            nn.AdaptiveAvgPool2d(config.latent_size)  # [B,32,16,20]
        )

        # 主网络
        self.time_embed = nn.Embedding(config.num_timesteps, 32)
        self.main_net = nn.Sequential(
            nn.Conv2d(1 + 32, 64, 3, padding=1),  # 输入通道：图像+条件+时间
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t, cond):
        # 条件处理 [B,1,64,80] -> [B,32,16,20]
        cond_feat = self.cond_net(cond.unsqueeze(1))

        # 时间嵌入 [B] -> [B,32,16,20]
        t_emb = self.time_embed(t).view(-1, 32, 1, 1).expand(-1, -1, *config.latent_size)

        # 特征融合
        x = torch.cat([
            x.unsqueeze(1),  # 噪声图像 [B,1,64,80]
            F.interpolate(cond_feat, scale_factor=4, mode="bilinear"),  # 上采样条件
            F.interpolate(t_emb, scale_factor=4, mode="nearest")  # 时间特征
        ], dim=1)

        return self.main_net(x).squeeze(1)


# 精简扩散流程
class LiteDiffusion:
    def __init__(self, model):
        self.model = model
        self.beta = torch.linspace(1e-4, 0.02, config.num_timesteps).to(config.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def train_step(self, clean, cond):
        noise = torch.randn_like(clean)
        t = torch.randint(0, config.num_timesteps, (clean.size(0),), device=config.device)
        alpha_t = self.alpha_bar[t][:,  None, None]

        # 前向加噪
        noisy = torch.sqrt(alpha_t) * clean + torch.sqrt(1 - alpha_t) * noise

        # 预测噪声
        pred_noise = self.model(noisy, t, cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def generate(self, cond, steps=100):
        self.model.eval()
        x = torch.randn_like(cond)

        for t in reversed(range(0, config.num_timesteps, config.num_timesteps // steps)):
            alpha_t = self.alpha_bar[t]
            pred_noise = self.model(x, torch.tensor([t] * len(x))).to(config.device)

            x = (x - (1 - alpha_t) / torch.sqrt(1 - self.alpha_bar[t]) * pred_noise) / torch.sqrt(alpha_t)
            if t > 0:
                x += torch.sqrt(self.beta[t]) * torch.randn_like(x)

        return torch.clamp(x, -1, 1)


# 训练流程
def lite_train():
    # 初始化
    torch.manual_seed(config.seed)
    dataset = LiteDataset()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = MicroConditionUNet().to(config.device)
    diffusion = LiteDiffusion(model)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (cond, target) in enumerate(loader):
            cond = cond.to(config.device)
            target = target.to(config.device)

            optim.zero_grad()
            loss = diffusion.train_step(target, cond)
            loss.backward()
            optim.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # 保存模型
        avg_loss = total_loss / len(loader)
        logger.success(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), config.model_path)


# 生成演示
def lite_generate(test_img_path):
    model = MicroConditionUNet().to(config.device)
    model.load_state_dict(torch.load(config.model_path))

    # 预处理输入
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    cond_img = transform(Image.open(test_img_path)).to(config.device)

    # 生成
    diffusion = LiteDiffusion(model)
    with torch.no_grad():
        output = diffusion.generate(cond_img.unsqueeze(0))

    # 显示结果
    plt.imshow(output.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    lite_train()
    # 使用示例:
    # lite_generate("test_portrait.png")