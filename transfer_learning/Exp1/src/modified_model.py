from typing import Optional

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from torchvision import transforms


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return x * self.activation(x)


class ResidualBlock(nn.Module):
    # n_groups, hyper-parameter of group norm
    # group norm, group normalize; first, split channels into different groups; then, normalize feature in every group, as batch_normalization, it has some hyper-parameters.
    # feat_map
    # cv1(feat_map)+ cv2(time_emb) -> feat_map' + cv(feat_map) -> output
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        # in_channels// n_groups

        self.res1 = nn.Sequential(nn.GroupNorm(n_groups, in_channels),
                                  Swish(),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))

        self.res_time = nn.Sequential(nn.Linear(out_channels, out_channels), Swish())

        self.res2 = nn.Sequential(nn.GroupNorm(n_groups, out_channels),
                                  Swish(),
                                  nn.Dropout(dropout),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res1(x) + self.res_time(t)[:, :, None, None]
        h = self.res2(h) + self.shortcut(x)
        return h


class DownBlock(nn.Module):
    # Encoder
    # DownBlock= ResidualBlock+ AttentionBlock
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x



class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, k_dims: int = None, n_groups: int = 16):
        super().__init__()
        if k_dims is None:
            k_dims = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # as n_channels= 64, 64>> 8* 128* 3
        self.projection = nn.Linear(n_channels, n_heads * k_dims * 3)
        self.output = nn.Linear(n_heads * k_dims, n_channels)
        self.scale = k_dims ** -0.5
        self.n_heads = n_heads
        self.k_dims = k_dims

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        b, c, h, w = x.shape
        # pull x straight, (batch_size, n_channels, H* W)>> (batch_size, H* W, n_channels)
        x = x.view(b, c, -1).permute(0, 2, 1)
        # (batch_size, H* W, channels)>> (batch_size, H* W, head, 3* k_head_dim)
        qkv = self.projection(x).view(b, -1, self.n_heads, 3 * self.k_dims)
        # q, (batch_size, H* W, head, k_head_dim); k, (batch_size, H* W, head, k_head_dim); v, (,,).
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # (batch_size, H* W, head, dim), (batch_size, H* W, head, dim) -> (batch_size, H* W, H* W, head)
        # This writing style is really good!
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        # (batch_size, H* W, H* W, head), (batch_size, H* W, head, dim)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # (batch_size, H* W, head* dim)
        res = res.view(b, -1, self.n_heads * self.k_dims)
        # (batch_size, H* W, C)
        res = self.output(res) + x
        res = res.permute(0, 2, 1).view(b, c, h, w)
        return res


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.tensor, t: torch.tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # we concatenate the output of the same resolution
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


# Configuration
class Config:
    seed = 42
    batch_size = 32
    image_size = (64, 80)
    channels = 1
    lr = 2e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 200
    early_stop_patience = 20
    num_timesteps = 1000
    model_save_path = "portrait2sketch.pth"
    dataset_path = "../dataset"  # 包含portrait/和sketch/子目录


# 初始化配置
config = Config()
logger.add("training.log", rotation="10 MB")


# 数据加载
class PortraitDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.portrait_paths = list(Path(dataset_dir).glob("origin/*.png"))
        self.sketch_paths = list(Path(dataset_dir).glob("target/*.png"))
        self.transform = transform

    def __len__(self):
        return min(len(self.portrait_paths), len(self.sketch_paths))

    def __getitem__(self, idx):
        portrait = read_image(self.portrait_paths[idx])
        sketch = read_image(self.sketch_paths[idx])
        return portrait, sketch


# 数据预处理
def get_transforms():
    return torch.nn.Sequential(
        transforms.Resize(config.image_size),
        # transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    )


# 模型架构
class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = TimeEmbedding(256)
        self.condition_encoder = ConditionEncoder()

        self.down_blocks = nn.ModuleList([
            DownBlock(16, 32, 256, has_attn=False),
            DownBlock(32, 64, 256, has_attn=True),
            DownBlock(64, 128, 256, has_attn=True)
        ])

        self.mid_block = MiddleBlock(128, 256)

        self.up_blocks = nn.ModuleList([
            UpBlock(128 + 64, 64, 256, has_attn=True),
            UpBlock(64 + 32, 32, 256, has_attn=True),
            UpBlock(32, 16, 256, has_attn=False)
        ])

    def forward(self, x, t, condition):
        t_emb = self.time_embed(t)
        condition = condition.unsqueeze(1)
        cond_feat = self.condition_encoder(condition)

        # 下采样
        features = []
        for block in self.down_blocks:
            x = block(x, t_emb, cond_feat)
            features.append(x)
            x = F.avg_pool2d(x, 2)

        # 中间层
        x = self.mid_block(x, t_emb, cond_feat)

        # 上采样
        for block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x = torch.cat([x, features.pop()], dim=1)
            x = block(x, t_emb, cond_feat)

        return x


class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            Swish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            Swish(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            Swish()
        )

    def forward(self, x):
        return self.layers(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, dim // 2),
            Swish(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, t):
        return self.layers(t.float().view(-1, 1))


# 其他组件定义（DownBlock、UpBlock等与原始代码类似，需要添加条件特征融合）

# 扩散过程
class PortraitDiffusion:
    def __init__(self, model):
        self.model = model
        self.timesteps = config.num_timesteps
        self.beta = torch.linspace(1e-4, 0.02, self.timesteps).to(config.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,))

    def q_sample(self, x0, t, noise):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t][:, None, None, None])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t][:, None, None, None])
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def p_loss(self, x0, condition, t=None):
        if t is None:
            t = self.sample_timesteps(x0.shape[0]).to(config.device)

        noise = torch.randn_like(x0).to(x0.device)
        x_noisy = self.q_sample(x0, t, noise)
        pred_noise = self.model(x_noisy, t, condition)
        return F.mse_loss(noise, pred_noise)

    @torch.no_grad()
    def sample(self, condition, return_process=False):
        self.model.eval()
        x = torch.randn_like(condition)
        process = []

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((1,), t, device=config.device)
            pred_noise = self.model(x, t_tensor, condition)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (x - beta_t * pred_noise / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_t)
            x += torch.sqrt(beta_t) * noise

            if return_process and t % 50 == 0:
                process.append(x.detach().cpu())

        return (x.clamp(-1, 1) + 1) / 2, process


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = torch.Tensor(img.astype(np.float32))
    img = (img / 255.).to(config.device)
    return img


# 训练流程
def train():
    # 初始化
    torch.manual_seed(config.seed)
    transform = get_transforms()
    dataset = PortraitDataset(config.dataset_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = ConditionalUNet().to(config.device)
    diffusion = PortraitDiffusion(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_loss = float("inf")
    patience_counter = 0

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = []

        for batch_idx, (portraits, sketches) in enumerate(train_loader):
            portraits = portraits.to(config.device)
            sketches = sketches.to(config.device)

            optimizer.zero_grad()
            loss = diffusion.p_loss(sketches, portraits)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = np.mean(epoch_loss)
        logger.info(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            logger.success(f"Model saved at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                logger.warning("Early stopping triggered")
                break


# 测试生成
def generate_sample(portrait_path):
    model = ConditionalUNet().to(config.device)
    model.load_state_dict(torch.load(config.model_save_path))
    diffusion = PortraitDiffusion(model)

    portrait = read_image(portrait_path)
    portrait_tensor = portrait.unsqueeze(0).to(config.device)

    with torch.no_grad():
        sketch_tensor, _ = diffusion.sample(portrait_tensor)

    sketch = sketch_tensor.squeeze().cpu().numpy()
    plt.imshow(sketch, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    train()
    # 使用示例
    # generate_sample("test_portrait.png")
