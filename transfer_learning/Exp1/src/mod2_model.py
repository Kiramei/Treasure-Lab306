from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional, Union, List, Tuple
from alive_progress import alive_bar


# =============================================================================
# 配置管理：统一保存所有超参数与路径
# =============================================================================
class Config:
    def __init__(self):
        # 随机种子与设备设置
        self.seed = np.random.randint(2 ** 32)
        self.device = 'cuda:0'
        # 数据与模型路径
        self.train_csv = '/kiramei_data/WorkSpace/py/Experiment/tl/Exp1/dataset/mnist/train/mnist_train.csv'
        self.test_csv = '/kiramei_data/WorkSpace/py/Experiment/tl/Exp1/dataset/mnist/train/mnist_test.csv'
        self.save_path = './u_net.pt'
        # 数据加载相关
        self.batch_size = 8
        # 模型与训练参数
        self.lr = 0.001
        self.n_diffusion_steps = 600
        self.num_epochs = 200
        self.early_stop_threshold = 40
        # UNet模型参数
        self.image_channels = 1
        self.base_channels = 16
        self.channel_multipliers = [1, 2, 2]
        self.attention_flags = [False, False, False]
        self.n_residual_blocks = 1
        self.dataset_dir = '../dataset/CUFS'


# =============================================================================
# 数据模块：读取 MNIST CSV 文件，并划分训练/验证/测试集
# =============================================================================
class MNISTData:
    def __init__(self, config: Config):
        self.train_csv = config.train_csv
        self.batch_size = config.batch_size

    @staticmethod
    def read_csv(file_path: str):
        with open(file_path, "rb") as f:
            # 解码所有行，过滤掉无法解码的行
            lines = []
            for line in f:
                try:
                    lines.append(line.decode("utf-8"))
                except UnicodeDecodeError:
                    pass

        data = np.array(
            [int(item) for row in lines for item in row.strip().split(b',' if isinstance(row, bytes) else ',')])
        data = data.reshape(len(lines), -1)
        # 第一列为标签，其余为像素值
        imgs = torch.from_numpy(data[:, 1:]).float() / 255.0
        labels = torch.from_numpy(data[:, 0])
        return imgs, labels

    def setup(self):
        imgs, labels = self.read_csv(self.train_csv)
        rand_idx = torch.rand(len(imgs))
        train_mask = rand_idx < 0.8
        valid_mask = (rand_idx >= 0.8) & (rand_idx < 0.9)
        test_mask = rand_idx >= 0.9

        train_set = TensorDataset(imgs[train_mask], labels[train_mask])
        valid_set = TensorDataset(imgs[valid_mask], labels[valid_mask])
        test_set = TensorDataset(imgs[test_mask], labels[test_mask])
        return train_set, valid_set, test_set

    def get_loaders(self):
        train_set, valid_set, test_set = self.setup()
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)
        return train_loader, valid_loader, test_loader


class SketchData(Dataset):
    def __init__(self, config, split=0):
        self.split = split
        self.config = config
        dataset_dir = Path(config.dataset_dir)
        data_select_dict = {
            0: dataset_dir / 'train',
            1: dataset_dir / 'valid',
            2: dataset_dir / 'test'
        }
        self.portrait_paths = list(data_select_dict[split].glob("origin/*.jpg"))
        # if split < 2:
        self.sketch_paths = list(data_select_dict[split].glob("target/*.jpg"))

    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 200))
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        img = torch.Tensor(img.astype(np.float32))
        img = (img / 255.).to(self.config.device)
        return img

    def get_loader(self):
        return DataLoader(self, batch_size=self.config.batch_size, shuffle=True)

    def __len__(self):
        return len(self.portrait_paths)

    def __getitem__(self, idx):
        portrait = self.read_image(self.portrait_paths[idx])
        sketch = self.read_image(self.sketch_paths[idx])
        return portrait, sketch


# =============================================================================
# 模型模块：UNet 及其各构建块
# =============================================================================
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

        self.res_time = nn.Sequential(nn.Linear(time_channels, out_channels), Swish())

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


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1, key_dim: Optional[int] = None, n_groups: int = 16):
        super().__init__()
        key_dim = channels if key_dim is None else key_dim
        self.proj = nn.Linear(channels, num_heads * key_dim * 3)
        self.out_proj = nn.Linear(num_heads * key_dim, channels)
        self.scale = key_dim ** -0.5
        self.num_heads = num_heads
        self.key_dim = key_dim

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        # 忽略 t
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        qkv = self.proj(x_flat).view(b, -1, self.num_heads, 3 * self.key_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        out = torch.einsum('bijh,bjhd->bihd', attn, v)
        out = out.reshape(b, -1, self.num_heads * self.key_dim)
        out = self.out_proj(out) + x_flat
        return out.permute(0, 2, 1).view(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.tensor, _: torch.tensor):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.deconv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, use_attention: bool):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn_block = AttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res_block(x, t)
        return self.attn_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, use_attention: bool):
        super().__init__()
        # 注意此处输入为 concat 后的通道数
        self.res_block = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        self.attn_block = AttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, skip: torch.Tensor):
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x, t)
        return self.attn_block(x)


class MiddleBlock(nn.Module):
    def __init__(self, channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_channels)
        self.attn = AttentionBlock(channels)
        self.res2 = ResidualBlock(channels, channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        return self.res2(x, t)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(embed_dim // 4, embed_dim)
        self.act = Swish()
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t: torch.Tensor):
        half_dim = self.embed_dim // 8
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_factor)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.linear2(self.act(self.linear1(emb)))


class UNetModel(nn.Module):
    def __init__(self,
                 image_channels: int,
                 base_channels: int,
                 channel_multipliers: List[int],
                 attention_flags: List[bool],
                 n_residual_blocks: int):
        super().__init__()
        self.image_projection = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.time_embedding = TimeEmbedding(base_channels * 4)
        # Encoder 构建
        in_channels = base_channels
        self.down_blocks = nn.ModuleList()
        for i, mult in enumerate(channel_multipliers):
            out_channels = in_channels * mult
            for _ in range(n_residual_blocks):
                self.down_blocks.append(DownBlock(in_channels, out_channels, base_channels * 4, attention_flags[i]))
                in_channels = out_channels
            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(Downsample(in_channels))
        # Middle block
        self.middle_block = MiddleBlock(in_channels, base_channels * 4)
        # Decoder 构建
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            for _ in range(n_residual_blocks):
                self.up_blocks.append(UpBlock(in_channels, in_channels, base_channels * 4, attention_flags[i]))
            out_channels = in_channels // mult
            self.up_blocks.append(UpBlock(in_channels, out_channels, base_channels * 4, attention_flags[i]))
            in_channels = out_channels
            if i > 0:
                self.up_blocks.append(Upsample(in_channels))
        self.norm = nn.GroupNorm(8, in_channels)
        self.act = Swish()
        self.final_conv = nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_embedding(t)
        x = self.image_projection(x)
        skip_connections = [x]
        for layer in self.down_blocks:
            x = layer(x, t_emb)
            skip_connections.append(x)
        x = self.middle_block(x, t_emb)
        for layer in self.up_blocks:
            if isinstance(layer, Upsample):
                x = layer(x, t_emb)
            else:
                skip = skip_connections.pop()
                x = layer(x, t_emb, skip)
        return self.final_conv(self.act(self.norm(x)))


# =============================================================================
# 扩散过程模块：封装正向扩散、采样及损失计算
# =============================================================================
def gather_tensor(consts: torch.Tensor, t: torch.Tensor):
    return consts.gather(-1, t).reshape(-1, 1, 1, 1)


class DiffusionProcess:
    def __init__(self, model: nn.Module, n_steps: int, device: Union[str, torch.device]):
        self.model = model
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, n_steps, device=device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor):
        mean = (gather_tensor(self.alpha_bar, t) ** 0.5) * x0
        var = 1. - gather_tensor(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + var.sqrt() * noise

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_pred = self.model(xt, t)
        alpha_bar_t = gather_tensor(self.alpha_bar, t)
        alpha_t = gather_tensor(self.alpha, t)
        eps_coef = (1. - alpha_t) / (1. - alpha_bar_t).sqrt()
        mean = (1. / alpha_t.sqrt()) * (xt - eps_coef * eps_pred)
        var = gather_tensor(self.sigma, t)
        return mean + var.sqrt() * torch.randn_like(xt)

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        bsz = x0.shape[0]
        t = torch.randint(0, self.n_steps, (bsz,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_pred = self.model(xt, t)
        return F.mse_loss(noise, eps_pred)


# =============================================================================
# 训练与采样模块：封装训练流程、模型保存、采样及结果展示
# =============================================================================
class DiffusionTrainer:
    def __init__(self, config: Config):
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        # 初始化模型
        self.model = UNetModel(
            image_channels=config.image_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            attention_flags=config.attention_flags,
            n_residual_blocks=config.n_residual_blocks
        ).to(config.device)
        self.diffusion = DiffusionProcess(self.model, config.n_diffusion_steps, config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def train(self, train_loader, valid_loader):
        best_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_losses = []

            with alive_bar(len(train_loader), title="Training", bar="smooth") as bar:
                for portrait, sketch in train_loader:
                    sketch = sketch.view(-1, 1, 200, 256).to(self.config.device)
                    self.optimizer.zero_grad()
                    loss = self.diffusion.loss(sketch)
                    loss.backward()
                    self.optimizer.step()
                    train_losses.append(loss.item())
                    bar()
            avg_train_loss = np.mean(train_losses)

            self.model.eval()
            valid_losses = []
            with torch.no_grad():
                for imgs, _ in valid_loader:
                    imgs = imgs.view(-1, 1, 200, 256).to(self.config.device)
                    valid_losses.append(self.diffusion.loss(imgs).item())
            avg_valid_loss = np.mean(valid_losses)
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Valid Loss = {avg_valid_loss:.6f}")

            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                early_stop_counter = 0
                torch.save(self.model, self.config.save_path)
            else:
                early_stop_counter += 1
            print(f"Early Stop Counter: {early_stop_counter}/{self.config.early_stop_threshold}")
            if early_stop_counter > self.config.early_stop_threshold:
                print("Early stopping triggered.")
                break

    def sample_and_show(self):
        self.model = torch.load(self.config.save_path, weights_only=False)
        self.diffusion = DiffusionProcess(self.model, self.config.n_diffusion_steps, self.config.device)
        with torch.no_grad():
            x_sample = torch.randn((1, 1, 256, 200), device=self.config.device)
            samples, steps = [], []
            for t in reversed(range(self.config.n_diffusion_steps)):
                t_tensor = torch.tensor([t], device=self.config.device)
                x_sample = self.diffusion.p_sample(x_sample, t_tensor)
                if (t + 1) % 100 == 1:
                    samples.append(x_sample.cpu().detach())
                    steps.append(t + 1)
        self.show_samples(samples, steps)

    @staticmethod
    def show_samples(images: List[torch.Tensor], steps: List[int]):
        n = len(images)
        _, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        for step, ax, img in zip(steps, axes, images):
            ax.imshow(img.view(256, 200), cmap='gray')
            ax.axis('off')
            ax.set_title(f"Step {step}")
        plt.show()


# =============================================================================
# 主程序：初始化配置、数据、训练与采样
# =============================================================================
if __name__ == "__main__":
    cfg = Config()
    t_loader = SketchData(cfg, split=0).get_loader()
    v_loader = SketchData(cfg, split=2).get_loader()
    torch.manual_seed(cfg.seed)
    trainer = DiffusionTrainer(cfg)
    # 若需要训练，请取消下面注释：
    # trainer.train(t_loader, v_loader)

    # 采样展示
    trainer.sample_and_show()
