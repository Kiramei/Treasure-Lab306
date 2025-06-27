import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPVisionModel, CLIPImageProcessor
from loguru import logger
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


# 配置参数
class Config:
    seed = 42
    image_size = (64, 80)
    batch_size = 4
    lr = 3e-5
    num_epochs = 200
    gradient_accumulation_steps = 2
    mixed_precision = "fp16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = "../dataset/CUFSF/train"
    output_dir = "./portrait2sketch"
    condition_scale = 1.5  # 条件控制强度


config = Config()
logger.add("training.log")


# 自定义条件UNet
class ConditionUNet(UNet2DConditionModel):
    def __init__(self):
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=768,  # CLIP视觉编码维度
        )


# 数据集类
class PortraitDataset(Dataset):
    def __init__(self):
        self.portraits = sorted(Path(config.dataset_path).glob("origin/*.png"))
        self.sketches = sorted(Path(config.dataset_path).glob("target/*.png"))
        self.processor = CLIPImageProcessor(
            do_rescale=False,
            crop_size={
            "width": 64,
            "height": 80
        })

    def __len__(self):
        return len(self.portraits)

    def __getitem__(self, idx):
        # 加载并预处理图像
        def process_image(path):
            img = Image.open(path).convert("L")
            img = img.resize(config.image_size)
            return self.processor(img, return_tensors="pt").pixel_values.squeeze(0)

        return {
            "condition": process_image(self.portraits[idx]),
            "target": process_image(self.sketches[idx]),
        }


# 条件编码器
class ConditionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.mapper = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 768)
        )

    def forward(self, images):
        with torch.no_grad():
            clip_features = self.clip_vision(images).last_hidden_state
        return self.mapper(clip_features)


# 训练流程
def train():
    # 初始化组件
    unet = ConditionUNet().to(config.device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    condition_net = ConditionNet().to(config.device)

    # 优化器和数据加载
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(condition_net.mapper.parameters()),
        lr=config.lr
    )
    dataset = PortraitDataset()
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 学习率调度
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * config.num_epochs
    )

    # 训练循环
    global_step = 0
    for epoch in range(config.num_epochs):
        unet.train()
        condition_net.train()

        for batch in train_loader:
            # 准备数据
            clean_images = batch["target"].to(config.device)
            conditions = condition_net(batch["condition"].to(config.device))

            # 添加噪声
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],), device=config.device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # 模型预测
            noise_pred = unet(
                noisy_images, timesteps,
                encoder_hidden_states=conditions
            ).sample

            # 计算损失
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            # 梯度累积
            if global_step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # 记录日志
            if global_step % 50 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}"
                )

            global_step += 1

        # 保存检查点
        if epoch % 10 == 0:
            pipeline = DDPMPipeline(
                unet=unet,
                scheduler=noise_scheduler
            )
            pipeline.save_pretrained(config.output_dir)
            logger.success(f"Checkpoint saved at epoch {epoch}")


# 推理生成
def generate(portrait_path):
    # 加载模型
    pipeline = DDPMPipeline.from_pretrained(config.output_dir).to(config.device)
    condition_net = ConditionNet().to(config.device)

    # 预处理输入
    processor = CLIPImageProcessor()
    condition_img = Image.open(portrait_path).convert("L").resize(config.image_size)
    condition = processor(condition_img, return_tensors="pt").pixel_values.to(config.device)

    # 生成素描
    with torch.no_grad():
        condition_emb = condition_net(condition)
        output = pipeline(
            num_inference_steps=50,
            generator=torch.manual_seed(config.seed),
            encoder_hidden_states=condition_emb,
            guidance_scale=config.condition_scale
        ).images[0]

    # 后处理
    output = output.convert("L")
    plt.imshow(output, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    train()
    # 使用示例:
    # generate("test_portrait.png")