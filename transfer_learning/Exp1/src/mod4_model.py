import os
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from diffusers import (
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
    AutoencoderKL,
)

# -------------------------------
# 数据集定义：读取配对的肖像和素描图
# -------------------------------
class PairedFaceDataset(Dataset):
    def __init__(self, origin_dir: str, target_dir: str, transform=None):
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.filenames = sorted(os.listdir(origin_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        origin_path = os.path.join(self.origin_dir, filename)
        target_path = os.path.join(self.target_dir, filename)
        origin_img = Image.open(origin_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        if self.transform:
            origin_img = self.transform(origin_img)
            target_img = self.transform(target_img)
        return {"condition": origin_img, "target": target_img}

# -------------------------------
# 主训练流程
# -------------------------------
def main():
    logger.add("train.log", level="INFO")
    logger.info("启动 ControlNet 风格迁移训练流程")

    # 定义预处理：调整至 64x80，转换为 tensor，并归一化到 [-1, 1]
    transform = T.Compose([
        T.Resize((64, 80)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 创建数据集与 DataLoader
    dataset = PairedFaceDataset("../dataset/CUFSF/train/origin", "../dataset/CUFSF/train/target", transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    logger.info(f"加载数据集，共 {len(dataset)} 张配对图像")

    # -------------------------------
    # 加载模型：ControlNet、UNet、VAE 以及扩散调度器
    # -------------------------------
    logger.info("加载预训练模型")
    # 这里示例使用公开预训练模型，实际训练时可选择适合的版本
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny"
    )
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )
    scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controlnet.to(device)
    unet.to(device)
    vae.to(device)

    # 优化器只更新 ControlNet 和 UNet 参数
    optimizer = torch.optim.AdamW(
        list(controlnet.parameters()) + list(unet.parameters()), lr=1e-4
    )
    num_epochs = 10

    # -------------------------------
    # 开始训练
    # -------------------------------
    logger.info("开始训练循环")
    for epoch in range(num_epochs):
        logger.info(f"第 {epoch+1}/{num_epochs} 个 Epoch")
        for step, batch in enumerate(dataloader):
            # 将数据移到 device 上
            condition_img = batch["condition"].to(device)  # 肖像图：作为条件输入
            target_img = batch["target"].to(device)         # 素描图：作为训练目标

            # 将目标图像编码为潜在表示
            with torch.no_grad():
                latents = vae.encode(target_img).latent_dist.sample() * 0.18215

            # 随机采样时间步
            batch_size = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()

            # 按时间步添加噪声到潜在向量
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # -------------------------------
            # ControlNet 条件提取（此处简单使用条件图像，实际可加边缘检测、Canny 等预处理）
            # -------------------------------
            control = controlnet(condition_img, timesteps).down_block_additional_residuals

            # -------------------------------
            # UNet 噪声预测
            # -------------------------------
            # 此处假设 UNet 前向支持传入 controlnet 输出作为额外条件，
            # 例如通过参数 controlnet_hint（具体接口请参考 diffusers 文档）
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=None,  # 如有文本条件，可传入文本 embedding
                controlnet_hint=control  # 传入 ControlNet 条件信号
            ).sample

            # 计算 MSE 损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f}")

        # 每个 epoch 保存一次模型
        ckpt_dir = f"checkpoint_epoch_{epoch+1}"
        os.makedirs(ckpt_dir, exist_ok=True)
        controlnet.save_pretrained(os.path.join(ckpt_dir, "controlnet"))
        unet.save_pretrained(os.path.join(ckpt_dir, "unet"))
        logger.info(f"保存第 {epoch+1} 个 Epoch 的模型到 {ckpt_dir}")

    logger.info("训练结束！")

if __name__ == "__main__":
    main()
