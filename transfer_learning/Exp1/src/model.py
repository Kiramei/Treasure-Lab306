import os
from PIL import Image
from torchvision import transforms

import torch
from torch import nn
from diffusers import UNet2DModel, DDPMScheduler
from diffusers import DDIMScheduler

def main():
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((80, 64)),
        transforms.Grayscale(num_output_channels=1),  # 确保输入是灰度图
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
    ])


    def load_data(img_folder, sketch_folder):
        img_paths = sorted(os.listdir(img_folder))
        sketch_paths = sorted(os.listdir(sketch_folder))

        data = []
        for img_name, sketch_name in zip(img_paths, sketch_paths):
            img = Image.open(os.path.join(img_folder, img_name)).convert("RGB")
            sketch = Image.open(os.path.join(sketch_folder, sketch_name)).convert("RGB")

            img = transform(img)
            sketch = transform(sketch)

            data.append((img, sketch))

        return data


    train_data = load_data("dataset/origin", "dataset/target")
    # train_set = torch.utils.data.dataset.TensorDataset(*train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)


    # 定义 U-Net 作为扩散模型的核心
    model = UNet2DModel(
        sample_size=256,
        in_channels=1,  # 灰度图作为输入
        out_channels=1,  # 生成素描图
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    # 扩散噪声调度器
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 训练损失
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练循环
    loss = None
    for epoch in range(100):
        for img, sketch in train_loader:
            img, sketch = img.to(device), sketch.to(device)

            noise = torch.randn_like(sketch).to(device)
            time_steps = torch.randint(0, 1000, (img.shape[0],), device=device)
            noisy_sketch = scheduler.add_noise(sketch, noise, time_steps)

            pred_noise = model(img, time_steps).sample  # 预测噪声

            loss = criterion(pred_noise, noise)  # 计算损失

        print(f"Epoch {epoch} - Loss: {loss.item()}")

    # 选择更快的DDIM采样
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    model.eval()


    def generate_sketch(img):
        img = transform(img).unsqueeze(0).to(device)
        timesteps = scheduler.timesteps.to(device)

        noise = torch.randn_like(img)
        for t in timesteps:
            noise_pred = model(img, t).sample
            img = scheduler.step(noise_pred, t, img).prev_sample

        img = img.squeeze(0).detach().cpu().numpy()
        return (img * 255).astype("uint8")  # 反归一化


    # 测试推理
    test_img = Image.open("test_face.jpg").convert("RGB")
    sketch = generate_sketch(test_img)
    Image.fromarray(sketch).show()
