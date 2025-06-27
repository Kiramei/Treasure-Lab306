import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义数据预处理
transform = transforms.Compose([
    # transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# 下载并加载VOC2007数据集（仅trainval部分）
dataset = VOCDetection(
    root="./data",
    year="2007",
    image_set="trainval",
    download=False,
    transform=transform,
)

# 构建数据加载器
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# 可视化一张样本
def show_sample(sample):
    image, target = sample
    image = image.squeeze().permute(1, 2, 0).numpy()
    annots = target["annotation"]["object"]
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if not isinstance(annots, list):
        annots = [annots]

    for obj in annots:
        bbox = obj["bndbox"]
        xmin = int(bbox["xmin"][0])
        ymin = int(bbox["ymin"][0])
        xmax = int(bbox["xmax"][0])
        ymax = int(bbox["ymax"][0])
        label = obj["name"]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, label, color='white', backgroundcolor='red')

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# 从加载器中取一张图片进行展示
# sample = next(iter(dataloader))
# show_sample(sample)
