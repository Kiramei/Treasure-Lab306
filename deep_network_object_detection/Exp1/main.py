import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 加载预训练的Faster-RCNN模型
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()  # 设置为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量并归一化到[0, 1]
])

# 加载测试图像
image_path =r"C:\Users\m1710\Pictures\Screenshots\屏幕截图 2024-04-02 233202.png" # 替换为实际图像路径
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # 增加batch维度

# 进行推理
with torch.no_grad():
    prediction = model(image_tensor)  # 返回检测结果

# 获取检测结果
boxes = prediction[0]['boxes']  # 边界框坐标
labels = prediction[0]['labels']  # 类别标签
scores = prediction[0]['scores']  # 置信度得分

# 过滤低置信度的检测结果
threshold = 0.5  # 设置置信度阈值
filtered_boxes = boxes[scores > threshold]
filtered_labels = labels[scores > threshold]

# 可视化结果
fig, ax = plt.subplots(1)
ax.imshow(image)

for box, label in zip(filtered_boxes, filtered_labels):
    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)  # 绘制边界框
    ax.text(x, y, f'Class: {label.item()}', color='white', fontsize=12,
            bbox=dict(facecolor='red', alpha=0.5))  # 添加类别标签

plt.axis('off')  # 隐藏坐标轴
plt.show()