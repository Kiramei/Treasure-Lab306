import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO 数据集的类别名（索引从1开始）
COCO_INSTANCE_CATEGORY_NAMES = ['__background__',  # always index 0
'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# 加载图像并预处理
image = Image.open(r'F:\WorkSpace\py\Experiment\od\Exp3\data\VOCdevkit\VOC2007\JPEGImages\002944.jpg').convert("RGB")
transform = Compose([
    # Resize((300, 300)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).cuda()  # 添加 batch 维度并转移到GPU
# 加载模型
model = ssd300_vgg16(pretrained=True).cuda()
model.eval()
# 推理
with torch.no_grad():
    outputs = model(input_tensor)

# 筛选出得分高于阈值的框
threshold = 0.5
boxes = outputs[0]['boxes']
scores = outputs[0]['scores']
labels = outputs[0]['labels']

keep = scores > threshold
boxes = boxes[keep].cpu()
labels = labels[keep].cpu()
scores = scores[keep].cpu()
# 显示图像
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)
for box, score, label in zip(boxes, scores, labels):
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    class_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
    ax.text(xmin, ymin, f"{class_name}:{score:.2f}", color='white', fontsize=10,
            bbox=dict(facecolor='red', alpha=0.5))
plt.axis('off')
plt.show()
