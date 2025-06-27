import torch
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np

# 图像路径（替换为你本地图像）
image_path = r'F:\WorkSpace\py\Experiment\od\Exp3\data\VOCdevkit\VOC2007\JPEGImages\002944.jpg'

# 类别名（COCO）
COCO_INSTANCE_CATEGORY_NAMES = [ '背景', '人', '自行车', '汽车', '摩托车', '飞机', '公交车', '火车', '卡车',
    '船', '红绿灯', '消防栓', '停止标志', '停车收费表', '长椅', '鸟', '猫', '狗', '马', '羊', '牛',
    '大象', '熊', '斑马', '长颈鹿', '背包', '伞', '手提包', '领带', '行李箱', '飞盘', '滑雪板',
    '滑雪板鞋', '运动球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', '网球拍', '瓶子',
    '酒杯', '杯子', '叉子', '刀', '勺子', '碗', '香蕉', '苹果', '三明治', '橙子', '西兰花',
    '胡萝卜', '热狗', '披萨', '甜甜圈', '蛋糕', '椅子', '沙发', '盆栽', '床', '餐桌',
    '马桶', '电视', '笔记本', '鼠标', '遥控器', '键盘', '手机', '微波炉', '烤箱', '烤面包机',
    '水槽', '冰箱', '书', '钟表', '花瓶', '剪刀', '玩具熊', '吹风机', '牙刷']

# 图像读取与预处理
image = Image.open(image_path).convert("RGB")
W, H = image.size
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).cuda()

# 模型加载（SSD + Faster R-CNN）
ssd = ssd300_vgg16(pretrained=True).cuda().eval()
rcnn = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()

# 推理
with torch.no_grad():
    out_ssd = ssd(input_tensor)[0]
    out_rcnn = rcnn(input_tensor)[0]

# 过滤高置信度检测
def filter_output(output, threshold=0.5):
    keep = output['scores'] > threshold
    return output['boxes'][keep].cpu(), output['scores'][keep].cpu(), output['labels'][keep].cpu()

boxes_ssd, scores_ssd, labels_ssd = filter_output(out_ssd)
boxes_rcnn, scores_rcnn, labels_rcnn = filter_output(out_rcnn)

# ✅ 柱状图比较检测数与平均置信度
methods = ['SSD', 'Faster R-CNN']
counts = [len(scores_ssd), len(scores_rcnn)]
avg_scores = [scores_ssd.mean().item() if len(scores_ssd) > 0 else 0,
              scores_rcnn.mean().item() if len(scores_rcnn) > 0 else 0]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(methods, counts)
plt.title('检测框数量')

plt.subplot(1, 2, 2)
plt.bar(methods, avg_scores)
plt.title('平均置信度')

plt.tight_layout()
plt.show()

# ✅ 热力图展示检测集中区域
def heatmap_from_boxes(boxes, img_size, grid=10):
    heatmap = np.zeros((grid, grid))
    H, W = img_size
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        cx = int(((xmin + xmax) / 2) / W * grid)
        cy = int(((ymin + ymax) / 2) / H * grid)
        if 0 <= cx < grid and 0 <= cy < grid:
            heatmap[cy, cx] += 1
    return heatmap

ssd_heat = heatmap_from_boxes(boxes_ssd, (H, W))
rcnn_heat = heatmap_from_boxes(boxes_rcnn, (H, W))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(ssd_heat, cmap='Reds', cbar=True)
plt.title('SSD 热力图')

plt.subplot(1, 2, 2)
sns.heatmap(rcnn_heat, cmap='Blues', cbar=True)
plt.title('Faster R-CNN 热力图')
plt.tight_layout()
plt.show()

