import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models.detection import (
    ssd300_vgg16,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn
)
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# ---- 配置 ----
image_dir = r"F:/WorkSpace/py/Experiment/od/Exp3/data/VOCdevkit/VOC2007/JPEGImages"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')][:200]

score_thresh = 0.5
IoU_thresh = 0.5
model_list = ['ssd', 'fasterrcnn']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- 通用函数 ----
def transform_image(image):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def compute_iou(box1, box2):
    """box1: [N, 4], box2: [M, 4] in xyxy"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    iou = torch.zeros((box1.size(0), box2.size(0)))
    for i in range(box1.size(0)):
        x1 = torch.max(box1[i, 0], box2[:, 0])
        y1 = torch.max(box1[i, 1], box2[:, 1])
        x2 = torch.min(box1[i, 2], box2[:, 2])
        y2 = torch.min(box1[i, 3], box2[:, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        union = area1[i] + area2 - inter
        iou[i] = inter / union
    return iou


def calculate_map(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    iou = compute_iou(pred_boxes, gt_boxes)
    hits = (iou.max(dim=1).values > iou_thresh).float()
    return hits.mean().item()


# ---- 模型加载 ----
print("[INFO] Loading models...")
models = {
    'ssd': ssd300_vgg16(pretrained=True).to(device).eval(),
    'fasterrcnn': fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval(),
    'maskrcnn': maskrcnn_resnet50_fpn(pretrained=True).to(device).eval(),
    'yolov8': YOLO('yolov8n.pt')  # ultralytics
}

# ---- 主评估循环 ----
results = {name: {'count': [], 'conf': [], 'map': []} for name in model_list}

for img_path in tqdm(image_paths, desc="Evaluating"):
    img = Image.open(img_path).convert("RGB")
    tensor = transform_image(img)
    W, H = img.size
    gt_boxes = torch.tensor([[0, 0, W, H]], dtype=torch.float32)  # 假设整张图像是GT

    for name in model_list:
        if name == 'yolov8':
            yolo_out = models['yolov8'](img_path, verbose=False)[0]
            boxes = torch.tensor(yolo_out.boxes.xyxy.cpu().numpy())
            scores = torch.tensor(yolo_out.boxes.conf.cpu().numpy())
        else:
            with torch.no_grad():
                output = models[name](tensor)[0]
            boxes = output['boxes'].detach().cpu()
            scores = output['scores'].detach().cpu()

        # filter
        keep = scores > score_thresh
        boxes = boxes[keep]
        scores = scores[keep]

        results[name]['count'].append(len(boxes))
        results[name]['conf'].append(scores.mean().item() if len(scores) > 0 else 0.0)
        results[name]['map'].append(calculate_map(boxes, gt_boxes, IoU_thresh))

# ---- 可视化 ----
model_names = list(results.keys())


def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(8, 5))
    for model in model_names:
        plt.plot(results[model][metric_name], label=model)
    plt.xlabel('Image Index')
    plt.ylabel(ylabel)
    plt.title(f'{metric_name.upper()} across images')
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_metric('count', 'Number of Detections')
plot_metric('conf', 'Mean Confidence')
plot_metric('map', 'Pseudo mAP (GT=full image)')


# 热力图分析（聚合）
def heatmap_aggregate(results, image_paths, grid=10):
    heatmaps = {m: np.zeros((grid, grid)) for m in model_names}
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        for name in model_names:
            if name == 'yolov8':
                output = models['yolov8'](img_path, verbose=False)[0]
                boxes = torch.tensor(output.boxes.xyxy.cpu().numpy())
                scores = torch.tensor(output.boxes.conf.cpu().numpy())
            else:
                tensor = transform_image(img)
                with torch.no_grad():
                    output = models[name](tensor)[0]
                boxes = output['boxes'].detach().cpu()
                scores = output['scores'].detach().cpu()

            keep = scores > score_thresh
            boxes = boxes[keep]

            for box in boxes:
                xmin, ymin, xmax, ymax = box
                cx = int(((xmin + xmax) / 2) / W * grid)
                cy = int(((ymin + ymax) / 2) / H * grid)
                if 0 <= cx < grid and 0 <= cy < grid:
                    heatmaps[name][cy, cx] += 1

    for name in model_names:
        plt.figure(figsize=(4, 4))
        sns.heatmap(heatmaps[name], cmap='coolwarm', square=True, cbar=True)
        plt.title(f"{name} Detection Heatmap")
        plt.show()


heatmap_aggregate(results, image_paths)
print("[DONE] All metrics visualized.")
