import os
import random

import cv2
import torch
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import functional as TF

from torchvision.models.detection import (
    ssd300_vgg16,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    SSD300_VGG16_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights
)
from ultralytics import YOLO

# ========== 路径配置 ==========
IMAGE_DIR = r'F:\WorkSpace\py\Experiment\od\Exp3\data\VOCdevkit\VOC2007\JPEGImages'
ANNOTATION_DIR = r'F:\WorkSpace\py\Experiment\od\Exp3\data\VOCdevkit\VOC2007\Annotations'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'vis'), exist_ok=True)

classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

YOLO_CLASS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
              8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
              14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
              22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
              29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
              35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
              40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
              48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
              55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
              62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
              69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
              76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

VOC_COLORS = {
    name: tuple(random.randint(60, 255) for _ in range(3))
    for name in [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
}


# ========== 工具函数 ==========
def compute_iou(box1, box2):
    """
    box: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        box = [
            float(bbox.find('xmin').text),
            float(bbox.find('ymin').text),
            float(bbox.find('xmax').text),
            float(bbox.find('ymax').text)
        ]
        boxes.append(box)
        labels.append(name)
    return boxes, labels


def calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, model_name, iou_threshold=0.5):
    """
    简化版 VOC mAP：IoU > threshold 视为TP，否则为FP
    """
    if len(pred_boxes) == 0:
        return 0.0

    pred_sorted = sorted(zip(pred_boxes, pred_labels, pred_scores), key=lambda x: -x[2])
    tp, fp = [], []
    matched = set()
    for pred_box, pred_label, _ in pred_sorted:
        found_match = False
        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if i in matched:
                continue
            _pred_label = classNames[int(pred_label)] if model_name != 'yolov8n' else YOLO_CLASS[int(pred_label)]
            if _pred_label == gt_label and compute_iou(pred_box, gt_box) >= iou_threshold:
                found_match = True
                matched.add(i)
                break
        if found_match:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (len(gt_boxes) + 1e-6)
    if len(precision) < 2:
        ap = precision[0] if recall[0] > 0 else 0.0
    else:
        ap = np.trapz(precision, recall)
    return ap


# ========== 模型加载 ==========
def load_models():
    print("Loading models...")

    models = {
        'ssd': ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).cuda().eval(),
        'rcnn': fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).cuda().eval(),
        'maskrcnn': maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).cuda().eval(),
        'yolov8n': YOLO('yolov8n.pt')
    }
    return models


# ========== 图像处理 ==========
def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = TF.to_tensor(image_rgb).cuda()
    return image, image_tensor.unsqueeze(0)  # 返回原图和 batch 维度张量


# ========== 模型推理统一接口 ==========
def run_model(model_name, model, image_path):
    image_orig, image_tensor = load_image(image_path)
    H, W, _ = image_orig.shape

    if model_name == 'yolov8n':
        results = model(image_path, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = [int(i) for i in results.boxes.cls.cpu().numpy()]
    else:
        with torch.no_grad():
            output = model(image_tensor)[0]
        keep = output['scores'] > 0.5
        boxes = output['boxes'][keep].cpu().numpy()
        scores = output['scores'][keep].cpu().numpy()
        labels = output['labels'][keep].cpu().numpy()

    # 转为 VOC 格式 box: [xmin, ymin, xmax, ymax]
    boxes = boxes.tolist()
    scores = scores.tolist()
    labels = [str(l) for l in labels]  # 暂用 str(label index)，后续映射标签名对齐 VOC

    return boxes, labels, scores, image_orig


# ========== 创建子目录 ==========
def ensure_vis_dir(model_name):
    vis_path = os.path.join(OUTPUT_DIR, 'vis', model_name)
    os.makedirs(vis_path, exist_ok=True)
    return vis_path


# ========== 绘制检测框图像 ==========
def draw_and_save(image, boxes, labels, scores, image_id, model_name, label_map=None):
    vis_dir = ensure_vis_dir(model_name)
    out_path = os.path.join(vis_dir, f"{image_id}.jpg")
    img = image.copy()

    label_map = classNames if model_name != 'yolov8n' else YOLO_CLASS

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = map(int, box)
        label_name = label_map[int(label)] if label_map and str(label).isdigit() else str(label)
        caption = f"{label_name} {score:.2f}"
        color = VOC_COLORS.get(label_name, (0, 255, 0))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(caption, font, font_scale, font_thickness)

        pad = 4
        if ymin - th - 2 * pad < 0:
            text_bg_topleft = (xmin, ymin + pad)
            text_bg_bottomright = (xmin + tw + 2 * pad, ymin + th + 3 * pad)
            text_org = (xmin + pad, ymin + th + pad)
        else:
            text_bg_topleft = (xmin, ymin - th - 2 * pad)
            text_bg_bottomright = (xmin + tw + 2 * pad, ymin)
            text_org = (xmin + pad, ymin - pad)

        # 直接绘制框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        # 绘制标签背景（不透明）
        cv2.rectangle(img, text_bg_topleft, text_bg_bottomright, color, thickness=-1)
        # 黑描边 + 白字
        cv2.putText(img, caption, text_org, font, font_scale,
                    (0, 0, 0), font_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(img, caption, text_org, font, font_scale,
                    (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, img)


# ========== 热力图聚合 ==========
def update_heatmap(heatmap, boxes, image_shape, grid=20):
    H, W = image_shape[:2]
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        cx = int(((xmin + xmax) / 2) / W * grid)
        cy = int(((ymin + ymax) / 2) / H * grid)
        if 0 <= cx < grid and 0 <= cy < grid:
            heatmap[cy, cx] += 1


def plot_heatmap(heatmap, model_name, grid=20):
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap, cmap='magma', cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f'Heatmap - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_{model_name}.png'))
    plt.close()


# ========== 主处理逻辑 ==========
def process_all_images():
    models = load_models()
    model_names = list(models.keys())
    label_map = [  # VOC类顺序，0是背景
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    label_map = ['background'] + label_map  # VOC 类别从 1 起

    records = []
    heatmaps = {name: np.zeros((20, 20)) for name in model_names}

    image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg'))
    print(f"Found {len(image_files)} images.")

    for image_file in tqdm(image_files, desc="Processing images"):
        image_id = image_file.replace(".jpg", "")
        image_path = os.path.join(IMAGE_DIR, image_file)
        gt_path = os.path.join(ANNOTATION_DIR, image_id + ".xml")
        gt_boxes, gt_labels = parse_voc_annotation(gt_path)

        can_show_image = random.Random().random() < 0.01
        for model_name in model_names:
            boxes, labels, scores, image = run_model(model_name, models[model_name], image_path)

            # 画图保存
            if can_show_image:
                draw_and_save(image, boxes, labels, scores, image_id, model_name, label_map)

            # mAP评估
            ap = calculate_map(boxes, labels, scores, gt_boxes, gt_labels, model_name)

            # 热力图更新
            update_heatmap(heatmaps[model_name], boxes, image.shape)

            records.append({
                "ImageID": image_id,
                "Model": model_name,
                "NumDetections": len(boxes),
                "MeanScore": np.mean(scores) if scores else 0.0,
                "mAP@0.5": ap
            })

    return records, heatmaps


# ========== 保存 CSV ==========
def save_csv(records):
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, 'detection_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"[✔] 结果已保存到：{csv_path}")
    return df


# ========== 绘制 mAP 曲线 ==========
def plot_map_curve(df):
    plt.figure(figsize=(10, 5))
    model_names = df['Model'].unique()
    image_ids = sorted(df['ImageID'].unique())

    for model in model_names:
        ap_series = df[df['Model'] == model].sort_values(by='ImageID')['mAP@0.5']
        plt.plot(image_ids, ap_series, label=model)

    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel('Image ID')
    plt.ylabel('mAP@0.5')
    plt.title('mAP@0.5 per Image')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'map_curve.png'))
    plt.close()
    print(f"[✔] mAP 曲线图已保存")


# ========== 主函数 ==========
if __name__ == '__main__':
    all_records, heatmaps = process_all_images()
    df = save_csv(all_records)
    plot_map_curve(df)

    for model_name, heat in heatmaps.items():
        plot_heatmap(heat, model_name)
