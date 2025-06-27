import torch
import itertools
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCDetection

scaler = GradScaler()

# from start import sample

VOC_LABELS = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


class VOCDataset(Dataset):
    def __init__(self, root, year="2007", image_set="trainval", transform=None):
        self.voc = VOCDetection(root=root, year=year, image_set=image_set, download=False)
        self.transform = transform or transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])
        self.label_map = {name: i + 1 for i, name in enumerate([
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'])}

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        img = self.transform(img)

        objects = target["annotation"]["object"]
        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        labels = []
        for obj in objects:
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            cx = (xmin + xmax) / 2 / 300
            cy = (ymin + ymax) / 2 / 300
            w = (xmax - xmin) / 300
            h = (ymax - ymin) / 300
            boxes.append([cx, cy, w, h])
            labels.append(self.label_map[obj["name"]])

        return img, torch.tensor(boxes), torch.tensor(labels, dtype=torch.long)


# 构建主干网络（VGG16 简化版）
class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Conv1_1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv1_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 150x150

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv2_1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Conv2_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 75x75

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv3_1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv3_2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv3_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 38x38
        )

    def forward(self, x):
        return self.features(x)


# 检测层：用于生成预测类别和框偏移
class SSDDetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(SSDDetectionHead, self).__init__()
        self.loc = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.cls = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        loc = self.loc(x)  # (N, num_anchors*4, H, W)
        cls = self.cls(x)  # (N, num_anchors*num_classes, H, W)
        return loc, cls


# 整体SSD模型
class SimpleSSD(nn.Module):
    def __init__(self, num_classes=21):  # VOC有20类 + 背景
        super(SimpleSSD, self).__init__()
        self.backbone = VGGBase()
        self.detect = SSDDetectionHead(256, num_anchors=4, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)  # 提取特征图
        loc_preds, cls_preds = self.detect(features)  # 位置 & 类别预测
        return loc_preds, cls_preds


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3, alpha=1.0):
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

    def forward(self, pred_locs, pred_scores, gt_locs, gt_labels, positive_mask):
        """
        pred_locs: (N, 4) location predictions
        pred_scores: (N, C) class predictions
        gt_locs: (N, 4) encoded gt for each matched dbox
        gt_labels: (N,) int64 tensor with 0 = background, 1~C = classes
        positive_mask: (N,) boolean
        """

        # 回归损失（仅正样本参与）
        pos_idx = positive_mask
        loc_loss = F.smooth_l1_loss(pred_locs[pos_idx], gt_locs[pos_idx], reduction='sum')
        gt_labels = gt_labels.to(pred_scores.device)  # 确保 gt_labels 在同一设备上
        # 分类损失（正样本 + 硬负样本）
        conf_loss = F.cross_entropy(pred_scores, gt_labels, reduction='none')  # (N,)
        pos_conf_loss = conf_loss[pos_idx]
        neg_conf_loss = conf_loss[~pos_idx]

        # Hard Negative Mining
        num_pos = pos_idx.sum().item()
        num_neg = min(self.neg_pos_ratio * num_pos, (~pos_idx).sum().item())

        neg_conf_loss_sorted, neg_idx = neg_conf_loss.sort(descending=True)
        hard_neg_idx = neg_idx[:num_neg]
        conf_loss_sum = pos_conf_loss.sum() + neg_conf_loss_sorted[:num_neg].sum()

        # 总损失归一化
        total_loss = (loc_loss + self.alpha * conf_loss_sum) / num_pos
        return total_loss


class DefaultBoxGenerator:
    def __init__(self, fmap_size, image_size, scales, aspect_ratios):
        self.fmap_size = fmap_size  # (fh, fw)
        self.image_size = image_size  # (H, W)
        self.scales = scales  # [s1, s2, ..., sn]
        self.aspect_ratios = aspect_ratios  # like [1.0, 2.0, 0.5]
        self.default_boxes = self._generate_default_boxes()

    def _generate_default_boxes(self):
        boxes = []
        fh, fw = self.fmap_size
        for i, j in itertools.product(range(fh), range(fw)):
            cx = (j + 0.5) / fw
            cy = (i + 0.5) / fh
            for scale in self.scales:
                for ar in self.aspect_ratios:
                    w = scale * np.sqrt(ar)
                    h = scale / np.sqrt(ar)
                    boxes.append([cx, cy, w, h])
        return torch.tensor(boxes)  # (N, 4) in [cx, cy, w, h] format


def compute_iou(boxes1, boxes2):
    """
    boxes1: (N, 4) [cx, cy, w, h]
    boxes2: (M, 4) [cx, cy, w, h]
    return: (N, M) IoU matrix
    """
    N, M = boxes1.size(0), boxes2.size(0)

    # Convert to corner format
    boxes1_xy = torch.cat([
        boxes1[:, :2] - boxes1[:, 2:] / 2,
        boxes1[:, :2] + boxes1[:, 2:] / 2
    ], dim=1)  # (N, 4) [xmin, ymin, xmax, ymax]

    boxes2_xy = torch.cat([
        boxes2[:, :2] - boxes2[:, 2:] / 2,
        boxes2[:, :2] + boxes2[:, 2:] / 2
    ], dim=1)  # (M, 4)

    iou = torch.zeros(N, M)

    for i in range(N):
        for j in range(M):
            x1 = max(boxes1_xy[i, 0], boxes2_xy[j, 0])
            y1 = max(boxes1_xy[i, 1], boxes2_xy[j, 1])
            x2 = min(boxes1_xy[i, 2], boxes2_xy[j, 2])
            y2 = min(boxes1_xy[i, 3], boxes2_xy[j, 3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)

            area1 = (boxes1_xy[i, 2] - boxes1_xy[i, 0]) * (boxes1_xy[i, 3] - boxes1_xy[i, 1])
            area2 = (boxes2_xy[j, 2] - boxes2_xy[j, 0]) * (boxes2_xy[j, 3] - boxes2_xy[j, 1])
            union = area1 + area2 - inter
            iou[i, j] = inter / union if union > 0 else 0.0

    return iou


def encode_offsets(gt_boxes, default_boxes):
    """
    gt_boxes: (M, 4) [cx, cy, w, h] ground truth
    default_boxes: (N, 4) [cx, cy, w, h]
    return: (N, 4) offsets
    """
    g_cx, g_cy, g_w, g_h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    d_cx, d_cy, d_w, d_h = default_boxes[:, 0], default_boxes[:, 1], default_boxes[:, 2], default_boxes[:, 3]

    offsets = torch.zeros_like(default_boxes)
    offsets[:, 0] = (g_cx - d_cx) / d_w
    offsets[:, 1] = (g_cy - d_cy) / d_h
    offsets[:, 2] = torch.log(g_w / d_w)
    offsets[:, 3] = torch.log(g_h / d_h)
    return offsets


def encode_ground_truth(gt_boxes, gt_labels, default_boxes, iou_threshold=0.5):
    # ensure gt_boxes and default_boxes are on the same device
    device = gt_boxes.device
    default_boxes = default_boxes.to(device)

    matched_gt, positive_mask, matched_gt_idx = match_boxes(gt_boxes, default_boxes, iou_threshold)
    positive_mask = positive_mask.to(device)  # 确保 positive_mask 在同一设备上
    matched_gt = matched_gt.to(device)  # 确保 matched_gt 在同一设备上
    matched_gt_idx = matched_gt_idx.to(device)  # 确保 matched_gt_idx 在同一设备上

    encoded_locs = encode_offsets(matched_gt, default_boxes)

    # 构造标签
    labels = torch.zeros(default_boxes.size(0), dtype=torch.long).to(device)
    labels[positive_mask] = gt_labels[matched_gt_idx[positive_mask]]  # ✅ 取正样本对应的gt标签
    return encoded_locs, labels, positive_mask


def match_boxes(gt_boxes, default_boxes, iou_threshold=0.5):
    """
    Returns:
      matched_gt: (N, 4)
      positive_mask: (N,)
      matched_gt_idx: (N,) 每个 default box 匹配的 gt 编号
    """
    iou = compute_iou(default_boxes, gt_boxes)  # (N, M)
    best_iou, best_gt_idx = iou.max(dim=1)  # 每个 default box 匹配哪个 gt

    # 强制匹配每个 gt 的最大 iou 的那个 default box
    _, forced_dbox_idx = iou.max(dim=0)  # 每个 gt 匹配哪个 default box
    positive_mask = best_iou >= iou_threshold
    positive_mask[forced_dbox_idx] = True

    matched_gt = gt_boxes[best_gt_idx]  # shape (N, 4)
    return matched_gt, positive_mask, best_gt_idx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def voc_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets


def train():
    # 初始化模型
    model = SimpleSSD(num_classes=21).to(device)
    criterion = MultiBoxLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 初始化 default boxes（只支持 38x38）
    default_box_generator = DefaultBoxGenerator(
        fmap_size=(37, 37),
        image_size=(300, 300),
        scales=[0.1],
        aspect_ratios=[1.0, 2.0, 0.5, 1.5]
    )
    default_boxes = default_box_generator.default_boxes.to(device)
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

    dataloader = DataLoader(
        dataset=VOCDataset(root="./data"),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    # 训练主循环
    EPOCHS = 10

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, boxes_list, labels_list in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images = images.to(device)
            gt_boxes = boxes_list[0].to(device)  # 只取第一个样本（batch_size=1）
            gt_labels = labels_list[0].to(device)

            # 匹配、编码
            encoded_locs, encoded_labels, positive_mask = encode_ground_truth(
                gt_boxes, gt_labels, default_boxes
            )

            with autocast():
                pred_locs, pred_scores = model(images)

                pred_locs = pred_locs.permute(0, 2, 3, 1).reshape(-1, 4)
                pred_scores = pred_scores.permute(0, 2, 3, 1).reshape(-1, 21)

                loss = criterion(pred_locs, pred_scores, encoded_locs, encoded_labels, positive_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()
