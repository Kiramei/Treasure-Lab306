from ultralytics import YOLO

# 加载YOLOv8模型，使用预训练模型
model = YOLO('yolov8n.pt')

# 使用测试图像进行推理（图像路径替换为你的测试图）
results = model(r'F:\WorkSpace\py\Experiment\od\Exp3\data\VOCdevkit\VOC2007\JPEGImages\000091.jpg', show=True, save=True)
# 训练模型（例如 voc.yaml 已定义训练数据集路径和类别）
model = YOLO('yolov8n.yaml')  # 从零开始训练新模型
model.train(data='voc.yaml', epochs=50, imgsz=640, batch=16, device=0)

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 使用验证集或自定义图像推理
model('test.jpg', show=True, save=True)
