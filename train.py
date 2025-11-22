# 安装 YOLOv8 官方库（如果未安装）
# !pip install ultralytics -q

# -------------------------------
# 导入
# -------------------------------
from ultralytics import YOLO
import os

# -------------------------------
# 配置
# -------------------------------
data_yaml = "./IP102_YOLOv8/data.yaml"  # 数据集 YAML
output_dir = "./IP102_YOLOv8/runs"     # 保存模型 checkpoint
epochs = 30                                   # 训练轮数
batch_size = 16                               # 可根据显存调节
img_size = 640                                # 输入尺寸

# -------------------------------
# 创建 YOLO 模型（从头训练）
# -------------------------------
model = YOLO("yolov8n.pt")  # 也可选择 yolov8s.pt / yolov8m.pt

# -------------------------------
# 开始训练
# -------------------------------
model.train(
    data=data_yaml,
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    project=output_dir,
    name="IP102_detection",
    exist_ok=True,
    save_period=1,       # 每个 epoch 保存一次 checkpoint
    save=True,
    # 自动保存最佳模型
    # ultralytics >= 8.1 默认会保存 best.pt
)
