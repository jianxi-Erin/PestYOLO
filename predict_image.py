# !pip install ultralytics
from ultralytics import YOLO
import os
import random
import matplotlib.pyplot as plt
import cv2

# -------------------------------
# 模型路径
# -------------------------------
best_model_path = "./IP102_YOLOv8/runs/IP102_detection/weights/best.pt"
model = YOLO(best_model_path)

# -------------------------------
# 验证集图片路径
# -------------------------------
val_img_dir = "./IP102_YOLOv8/val/images"
val_images = [f for f in os.listdir(val_img_dir) if f.endswith(".jpg")]

# 随机选择一张图片
img_name = random.choice(val_images)
img_path = os.path.join(val_img_dir, img_name)

# -------------------------------
# 预测
# -------------------------------
results = model.predict(
    source=img_path,   # 图片路径或文件夹
    imgsz=640,         # 输入尺寸
    conf=0.25,         # 置信度阈值
    show=False,        # 不用YOLO自带窗口显示
    save=False         # 不保存到磁盘
)

# -------------------------------
# 可视化结果
# -------------------------------
# YOLOv8 返回结果是 list，每个元素对应一张图片
result = results[0]

# 读取原图
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 绘制预测框和类别
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    label = f"{model.names[cls_id]} {conf:.2f}"

    # 绘制矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)

# 显示图片
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis("off")
plt.title(f"Prediction: {img_name}")
plt.show()
