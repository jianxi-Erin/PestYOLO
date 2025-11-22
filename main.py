import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import tempfile
import shutil
import json
import time

# 加载预训练模型
model_path = "./IP102_YOLOv8/runs/IP102_detection/weights/best.pt"

def load_model():
    """加载YOLOv8模型"""
    try:
        model = YOLO(model_path)
        print("模型加载成功!")
        print(f"模型类别: {model.names}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

# 全局模型变量
model = load_model()

# 类别颜色映射
def get_color_map(num_classes):
    colors = []
    for i in range(num_classes):
        # 生成不同的颜色
        r = int((i * 50) % 255)
        g = int((i * 100) % 255)
        b = int((i * 150) % 255)
        colors.append((b, g, r))
    return colors

def predict_single_image(input_image, confidence_threshold=0.25):
    """单张图片预测"""
    if model is None:
        return "模型未加载成功，请检查模型路径", None
    
    try:
        # 处理不同类型的输入
        if isinstance(input_image, str):
            # 文件路径
            if not os.path.exists(input_image):
                return f"文件不存在: {input_image}", None
            image = cv2.imread(input_image)
            if image is None:
                return "无法读取图片文件", None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(input_image, np.ndarray):
            # numpy数组
            image = input_image.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 已经是RGB格式
                pass
            else:
                # 转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return "不支持的图片格式", None
        
        # 进行预测
        results = model.predict(
            source=image,
            conf=confidence_threshold,
            save=False,
            verbose=False
        )
        
        # 获取预测结果
        result = results[0]
        
        # 创建副本用于绘制
        image_with_boxes = image.copy()
        
        # 绘制检测框
        if result.boxes is not None and len(result.boxes) > 0:
            # 获取类别颜色
            num_classes = len(model.names)
            colors = get_color_map(num_classes)
            
            # 绘制检测框和标签
            for box in result.boxes:
                # 获取坐标和类别
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 绘制矩形框
                color = colors[cls % len(colors)]
                cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # 绘制标签
                label = f"{model.names[cls]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # 标签背景
                cv2.rectangle(image_with_boxes, 
                            (int(x1), int(y1 - label_size[1] - 5)), 
                            (int(x1 + label_size[0]), int(y1)), 
                            color, -1)
                # 标签文字
                cv2.putText(image_with_boxes, label, (int(x1), int(y1 - 5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 统计检测结果
        detection_count = len(result.boxes) if result.boxes is not None else 0
        result_text = f"检测到 {detection_count} 个目标"
        
        if detection_count > 0:
            class_counts = {}
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            result_text += "\n详细统计:"
            for class_name, count in class_counts.items():
                result_text += f"\n- {class_name}: {count}个"
        
        return result_text, image_with_boxes
        
    except Exception as e:
        return f"预测出错: {str(e)}", None

def predict_batch_images(input_files, confidence_threshold=0.25):
    """批量图片预测"""
    if model is None:
        return "模型未加载成功，请检查模型路径", []
    
    if not input_files:
        return "请选择图片文件", []
    
    try:
        # 创建临时输出文件夹
        output_folder = tempfile.mkdtemp(prefix="yolo_batch_")
        
        results = []
        processed_count = 0
        error_count = 0
        
        for file_info in input_files:
            try:
                file_path = file_info if isinstance(file_info, str) else file_info.name
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    error_count += 1
                    continue
                
                # 读取图片
                image = cv2.imread(file_path)
                if image is None:
                    print(f"无法读取图片: {file_path}")
                    error_count += 1
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 进行预测
                results_pred = model.predict(
                    source=image_rgb,
                    conf=confidence_threshold,
                    save=False,
                    verbose=False
                )
                
                result = results_pred[0]
                
                # 绘制检测结果
                image_with_boxes = image.copy()
                if result.boxes is not None and len(result.boxes) > 0:
                    num_classes = len(model.names)
                    colors = get_color_map(num_classes)
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        color = colors[cls % len(colors)]
                        cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        label = f"{model.names[cls]}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(image_with_boxes, 
                                    (int(x1), int(y1 - label_size[1] - 5)), 
                                    (int(x1 + label_size[0]), int(y1)), 
                                    color, -1)
                        cv2.putText(image_with_boxes, label, (int(x1), int(y1 - 5)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 保存结果图片
                filename = Path(file_path).name
                output_path = os.path.join(output_folder, f"pred_{filename}")
                # 转换回RGB保存
                image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_path, image_with_boxes_rgb)
                
                # 统计信息
                detection_count = len(result.boxes) if result.boxes is not None else 0
                results.append((output_path, detection_count))
                processed_count += 1
                
            except Exception as e:
                print(f"处理图片时出错: {e}")
                error_count += 1
                continue
        
        # 准备返回的图片列表
        result_images = [(path, f"检测到 {count} 个目标") for path, count in results]
        
        summary = f"批量预测完成！\n总文件数: {len(input_files)}\n成功处理: {processed_count}\n处理失败: {error_count}"
        
        return summary, result_images
        
    except Exception as e:
        return f"批量预测出错: {str(e)}", []

def create_interface():
    """创建GradIO界面"""
    with gr.Blocks(title="YOLOv8 害虫检测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 YOLOv8 害虫检测系统")
        gr.Markdown(f"使用模型: `{model_path}`")
        
        with gr.Tab("单张图片预测"):
            with gr.Row():
                with gr.Column():
                    single_image_input = gr.Image(
                        label="上传图片", 
                        type="filepath",
                        sources=["upload"],
                        height=300
                    )
                    single_confidence = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.25, 
                        label="置信度阈值", step=0.05
                    )
                    single_predict_btn = gr.Button("开始预测", variant="primary")
                
                with gr.Column():
                    single_result_text = gr.Textbox(
                        label="预测结果", 
                        lines=4,
                        placeholder="预测结果将显示在这里..."
                    )
                    single_result_image = gr.Image(
                        label="检测结果", 
                        interactive=False,
                        height=300
                    )
            
            single_predict_btn.click(
                fn=predict_single_image,
                inputs=[single_image_input, single_confidence],
                outputs=[single_result_text, single_result_image]
            )
        
        
        with gr.Tab("批量图片预测"):
            with gr.Row():
                with gr.Column():
                    batch_file_input = gr.File(
                        label="选择图片文件",
                        file_count="multiple",
                        file_types=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                        height=200
                    )
                    batch_confidence = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.25,
                        label="置信度阈值", step=0.05
                    )
                    batch_predict_btn = gr.Button("开始批量预测", variant="primary")
                
                with gr.Column():
                    batch_result_text = gr.Textbox(
                        label="批量预测结果",
                        lines=4,
                        placeholder="批量预测结果将显示在这里..."
                    )
                    batch_gallery = gr.Gallery(
                        label="检测结果预览",
                        show_label=True,
                        elem_id="gallery",
                        columns=3,
                        rows=2,
                        object_fit="contain",
                        height="auto"
                    )
            
            batch_predict_btn.click(
                fn=predict_batch_images,
                inputs=[batch_file_input, batch_confidence],
                outputs=[batch_result_text, batch_gallery]
            )
        
        with gr.Tab("系统信息"):
            gr.Markdown("### 模型信息")
            if model is not None:
                model_info = f"""
                - **模型路径**: {model_path}
                - **类别数量**: {len(model.names)}
                - **任务类型**: {model.task}
                
                ### 主要害虫类别:
                """
                # 显示前20个类别作为示例
                for i, name in list(model.names.items())[:20]:
                    model_info += f"- {i}: {name}\n"
                if len(model.names) > 20:
                    model_info += f"\n... 还有 {len(model.names) - 20} 个类别"
            else:
                model_info = "模型加载失败，请检查模型路径"
            
            gr.Markdown(model_info)
            
            gr.Markdown("### 使用说明")
            gr.Markdown("""
            1. **单张图片预测**: 上传单张图片进行害虫检测
            2. **摄像头预测**: 使用摄像头实时捕获画面进行检测
            3. **批量图片预测**: 选择多张图片文件进行批量检测
            4. **置信度阈值**: 调整检测的敏感度，值越高要求越严格
            
            **注意**: 
            - 首次使用摄像头需要授权浏览器访问摄像头权限
            - 批量处理大量图片可能需要较长时间
            """)
    
    return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_interface()
    
    # 启动GradIO应用
    print("启动YOLOv8害虫检测Web界面...")
    print("请在浏览器中访问: http://localhost:7860")
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True
        )
    except Exception as e:
        print(f"启动失败: {e}")
        print("尝试使用默认设置启动...")
        demo.launch(share=False, inbrowser=True)