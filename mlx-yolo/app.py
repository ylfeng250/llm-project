import os
import time
import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download
from optiq.models.yolo import load_quantized_yolo

MODEL_ID = "mlx-community/YOLO26s-OptiQ-6bit"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_ID.replace("/", "--"))

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

print(f"Downloading model {MODEL_ID} to {MODEL_PATH} ...")
snapshot_download(MODEL_ID, local_dir=MODEL_PATH)
print("Model ready.")
model = load_quantized_yolo(MODEL_PATH)
print("Model loaded.")


def detect(image, conf_threshold, img_size):
    if image is None:
        return None, "请上传图片"

    img_pil = Image.fromarray(image).convert("RGB")
    start = time.time()
    results = model.predict(img_pil, conf=conf_threshold, imgsz=img_size)
    elapsed = time.time() - start

    result = results[0]
    annotated = Image.fromarray(result.plot())

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        info = f"未检测到目标 | 耗时: {elapsed * 1000:.1f}ms | 推理尺寸: {img_size}"
    else:
        lines = [f"检测到 {len(boxes)} 个目标 | 耗时: {elapsed * 1000:.1f}ms | 推理尺寸: {img_size}", ""]
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class{cls_id}"
            conf = float(boxes.conf[i])
            lines.append(f"  {cls_name}: {conf:.2f}")
        info = "\n".join(lines)

    return annotated, info


with gr.Blocks(title="YOLO26s-OptiQ Object Detection") as demo:
    gr.Markdown(
        "# YOLO26s-OptiQ (6bit MLX) 目标检测\n"
        f"Running `{MODEL_ID}` on Apple Silicon via MLX"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传图片", type="numpy")
            with gr.Row():
                conf_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值 (Conf)")
                imgsz_slider = gr.Slider(320, 1280, value=640, step=32, label="推理尺寸 (imgsz)")
            detect_btn = gr.Button("检测", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="检测结果", type="pil")
            info_text = gr.Textbox(label="检测信息", lines=8)

    detect_btn.click(
        fn=detect,
        inputs=[input_image, conf_slider, imgsz_slider],
        outputs=[output_image, info_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7863)
