import json
import os
import time

import gradio as gr
from huggingface_hub import snapshot_download
from mlx_vlm import load as load_vlm, generate as vlm_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_ID = "mlx-community/DeepSeek-OCR-2-4bit"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_ID.replace("/", "--"))

print(f"Downloading model {MODEL_ID} to {MODEL_PATH} ...")
snapshot_download(MODEL_ID, local_dir=MODEL_PATH)
print("Model ready.")

# The downloaded model's processor_config.json and tokenizer_config.json
# reference "DeepseekVLV2Processor" (PyTorch class), but mlx-vlm registers
# "DeepseekOCR2Processor" (MLX class). Fix the class name so AutoProcessor
# can resolve it correctly.
for cfg_name in ("processor_config.json", "tokenizer_config.json"):
    cfg_path = os.path.join(MODEL_PATH, cfg_name)
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        changed = False
        if cfg.get("processor_class") == "DeepseekVLV2Processor":
            cfg["processor_class"] = "DeepseekOCR2Processor"
            changed = True
        if changed:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(f"Fixed {cfg_name}: processor_class -> DeepseekOCR2Processor")

# Remove auto_map from config.json that references PyTorch modeling files.
# These are not needed for MLX inference and cause ImportError when
# transformers tries to dynamically import them (missing torch/addict/etc).
config_path = os.path.join(MODEL_PATH, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    changed = False
    if "auto_map" in cfg:
        del cfg["auto_map"]
        changed = True
    lang_cfg = cfg.get("language_config", {})
    if "auto_map" in lang_cfg:
        del lang_cfg["auto_map"]
        changed = True
    if changed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print("Fixed config.json: removed auto_map (PyTorch refs)")

for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(key, None)

model, processor = load_vlm(MODEL_PATH)
config = load_config(MODEL_PATH)
print("Model loaded.")


def ocr_image(image, prompt, max_tokens, temperature):
    if image is None:
        return "请上传图片。", ""

    try:
        messages = [{"role": "user", "content": "<image>" + prompt}]
        chat_prompt = apply_chat_template(processor, config, messages)

        t0 = time.time()
        result = vlm_generate(
            model,
            processor,
            prompt=chat_prompt,
            image=image,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
        elapsed = time.time() - t0

        text = result.text.strip()
        info = f"耗时: {elapsed:.2f}s"
        return text, info
    except Exception as e:
        return f"OCR 失败: {e}", ""


with gr.Blocks(title="DeepSeek-OCR-2 MLX") as demo:
    gr.Markdown(
        "# DeepSeek-OCR-2 (4-bit MLX)\n"
        f"Running `{MODEL_ID}` on Apple Silicon via MLX\n\n"
        "基于 DeepSeek-VL2 架构的 OCR 模型，支持中英文文档、表格、公式等文字识别。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="上传图片", type="filepath")
            prompt_input = gr.Textbox(
                label="提示词",
                value="请识别图片中的所有文字，并按原始格式输出。",
                lines=2,
            )
            with gr.Row():
                max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Temperature")
            ocr_btn = gr.Button("开始识别", variant="primary")

        with gr.Column(scale=1):
            result_output = gr.Textbox(label="识别结果", lines=20, interactive=False)
            info_output = gr.Textbox(label="信息", interactive=False)

    ocr_btn.click(
        fn=ocr_image,
        inputs=[image_input, prompt_input, max_tokens, temperature],
        outputs=[result_output, info_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7864)
