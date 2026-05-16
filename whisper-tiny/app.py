from mlx_audio.stt import load as load_stt
from huggingface_hub import snapshot_download
import gradio as gr
import os
import tempfile

import mlx.core as mx

mx.set_default_device(mx.cpu)


MODEL_ID = "mlx-community/SenseVoiceSmall"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_ID.replace("/", "--"))

print(f"Downloading model {MODEL_ID} to {MODEL_PATH} ...")
snapshot_download(MODEL_ID, local_dir=MODEL_PATH)
print("Model ready.")

for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(key, None)

model = load_stt(MODEL_PATH)
print("Model loaded.")


def transcribe(audio_path, language, task, temperature):
    if not audio_path:
        return "请上传或录制音频。", ""

    try:
        lang = language if language else None
        result = model.generate(
            audio_path,
            language=lang,
            task=task,
            temperature=temperature,
        )

        text = result.text
        segments_info = ""
        if result.segments:
            for seg in result.segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                seg_text = seg.get("text", "")
                segments_info += f"[{start:.1f}s - {end:.1f}s] {seg_text}\n"

        detected_lang = result.language or (language or "未知")
        header = f"检测语言: {detected_lang}\n\n"
        return header + text, segments_info
    except Exception as e:
        return f"识别失败: {e}", ""


with gr.Blocks(title="Whisper Tiny ASR MLX") as demo:
    gr.Markdown(
        "# Whisper Tiny ASR (FP16 MLX)\n"
        f"Running `{MODEL_ID}` on Apple Silicon via MLX\n\n"
        "基于 Whisper Tiny 的语音识别，支持多语言转录与翻译。"
    )

    audio_input = gr.Audio(
        label="上传音频",
        type="filepath",
        sources=["microphone", "upload"],
    )

    with gr.Row():
        language = gr.Dropdown(
            choices=["", "zh", "en", "ja", "ko",
                     "de", "fr", "es", "ru", "it", "pt"],
            value="",
            label="语言（留空自动检测）",
            scale=2,
        )
        task = gr.Radio(
            choices=["transcribe", "translate"],
            value="transcribe",
            label="任务",
            scale=1,
        )
        temperature = gr.Slider(0.0, 1.0, value=0.0,
                                step=0.1, label="Temperature")

    transcribe_btn = gr.Button("识别", variant="primary")

    text_output = gr.Textbox(label="识别结果", lines=6)
    segments_output = gr.Textbox(label="分段详情", lines=8)

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input, language, task, temperature],
        outputs=[text_output, segments_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7862)
