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


def transcribe(audio_path, language):
    if not audio_path:
        return "请上传或录制音频。"

    try:
        kwargs = {}
        if language:
            kwargs["language"] = language
        result = model.generate(audio_path, **kwargs)

        detected_lang = getattr(result, "language", None) or language or "未知"
        header = f"检测语言: {detected_lang}\n\n"
        return header + result.text
    except Exception as e:
        return f"识别失败: {e}"


with gr.Blocks(title="SenseVoice Small ASR MLX") as demo:
    gr.Markdown(
        "# SenseVoice Small ASR (MLX)\n"
        f"Running `{MODEL_ID}` on Apple Silicon via MLX\n\n"
        "基于 SenseVoice Small 的语音识别，支持中英日韩等 50+ 语言自动检测。"
    )

    audio_input = gr.Audio(
        label="上传音频",
        type="filepath",
        sources=["microphone", "upload"],
    )

    language = gr.Dropdown(
        choices=["", "zh", "en", "ja", "ko",
                 "de", "fr", "es", "ru", "it", "pt"],
        value="",
        label="语言（留空自动检测）",
    )

    transcribe_btn = gr.Button("识别", variant="primary")

    text_output = gr.Textbox(label="识别结果", lines=8)

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input, language],
        outputs=text_output,
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7862)
