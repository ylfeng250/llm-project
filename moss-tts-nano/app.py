import os
import tempfile

import gradio as gr
import numpy as np
from huggingface_hub import snapshot_download
from mlx_audio.tts import load_model
from mlx_audio.audio_io import write as audio_write

MODEL_ID = "mlx-community/MOSS-TTS-Nano-100M"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_ID.replace("/", "--"))

print(f"Downloading model {MODEL_ID} to {MODEL_PATH} ...")
snapshot_download(MODEL_ID, local_dir=MODEL_PATH)
print("Model ready.")

# Clear proxy env after download to avoid SOCKS issues at runtime
for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(key, None)

model = load_model(MODEL_PATH)
print("Model loaded.")


def generate_tts(text, ref_audio_path, temperature, top_p, top_k, repetition_penalty, max_tokens):
    if not text.strip():
        return None, "请输入要合成的文本。"
    if not ref_audio_path:
        return None, "请上传参考音频（用于声音克隆）。"

    try:
        results = model.generate(
            text=text,
            ref_audio=ref_audio_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )
        audio_chunks = []
        sample_rate = None
        for result in results:
            audio_chunks.append(np.array(result.audio))
            if sample_rate is None:
                sample_rate = result.sample_rate

        if not audio_chunks:
            return None, "未生成音频。"

        audio = np.concatenate(audio_chunks)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_write(tmp.name, audio, sample_rate, format="wav")

        info = (
            f"采样率: {sample_rate} Hz | "
            f"时长: {result.audio_duration} | "
            f"实时率: {result.real_time_factor:.2f}x | "
            f"峰值内存: {result.peak_memory_usage:.2f} GB"
        )
        return tmp.name, info
    except Exception as e:
        return None, f"生成失败: {e}"


with gr.Blocks(title="MOSS-TTS-Nano MLX") as demo:
    gr.Markdown(
        "# MOSS-TTS-Nano (100M MLX)\n"
        f"Running `{MODEL_ID}` on Apple Silicon via MLX\n\n"
        "基于参考音频的声音克隆 TTS，支持中英等 20 种语言。"
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="合成文本",
            placeholder="输入要合成的文本...",
            lines=3,
            scale=4,
        )
        ref_audio = gr.Audio(
            label="参考音频（声音克隆）",
            type="filepath",
            sources=["upload"],
            scale=2,
        )

    with gr.Row():
        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        top_k = gr.Slider(1, 100, value=25, step=1, label="Top-k")
        max_tokens = gr.Slider(50, 2048, value=375, step=25, label="Max Tokens")

    with gr.Row():
        repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition Penalty")
        generate_btn = gr.Button("生成语音", variant="primary", scale=2)

    audio_output = gr.Audio(label="生成结果", type="filepath")
    info_output = gr.Textbox(label="信息", interactive=False)

    generate_btn.click(
        fn=generate_tts,
        inputs=[text_input, ref_audio, temperature, top_p, top_k, repetition_penalty, max_tokens],
        outputs=[audio_output, info_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7861)
