import json
import os
import re
import tempfile

import gradio as gr
import numpy as np
from huggingface_hub import snapshot_download
from mlx_audio.tts import load_model
from mlx_audio.audio_io import write as audio_write
from openai import OpenAI

MODEL_ID = "mlx-community/MOSS-TTS-Nano-100M"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_ID.replace("/", "--"))

SIBLING_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "moss-tts-nano", "models", MODEL_ID.replace("/", "--"),
)

if os.path.isdir(SIBLING_MODEL_PATH):
    MODEL_PATH = SIBLING_MODEL_PATH
    print(f"Using shared model from {MODEL_PATH}")
else:
    print(f"Downloading model {MODEL_ID} to {MODEL_PATH} ...")
    snapshot_download(MODEL_ID, local_dir=MODEL_PATH)
    print("Model ready.")

for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(key, None)

model = load_model(MODEL_PATH)
print("TTS Model loaded.")

# --- Voice Presets ---
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def scan_voices():
    """Scan voices/ directory for WAV files, return (choices, path_map)."""
    choices = ["（使用上传/录制）"]
    path_map = {}
    if os.path.isdir(VOICES_DIR):
        for fname in sorted(os.listdir(VOICES_DIR)):
            if fname.lower().endswith(".wav"):
                label = os.path.splitext(fname)[0]
                choices.append(label)
                path_map[label] = os.path.join(VOICES_DIR, fname)
    return choices, path_map


voice_choices, voice_path_map = scan_voices()
print(f"Found {len(voice_choices) - 1} preset voices.")

# Default preset voices (first available for Speaker A, second for Speaker B, fallback to first)
_preset_voice_labels = [c for c in voice_choices if c != "（使用上传/录制）"]
DEFAULT_VOICE_A = _preset_voice_labels[0] if len(_preset_voice_labels) >= 1 else voice_choices[0]
DEFAULT_VOICE_B = _preset_voice_labels[1] if len(_preset_voice_labels) >= 2 else (_preset_voice_labels[0] if _preset_voice_labels else voice_choices[0])

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = """你是一个专业的播客对话生成专家。你的任务是将用户提供的文章/文本转换成一段自然流畅的双人播客对话。

## 角色定义
- Speaker A（主持人）：负责开场、引导话题、提问、承上启下、总结收尾
- Speaker B（嘉宾/专家）：负责深入解读、提供见解、回答问题、分享细节

## 对话要求
1. 对话内容必须严格基于输入文本，不能编造原文中没有的事实或数据
2. 对话要自然口语化，像真实的播客节目，适当使用"嗯"、"其实"、"我觉得"等口语表达
3. Speaker A 提出好问题引导嘉宾，Speaker B 给出有深度的回答
4. 每个发言不宜过长，中文建议 30-150 字，便于语音合成
5. 对话轮次 8-20 轮，根据内容丰富程度灵活调整
6. 开场要有简短介绍，结尾要有总结收尾

## 输出格式
必须输出严格合法的 JSON 数组，每个元素包含 speaker 和 text 字段：
```json
[
  {"speaker": "A", "text": "欢迎收听今天的节目，今天我们来聊聊..."},
  {"speaker": "B", "text": "谢谢主持人，这个话题确实很有意思。"}
]
```
只输出 JSON 数组，不要包含任何 markdown 标记或额外解释。"""


def generate_dialogue(text, api_key, model_name):
    try:
        if not text or not text.strip():
            yield [], "请输入源文本。"
            return

        key = (api_key or "").strip() or os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            yield [], "错误：未设置 DeepSeek API Key。请在界面中输入或设置环境变量 DEEPSEEK_API_KEY。"
            return

        client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请将以下文章转换为一段双人播客对话：\n\n{text}"},
            ],
            temperature=0.7,
            max_tokens=4096,
            stream=True,
            extra_body={"thinking": {"type": "enabled"}},
        )

        full_content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                full_content += delta.content
                yield [], full_content

        if not full_content:
            yield [], "API 返回空内容，请重试。"
            return

        content = full_content.strip()
        content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE).strip()

        dialogue = json.loads(content)

        if not isinstance(dialogue, list):
            yield [], f"返回结果不是数组格式：\n{content}"
            return

        for item in dialogue:
            if not isinstance(item, dict) or "speaker" not in item or "text" not in item:
                yield [], f"对话格式不正确，每项需包含 speaker 和 text 字段：\n{content}"
                return
            if item["speaker"] not in ("A", "B"):
                yield [], f"speaker 字段必须为 A 或 B：\n{content}"
                return

        preview = format_dialogue_preview(dialogue)
        yield dialogue, preview

    except json.JSONDecodeError as e:
        raw = full_content if 'full_content' in locals() else 'N/A'
        yield [], f"JSON 解析失败: {e}\n\n原始响应:\n{raw}"
    except Exception as e:
        yield [], f"API 调用失败: {e}"


def format_dialogue_preview(dialogue):
    lines = []
    for turn in dialogue:
        label = "主持人" if turn["speaker"] == "A" else "嘉宾"
        lines.append(f"[{turn['speaker']}] {label}：{turn['text']}")
    return "\n\n".join(lines)


def parse_preview_to_dialogue(preview_text):
    if not preview_text:
        return None
    dialogue = []
    pattern = re.compile(r'^\[([AB])\]\s*(?:主持人|嘉宾)[：:]\s*(.*)')
    for line in preview_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            dialogue.append({"speaker": m.group(1), "text": m.group(2).strip()})
        else:
            return None
    return dialogue if dialogue else None


def toggle_voice_mode(mode):
    """Toggle visibility of upload/record components based on voice mode."""
    is_custom = mode == "自定义音色（上传/录制参考音频）"
    return (
        gr.update(visible=is_custom),  # ref_audio_a
        gr.update(visible=is_custom),  # ref_audio_b
    )


def synthesize_from_preview(preview_text, ref_audio_a, ref_audio_b,
                            voice_a_preset, voice_b_preset, voice_mode,
                            temperature, top_p, top_k, repetition_penalty, max_tokens,
                            silence_gap):
    # Resolve reference audio
    is_preset_mode = voice_mode == "预设音色（无需录制，开箱即用）"

    if is_preset_mode:
        ref_a = voice_path_map.get(voice_a_preset)
        ref_b = voice_path_map.get(voice_b_preset)
        if not ref_a:
            yield gr.update(), f"主持人音色预设无效: {voice_a_preset}"
            return
        if not ref_b:
            yield gr.update(), f"嘉宾音色预设无效: {voice_b_preset}"
            return
    else:
        ref_a = voice_path_map.get(voice_a_preset) if voice_a_preset != "（使用上传/录制）" else None
        ref_b = voice_path_map.get(voice_b_preset) if voice_b_preset != "（使用上传/录制）" else None
        if not ref_a:
            ref_a = ref_audio_a
        if not ref_b:
            ref_b = ref_audio_b
        if not ref_a or not ref_b:
            yield gr.update(), "请为两位说话人选择预设音色或上传参考音频。"
            return

    dialogue = parse_preview_to_dialogue(preview_text)
    if dialogue is None:
        yield gr.update(), (
            "对话脚本格式有误，请检查编辑。\n"
            "每行格式：[A] 主持人：文本 或 [B] 嘉宾：文本"
        )
        return

    # Filter out empty turns
    active_turns = [(i, t) for i, t in enumerate(dialogue) if t["text"].strip()]
    total = len(active_turns)
    label_map = {"A": "主持人", "B": "嘉宾"}

    yield gr.update(), f"开始合成播客音频，共 {total} 句..."

    try:
        all_segments = []
        sample_rate = None

        for step, (orig_idx, turn) in enumerate(active_turns):
            speaker = turn["speaker"]
            text = turn["text"].strip()

            yield gr.update(), f"🔊 正在合成第 {step + 1}/{total} 句 — {label_map[speaker]}：{text[:30]}..."

            ref_path = ref_a if speaker == "A" else ref_b

            results = model.generate(
                text=text,
                ref_audio=ref_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            )

            segment_frames = []
            for result in results:
                segment_frames.append(np.array(result.audio))
                if sample_rate is None:
                    sample_rate = result.sample_rate

            if segment_frames:
                segment_audio = np.concatenate(segment_frames)
                all_segments.append(segment_audio)

                if orig_idx < len(dialogue) - 1:
                    silence_len = int(sample_rate * silence_gap)
                    silence = np.zeros((silence_len, segment_audio.shape[1]), dtype=np.float32)
                    all_segments.append(silence)

        if not all_segments:
            yield gr.update(), "未生成任何音频。"
            return

        yield gr.update(), "🔧 正在拼接音频..."

        full_audio = np.concatenate(all_segments)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"podcast_{timestamp}.wav")
        audio_write(out_path, full_audio, sample_rate, format="wav")

        duration = len(full_audio) / sample_rate
        info = (
            f"✅ 合成完成 | "
            f"采样率: {sample_rate} Hz | "
            f"总时长: {duration:.1f} 秒 | "
            f"对话轮次: {total}\n"
            f"已保存: {out_path}"
        )
        yield out_path, info

    except Exception as e:
        yield gr.update(), f"音频合成失败: {e}"


with gr.Blocks(title="播客生成工具 Podcast Generator") as demo:
    gr.Markdown(
        """# 播客生成工具 Podcast Generator

将文章/文本转换为双人播客对话，内置预设音色，无需录音即可使用。

**工作流程**：① 选择音色（预设/自定义） → ② 输入文本 → ③ 生成对话脚本 → ④ 合成播客音频
"""
    )

    dialogue_state = gr.State([])

    # --- Voice Mode ---
    voice_mode = gr.Radio(
        choices=["预设音色（无需录制，开箱即用）", "自定义音色（上传/录制参考音频）"],
        value="预设音色（无需录制，开箱即用）",
        label="声音模式",
        info="预设模式使用内置音色，无需录音；自定义模式可上传或录制自己的声音进行克隆",
    )

    # --- Reference Audio ---
    with gr.Row():
        with gr.Column():
            voice_a_preset = gr.Dropdown(
                choices=voice_choices,
                value=DEFAULT_VOICE_A,
                label="主持人音色预设 (Speaker A)",
                info="预设模式下选择内置音色即可",
            )
            ref_audio_a = gr.Audio(
                label="上传或录制参考音频（自定义模式）",
                type="filepath",
                sources=["microphone", "upload"],
                visible=False,
            )
        with gr.Column():
            voice_b_preset = gr.Dropdown(
                choices=voice_choices,
                value=DEFAULT_VOICE_B,
                label="嘉宾音色预设 (Speaker B)",
                info="预设模式下选择内置音色即可",
            )
            ref_audio_b = gr.Audio(
                label="上传或录制参考音频（自定义模式）",
                type="filepath",
                sources=["microphone", "upload"],
                visible=False,
            )

    # --- Source Text ---
    source_text = gr.Textbox(
        label="源文本",
        placeholder="粘贴要转换为播客的文章或文本...",
        lines=8,
    )

    # --- API Config ---
    with gr.Accordion("API 配置", open=False):
        api_key = gr.Textbox(
            label="DeepSeek API Key（留空则使用环境变量 DEEPSEEK_API_KEY）",
            type="password",
            placeholder="sk-...",
        )
        model_selector = gr.Radio(
            choices=["deepseek-v4-flash", "deepseek-v4-pro"],
            value="deepseek-v4-flash",
            label="模型",
        )

    # --- Step 1: Generate Dialogue ---
    generate_btn = gr.Button("生成对话脚本", variant="primary")
    dialogue_preview = gr.Textbox(
        label="对话脚本预览（可编辑后重新合成）",
        lines=14,
        interactive=True,
        placeholder="对话脚本将在此显示，可编辑后重新合成...",
    )

    # --- Step 2: Synthesize Audio ---
    with gr.Accordion("TTS 参数", open=False):
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            top_k = gr.Slider(1, 100, value=25, step=1, label="Top-k")
        with gr.Row():
            max_tokens = gr.Slider(50, 2048, value=375, step=25, label="Max Tokens")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition Penalty")
            silence_gap = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="句间静音 (秒)")

    synthesize_btn = gr.Button("合成播客音频", variant="primary")
    info_output = gr.Textbox(
        label="合成状态",
        lines=2,
        interactive=False,
        placeholder="点击「合成播客音频」开始...",
    )
    audio_output = gr.Audio(label="播客音频", type="filepath")

    # --- Event Wiring ---
    generate_btn.click(
        fn=generate_dialogue,
        inputs=[source_text, api_key, model_selector],
        outputs=[dialogue_state, dialogue_preview],
    )

    voice_mode.change(
        fn=toggle_voice_mode,
        inputs=[voice_mode],
        outputs=[ref_audio_a, ref_audio_b],
    )

    synthesize_btn.click(
        fn=synthesize_from_preview,
        inputs=[
            dialogue_preview, ref_audio_a, ref_audio_b,
            voice_a_preset, voice_b_preset, voice_mode,
            temperature, top_p, top_k, repetition_penalty, max_tokens,
            silence_gap,
        ],
        outputs=[audio_output, info_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7865)
