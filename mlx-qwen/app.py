import os
import gradio as gr
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_ID.replace("/", "--"))

print(f"Downloading model {MODEL_ID} to {MODEL_PATH} ...")
snapshot_download(MODEL_ID, local_dir=MODEL_PATH)
print("Model ready.")
model, tokenizer = load(MODEL_PATH)
print("Model loaded.")


def build_messages(message, history):
    messages = [{"role": "system", "content": "你是一个智能助手。"}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    return messages


def chat_stream(message, history, temperature, top_p, max_tokens, enable_thinking):
    messages = build_messages(message, history)

    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if not enable_thinking:
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    sampler = make_sampler(temp=temperature, top_p=top_p)
    response = ""
    for chunk in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        response += chunk.text
        yield response.replace("<|im_end|>", "").strip()


def user_send(message, history):
    if not message.strip():
        return "", history
    history = history + [{"role": "user", "content": message}]
    return "", history


def bot_respond(history, temperature, top_p, max_tokens, enable_thinking):
    if not history:
        return history
    user_message = history[-1]["content"]
    prev_history = history[:-1]
    history = history + [{"role": "assistant", "content": ""}]
    for partial in chat_stream(user_message, prev_history, temperature, top_p, max_tokens, enable_thinking):
        history[-1]["content"] = partial
        yield history


with gr.Blocks(title="Qwen3.5-0.8B MLX Chat") as demo:
    gr.Markdown(
        "# Qwen3.5-0.8B-OptiQ (4bit MLX)\n"
        f"Running `{MODEL_ID}` on Apple Silicon via MLX"
    )

    with gr.Row():
        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
        max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
        enable_thinking = gr.Checkbox(value=False, label="启用思考模式")

    chatbot = gr.Chatbot(label="对话", height=500)

    with gr.Row():
        msg_input = gr.Textbox(placeholder="输入你的问题...", show_label=False, scale=4)
        send_btn = gr.Button("发送", variant="primary", scale=1)
    clear_btn = gr.Button("清空对话")

    msg_input.submit(
        fn=user_send, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot],
    ).then(
        fn=bot_respond,
        inputs=[chatbot, temperature, top_p, max_tokens, enable_thinking],
        outputs=chatbot,
    )
    send_btn.click(
        fn=user_send, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot],
    ).then(
        fn=bot_respond,
        inputs=[chatbot, temperature, top_p, max_tokens, enable_thinking],
        outputs=chatbot,
    )
    clear_btn.click(fn=lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)
