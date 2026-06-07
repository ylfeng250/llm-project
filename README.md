# llm-project

基于 Apple Silicon MLX 的 LLM 应用合集。

- [MLX Qwen3.5-0.8B Gradio Web Demo](./mlx-qwen) — LLM 文本生成
- [MLX MOSS-TTS-Nano Gradio Web Demo](./moss-tts-nano) — TTS 声音克隆
- [MLX SenseVoice Small ASR Gradio Web Demo](./sense-voice) — ASR 语音识别
- [MLX YOLO26s-OptiQ Gradio Web Demo](./mlx-yolo) — 目标检测
- [MLX DeepSeek-OCR-2 Gradio Web Demo](./deepseek-ocr) — OCR 文字识别
- [播客生成工具 Podcast Generator](./podcast) — 双人播客对话生成 + 声音克隆 TTS

## 快速启动

每个项目目录下都有 `start.sh` 一键启动脚本，自动完成 conda 环境创建、依赖安装和服务启动：

```bash
# 例如
cd mlx-qwen && ./start.sh      # 浏览器打开 http://localhost:7860
cd moss-tts-nano && ./start.sh # 浏览器打开 http://localhost:7861
cd podcast && ./start.sh       # 浏览器打开 http://localhost:7865
```

| 项目 | 端口 | 环境名 |
|------|------|--------|
| MLX Qwen3.5-0.8B | 7860 | mlx-qwen |
| MOSS-TTS-Nano | 7861 | moss-tts-nano |
| SenseVoice ASR | 7862 | sense-voice |
| YOLO26s-OptiQ | 7863 | mlx-yolo |
| DeepSeek-OCR-2 | 7864 | deepseek-ocr |
| Podcast Generator | 7865 | podcast