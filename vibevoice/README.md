# VibeVoice-Realtime-0.5B (MLX)

Microsoft 实时 TTS 模型，基于 Apple Silicon MLX 运行。

## 模型信息

- **原始模型**: [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- **MLX 版本**: [mlx-community/VibeVoice-Realtime-0.5B-5bit](https://huggingface.co/mlx-community/VibeVoice-Realtime-0.5B-5bit)
- **参数量**: 0.5B（5-bit 量化后约 0.2B）
- **采样率**: 24 kHz
- **首音延迟**: ~300ms
- **最大生成长度**: ~10 分钟

## 预设音色

| 音色 | 类型 |
|------|------|
| Carter | 男声 |
| Davis | 男声 |
| Emma | 女声 |
| Frank | 男声 |
| Grace | 女声 |
| Mike | 男声 |

## 快速启动

```bash
chmod +x start.sh && ./start.sh
```

浏览器打开 http://localhost:7866

## 依赖

- Python 3.12
- mlx-audio
- gradio

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Temperature | 0.8 | 控制生成随机性，越高越多样 |
| Top-p | 0.95 | 核采样阈值 |
| Top-k | 25 | 采样时保留概率最高的 k 个 token |
| Max Tokens | 375 | 最大生成 token 数 |
| Repetition Penalty | 1.2 | 重复惩罚系数 |
