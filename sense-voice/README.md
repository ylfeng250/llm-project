# MLX SenseVoice Small ASR Gradio Web Demo

在 Apple Silicon 上通过 MLX 运行 [SenseVoiceSmall](https://huggingface.co/mlx-community/SenseVoiceSmall)，使用 Gradio 提供语音识别 Web 界面。

![](./result.png)

## 模型简介

SenseVoice 是阿里通义实验室开源的语音识别模型，支持多语言转录、语种检测和情感识别。本仓库使用 `mlx-audio` 将模型转换为 MLX 格式，在 Apple Silicon 上高效推理。

主要特性：

- **50+ 语言**：支持中、英、日、韩等语言的自动语音转录
- **语种自动检测**：无需指定语言即可识别
- **情感识别**：可输出语音中的情感标注
- **快速推理**：模型轻量，适合本地实时转录

## 环境要求

- macOS Apple Silicon (M1/M2/M3/M4)
- Conda (Miniconda / Anaconda)
- ~1GB 磁盘空间（模型首次运行自动下载）

## 快速启动

```bash
# 1. 创建并激活 conda 环境
conda create -n sense-voice python=3.12 -y
conda activate sense-voice

# 2. 安装依赖
pip install mlx-audio gradio

# 3. 启动
python app.py
```

首次运行会自动从 HuggingFace 下载模型到项目目录下的 `models/` 文件夹，后续直接使用本地模型。

启动后浏览器打开 **http://localhost:7862** 即可使用。

## 使用方法

1. 上传音频文件或使用麦克风录制
2. 可选：指定语言（留空自动检测）
3. 点击「识别」

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 语言 | 自动检测 | 指定语言可提高准确率 |

## 关于 MLX Audio

MLX Audio 是基于 MLX 框架的音频模型推理库，支持 TTS、STT、STS 等多种任务。与 PyTorch + CUDA 的传统方案不同，MLX Audio 充分利用 Apple Silicon 的统一内存架构，数据在 CPU 和 GPU 之间零拷贝传输，特别适合本地单用户场景。

```
# PyTorch + CUDA 音频推理架构
┌───────────────────────────────┐
│        应用层 / 服务层         │
│  TTS / STT API Server         │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│     PyTorch + Transformers    │
│  - GPU Tensor 运算             │
│  - CUDA Kernel                │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│     CUDA / cuDNN / cuBLAS     │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│        NVIDIA GPU             │
│  独立显存 (HBM / GDDR)        │
│  CPU ↔ GPU 数据拷贝开销大     │
└───────────────────────────────┘

# MLX Audio 推理架构
┌───────────────────────────────┐
│        应用层 / 服务层         │
│  Gradio Web Demo / CLI        │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│         mlx-audio             │
│  - TTS / STT / STS 模型封装   │
│  - 量化推理支持               │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│              MLX              │
│  - Tensor / Autograd          │
│  - Metal GPU 后端             │
│  - 统一内存零拷贝              │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│        Apple Silicon          │
│  CPU + GPU + NPU              │
│  Unified Memory (统一内存)     │
└───────────────────────────────┘
```

## 常见问题

**Q: 模型下载慢？**

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

**Q: 使用代理时启动报 SOCKS 错误？**

程序在模型下载完成后会自动清除代理环境变量。如果仍有问题，可在启动前安装 socks 支持：
```bash
pip install "httpx[socks]"
```

**Q: 如何卸载环境？**

```bash
conda deactivate
conda remove -n sense-voice --all
```

**Q: 如何清除模型缓存？**

```bash
rm -rf models/mlx-community--SenseVoiceSmall
```
