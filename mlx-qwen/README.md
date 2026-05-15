# MLX Qwen3.5-0.8B Gradio Web Demo

在 Apple Silicon 上通过 MLX 运行 [Qwen3.5-0.8B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.5-0.8B-OptiQ-4bit)，使用 Gradio 提供 Web 聊天界面。
![app](./app.png)

## 环境要求

- macOS Apple Silicon (M1/M2/M3/M4)
- Conda (Miniconda / Anaconda)
- ~2GB 磁盘空间（模型首次运行自动下载）

## 快速启动

```bash
# 1. 创建并激活 conda 环境
conda create -n mlx-qwen python=3.11 -y
conda activate mlx-qwen

# 2. 安装依赖
pip install mlx-lm gradio

# 3. 启动
python app.py
```

首次运行会自动从 HuggingFace 下载模型到项目目录下的 `models/` 文件夹，后续直接使用本地模型。

启动后浏览器打开 **http://localhost:7860** 即可使用。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Max Tokens | 1024 | 单次回复最大生成长度 |
| Temperature | 0.7 | 越高越随机，0 为贪心解码 |
| Top-p | 0.9 | 核采样概率阈值 |
| Enable Thinking | 关闭 | 开启后模型会先输出思考过程再给答案 |



## 关于 MLX

MLX 是由苹果公司发布的在 Apple Silicon 芯片上的深度学习框架。与 PyTorch 等传统框架不同，MLX 充分利用了 Apple M 系列芯片的硬件优势，将数据维护在共享内存中，不需要频繁地在 CPU 和 GPU 之间传输数据，这样的共享内存架构大大提升了模型推理的效率。

### 与 vLLM 的区别

MLX-LM 是基于 MLX 框架构建的用于大模型推理的库，支持多种大语言模型（LLM）的部署和使用。通过 MLX-LM，用户可以在搭载 Apple M 系列芯片的设备上高效地运行大模型，实现本地推理。相比于 vLLM 等推理框架，架构区别如下所示：

```
# vLLM 架构示意图
┌───────────────────────────────┐
│        应用层 / 服务层         │
│  OpenAI-compatible API        │
│  (chat / completions)         │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│            vLLM               │
│  - Continuous Batching        │
│  - PagedAttention (KV Cache)  │
│  - 高并发 / 高吞吐             │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│     CUDA / Triton / NCCL      │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│        NVIDIA GPU             │
│  A100 / H100 / RTX            │
│  独立显存 (HBM / GDDR)        │
└───────────────────────────────┘
```

```
# MLX-LM 架构示意图
┌───────────────────────────────┐
│        用户接口层             │
│  CLI / Chat / Script          │
│  (mlx_lm.generate / chat)     │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│            mlx-lm             │
│  - 模型封装                   │
│  - 4bit / 8bit 推理            │
│  - 单用户 / 本地               │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│              MLX              │
│  - Tensor / Autograd          │
│  - Metal 后端                 │
└───────────────▲───────────────┘
                │
┌───────────────┴───────────────┐
│        Apple Silicon          │
│  CPU + GPU + NPU              │
│  Unified Memory (统一内存)     │
└───────────────────────────────┘
```

可以看到，vLLM 主要针对 NVIDIA GPU 进行优化，适合高并发和高吞吐的场景，而 MLX-LM 则充分利用了 Apple Silicon 的统一内存架构，适合单用户的本地推理需求。在 Apple M 系列芯片上使用 MLX-LM，可以实现高效的大模型推理体验。

具体使用差异对比如下表所示：

| 特性               | vLLM                          | MLX-LM                        |
|--------------------|-------------------------------|-------------------------------|
| 目标硬件           | NVIDIA GPU                    | Apple M 系列芯片              |
| 内存架构           | 独立显存 (HBM / GDDR)            | 统一内存 (Unified Memory)     |
| 并发支持           | 高并发 / 高吞吐                 | 单用户 / 本地                   |
| 模型量化         | 支持，但需自行实现               | 原生支持 4bit / 8bit 推理    |
| 框架依赖           | CUDA / Triton / NCCL           | MLX (Metal 后端)               |
| 使用复杂度         | 较高，需要配置环境和依赖       | 较低，适合本地快速部署         |

## 性能评测

运行 `benchmark.py` 可对模型在本地 Apple Silicon 上的推理效率进行评测：

```bash
python benchmark.py
```

评测指标：

| 指标 | 说明 |
|------|------|
| TTFT | 首 Token 延迟，越低交互响应越快 |
| TPOT | 每 Token 平均延迟 |
| TPS | 生成吞吐量 (tokens/s) |
| Prefill | Prompt 预填充吞吐量 (tokens/s) |
| 峰值内存 | 推理过程最大内存占用 |

以下为 **M1 Pro** 上的实测结果（4-bit 量化，贪心解码，每场景重复 3 次取均值）：

| 场景 | Prompt tokens | Gen tokens | TTFT (ms) | TPOT (ms) | TPS (tok/s) | Prefill (tok/s) | 峰值内存 (GB) |
|------|--------------|------------|-----------|-----------|-------------|-----------------|--------------|
| 短文本 | 27 | 23 | 249.1 | 6.1 | 58.1 | 486.5 | 0.78 |
| 中文本 | 35 | 512 | 262.1 | 6.3 | 145.9 | 499.0 | 0.78 |
| 长文本 | 39 | 220 | 257.3 | 6.2 | 135.8 | 562.5 | 0.78 |
| 代码 | 34 | 512 | 315.3 | 6.3 | 143.6 | 365.6 | 0.78 |
| 推理 | 50 | 131 | 270.5 | 6.2 | 120.5 | 638.8 | 0.81 |
| **平均** | — | — | **270.9** | **6.2** | **120.8** | — | **0.81** |


![benchmark_result](benchmark_result.png)

评测结果会同时保存到 `benchmark_result.json`。

## 常见问题

**Q: 模型下载慢？**

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

**Q: 显存/内存不足？**

此模型为 4-bit 量化版本，约占用 ~1GB 内存，M1 8GB 机型可流畅运行。

**Q: 如何卸载环境？**

```bash
conda deactivate
conda remove -n mlx-qwen --all
```

**Q: 如何清除模型缓存？**

```bash
rm -rf models/mlx-community--Qwen3.5-0.8B-OptiQ-4bit
```
