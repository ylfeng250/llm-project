# MLX YOLO26s-OptiQ Gradio Web Demo

在 Apple Silicon 上通过 MLX 运行 [YOLO26s-OptiQ-6bit](https://huggingface.co/mlx-community/YOLO26s-OptiQ-6bit)，使用 Gradio 提供目标检测 Web 界面。


## 环境要求

- macOS Apple Silicon (M1/M2/M3/M4)
- Conda (Miniconda / Anaconda)
- ~10MB 磁盘空间（模型首次运行自动下载）

## 快速启动

```bash
# 1. 创建并激活 conda 环境
conda create -n mlx-yolo python=3.11 -y
conda activate mlx-yolo

# 2. 安装依赖
pip install mlx-optiq yolo-mlx gradio

# 3. 启动
python app.py
```

首次运行会自动从 HuggingFace 下载模型到项目目录下的 `models/` 文件夹，后续直接使用本地模型。

启动后浏览器打开 **http://localhost:7863** 即可使用。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 置信度阈值 (Conf) | 0.25 | 检测结果的最低置信度，低于此值的目标将被过滤 |
| 推理尺寸 (imgsz) | 640 | 输入图像分辨率，越大精度越高但速度越慢 |

## 关于模型

YOLO26 是 Ultralytics 发布的最新一代 YOLO 实时目标检测模型，采用 NMS-free 端到端检测和简化的 DFL-free 边界框回归。本项目使用 [yolo-mlx](https://pypi.org/project/yolo-mlx/) 在 MLX 框架上的纯实现，无需 PyTorch 运行时依赖。

### OptiQ 混合精度量化

本模型通过 [mlx-optiq](https://pypi.org/project/mlx-optiq/) 进行混合精度量化。与传统的"所有层统一 4-bit"不同，OptiQ 通过 KL 散度测量每个卷积层对量化的敏感度，然后使用贪心背包优化为每层分配最优位宽：

- 敏感层（检测头、特征金字塔）保持 8-bit 精度
- 鲁棒的骨干网络层使用 4-bit 精度

| 属性 | 值 |
|------|------|
| 目标 BPW | 6.0 |
| 实际 BPW | 5.97 |
| 4-bit 层数 | 11 |
| 8-bit 层数 | 115 |
| 原始大小 | 38.4 MB |
| 量化后大小 | 8.9 MB |
| 压缩比 | 4.3x |

### COCO128 基准结果

| 模型 | 总检测数 | 平均/图 |
|------|----------|---------|
| **OptiQ 6-bit** | **633** | **4.9** |
| 原始 FP32 | 681 | 5.3 |

检测差异: -48 (-7.0%)，在 4.3x 压缩下。

## 常见问题

**Q: 模型下载慢？**

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

**Q: 显存/内存不足？**

此模型为 6-bit 混合精度量化版本，约占用 ~9MB 内存，所有 Apple Silicon 机型均可流畅运行。

**Q: 如何卸载环境？**

```bash
conda deactivate
conda remove -n mlx-yolo --all
```

**Q: 如何清除模型缓存？**

```bash
rm -rf models/mlx-community--YOLO26s-OptiQ-6bit
```
