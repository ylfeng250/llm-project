#!/bin/bash
set -e

ENV_NAME="mlx-yolo"
PYTHON_VERSION="3.11"
PORT="7863"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== MLX YOLO26s-OptiQ Gradio Web Demo ==="

# Check conda
if ! command -v conda &>/dev/null; then
    echo "错误: 未找到 conda，请先安装 Miniconda/Anaconda。"
    exit 1
fi

# Create env if not exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo "安装依赖..."
    conda run -n "${ENV_NAME}" pip install mlx-optiq yolo-mlx gradio
    echo "环境创建完成。"
fi

# Start
echo "启动 MLX YOLO26s-OptiQ..."
echo "浏览器打开 http://localhost:${PORT}"
cd "${SCRIPT_DIR}"
conda run -n "${ENV_NAME}" python app.py
