#!/bin/bash
# ============================================
# 中文→日语实时语音转换系统 - macOS/Linux 环境安装
# 适用于 M4 Max 开发环境
# ============================================

set -e

echo "============================================"
echo "中文→日语实时语音转换系统 - 环境安装"
echo "============================================"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3，请安装 Python 3.10+"
    exit 1
fi

PYTHON="python3"
$PYTHON --version

# 创建虚拟环境
echo "[1/5] 创建 Python 虚拟环境..."
$PYTHON -m venv venv
source venv/bin/activate

# 安装 PyTorch (CPU/MPS - macOS)
echo "[2/5] 安装 PyTorch (CPU/MPS)..."
pip install torch torchvision torchaudio

# 安装项目依赖
echo "[3/5] 安装项目依赖..."
pip install -r requirements.txt

# 下载模型
echo "[4/5] 下载模型（可能需要较长时间）..."
python download_models.py --whisper-size small

# 搭建 GPT-SoVITS（独立 venv，因 transformers 版本冲突）
echo "[5/5] 搭建 GPT-SoVITS 独立环境..."
python setup_gptsovits.py
bash scripts/setup_gptsovits_env.sh

echo ""
echo "============================================"
echo "安装完成！"
echo ""
echo "使用步骤:"
echo "  1. 录制参考音频: reference_audio/my_voice.wav"
echo "  2. 启动 GPT-SoVITS: bash scripts/start_gptsovits.sh"
echo "  3. 启动管道: python main.py --env m4max"
echo "============================================"
