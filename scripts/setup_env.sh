#!/bin/bash
# ============================================
# 中文→日语实时语音转换系统 - macOS/Linux 环境安装
# 适用于 M4 Max 开发环境
# ============================================

set -e

echo "============================================"
echo "中文→日语实时语音转换系统 - 环境安装"
echo "============================================"

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到 conda！请先安装 Miniforge/Anaconda"
    exit 1
fi

ENV_NAME="cn2jp"
eval "$(conda shell.bash hook)"

# 创建/激活 conda 环境
echo "[1/5] 创建/激活 conda 环境: $ENV_NAME ..."
if conda info --envs 2>/dev/null | grep -q "^$ENV_NAME "; then
    echo "  环境 $ENV_NAME 已存在，直接激活"
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"
python --version

# 安装 PyTorch (CPU/MPS - macOS)
echo "[2/5] 安装 PyTorch (CPU/MPS)..."
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# 安装项目依赖
echo "[3/5] 安装项目依赖..."
pip install -r requirements.txt
pip install \
    faster-whisper==1.2.1 \
    ctranslate2==4.7.1 \
    transformers==5.8.0 \
    tokenizers==0.22.2 \
    peft==0.19.1 \
    accelerate==1.13.0 \
    sentencepiece==0.2.1 \
    huggingface-hub==1.14.0 \
    numpy==2.4.4 \
    pyaudio==0.2.14 \
    requests==2.34.0 \
    pykakasi==2.3.0 \
    pypinyin==0.55.0 \
    pyyaml==6.0.3

# 下载模型
echo "[4/5] 下载模型（可能需要较长时间）..."
python download_models.py --whisper-size small

# 搭建 GPT-SoVITS（独立 conda 环境）
echo "[5/5] 搭建 GPT-SoVITS 独立环境..."
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
