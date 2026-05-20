#!/bin/bash
# ============================================
# GPT-SoVITS 启动脚本（使用独立 conda 环境）
#
# 用法:
#   cd ~/Downloads/cn2jp-live-voice
#   bash scripts/start_gptsovits.sh
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GPTSOVITS_DIR="$PROJECT_DIR/GPT-SoVITS"
ENV_NAME="gptsovits"

if ! command -v conda &> /dev/null; then
	echo "[错误] 未找到 conda，请先安装 Miniforge/Anaconda"
	exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$GPTSOVITS_DIR"
echo "启动 GPT-SoVITS API Server (conda: $ENV_NAME, transformers<=4.46)..."
python3 api_v2.py -a 127.0.0.1 -p 9880