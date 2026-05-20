#!/bin/bash
# ============================================
# GPT-SoVITS 启动脚本（使用独立 venv）
#
# 用法:
#   cd ~/Downloads/cn2jp-live-voice
#   bash scripts/start_gptsovits.sh
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GPTSOVITS_DIR="$PROJECT_DIR/GPT-SoVITS"
GPTSOVITS_VENV="$/venv"


cd "$GPTSOVITS_DIR"
echo "启动 GPT-SoVITS API Server (独立 venv, transformers<=4.46)..."
python3 api_v2.py -a 127.0.0.1 -p 9880