#!/bin/bash
# ============================================
# GPT-SoVITS 独立 conda 环境搭建脚本
#
# 原因: HY-MT 翻译模型需要 transformers>=4.56.0，
#       但 GPT-SoVITS 与新版 transformers 不兼容
#       (HybridCache ImportError / tensor维度不匹配)，
#       所以需要独立 conda 环境。
#
# 用法:
#   cd ~/Downloads/cn2jp-live-voice
#   bash scripts/setup_gptsovits_env.sh
#
# 启动 GPT-SoVITS:
#   bash scripts/start_gptsovits.sh
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GPTSOVITS_DIR="$PROJECT_DIR/GPT-SoVITS"
ENV_NAME="gptsovits"

echo "============================================"
echo "GPT-SoVITS 独立 conda 环境搭建"
echo "============================================"
echo "项目目录: $PROJECT_DIR"
echo "GPT-SoVITS: $GPTSOVITS_DIR"
echo "conda 环境: $ENV_NAME"
echo ""

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到 conda！"
    echo "请先安装 Miniforge 或 Anaconda:"
    echo "  macOS:  brew install miniforge"
    echo "  Linux:  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    exit 1
fi

# 检查 GPT-SoVITS 目录
if [ ! -d "$GPTSOVITS_DIR" ]; then
    echo "[1/5] 克隆 GPT-SoVITS 仓库..."
    git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git "$GPTSOVITS_DIR"
else
    echo "[1/5] GPT-SoVITS 目录已存在，跳过克隆"
fi

# 创建 conda 环境
echo ""
echo "[2/5] 创建 conda 环境: $ENV_NAME ..."
if conda info --envs 2>/dev/null | grep -q "^$ENV_NAME "; then
    echo "  环境 $ENV_NAME 已存在，跳过创建"
else
    conda create -n "$ENV_NAME" python=3.11 -y
    echo "  创建完成"
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 安装 PyTorch (CPU版 for macOS, CUDA版 for Linux)
echo ""
echo "[3/5] 安装 PyTorch + GPT-SoVITS 依赖..."
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: MPS 加速
    pip install --quiet torch torchvision torchaudio
else
    # Linux: CUDA 加速
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# 安装 GPT-SoVITS 依赖（限制 transformers 版本）
pip install --quiet 'transformers==4.45.0' 'peft>=0.12.0,<0.14.0'

REQ_FILE="$GPTSOVITS_DIR/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    pip install --quiet -r "$REQ_FILE" || true
else
    pip install --quiet scipy numpy librosa soundfile
fi

# 关键: 安装 GPT-SoVITS 日语/英文处理依赖
pip install --quiet pyopenjtalk nltk pypinyin

# 下载预训练模型
echo ""
echo "[4/5] 下载 GPT-SoVITS 预训练模型..."
PRETRAINED_DIR="$GPTSOVITS_DIR/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained"
if [ -d "$PRETRAINED_DIR" ]; then
    echo "  预训练模型已存在，跳过下载"
else
    DOWNLOAD_SCRIPT="$GPTSOVITS_DIR/download.py"
    if [ -f "$DOWNLOAD_SCRIPT" ]; then
        echo "  运行 GPT-SoVITS 下载脚本..."
        python3 "$DOWNLOAD_SCRIPT" || echo "  [警告] 自动下载失败，请手动下载（见下方说明）"
    else
        echo "  [警告] 未找到 download.py，需要手动下载预训练模型"
    fi
fi

# 再次检查模型是否下载成功
if [ ! -d "$PRETRAINED_DIR" ]; then
    echo ""
    echo "  ⚠️  预训练模型未下载！需要手动操作："
    echo "  "
    echo "  方法1: 使用 GPT-SoVITS 自带下载"
    echo "    cd $GPTSOVITS_DIR"
    echo "    conda activate $ENV_NAME"
    echo "    python3 download.py"
    echo ""
    echo "  方法2: 手动从 HuggingFace 下载"
    echo "    https://huggingface.co/lj1995/GPT-SoVITS"
    echo "    下载 gsv-v2final-pretrained 目录到:"
    echo "    $PRETRAINED_DIR"
    echo ""
    echo "  方法3: 国内镜像"
    echo "    https://hf-mirror.com/lj1995/GPT-SoVITS"
fi

# 下载 NLTK 数据
echo ""
echo "[5/5] 下载 NLTK 数据..."
python3 -c "
import nltk
for res in ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'punkt_tab', 'cmudict']:
    try:
        nltk.data.find(f'taggers/{res}' if 'tagger' in res else f'corpora/{res}' if 'cmudict' in res else f'tokenizers/{res}')
        print(f'  ✓ {res} 已存在')
    except LookupError:
        print(f'  ⬇ {res} 下载中...')
        nltk.download(res, quiet=True)
print('NLTK 数据检查完成')
" || echo "  NLTK 下载失败，可稍后手动运行: python3 main.py --fix-nltk"

conda deactivate

echo ""
echo "============================================"
echo "GPT-SoVITS 环境搭建完成！"
echo ""
echo "启动方式:"
echo "  bash scripts/start_gptsovits.sh"
echo ""
echo "手动启动:"
echo "  conda activate $ENV_NAME"
echo "  cd $GPTSOVITS_DIR"
echo "  python3 api_v2.py -a 127.0.0.1 -p 9880"
echo "============================================"
