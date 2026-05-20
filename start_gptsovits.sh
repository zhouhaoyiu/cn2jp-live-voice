#!/bin/bash
echo "启动 GPT-SoVITS API Server v2..."
echo "参考音频等参数通过 API 请求传递，无需命令行指定"

# 预下载 NLTK 数据（处理中英混合文本必需，如 "大家好我是saki"）
cd "$(dirname "$0")/GPT-SoVITS"
PYTHON="${VIRTUAL_ENV}/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "检查 NLTK 数据..."
$PYTHON -c "
import nltk
for res in ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'punkt_tab']:
    try:
        nltk.data.find(f'taggers/{res}' if 'tagger' in res else f'tokenizers/{res}')
    except LookupError:
        print(f'  下载 NLTK 资源: {res}')
        nltk.download(res, quiet=True)
print('NLTK 数据检查完成')
" 2>/dev/null || echo "NLTK 检查跳过（不影响纯中文文本合成）"

echo "启动 API Server..."
$PYTHON api_v2.py -a 127.0.0.1 -p 9880
