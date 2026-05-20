#!/bin/bash
set -e

echo "=== Step 1: Removing large files from Git history ==="
git filter-repo --path "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin" --invert-paths
git filter-repo --path "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" --invert-paths
git filter-repo --path "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth" --invert-paths

echo "=== Step 2: Configuring Git LFS ==="
git lfs install
git lfs track "*.bin" "*.ckpt" "*.pth" "*.pt" "*.onnx" "*.pb" "*.h5"
git add .gitattributes
git commit -m "Configure Git LFS for AI model files"

echo "=== Step 3: Done ==="
echo ""
echo "✅ 历史清理完成"
echo "👉 现在请将模型文件放回原位置，然后执行："
echo "   git add GPT_SoVITS/pretrained_models/"
echo "   git commit -m 'Add pretrained models via LFS'"
echo "   git push --force-with-lease origin main"