#!/usr/bin/env python3
"""
GPT-SoVITS 环境搭建脚本
自动克隆 GPT-SoVITS 仓库并安装依赖
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("setup_gptsovits")
logging.basicConfig(level=logging.INFO)

GPTSOVITS_REPO = "https://github.com/RVC-Boss/GPT-SoVITS.git"
GPTSOVITS_DIR = "GPT-SoVITS"


def run_cmd(cmd: str, cwd: str = None):
    """运行命令"""
    logger.info(f"执行: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        logger.error(f"命令执行失败 (code={result.returncode}): {cmd}")
        return False
    return True


def clone_repo():
    """克隆 GPT-SoVITS 仓库"""
    if Path(GPTSOVITS_DIR).exists():
        logger.info(f"{GPTSOVITS_DIR} 目录已存在，跳过克隆")
        return True

    logger.info(f"克隆 GPT-SoVITS: {GPTSOVITS_REPO}")
    return run_cmd(f"git clone --depth 1 {GPTSOVITS_REPO}")


def install_dependencies():
    """安装 GPT-SoVITS 依赖"""
    req_file = Path(GPTSOVITS_DIR) / "requirements.txt"
    if req_file.exists():
        logger.info("安装 GPT-SoVITS 依赖...")
        run_cmd(f"{sys.executable} -m pip install -r requirements.txt", cwd=GPTSOVITS_DIR)
    else:
        logger.warning("未找到 requirements.txt，请手动安装依赖")


def download_nltk_data():
    """下载 NLTK 数据（GPT-SoVITS 处理混合语言文本时需要）"""
    logger.info("下载 NLTK 数据（GPT-SoVITS 处理英文/混合文本时需要）...")
    try:
        import nltk
        # GPT-SoVITS 处理含英文字母的文本时需要这些资源
        resources = [
            "averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger",
            "punkt_tab",
            "cmudict",
        ]
        for res in resources:
            try:
                nltk.data.find(f"taggers/{res}") if "tagger" in res else None
            except LookupError:
                logger.info(f"  下载 NLTK 资源: {res}")
                nltk.download(res, quiet=True)
        logger.info("NLTK 数据下载完成")
    except ImportError:
        logger.warning("NLTK 未安装，跳过数据下载")
        logger.warning("如果 GPT-SoVITS 报 NLTK 错误，请运行:")
        logger.warning("  pip install nltk && python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")
    except Exception as e:
        logger.warning(f"NLTK 数据下载失败: {e}")
        logger.warning("如果 GPT-SoVITS 报 NLTK 错误，请手动运行:")
        logger.warning("  python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")


def download_pretrained_models():
    """下载 GPT-SoVITS 预训练模型"""
    logger.info("下载 GPT-SoVITS 预训练模型...")
    # GPT-SoVITS v2 预训练模型
    models_dir = Path(GPTSOVITS_DIR) / "GPT_SoVITS" / "pretrained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 优先使用项目自带的下载脚本（支持 HF 镜像自动切换）
    model_download_script = Path("scripts") / "download_gptsovits_models.py"
    if model_download_script.exists():
        logger.info("使用专用下载脚本（支持国内镜像）...")
        run_cmd(f"{sys.executable} scripts/download_gptsovits_models.py")
    else:
        # 回退到 GPT-SoVITS 自带的下载脚本
        download_script = Path(GPTSOVITS_DIR) / "download.py"
        if download_script.exists():
            logger.info("使用 GPT-SoVITS 自带下载脚本...")
            run_cmd(f"{sys.executable} download.py", cwd=GPTSOVITS_DIR)
        else:
            logger.warning("未找到下载脚本，请手动下载模型:")
            logger.warning("  python3 scripts/download_gptsovits_models.py")


def create_start_script():
    """创建启动脚本"""
    # Windows
    # 注意: api_v2.py 只接受 -a 和 -p 参数
    # 参考音频等参数通过 API 请求体传递（已由 tts.py 模块自动处理）
    bat_content = """@echo off
echo 启动 GPT-SoVITS API Server v2...
echo 参考音频等参数通过 API 请求传递，无需命令行指定
cd /d "%~dp0GPT-SoVITS"
python api_v2.py -a 127.0.0.1 -p 9880
pause
"""
    with open("start_gptsovits.bat", "w", encoding="utf-8") as f:
        f.write(bat_content)

    # Linux/macOS - 使用 python3 兼容 macOS
    sh_content = """#!/bin/bash
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
"""
    with open("start_gptsovits.sh", "w", encoding="utf-8") as f:
        f.write(sh_content)
    os.chmod("start_gptsovits.sh", 0o755)

    logger.info("已创建启动脚本: start_gptsovits.bat / start_gptsovits.sh")


def main():
    logger.info("=" * 60)
    logger.info("GPT-SoVITS 环境搭建")
    logger.info("=" * 60)

    # Step 1: 克隆仓库
    logger.info("[1/5] 克隆 GPT-SoVITS 仓库...")
    if not clone_repo():
        logger.error("克隆失败，请检查网络连接")
        return

    # Step 2: 安装依赖
    logger.info("[2/5] 安装依赖...")
    install_dependencies()

    # Step 3: 下载预训练模型
    logger.info("[3/5] 下载预训练模型...")
    download_pretrained_models()

    # Step 4: 下载 NLTK 数据（处理中英混合文本必需）
    logger.info("[4/5] 下载 NLTK 数据...")
    download_nltk_data()

    # Step 5: 创建启动脚本
    logger.info("[5/5] 创建启动脚本...")
    create_start_script()

    logger.info("=" * 60)
    logger.info("GPT-SoVITS 搭建完成！")
    logger.info("")
    logger.info("使用步骤:")
    logger.info("  1. 录制 5-15 秒自己的语音，保存为 reference_audio/my_voice.wav")
    logger.info("  2. 修改 start_gptsovits.bat/sh 中的参考音频文本参数")
    logger.info("  3. 运行 start_gptsovits.bat/sh 启动 API 服务")
    logger.info("  4. 运行 python main.py 启动语音转换管道")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
