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
ENV_NAME = "gptsovits"


def run_cmd(cmd, cwd: str = None):
    """运行命令"""
    printable = " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd
    logger.info(f"执行: {printable}")
    try:
        result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd)
    except FileNotFoundError:
        logger.error(f"未找到命令: {cmd[0] if isinstance(cmd, (list, tuple)) else printable}")
        return False
    if result.returncode != 0:
        logger.error(f"命令执行失败 (code={result.returncode}): {printable}")
        return False
    return True


def conda_cmd(*args, cwd: str = None):
    return run_cmd(["conda", *args], cwd=cwd)


def conda_run(*args, cwd: str = None):
    return conda_cmd("run", "-n", ENV_NAME, *args, cwd=cwd)


def conda_env_exists():
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return any(line.split() and line.split()[0] == ENV_NAME for line in result.stdout.splitlines())


def clone_repo():
    """克隆 GPT-SoVITS 仓库"""
    if Path(GPTSOVITS_DIR).exists():
        logger.info(f"{GPTSOVITS_DIR} 目录已存在，跳过克隆")
        return True

    logger.info(f"克隆 GPT-SoVITS: {GPTSOVITS_REPO}")
    return run_cmd(f"git clone --depth 1 {GPTSOVITS_REPO}")


def ensure_conda_env():
    """创建 GPT-SoVITS 独立 conda 环境"""
    if not conda_cmd("--version"):
        logger.error("未找到 conda，请先安装 Miniforge/Anaconda")
        return False

    if conda_env_exists():
        logger.info(f"conda 环境 {ENV_NAME} 已存在，跳过创建")
        return True

    logger.info(f"创建 conda 环境: {ENV_NAME}")
    return conda_cmd("create", "-n", ENV_NAME, "python=3.11", "-y")


def install_dependencies():
    """安装 GPT-SoVITS 依赖"""
    if not ensure_conda_env():
        return False

    torch_cmd = [
        "python", "-m", "pip", "install",
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
    ]
    if os.name == "nt":
        torch_cmd += ["--index-url", "https://download.pytorch.org/whl/cu121"]

    if not conda_run(*torch_cmd):
        return False

    if not conda_cmd("install", "-n", ENV_NAME, "-c", "conda-forge", "ffmpeg>=6,<7", "-y"):
        return False

    if not conda_run(
        "python", "-m", "pip", "install",
        "transformers==4.45.0",
        "tokenizers==0.20.3",
        "peft==0.12.0",
        "accelerate==1.13.0",
        "sentencepiece==0.2.1",
        "huggingface-hub==0.36.2",
    ):
        return False

    req_file = Path(GPTSOVITS_DIR) / "requirements.txt"
    if req_file.exists():
        logger.info("安装 GPT-SoVITS 依赖...")
        if not conda_run("python", "-m", "pip", "install", "-r", "requirements.txt", cwd=GPTSOVITS_DIR):
            return False
    else:
        logger.warning("未找到 requirements.txt，请手动安装依赖")

    return conda_run(
        "python", "-m", "pip", "install",
        "numpy==1.26.4",
        "scipy==1.17.1",
        "librosa==0.10.2",
        "soundfile==0.13.1",
        "pyopenjtalk==0.4.1",
        "nltk==3.9.4",
        "pypinyin==0.55.0",
    )


def download_nltk_data():
    """下载 NLTK 数据（GPT-SoVITS 处理混合语言文本时需要）"""
    logger.info("下载 NLTK 数据（GPT-SoVITS 处理英文/混合文本时需要）...")
    code = """
import nltk
resources = {
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'punkt_tab': 'tokenizers/punkt_tab',
    'cmudict': 'corpora/cmudict',
}
for name, path in resources.items():
    try:
        nltk.data.find(path)
        print(f'  {name} 已存在')
    except LookupError:
        print(f'  下载 NLTK 资源: {name}')
        nltk.download(name, quiet=True)
print('NLTK 数据下载完成')
"""
    return conda_run("python", "-c", code)


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
        return conda_run("python", "scripts/download_gptsovits_models.py")
    else:
        # 回退到 GPT-SoVITS 自带的下载脚本
        download_script = Path(GPTSOVITS_DIR) / "download.py"
        if download_script.exists():
            logger.info("使用 GPT-SoVITS 自带下载脚本...")
            return conda_run("python", "download.py", cwd=GPTSOVITS_DIR)
        else:
            logger.warning("未找到下载脚本，请手动下载模型:")
            logger.warning("  python3 scripts/download_gptsovits_models.py")
            return True


def create_start_script():
    """创建启动脚本"""
    # Windows
    # 注意: api_v2.py 只接受 -a 和 -p 参数
    # 参考音频等参数通过 API 请求体传递（已由 tts.py 模块自动处理）
    bat_content = """@echo off
echo 启动 GPT-SoVITS API Server v2...
echo 参考音频等参数通过 API 请求传递，无需命令行指定

conda --version >nul 2>&1
if not errorlevel 1 (
    call conda activate gptsovits
)

cd /d "%~dp0GPT-SoVITS"
python api_v2.py -a 127.0.0.1 -p 9880
pause
"""
    with open("start_gptsovits.bat", "w", encoding="utf-8") as f:
        f.write(bat_content)

    # Linux/macOS - 转发到维护中的脚本，避免两个入口漂移
    sh_content = """#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/start_gptsovits.sh"
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
        return 1

    # Step 2: 安装依赖
    logger.info("[2/5] 安装依赖...")
    if not install_dependencies():
        logger.error("GPT-SoVITS 依赖安装失败")
        return 1

    # Step 3: 下载预训练模型
    logger.info("[3/5] 下载预训练模型...")
    if not download_pretrained_models():
        logger.error("GPT-SoVITS 预训练模型下载失败")
        return 1

    # Step 4: 下载 NLTK 数据（处理中英混合文本必需）
    logger.info("[4/5] 下载 NLTK 数据...")
    if not download_nltk_data():
        logger.error("NLTK 数据下载失败")
        return 1

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
