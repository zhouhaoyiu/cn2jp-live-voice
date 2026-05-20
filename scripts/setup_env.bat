@echo off
REM ============================================
REM 中文→日语实时语音转换系统 - Windows 环境安装
REM 适用于 RTX 4050 笔记本
REM ============================================

echo ============================================
echo 中文-日语实时语音转换系统 - 环境安装
echo ============================================

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查 CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到 NVIDIA GPU / CUDA
    echo 如有 RTX 4050，请安装 NVIDIA 驱动
)

REM 创建虚拟环境
echo [1/5] 创建 Python 虚拟环境...
python -m venv venv
call venv\Scripts\activate.bat

REM 安装 PyTorch (CUDA 12.1)
echo [2/5] 安装 PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM 安装项目依赖
echo [3/5] 安装项目依赖...
pip install -r requirements.txt

REM 下载模型
echo [4/5] 下载模型（可能需要较长时间）...
python download_models.py --whisper-size base

REM 搭建 GPT-SoVITS
echo [5/5] 搭建 GPT-SoVITS...
python setup_gptsovits.py

echo.
echo ============================================
echo 安装完成！
echo.
echo 使用步骤:
echo   1. 录制参考音频: reference_audio\my_voice.wav
echo   2. 启动 GPT-SoVITS: start_gptsovits.bat
echo   3. 启动管道: python main.py --env rtx4050
echo ============================================
pause
