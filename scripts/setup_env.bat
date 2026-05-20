@echo off
REM ============================================
REM 中文→日语实时语音转换系统 - Windows 环境安装
REM 适用于 RTX 4050 笔记本
REM ============================================

echo ============================================
echo 中文-日语实时语音转换系统 - 环境安装
echo ============================================

set ENV_NAME=cn2jp

REM 检查 conda
conda --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 conda，请先安装 Miniforge/Anaconda
    pause
    exit /b 1
)

REM 检查 CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到 NVIDIA GPU / CUDA
    echo 如有 RTX 4050，请安装 NVIDIA 驱动
)

REM 创建/激活 conda 环境
echo [1/5] 创建/激活 conda 环境: %ENV_NAME% ...
set ENV_EXISTS=
for /f "tokens=1" %%i in ('conda info --envs ^| findstr /R /C:"^%ENV_NAME% "') do set ENV_EXISTS=1
if not defined ENV_EXISTS (
    conda create -n %ENV_NAME% python=3.11 -y
)
call conda activate %ENV_NAME%
python --version

REM 安装 PyTorch (CUDA 12.1)
echo [2/5] 安装 PyTorch (CUDA 12.1)...
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu121

REM 安装项目依赖
echo [3/5] 安装项目依赖...
pip install -r requirements.txt
pip install faster-whisper==1.2.1 ctranslate2==4.7.1 transformers==5.8.0 tokenizers==0.22.2 peft==0.19.1 accelerate==1.13.0 sentencepiece==0.2.1 huggingface-hub==1.14.0 numpy==2.4.4 pyaudio==0.2.14 requests==2.34.0 pykakasi==2.3.0 pypinyin==0.55.0 pyyaml==6.0.3

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
