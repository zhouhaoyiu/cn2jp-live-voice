@echo off
echo 启动 GPT-SoVITS API Server v2...
echo 参考音频等参数通过 API 请求传递，无需命令行指定

REM 尝试激活 conda 环境（如果已安装）
conda --version >nul 2>&1
if not errorlevel 1 (
	call conda activate gptsovits
)

cd /d "%~dp0GPT-SoVITS"
python api_v2.py -a 127.0.0.1 -p 9880
pause
