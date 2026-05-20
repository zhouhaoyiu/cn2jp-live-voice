@echo off
echo 启动 GPT-SoVITS API Server v2...
echo 参考音频等参数通过 API 请求传递，无需命令行指定
cd /d "%~dp0GPT-SoVITS"
python api_v2.py -a 127.0.0.1 -p 9880
pause
