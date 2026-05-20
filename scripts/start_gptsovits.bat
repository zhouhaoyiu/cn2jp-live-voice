@echo off
echo 启动 GPT-SoVITS API Server v2...
cd /d "%~dp0GPT-SoVITS"
python api_v2.py -a 127.0.0.1 -p 9880 -dr reference_audio/my_voice.wav -dt "参考音频文本" -dl zh
pause
