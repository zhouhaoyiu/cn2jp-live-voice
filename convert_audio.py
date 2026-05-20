 #!/usr/bin/env python3
"""
音频格式转换脚本
将任意格式的音频文件（MP3, M4A, FLAC, OGG 等）转换为标准 WAV 格式
用于准备 GPT-SoVITS 所需的参考音频

用法:
    python3 convert_audio.py input.mp3 reference_audio/my_voice.wav
    python3 convert_audio.py input.m4a reference_audio/my_voice.wav --sample-rate 32000
"""
import argparse
import subprocess
import sys
from pathlib import Path


def convert_with_ffmpeg(input_path: str, output_path: str, sample_rate: int = 32000):
    """使用 ffmpeg 转换音频（最可靠的方式）"""
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sample_rate),   # 采样率
        "-ac", "1",                # 单声道
        "-sample_fmt", "s16",      # 16bit
        output_path
    ]

    print(f"执行: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"转换成功！输出: {output_path} ({sample_rate}Hz, 单声道, 16bit)")
            return True
        else:
            print(f"ffmpeg 错误: {result.stderr[:500]}")
            return False
    except FileNotFoundError:
        print("ffmpeg 未安装！")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: 从 https://ffmpeg.org 下载")
        return False
    except subprocess.TimeoutExpired:
        print("ffmpeg 超时")
        return False


def convert_with_pydub(input_path: str, output_path: str, sample_rate: int = 32000):
    """使用 pydub 转换音频（需要 pip install pydub）"""
    try:
        from pydub import AudioSegment
    except ImportError:
        print("pydub 未安装，请运行: pip install pydub")
        return False

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"读取音频: {input_path}")
    audio = AudioSegment.from_file(input_path)

    # 转单声道
    if audio.channels > 1:
        audio = audio.set_channels(1)
        print("已转为单声道")

    # 重采样
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
        print(f"已重采样到 {sample_rate}Hz")

    # 16bit
    audio = audio.set_sample_width(2)

    audio.export(output_path, format="wav")
    print(f"转换成功！输出: {output_path} ({sample_rate}Hz, 单声道, 16bit)")
    return True


def convert_with_python(input_path: str, output_path: str, sample_rate: int = 32000):
    """使用纯 Python 读取音频（最后的备选方案）"""
    import wave
    import numpy as np

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 尝试用 soundfile 读取
    try:
        import soundfile as sf
        audio, sr = sf.read(input_path)
        print(f"soundfile 读取成功: {sr}Hz, {len(audio)} samples")
    except Exception as e:
        print(f"soundfile 读取失败: {e}")
        # 尝试用 scipy
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(input_path)
            print(f"scipy 读取成功: {sr}Hz")
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
        except Exception as e2:
            print(f"scipy 也读取失败: {e2}")
            print("无法读取此文件，请安装 ffmpeg: brew install ffmpeg")
            return False

    audio = np.asarray(audio, dtype=np.float32)

    # 转单声道
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        print("已转为单声道")

    # 重采样
    if sr != sample_rate:
        duration = len(audio) / sr
        target_len = int(duration * sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)
        print(f"已重采样到 {sample_rate}Hz")

    # 保存为标准 WAV
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"转换成功！输出: {output_path} ({sample_rate}Hz, 单声道, 16bit, {len(audio)/sample_rate:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="音频格式转换为标准 WAV")
    parser.add_argument("input", help="输入音频文件路径（支持 MP3/M4A/FLAC/OGG/WAV 等）")
    parser.add_argument("output", nargs="?", default="reference_audio/my_voice.wav",
                        help="输出 WAV 文件路径 (默认: reference_audio/my_voice.wav)")
    parser.add_argument("--sample-rate", type=int, default=32000,
                        help="输出采样率 (默认: 32000，GPT-SoVITS v2 推荐)")
    parser.add_argument("--method", choices=["ffmpeg", "pydub", "python", "auto"],
                        default="auto", help="转换方法 (默认: auto=依次尝试)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    sample_rate = args.sample_rate

    if not Path(input_path).exists():
        print(f"错误：输入文件不存在: {input_path}")
        sys.exit(1)

    # 打印输入文件信息
    print(f"输入文件: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"目标采样率: {sample_rate}Hz")
    print()

    success = False

    if args.method == "auto":
        # 依次尝试 ffmpeg -> pydub -> python
        print("[1/3] 尝试 ffmpeg...")
        success = convert_with_ffmpeg(input_path, output_path, sample_rate)

        if not success:
            print("\n[2/3] 尝试 pydub...")
            success = convert_with_pydub(input_path, output_path, sample_rate)

        if not success:
            print("\n[3/3] 尝试纯 Python...")
            success = convert_with_python(input_path, output_path, sample_rate)

    elif args.method == "ffmpeg":
        success = convert_with_ffmpeg(input_path, output_path, sample_rate)
    elif args.method == "pydub":
        success = convert_with_pydub(input_path, output_path, sample_rate)
    elif args.method == "python":
        success = convert_with_python(input_path, output_path, sample_rate)

    if success:
        print(f"\n下一步:")
        print(f"  1. 将录音对应的文本填入 configs/m4max.yaml 的 prompt_text 字段")
        print(f"  2. 启动 GPT-SoVITS: cd GPT-SoVITS && python3 api_v2.py -a 127.0.0.1 -p 9880")
        print(f"  3. 测试 TTS: python3 main.py --test-tts 'こんにちは'")
    else:
        print(f"\n所有转换方法都失败了！请安装 ffmpeg:")
        print(f"  macOS: brew install ffmpeg")
        sys.exit(1)


if __name__ == "__main__":
    main()
