#!/usr/bin/env python3
"""
参考音频录制脚本
录制 5-15 秒自己的语音，用于 GPT-SoVITS 音色克隆

⚠️ 录制时请说一段完整的中文句子，并在配置文件 prompt_text 中填入相同的文本！
"""
import wave
import numpy as np
import sys
from pathlib import Path

FORMAT_PA16 = 8


def record_reference(output_path: str = "reference_audio/my_voice.wav",
                     duration: int = 7,
                     sample_rate: int = 32000):
    """
    录制参考音频

    Args:
        output_path: 输出路径
        duration: 录制时长(秒)，默认7秒，建议3~10秒
        sample_rate: 采样率 (GPT-SoVITS v2 推荐 32000Hz)
    """
    import pyaudio

    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    chunk_size = 1024
    channels = 1

    print("=" * 60)
    print(f"  参考音频录制 - {duration}秒 @ {sample_rate}Hz")
    print("=" * 60)
    print()
    print("⚠️ 重要：请用正常语速朗读一段中文！")
    print()
    print("建议朗读内容（3~10秒）：")
    print("  '大家好，欢迎来到我的直播间。'")
    print("  '今天天气真不错呀。'")
    print()
    print("⚠️ 关键：prompt_text 必须与录音内容完全一致！")
    print("  录多长就说多长，不要在 prompt_text 里写录音里没有的内容！")
    print("  否则 GPT-SoVITS 会把 prompt_text 多出的部分当作要生成的文本，")
    print("  导致输出音频开头出现 prompt_text 的尾巴！")
    print()
    print(f"录制时长: {duration} 秒（建议 3~10 秒）")
    print()

    pa = pyaudio.PyAudio()

    # 列出可用设备
    print("可用的输入设备：")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} ({info['maxInputChannels']}ch, {info['defaultSampleRate']}Hz)")
    print()

    input("按 Enter 开始录制...")
    print(f"🔴 录制中... ({duration}秒，请开始说话)")

    stream = pa.open(
        format=FORMAT_PA16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    frames = []
    num_chunks = int(sample_rate / chunk_size * duration)

    for i in range(num_chunks):
        data = stream.read(chunk_size)
        frames.append(data)
        # 进度条
        progress = (i + 1) / num_chunks
        bar = "█" * int(progress * 30) + "░" * (30 - int(progress * 30))
        print(f"\r  [{bar}] {progress * 100:.0f}%", end="", flush=True)

    print("\n")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # 保存 WAV 文件
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    # 验证文件
    with wave.open(output_path, 'rb') as wf:
        verify_sr = wf.getframerate()
        verify_frames = wf.getnframes()
        verify_duration = verify_frames / verify_sr

    print(f"✅ 录制完成！保存到: {output_path}")
    print(f"   格式: {verify_sr}Hz, 单声道, 16bit, {verify_duration:.1f}秒")
    print()
    print("下一步:")
    print("  1. 确认音频质量（听一下录制的音频，确保清晰无杂音）")
    print("  2. ⚠️ 将录音时说的文本【逐字】填入配置文件的 prompt_text 字段")
    print("     clone模式: configs/m4max_clone.yaml -> tts.prompt_text")
    print("     翻译模式: configs/m4max.yaml -> tts.prompt_text")
    print("     例如: prompt_text: \"大家好，欢迎来到我的直播间\"")
    print("  3. ⚠️ prompt_text 必须和录音内容完全一致，不超过10秒的文本量！")
    print("  4. 启动 GPT-SoVITS API Server:")
    print("     cd GPT-SoVITS && python3 api_v2.py -a 127.0.0.1 -p 9880")
    print("  5. 测试 TTS:")
    print("     python3 main.py --env m4max_clone --test-tts '你好世界'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="录制参考音频（用于音色克隆）")
    parser.add_argument("--output", default="reference_audio/my_voice.wav", help="输出路径")
    parser.add_argument("--duration", type=int, default=7, help="录制时长(秒)，建议3~10秒")
    parser.add_argument("--sample-rate", type=int, default=32000,
                        help="采样率 (默认 32000，GPT-SoVITS v2 推荐)")
    args = parser.parse_args()

    record_reference(args.output, args.duration, args.sample_rate)
