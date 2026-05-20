#!/usr/bin/env python3
"""
中文→日语实时语音转换系统 - 主入口

功能：说中文，实时输出日语语音（保留本人音色）
流程：麦克风 → ASR(语音识别) → 翻译(中→日) → TTS(音色克隆) → 虚拟音频输出

使用方式:
    # 默认配置启动
    python main.py

    # 指定环境配置
    python main.py --env m4max
    python main.py --env rtx4050

    # 列出音频设备
    python main.py --list-devices

    # 测试 ASR
    python main.py --test-asr test_audio.wav

    # 测试翻译
    python main.py --test-translate "你好，世界"

    # 测试 TTS（需要 GPT-SoVITS 服务）
    python main.py --test-tts "こんにちは世界"
"""
import argparse
import sys
import signal
import time
import numpy as np
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_config, setup_logging, print_system_info, get_config_path


def main():
    parser = argparse.ArgumentParser(
        description="实时语音转换系统（翻译/音色克隆）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译模式：中文→日语
  python3 main.py --env m4max
  python3 main.py --env rtx4050

  # 纯克隆模式：中文→克隆音色中文
  python3 main.py --env m4max --mode clone
  python3 main.py --env rtx4050_clone

  # 测试
  python3 main.py --list-devices
  python3 main.py --test-asr audio.wav
  python3 main.py --test-translate "你好"
  python3 main.py --test-tts "你好"          # 翻译模式测试
  python3 main.py --test-tts "你好" --mode clone  # 纯克隆测试
  python3 main.py --fix-nltk             # 下载 GPT-SoVITS 所需 NLTK 数据
  python3 main.py --download-models      # 预下载所有模型（之后可离线使用）
        """,
    )
    parser.add_argument(
        "--env",
        choices=[
            "m4max", "rtx4050", "rtx4070",
            "m4max_clone", "rtx4050_clone", "rtx4070_clone",
            "m4max_yue2zh", "rtx4050_yue2zh", "rtx4070_yue2zh",
        ],
        help="运行环境",
    )
    parser.add_argument(
        "--mode",
        choices=["translate", "clone", "yue2zh", "yue2zh_text"],
        help="运行模式: translate=翻译+克隆, clone=纯音色克隆, yue2zh=粤语转普通话+TTS, yue2zh_text=粤语转普通话纯文本",
    )
    parser.add_argument("--config", type=str, help="自定义配置文件路径")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备")
    parser.add_argument("--test-asr", type=str, metavar="AUDIO_FILE", help="测试 ASR")
    parser.add_argument("--test-translate", type=str, metavar="TEXT", help="测试翻译")
    parser.add_argument("--test-tts", type=str, metavar="TEXT", help="测试 TTS")
    parser.add_argument("--fix-nltk", action="store_true", help="下载 GPT-SoVITS 所需的 NLTK 数据")
    parser.add_argument("--download-models", action="store_true", help="预下载所有模型到本地缓存（之后可离线使用）")
    parser.add_argument("--log-level", default="INFO", help="日志级别")

    args = parser.parse_args()

    # 设置日志
    setup_logging(level=args.log_level)
    import logging
    logger = logging.getLogger("main")

    # 打印系统信息
    print_system_info()

    # 列出音频设备
    if args.list_devices:
        logger.info("音频输入设备:")
        from audio.capture import AudioCapture
        AudioCapture.list_devices()
        logger.info("\n音频输出设备:")
        from audio.player import AudioPlayer
        AudioPlayer.list_devices()
        return

    # 加载配置
    if args.config:
        config = load_config(args.config)
    elif args.env:
        config_path = get_config_path(args.env)
        config = load_config(str(config_path))
    else:
        config_path = get_config_path()
        config = load_config(str(config_path))

    # --mode 参数覆盖配置文件中的 mode
    if args.mode:
        config["mode"] = args.mode
        # clone 模式自动修改 TTS 输出语言为中文
        if args.mode == "clone" and "tts" in config:
            config["tts"]["output_language"] = "zh"

    # 处理测试命令
    if args.fix_nltk:
        _fix_nltk()
        return

    if args.download_models:
        _download_models(config)
        return

    if args.test_asr:
        _test_asr(config, args.test_asr)
        return

    if args.test_translate:
        _test_translate(config, args.test_translate)
        return

    if args.test_tts:
        _test_tts(config, args.test_tts)
        return

    # 启动完整管道
    _run_pipeline(config, logger)


def _fix_nltk():
    """下载 GPT-SoVITS 所需的 NLTK 数据"""
    import logging
    logger = logging.getLogger("fix_nltk")

    logger.info("下载 GPT-SoVITS 所需的 NLTK 数据...")
    logger.info("（处理含英文/字母的文本时需要，如 '大家好我是saki'）")

    try:
        import nltk

        # GPT-SoVITS 需要的 NLTK 资源
        resources = [
            ("averaged_perceptron_tagger_eng", "taggers"),
            ("averaged_perceptron_tagger", "taggers"),
            ("punkt_tab", "tokenizers"),
            ("cmudict", "corpora"),
        ]

        downloaded = 0
        for res_name, res_type in resources:
            try:
                nltk.data.find(f"{res_type}/{res_name}")
                logger.info(f"  ✓ {res_name} - 已存在")
            except LookupError:
                logger.info(f"  ⬇ {res_name} - 下载中...")
                nltk.download(res_name, quiet=True)
                downloaded += 1

        if downloaded > 0:
            logger.info(f"下载完成！共下载 {downloaded} 个资源")
            logger.info("⚠️  请重启 GPT-SoVITS API Server 后再试")
        else:
            logger.info("所有 NLTK 数据已就绪，无需下载")

    except ImportError:
        logger.error("NLTK 未安装！请先安装:")
        logger.error("  pip install nltk")
        logger.error("然后重新运行: python3 main.py --fix-nltk")
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.error("请手动运行:")
        logger.error("  python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")


def _download_models(config: dict):
    """预下载所有模型到本地缓存，之后可离线使用"""
    import os
    import logging
    logger = logging.getLogger("download_models")

    logger.info("=" * 60)
    logger.info("预下载模型到本地缓存（需要联网）")
    logger.info("下载后可永久离线使用，无需再连 HuggingFace")
    logger.info("=" * 60)

    # 🌐 预先设置镜像（国内必备）
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if not hf_endpoint:
        hf_endpoint = config.get("translator", {}).get("hf_endpoint", "")
    if not hf_endpoint:
        try:
            import urllib.request
            urllib.request.urlopen("https://huggingface.co", timeout=5)
        except Exception:
            hf_endpoint = "https://hf-mirror.com"
            logger.info("🌐 检测到无法直连 HuggingFace，自动使用镜像: hf-mirror.com")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        logger.info(f"HF_ENDPOINT: {hf_endpoint}")

    # 1. 下载 ASR 模型 (faster-whisper)
    asr_config = config.get("asr", {})
    model_size = asr_config.get("model_size", "base")
    logger.info(f"\n[1/2] 下载 ASR 模型 (faster-whisper/{model_size})...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, local_files_only=False)
        del model
        logger.info(f"  ✅ ASR 模型 {model_size} 已缓存")
    except Exception as e:
        logger.error(f"  ❌ ASR 模型下载失败: {e}")

    # 2. 下载翻译模型（使用 huggingface-cli 断点续传）
    translator_config = config.get("translator", {})
    model_name = translator_config.get("model_name", "tencent/HY-MT1.5-1.8B")
    logger.info(f"\n[2/2] 下载翻译模型 ({model_name})...")
    logger.info(f"  模型大小约 3.6GB，支持断点续传")

    download_ok = False

    # 方法1: huggingface-cli（支持断点续传，推荐）
    try:
        from huggingface_hub import snapshot_download
        logger.info("  使用 huggingface_hub 下载（支持断点续传）...")
        snapshot_download(
            model_name,
            resume_download=True,
            max_workers=4,
        )
        download_ok = True
        logger.info(f"  ✅ 翻译模型已缓存")
    except ImportError:
        logger.warning("  huggingface_hub 未安装，尝试备选方式...")
    except Exception as e:
        logger.warning(f"  huggingface_hub 下载失败: {e}")

    # 方法2: transformers from_pretrained（无断点续传，但不需要额外依赖）
    if not download_ok:
        try:
            logger.info("  使用 transformers 加载下载（无断点续传）...")
            translator_config_copy = dict(translator_config)
            translator_config_copy["local_files_only"] = False
            from modules.translator import TranslatorModule
            translator = TranslatorModule(translator_config_copy)
            translator.load_model()
            download_ok = True
            logger.info(f"  ✅ 翻译模型已缓存")
        except Exception as e:
            logger.error(f"  ❌ 翻译模型下载失败: {e}")

    if not download_ok:
        logger.error("")
        logger.error("=" * 60)
        logger.error("⚠️ 翻译模型下载失败！请手动下载：")
        logger.error("")
        logger.error("方法1: 使用 huggingface-cli（推荐，支持断点续传）")
        logger.error(f"  pip install -U huggingface_hub")
        logger.error(f"  export HF_ENDPOINT=https://hf-mirror.com  # 国内用镜像")
        logger.error(f"  huggingface-cli download {model_name}")
        logger.error("")
        logger.error("方法2: 使用镜像网站手动下载")
        logger.error(f"  打开 https://hf-mirror.com/{model_name}")
        logger.error("  下载所有文件到 ~/.cache/huggingface/hub/models--tencent--HY-MT1.5-1.8B/")
        logger.error("=" * 60)

    logger.info("\n" + "=" * 60)
    logger.info("模型下载流程结束！之后启动将优先使用本地缓存")
    logger.info("缓存位置: ~/.cache/huggingface/hub/")
    logger.info("=" * 60)


def _test_asr(config: dict, audio_file: str):
    """测试 ASR 模块"""
    import logging
    logger = logging.getLogger("test_asr")

    from modules.asr import ASRModule
    asr = ASRModule(config.get("asr", {}))
    logger.info(f"识别音频文件: {audio_file}")

    results = asr.transcribe_file(audio_file)
    logger.info(f"识别结果 ({len(results)} 段):")
    for r in results:
        print(f"  [{r['start']:.1f}s - {r['end']:.1f}s] (置信度: {r['confidence']:.2f}) {r['text']}")


def _test_translate(config: dict, text: str):
    """测试翻译模块"""
    import logging
    logger = logging.getLogger("test_translate")

    from modules.translator import TranslatorModule
    translator = TranslatorModule(config.get("translator", {}))
    translator.load_model()

    logger.info(f"原文: {text}")
    result = translator.translate(text)
    logger.info(f"译文: {result}")


def _test_tts(config: dict, text: str):
    """测试翻译+TTS 或 纯克隆+TTS"""
    import logging
    import wave
    import subprocess
    import sys
    logger = logging.getLogger("test_tts")

    is_clone = config.get("mode", "translate") == "clone"

    if is_clone:
        # 纯克隆模式：中文直接合成
        logger.info(f"📝 [克隆模式] 合成文本: {text}")
        tts_text = text
    else:
        # 翻译模式：先翻译再合成
        from modules.translator import TranslatorModule
        translator = TranslatorModule(config.get("translator", {}))
        translator.load_model()

        logger.info(f"📝 原文(中文): {text}")
        translated = translator.translate(text)
        logger.info(f"🌐 译文(日文): {translated}")

        # ⚠️ 翻译失败时不应该把中文原文发给 ja 模式 TTS（会读不出声）
        if translated:
            tts_text = translated
        else:
            logger.error("❌ 翻译失败，跳过 TTS 合成")
            return

    # 🛡️ 安全阀：TTS 文本过长时截断
    max_tts_len = 200  # 日语200字符大约40-60秒语音
    if len(tts_text) > max_tts_len:
        logger.warning(f"⚠️ TTS 文本过长 ({len(tts_text)} chars)，截断到 {max_tts_len} chars")
        logger.warning(f"  原始: {tts_text[:80]}...")
        tts_text = tts_text[:max_tts_len]
        # 截断到最近一个完整句子
        import re
        last_sentence = max(tts_text.rfind('。'), tts_text.rfind('！'), tts_text.rfind('？'),
                          tts_text.rfind('、'), tts_text.rfind(' '))
        if last_sentence > max_tts_len // 2:  # 至少保留一半内容
            tts_text = tts_text[:last_sentence + 1]
        logger.warning(f"  截断后: {tts_text}")

    # TTS 合成
    from modules.tts import TTSModule
    tts = TTSModule(config.get("tts", {}))

    # 检查服务
    if not tts.check_server():
        logger.error("GPT-SoVITS API Server 未启动！")
        logger.error("请先启动: cd GPT-SoVITS && python3 api_v2.py")
        return

    logger.info(f"🔊 合成文本: {tts_text}")
    result = tts.synthesize(tts_text)

    if result is not None:
        audio, sample_rate = result
        # 保存为 WAV 文件
        output_path = "test_tts_output.wav"
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        logger.info(f"已保存到: {output_path} ({len(audio)} samples, {sample_rate}Hz, {len(audio)/sample_rate:.1f}s)")

        # macOS 自动播放
        if sys.platform == "darwin":
            logger.info("正在播放...")
            subprocess.run(["afplay", output_path])
        elif sys.platform == "linux":
            subprocess.run(["aplay", output_path])
        else:
            logger.info(f"请手动播放: {output_path}")
    else:
        logger.error("TTS 合成失败")


def _run_pipeline(config: dict, logger):
    """运行完整管道"""
    from pipeline.orchestrator import StreamPipeline

    pipeline = StreamPipeline(config)
    pipeline.start()

    # 信号处理 - 优雅退出
    def signal_handler(sig, frame):
        logger.info("\n收到退出信号，正在停止...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 主线程循环 - 定期打印状态
    mode_label = "克隆" if config.get("mode") == "clone" else "翻译"
    try:
        while True:
            time.sleep(10)
            stats = pipeline.get_stats()
            logger.info(
                f"[{mode_label}] 状态: ASR={stats['asr_count']} "
                f"翻译={stats['translate_count']} TTS={stats['tts_count']}"
            )
    except KeyboardInterrupt:
        pass

    pipeline.stop()


if __name__ == "__main__":
    main()
