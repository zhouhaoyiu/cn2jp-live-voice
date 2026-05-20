#!/usr/bin/env python3
"""
模型下载脚本 - 自动下载所需模型到本地缓存

下载的模型:
1. faster-whisper (base/small) - 语音识别
2. NLLB-200-Distilled-600M - 中日翻译
3. GPT-SoVITS 需要单独克隆（见 setup_gptsovits.py）
"""
import logging
import sys
from pathlib import Path

logger = logging.getLogger("download_models")
logging.basicConfig(level=logging.INFO)


def download_whisper_model(model_size: str = "base"):
    """下载 faster-whisper 模型"""
    logger.info(f"下载 faster-whisper 模型: {model_size}")
    from faster_whisper import WhisperModel

    # 下载并缓存
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    logger.info(f"faster-whisper {model_size} 下载完成")
    del model


def download_nllb_model(model_name: str = "facebook/nllb-200-distilled-600M"):
    """下载 NLLB 翻译模型"""
    logger.info(f"下载 NLLB 翻译模型: {model_name}")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    logger.info("下载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="zho_Hans")

    logger.info("下载模型（约 1.2GB）...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    logger.info("NLLB 模型下载完成")
    del model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载所需模型")
    parser.add_argument("--whisper-size", default="base", help="Whisper 模型大小")
    parser.add_argument("--skip-whisper", action="store_true", help="跳过 Whisper 下载")
    parser.add_argument("--skip-nllb", action="store_true", help="跳过 NLLB 下载")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("开始下载模型文件")
    logger.info("=" * 50)

    if not args.skip_whisper:
        try:
            download_whisper_model(args.whisper_size)
        except Exception as e:
            logger.error(f"Whisper 模型下载失败: {e}")

    if not args.skip_nllb:
        try:
            download_nllb_model()
        except Exception as e:
            logger.error(f"NLLB 模型下载失败: {e}")

    logger.info("=" * 50)
    logger.info("模型下载完成！")
    logger.info("注意: GPT-SoVITS 需要单独设置，请运行:")
    logger.info("  python setup_gptsovits.py")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
