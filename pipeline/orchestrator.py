"""
流式管道编排器 - 多线程生产者-消费者模式
实现 ASR → 翻译 → TTS 的流水线并行处理
"""
import logging
import queue
import threading
import time
import numpy as np
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class StreamPipeline:
    """
    流式语音转换管道

    支持四种模式：
        translate (默认): ASR → 翻译 → TTS → 输出 (中文→日语)
        clone:           ASR → TTS → 输出 (中文→中文，纯音色克隆)
        yue2zh:          ASR → 翻译 → TTS → 输出 (粤语→普通话，克隆音色)
        yue2zh_text:     ASR → 翻译 → 终端文本输出 (粤语→普通话，纯文本流式)

    使用方式:
        pipeline = StreamPipeline(config)
        pipeline.start()
        # ... 运行中，自动处理 ...
        pipeline.stop()
    """

    def __init__(self, config: dict):
        """
        初始化流式管道

        Args:
            config: 完整配置字典，包含各模块和全局配置
        """
        self.config = config
        self.pipeline_config = config.get("pipeline", {})

        # 🔑 运行模式: translate(翻译+克隆) / clone(纯克隆) / yue2zh(粤语转普通话+TTS) / yue2zh_text(粤语转普通话纯文本)
        self.mode = config.get("mode", "translate")
        self.is_clone_mode = (self.mode == "clone")
        self.is_yue2zh_mode = (self.mode in ("yue2zh", "yue2zh_text"))
        self.is_text_only_mode = (self.mode == "yue2zh_text")

        # 队列设置
        self.max_queue_size = self.pipeline_config.get("max_queue_size", 32)
        self._asr_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self._translate_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self._tts_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)

        # 模块实例
        self._asr = None
        self._translator = None
        self._tts = None
        self._capture = None
        self._player = None

        # 状态
        self._running = False
        self._stats_lock = threading.Lock()
        self._stats = {
            "asr_count": 0,
            "translate_count": 0,
            "tts_count": 0,
            "total_latency_ms": 0,
            "start_time": 0,
        }

        # 延迟追踪
        self._sentence_timestamps = {}  # sentence_id -> timestamp
        self._sentence_counter = 0
        self._counter_lock = threading.Lock()

        # 🔇 防回环：TTS 播放时静音 ASR
        self._mute_lock = threading.Lock()
        self._is_muted = False
        self._mute_resume_delay = 0.3  # TTS 播放结束后再等 0.3s 才恢复采集

        # 📝 最近翻译结果缓存，用于终端显示
        self._recent_translations = []
        self._max_recent = 20

        # 📝 流式文本输出回调（yue2zh_text 模式用）
        self._on_stream_text: Optional[Callable[[str, str], None]] = None

    def set_on_stream_text(self, callback: Callable[[str, str], None]):
        """
        设置流式文本输出回调（yue2zh_text 模式专用）

        Args:
            callback: 回调函数，参数为 (cantonese_text, mandarin_text)
                      mandarin_text 为空表示正在翻译中
        """
        self._on_stream_text = callback

    def start(self):
        """启动完整管道"""
        mode_labels = {
            "clone": "纯音色克隆",
            "yue2zh": "粤语→普通话",
            "yue2zh_text": "粤语→普通话(纯文本)",
            "translate": "中文→日语翻译+克隆",
        }
        mode_label = mode_labels.get(self.mode, self.mode)
        logger.info("=" * 60)
        logger.info(f"启动实时语音转换管道 [{mode_label}模式]")
        logger.info("=" * 60)

        self._running = True
        self._stats["start_time"] = time.time()

        # ─── yue2zh_text 纯文本模式：跳过音频输出和 TTS ───
        if self.is_text_only_mode:
            # 1. 初始化翻译
            from modules.translator import TranslatorModule
            translator_config = self.config.get("translator", {})
            self._translator = TranslatorModule(translator_config)
            self._translator.start(
                on_translated=self._on_translated,
                text_queue=self._translate_queue,
            )
            logger.info("[1/3] 翻译模块已启动 (粤语→普通话文本转换)")

            # 2. 初始化 ASR
            from modules.asr import ASRModule
            asr_config = self.config.get("asr", {})
            self._asr = ASRModule(asr_config)
            self._asr.start(on_text=self._on_asr_text)
            logger.info("[2/3] ASR 模块已启动 (粤语识别)")

            # 3. 初始化音频采集
            from audio.capture import AudioCapture
            capture_config = self.config.get("capture", {})
            self._capture = AudioCapture(capture_config)
            self._capture.start(on_audio=self._on_audio_capture)
            logger.info("[3/3] 音频采集模块已启动")

            logger.info("=" * 60)
            logger.info("🟢 管道启动完成！[粤语转普通话-纯文本] 说粤语→实时输出普通话文字")
            logger.info("=" * 60)
            return

        # ─── 完整模式（包含 TTS + 音频输出）───

        # 1. 初始化并启动音频输出
        from audio.player import AudioPlayer
        player_config = self.config.get("player", {})
        self._player = AudioPlayer(player_config)
        self._player.start()
        step = 1
        logger.info(f"[{step}/N] 音频输出模块已启动")

        # 2. 初始化并启动 TTS
        from modules.tts import TTSModule
        tts_config = self.config.get("tts", {})
        self._tts = TTSModule(tts_config)

        # 检查 GPT-SoVITS 服务
        if not self._tts.check_server():
            logger.warning("⚠ GPT-SoVITS API Server 未响应！请先启动 GPT-SoVITS")
            logger.warning("  启动方式: cd GPT-SoVITS && python3 api_v2.py")
            logger.warning("  管道将继续启动，但 TTS 会失败")

        self._tts.start(
            on_audio=self._on_tts_audio,
            text_queue=self._tts_queue,
        )
        step += 1
        logger.info(f"[{step}/N] TTS 模块已启动 (输出语言: {tts_config.get('output_language', 'ja')})")

        # 3. 初始化翻译（translate 和 yue2zh 模式）
        if not self.is_clone_mode:
            from modules.translator import TranslatorModule
            translator_config = self.config.get("translator", {})
            self._translator = TranslatorModule(translator_config)
            self._translator.start(
                on_translated=self._on_translated,
                text_queue=self._translate_queue,
            )
            step += 1
            logger.info(f"[{step}/N] 翻译模块已启动")
        else:
            logger.info("[--] 翻译模块已跳过 (clone 模式)")

        # 粤语模式提示翻译模块用途
        if self.is_yue2zh_mode and self._translator:
            src_lang = self.config.get("translator", {}).get("src_lang", "yue")
            tgt_lang = self.config.get("translator", {}).get("tgt_lang", "zh")
            logger.info(f"  粤语模式: 翻译模块用于 {src_lang}→{tgt_lang} 口语文本转换")

        # 4. 初始化并启动 ASR
        from modules.asr import ASRModule
        asr_config = self.config.get("asr", {})
        self._asr = ASRModule(asr_config)
        self._asr.start(on_text=self._on_asr_text)
        step += 1
        logger.info(f"[{step}/N] ASR 模块已启动")

        # 5. 初始化并启动音频采集
        from audio.capture import AudioCapture
        capture_config = self.config.get("capture", {})
        self._capture = AudioCapture(capture_config)
        self._capture.start(on_audio=self._on_audio_capture)
        step += 1
        logger.info(f"[{step}/N] 音频采集模块已启动")

        logger.info("=" * 60)
        if self.is_clone_mode:
            logger.info("🟢 管道启动完成！[纯克隆模式] 说中文→克隆音色输出中文")
        elif self.is_yue2zh_mode:
            logger.info("🟢 管道启动完成！[粤语转普通话] 说粤语→转普通话→克隆音色输出普通话")
        else:
            logger.info("🟢 管道启动完成！[翻译模式] 说中文→翻译→克隆音色输出日语")
        logger.info("=" * 60)

    def stop(self):
        """停止完整管道"""
        logger.info("正在停止管道...")
        self._running = False

        # 按逆序停止
        if self._capture:
            self._capture.stop()
        if self._asr:
            self._asr.stop()
        if self._translator:
            self._translator.stop()
        if self._tts:
            self._tts.stop()
        if self._player:
            self._player.stop()

        # 打印统计
        self._print_stats()
        logger.info("管道已停止")

    def _on_audio_capture(self, audio_data: np.ndarray):
        """
        音频采集回调 → 送入 ASR

        防回环：当 TTS 正在播放音频时，丢弃麦克风输入，
        避免把日语音频再识别成中文形成反馈循环。

        Args:
            audio_data: float32 numpy 数组, 16kHz 单声道
        """
        with self._mute_lock:
            muted = self._is_muted

        if muted:
            return  # 🔇 TTS 正在播放，丢弃麦克风数据防回环

        if self._asr:
            self._asr.feed_audio(audio_data)

    def _on_asr_text(self, text: str, confidence: float):
        """
        ASR 识别回调 → 送入翻译队列

        Args:
            text: 识别的中文/粤语文本
            confidence: 识别置信度
        """
        # 过滤短文本和低置信度
        if len(text) < 2:
            logger.debug(f"跳过短文本: '{text}'")
            return

        if confidence < -1.0:
            logger.debug(f"跳过低置信度文本: [{confidence:.2f}] '{text}'")
            return

        # 过滤明显的非语音识别结果（纯标点、单个重复字等）
        stripped = text.replace(' ', '').replace('。', '').replace('，', '').replace('！', '').replace('？', '')
        if len(stripped) < 2:
            logger.debug(f"跳过无意义文本: '{text}'")
            return

        # 记录时间戳
        with self._counter_lock:
            self._sentence_counter += 1
            sid = self._sentence_counter
        self._sentence_timestamps[sid] = time.time()

        logger.info(f"🎤 ASR [{confidence:.2f}]: {text}")

        # 🔑 yue2zh_text 模式：立即输出粤语原文，再异步翻译
        if self.is_text_only_mode:
            # 立即显示粤语原文
            if self._on_stream_text:
                self._on_stream_text(text, "")  # 粤语原文，普通话待翻译
            else:
                print(f"\n🗣️ 粤语: {text}")

        # 🔑 clone 模式：直接把中文送入 TTS 队列，跳过翻译
        if self.is_clone_mode:
            try:
                self._tts_queue.put_nowait(text)
                with self._stats_lock:
                    self._stats["asr_count"] += 1
            except queue.Full:
                logger.warning("TTS 队列已满，丢弃文本")
            return

        # translate / yue2zh / yue2zh_text 模式：送入翻译队列
        try:
            self._translate_queue.put_nowait(text)
            with self._stats_lock:
                self._stats["asr_count"] += 1
        except queue.Full:
            logger.warning("翻译队列已满，丢弃文本")

    def _on_translated(self, original: str, translated: str):
        """
        翻译完成回调 → 送入 TTS 队列或文本输出

        Args:
            original: 原始文本（粤语/中文）
            translated: 翻译后的文本（普通话/日语）
        """
        if not translated or not translated.strip():
            logger.warning("翻译结果为空，跳过")
            return

        # 🛡️ 安全阀：翻译结果过长时截断（防止 NLLB 重复 bug）
        max_tts_len = 200
        if len(translated) > max_tts_len:
            logger.warning(f"⚠️ 翻译结果过长 ({len(translated)} chars)，截断到 {max_tts_len}")
            translated = translated[:max_tts_len]
            # 截断到最近句号
            for sep in ['。', '！', '？', '、', ' ']:
                idx = translated.rfind(sep)
                if idx > max_tts_len // 2:
                    translated = translated[:idx + 1]
                    break

        # 根据模式显示不同的日志标签
        if self.is_yue2zh_mode:
            logger.info(f"🗣️ 粤语转普通话: {original} → {translated}")
        else:
            logger.info(f"🌐 翻译: {original} → {translated}")

        # 🔑 yue2zh_text 模式：直接输出文本，不入 TTS 队列
        if self.is_text_only_mode:
            if self._on_stream_text:
                self._on_stream_text(original, translated)
            else:
                print(f"   → 普通话: {translated}")
            with self._stats_lock:
                self._stats["translate_count"] += 1
            return

        try:
            self._tts_queue.put_nowait(translated)
            with self._stats_lock:
                self._stats["translate_count"] += 1
        except queue.Full:
            logger.warning("TTS 队列已满，丢弃文本")

    def _on_tts_audio(self, audio_data: np.ndarray, sample_rate: int = 32000):
        """
        TTS 合成完成回调 → 送入音频输出

        防回环：播放前静音 ASR，播放结束后延迟恢复。

        Args:
            audio_data: float32 numpy 数组
            sample_rate: 音频采样率（GPT-SoVITS v2 默认 32000Hz）
        """
        # 如果播放器采样率与音频不同，需要重采样
        if self._player and self._player.sample_rate != sample_rate:
            audio_data = self._resample(audio_data, sample_rate, self._player.sample_rate)

        # 计算延迟（取最早未完成的时间戳）
        now = time.time()
        with self._counter_lock:
            if self._sentence_timestamps:
                oldest_sid = min(self._sentence_timestamps.keys())
                start_time = self._sentence_timestamps.pop(oldest_sid)
                latency_ms = (now - start_time) * 1000
            else:
                latency_ms = 0

        # 🔇 静音 ASR：防止麦克风录到 TTS 输出形成回环
        with self._mute_lock:
            self._is_muted = True

        logger.info(f"🔊 TTS 输出: {len(audio_data)} samples @ {sample_rate}Hz, 延迟: {latency_ms:.0f}ms")

        if self._player:
            self._player.play(audio_data)

        # 计算音频播放时长，播放结束后延迟恢复 ASR
        audio_duration = len(audio_data) / (self._player.sample_rate if self._player else sample_rate)
        resume_delay = audio_duration + self._mute_resume_delay
        threading.Timer(resume_delay, self._unmute_asr).start()

        with self._stats_lock:
            self._stats["tts_count"] += 1
            self._stats["total_latency_ms"] += latency_ms

    def _unmute_asr(self):
        """TTS 播放结束后恢复 ASR 采集"""
        # 确保播放器缓冲区也播放完毕
        if self._player and self._player.is_playing():
            # 还在播放，再等 0.5 秒
            threading.Timer(0.5, self._unmute_asr).start()
            return

        # 恢复前清空 ASR 缓冲区，丢弃回环期间积压的音频
        if self._asr:
            self._asr.clear_buffer()

        with self._mute_lock:
            was_muted = self._is_muted
            self._is_muted = False
        if was_muted:
            logger.debug("🔇→🎙️ ASR 恢复采集")

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样"""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio)

    def _print_stats(self):
        """打印运行统计"""
        with self._stats_lock:
            uptime = time.time() - self._stats["start_time"] if self._stats["start_time"] > 0 else 0
            asr_count = self._stats["asr_count"]
            translate_count = self._stats["translate_count"]
            tts_count = self._stats["tts_count"]
            avg_latency = (
                self._stats["total_latency_ms"] / tts_count
                if tts_count > 0 else 0
            )

        logger.info("=" * 60)
        logger.info("运行统计:")
        logger.info(f"  运行时间: {uptime:.1f}s")
        logger.info(f"  ASR 句数: {asr_count}")
        logger.info(f"  翻译句数: {translate_count}")
        logger.info(f"  TTS 句数: {tts_count}")
        logger.info(f"  平均延迟: {avg_latency:.0f}ms")
        logger.info("=" * 60)

    def get_stats(self) -> dict:
        """获取当前统计信息"""
        with self._stats_lock:
            return dict(self._stats)
