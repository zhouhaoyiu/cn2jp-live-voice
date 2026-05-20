"""
ASR 模块 - 基于 faster-whisper 的实时语音识别
支持流式识别，集成 VAD (语音活动检测)
"""
import logging
import queue
import threading
import time
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class ASRModule:
    """
    实时语音识别模块

    使用 faster-whisper 进行流式语音识别，
    内置 VAD 检测自动分句，识别完成后通过回调输出文本。

    使用方式:
        asr = ASRModule(config)
        asr.start(on_text_callback)
        # 向 asr.feed_audio() 送入 PCM 数据
        asr.stop()
    """

    def __init__(self, config: dict):
        """
        初始化 ASR 模块

        Args:
            config: 配置字典，包含以下键:
                - model_size: 模型大小 (tiny/base/small/medium/large-v2)
                - device: 推理设备 (cuda/cpu/auto)
                - compute_type: 计算精度 (int8/float16/float32)
                - language: 识别语言 (zh/en/ja/auto)
                - vad_filter: 是否启用 VAD 过滤
                - vad_parameters: VAD 参数
                - beam_size: 束搜索大小
                - chunk_duration_sec: 每次处理的音频时长(秒)
        """
        self.config = config
        self.model_size = config.get("model_size", "base")
        self.device = config.get("device", "auto")
        self.compute_type = config.get("compute_type", "int8")
        self.language = config.get("language", "zh")
        self.vad_filter = config.get("vad_filter", True)
        self.vad_parameters = config.get("vad_parameters", {
            "min_silence_duration_ms": 600,
            "speech_pad_ms": 200,
            "threshold": 0.5,
        })
        self.beam_size = config.get("beam_size", 3)
        self.chunk_duration_sec = config.get("chunk_duration_sec", 3.0)

        self.model = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sample_rate = 16000
        self._audio_buffer = np.array([], dtype=np.float32)
        self._on_text_callback: Optional[Callable] = None

    def load_model(self):
        """加载 faster-whisper 模型"""
        from faster_whisper import WhisperModel

        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = self.compute_type
        if device == "cpu":
            compute_type = "int8"

        # 🔑 优先使用本地缓存，避免每次启动都联网
        local_only = self.config.get("local_files_only", True)

        logger.info(f"加载 ASR 模型: {self.model_size}, 设备: {device}, 精度: {compute_type}, 本地优先: {local_only}")
        try:
            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                local_files_only=local_only,
            )
        except Exception as e:
            if local_only:
                logger.warning(f"本地缓存加载失败: {e}")
                logger.info("尝试联网下载 ASR 模型...")
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=False,
                )
                logger.info("✅ ASR 模型已下载到本地缓存，下次启动可离线使用")
            else:
                raise
        logger.info("ASR 模型加载完成")

    def start(self, on_text: Callable[[str, float], None]):
        """
        启动 ASR 处理线程

        Args:
            on_text: 文本回调函数，参数为 (text, confidence)
        """
        if self.model is None:
            self.load_model()

        self._on_text_callback = on_text
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("ASR 处理线程已启动")

    def stop(self):
        """停止 ASR 处理"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("ASR 处理线程已停止")

    def feed_audio(self, audio_data: np.ndarray):
        """
        送入音频数据

        Args:
            audio_data: float32 numpy 数组, 16kHz 单声道
        """
        if self._running:
            self._audio_queue.put(audio_data)

    def clear_buffer(self):
        """清空音频缓冲区和队列，用于防回环时丢弃积压数据"""
        # 清空队列
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Exception:
                break
        # 清空内部缓冲区
        self._audio_buffer = np.array([], dtype=np.float32)
        logger.debug("ASR 缓冲区已清空")

    def _process_loop(self):
        """ASR 处理主循环"""
        chunk_samples = int(self._sample_rate * self.chunk_duration_sec)

        while self._running:
            try:
                # 从队列获取音频块，超时 100ms
                audio_chunk = self._audio_queue.get(timeout=0.1)
                self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])
            except queue.Empty:
                # 没有新数据，检查缓冲区是否有残余需要处理
                if len(self._audio_buffer) > self._sample_rate * 0.5:  # >0.5秒
                    self._process_audio(force=True)
                continue

            # 缓冲区达到指定时长，进行识别
            if len(self._audio_buffer) >= chunk_samples:
                self._process_audio(force=False)

    def _process_audio(self, force: bool = False):
        """
        处理音频缓冲区中的数据

        Args:
            force: 是否强制处理（即使缓冲区不够长）
        """
        if len(self._audio_buffer) < self._sample_rate * 0.3:  # 少于0.3秒不处理
            return

        audio_data = self._audio_buffer.copy()
        self._audio_buffer = np.array([], dtype=np.float32)

        # 确保数据类型是 float32（重采样可能产生 float64）
        audio_data = audio_data.astype(np.float32)

        try:
            segments, info = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters,
            )

            for segment in segments:
                text = segment.text.strip()
                if text and self._on_text_callback:
                    # faster-whisper 用 avg_logprob 不是 avg_log_probability
                    confidence = getattr(segment, 'avg_logprob',
                               getattr(segment, 'avg_log_probability', -1.0))
                    logger.debug(f"ASR 识别: [{confidence:.2f}] {text}")
                    self._on_text_callback(text, confidence)

        except Exception as e:
            logger.error(f"ASR 处理出错: {e}")

    def transcribe_file(self, audio_path: str) -> list:
        """
        识别音频文件（用于测试）

        Args:
            audio_path: 音频文件路径

        Returns:
            识别结果列表 [{"text": ..., "start": ..., "end": ..., "confidence": ...}]
        """
        if self.model is None:
            self.load_model()

        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )

        results = []
        for seg in segments:
            results.append({
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
                "confidence": getattr(seg, 'avg_logprob',
                            getattr(seg, 'avg_log_probability', -1.0)),
            })
        return results
