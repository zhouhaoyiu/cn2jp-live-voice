"""
音频采集模块 - 麦克风实时录音
支持多种采样率和声道配置，自动重采样到 16kHz 单声道
"""
import logging
import threading
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# PyAudio 常量
FORMAT_PA16 = 8  # paInt16
INPUT_MAP = {}


class AudioCapture:
    """
    实时音频采集模块

    从麦克风采集音频，自动重采样到 16kHz 单声道 float32 格式，
    通过回调函数实时输出音频数据。

    使用方式:
        capture = AudioCapture(config)
        capture.start(on_audio_callback)
        # ... 运行中 ...
        capture.stop()
    """

    def __init__(self, config: dict):
        """
        初始化音频采集模块

        Args:
            config: 配置字典，包含以下键:
                - sample_rate: 采集采样率 (默认 16000)
                - channels: 采集声道数 (默认 1)
                - chunk_size: 每次读取的帧数 (默认 1024)
                - device_index: 麦克风设备索引 (默认 None=系统默认)
                - target_sample_rate: 输出采样率 (默认 16000)
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.chunk_size = config.get("chunk_size", 1024)
        self.device_index = config.get("device_index", None)
        self.target_sample_rate = config.get("target_sample_rate", 16000)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pyaudio = None
        self._stream = None
        self._on_audio: Optional[Callable] = None

    @staticmethod
    def list_devices():
        """列出所有可用的音频输入设备"""
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    "index": i,
                    "name": info['name'],
                    "sample_rate": int(info['defaultSampleRate']),
                    "channels": info['maxInputChannels'],
                })
                logger.info(f"  [{i}] {info['name']} ({info['maxInputChannels']}ch, {info['defaultSampleRate']}Hz)")
        pa.terminate()
        return devices

    def start(self, on_audio: Callable[[np.ndarray], None]):
        """
        启动音频采集

        Args:
            on_audio: 音频数据回调，参数为 float32 numpy 数组 (16kHz 单声道)
        """
        import pyaudio

        self._on_audio = on_audio
        self._running = True

        # 初始化 PyAudio
        self._pyaudio = pyaudio.PyAudio()

        # 打开音频流
        stream_kwargs = {
            "format": FORMAT_PA16,
            "channels": self.channels,
            "rate": self.sample_rate,
            "input": True,
            "frames_per_buffer": self.chunk_size,
            "stream_callback": self._audio_callback,
        }
        if self.device_index is not None:
            stream_kwargs["input_device_index"] = self.device_index

        self._stream = self._pyaudio.open(**stream_kwargs)
        self._stream.start_stream()

        logger.info(f"音频采集已启动: {self.sample_rate}Hz, {self.channels}ch, chunk={self.chunk_size}")

    def stop(self):
        """停止音频采集"""
        self._running = False

        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

        logger.info("音频采集已停止")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 回调函数"""
        if not self._running:
            return (None, 2)  # paComplete

        if status:
            logger.warning(f"音频采集状态: {status}")

        try:
            # 将字节数据转换为 numpy 数组
            audio_int16 = np.frombuffer(in_data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # 多声道取平均
            if self.channels > 1:
                audio_float32 = audio_float32.reshape(-1, self.channels).mean(axis=1)

            # 重采样到目标采样率
            if self.sample_rate != self.target_sample_rate:
                audio_float32 = self._resample(audio_float32, self.sample_rate, self.target_sample_rate)
                audio_float32 = audio_float32.astype(np.float32)  # np.interp 返回 float64

            # 回调
            if self._on_audio:
                self._on_audio(audio_float32)

        except Exception as e:
            logger.error(f"音频回调出错: {e}")

        return (in_data, 0)  # paContinue

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """线性重采样"""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio)
