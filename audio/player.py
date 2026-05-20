"""
音频输出模块 - 将合成的语音输出到虚拟音频设备
用于 OBS 等直播软件采集
"""
import logging
import queue
import threading
import time
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

FORMAT_PA16 = 8  # paInt16


class AudioPlayer:
    """
    音频输出模块

    将合成的 float32 音频数据输出到指定音频设备，
    用于虚拟音频线缆 (VB-Audio Cable / BlackHole) 等设备，
    使 OBS 可以采集到合成的日语语音。

    使用方式:
        player = AudioPlayer(config)
        player.start()
        player.play(audio_data)  # float32 numpy array
        player.stop()
    """

    def __init__(self, config: dict):
        """
        初始化音频输出模块

        Args:
            config: 配置字典，包含以下键:
                - sample_rate: 输出采样率 (默认 16000)
                - channels: 输出声道数 (默认 1)
                - device_index: 输出设备索引 (None=默认)
                - buffer_size: 内部缓冲区大小 (秒)
                - volume: 音量 (0.0-2.0, 默认 1.0)
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.device_index = config.get("device_index", None)
        self.buffer_size = config.get("buffer_size", 10.0)
        self.volume = config.get("volume", 1.0)

        self._running = False
        self._max_buffer_samples = int(self.buffer_size * self.sample_rate)
        self._pyaudio = None
        self._stream = None
        self._buffer_lock = threading.Lock()
        self._play_buffer = np.array([], dtype=np.float32)

    @staticmethod
    def list_devices():
        """列出所有可用的音频输出设备"""
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                devices.append({
                    "index": i,
                    "name": info['name'],
                    "sample_rate": int(info['defaultSampleRate']),
                    "channels": info['maxOutputChannels'],
                })
                logger.info(f"  [{i}] {info['name']} ({info['maxOutputChannels']}ch, {info['defaultSampleRate']}Hz)")
        pa.terminate()
        return devices

    def start(self):
        """启动音频输出"""
        import pyaudio

        self._running = True
        self._pyaudio = pyaudio.PyAudio()

        stream_kwargs = {
            "format": FORMAT_PA16,
            "channels": self.channels,
            "rate": self.sample_rate,
            "output": True,
            "stream_callback": self._output_callback,
        }
        if self.device_index is not None:
            stream_kwargs["output_device_index"] = self.device_index

        self._stream = self._pyaudio.open(**stream_kwargs)
        self._stream.start_stream()

        logger.info(f"音频输出已启动: {self.sample_rate}Hz, {self.channels}ch")

    def stop(self):
        """停止音频输出"""
        self._running = False

        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

        logger.info("音频输出已停止")

    def play(self, audio_data: np.ndarray):
        """
        播放音频数据

        Args:
            audio_data: float32 numpy 数组 (16kHz 单声道)
        """
        if not self._running:
            logger.warning("音频输出未启动，忽略播放请求")
            return

        # 音量调节
        if self.volume != 1.0:
            audio_data = audio_data * self.volume

        # 裁剪防止爆音
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # 加入播放队列（限制最大缓冲）
        with self._buffer_lock:
            self._play_buffer = np.concatenate([self._play_buffer, audio_data])
            # 防止缓冲区无限增长
            if len(self._play_buffer) > self._max_buffer_samples:
                self._play_buffer = self._play_buffer[-self._max_buffer_samples:]

        logger.debug(f"音频入队: +{len(audio_data)} samples, 缓冲区: {len(self._play_buffer)} samples")

    def _output_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 输出回调"""
        if not self._running:
            return (b'\x00' * frame_count * self.channels * 2, 2)

        with self._buffer_lock:
            if len(self._play_buffer) >= frame_count:
                # 从缓冲区取数据
                out_float = self._play_buffer[:frame_count]
                self._play_buffer = self._play_buffer[frame_count:]
            else:
                # 缓冲区不足，补零
                out_float = np.zeros(frame_count, dtype=np.float32)
                available = len(self._play_buffer)
                if available > 0:
                    out_float[:available] = self._play_buffer
                    self._play_buffer = np.array([], dtype=np.float32)

        # 转换为 int16
        out_int16 = (out_float * 32767).astype(np.int16)

        # 多声道复制
        if self.channels > 1:
            out_int16 = np.repeat(out_int16[:, np.newaxis], self.channels, axis=1).flatten()

        return (out_int16.tobytes(), 0)  # paContinue

    def get_buffer_duration(self) -> float:
        """获取当前缓冲区时长（秒）"""
        with self._buffer_lock:
            return len(self._play_buffer) / self.sample_rate

    def is_playing(self) -> bool:
        """是否有音频正在播放"""
        return self.get_buffer_duration() > 0.05
