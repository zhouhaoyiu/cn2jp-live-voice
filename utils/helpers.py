"""
工具函数 - 设备检测、配置加载、日志设置等
"""
import logging
import sys
import platform
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """
    自动检测最佳推理设备

    Returns:
        "cuda" | "mps" | "cpu"
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger.info(f"检测到 CUDA GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("检测到 Apple MPS 加速")
            return "mps"
    except ImportError:
        pass

    logger.info("未检测到 GPU 加速，使用 CPU")
    return "cpu"


def get_config_path(env: Optional[str] = None) -> Path:
    """
    获取配置文件路径

    Args:
        env: 环境名称 (m4max/rtx4050/rtx4070 + _clone/_yue2zh 后缀 / None=自动检测)

    Returns:
        配置文件的完整路径
    """
    config_dir = Path(__file__).parent.parent / "configs"
    if env == "m4max":
        return config_dir / "m4max.yaml"
    elif env == "rtx4050":
        return config_dir / "rtx4050.yaml"
    elif env == "rtx4070":
        return config_dir / "rtx4070.yaml"
    elif env == "m4max_clone":
        return config_dir / "m4max_clone.yaml"
    elif env == "rtx4050_clone":
        return config_dir / "rtx4050_clone.yaml"
    elif env == "rtx4070_clone":
        return config_dir / "rtx4070_clone.yaml"
    elif env == "m4max_yue2zh":
        return config_dir / "m4max_yue2zh.yaml"
    elif env == "rtx4050_yue2zh":
        return config_dir / "rtx4050_yue2zh.yaml"
    elif env == "rtx4070_yue2zh":
        return config_dir / "rtx4070_yue2zh.yaml"
    else:
        # 自动检测：macOS 用 m4max，其他用 default
        if sys.platform == "darwin":
            m4max_path = config_dir / "m4max.yaml"
            if m4max_path.exists():
                logger.info("检测到 macOS，自动使用 m4max 配置")
                return m4max_path
        return config_dir / "default.yaml"


def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 自动检测设备
    if config.get("auto_detect_device", True):
        device = detect_device()
        for section in ["asr", "translator", "tts"]:
            if section in config and config[section].get("device") == "auto":
                config[section]["device"] = device

    # 设置相对路径为绝对路径
    project_root = path.parent.parent
    for section in ["tts"]:
        if section in config:
            key = "refer_wav_path"
            if key in config[section]:
                p = Path(config[section][key])
                if not p.is_absolute():
                    config[section][key] = str(project_root / p)

    return config


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    配置日志系统

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径（可选）
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 格式化器
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_fmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, date_fmt)

    # 控制台处理器
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def print_system_info():
    """打印系统信息"""
    import torch

    _logger = logging.getLogger(__name__)
    _logger.info("=" * 60)
    _logger.info("系统信息:")
    _logger.info(f"  Python: {sys.version}")
    _logger.info(f"  平台: {platform.system()} {platform.release()}")
    _logger.info(f"  PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        _logger.info(f"  CUDA: {torch.version.cuda}")
        _logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        _logger.info(f"  VRAM: {mem:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _logger.info("  加速: Apple MPS")
    else:
        _logger.info("  加速: CPU only")

    _logger.info("=" * 60)
