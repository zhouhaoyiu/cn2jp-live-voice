"""
TTS 模块 - 基于 GPT-SoVITS 的语音克隆合成
通过 GPT-SoVITS API Server 实现日语语音克隆
"""
import logging
import io
import queue
import threading
import time
import wave
import numpy as np
from typing import Optional, Callable, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSModule:
    """
    语音克隆 TTS 模块

    连接 GPT-SoVITS API Server 进行语音合成，
    支持参考音频音色克隆，将日语文本合成为目标音色语音。

    使用方式:
        tts = TTSModule(config)
        # 同步调用
        audio, sr = tts.synthesize("こんにちは")
        # 异步流式
        tts.start(on_audio_callback)
        tts.feed_text("こんにちは")
        tts.stop()
    """

    def __init__(self, config: dict):
        """
        初始化 TTS 模块

        Args:
            config: 配置字典，包含以下键:
                - api_url: GPT-SoVITS API 地址
                - refer_wav_path: 参考音频路径（用于音色克隆）
                - prompt_text: 参考音频对应文本
                - prompt_language: 参考音频语言 (zh/ja/en)
                - output_language: 输出语言 (ja)
                - speed: 语速 (0.5-2.0)
                - top_k: Top-K 采样参数
                - top_p: Top-P 采样参数
                - temperature: 温度参数
                - timeout: API 请求超时(秒)
                - retry_count: 重试次数
        """
        self.config = config
        self.api_url = config.get("api_url", "http://127.0.0.1:9880")
        self.refer_wav_path = config.get("refer_wav_path", "reference_audio/my_voice.wav")
        self.prompt_text = config.get("prompt_text", "")
        self.prompt_language = config.get("prompt_language", "zh")
        self.output_language = config.get("output_language", "ja")
        self.speed = config.get("speed", 1.0)
        self.top_k = config.get("top_k", 5)
        self.top_p = config.get("top_p", 0.8)
        self.temperature = config.get("temperature", 0.6)
        self.min_text_length = config.get("min_text_length", 3)
        self.seed = config.get("seed", 42)  # 固定种子保证直播音色一致
        self.repetition_penalty = config.get("repetition_penalty", 1.35)
        self.text_split_method = config.get("text_split_method", "cut5")
        self.timeout = config.get("timeout", 120)  # CPU 模式需要更长时间
        self.retry_count = config.get("retry_count", 3)  # 增加：需要尝试 mix 模式 + 换 seed

        self._running = False
        self._thread = None
        self._text_queue: Optional[queue.Queue] = None
        self._on_audio: Optional[Callable] = None
        self._session = None

        # 启动时检查关键配置
        self._check_config()

    def _check_config(self):
        """检查关键配置项"""
        # 检查参考音频
        refer_path = Path(self.refer_wav_path)
        if not refer_path.exists():
            logger.warning(f"⚠ 参考音频文件不存在: {self.refer_wav_path}")
            logger.warning("  请先录制或转换参考音频:")
            logger.warning("    python3 record_reference.py --duration 10")
            logger.warning("    python3 convert_audio.py your_audio.mp3 reference_audio/my_voice.wav")
        else:
            # 检查是否为有效 WAV 文件
            try:
                with wave.open(str(refer_path), 'rb') as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    frames = wf.getnframes()
                    duration = frames / sr
                logger.info(f"参考音频: {self.refer_wav_path} ({sr}Hz, {ch}ch, {sw*8}bit, {duration:.1f}s)")
                if duration < 3:
                    logger.warning(f"  ⚠ 参考音频太短 ({duration:.1f}s)，建议 3-10 秒")
                if duration > 10:
                    logger.warning(f"  ⚠ 参考音频太长 ({duration:.1f}s)，建议 3-10 秒，超过10秒会被自动截取")
            except Exception as e:
                logger.error(f"⚠ 参考音频文件格式无效: {e}")
                logger.error("  文件可能不是标准 WAV 格式，请用 convert_audio.py 转换:")
                logger.error("    python3 convert_audio.py your_audio.mp3 reference_audio/my_voice.wav")

        # 检查 prompt_text
        if not self.prompt_text:
            logger.warning("⚠ prompt_text 为空！GPT-SoVITS 需要参考音频对应的文本才能克隆音色")
            logger.warning("  请在你的 YAML 配置文件中设置 prompt_text 字段")
            logger.warning("  例如: prompt_text: \"大家好，欢迎来到我的直播间\"")
            logger.warning("  提示: clone 模式推荐使用 --env m4max_clone / rtx4050_clone / rtx4070_clone")

    def check_server(self) -> bool:
        """检查 GPT-SoVITS API Server 是否可用"""
        import requests
        try:
            # GPT-SoVITS 没有根路径，404 也说明服务在运行
            resp = requests.get(f"{self.api_url}/", timeout=3)
            return resp.status_code in (200, 404)
        except Exception:
            return False

    @staticmethod
    def _has_rare_chars(text: str) -> bool:
        """快速检测文本是否含 CJK 汉字（可能包含 GPT-SoVITS 不认识的字）"""
        for ch in text:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF:
                return True
        return False

    @staticmethod
    def _has_english_chars(text: str) -> bool:
        """检测文本是否包含英文字母（GPT-SoVITS 中文模式遇到英文会崩）"""
        return any('a' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in text)

    @staticmethod
    def _sanitize_english_for_zh(text: str) -> str:
        """
        将中文文本中的英文字母转成中文音译，避免 GPT-SoVITS 报 text.english 模块错误。

        GPT-SoVITS 的中文模式 (text_lang="zh") 遇到英文字母时，
        内部会尝试加载 text.english 模块做英文发音处理，
        但如果 NLTK 未安装或数据缺失，就会报 "No module named 'text.english'" 错误。

        解决方案：在发送前把英文字母替换成中文谐音。

        例: "saki爬上床" → "莎纪爬上床"
             "ika君" → "伊卡君"
             "ABCD测试" → "A B C D测试" (未在映射表中的字母保留并加空格防崩)
        """
        import re

        # 常见英文名字/单词 → 中文音译映射
        # 主要覆盖日文名、常见人名、直播常用词
        NAME_MAP = {
            # 日文名/常见名
            "saki": "莎纪", "ika": "伊卡", "yuki": "由纪", "miku": "未来",
            "nana": "奈奈", "rina": "理奈", "hana": "花奈", "luna": "露娜",
            "kiki": "琪琪", "mimi": "咪咪", "nini": "妮妮",
            "aki": "秋", "kai": "凯", "rei": "玲", "mai": "舞",
            "ren": "莲", "shin": "新", "taro": "太郎", "jiro": "次郎",
            # 英文常见词
            "ok": "好的", "hi": "嗨", "hello": "哈喽", "hey": "嘿",
            "thanks": "谢谢", "sorry": "抱歉", "wow": "哇", "yeah": "耶",
            "no": "不", "yes": "是的", "oh": "哦", "lol": "哈哈",
            "gg": "吉吉", "afk": "暂离", "boss": "老板", "vip": "贵宾",
        }

        # 先尝试整词替换（优先匹配长词）
        result = text
        # 按词长降序排列，避免短词先匹配导致长词被截断
        sorted_names = sorted(NAME_MAP.keys(), key=len, reverse=True)
        for eng, chn in sorted_names:
            # 大小写不敏感匹配
            pattern = re.compile(re.escape(eng), re.IGNORECASE)
            result = pattern.sub(chn, result)

        # 剩余的零散英文字母：逐个替换为中文谐音
        LETTER_MAP = {
            'a': '诶', 'b': '比', 'c': '西', 'd': '迪', 'e': '伊',
            'f': '艾弗', 'g': '吉', 'h': '艾奇', 'i': '艾', 'j': '杰',
            'k': '凯', 'l': '艾尔', 'm': '艾姆', 'n': '恩', 'o': '欧',
            'p': '皮', 'q': '丘', 'r': '阿', 's': '艾斯', 't': '提',
            'u': '尤', 'v': '维', 'w': '达不溜', 'x': '艾克斯', 'y': '歪',
            'z': '贼',
        }

        # 把剩余连续英文字母片段逐个替换
        def replace_english_word(match):
            word = match.group(0)
            # 如果全小写或全大写且3字母以上，可能是未收录的名字
            # 尝试逐字母转中文
            chars = []
            for ch in word:
                lower = ch.lower()
                if lower in LETTER_MAP:
                    chars.append(LETTER_MAP[lower])
                else:
                    chars.append(ch)
            return ''.join(chars)

        result = re.sub(r'[a-zA-Z]{1,}', replace_english_word, result)

        if result != text:
            logger.info(f"🔧 英文→中文音译: '{text}' → '{result}'")

        return result

    @staticmethod
    def _japanese_to_zh_fallback(text: str) -> str:
        """
        当 GPT-SoVITS 缺少 text.japanese 模块时，将日语文本转为中文模式可读的格式。

        策略：
        1. 保留汉字（GPT-SoVITS 中文模式能读）
        2. 片假名 → 对应中文音译（サキ→莎纪、イカ→伊卡）
        3. 平假名 → 对应中文音译（さき→莎纪、いか→伊卡）
        4. 标点保留（中文TTS能处理日文标点）

        注意：这是降级方案，音质不如日语模式。推荐安装 pyopenjtalk 修复。

        例: "サキがベッドに乗る時" → "莎纪がベッドに乗る時"
             → 进一步 → "莎纪被别多に乗る時"（片假名逐字转换）
        """
        # 片假名 → 中文音译（常见词汇整词映射）
        KATAKANA_WORDS = {
            "サキ": "莎纪", "イカ": "伊卡", "ユキ": "由纪", "ミク": "未来",
            "ベッド": "床", "リンゴ": "苹果", "テレビ": "电视", "ノート": "笔记本",
            "フォーク": "叉子", "ナイフ": "刀", "バター": "黄油", "ミルク": "牛奶",
            "コーヒー": "咖啡", "ビール": "啤酒", "ワイン": "红酒", "ケーキ": "蛋糕",
            "レストラン": "餐厅", "ホテル": "酒店", "タクシー": "出租车",
            "エレベーター": "电梯", "コンビニ": "便利店",
        }

        # 片假名单字 → 中文谐音（降级用）
        KATAKANA_CHAR = {
            'ア': '阿', 'イ': '伊', 'ウ': '乌', 'エ': '诶', 'オ': '欧',
            'カ': '卡', 'キ': '基', 'ク': '库', 'ケ': '凯', 'コ': '考',
            'サ': '萨', 'シ': '西', 'ス': '斯', 'セ': '塞', 'ソ': '索',
            'タ': '塔', 'チ': '奇', 'ツ': '茨', 'テ': '特', 'ト': '托',
            'ナ': '纳', 'ニ': '尼', 'ヌ': '努', 'ネ': '内', 'ノ': '诺',
            'ハ': '哈', 'ヒ': '希', 'フ': '夫', 'ヘ': '赫', 'ホ': '霍',
            'マ': '马', 'ミ': '米', 'ム': '姆', 'メ': '梅', 'モ': '莫',
            'ヤ': '亚', 'ユ': '尤', 'ヨ': '约',
            'ラ': '拉', 'リ': '里', 'ル': '鲁', 'レ': '雷', 'ロ': '洛',
            'ワ': '瓦', 'ヲ': '沃', 'ン': '恩',
            # 浊音/半浊音
            'ガ': '嘎', 'ギ': '吉', 'グ': '古', 'ゲ': '盖', 'ゴ': '戈',
            'ザ': '扎', 'ジ': '吉', 'ズ': '兹', 'ゼ': '泽', 'ゾ': '佐',
            'ダ': '达', 'ヂ': '吉', 'ヅ': '兹', 'デ': '德', 'ド': '多',
            'バ': '巴', 'ビ': '比', 'ブ': '布', 'ベ': '贝', 'ボ': '波',
            'パ': '帕', 'ピ': '皮', 'プ': '普', 'ペ': '佩', 'ポ': '波',
        }

        # 平假名单字 → 中文谐音
        HIRAGANA_CHAR = {
            'あ': '阿', 'い': '伊', 'う': '乌', 'え': '诶', 'お': '欧',
            'か': '卡', 'き': '基', 'く': '库', 'け': '凯', 'こ': '考',
            'さ': '萨', 'し': '西', 'す': '斯', 'せ': '塞', 'そ': '索',
            'た': '塔', 'ち': '奇', 'つ': '茨', 'て': '特', 'と': '托',
            'な': '纳', 'に': '尼', 'ぬ': '努', 'ね': '内', 'の': '诺',
            'は': '哈', 'ひ': '希', 'ふ': '夫', 'へ': '赫', 'ほ': '霍',
            'ま': '马', 'み': '米', 'む': '姆', 'め': '梅', 'も': '莫',
            'や': '亚', 'ゆ': '尤', 'よ': '约',
            'ら': '拉', 'り': '里', 'る': '鲁', 'れ': '雷', 'ろ': '洛',
            'わ': '瓦', 'を': '沃', 'ん': '恩',
            # 浊音/半浊音
            'が': '嘎', 'ぎ': '吉', 'ぐ': '古', 'げ': '盖', 'ご': '戈',
            'ざ': '扎', 'じ': '吉', 'ず': '兹', 'ぜ': '泽', 'ぞ': '佐',
            'だ': '达', 'ぢ': '吉', 'づ': '兹', 'で': '德', 'ど': '多',
            'ば': '巴', 'び': '比', 'ぶ': '布', 'べ': '贝', 'ぼ': '波',
            'ぱ': '帕', 'ぴ': '皮', 'ぷ': '普', 'ぺ': '佩', 'ぽ': '波',
            # 拗音
            'きゃ': '卡', 'きゅ': '丘', 'きょ': '乔',
            'しゃ': '夏', 'しゅ': '修', 'しょ': '肖',
            'ちゃ': '查', 'ちゅ': '丘', 'ちょ': '乔',
            'にゃ': '尼亚', 'にゅ': '纽', 'にょ': '尼约',
            'ひゃ': '夏', 'ひゅ': '休', 'ひょ': '肖',
            'みゃ': '米亚', 'みゅ': '缪', 'みょ': '米约',
            'りゃ': '利亚', 'りゅ': '刘', 'りょ': '里约',
            'ぎゃ': '加', 'ぎゅ': '究', 'ぎょ': '乔',
            'じゃ': '加', 'じゅ': '就', 'じょ': '乔',
            'びゃ': '比亚', 'びゅ': '比尤', 'びょ': '比约',
            'ぴゃ': '皮亚', 'ぴゅ': '皮尤', 'ぴょ': '皮约',
        }

        import re

        result = text

        # 1. 先替换片假名整词（优先匹配长词）
        for kata, zh in sorted(KATAKANA_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
            result = result.replace(kata, zh)

        # 2. 替换剩余片假名单字
        def replace_katakana(match):
            ch = match.group(0)
            # 处理带长音符号的片假名（如 "ー" 重复前一个元音）
            return KATAKANA_CHAR.get(ch, ch)

        result = re.sub(r'[ア-ン゙゚ー]', replace_katakana, result)

        # 3. 替换平假名（先处理拗音，再处理单字）
        for hira, zh in sorted(HIRAGANA_CHAR.items(), key=lambda x: len(x[0]), reverse=True):
            result = result.replace(hira, zh)

        if result != text:
            logger.info(f"🔧 日语→中文降级转换: '{text[:50]}' → '{result[:50]}'")

        return result

    @staticmethod
    def _replace_rare_with_pinyin(text: str) -> str:
        """
        将汉字替换为拼音（GPT-SoVITS 不认识某些汉字时可读拼音）

        例: "我是明璐测试测试" → "wo shi ming lu ce shi ce shi"
        """
        try:
            from pypinyin import pinyin, Style
            result = pinyin(text, style=Style.NORMAL, heteronym=False)
            pinyin_text = ' '.join(py[0] for py in result if py)
            return pinyin_text
        except ImportError:
            logger.warning("pypinyin 未安装，无法替换生僻字为拼音")
            return text
        except Exception as e:
            logger.warning(f"拼音替换失败: {e}")
            return text

    @staticmethod
    def _replace_cjk_with_katakana(text: str) -> str:
        """
        将日文文本中的汉字替换为片假名（GPT-SoVITS 日语模式读不了中文汉字时的回退）

        使用翻译模块的 _cjk_to_katakana 做智能转换：
        - 保留日语常用词汇的汉字（元気、野田 等）
        - 转换中文名字汉字为片假名（李晶 → リージン）

        例: "私は李晶です" → "私はリージンです"
        """
        try:
            from modules.translator import _cjk_to_katakana
            return _cjk_to_katakana(text)
        except ImportError:
            logger.warning("translator 模块未找到，无法做汉字→片假名转换")
            return text
        except Exception as e:
            logger.warning(f"汉字→片假名转换失败: {e}")
            return text

    def _try_fix_nltk(self, error_text: str):
        """
        检测到 GPT-SoVITS 报 NLTK 缺失错误时，尝试自动下载 NLTK 数据

        GPT-SoVITS 处理含英文字母的文本（如 "大家好我是saki"）时，
        内部使用 NLTK 做 English text POS tagging，需要：
        - averaged_perceptron_tagger_eng
        - averaged_perceptron_tagger

        这个方法会尝试在用户级目录 ~/nltk_data 下载，
        GPT-SoVITS 重启后即可使用。
        """
        import re

        # 提取缺失的资源名
        missing = []
        match = re.search(r"Resource '(\w+)' not found", error_text)
        if match:
            missing.append(match.group(1))

        logger.error("=" * 60)
        logger.error("⚠️ GPT-SoVITS 缺少 NLTK 数据！")
        logger.error(f"   缺失资源: {missing if missing else 'averaged_perceptron_tagger_eng'}")
        logger.error("   原因: 文本中包含英文字母（如 'saki'），GPT-SoVITS 需要 NLTK 做英文处理")
        logger.error("")

        # 尝试自动下载到用户级目录
        try:
            import nltk
            logger.info("🔧 正在尝试自动下载 NLTK 数据到 ~/nltk_data ...")
            resources_to_download = missing if missing else [
                "averaged_perceptron_tagger_eng",
                "averaged_perceptron_tagger",
            ]
            for res in resources_to_download:
                logger.info(f"  下载: {res}")
                nltk.download(res, quiet=True)

            # 补充下载常用资源
            for extra in ["punkt_tab", "cmudict"]:
                try:
                    nltk.data.find(f"tokenizers/{extra}" if "punkt" in extra else f"corpora/{extra}")
                except LookupError:
                    logger.info(f"  下载: {extra}")
                    nltk.download(extra, quiet=True)

            logger.info("✅ NLTK 数据下载完成！")
            logger.info("⚠️  请重启 GPT-SoVITS API Server 后再试")
            logger.info("   重启: 在 GPT-SoVITS 终端 Ctrl+C 后重新运行 start_gptsovits.sh")
        except ImportError:
            logger.error("❌ 本环境未安装 nltk，无法自动修复")
            logger.error("   请在 GPT-SoVITS 的 Python 环境中运行:")
            logger.error("   pip install nltk")
            logger.error("   python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")
        except Exception as e:
            logger.error(f"❌ 自动下载失败: {e}")
            logger.error("   请手动在 GPT-SoVITS 的 Python 环境中运行:")
            logger.error("   python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")

        logger.error("")
        logger.error("💡 临时绕过: 使用纯中文文本（不含英文/字母）可以避免此错误")
        logger.error("=" * 60)

    def synthesize(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        同步合成语音

        Args:
            text: 待合成文本（日语）

        Returns:
            (float32 numpy 数组, sample_rate) 元组或 None
        """
        import requests

        if not text or not text.strip():
            logger.warning("TTS 收到空文本，跳过")
            return None

        # 过滤过短文本：GPT-SoVITS 对极短文本效果很差
        # 去除标点后检查实际字符数
        import re
        text_clean = re.sub(r'[^\w]', '', text, flags=re.UNICODE)
        if len(text_clean) < self.min_text_length:
            logger.warning(f"跳过过短文本: '{text}' (有效字符: {len(text_clean)} < {self.min_text_length})")
            return None

        # 检查参考音频
        refer_path = Path(self.refer_wav_path)
        refer_abs = ""
        if refer_path.exists():
            refer_abs = str(refer_path.resolve())
            # 验证 WAV 格式，并自动截取过长音频
            try:
                refer_abs = self._maybe_trim_ref_audio(refer_abs)
            except Exception as e:
                logger.error(f"参考音频处理失败: {e}")
                logger.error("请运行: python3 convert_audio.py your_audio.mp3 reference_audio/my_voice.wav")
                return None
        else:
            logger.warning(f"参考音频不存在: {self.refer_wav_path}，尝试不带参考音频合成")

        # 检查 prompt_text
        if not self.prompt_text and refer_abs:
            logger.warning("prompt_text 为空，音色克隆效果会大打折扣！")
            logger.warning("请在配置文件中设置 prompt_text（参考音频对应的文本）")
            logger.warning("提示: clone 模式推荐使用 --env m4max_clone / rtx4050_clone / rtx4070_clone")

        # 🔑 决定 text_lang：
        # - clone 模式（中文→中文）：用 "zh"
        # - translate 模式（翻译→日文）：用 "ja"
        #   翻译模块会预处理日文中的汉字→片假名，所以纯 ja 模式即可
        actual_text_lang = self.output_language

        # 🔧 中文模式预处理：把英文转成中文音译
        # GPT-SoVITS 中文模式遇到英文字母会尝试加载 text.english 模块，
        # 如果 NLTK 未安装就会报 "No module named 'text.english'" 错误
        sanitized_text = text
        if actual_text_lang == "zh" and self._has_english_chars(text):
            sanitized_text = self._sanitize_english_for_zh(text)
            if sanitized_text != text:
                logger.info(f"🔧 中文模式英文预处理: '{text[:50]}' → '{sanitized_text[:50]}'")

        # 构建请求体 - GPT-SoVITS API v2
        payload = {
            "ref_audio_path": refer_abs,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_language,
            "text": sanitized_text,
            "text_lang": actual_text_lang,
            "speed": self.speed,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "seed": self.seed,                         # 🔑 固定种子：保证同文本同音色
            "repetition_penalty": self.repetition_penalty,  # 防止重复生成
            "text_split_method": self.text_split_method,    # 文本切分方式
            "parallel_infer": True,
            "split_bucket": True,
            "batch_size": 1,
            "media_type": "wav",
        }

        prompt_preview = self.prompt_text[:20] if self.prompt_text else "(空)"
        logger.info(f"TTS 请求: text='{text}', lang={actual_text_lang}, "
                    f"ref_audio={bool(refer_abs)}, prompt='{prompt_preview}', "
                    f"seed={self.seed}, top_k={self.top_k}, temp={self.temperature}, "
                    f"timeout={self.timeout}s")

        for attempt in range(self.retry_count + 1):
            try:
                t0 = time.time()
                resp = requests.post(
                    f"{self.api_url}/tts",
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed = time.time() - t0

                if resp.status_code == 200:
                    content_type = resp.headers.get('content-type', '')
                    logger.debug(f"TTS 响应: status=200, content-type={content_type}, "
                                f"size={len(resp.content)} bytes, 耗时 {elapsed:.1f}s")

                    # 检查是否真的是音频数据
                    if len(resp.content) < 44:
                        logger.error(f"TTS 返回数据太短 ({len(resp.content)} bytes)，可能不是音频")
                        logger.error(f"响应内容: {resp.content[:200]}")
                        continue

                    # 检查 WAV 文件头
                    if resp.content[:4] != b'RIFF':
                        logger.error(f"TTS 返回的不是 WAV 格式，前4字节: {resp.content[:4]}")
                        logger.error(f"响应前100字节: {resp.content[:100]}")
                        # 可能是 JSON 错误信息
                        try:
                            error_json = resp.json()
                            logger.error(f"GPT-SoVITS 错误: {error_json}")
                        except Exception:
                            pass
                        continue

                    # 解码 WAV 音频
                    audio_data, sample_rate = self._decode_wav_bytes(resp.content)
                    duration = len(audio_data) / sample_rate

                    # 🎯 音频质量检查：检测静音/异常输出
                    rms = np.sqrt(np.mean(audio_data ** 2))
                    peak = np.max(np.abs(audio_data))
                    if rms < 0.001 or peak < 0.01:
                        logger.warning(f"TTS 输出疑似静音: rms={rms:.4f}, peak={peak:.4f}")
                        if attempt < self.retry_count:
                            # 🔧 重试策略（区分 ja/zh 模式）：
                            # ja 模式：汉字→片假名（GPT-SoVITS 日语模式读不了中文汉字）
                            # zh 模式：汉字→拼音（生僻字拼音回退）
                            import random
                            if attempt == 0 and self._has_rare_chars(text):
                                if self.output_language == "ja":
                                    # 日语模式：用翻译模块的 _cjk_to_katakana 转片假名
                                    kata_text = self._replace_cjk_with_katakana(text)
                                    if kata_text != text:
                                        payload["text"] = kata_text
                                        logger.warning(f"  汉字→片假名重试: '{text}' → '{kata_text}'")
                                    else:
                                        old_seed = payload.get("seed", -1)
                                        new_seed = random.randint(1, 999999)
                                        payload["seed"] = new_seed
                                        logger.warning(f"  换 seed 重试: {old_seed} → {new_seed}")
                                else:
                                    # 中文模式：拼音回退
                                    pinyin_text = self._replace_rare_with_pinyin(text)
                                    if pinyin_text != text:
                                        payload["text"] = pinyin_text
                                        logger.warning(f"  生僻字→拼音重试: '{text}' → '{pinyin_text}'")
                                    else:
                                        old_seed = payload.get("seed", -1)
                                        new_seed = random.randint(1, 999999)
                                        payload["seed"] = new_seed
                                        logger.warning(f"  换 seed 重试: {old_seed} → {new_seed}")
                            else:
                                old_seed = payload.get("seed", -1)
                                new_seed = random.randint(1, 999999)
                                payload["seed"] = new_seed
                                logger.warning(f"  换 seed 重试: {old_seed} → {new_seed}")
                            time.sleep(0.3)
                            continue
                        else:
                            logger.error(f"  重试 {self.retry_count} 次后仍为静音，放弃此文本")
                            break

                    # 检测异常时长（太短或太长）
                    # 日语正常语速约 8-12 字符/秒，中文约 4-6 字/秒
                    chars_per_sec = 6 if self.output_language == "zh" else 10
                    expected_min = max(len(text_clean) / chars_per_sec * 0.5, 0.3)  # 最少一半语速
                    expected_max = max(len(text_clean) / chars_per_sec * 3.0, 5.0)  # 最多三倍语速

                    if duration < 0.2:
                        logger.warning(f"TTS 输出太短 ({duration:.1f}s)，可能合成异常")
                        if attempt < self.retry_count:
                            logger.warning(f"  重试中 ({attempt + 1}/{self.retry_count})...")
                            time.sleep(0.5)
                            continue
                        else:
                            logger.error(f"  重试 {self.retry_count} 次后仍太短，放弃此文本")
                            break

                    if duration > expected_max:
                        logger.warning(f"TTS 输出异常长 ({duration:.1f}s > 预期最大 {expected_max:.1f}s)，"
                                      f"文本 '{text[:20]}' ({len(text_clean)} 字符)")
                        if duration > 30:
                            audio_data = audio_data[:30 * sample_rate]
                        elif attempt < self.retry_count:
                            # 换 seed 重试，可能得到正常长度
                            old_seed = payload.get("seed", -1)
                            import random
                            new_seed = random.randint(1, 999999)
                            payload["seed"] = new_seed
                            logger.warning(f"  时长异常，换 seed 重试: {old_seed} → {new_seed}")
                            time.sleep(0.3)
                            continue

                    logger.info(f"TTS 合成成功: '{text[:30]}' -> {duration:.1f}s 音频 @ {sample_rate}Hz, "
                                f"耗时 {elapsed:.1f}s, seed={payload.get('seed', self.seed)}")
                    return audio_data, sample_rate
                else:
                    logger.error(f"TTS API 返回错误 ({elapsed:.1f}s): HTTP {resp.status_code}")
                    logger.error(f"响应内容: {resp.text[:500]}")

                    # 🔧 检测 NLTK 缺失错误 - 自动尝试修复
                    if resp.status_code == 400 and "nltk" in resp.text.lower():
                        self._try_fix_nltk(resp.text)

                    # 如果是 400 错误，说明参数有问题，不需要重试
                    if resp.status_code == 400:
                        try:
                            error_json = resp.json()
                            logger.error(f"GPT-SoVITS 错误详情: {error_json}")
                        except Exception:
                            pass
                        break

            except requests.Timeout:
                logger.warning(f"TTS 请求超时 ({self.timeout}s) (尝试 {attempt + 1}/{self.retry_count + 1})")
                logger.warning("  CPU 模式推理较慢，可以增大配置中的 timeout 值")
            except requests.ConnectionError:
                logger.error("TTS 服务连接失败，请确认 GPT-SoVITS API Server 已启动")
                logger.error("  启动方式: cd GPT-SoVITS && python3 api_v2.py -a 127.0.0.1 -p 9880")
                break
            except Exception as e:
                logger.error(f"TTS 合成出错: {type(e).__name__}: {e}")

            if attempt < self.retry_count:
                wait_time = 1.0 * (attempt + 1)
                logger.info(f"等待 {wait_time:.1f}s 后重试...")
                time.sleep(wait_time)

        logger.error(f"TTS 合成最终失败: '{text[:50]}'")
        return None

    # GPT-SoVITS 参考音频最大时长（秒），超过会自动截取
    # 实测 token 率约 3.78 tokens/秒 (306s 音频 → 1157 tokens)
    # GPT 模型位置编码上限 512，预留 130 tokens 给文本
    # 安全上限 = (512 - 130) / 3.78 ≈ 101 秒，但留余量设为 10 秒
    # 建议：参考音频 3~10 秒，prompt_text 与音频内容完全一致
    MAX_REF_DURATION_SEC = 10
    # 截取后的临时文件名模板
    _TRIMMED_REF_NAME = "_trimmed_10s.wav"

    def _maybe_trim_ref_audio(self, ref_path: str) -> str:
        """
        如果参考音频超过 MAX_REF_DURATION_SEC 秒，自动截取前 N 秒并保存为临时文件。

        GPT-SoVITS v2 GPT 模型的位置编码上限为 512 tokens。
        实测 token 率约 3.78 tokens/秒，10 秒 ≈ 38 tokens + 文本 ≈ 130 = 168 < 512，安全。
        过长的参考音频（如 306s → 1157 tokens）会导致 tensor 维度不匹配:
          "The size of tensor a (1157) must match the size of tensor b (512)"

        截取策略：取前 10 秒。截取后 prompt_text 可能与音频不匹配，
        建议重新录制 3~10 秒的参考音频，确保 prompt_text 与音频内容完全一致。

        Returns:
            截取后的音频路径（原文件或截取后的临时文件）
        """
        with wave.open(ref_path, 'rb') as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            total_frames = wf.getnframes()
            duration = total_frames / sr

        if duration <= self.MAX_REF_DURATION_SEC:
            return ref_path  # 不需要截取

        # 检查是否已有截取版本
        from pathlib import PurePath
        stem = PurePath(ref_path).stem
        parent = PurePath(ref_path).parent
        trimmed_path = str(parent) + "/" + stem + self._TRIMMED_REF_NAME

        # 如果截取版本已存在且比原文件新，直接使用
        import os
        if os.path.exists(trimmed_path):
            if os.path.getmtime(trimmed_path) >= os.path.getmtime(ref_path):
                logger.info(f"📏 使用已截取的参考音频: {trimmed_path}")
                return trimmed_path

        # 执行截取
        max_frames = int(self.MAX_REF_DURATION_SEC * sr)
        logger.info(f"📏 参考音频太长 ({duration:.1f}s)，截取前 {self.MAX_REF_DURATION_SEC}s → {trimmed_path}")

        with wave.open(ref_path, 'rb') as wf_in:
            frames = wf_in.readframes(max_frames)
            with wave.open(trimmed_path, 'wb') as wf_out:
                wf_out.setnchannels(ch)
                wf_out.setsampwidth(sw)
                wf_out.setframerate(sr)
                wf_out.writeframes(frames)

        logger.info(f"📏 截取完成: {trimmed_path} ({self.MAX_REF_DURATION_SEC}s)")
        return trimmed_path

    def _decode_wav_bytes(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        解码 WAV 字节数据为 numpy 数组

        Args:
            wav_bytes: WAV 格式字节数据

        Returns:
            (float32 numpy 数组, sample_rate) 元组
        """
        with io.BytesIO(wav_bytes) as f:
            with wave.open(f, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

        logger.debug(f"WAV 解码: {sample_rate}Hz, {n_channels}ch, {sample_width*8}bit, {n_frames} frames")

        # 转换为 numpy 数组
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sample_width, np.int16)
        audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

        # 归一化到 [-1, 1]
        if sample_width == 2:
            audio /= 32768.0
        elif sample_width == 4:
            audio /= 2147483648.0
        elif sample_width == 1:
            audio /= 128.0
            audio -= 1.0

        # 多声道取平均
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        # 保留原始采样率，不重采样！
        # GPT-SoVITS v2 输出 32kHz
        return audio, sample_rate

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """简单线性重采样"""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def start(self, on_audio: Callable[[np.ndarray, int], None], text_queue=None):
        """
        启动异步 TTS 线程

        Args:
            on_audio: 音频回调函数，参数为 (float32 numpy 数组, sample_rate)
            text_queue: 文本队列（可选）
        """
        self._on_audio = on_audio
        self._text_queue = text_queue or queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._tts_loop, daemon=True)
        self._thread.start()
        logger.info("TTS 线程已启动")

    def stop(self):
        """停止 TTS 线程"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("TTS 线程已停止")

    def feed_text(self, text: str):
        """
        送入待合成文本

        Args:
            text: 日语文本
        """
        if self._running and self._text_queue:
            self._text_queue.put(text)

    def _tts_loop(self):
        """TTS 处理主循环"""
        while self._running:
            try:
                text = self._text_queue.get(timeout=0.5)
            except Exception:
                continue

            if not text or not text.strip():
                continue

            logger.info(f"🔊 TTS 合成中: '{text}'")
            try:
                result = self.synthesize(text)
                if result is not None and self._on_audio:
                    audio, sample_rate = result
                    duration = len(audio) / sample_rate
                    logger.info(f"🔊 TTS 完成: '{text}' -> {duration:.1f}s 音频 @ {sample_rate}Hz")
                    self._on_audio(audio, sample_rate)
                else:
                    logger.warning(f"🔊 TTS 失败: '{text}' - synthesize 返回 None")
            except Exception as e:
                logger.error(f"🔊 TTS 异常: '{text}' - {type(e).__name__}: {e}")
