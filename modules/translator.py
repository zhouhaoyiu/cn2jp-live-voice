"""
翻译模块 - 支持多种翻译模型的中日翻译

支持模型:
  - HY-MT1.5-1.8B (腾讯): 推荐，高质量低重复，1.8B decoder-only LLM
  - NLLB-200-Distilled-600M (Meta): 回退方案，轻量但有重复问题
"""
import logging
import re
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ━━━ 语言代码映射 ━━━
# NLLB 专用语言代码
LANG_CODE_MAP = {
    "zh": "zho_Hans",    # 简体中文
    "ja": "jpn_Jpan",    # 日语
    "en": "eng_Latn",    # 英语
    "ko": "kor_Hang",    # 韩语
}

# HY-MT 目标语言名称（中文指令用中文语言名）
HYMT_TGT_LANG = {
    "ja": "日语",
    "en": "英语",
    "ko": "韩语",
    "zh": "中文",
}

# ━━━ 模型类型自动检测 ━━━
def _detect_model_type(model_name: str) -> str:
    """
    根据模型名称自动检测模型类型

    Returns:
        "hymt" 或 "nllb"
    """
    name_lower = model_name.lower()
    if "hy-mt" in name_lower or "hymt" in name_lower:
        return "hymt"
    if "nllb" in name_lower:
        return "nllb"
    # 默认使用 hymt（推荐模型）
    return "hymt"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  辅助函数：重复检测、汉字→片假名
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _detect_repetition_unit(text: str, min_unit_len: int = 1, max_unit_len: int = 20) -> str:
    """
    检测文本中的重复单元

    使用高效算法检测文本中连续重复出现的最短子串。
    例如: "私はリ・リ・リ・リ・..." → 返回 "リ・"
    """
    if len(text) < min_unit_len * 4:
        return ""

    for unit_len in range(min_unit_len, min(max_unit_len + 1, len(text) // 3)):
        unit = text[:unit_len]
        count = 0
        pos = 0
        while pos <= len(text) - unit_len:
            if text[pos:pos + unit_len] == unit:
                count += 1
                pos += unit_len
            else:
                break
        if count >= 4:
            return unit

    for start in range(1, min(max_unit_len * 2, len(text) // 3)):
        remaining = text[start:]
        for unit_len in range(min_unit_len, min(max_unit_len + 1, len(remaining) // 3)):
            unit = remaining[:unit_len]
            count = 0
            pos = 0
            while pos <= len(remaining) - unit_len:
                if remaining[pos:pos + unit_len] == unit:
                    count += 1
                    pos += unit_len
                else:
                    break
            if count >= 4:
                return unit

    return ""


def _is_cjk_char(ch: str) -> bool:
    """判断字符是否为 CJK 汉字"""
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2A6DF)


# ━━━ 拼音→片假名 映射表 ━━━
_PINYIN_KATAKANA = {
    "a": "ア", "ai": "アイ", "an": "アン", "ang": "アン", "ao": "アオ",
    "ba": "バ", "bai": "バイ", "ban": "バン", "bang": "バン", "bao": "バオ", "bei": "ベイ", "ben": "ベン", "beng": "ベン", "bi": "ビ", "bian": "ビェン", "biao": "ビャオ", "bie": "ビェ", "bin": "ビン", "bing": "ビン", "bo": "ボ", "bu": "ブ",
    "ca": "ツァ", "cai": "ツァイ", "can": "ツァン", "cang": "ツァン", "cao": "ツァオ", "ce": "ツァ", "cen": "ツェン", "ceng": "ツェン", "cha": "チャ", "chai": "チャイ", "chan": "チャン", "chang": "チャン", "chao": "チャオ", "che": "チェ", "chen": "チェン", "cheng": "チェン", "chi": "チー", "chong": "チョン", "chou": "チョウ", "chu": "チュ", "chua": "チュア", "chuai": "チュアイ", "chuan": "チュアン", "chuang": "チュアン", "chui": "チュイ", "chun": "チュン", "chuo": "チュオ",
    "ci": "ツー", "cong": "ツォン", "cou": "ツォウ", "cu": "ツ", "cuan": "ツァン", "cui": "ツイ", "cun": "ツン", "cuo": "ツォ",
    "da": "ダ", "dai": "ダイ", "dan": "ダン", "dang": "ダン", "dao": "ダオ", "de": "ダ", "dei": "デイ", "den": "デン", "deng": "デン", "di": "ディ", "dia": "ディア", "dian": "ディェン", "diao": "ディャオ", "die": "ディェ", "ding": "ディン", "diu": "ディウ", "dong": "ドン", "dou": "ドウ", "du": "ドゥ", "duan": "ドゥアン", "dui": "ドゥイ", "dun": "ドゥン", "duo": "ドゥオ",
    "e": "エ", "ei": "エイ", "en": "エン", "eng": "エン", "er": "アル",
    "fa": "ファ", "fan": "ファン", "fang": "ファン", "fei": "フェイ", "fen": "フェン", "feng": "フォン", "fo": "フォ", "fou": "フォウ", "fu": "フー",
    "ga": "ガ", "gai": "ガイ", "gan": "ガン", "gang": "ガン", "gao": "ガオ", "ge": "ガ", "gei": "ゲイ", "gen": "ゲン", "geng": "ゲン", "gong": "ゴン", "gou": "ゴウ", "gu": "グ", "gua": "グア", "guai": "グアイ", "guan": "グァン", "guang": "グァン", "gui": "グイ", "gun": "グン", "guo": "グオ",
    "ha": "ハ", "hai": "ハイ", "han": "ハン", "hang": "ハン", "hao": "ハオ", "he": "ハ", "hei": "ヘイ", "hen": "ヘン", "heng": "ヘン", "hong": "ホン", "hou": "ホウ", "hu": "フー", "hua": "ファ", "huai": "ファイ", "huan": "ファン", "huang": "ファン", "hui": "フィ", "hun": "フン", "huo": "フォ",
    "ji": "ジ", "jia": "ジャ", "jian": "ジェン", "jiang": "ジャン", "jiao": "ジャオ", "jie": "ジェ", "jin": "ジン", "jing": "ジン", "jiong": "ジョン", "jiu": "ジウ", "ju": "ジュ", "juan": "ジュァン", "jue": "ジュエ", "jun": "ジュン",
    "ka": "カ", "kai": "カイ", "kan": "カン", "kang": "カン", "kao": "カオ", "ke": "カ", "ken": "ケン", "keng": "ケン", "kong": "コン", "kou": "コウ", "ku": "ク", "kua": "クア", "kuai": "クアイ", "kuan": "クァン", "kuang": "クァン", "kui": "クイ", "kun": "クン", "kuo": "クオ",
    "la": "ラ", "lai": "ライ", "lan": "ラン", "lang": "ラン", "lao": "ラオ", "le": "ラ", "lei": "レイ", "leng": "レン", "li": "リー", "lia": "リア", "lian": "リェン", "liang": "リャン", "liao": "リャオ", "lie": "リェ", "lin": "リン", "ling": "リン", "liu": "リウ", "lo": "ロ", "long": "ロン", "lou": "ロウ", "lu": "ルー", "lv": "リュ", "luan": "ルァン", "lve": "リュエ", "lun": "ルン", "luo": "ルオ",
    "ma": "マ", "mai": "マイ", "man": "マン", "mang": "マン", "mao": "マオ", "me": "マ", "mei": "メイ", "men": "メン", "meng": "モン", "mi": "ミ", "mian": "ミェン", "miao": "ミャオ", "mie": "ミェ", "min": "ミン", "ming": "ミン", "miu": "ミウ", "mo": "モ", "mou": "モウ", "mu": "ムー",
    "na": "ナ", "nai": "ナイ", "nan": "ナン", "nang": "ナン", "nao": "ナオ", "ne": "ナ", "nei": "ネイ", "nen": "ネン", "neng": "ネン", "ni": "ニー", "nian": "ニェン", "niang": "ニャン", "niao": "ニャオ", "nie": "ニェ", "nin": "ニン", "ning": "ニン", "niu": "ニウ", "nong": "ノン", "nou": "ノウ", "nu": "ヌー", "nv": "ニュ", "nuan": "ヌァン", "nve": "ニュエ", "nun": "ヌン", "nuo": "ノオ",
    "o": "オ", "ou": "オウ",
    "pa": "パ", "pai": "パイ", "pan": "パン", "pang": "パン", "pao": "パオ", "pei": "ペイ", "pen": "ペン", "peng": "ポン", "pi": "ピー", "pian": "ピェン", "piao": "ピャオ", "pie": "ピェ", "pin": "ピン", "ping": "ピン", "po": "ポ", "pou": "ポウ", "pu": "プー",
    "qi": "チー", "qia": "チャ", "qian": "チェン", "qiang": "チャン", "qiao": "チャオ", "qie": "チェ", "qin": "チン", "qing": "チン", "qiong": "チョン", "qiu": "チウ", "qu": "チュー", "quan": "チュァン", "que": "チュエ", "qun": "チュン",
    "ran": "ラン", "rang": "ラン", "rao": "ラオ", "re": "ラ", "ren": "レン", "reng": "レン", "ri": "リー", "rong": "ロン", "rou": "ロウ", "ru": "ルー", "rua": "ルア", "ruan": "ルァン", "rui": "ルイ", "run": "ルン", "ruo": "ルオ",
    "sa": "サ", "sai": "サイ", "san": "サン", "sang": "サン", "sao": "サオ", "se": "サ", "sen": "セン", "seng": "セン", "sha": "シャ", "shai": "シャイ", "shan": "シャン", "shang": "シャン", "shao": "シャオ", "she": "シェ", "shei": "シェイ", "shen": "シェン", "sheng": "ション", "shi": "シー", "shou": "ショウ", "shu": "シュ", "shua": "シュア", "shuai": "シュアイ", "shuan": "シュァン", "shuang": "シュァン", "shui": "シュイ", "shun": "シュン", "shuo": "シュオ",
    "si": "スー", "song": "ソン", "sou": "ソウ", "su": "ス", "suan": "スァン", "sui": "スイ", "sun": "スン", "suo": "ソ",
    "ta": "タ", "tai": "タイ", "tan": "タン", "tang": "タン", "tao": "タオ", "te": "タ", "teng": "テン", "ti": "ティー", "tian": "ティェン", "tiao": "ティャオ", "tie": "ティェ", "ting": "ティン", "tong": "トン", "tou": "トウ", "tu": "トゥ", "tuan": "トゥァン", "tui": "トゥイ", "tun": "トゥン", "tuo": "トゥオ",
    "wa": "ワ", "wai": "ワイ", "wan": "ワン", "wang": "ワン", "wei": "ウェイ", "wen": "ウェン", "weng": "ウォン", "wo": "ウォ", "wu": "ウー",
    "xi": "シー", "xia": "シャ", "xian": "シェン", "xiang": "シャン", "xiao": "シャオ", "xie": "シェ", "xin": "シン", "xing": "シン", "xiong": "ション", "xiu": "シウ", "xu": "シュー", "xuan": "シュァン", "xue": "シュエ", "xun": "シュン",
    "ya": "ヤ", "yan": "イェン", "yang": "ヤン", "yao": "ヤオ", "ye": "イェ", "yi": "イー", "yin": "イン", "ying": "イン", "yo": "ヨ", "yong": "ヨン", "you": "ヨウ", "yu": "ユー", "yuan": "ユァン", "yue": "ユエ", "yun": "ユン",
    "za": "ザ", "zai": "ザイ", "zan": "ザン", "zang": "ザン", "zao": "ザオ", "ze": "ザ", "zei": "ゼイ", "zen": "ゼン", "zeng": "ゼン", "zha": "ジャ", "zhai": "ジャイ", "zhan": "ジャン", "zhang": "ジャン", "zhao": "ジャオ", "zhe": "ジャ", "zhei": "ジェイ", "zhen": "ジェン", "zheng": "ジェン", "zhi": "ジー", "zhong": "ジョン", "zhou": "ジョウ", "zhu": "ジュ", "zhua": "ジュア", "zhuai": "ジュアイ", "zhuan": "ジュァン", "zhuang": "ジュァン", "zhui": "ジュイ", "zhun": "ジュン", "zhuo": "ジュオ",
    "zi": "ズー", "zong": "ゾン", "zou": "ゾウ", "zu": "ズ", "zuan": "ズァン", "zui": "ズイ", "zun": "ズン", "zuo": "ゾ",
}

# 常见日本汉字的日文音読み
_JP_ONYOMI = {
    "世界": "セカイ", "今日": "キョウ", "日本": "ニホン", "先生": "センセイ",
    "元気": "ゲンキ", "出来": "デキ",
    "私": "ワタシ", "日": "ニチ", "本": "ホン", "大": "ダイ", "中": "チュウ", "国": "コク",
    "人": "ジン", "年": "ネン", "月": "ゲツ", "時": "ジ", "分": "フン",
    "生": "セイ", "前": "ゼン", "後": "ゴ", "新": "シン", "長": "チョウ",
    "高": "コウ", "小": "ショウ", "学": "ガク", "校": "コウ", "会": "カイ",
    "社": "シャ", "電": "デン", "車": "シャ", "東": "トウ", "西": "サイ",
    "南": "ナン", "北": "ホク", "山": "サン", "水": "スイ", "火": "カ",
    "風": "フウ", "地": "チ", "天": "テン", "気": "キ", "明": "メイ",
    "京": "キョウ", "都": "ト", "道": "ドウ", "県": "ケン", "市": "シ",
    "区": "ク", "町": "チョウ", "力": "リキ", "業": "ギョウ", "商": "ショウ",
    "経": "ケイ", "理": "リ", "法": "ホウ", "政": "セイ", "務": "ム",
    "外": "ガイ", "交": "コウ", "文": "ブン", "化": "カ", "語": "ゴ",
    "書": "ショ", "館": "カン", "院": "イン", "所": "ショ", "局": "キョク",
    "部": "ブ", "心": "シン", "思": "シ", "意": "イ", "見": "ケン",
    "話": "ワ", "問": "モン", "答": "トウ", "知": "チ", "信": "シン",
    "考": "コウ", "主": "シュ", "義": "ギ", "民": "ミン",
    "和": "ワ", "平": "ヘイ", "安": "アン", "全": "ゼン", "体": "タイ",
    "目": "モク", "手": "シュ", "口": "コウ", "耳": "ジ", "足": "ソク",
    "立": "リツ", "入": "ニュウ", "出": "シュツ", "開": "カイ", "閉": "ヘイ",
    "行": "コウ", "来": "ライ", "帰": "キ", "持": "ジ", "使": "シ",
    "作": "サク", "成": "セイ", "代": "ダイ",
}


def _pinyin_to_katakana(ch: str) -> str:
    """将单个汉字通过拼音转为片假名"""
    try:
        from pypinyin import pinyin, Style
        py = pinyin(ch, style=Style.TONE2, heteronym=False)
        if py and py[0]:
            raw = py[0][0]
            syllable = re.sub(r'\d', '', raw)
            return _PINYIN_KATAKANA.get(syllable, '')
    except Exception:
        pass
    return ''


def _cjk_to_katakana(text: str) -> str:
    """
    将日文文本中的中文汉字智能转换为片假名（pykakasi 版）

    利用 pykakasi 分词区分日语词汇和中文名字：
    - 多字 CJK 复合词 → 保留（GPT-SoVITS 能读）
    - 连续单字 CJK ≥2 → 中文人名 → pypinyin→片假名
    - 孤立单字 CJK → 日语常用字 → 保留
    - 无假名纯汉字 → 不转换
    """
    if not text:
        return text

    if not any(_is_cjk_char(ch) for ch in text):
        return text

    # 无假名（纯汉字）→ 不转换
    kana_count = sum(1 for ch in text
                     if (0x3040 <= ord(ch) <= 0x309F or 0x30A0 <= ord(ch) <= 0x30FF))
    if kana_count == 0:
        return text

    try:
        import pykakasi
        kks = pykakasi.kakasi()
        result = kks.convert(text)
    except ImportError:
        logger.debug("pykakasi 未安装，使用简化汉字转换策略")
        return _cjk_to_katakana_fallback(text)

    segments = []
    for item in result:
        orig = item['orig']
        kana = item.get('kana', orig)
        cjk_count = sum(1 for c in orig if _is_cjk_char(c))
        is_all_cjk = cjk_count == len(orig) and cjk_count > 0
        segments.append({
            'orig': orig, 'kana': kana,
            'is_cjk': is_all_cjk, 'cjk_len': cjk_count,
        })

    output_parts = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg['is_cjk'] and seg['cjk_len'] == 1:
            group = [seg]
            j = i + 1
            while j < len(segments) and segments[j]['is_cjk'] and segments[j]['cjk_len'] == 1:
                group.append(segments[j])
                j += 1
            if len(group) >= 2:
                for g in group:
                    kata = _pinyin_to_katakana(g['orig'])
                    output_parts.append(kata if kata else g['kana'])
                i = j
            else:
                output_parts.append(seg['orig'])
                i += 1
        else:
            output_parts.append(seg['orig'])
            i += 1

    converted = ''.join(output_parts)
    if converted != text:
        logger.info(f"🔤 汉字→片假名: '{text}' → '{converted}'")
    return converted


def _cjk_to_katakana_fallback(text: str) -> str:
    """_cjk_to_katakana 的回退版本（不依赖 pykakasi）"""
    if not text:
        return text

    kana_count = 0
    cjk_chars = []
    for i, ch in enumerate(text):
        cp = ord(ch)
        if (0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF):
            kana_count += 1
        elif _is_cjk_char(ch):
            cjk_chars.append((i, ch))

    if not cjk_chars:
        return text

    total_chars = len(text.replace(' ', ''))
    if total_chars > 0 and kana_count / total_chars < 0.2:
        return text

    output = list(text)

    for word in sorted(_JP_ONYOMI.keys(), key=len, reverse=True):
        if len(word) < 2:
            continue
        word_text = ''.join(output)
        idx = 0
        while True:
            pos = word_text.find(word, idx)
            if pos == -1:
                break
            replacement = _JP_ONYOMI[word]
            output[pos:pos + len(word)] = list(replacement) + [''] * (len(word) - len(replacement))
            word_text = ''.join(output)
            idx = pos + len(replacement)

    for i, ch in enumerate(output):
        if not ch or not _is_cjk_char(ch):
            continue
        kata = _JP_ONYOMI.get(ch) or _pinyin_to_katakana(ch)
        if kata:
            output[i] = kata

    converted = ''.join(ch for ch in output if ch)
    if converted != text:
        logger.info(f"🔤 汉字→片假名(fallback): '{text}' → '{converted}'")
    return converted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  翻译模块主类
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TranslatorModule:
    """
    中日翻译模块（支持 HY-MT1.5 和 NLLB）

    使用方式:
        translator = TranslatorModule(config)
        translator.load_model()
        result = translator.translate("你好世界")
        # 异步:
        translator.start(on_translated_callback)
        translator.feed_text("你好世界")
        translator.stop()
    """

    def __init__(self, config: dict):
        """
        初始化翻译模块

        Config 键（通用）:
            - model_name: 模型名称/路径（自动检测类型）
            - model_type: 显式指定 "hymt" 或 "nllb"（可选，默认自动检测）
            - device: 推理设备 (cuda/cpu/auto)
            - src_lang: 源语言 (zh)
            - tgt_lang: 目标语言 (ja)

        Config 键（HY-MT 专用）:
            - max_new_tokens: 最大生成 token 数（默认 512）
            - temperature: 采样温度（默认 0.7）
            - top_k: Top-K 采样（默认 20）
            - top_p: Top-P 采样（默认 0.6）
            - repetition_penalty: 重复惩罚（默认 1.05）

        Config 键（NLLB 专用）:
            - max_length: 最大生成长度（默认 128）
            - beam_size: 束搜索大小（默认 4）
            - no_repeat_ngram_size: 禁止 n-gram 重复（默认 2）
        """
        self.config = config
        self.model_name = config.get("model_name", "tencent/HY-MT1.5-1.8B")
        self.model_type = config.get("model_type") or _detect_model_type(self.model_name)
        self.device = config.get("device", "auto")
        self.src_lang = config.get("src_lang", "zh")
        self.tgt_lang = config.get("tgt_lang", "ja")

        # HY-MT 专用参数
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.temperature = config.get("temperature", 0.7)
        self.top_k = config.get("top_k", 20)
        self.top_p = config.get("top_p", 0.6)
        self.hymt_repetition_penalty = config.get("repetition_penalty", 1.05)

        # NLLB 专用参数
        self.max_length = config.get("max_length", 128)
        self.beam_size = config.get("beam_size", 4)
        self.nllb_repetition_penalty = config.get("repetition_penalty", 1.5)
        self.no_repeat_ngram_size = config.get("no_repeat_ngram_size", 2)

        self.tokenizer = None
        self.model = None
        self._device = None
        self._running = False
        self._thread = None
        self._text_queue = None
        self._on_translated: Optional[Callable] = None

        logger.info(f"翻译模型类型: {self.model_type} ({self.model_name})")

    def load_model(self):
        """加载翻译模型和分词器"""
        import torch

        if self.model_type == "hymt":
            self._load_hymt()
        else:
            self._load_nllb()

    def _load_hymt(self):
        """加载 HY-MT1.5 模型（decoder-only LLM）"""
        import os
        import torch
        import logging as _logging
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"加载 HY-MT 翻译模型: {self.model_name}")

        # 🔇 压缩 HTTP / HF Hub 的 INFO 日志（模型加载时太啰嗦）
        _logging.getLogger("httpx").setLevel(_logging.WARNING)
        _logging.getLogger("httpcore").setLevel(_logging.WARNING)
        _logging.getLogger("huggingface_hub").setLevel(_logging.WARNING)

        # 🔑 优先使用本地缓存，避免每次启动都联网检查
        # 策略：先 local_files_only=True，失败后再联网（含镜像回退）
        local_only = self.config.get("local_files_only", True)  # 默认优先本地

        # 🌐 联网时自动设置 HuggingFace 镜像（国内无法直连 huggingface.co）
        if not local_only:
            hf_endpoint = os.environ.get("HF_ENDPOINT", "")
            if not hf_endpoint:
                hf_endpoint = self.config.get("hf_endpoint", "")
            if not hf_endpoint:
                try:
                    import urllib.request
                    urllib.request.urlopen("https://huggingface.co", timeout=5)
                except Exception:
                    hf_endpoint = "https://hf-mirror.com"
                    logger.info("  🌐 检测到无法连接 HuggingFace，自动使用镜像: hf-mirror.com")
            if hf_endpoint:
                os.environ["HF_ENDPOINT"] = hf_endpoint
                logger.info(f"  HF_ENDPOINT: {hf_endpoint}")

        # 确定设备
        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # 选择精度：MPS 用 float16，CUDA 用 bfloat16，CPU 用 float32
        if device == "cpu":
            dtype = torch.float32
        elif device == "mps":
            dtype = torch.float16  # MPS 不完全支持 bfloat16
        else:
            dtype = torch.bfloat16

        logger.info(f"  设备: {device}, 精度: {dtype}, 本地优先: {local_only}")

        # HY-MT 使用 hunyuan_v1_dense 架构，需要 transformers>=4.56.0
        # 该架构从 4.56.0 起原生内置，无需 trust_remote_code
        import transformers
        min_version = "4.56.0"
        if tuple(int(x) for x in transformers.__version__.split('.')[:3]) < tuple(int(x) for x in min_version.split('.')):
            raise ImportError(
                f"HY-MT 模型需要 transformers>={min_version}（当前: {transformers.__version__}），"
                f"hunyuan_v1_dense 架构从此版本起才内置支持。\n"
                f"请升级: pip install transformers>={min_version}\n"
                f"⚠️  注意: GPT-SoVITS 与新版 transformers 不兼容，"
                f"需在独立 venv 中运行，详见 scripts/setup_gptsovits_env.sh"
            )

        # 🔑 加载策略：先本地，后联网
        # local_files_only=True 时不会联网，完全依赖缓存，无网也能用
        loaded = False
        load_error = None

        if local_only:
            # 第一次尝试：纯本地加载
            try:
                logger.info("  从本地缓存加载（不联网）...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                if device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, device_map="auto",
                        torch_dtype=dtype, local_files_only=True,
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, torch_dtype=dtype, local_files_only=True,
                    )
                    self.model = self.model.to(device)
                loaded = True
            except Exception as e:
                load_error = e
                logger.warning(f"  本地缓存加载失败: {e}")
                logger.info("  尝试联网下载...")

        if not loaded:
            # 联网加载（含镜像设置）
            if not local_only:
                # 还没设置过镜像，现在设置
                hf_endpoint = os.environ.get("HF_ENDPOINT", "")
                if not hf_endpoint:
                    hf_endpoint = self.config.get("hf_endpoint", "")
                if not hf_endpoint:
                    try:
                        import urllib.request
                        urllib.request.urlopen("https://huggingface.co", timeout=5)
                    except Exception:
                        hf_endpoint = "https://hf-mirror.com"
                        logger.info("  🌐 检测到无法连接 HuggingFace，自动使用镜像: hf-mirror.com")
                if hf_endpoint:
                    os.environ["HF_ENDPOINT"] = hf_endpoint
                    logger.info(f"  HF_ENDPOINT: {hf_endpoint}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="auto", torch_dtype=dtype,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=dtype,
                )
                self.model = self.model.to(device)

            # 下载成功后提示：下次可离线使用
            logger.info("  ✅ 模型已下载到本地缓存，下次启动可离线使用")

        self.model.eval()
        self._device = device
        logger.info(f"HY-MT 翻译模型加载完成 (设备: {device})")

    def _load_nllb(self):
        """加载 NLLB 模型（encoder-decoder）"""
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载 NLLB 翻译模型: {self.model_name}, 设备: {device}")

        local_only = self.config.get("local_files_only", True)
        loaded = False

        if local_only:
            try:
                logger.info("  从本地缓存加载（不联网）...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    src_lang=LANG_CODE_MAP[self.src_lang],
                    use_fast=False,
                    local_files_only=True,
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, local_files_only=True,
                )
                loaded = True
            except Exception as e:
                logger.warning(f"  本地缓存加载失败: {e}")
                logger.info("  尝试联网下载...")

        if not loaded:
            # 🌐 联网时自动设置镜像
            hf_endpoint = os.environ.get("HF_ENDPOINT", "") or self.config.get("hf_endpoint", "")
            if not hf_endpoint:
                try:
                    import urllib.request
                    urllib.request.urlopen("https://huggingface.co", timeout=5)
                except Exception:
                    hf_endpoint = "https://hf-mirror.com"
                    logger.info("  🌐 自动使用 HuggingFace 镜像: hf-mirror.com")
            if hf_endpoint:
                os.environ["HF_ENDPOINT"] = hf_endpoint

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                src_lang=LANG_CODE_MAP[self.src_lang],
                use_fast=False,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info("  ✅ 模型已下载到本地缓存，下次启动可离线使用")

        if device == "cuda":
            self.model = self.model.half()

        self.model = self.model.to(device)
        self.model.eval()
        self._device = device
        logger.info("NLLB 翻译模型加载完成")

    def translate(self, text: str) -> str:
        """
        同步翻译文本

        Args:
            text: 源语言文本

        Returns:
            翻译后的文本
        """
        if self.model is None:
            self.load_model()

        if not text or not text.strip():
            return ""

        try:
            if self.model_type == "hymt":
                translated = self._translate_hymt(text)
            else:
                translated = self._translate_nllb(text)

            # 🔄 重复过滤（NLLB 严重需要，HY-MT 轻度需要）
            translated = self._filter_repetition(translated)

            # 🧹 清理翻译结果：汉字→片假名、去中点等
            translated = self._clean_for_tts(translated)

            logger.info(f"翻译完成: {text} -> {translated}")
            return translated

        except Exception as e:
            import traceback
            logger.error(f"翻译出错: {e}")
            logger.error(traceback.format_exc())
            return ""

    def _translate_hymt(self, text: str) -> str:
        """HY-MT1.5 翻译（decoder-only LLM + chat template）"""
        import torch

        tgt_lang_name = HYMT_TGT_LANG.get(self.tgt_lang, self.tgt_lang)
        prompt = f"将以下文本翻译为{tgt_lang_name}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"

        messages = [{"role": "user", "content": prompt}]

        # 🔑 先用 apply_chat_template 生成文本 prompt，再手动 tokenize
        # 直接用 tokenize=True + return_tensors="pt" 在某些 transformers 版本
        # 会返回 BatchEncoding 而非 tensor，导致 .shape 报 AttributeError
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        input_len = input_ids.shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.hymt_repetition_penalty,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只取生成部分（跳过 prompt）
        generated_ids = outputs[0][input_len:]
        translated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if not translated or not translated.strip():
            logger.warning(f"HY-MT 生成为空，原始输出 token 数: {len(generated_ids)}")
            return ""

        # HY-MT 有时会在翻译后加解释文字，截取第一行
        translated = translated.split('\n')[0].strip()

        # 移除可能残留的引号
        if (translated.startswith('"') and translated.endswith('"')) or \
           (translated.startswith('「') and translated.endswith('」')):
            translated = translated[1:-1].strip()

        return translated

    def _translate_nllb(self, text: str) -> str:
        """NLLB 翻译（encoder-decoder）"""
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self._device)

        tgt_lang_code = LANG_CODE_MAP[self.tgt_lang]
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_code]
        else:
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=self.max_length,
                num_beams=self.beam_size,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                repetition_penalty=self.nllb_repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                length_penalty=0.6,
            )

        result = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return result[0] if result else ""

    @staticmethod
    def _filter_repetition(text: str, max_repeat: int = 3) -> str:
        """
        过滤翻译重复问题（NLLB 严重，HY-MT 轻度）

        三层策略：长度安全阀 → 子串重复检测 → 词频统计
        """
        if not text:
            return text

        original_text = text
        original_len = len(text)

        # 策略0: 长度安全阀
        max_safe_len = max(original_len * 8, 50)
        if len(text) > max_safe_len:
            unit = _detect_repetition_unit(text)
            if unit:
                prefix = text[:text.find(unit)]
                result = prefix + (unit * max_repeat)
                result = re.sub(r'[・！!?？、。,\s]+$', '', result)
                logger.info(f"🔄 重复过滤(长度安全): {len(original_text)}→{len(result)}chars")
                return result
            else:
                result = text[:max_safe_len]
                result = re.sub(r'[・！!?？、。,\s]+$', '', result)
                logger.warning(f"🔄 翻译结果异常长({len(original_text)}chars)，截断到 {len(result)}chars")
                return result

        # 策略1: 子串重复检测
        unit = _detect_repetition_unit(text)
        if unit and len(unit) <= len(text) // (max_repeat + 1):
            count = text.count(unit)
            if count > max_repeat:
                prefix_end = text.find(unit)
                prefix = text[:prefix_end]
                result = prefix + (unit * max_repeat)
                result = re.sub(r'[・！!?？、。,\s]+$', '', result)
                logger.info(f"🔄 重复过滤(子串): {count}次→{max_repeat}次")
                return result

        # 策略2: 词频统计
        sep_pattern = r'([・！!?？、。,、\s·\u30FB\uFF65]+)'
        parts = re.split(sep_pattern, text)
        words = [p for p in parts if p and not re.match(sep_pattern.strip('()[]+'), p)]

        if len(words) > max_repeat:
            from collections import Counter
            word_counts = Counter(words)
            most_common_word, most_common_count = word_counts.most_common(1)[0]

            if most_common_count > max_repeat and most_common_count / len(words) > 0.4:
                seen = {}
                filtered_words = []
                for w in words:
                    seen[w] = seen.get(w, 0) + 1
                    if seen[w] <= max_repeat:
                        filtered_words.append(w)
                result = ""
                word_idx = 0
                for p in parts:
                    if p and not re.match(sep_pattern.strip('()[]+'), p):
                        if word_idx < len(filtered_words) and p == filtered_words[word_idx]:
                            result += p
                            word_idx += 1
                    else:
                        if word_idx > 0 or not p:
                            result += p
                result = re.sub(r'[・！!?？、。,、\s·]+$', '', result)
                logger.info(f"🔄 重复过滤(词频): '{original_text[:40]}...' → '{result}'")
                return result

        return text

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """
        清理翻译结果中影响 TTS 合成质量的字符

        1. 汉字→片假名（GPT-SoVITS ja 模式读不了中文汉字）
        2. 移除日文中点 "・"
        3. 移除多余空格和首尾标点
        """
        if not text:
            return text

        original = text

        # 1. 汉字→片假名
        text = _cjk_to_katakana(text)

        # 2. 移除中点
        text = text.replace('・', '').replace('･', '')

        # 3. 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 4. 移除首尾标点
        text = re.sub(r'^[、。！？\s]+', '', text)
        text = re.sub(r'[、。！？\s]+$', '', text)

        if text != original:
            logger.info(f"🧹 TTS文本清理: '{original}' → '{text}'")

        return text

    def start(self, on_translated: Callable[[str, str], None], text_queue=None):
        """启动异步翻译线程"""
        import queue

        if self.model is None:
            self.load_model()

        self._on_translated = on_translated
        self._text_queue = text_queue or queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._translate_loop, daemon=True)
        self._thread.start()
        logger.info("翻译线程已启动")

    def stop(self):
        """停止翻译线程"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("翻译线程已停止")

    def feed_text(self, text: str):
        """送入待翻译文本"""
        if self._running and self._text_queue:
            self._text_queue.put(text)

    def _translate_loop(self):
        """翻译处理主循环"""
        while self._running:
            try:
                text = self._text_queue.get(timeout=0.5)
            except Exception:
                continue

            if not text or not text.strip():
                continue

            translated = self.translate(text)
            if translated and self._on_translated:
                self._on_translated(text, translated)
