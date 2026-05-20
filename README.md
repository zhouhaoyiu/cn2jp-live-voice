# 中文→日语实时语音转换系统

说中文，实时输出日语语音（保留本人音色），适用于直播场景。

## 系统架构

```
麦克风 → ASR(语音识别) → 翻译(中→日) → TTS(音色克隆) → 虚拟音频输出 → OBS
```

```mermaid
flowchart LR
	mic[麦克风] --> asr[ASR
faster-whisper]
	asr --> tr[翻译
HY-MT / NLLB]
	tr --> tts[TTS
GPT-SoVITS v2]
	tts --> out[虚拟音频输出]
	out --> obs[OBS]

	cfg[配置文件
configs/*.yaml] --> asr
	cfg --> tr
	cfg --> tts
	cfg --> out
	gsv[GPT-SoVITS
API Server] --> tts
```

- **ASR**: faster-whisper (中文语音识别)
- **翻译**: HY-MT1.5-1.8B (默认) / NLLB-200-Distilled-600M (可选)
- **TTS**: GPT-SoVITS v2 (语音克隆合成)
- **管道**: 多线程流水线并行处理

## 支持模式

- **翻译模式**：中文 → 日语 + 音色克隆（默认）
- **克隆模式**：中文 → 中文（不翻译，音色最稳定）
- **粤语转普通话**：粤语 → 普通话 + 音色克隆

## 快速开始

### 1. 环境安装

> 需要先安装 conda（Miniforge/Anaconda）。

**macOS (M4 Max 开发机):**
```bash
bash scripts/setup_env.sh
```

**Windows (RTX 4050 直播机):**
```bat
scripts\setup_env.bat
```

> 脚本会自动安装依赖、下载模型并搭建 GPT-SoVITS。
> 如需手动执行，可参考：
> ```bash
> conda create -n cn2jp python=3.11 -y
> conda activate cn2jp
> pip install torch torchvision torchaudio
> pip install -r requirements.txt
> python download_models.py
> ```
> `download_models.py` 默认下载 faster-whisper + NLLB；HY-MT 会在首次使用时自动从 Hugging Face 拉取。

> **重要：环境隔离**
> HY-MT 需要 `transformers>=4.56`，但 GPT-SoVITS 与新版 `transformers` 不兼容。
> macOS/Linux 建议使用独立 conda 环境运行 GPT-SoVITS：
> ```bash
> bash scripts/setup_gptsovits_env.sh
> conda activate gptsovits
> ```
> Windows 可手动创建 `gptsovits` conda 环境并固定 `transformers==4.45.0`。

### 环境版本（当前系统）

**cn2jp (主流程环境)**

| 包 | 版本 |
|---|---|
| python | 3.11.15 |
| pip | 26.0.1 |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| transformers | 5.8.0 |
| tokenizers | 0.22.2 |
| peft | 0.19.1 |
| accelerate | 1.13.0 |
| sentencepiece | 0.2.1 |
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.7.1 |
| huggingface-hub | 1.14.0 |
| numpy | 2.4.4 |
| pyaudio | 0.2.14 |
| requests | 2.34.0 |
| pykakasi | 2.3.0 |
| pypinyin | 0.55.0 |
| pyyaml | 6.0.3 |

**gptsovits (GPT-SoVITS 环境)**

| 包 | 版本 |
|---|---|
| python | 3.11.15 |
| pip | 26.0.1 |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| transformers | 4.45.0 |
| tokenizers | 0.20.3 |
| peft | 0.12.0 |
| accelerate | 1.13.0 |
| sentencepiece | 0.2.1 |
| huggingface-hub | 0.36.2 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| librosa | 0.10.2 |
| soundfile | 0.13.1 |
| pyopenjtalk | 0.4.1 |
| nltk | 3.9.4 |
| pypinyin | 0.55.0 |

### 2. 录制参考音频

```bash
# 激活 conda 环境
conda activate cn2jp

# 录制 10 秒自己的语音
python record_reference.py --duration 10
```

录制完成后：
- 将录音对应的**逐字文本**填写到配置文件的 `tts.prompt_text`
- 确认 `tts.refer_wav_path` 指向录音文件路径

### 3. 启动 GPT-SoVITS API Server

参考音频等参数通过 API 请求传递，无需命令行指定。

**macOS/Linux（独立 conda 环境，推荐）:**
```bash
bash scripts/start_gptsovits.sh
```

**macOS/Linux（已激活 gptsovits 环境）:**
```bash
./start_gptsovits.sh
```

**Windows:**
```bat
scripts\start_gptsovits.bat
```

**手动启动（可选）：**
```bash
conda activate gptsovits
cd GPT-SoVITS
python3 api_v2.py -a 127.0.0.1 -p 9880
```

### 4. 启动语音转换管道

**翻译模式（默认）:**
```bash
conda activate cn2jp
python main.py --env m4max
python main.py --env rtx4050
python main.py --env rtx4070
```

**克隆模式（不翻译）:**
```bash
conda activate cn2jp
python main.py --env m4max_clone
python main.py --env rtx4050_clone
python main.py --env rtx4070_clone
```

**粤语转普通话:**
```bash
conda activate cn2jp
python main.py --env m4max_yue2zh
```

> `--mode` 参数可以覆盖配置文件中的模式，例如：`--mode clone` / `--mode yue2zh`

## 测试各模块

```bash
# 列出音频设备
python main.py --list-devices

# 测试 ASR
python main.py --test-asr test_audio.wav

# 测试翻译
python main.py --test-translate "你好，欢迎来到我的直播间"

# 测试 TTS
python main.py --test-tts "こんにちは、私の配信へようこそ"

# 克隆模式测试
python main.py --test-tts "你好，欢迎来到我的直播间" --mode clone

# 粤语转普通话测试
python main.py --test-tts "我哋今晚开播" --mode yue2zh

# 修复 GPT-SoVITS 所需 NLTK 数据（处理中英混合文本）
python main.py --fix-nltk
```

## OBS 集成

### Windows (RTX 4050)
1. 安装 [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
2. 在 `configs/rtx4050.yaml` 中设置 `player.device_index` 为虚拟音频设备
3. OBS 音频源选择 "CABLE Output"
4. 设备索引可通过 `python main.py --list-devices` 查看

### macOS (M4 Max)
1. 安装 [BlackHole](https://existential.audio/blackhole/)
2. 在 `configs/m4max.yaml` 中设置 `player.device_index`
3. OBS 音频源选择 BlackHole
4. 设备索引可通过 `python main.py --list-devices` 查看

## 配置说明

| 配置文件 | 用途 |
|---------|------|
| `configs/default.yaml` | 默认配置，自动检测设备 |
| `configs/m4max.yaml` | M4 Max 36GB 开发环境（翻译模式） |
| `configs/m4max_clone.yaml` | M4 Max 纯音色克隆 |
| `configs/m4max_yue2zh.yaml` | M4 Max 粤语→普通话 |
| `configs/rtx4050.yaml` | RTX 4050 6GB 直播环境（翻译模式） |
| `configs/rtx4050_clone.yaml` | RTX 4050 纯音色克隆 |
| `configs/rtx4070.yaml` | RTX 4070 12GB 直播环境（翻译模式） |
| `configs/rtx4070_clone.yaml` | RTX 4070 纯音色克隆 |

## 项目结构

```
cn2jp-live-voice/
├── main.py                 # 主入口
├── configs/                # 配置文件
│   ├── default.yaml
│   ├── m4max.yaml
│   ├── m4max_clone.yaml
│   ├── m4max_yue2zh.yaml
│   ├── rtx4050.yaml
│   ├── rtx4050_clone.yaml
│   ├── rtx4070.yaml
│   └── rtx4070_clone.yaml
├── modules/                # 核心模块
│   ├── asr.py             # ASR 语音识别
│   ├── translator.py      # 中日翻译
│   └── tts.py             # TTS 音色克隆
├── audio/                  # 音频 I/O
│   ├── capture.py         # 麦克风采集
│   └── player.py          # 音频输出
├── pipeline/               # 管道编排
│   └── orchestrator.py    # 流式管道
├── utils/                  # 工具函数
│   └── helpers.py
├── scripts/                # 安装脚本
│   ├── setup_env.sh       # macOS/Linux 安装
│   ├── setup_env.bat      # Windows 安装
│   ├── setup_gptsovits_env.sh # GPT-SoVITS 独立 conda 环境
│   ├── start_gptsovits.sh  # GPT-SoVITS 启动（macOS/Linux）
│   └── start_gptsovits.bat # GPT-SoVITS 启动（Windows）
├── start_gptsovits.sh      # 启动 GPT-SoVITS API (macOS/Linux)
├── start_gptsovits.bat     # 启动 GPT-SoVITS API (Windows)
├── reference_audio/        # 参考音频目录
├── download_models.py      # 模型下载
├── setup_gptsovits.py      # GPT-SoVITS 搭建
├── record_reference.py     # 参考音频录制
└── requirements.txt        # Python 依赖
```

## 显存分配 (RTX 4050 - 6GB)

| 模块 | 模型 | 显存占用 |
|------|------|---------|
| ASR | whisper-base (int8) | ~1.0 GB |
| 翻译 | HY-MT1.5-1.8B (fp16) | ~4.0 GB |
| 翻译 | NLLB-600M (fp16, 可选) | ~1.5 GB |
| TTS | GPT-SoVITS (独立进程) | ~2.0 GB |
| 系统 | CUDA + 其他 | ~1.5 GB |
| **合计** | | **~6.0 GB (NLLB) / ~8.5 GB (HY-MT)** |

## 延迟预估

| 阶段 | 耗时 |
|------|------|
| ASR (whisper-base) | 200-400ms |
| 翻译 (HY-MT / NLLB) | 100-500ms |
| TTS (GPT-SoVITS) | 500-1000ms |
| **流水线端到端** | **~1.2-1.8s** |

## 常见问题

**GPT-SoVITS 报 transformers 版本冲突**
```bash
bash scripts/setup_gptsovits_env.sh
conda activate gptsovits
bash scripts/start_gptsovits.sh
```

**TTS 开头复述 prompt_text**
- 确保 `tts.prompt_text` 与参考音频内容**逐字一致**，不要多写文本。

**NLTK 数据缺失 / LookupError**
```bash
python main.py --fix-nltk
```

**没有声音 / OBS 没有输入**
- 用 `python main.py --list-devices` 找到虚拟音频设备索引。
- 确认 `player.sample_rate` 与 GPT-SoVITS 输出一致（默认 32000Hz）。

**HY-MT 显存不足**
- 4050 6GB 可能 OOM，改用 `facebook/nllb-200-distilled-600M`。
- 关闭其他占用 GPU 的程序，或改用 CPU/MPS。

**模型下载慢或失败**
- 设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- Windows: `set HF_ENDPOINT=https://hf-mirror.com` 或 PowerShell: `$env:HF_ENDPOINT="https://hf-mirror.com"`
