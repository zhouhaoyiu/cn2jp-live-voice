#!/usr/bin/env python3
"""
GPT-SoVITS v2 预训练模型下载脚本

自动下载 GPT-SoVITS v2 所需的全部预训练模型:
  - gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt (GPT)
  - gsv-v2final-pretrained/s2G2333k.pth (SoVITS)
  - chinese-roberta-wwm-ext-large/ (BERT) — 独立 HF 仓库
  - chinese-hubert-base/ (HuBERT) — 独立 HF 仓库

用法:
  conda activate gptsovits
  python scripts/download_gptsovits_models.py

支持国内 HF 镜像自动切换 (hf-mirror.com)
"""

import os
import sys
from pathlib import Path

# ─── 配置 ──────────────────────────────────────────────
HF_REPO_GPTSOVITS = "lj1995/GPT-SoVITS"       # GPT + SoVITS 主仓库
HF_REPO_BERT = "hfl/chinese-roberta-wwm-ext-large"  # BERT 独立仓库
HF_REPO_HUBERT = "TencentGameMate/chinese-hubert-base"  # HuBERT 独立仓库
HF_MIRROR = "hf-mirror.com"

# GPT-SoVITS 主仓库中的预训练模型路径前缀
GPTSOVITS_REPO_PREFIX = "GPT_SoVITS/pretrained_models"

# 需要从 GPT-SoVITS 主仓库下载的文件
GPTSOVITS_FILES = {
    "gsv-v2final-pretrained": {
        "files": [
            "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "s2G2333k.pth",
        ],
        "description": "GPT-SoVITS v2 预训练模型 (GPT + SoVITS)",
    },
}

# 独立 HuggingFace 模型仓库
SEPARATE_REPOS = {
    "chinese-roberta-wwm-ext-large": {
        "repo_id": HF_REPO_BERT,
        "description": "中文 BERT 模型 (文本语义提取)",
        # 典型文件列表，实际下载时先列出仓库文件
        "essential_patterns": ["config.json", "tokenizer", "model"],
    },
    "chinese-hubert-base": {
        "repo_id": HF_REPO_HUBERT,
        "description": "中文 HuBERT 模型 (语音特征提取)",
        "essential_patterns": ["config.json", "preprocessor_config.json", "model"],
    },
}


def detect_china_network():
    """检测是否在国内网络环境（无法直接访问 HuggingFace）"""
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://huggingface.co",
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        urllib.request.urlopen(req, timeout=8)
        return False
    except Exception:
        return True


def setup_hf_mirror():
    """设置 HF 镜像（国内用户）"""
    is_china = detect_china_network()
    if is_china:
        print(f"  [检测] 国内网络环境，使用 HF 镜像: {HF_MIRROR}")
        os.environ["HF_ENDPOINT"] = f"https://{HF_MIRROR}"
        return True
    else:
        print("  [检测] 可直接访问 HuggingFace")
        return False


def get_project_dir():
    """获取项目根目录"""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def get_pretrained_dir():
    """获取预训练模型目录"""
    return get_project_dir() / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"


def check_file_exists(pretrained_dir: Path, model_name: str, filename: str) -> bool:
    """检查单个模型文件是否存在且非空"""
    filepath = pretrained_dir / model_name / filename
    return filepath.exists() and filepath.stat().st_size > 0


def check_model_dir_exists(pretrained_dir: Path, model_name: str) -> bool:
    """检查模型目录是否包含有效文件"""
    model_dir = pretrained_dir / model_name
    if not model_dir.exists():
        return False
    # 至少要有 config.json + 一个模型文件
    has_config = (model_dir / "config.json").exists()
    has_model = (
        (model_dir / "model.safetensors").exists() or
        (model_dir / "pytorch_model.bin").exists() or
        list(model_dir.glob("*.safetensors.index.*")) or
        list(model_dir.glob("pytorch_model*.bin"))
    )
    return has_config and has_model


# ═══════════════════════════════════════════════════════
# 方式1: huggingface_hub Python API (hf_hub_download)
# ═══════════════════════════════════════════════════════

def _ensure_huggingface_hub():
    """确保 huggingface_hub 已安装"""
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        print("  [安装] huggingface_hub...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
            check=True
        )
        import huggingface_hub
        return huggingface_hub


def download_gptsovits_repo_files(pretrained_dir: Path, hf) -> bool:
    """从 GPT-SoVITS 主仓库下载特定文件"""
    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"\n  从 {HF_REPO_GPTSOVITS} 下载 GPT/SoVITS 模型...")

    # 先列出仓库文件，确认路径
    print("    探测仓库文件列表...")
    try:
        all_files = list_repo_files(HF_REPO_GPTSOVITS)
    except Exception as e:
        print(f"    ✗ 无法列出仓库文件: {e}")
        return False

    # 筛选出预训练模型相关文件
    pretrained_files = [f for f in all_files if f.startswith(f"{GPTSOVITS_REPO_PREFIX}/")]
    print(f"    仓库中 pretrained_models/ 下共 {len(pretrained_files)} 个文件")

    # 按 model_name 分组
    for model_name, info in GPTSOVITS_FILES.items():
        model_dir = pretrained_dir / model_name
        print(f"\n  [{model_name}]")
        print(f"    说明: {info['description']}")

        all_exist = all(
            check_file_exists(pretrained_dir, model_name, f)
            for f in info["files"]
        )
        if all_exist:
            print("    状态: ✓ 所有文件已存在，跳过")
            continue

        model_dir.mkdir(parents=True, exist_ok=True)

        for filename in info["files"]:
            if check_file_exists(pretrained_dir, model_name, filename):
                size_mb = (model_dir / filename).stat().st_size / (1024 * 1024)
                print(f"    ✓ {filename} 已存在 ({size_mb:.1f} MB)")
                continue

            # 在仓库中查找该文件
            repo_path = f"{GPTSOVITS_REPO_PREFIX}/{model_name}/{filename}"
            matching = [f for f in pretrained_files if f == repo_path]

            if not matching:
                # 尝试宽松匹配（可能路径略有不同）
                matching = [f for f in pretrained_files if filename in f]

            if not matching:
                print(f"    ✗ 文件不在仓库中: {repo_path}")
                print(f"    可用文件:")
                relevant = [f for f in pretrained_files if model_name in f]
                for f in relevant[:10]:
                    print(f"      {f}")
                return False

            actual_repo_path = matching[0]
            print(f"    ⬇ 下载: {actual_repo_path}")

            try:
                downloaded = hf_hub_download(
                    repo_id=HF_REPO_GPTSOVITS,
                    filename=actual_repo_path,
                    local_dir=str(pretrained_dir.parent.parent),  # GPT-SoVITS/
                )
                # hf_hub_download 下载到仓库目录结构中
                # 验证文件是否出现在正确位置
                target = model_dir / filename
                if target.exists() and target.stat().st_size > 0:
                    size_mb = target.stat().st_size / (1024 * 1024)
                    print(f"    ✓ 下载完成: {filename} ({size_mb:.1f} MB)")
                else:
                    # 文件可能下载到缓存但未复制到 local_dir
                    # 尝试从缓存复制
                    if downloaded and Path(downloaded).exists():
                        import shutil
                        shutil.copy2(downloaded, str(target))
                        size_mb = target.stat().st_size / (1024 * 1024)
                        print(f"    ✓ 复制完成: {filename} ({size_mb:.1f} MB)")
                    else:
                        print(f"    ✗ 下载后文件未找到: {filename}")
                        return False
            except Exception as e:
                print(f"    ✗ 下载失败: {e}")
                return False

    return True


def download_separate_repo(pretrained_dir: Path, model_name: str, repo_id: str,
                            description: str, hf) -> bool:
    """从独立的 HuggingFace 模型仓库下载整个模型"""
    from huggingface_hub import snapshot_download

    model_dir = pretrained_dir / model_name
    print(f"\n  [{model_name}]")
    print(f"    说明: {description}")

    if check_model_dir_exists(pretrained_dir, model_name):
        print("    状态: ✓ 模型已存在，跳过")
        return True

    print(f"    ⬇ 从 {repo_id} 下载...")

    try:
        # 下载整个模型仓库到指定目录
        downloaded = snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
        )
        print(f"    ✓ 下载完成: {model_dir}")
        return True
    except Exception as e:
        print(f"    ✗ 下载失败: {e}")
        return False


def download_method1_hf_hub(pretrained_dir: Path) -> bool:
    """方式1: 使用 huggingface_hub Python API"""
    hf = _ensure_huggingface_hub()

    # Step 1: 从 GPT-SoVITS 主仓库下载 GPT/SoVITS 模型
    if not download_gptsovits_repo_files(pretrained_dir, hf):
        return False

    # Step 2: 从独立仓库下载 BERT 和 HuBERT
    for model_name, info in SEPARATE_REPOS.items():
        if not download_separate_repo(
            pretrained_dir, model_name,
            info["repo_id"], info["description"], hf
        ):
            return False

    return True


# ═══════════════════════════════════════════════════════
# 方式2: huggingface-cli
# ═══════════════════════════════════════════════════════

def download_method2_hf_cli(pretrained_dir: Path) -> bool:
    """方式2: 使用 huggingface-cli 命令行工具"""
    import subprocess

    print("  检查 huggingface-cli...")
    try:
        subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  [安装] huggingface_hub CLI...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[cli]"],
            check=True
        )

    gptsovits_root = pretrained_dir.parent.parent  # GPT-SoVITS/

    # Step 1: GPT-SoVITS 主仓库的特定文件
    for model_name, info in GPTSOVITS_FILES.items():
        model_dir = pretrained_dir / model_name
        print(f"\n  [{model_name}]")
        print(f"    说明: {info['description']}")

        all_exist = all(
            check_file_exists(pretrained_dir, model_name, f)
            for f in info["files"]
        )
        if all_exist:
            print("    状态: ✓ 已存在，跳过")
            continue

        model_dir.mkdir(parents=True, exist_ok=True)

        for filename in info["files"]:
            if check_file_exists(pretrained_dir, model_name, filename):
                continue
            repo_path = f"{GPTSOVITS_REPO_PREFIX}/{model_name}/{filename}"
            print(f"    ⬇ 下载: {repo_path}")
            result = subprocess.run(
                [
                    "huggingface-cli", "download",
                    HF_REPO_GPTSOVITS,
                    repo_path,
                    "--local-dir", str(gptsovits_root),
                ],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"    ✗ 失败: {result.stderr.strip()}")
                return False
            if check_file_exists(pretrained_dir, model_name, filename):
                size_mb = (model_dir / filename).stat().st_size / (1024 * 1024)
                print(f"    ✓ 完成 ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ 下载后文件未找到")
                return False

    # Step 2: BERT 和 HuBERT 独立仓库
    for model_name, info in SEPARATE_REPOS.items():
        model_dir = pretrained_dir / model_name
        print(f"\n  [{model_name}]")
        print(f"    说明: {info['description']}")

        if check_model_dir_exists(pretrained_dir, model_name):
            print("    状态: ✓ 已存在，跳过")
            continue

        print(f"    ⬇ 从 {info['repo_id']} 下载...")
        result = subprocess.run(
            [
                "huggingface-cli", "download",
                info["repo_id"],
                "--local-dir", str(model_dir),
            ],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    ✗ 失败: {result.stderr.strip()}")
            return False
        print(f"    ✓ 下载完成")

    return True


# ═══════════════════════════════════════════════════════
# 方式3: wget/curl 直接下载 (最底层备用方案)
# ═══════════════════════════════════════════════════════

def _wget_download(url: str, output_path: str) -> bool:
    """用 wget 下载文件"""
    import subprocess
    result = subprocess.run(
        ["wget", "-q", "--show-progress", "-c", "-O", output_path, url],
    )
    return result.returncode == 0


def _curl_download(url: str, output_path: str) -> bool:
    """用 curl 下载文件"""
    import subprocess
    result = subprocess.run(
        ["curl", "-L", "-o", output_path, "-#", "-C", "-", url],
    )
    return result.returncode == 0


def download_method3_direct(pretrained_dir: Path) -> bool:
    """方式3: wget/curl 直接下载 (使用 HF 镜像)"""
    import subprocess

    # 确定下载工具
    download_fn = None
    if subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
        download_fn = _wget_download
        print("  使用 wget 下载")
    elif subprocess.run(["which", "curl"], capture_output=True).returncode == 0:
        download_fn = _curl_download
        print("  使用 curl 下载")
    else:
        print("  ✗ 未找到 wget 或 curl")
        return False

    # 强制使用 HF 镜像
    hf_host = os.environ.get("HF_ENDPOINT", f"https://{HF_MIRROR}").rstrip("/")
    base_url_main = f"{hf_host}/{HF_REPO_GPTSOVITS}/resolve/main"

    # Step 1: GPT-SoVITS 主仓库文件
    for model_name, info in GPTSOVITS_FILES.items():
        model_dir = pretrained_dir / model_name
        print(f"\n  [{model_name}]")
        print(f"    说明: {info['description']}")

        all_exist = all(
            check_file_exists(pretrained_dir, model_name, f)
            for f in info["files"]
        )
        if all_exist:
            print("    状态: ✓ 已存在，跳过")
            continue

        model_dir.mkdir(parents=True, exist_ok=True)

        for filename in info["files"]:
            if check_file_exists(pretrained_dir, model_name, filename):
                continue
            url = f"{base_url_main}/{GPTSOVITS_REPO_PREFIX}/{model_name}/{filename}"
            output_path = str(model_dir / filename)
            print(f"    ⬇ 下载: {filename}")
            print(f"       URL: {url}")
            if not download_fn(url, output_path):
                print(f"    ✗ 下载失败")
                return False
            if check_file_exists(pretrained_dir, model_name, filename):
                size_mb = (model_dir / filename).stat().st_size / (1024 * 1024)
                print(f"    ✓ 完成 ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ 下载后文件未找到")
                return False

    # Step 2: BERT 和 HuBERT 独立仓库
    for model_name, info in SEPARATE_REPOS.items():
        model_dir = pretrained_dir / model_name
        print(f"\n  [{model_name}]")
        print(f"    说明: {info['description']}")

        if check_model_dir_exists(pretrained_dir, model_name):
            print("    状态: ✓ 已存在，跳过")
            continue

        model_dir.mkdir(parents=True, exist_ok=True)
        repo_id = info["repo_id"]
        base_url = f"{hf_host}/{repo_id}/resolve/main"

        # 典型的模型文件列表
        possible_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
        ]

        # 先尝试下载 config.json 看仓库是否可访问
        test_url = f"{base_url}/config.json"
        test_output = str(model_dir / "config.json")
        print(f"    ⬇ 下载配置文件测试连接...")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        if not download_fn(test_url, tmp_path):
            # 尝试不带 /main 路径
            base_url_alt = f"{hf_host}/{repo_id}/resolve/master"
            test_url = f"{base_url_alt}/config.json"
            if download_fn(test_url, tmp_path):
                base_url = base_url_alt
            else:
                print(f"    ✗ 无法访问仓库: {repo_id}")
                Path(tmp_path).unlink(missing_ok=True)
                return False

        # 复制临时文件到目标位置
        import shutil
        shutil.move(tmp_path, test_output)

        # 下载所有可能的文件
        for filename in possible_files:
            output_path = str(model_dir / filename)
            if Path(output_path).exists():
                continue
            url = f"{base_url}/{filename}"
            print(f"    ⬇ {filename}", end=" ")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=f".{filename.split('.')[-1]}", delete=False) as tmp:
                tmp_path = tmp.name
            if download_fn(url, tmp_path):
                # 检查下载的文件是否有效（不是 HTML 错误页）
                if Path(tmp_path).stat().st_size > 100:  # 至少100字节
                    shutil.move(tmp_path, output_path)
                    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                    print(f"✓ ({size_mb:.1f} MB)")
                else:
                    Path(tmp_path).unlink(missing_ok=True)
                    print("(跳过，文件过小)")
            else:
                Path(tmp_path).unlink(missing_ok=True)
                print("(跳过)")

        # 检查是否有分片模型文件
        shard_patterns = [
            "model-00001-of-0000{}.safetensors",
            "model-00002-of-0000{}.safetensors",
            "model-00003-of-0000{}.safetensors",
            "pytorch_model-00001-of-0000{}.bin",
            "pytorch_model-00002-of-0000{}.bin",
        ]
        # 尝试下载分片（最多试4个分片）
        for pattern in shard_patterns:
            for n in range(1, 5):
                filename = pattern.format(n)
                output_path = str(model_dir / filename)
                if Path(output_path).exists():
                    continue
                url = f"{base_url}/{filename}"
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
                    tmp_path = tmp.name
                if download_fn(url, tmp_path) and Path(tmp_path).stat().st_size > 100:
                    shutil.move(tmp_path, output_path)
                    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                    print(f"    ⬇ {filename} ✓ ({size_mb:.1f} MB)")
                else:
                    Path(tmp_path).unlink(missing_ok=True)
                    break  # 没有更多分片

        print(f"    ✓ 下载完成")

    return True


# ═══════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════

def verify_models(pretrained_dir: Path) -> bool:
    """验证所有模型文件完整性"""
    print("\n" + "=" * 50)
    print("验证模型文件...")
    print("=" * 50)

    all_ok = True

    # 检查 GPT-SoVITS 文件
    for model_name, info in GPTSOVITS_FILES.items():
        model_dir = pretrained_dir / model_name
        print(f"\n  [{model_name}]")
        if not model_dir.exists():
            print("    ✗ 目录不存在!")
            all_ok = False
            continue
        for f in info["files"]:
            filepath = model_dir / f
            if filepath.exists() and filepath.stat().st_size > 0:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"    ✓ {f} ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ {f} 缺失!")
                all_ok = False

    # 检查 BERT / HuBERT 目录
    for model_name, info in SEPARATE_REPOS.items():
        model_dir = pretrained_dir / model_name
        print(f"\n  [{model_name}]")
        if not model_dir.exists():
            print("    ✗ 目录不存在!")
            all_ok = False
            continue
        file_count = len(list(model_dir.iterdir()))
        has_config = (model_dir / "config.json").exists()
        has_model = (
            (model_dir / "model.safetensors").exists() or
            (model_dir / "pytorch_model.bin").exists() or
            list(model_dir.glob("*.safetensors.index.*")) or
            list(model_dir.glob("pytorch_model*.bin"))
        )
        if has_config and has_model:
            print(f"    ✓ 模型完整 ({file_count} 个文件)")
        else:
            print(f"    ✗ 模型不完整 (有 {file_count} 个文件)")
            if not has_config:
                print("    ✗ config.json 缺失")
            if not has_model:
                print("    ✗ 模型权重文件缺失")
            all_ok = False

    return all_ok


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("GPT-SoVITS v2 预训练模型下载")
    print("=" * 50)

    pretrained_dir = get_pretrained_dir()
    print(f"\n模型目标目录: {pretrained_dir}")

    # 检查 GPT-SoVITS 是否已克隆
    gptsovits_root = pretrained_dir.parent.parent
    if not gptsovits_root.exists():
        print(f"\n[错误] GPT-SoVITS 目录不存在: {gptsovits_root}")
        print("请先运行: bash scripts/setup_gptsovits_env.sh")
        sys.exit(1)

    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # 设置 HF 镜像
    print("\n检查网络环境...")
    use_mirror = setup_hf_mirror()

    # 下载模型 - 依次尝试多种方式
    success = False

    # 方式1: huggingface_hub Python API (最推荐)
    print("\n" + "-" * 50)
    print("方式1: huggingface_hub Python API")
    print("-" * 50)
    try:
        success = download_method1_hf_hub(pretrained_dir)
    except Exception as e:
        print(f"  方式1异常: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # 方式2: huggingface-cli
    if not success:
        print("\n" + "-" * 50)
        print("方式2: huggingface-cli 命令行工具")
        print("-" * 50)
        try:
            success = download_method2_hf_cli(pretrained_dir)
        except Exception as e:
            print(f"  方式2异常: {e}")
            success = False

    # 方式3: wget/curl 直接下载
    if not success:
        print("\n" + "-" * 50)
        print("方式3: wget/curl 直接下载 (HF 镜像)")
        print("-" * 50)
        # 确保镜像设置
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = f"https://{HF_MIRROR}"
        try:
            success = download_method3_direct(pretrained_dir)
        except Exception as e:
            print(f"  方式3异常: {e}")
            success = False

    # 验证
    all_ok = verify_models(pretrained_dir)

    if all_ok:
        print("\n" + "=" * 50)
        print("所有模型下载完成！")
        print("")
        print("启动 GPT-SoVITS:")
        print("  bash scripts/start_gptsovits.sh")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("部分模型下载失败")
        print("")
        print("手动下载方法:")
        print(f"  1. GPT/SoVITS 模型:")
        print(f"     https://huggingface.co/{HF_REPO_GPTSOVITS}")
        print(f"     下载 GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/ 下的文件")
        if use_mirror:
            print(f"     国内镜像: https://{HF_MIRROR}/{HF_REPO_GPTSOVITS}")
        print(f"")
        print(f"  2. BERT 模型:")
        print(f"     https://huggingface.co/{HF_REPO_BERT}")
        print(f"     下载全部文件到 {pretrained_dir}/chinese-roberta-wwm-ext-large/")
        print(f"")
        print(f"  3. HuBERT 模型:")
        print(f"     https://huggingface.co/{HF_REPO_HUBERT}")
        print(f"     下载全部文件到 {pretrained_dir}/chinese-hubert-base/")
        if use_mirror:
            print(f"     国内镜像: https://{HF_MIRROR}/{HF_REPO_HUBERT}")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
