#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Hugging Face 下载 Qwen2.5-7B-Instruct 到本地 models/Qwen/Qwen2.5-7B-Instruct。

依赖: pip install huggingface_hub

可选:
  - 国内网络可设置镜像: HF_ENDPOINT=https://hf-mirror.com python scripts/download_qwen25_7b_instruct.py
  - 若模型需登录: huggingface-cli login 或设置 HF_TOKEN
"""
import os
import sys
from pathlib import Path

# 国内镜像（若直连失败可取消注释或设置 HF_ENDPOINT=https://hf-mirror.com）
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
SAVE_DIR = PROJECT_ROOT / "models" / "Qwen" / "Qwen2.5-7B-Instruct"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        sys.exit(1)

    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print(f"使用镜像: {endpoint}")
    print(f"正在下载 {MODEL_REPO} 到 {SAVE_DIR} ...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=str(SAVE_DIR),
            local_dir_use_symlinks=False,
            local_files_only=False,
        )
        print(f"下载完成: {path}")
    except Exception as e:
        print(f"下载失败: {e}")
        if not endpoint:
            print("国内网络可尝试: HF_ENDPOINT=https://hf-mirror.com python scripts/download_qwen25_7b_instruct.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
