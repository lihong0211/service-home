#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Hugging Face 下载 Qwen2.5-1.5B（Base，非 Instruct）到本地 models/Qwen/Qwen2.5-1.5B。

Base 仅做因果语言模型预训练，未做指令/对话微调，几乎没专门学过「法律问答」格式，
适合作为「纯基座」做法律 LoRA 微调，便于观察微调数据的真实贡献。

依赖: pip install huggingface_hub

国内优先：默认使用 HF 国内镜像。取消镜像：
  HF_ENDPOINT= python scripts/download_qwen25_1.5b_base.py
"""
import os
import sys
from pathlib import Path

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "Qwen/Qwen2.5-1.5B"
SAVE_DIR = PROJECT_ROOT / "models" / "Qwen" / "Qwen2.5-1.5B"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        sys.exit(1)

    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print(f"使用镜像: {endpoint}")
    print(f"正在下载 {MODEL_REPO}（Base，非 Instruct）到 {SAVE_DIR} ...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=str(SAVE_DIR),
            local_files_only=False,
        )
        print(f"下载完成: {path}")
    except Exception as e:
        print(f"下载失败: {e}")
        if not endpoint:
            print("国内网络可尝试: HF_ENDPOINT=https://hf-mirror.com python scripts/download_qwen25_1.5b_base.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
