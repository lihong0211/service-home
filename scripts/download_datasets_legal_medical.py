#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载高质量法律和医疗数据集（用于 LoRA 微调）。

法律：ShengbinYue/DISC-Law-SFT（复旦大学，专为法律 LLM 设计的 SFT 数据）
医疗：
  - FreedomIntelligence/HuatuoGPT-sft-data-v1（华驼，SFT 格式，干净）
  - FreedomIntelligence/huatuo_encyclopedia_qa（医学百科问答）
  - FreedomIntelligence/huatuo_consultation_qa（在线问诊对话）

用法：
  python scripts/download_datasets_legal_medical.py
  python scripts/download_datasets_legal_medical.py --only legal
  python scripts/download_datasets_legal_medical.py --only medical
  HF_ENDPOINT= python scripts/...  （关闭国内镜像，直连 HF）
"""
import os
import sys
import argparse
from pathlib import Path

# 默认走国内镜像；可通过环境变量覆盖
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

LEGAL_DATASETS = [
    {
        "repo_id": "ShengbinYue/DISC-Law-SFT",
        "local_dir": DATASET_ROOT / "【数据集】legal_hq" / "DISC-Law-SFT",
        "desc": "复旦 DISC-Law-SFT（法律指令微调数据，含判决/咨询/考试/推理四类）",
    },
]

MEDICAL_DATASETS = [
    {
        "repo_id": "FreedomIntelligence/HuatuoGPT-sft-data-v1",
        "local_dir": DATASET_ROOT / "【数据集】medical_hq" / "HuatuoGPT-sft-v1",
        "desc": "华驼 SFT v1（单轮医疗指令数据，格式干净，无垃圾模板）",
    },
    {
        "repo_id": "FreedomIntelligence/huatuo_encyclopedia_qa",
        "local_dir": DATASET_ROOT / "【数据集】medical_hq" / "huatuo_encyclopedia_qa",
        "desc": "华驼医学百科问答（权威医学知识，回答质量高）",
    },
    {
        "repo_id": "FreedomIntelligence/huatuo_consultation_qa",
        "local_dir": DATASET_ROOT / "【数据集】medical_hq" / "huatuo_consultation_qa",
        "desc": "华驼在线问诊对话（真实医患问答，多轮对话）",
    },
]


def download_one(repo_id: str, local_dir: Path, desc: str):
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT", "")
    print(f"\n{'─' * 60}")
    print(f"下载: {repo_id}")
    print(f"说明: {desc}")
    print(f"保存: {local_dir}")
    if endpoint:
        print(f"镜像: {endpoint}")
    print(f"{'─' * 60}")

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_files_only=False,
        )
        print(f"✓ 完成: {path}")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        print("  提示: 若网络超时，可重跑此脚本（会断点续传）")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["legal", "medical"],
        default=None,
        help="只下载法律或医疗，不传则两者都下载",
    )
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        sys.exit(1)

    targets = []
    if args.only in (None, "legal"):
        targets.extend(LEGAL_DATASETS)
    if args.only in (None, "medical"):
        targets.extend(MEDICAL_DATASETS)

    results = []
    for t in targets:
        ok = download_one(t["repo_id"], t["local_dir"], t["desc"])
        results.append((t["repo_id"], ok))

    print(f"\n{'=' * 60}")
    print("下载结果汇总:")
    for repo_id, ok in results:
        status = "✓ 成功" if ok else "✗ 失败"
        print(f"  {status}  {repo_id}")

    failed = [r for r, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} 个失败，可重跑脚本断点续传，或检查网络。")
        sys.exit(1)
    else:
        print("\n全部下载完成。")
        print(f"法律数据: {DATASET_ROOT / '【数据集】legal_hq'}")
        print(f"医疗数据: {DATASET_ROOT / '【数据集】medical_hq'}")


if __name__ == "__main__":
    main()
