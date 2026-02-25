#!/usr/bin/env python3
"""
下载 ModelScope 四川方言语音数据集（799 小时手机采集对话语音）。
搞个锤子，只能下10条样板数据
"""
import argparse
import sys
from pathlib import Path

# 允许从项目根或 scripts 目录执行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="下载四川方言对话语音数据集 (799Hours)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="可选：将数据集复制/保存到的本地目录（默认使用 ModelScope 缓存）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="仅加载到内存/缓存，不额外导出到 --out-dir",
    )
    args = parser.parse_args()

    from modelscope.msdatasets import MsDataset

    dataset_name = (
        "DatatangBeijing/799Hours-SichuanDialectConversationalSpeechDataByMobilePhone"
    )
    subset_name = "default"
    split = "train"

    print(f"Loading dataset: {dataset_name} (subset={subset_name}, split={split}) ...")
    ds = MsDataset.load(
        dataset_name,
        subset_name=subset_name,
        split=split,
    )
    print("Dataset loaded (cached or downloaded).")

    if args.out_dir and not args.no_save:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if hasattr(ds, "save_to_disk"):
            print(f"Saving to {out_path} ...")
            ds.save_to_disk(str(out_path))
            print(f"Done. Saved to {out_path}")
        else:
            print("Dataset does not support save_to_disk; data is in ModelScope cache.")
    else:
        n = 0
        for _ in ds:
            n += 1
            if n >= 3:
                break
        print(f"Dataset ready (sampled {n} items). Use --out-dir to save to disk.")


if __name__ == "__main__":
    main()
