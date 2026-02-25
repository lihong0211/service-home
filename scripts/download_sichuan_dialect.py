#!/usr/bin/env python3
"""
下载 ModelScope 四川方言语音数据集（799 小时手机采集对话语音）。
使用前请安装：pip install modelscope
若需登录/鉴权，请先：modelscope login --token YOUR_TOKEN

说明：ModelScope 上该数据集当前仅开放约 10 条试听样本；完整 799 小时需向数据堂申请：
  https://datatang.com 或 ModelScope 数据集页面的「申请」/联系方式。
"""
import argparse
import sys
from pathlib import Path

# 允许从项目根或 scripts 目录执行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="下载四川方言对话语音数据集 (799Hours)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="将数据集保存到的本地目录（默认：项目根目录下的 dataset）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="仅加载到内存/缓存，不额外导出到 --out-dir",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="遍历全部样本以触发完整音频下载（数据量大，时间较长）",
    )
    args = parser.parse_args()

    # 兼容 modelscope 在路径拼接时遇到 NaN/float 的 bug（dataset_builder 中 os.path.join 收到 float 会报错）
    import os.path as _os_path

    _orig_join = _os_path.join

    def _safe_join(a, *p):
        p = [str(x) if isinstance(x, float) else x for x in p]
        return _orig_join(a, *p)

    _os_path.join = _safe_join

    from modelscope.msdatasets import MsDataset

    dataset_name = "DatatangBeijing/799Hours-SichuanDialectConversationalSpeechDataByMobilePhone"
    subset_name = "default"

    # 默认下载到项目根目录下的 dataset
    out_dir = args.out_dir
    if out_dir is None and not args.no_save:
        out_dir = str(ROOT / "dataset")
    if out_dir is None and args.download_all:
        out_dir = str(ROOT / "dataset")

    # 尝试加载所有 split（train/validation/test）以获取完整数据；若某 split 不存在会跳过
    out_path = Path(out_dir) if out_dir else None
    if out_path and not args.no_save:
        out_path.mkdir(parents=True, exist_ok=True)

    all_splits = ["train", "validation", "test"]
    total_copied = 0

    for split in all_splits:
        try:
            print(f"Loading dataset: {dataset_name} (subset={subset_name}, split={split}) ...")
            ds = MsDataset.load(
                dataset_name,
                subset_name=subset_name,
                split=split,
            )
        except Exception as e:
            print(f"  Skip split '{split}': {e}")
            continue
        try:
            n_total = len(ds)
            print(f"  Split '{split}' has {n_total} samples.")
        except Exception:
            n_total = None

        if out_path and not args.no_save:
            if hasattr(ds, "save_to_disk") and total_copied == 0 and split == "train":
                print(f"Saving to {out_path} ...")
                ds.save_to_disk(str(out_path))
                print(f"Done. Saved to {out_path}")
                total_copied = n_total or 0
                break
            # 遍历并复制音频到 dataset/audio
            import shutil
            (out_path / "audio").mkdir(parents=True, exist_ok=True)
            for i, item in enumerate(ds):
                if isinstance(item, dict):
                    for k, v in item.items():
                        if v is None:
                            continue
                        src = None
                        if hasattr(v, "path"):
                            src = getattr(v, "path", None)
                        elif isinstance(v, dict) and "path" in v:
                            src = v["path"]
                        elif isinstance(v, (str, Path)) and Path(str(v)).exists():
                            src = str(v)
                        if src and Path(src).exists():
                            ext = Path(src).suffix or ".wav"
                            out_name = f"{total_copied:06d}_{k}{ext}"
                            shutil.copy2(src, out_path / "audio" / out_name)
                total_copied += 1
                if total_copied % 100 == 0:
                    print(f"  copied {total_copied} samples to {out_path} ...")
        else:
            if args.download_all:
                for i, item in enumerate(ds):
                    if isinstance(item, dict):
                        for v in item.values():
                            _ = v
                    total_copied += 1
                    if (i + 1) % 100 == 0:
                        print(f"  downloaded {i + 1} (split {split}) ...")
            elif split == "train":
                n = 0
                for _ in ds:
                    n += 1
                    if n >= 3:
                        break
                print(f"Dataset ready (sampled {n} items). Use --out-dir or --download-all to save/download all.")
                return

    if out_path and not args.no_save:
        print(f"Done. Total copied {total_copied} samples to {out_path}")
        if total_copied <= 20:
            print("Tip: ModelScope 当前仅开放少量试听样本。完整 799 小时需在数据堂/ModelScope 页面申请：https://modelscope.cn/datasets/DatatangBeijing/799Hours-SichuanDialectConversationalSpeechDataByMobilePhone")
    elif args.download_all:
        print(f"Finished downloading all samples: {total_copied}")


if __name__ == "__main__":
    main()
