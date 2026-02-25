#!/usr/bin/env python
# coding: utf-8
"""
将某次运行目录下误生成在「父目录根」的 LoRA/Tokenizer 文件移入该目录下的 lora/ 文件夹。
若已存在 lora/ 且已有内容则跳过。
用法：python -m service.ai.finetuning.move_run_output_to_lora [运行目录名]
例如：python -m service.ai.finetuning.move_run_output_to_lora 20260224_Qwen2.5-7B-Instruct
"""

import sys
from pathlib import Path

# LoRA/Tokenizer 常见文件名（在父目录根时需移入 lora/）
LORA_FILE_NAMES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
}


def move_run_output_to_lora(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    lora_dir = run_dir / "lora"
    if not run_dir.is_dir():
        print(f"目录不存在: {run_dir}")
        return
    lora_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for name in LORA_FILE_NAMES:
        src = run_dir / name
        if not src.is_file():
            continue
        dst = lora_dir / name
        if dst.exists():
            print(f"已存在，跳过: {dst}")
            continue
        src.rename(dst)
        print(f"已移动: {src.name} -> lora/")
        moved += 1
    if moved == 0:
        print("无需移动（父目录下无待移动的 LoRA/Tokenizer 文件，或 lora/ 内已存在）。")
    else:
        print(f"共移动 {moved} 个文件到 {lora_dir}")


def main():
    root = Path(__file__).resolve().parent
    if len(sys.argv) >= 2:
        run_name = sys.argv[1].strip()
        run_dir = root / run_name
    else:
        # 未传参：找最新一个「日期_模型名」目录
        from service.ai.finetuning.paths import RUN_MODEL_NAME
        prefix = "_" + RUN_MODEL_NAME
        candidates = [d for d in root.iterdir() if d.is_dir() and d.name.endswith(prefix)]
        if not candidates:
            print("未找到任何运行目录（格式：YYYYMMDD_Qwen2.5-7B-Instruct）")
            print("用法: python -m service.ai.finetuning.move_run_output_to_lora 20260224_Qwen2.5-7B-Instruct")
            return
        candidates.sort(key=lambda d: d.name, reverse=True)
        run_dir = candidates[0]
        print(f"使用最新运行目录: {run_dir.name}")
    move_run_output_to_lora(run_dir)


if __name__ == "__main__":
    main()
