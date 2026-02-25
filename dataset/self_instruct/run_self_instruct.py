#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 dataset/self_instruct 目录下运行 Self-Instruct 生成。

用法（在项目根目录）:
  python -m dataset.self_instruct.run_self_instruct

或在 dataset/self_instruct 下:
  python run_self_instruct.py

环境变量:
  QWEN_API_KEY 或 OPENAI_API_KEY  必填（若用本地服务可配 OPENAI_BASE_URL）
  OPENAI_BASE_URL                 可选，本地/ Qwen 等兼容端点
"""
import os
import sys
from pathlib import Path

# 保证当前目录在 path 中，便于同目录 filters / seed 等
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 项目根目录，便于使用 dataset/self_instruct/seed_tasks.jsonl
PROJECT_ROOT = ROOT.parent.parent

os.chdir(ROOT)


def main():
    seed_path = ROOT / "seed_tasks.jsonl"
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "self_instruct_output.jsonl"

    try:
        from .self_instruct_pipeline import SelfInstructPipeline
    except ImportError:
        from self_instruct_pipeline import SelfInstructPipeline

    pipeline = SelfInstructPipeline(
        seed_path=str(seed_path),
        data_output_path=str(output_path),
        num_machine_instructions=int(os.environ.get("NUM_INSTRUCTIONS", "20")),
        human_to_machine_ratio=(5, 2),
        model=os.environ.get("SELF_INSTRUCT_MODEL", "gpt-3.5-turbo"),
        temperature=float(os.environ.get("SELF_INSTRUCT_TEMPERATURE", "0.7")),
        per_call=5,
    )
    out = pipeline.generate()
    print(f"生成完成，输出: {out}")
    return out


if __name__ == "__main__":
    main()
