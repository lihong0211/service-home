#!/usr/bin/env python
# coding: utf-8
"""
一次性迁移：将已有的 lora_model_medical_hf 与 outputs_hf 移入新结构「日期+模型名/lora」与「日期+模型名/outputs_hf」。
运行一次即可，之后训练脚本会直接写入新目录。
"""

import os
import shutil
from pathlib import Path

from service.ai.finetuning.paths import (
    get_finetuning_root,
    get_run_parent_dir,
    get_lora_dir,
    get_outputs_hf_dir,
    RUN_MODEL_NAME,
)


def main():
    root = get_finetuning_root()
    old_lora = root / "lora_model_medical_hf"
    old_outputs = root / "outputs_hf"

    # 使用今天日期作为迁移后的父目录名（若希望用历史日期可改这里）
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    parent = get_run_parent_dir(root, model_name=RUN_MODEL_NAME, date_str=date_str)
    new_lora = get_lora_dir(parent)
    new_outputs = get_outputs_hf_dir(parent)

    if not old_lora.is_dir() and not old_outputs.is_dir():
        print("未发现 lora_model_medical_hf 或 outputs_hf，无需迁移。")
        return

    parent.mkdir(parents=True, exist_ok=True)

    if old_lora.is_dir():
        if new_lora.exists():
            print(f"目标已存在: {new_lora}，跳过 lora 迁移。")
        else:
            shutil.copytree(old_lora, new_lora)
            print(f"已复制: {old_lora} -> {new_lora}")
    else:
        print(f"不存在: {old_lora}")

    if old_outputs.is_dir():
        if new_outputs.exists():
            print(f"目标已存在: {new_outputs}，跳过 outputs_hf 迁移。")
        else:
            shutil.copytree(old_outputs, new_outputs)
            print(f"已复制: {old_outputs} -> {new_outputs}")
    else:
        print(f"不存在: {old_outputs}")

    print(f"迁移完成。父目录: {parent}")
    print("推理服务将自动使用最新一次运行目录下的 lora/。如需删除旧目录请手动执行。")


if __name__ == "__main__":
    main()
