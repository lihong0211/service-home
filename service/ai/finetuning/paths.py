"""
微调目录约定：lora/{日期}_{模型名}，该目录下直接保存 LoRA 适配器与 tokenizer，子目录 outputs_hf/ 存训练 checkpoint。
训练脚本与推理服务统一使用本模块解析路径。
"""

from datetime import datetime
from pathlib import Path

# 默认模型标识（get_run_parent_dir 未传 model_name 时使用）
RUN_MODEL_NAME = "Qwen2.5-7B-Instruct"


def get_finetuning_root() -> Path:
    """service/ai/finetuning 目录（与 paths.py 同目录）。"""
    return Path(__file__).resolve().parent


def get_project_root() -> Path:
    """项目根目录（service-home）：finetuning 上三级。"""
    return get_finetuning_root().parent.parent.parent


def get_run_parent_dir(
    finetuning_root: Path | None = None,
    model_name: str | None = None,
    date_str: str | None = None,
) -> Path:
    """本次运行的目录：项目根目录下的 lora/{日期}_{模型名}，例如 <project>/lora/20260224_Qwen2.5-7B-Instruct。"""
    project_root = get_project_root()
    name = model_name or RUN_MODEL_NAME
    date = date_str or datetime.now().strftime("%Y%m%d")
    return project_root / "lora" / f"{date}_{name}"


def get_lora_dir(parent: Path) -> Path:
    """LoRA 适配器保存目录，即本次运行目录本身（adapter 直接保存在该目录下）。"""
    return parent


def get_outputs_hf_dir(parent: Path) -> Path:
    """父目录下的 outputs_hf/，用于 SFTTrainer 的 output_dir。"""
    return parent / "outputs_hf"


def get_latest_lora_dir(
    finetuning_root: Path | None = None,
    model_name: str | None = None,
) -> Path | None:
    """取最新一次运行的 lora 目录（项目根/lora/{date}_{model} 按目录名排序）。不存在则返回 None。"""
    lora_base = get_project_root() / "lora"
    if not lora_base.is_dir():
        return None
    name = model_name or RUN_MODEL_NAME
    prefix = "_" + name
    candidates = [d for d in lora_base.iterdir() if d.is_dir() and d.name.endswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.name, reverse=True)
    return candidates[0]
