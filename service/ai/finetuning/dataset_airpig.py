# -*- coding: utf-8 -*-
"""
空气小猪数据集：从 dataset/【数据集】空气小猪/qa_train.json 加载问答对，供空气小猪 LoRA 微调用。
格式：JSON 数组，每项 {"instruction": "...", "input": "", "output": "..."}（Alpaca 风格）
"""
import json
from pathlib import Path
from datasets import Dataset


def get_airpig_qa_path():
    """空气小猪 QA 数据路径（项目根下 dataset/【数据集】空气小猪/qa_train.json）。"""
    root = Path(__file__).resolve().parents[3]
    return root / "dataset" / "【数据集】空气小猪" / "qa_train.json"


def load_airpig_data(json_path=None, max_answer_len=1200, max_question_len=500):
    """
    加载空气小猪客服/产品 QA，返回 HuggingFace Dataset，每条为 {"input", "output"}。
    input = instruction（或 instruction + "\\n" + input），output = 官方回答。
    """
    path = json_path or get_airpig_qa_path()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"空气小猪 QA 文件不存在: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raw = [raw]

    data = []
    for item in raw:
        inst = (item.get("instruction") or "").strip()
        inp = (item.get("input") or "").strip()
        out = (item.get("output") or "").strip()
        if not inst or not out:
            continue
        q = f"{inst}\n{inp}".strip() if inp else inst
        if len(q) > max_question_len or len(out) > max_answer_len:
            continue
        data.append({"input": q, "output": out})
    if not data:
        raise ValueError("没有加载到任何空气小猪 QA 数据")
    print(f"加载空气小猪 QA 共 {len(data)} 条")
    return Dataset.from_list(data)
