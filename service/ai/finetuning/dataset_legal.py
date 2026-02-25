# -*- coding: utf-8 -*-
"""
法律咨询数据集：从 dataset/【数据集】legal/qa_corpus.json 加载问答对，供法律 LoRA 微调用。
格式：JSONL，每行一个 JSON：{"question": "...", "answers": ["...", ...], "category": "..."}
"""
import os
import json
from pathlib import Path
from datasets import Dataset


def get_legal_qa_path():
    """法律 QA 数据路径（项目根下 dataset/【数据集】legal/qa_corpus.json）。"""
    root = Path(__file__).resolve().parents[3]
    return root / "dataset" / "【数据集】legal" / "qa_corpus.json"


def load_legal_data(
    json_path=None,
    max_answer_len=800,
    max_question_len=300,
    use_first_answer_only=True,
):
    """
    加载法律咨询 QA，返回 HuggingFace Dataset，每条为 {"instruction", "input", "output"}。

    :param json_path: qa_corpus.json 路径，默认项目下 dataset/【数据集】legal/qa_corpus.json
    :param max_question_len: 问题最大长度，过长跳过
    :param max_answer_len: 回答最大长度；若 use_first_answer_only=False 则多条回答拼接后截断
    :param use_first_answer_only: True 只取 answers[0]，False 则用 " ".join(answers)
    """
    path = json_path or get_legal_qa_path()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"法律 QA 文件不存在: {path}")

    data = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (item.get("question") or "").strip()
            answers = item.get("answers")
            if not q or not answers or not isinstance(answers, (list, tuple)):
                continue
            if use_first_answer_only:
                a = (answers[0] or "").strip()
            else:
                a = " ".join((str(x) or "").strip() for x in answers if x).strip()
            if not a or len(q) > max_question_len or len(a) > max_answer_len:
                continue
            data.append(
                {
                    "instruction": "请根据以下法律咨询问题给出专业、准确的回答。",
                    "input": q,
                    "output": a,
                }
            )
    if not data:
        raise ValueError("没有加载到任何法律 QA 数据")
    print(f"加载法律 QA 共 {len(data)} 条")
    return Dataset.from_list(data)
