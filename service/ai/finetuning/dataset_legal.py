# -*- coding: utf-8 -*-
"""
法律咨询数据集：从 dataset/【数据集】legal/ 加载数据，供法律 LoRA 微调用。
- qa_corpus.json：JSONL，每行 {"question", "answers", "category"}
- kg_crime.json：JSONL，每行罪名知识 {"crime_big", "crime_small", "gainian", "tezheng", "rending", "chufa", ...}
"""
import os
import json
from pathlib import Path
from datasets import Dataset


def get_legal_qa_path():
    """法律 QA 数据路径（项目根下 dataset/【数据集】legal/qa_corpus.json）。"""
    root = Path(__file__).resolve().parents[3]
    return root / "dataset" / "【数据集】legal" / "qa_corpus.json"


def get_legal_kg_crime_path():
    """罪名知识库路径（dataset/【数据集】legal/kg_crime.json）。"""
    root = Path(__file__).resolve().parents[3]
    return root / "dataset" / "【数据集】legal" / "kg_crime.json"


def _safe_join(texts, sep="\n", max_len=None):
    """将列表拼成字符串，可选截断。"""
    if not texts or not isinstance(texts, (list, tuple)):
        return ""
    parts = [str(x).strip() for x in texts if x]
    s = sep.join(parts).strip()
    if max_len and len(s) > max_len:
        s = s[:max_len].rsplit(sep, 1)[0] or s[:max_len]
    return s


def load_legal_kg_crime_data(
    json_path=None,
    max_answer_len=1200,
    max_question_len=200,
):
    """
    从 kg_crime.json 加载罪名知识，转为与 load_legal_data 相同格式的 QA 列表。
    每条罪名生成：问题「什么是{crime_small}？」或「请介绍{crime_small}（{crime_big}）」，回答为概念+特征+认定+处罚。
    """
    path = json_path or get_legal_kg_crime_path()
    path = Path(path)
    if not path.is_file():
        return []

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
            crime_big = (item.get("crime_big") or "").strip()
            crime_small = (item.get("crime_small") or "").strip()
            if not crime_small:
                continue
            gainian = _safe_join(item.get("gainian"), sep=" ", max_len=500)
            tezheng = _safe_join(item.get("tezheng"), sep=" ", max_len=400)
            rending = _safe_join(item.get("rending"), sep=" ", max_len=200)
            chufa = _safe_join(item.get("chufa"), sep=" ", max_len=200)
            parts = [p for p in [gainian, tezheng, rending, chufa] if p]
            answer = "\n\n".join(parts)
            if not answer or len(answer) > max_answer_len:
                answer = (answer or "")[:max_answer_len]
            if crime_big:
                question = f"请介绍{crime_small}（属于{crime_big}）"
            else:
                question = f"什么是{crime_small}？"
            if len(question) > max_question_len:
                question = question[:max_question_len]
            data.append(
                {
                    "instruction": "请根据以下法律咨询问题给出专业、准确的回答。",
                    "input": question,
                    "output": answer,
                }
            )
    if data:
        print(f"加载罪名知识 kg_crime 共 {len(data)} 条")
    return data


def load_legal_data(
    json_path=None,
    max_answer_len=800,
    max_question_len=300,
    use_first_answer_only=True,
    answer_choice="first",
    include_kg_crime=True,
    kg_crime_max_answer_len=1200,
):
    """
    加载法律咨询数据，返回 HuggingFace Dataset，每条为 {"instruction", "input", "output"}。
    可选合并 dataset/【数据集】legal/kg_crime.json 罪名知识。

    :param json_path: qa_corpus.json 路径，默认项目下 dataset/【数据集】legal/qa_corpus.json
    :param max_question_len: 问题最大长度，过长跳过
    :param max_answer_len: 回答最大长度；若 use_first_answer_only=False 则多条回答拼接后截断
    :param use_first_answer_only: True 只取一条回答（由 answer_choice 决定），False 则用 " ".join(answers)
    :param answer_choice: "first" 用 answers[0]，"longest" 用 answers 中最长的一条（更易学到完整表述）
    :param include_kg_crime: 是否同时加载 kg_crime.json 罪名知识并合并到训练集
    :param kg_crime_max_answer_len: 罪名知识条目的回答最大长度
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
                if answer_choice == "longest":
                    candidates = [(str(x) or "").strip() for x in answers if x]
                    a = max(candidates, key=len) if candidates else ""
                else:
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

    if include_kg_crime:
        kg_data = load_legal_kg_crime_data(
            max_answer_len=kg_crime_max_answer_len,
            max_question_len=max_question_len,
        )
        if kg_data:
            data = data + kg_data
            print(f"合并后总样本数: {len(data)} 条")

    return Dataset.from_list(data)
