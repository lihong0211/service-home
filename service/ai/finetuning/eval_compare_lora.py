#!/usr/bin/env python
# coding: utf-8
"""
用训练集前 N 条做 LoRA 输出对比：每条跑一遍推理，对比「标准答案」与「LoRA 生成答案」，结果写入 finetuning 目录。

用法:
  python -m service.ai.finetuning.eval_compare_lora --type medical --top 10
  python -m service.ai.finetuning.eval_compare_lora --type legal --top 10
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    _bootstrap_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _bootstrap_dir = os.getcwd()
_bootstrap_root = os.path.dirname(os.path.dirname(os.path.dirname(_bootstrap_dir)))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from service.ai.finetuning.paths import (
    get_finetuning_root,
    get_project_root,
    get_latest_lora_dir,
    get_finetuning_root as _finetuning_root,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = get_project_root()
FINETUNING_DIR = get_finetuning_root()

# 医疗：数据路径 + 默认 LoRA 目录名
MEDICAL_DATA_ROOT = PROJECT_ROOT / "dataset" / "【数据集】medical_hq"
MEDICAL_HUATUO = MEDICAL_DATA_ROOT / "HuatuoGPT-sft-v1" / "HuatuoGPT_sft_data_v1.jsonl"
MEDICAL_ENCYCLOPEDIA = (
    MEDICAL_DATA_ROOT / "huatuo_encyclopedia_qa" / "train_datasets.jsonl"
)
MEDICAL_LORA_MODEL_NAME = "Qwen2.5-1.5B-Instruct-medical-huatuo-5090"
MEDICAL_SYSTEM_PROMPT = (
    "你是一名专业医疗健康助手，专注于常见病症、用药知识与健康管理咨询。"
    "回答时请：①简要分析可能的原因或病因；②给出具体、可操作的居家处理或用药建议；"
    "③说明何种情况下需要及时就医。"
    "重要原则：不做确定性诊断，不替代医生处方；"
    "若描述的症状提示急症（如胸痛、呼吸困难、意识障碍等），请优先建议立即拨打急救电话或前往急诊。"
)

# 法律：数据路径 + 默认 LoRA 目录名
LEGAL_DATA_ROOT = PROJECT_ROOT / "dataset" / "【数据集】legal_hq" / "DISC-Law-SFT"
LEGAL_PAIR_QA = LEGAL_DATA_ROOT / "DISC-Law-SFT-Pair-QA-released.jsonl"
LEGAL_TRIPLET_QA = LEGAL_DATA_ROOT / "DISC-Law-SFT-Triplet-QA-released.jsonl"
LEGAL_LORA_MODEL_NAME = "Qwen2.5-1.5B-Instruct-legal-disc-5090"
LEGAL_SYSTEM_PROMPT = (
    "你是一名专业法律顾问，专注于中国法律法规的咨询与解答。"
    "请根据用户描述的问题，给出具体、准确、可操作的法律建议，"
    "在适用时引用相关法律条文（如《民法典》《劳动合同法》等）并说明处理流程。"
    "回答应聚焦于法律层面；若问题超出法律范畴（如医疗、心理等），"
    "请明确告知并建议用户寻求对应专业人士帮助。"
)

BASE_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen" / "Qwen2.5-1.5B-Instruct"


def load_medical_samples(n: int):
    """医疗：HuatuoGPT-SFT + 百科，合并后取前 n 条。"""

    def _huatuo(path, max_q=600, max_a=800):
        if not path.is_file():
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turns = item.get("data") or []
                q = a = None
                for turn in turns:
                    t = turn.strip()
                    if t.startswith("问：") and q is None:
                        q = t[2:].strip()
                    elif t.startswith("答：") and q is not None and a is None:
                        a = t[2:].strip()
                        break
                if not q or not a or len(q) < 5 or len(a) < 10:
                    continue
                if len(q) > max_q or len(a) > max_a:
                    continue
                out.append({"question": q, "answer": a})
        return out

    def _encyclopedia(path, max_q=300, max_a=800):
        if not path.is_file():
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qs = item.get("questions") or []
                _ans = item.get("answers") or ""
                if isinstance(_ans, list) and _ans:
                    a = max(_ans, key=len).strip()
                else:
                    a = (_ans if isinstance(_ans, str) else "").strip()
                first = qs[0] if qs else None
                q = (
                    (first[0] if isinstance(first, list) else first).strip()
                    if first
                    else ""
                )
                if not q or not a or len(q) < 5 or len(a) < 10:
                    continue
                if len(q) > max_q or len(a) > max_a:
                    continue
                out.append({"question": q, "answer": a})
        return out

    data = _huatuo(MEDICAL_HUATUO) + _encyclopedia(MEDICAL_ENCYCLOPEDIA)
    return data[:n]


def load_legal_samples(n: int):
    """法律：Pair-QA + Triplet-QA，合并后取前 n 条。"""

    def _load(path, max_q=800, max_a=1000):
        if not path.is_file():
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = (item.get("input") or "").strip()
                a = (item.get("output") or "").strip()
                if len(q) < 15 or len(a) < 15 or len(q) > max_q or len(a) > max_a:
                    continue
                out.append({"question": q, "answer": a})
        return out

    data = _load(LEGAL_PAIR_QA) + _load(LEGAL_TRIPLET_QA)
    return data[:n]


def load_model_and_tokenizer(lora_type: str, lora_dir_override: str | None = None):
    """加载基座 + LoRA，返回 (model, tokenizer)。lora_dir_override 优先（--lora-dir 或 环境变量 LORA_PATH）。"""
    base = str(BASE_MODEL_PATH)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"基座不存在: {base}")

    if lora_dir_override and os.path.isdir(lora_dir_override):
        lora_path = os.path.abspath(lora_dir_override)
    else:
        lora_base = PROJECT_ROOT / "lora"
        # 优先使用 lora/medical、lora/legal 目录（与 finetuning 推理一致）
        if lora_type == "medical" and (lora_base / "medical").is_dir():
            lora_path = str(lora_base / "medical")
        elif lora_type == "legal" and (lora_base / "legal").is_dir():
            lora_path = str(lora_base / "legal")
        else:
            model_name = (
                MEDICAL_LORA_MODEL_NAME
                if lora_type == "medical"
                else LEGAL_LORA_MODEL_NAME
            )
            lora_dir = get_latest_lora_dir(_finetuning_root(), model_name=model_name)
            if not lora_dir or not lora_dir.is_dir():
                hint = f"请将 LoRA 放到 {lora_base}/medical 或 {lora_base}/legal，或使用 --lora-dir 指定。"
                raise FileNotFoundError(f"未找到 {lora_type} LoRA 目录。{hint}")
            lora_path = str(lora_dir)

    # 从基座加载 tokenizer，避免 LoRA 目录里 Unsloth 保存的 tokenizer 与当前 transformers 不兼容
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    if not torch.cuda.is_available():
        model = model.to("cpu")
    return model, tokenizer


def generate(
    model, tokenizer, question: str, system_prompt: str, max_new_tokens: int = 1000
):
    """单条推理，与 medical_5090 / legal_5090 一致：chat template + generate。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "token_type_ids"}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 训练集前 N 条对比：标准答案 vs LoRA 生成"
    )
    parser.add_argument(
        "--type",
        choices=("medical", "legal"),
        default="medical",
        help="medical 或 legal",
    )
    parser.add_argument("--top", type=int, default=10, help="取训练集前几条")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出文件路径，默认 finetuning/eval_compare_{type}_{date}.md",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=None,
        help="LoRA 目录路径（不指定则用 lora/ 下最新同名目录）",
    )
    args = parser.parse_args()

    lora_dir_override = args.lora_dir or os.environ.get("LORA_PATH")

    lora_type = args.type
    top_n = args.top
    system_prompt = (
        MEDICAL_SYSTEM_PROMPT if lora_type == "medical" else LEGAL_SYSTEM_PROMPT
    )

    if lora_type == "medical":
        samples = load_medical_samples(top_n)
    else:
        samples = load_legal_samples(top_n)

    if not samples:
        print(f"未加载到任何 {lora_type} 样本，请检查数据路径")
        sys.exit(1)
    print(f"加载 {len(samples)} 条样本，开始加载模型并推理...")

    model, tokenizer = load_model_and_tokenizer(
        lora_type, lora_dir_override=lora_dir_override
    )

    results = []
    for i, s in enumerate(samples):
        q = s["question"]
        ref = s["answer"]
        pred = generate(model, tokenizer, q, system_prompt)
        results.append(
            {"idx": i + 1, "question": q, "reference": ref, "lora_answer": pred}
        )
        print(f"  [{i+1}/{len(samples)}] 完成")

    # 写入 finetuning 目录
    out_dir = Path(FINETUNING_DIR)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = out_dir / out_path
    else:
        out_path = (
            out_dir
            / f"eval_compare_{lora_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# LoRA 对比评估：{lora_type}（训练集前 {top_n} 条）\n\n")
        f.write(f"生成时间：{datetime.now().isoformat()}\n\n---\n\n")
        for r in results:
            f.write(f"## 第 {r['idx']} 条\n\n")
            f.write(f"**问题**：\n{r['question']}\n\n")
            f.write(f"**标准答案**：\n{r['reference']}\n\n")
            f.write(f"**LoRA 生成**：\n{r['lora_answer']}\n\n---\n\n")

    print(f"对比结果已保存: {out_path}")


if __name__ == "__main__":
    main()
