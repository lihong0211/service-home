#!/usr/bin/env python
# coding: utf-8
"""
医疗问答 LoRA 微调 - Qwen2.5-1.5B-Instruct + HuatuoGPT 高质量数据集（RTX 5090）。
结构对齐 legal_5090：Unsloth + flash_attn + Qwen chat template。

基座：Qwen2.5-1.5B-Instruct
数据：
  - dataset/【数据集】medical_hq/HuatuoGPT-sft-v1/HuatuoGPT_sft_data_v1.jsonl  (226,042条，问答多轮)
  - dataset/【数据集】medical_hq/huatuo_encyclopedia_qa/train_datasets.jsonl     (362,420条，医学百科)
  (huatuo_consultation_qa answers 是 URL，跳过)
"""

import os
import sys
import json
import random
from pathlib import Path

try:
    _bootstrap_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _bootstrap_dir = os.getcwd()
_bootstrap_root = os.path.dirname(os.path.dirname(os.path.dirname(_bootstrap_dir)))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)


# ── Patch：修复 transformers tokenizer bug ─────────────────────────────────────
def _fix_tokenizer_dict_bug():
    try:
        import importlib.util
        spec = importlib.util.find_spec("transformers.tokenization_utils_base")
        if not spec or not spec.origin:
            return
        with open(spec.origin, encoding="utf-8") as f:
            src = f.read()
        old = "if _is_local and _config.model_type not in ["
        new = 'if _is_local and getattr(_config, "model_type", None) not in ['
        if old in src:
            with open(spec.origin, "w", encoding="utf-8") as f:
                f.write(src.replace(old, new, 1))
            print("[startup] transformers tokenizer dict bug 已修复")
        else:
            print("[startup] transformers tokenizer bug 不存在或已修复，跳过")
    except Exception as e:
        print(f"[startup] tokenizer patch 失败: {e}")


_fix_tokenizer_dict_bug()


def _fix_sft_trainer_eos_check():
    """SFTTrainer eos_token 词表校验 → pass（Qwen2+unsloth 兼容）。"""
    try:
        import importlib.util
        spec = importlib.util.find_spec("trl.trainer.sft_trainer")
        if not spec or not spec.origin:
            return
        with open(spec.origin, encoding="utf-8") as f:
            src = f.read()
        marker = "Ensure that the `eos_token` exists in the vocabulary"
        if marker not in src:
            print("[startup] SFTTrainer eos_token check 不存在或已修复，跳过")
            return
        idx = src.find(marker)
        start = src.rfind("raise ValueError(", 0, idx)
        if start == -1:
            return
        depth, end = 0, start
        for i, ch in enumerate(src[start:]):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break
        with open(spec.origin, "w", encoding="utf-8") as f:
            f.write(src.replace(src[start:end], "pass  # patched for Qwen2+unsloth", 1))
        print("[startup] SFTTrainer eos_token check 已跳过")
    except Exception as e:
        print(f"[startup] SFTTrainer patch 失败: {e}")


_fix_sft_trainer_eos_check()

import torch
from datasets import Dataset

import unsloth.models._utils as _unsloth_utils
_unsloth_utils.snapshot_download = lambda *a, **kw: None

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from service.ai.finetuning.paths import (
    get_finetuning_root,
    get_run_parent_dir,
    get_lora_dir,
    get_outputs_hf_dir,
)

# ── 路径 ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen2.5-1.5B-Instruct")
if not os.path.isdir(BASE_MODEL_PATH):
    BASE_MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "Qwen", "Qwen2.5-1.5B-Instruct")

_DATA_ROOT = Path(_PROJECT_ROOT) / "dataset" / "【数据集】medical_hq"
_HUATUO_SFT   = _DATA_ROOT / "HuatuoGPT-sft-v1" / "HuatuoGPT_sft_data_v1.jsonl"
_ENCYCLOPEDIA = _DATA_ROOT / "huatuo_encyclopedia_qa" / "train_datasets.jsonl"

MEDICAL_RUN_NAME = "Qwen2.5-1.5B-Instruct-medical-huatuo-5090"
_RUN_PARENT = get_run_parent_dir(get_finetuning_root(), model_name=MEDICAL_RUN_NAME)
os.makedirs(_RUN_PARENT, exist_ok=True)
LORA_SAVE_DIR = str(get_lora_dir(_RUN_PARENT))
OUTPUT_DIR = str(get_outputs_hf_dir(_RUN_PARENT))
print(f"基座: {BASE_MODEL_PATH}")
print(f"LoRA 保存: {LORA_SAVE_DIR}")

max_seq_length = 1024
seed = 3407

# ── Unsloth：模型 + Tokenizer ──────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=False,
    local_files_only=True,
)
print(f"设备: {next(model.parameters()).device}")

tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

# ── LoRA ──────────────────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=seed,
    use_rslora=False,
)
model.print_trainable_parameters()

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一名专业医疗健康助手，专注于常见病症、用药知识与健康管理咨询。"
    "回答时请：①简要分析可能的原因或病因；②给出具体、可操作的居家处理或用药建议；"
    "③说明何种情况下需要及时就医。"
    "重要原则：不做确定性诊断，不替代医生处方；"
    "若描述的症状提示急症（如胸痛、呼吸困难、意识障碍等），请优先建议立即拨打急救电话或前往急诊。"
)

# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_huatuo_sft(path, max_q=600, max_a=800):
    """
    data 字段格式：["问：xxx\n", "答：xxx\n", ...]
    取每条对话第一对问答。
    """
    if not path.is_file():
        print(f"[跳过] 不存在: {path}")
        return []
    records, skip = [], 0
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
            # 找第一对 问/答
            q = a = None
            for turn in turns:
                t = turn.strip()
                if t.startswith("问：") and q is None:
                    q = t[2:].strip()
                elif t.startswith("答：") and q is not None and a is None:
                    a = t[2:].strip()
                    break
            if not q or not a:
                skip += 1
                continue
            if len(q) > max_q or len(a) > max_a:
                skip += 1
                continue
            if len(q) < 5 or len(a) < 10:
                skip += 1
                continue
            records.append({"question": q, "answer": a})
    print(f"加载 HuatuoGPT-SFT: {len(records):,} 条（跳过 {skip:,}）")
    return records


def load_encyclopedia(path, max_q=300, max_a=800):
    """
    questions: list of list，取第一个问题
    answers: 字符串
    """
    if not path.is_file():
        print(f"[跳过] 不存在: {path}")
        return []
    records, skip = [], 0
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
            if isinstance(_ans, list):
                a = max(_ans, key=len).strip() if _ans else ""
            else:
                a = _ans.strip()
            # questions 是 list of list 或 list of str
            q = ""
            if qs:
                first = qs[0]
                q = (first[0] if isinstance(first, list) else first).strip()
            if not q or not a:
                skip += 1
                continue
            if len(q) > max_q or len(a) > max_a:
                skip += 1
                continue
            if len(q) < 5 or len(a) < 10:
                skip += 1
                continue
            records.append({"question": q, "answer": a})
    print(f"加载 encyclopedia: {len(records):,} 条（跳过 {skip:,}）")
    return records


MAX_TRAIN_SAMPLES = 10000   # 试跑；全量设为 None

raw_data = load_huatuo_sft(_HUATUO_SFT) + load_encyclopedia(_ENCYCLOPEDIA)
if not raw_data:
    raise ValueError("没有加载到任何医疗数据，请检查路径")
print(f"合并后: {len(raw_data):,} 条")

if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < len(raw_data):
    random.seed(seed)
    raw_data = random.sample(raw_data, MAX_TRAIN_SAMPLES)
print(f"训练用: {len(raw_data):,} 条")

# ── 格式化：Qwen chat template ─────────────────────────────────────────────────
def format_sample(item):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": item["question"]},
        {"role": "assistant", "content": item["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


texts = [format_sample(item) for item in raw_data]
dataset = Dataset.from_dict({"text": texts})
print(f"格式化后样本数: {len(dataset):,}")
if texts:
    print(f"样本示例（前400字）:\n{texts[0][:400]}\n{'─'*60}")

# ── 训练参数 ──────────────────────────────────────────────────────────────────
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    bf16=True,
    fp16=False,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=seed,
    report_to="none",
    save_strategy="no",
    dataset_text_field="text",
    max_length=max_seq_length,
    packing=False,
    dataset_num_proc=4,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_args,
)

# ── 训练 ──────────────────────────────────────────────────────────────────────
print("开始训练医疗 LoRA (Unsloth + HuatuoGPT)...")
trainer.train()
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA 已保存到: {LORA_SAVE_DIR}")


# ── 推理示例 ──────────────────────────────────────────────────────────────────
def generate_medical_response(question, model=None, tokenizer=None):
    m = model or trainer.model
    tok = tokenizer or getattr(trainer, "processing_class", None)
    FastLanguageModel.for_inference(m)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(
            **enc,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


for q in ["我最近总是感觉头晕，应该怎么办？", "感冒发烧应该吃什么药？"]:
    print(f"\n问题: {q}")
    print(f"回答: {generate_medical_response(q)}")
