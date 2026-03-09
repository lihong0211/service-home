#!/usr/bin/env python
# coding: utf-8
"""
法律咨询 LoRA 微调 - Qwen2.5-1.5B-Instruct + DISC-Law-SFT（RTX 5090）。

基座：Qwen2.5-1.5B-Instruct（指令对齐版，不用 Base）
数据：
  - dataset/【数据集】legal_hq/DISC-Law-SFT/DISC-Law-SFT-Pair-QA-released.jsonl   (79,692条)
  - dataset/【数据集】legal_hq/DISC-Law-SFT/DISC-Law-SFT-Triplet-QA-released.jsonl (23,331条)
格式：Qwen chat template（与 Instruct 训练格式一致），替换原来的 ### 问题/回答 裸格式
"""

import os
import sys
import json
from pathlib import Path

try:
    _bootstrap_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _bootstrap_dir = os.getcwd()
_bootstrap_root = os.path.dirname(os.path.dirname(os.path.dirname(_bootstrap_dir)))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

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
_LOCAL_MODEL = os.path.isdir(BASE_MODEL_PATH)

LEGAL_RUN_NAME = "Qwen2.5-1.5B-Instruct-legal-disc-5090"
_RUN_PARENT = get_run_parent_dir(get_finetuning_root(), model_name=LEGAL_RUN_NAME)
os.makedirs(_RUN_PARENT, exist_ok=True)
LORA_SAVE_DIR = str(get_lora_dir(_RUN_PARENT))
OUTPUT_DIR = str(get_outputs_hf_dir(_RUN_PARENT))
print(f"基座: {BASE_MODEL_PATH}")
print(f"LoRA 保存: {LORA_SAVE_DIR}")

# ── 设备 ──────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
    use_4bit = os.environ.get("USE_4BIT", "").strip() in ("1", "true", "yes")
    use_bf16 = torch.cuda.is_bf16_supported() and not use_4bit
else:
    device = (
        "mps"
        if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        else "cpu"
    )
    use_4bit = False
    use_bf16 = False
print(f"设备: {device}, 4bit: {use_4bit}, bf16: {use_bf16}")

max_seq_length = 2048
seed = 3407

# ── Tokenizer ─────────────────────────────────────────────────────────────────
# 部分 transformers 版本 AutoTokenizer 本地加载时有 '_config.model_type' AttributeError，
# 直接用 Qwen2TokenizerFast 跳过 Auto 派发层。
try:
    from transformers import Qwen2TokenizerFast
    tokenizer = Qwen2TokenizerFast.from_pretrained(BASE_MODEL_PATH)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── 模型 ──────────────────────────────────────────────────────────────────────
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
else:
    _dtype = torch.bfloat16 if (device == "cuda" and use_bf16) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # alpha=2r，对 Instruct 模型收敛更稳
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
model.print_trainable_parameters()

# ── System Prompt ──────────────────────────────────────────────────────────────
# 具体、有约束力：明确职责范围、要求引用法条、禁止越界给非法律建议
SYSTEM_PROMPT = (
    "你是一名专业法律顾问，专注于中国法律法规的咨询与解答。"
    "请根据用户描述的问题，给出具体、准确、可操作的法律建议，"
    "在适用时引用相关法律条文（如《民法典》《劳动合同法》等）并说明处理流程。"
    "回答应聚焦于法律层面；若问题超出法律范畴（如医疗、心理等），"
    "请明确告知并建议用户寻求对应专业人士帮助。"
)

# ── 数据加载 ──────────────────────────────────────────────────────────────────
_DATA_DIR = Path(_PROJECT_ROOT) / "dataset" / "【数据集】legal_hq" / "DISC-Law-SFT"
_PAIR_QA = _DATA_DIR / "DISC-Law-SFT-Pair-QA-released.jsonl"
_TRIPLET_QA = _DATA_DIR / "DISC-Law-SFT-Triplet-QA-released.jsonl"


def load_disc_law_sft(max_input_len=800, max_output_len=1000):
    """
    加载 DISC-Law-SFT 的 Pair-QA 和 Triplet-QA 两份数据。
    Triplet-QA 的 input 已经包含了法律条文参考，直接使用 input/output 两个字段即可。
    过滤：输入或输出过短（<15字）、输入过长的条目。
    """
    records = []
    for path, label in [(_PAIR_QA, "Pair-QA"), (_TRIPLET_QA, "Triplet-QA")]:
        if not path.is_file():
            print(f"[跳过] 文件不存在: {path}")
            continue
        count = 0
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
                if len(q) < 15 or len(a) < 15:
                    continue
                if len(q) > max_input_len or len(a) > max_output_len:
                    continue
                records.append({"question": q, "answer": a})
                count += 1
        print(f"加载 {label}: {count:,} 条")
    return records


MAX_TRAIN_SAMPLES = 10000  # 试跑；正式训练设为 None 用全量数据

import random

raw_data = load_disc_law_sft(max_input_len=800, max_output_len=1000)
if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < len(raw_data):
    random.seed(seed)
    raw_data = random.sample(raw_data, MAX_TRAIN_SAMPLES)
print(f"总计: {len(raw_data):,} 条")


# ── 格式化：Qwen chat template ─────────────────────────────────────────────────
# Instruct 模型用 apply_chat_template 格式化，与预训练格式完全一致，效果远优于裸 ### 格式
def format_sample(item):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": item["answer"]},
    ]
    # add_generation_prompt=False：训练时 assistant 的回答要包含在 text 里
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


texts = [format_sample(item) for item in raw_data]
dataset = Dataset.from_dict({"text": texts})
print(f"格式化后样本数: {len(dataset):,}")
print(f"样本示例（前500字）:\n{texts[0][:500]}\n{'─'*60}")

# ── 训练参数 ──────────────────────────────────────────────────────────────────
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=2,  # Instruct 基座收敛更快，2 epoch 即可
    learning_rate=1e-4,  # Instruct 上 lr 适当降低，避免过拟合
    warmup_ratio=0.03,
    bf16=(device == "cuda" and use_bf16),
    fp16=(device == "cuda" and not use_bf16),
    logging_steps=10,
    optim="adamw_8bit" if use_4bit else "adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",  # cosine 比 linear 对 Instruct 模型更友好
    seed=seed,
    report_to="none",
    save_strategy="no",
    gradient_checkpointing=(device == "cuda"),
    dataset_text_field="text",
    max_length=max_seq_length,
    packing=True,  # packing=True 对齐长度，提升 5090 显存利用率
    dataset_num_proc=4,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_args,
)

# ── 训练 ──────────────────────────────────────────────────────────────────────
print("开始训练法律 LoRA (DISC-Law-SFT + Instruct 基座)...")
trainer.train()
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA 已保存到: {LORA_SAVE_DIR}")


# ── 推理示例 ──────────────────────────────────────────────────────────────────
def generate_legal_response(question, model=None, tokenizer=None):
    m = model or trainer.model
    tok = tokenizer or getattr(trainer, "processing_class", None)
    m.eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tok(prompt, return_tensors="pt").to(m.device)
    enc = {k: v for k, v in enc.items() if k != "token_type_ids"}
    with torch.no_grad():
        out = m.generate(
            **enc,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen = out[0][enc["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


for q in [
    "还没有满十六岁的去打工，还没有一个月，不干了，老板不发工资，我应该怎么做？",
    "交通事故对方全责，保险公司拖着不赔怎么办？",
    "公司突然裁员，没有提前通知也没有赔偿，合法吗？",
]:
    print(f"\n问题: {q}")
    print(f"回答: {generate_legal_response(q)}")
