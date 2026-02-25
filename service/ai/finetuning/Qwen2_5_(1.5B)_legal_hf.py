#!/usr/bin/env python
# coding: utf-8
"""
Qwen2.5-1.5B 法律咨询微调 - 使用 CrimeKgAssistant/data/qa_corpus.json 训练 LoRA。
数据来源：dataset/CrimeKgAssitant/data/qa_corpus.json（JSONL，question + answers + category）
"""

import os
import sys

try:
    _bootstrap_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _bootstrap_dir = os.getcwd()
_bootstrap_root = os.path.dirname(os.path.dirname(os.path.dirname(_bootstrap_dir)))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from service.ai.finetuning.dataset_legal import load_legal_data
from service.ai.finetuning.paths import (
    get_finetuning_root,
    get_run_parent_dir,
    get_lora_dir,
    get_outputs_hf_dir,
)

# ---------- 路径配置 ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen2.5-1.5B-Instruct")
if not os.path.isdir(BASE_MODEL_PATH):
    BASE_MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "Qwen", "Qwen2.5-1.5B-Instruct")
_LOCAL_MODEL = os.path.isdir(BASE_MODEL_PATH)

# 法律 LoRA 单独目录，不与医疗混用：lora/{日期}_Qwen2.5-1.5B-Instruct-legal
LEGAL_RUN_NAME = "Qwen2.5-1.5B-Instruct-legal"
_RUN_PARENT = get_run_parent_dir(get_finetuning_root(), model_name=LEGAL_RUN_NAME)
os.makedirs(_RUN_PARENT, exist_ok=True)
LORA_SAVE_DIR = str(get_lora_dir(_RUN_PARENT))
OUTPUT_DIR = str(get_outputs_hf_dir(_RUN_PARENT))
print(f"本次运行目录: {_RUN_PARENT} -> lora: {LORA_SAVE_DIR}, outputs_hf: {OUTPUT_DIR}")

# ---------- 设备与精度 ----------
if torch.cuda.is_available():
    device = "cuda"
    use_4bit = True
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    use_4bit = False
else:
    device = "cpu"
    use_4bit = False
print(f"使用设备: {device}, 4bit 量化: {use_4bit}")

max_seq_length = 2048
seed = 3407

# ---------- 加载 tokenizer ----------
if _LOCAL_MODEL:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(BASE_MODEL_PATH, "tokenizer.json")
    )
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.eos_token = "<|endoftext|>"
else:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
EOS_TOKEN = getattr(tokenizer, "eos_token", None) or "<|endoftext|>"

# ---------- 加载模型 ----------
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
        local_files_only=_LOCAL_MODEL,
    )
    model = prepare_model_for_kbit_training(model)
else:
    _load_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=_load_dtype,
        trust_remote_code=True,
        local_files_only=_LOCAL_MODEL,
    )
    model = model.to(device)

# ---------- LoRA ----------
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# ---------- 数据：法律 QA 模板 ----------
legal_prompt = """你是一个专业的法律咨询助手。请根据用户的问题提供专业、准确的法律建议。

### 问题：
{}

### 回答：
{}"""


def formatting_prompts_func(examples):
    texts = []
    for i, o in zip(examples["input"], examples["output"]):
        texts.append(legal_prompt.format(i, o) + EOS_TOKEN)
    return {"text": texts}


# 数据量：qa_corpus.json 约 3 万条，可限量做本地调试
MAX_TRAIN_SAMPLES = 10000  # None 表示全量
_seed = seed

dataset = load_legal_data(
    max_question_len=400,
    max_answer_len=800,
    use_first_answer_only=True,
)
if MAX_TRAIN_SAMPLES is not None:
    n = min(MAX_TRAIN_SAMPLES, len(dataset))
    dataset = dataset.shuffle(seed=_seed).select(range(n))
    print(f"截取 {len(dataset)} 条用于训练")
dataset = dataset.map(formatting_prompts_func, batched=True)

# ---------- 训练参数 ----------
batch_size = 2 if device in ("mps", "cpu") else 4
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=8,
    warmup_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
    bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
    logging_steps=1,
    optim="adamw_8bit" if use_4bit else "adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=seed,
    report_to="none",
    gradient_checkpointing=(device == "cuda"),
    dataset_text_field="text",
    max_length=max_seq_length,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_args,
)

# ---------- 训练 ----------
print("开始训练法律咨询 LoRA...")
trainer.train()
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA 已保存到: {LORA_SAVE_DIR}")

# ---------- 简单推理示例 ----------
def generate_legal_response(question, model=None, tokenizer=None):
    m = model or trainer.model
    tok = tokenizer or getattr(trainer, "processing_class", None)
    prompt = legal_prompt.format(question, "")
    enc = tok(prompt, return_tensors="pt").to(m.device)
    gen_kwargs = {k: v for k, v in enc.items() if k != "token_type_ids"}
    out = m.generate(
        **gen_kwargs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


for q in ["公司不发工资怎么办？", "交通事故对方全责怎么索赔？"]:
    print("\n问题:", q)
    print("回答:", generate_legal_response(q))
