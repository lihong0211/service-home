#!/usr/bin/env python
# coding: utf-8
"""
Qwen2.5-1.5B 医疗微调 - RTX 5090（32GB）版本。
默认全量 bf16 + batch=8 + gradient checkpointing，不保存 outputs_hf checkpoint。
若 OOM 可启用 4bit：USE_4BIT=1 python -m ...

数据来源同 Qwen2_5_(1.5B)_medical_hf.py：dataset/【数据集】中文医疗数据 或脚本同目录下。
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
import pandas as pd
from datasets import Dataset

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
    BASE_MODEL_PATH = os.path.join(
        _SCRIPT_DIR, "models", "Qwen", "Qwen2.5-1.5B-Instruct"
    )
_LOCAL_MODEL = os.path.isdir(BASE_MODEL_PATH)

# 医疗数据：优先项目 dataset，其次脚本同目录
MEDICAL_DATA_DIR = os.path.join(_PROJECT_ROOT, "dataset", "【数据集】中文医疗数据")
if not os.path.isdir(MEDICAL_DATA_DIR):
    MEDICAL_DATA_DIR = os.path.join(_SCRIPT_DIR, "【数据集】中文医疗数据")

# 5090 版单独目录
MEDICAL_RUN_NAME = "Qwen2.5-1.5B-Instruct-medical-5090"
_RUN_PARENT = get_run_parent_dir(get_finetuning_root(), model_name=MEDICAL_RUN_NAME)
os.makedirs(_RUN_PARENT, exist_ok=True)
LORA_SAVE_DIR = str(get_lora_dir(_RUN_PARENT))
OUTPUT_DIR = str(get_outputs_hf_dir(_RUN_PARENT))
print(f"本次运行目录: {_RUN_PARENT} -> lora: {LORA_SAVE_DIR}, outputs_hf: {OUTPUT_DIR}")

# ---------- 设备与精度 -----------
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
print(f"使用设备: {device}, 4bit: {use_4bit}, bf16: {use_bf16}")

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

# ---------- 加载模型 -----------
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
    _load_dtype = torch.bfloat16 if (device == "cuda" and use_bf16) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=_load_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=_LOCAL_MODEL,
    )
    if device != "cuda":
        model = model.to(device)

# ---------- LoRA ----------
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# ---------- 数据 ----------
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""


def read_csv_with_encoding(file_path):
    for enc in ["gbk", "gb2312", "gb18030", "utf-8"]:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取: {file_path}")


def load_medical_data(data_dir):
    data = []
    departments = {
        "Andriatria_男科": "男科",
        "IM_内科": "内科",
        "Surgical_外科": "外科",
        "Pediatric_儿科": "儿科",
        "Oncology_肿瘤科": "肿瘤科",
        "OAGD_妇产科": "妇产科",
    }
    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"目录不存在: {dept_path}")
            continue
        for f in os.listdir(dept_path):
            if not f.endswith(".csv"):
                continue
            fp = os.path.join(dept_path, f)
            try:
                df = read_csv_with_encoding(fp)
                for _, row in df.iterrows():
                    q = None
                    if "question" in row:
                        q = str(row["question"]).strip()
                    elif "问题" in row:
                        q = str(row["问题"]).strip()
                    elif "ask" in row:
                        q = str(row["ask"]).strip()
                    a = None
                    if "answer" in row:
                        a = str(row["answer"]).strip()
                    elif "回答" in row:
                        a = str(row["回答"]).strip()
                    elif "response" in row:
                        a = str(row["response"]).strip()
                    if not q or not a or len(q) > 200 or len(a) > 200:
                        continue
                    data.append(
                        {
                            "instruction": "请回答以下医疗相关问题",
                            "input": q,
                            "output": a,
                        }
                    )
            except Exception as e:
                print(f"处理 {f} 出错: {e}")
    if not data:
        raise ValueError("没有加载到任何数据")
    print(f"加载 {len(data)} 条数据")
    return Dataset.from_list(data)


def formatting_prompts_func(examples):
    texts = []
    for i, o in zip(examples["input"], examples["output"]):
        texts.append(medical_prompt.format(i, o) + EOS_TOKEN)
    return {"text": texts}


MAX_TRAIN_SAMPLES = 10000
SHUFFLE_BEFORE_SELECT = True
_seed = seed

dataset = load_medical_data(MEDICAL_DATA_DIR)
if MAX_TRAIN_SAMPLES is not None:
    n = min(MAX_TRAIN_SAMPLES, len(dataset))
    if SHUFFLE_BEFORE_SELECT and n < len(dataset):
        dataset = dataset.shuffle(seed=_seed)
    dataset = dataset.select(range(n))
    print(
        f"截取 {len(dataset)} 条"
        + ("（已打乱混合各科室）" if SHUFFLE_BEFORE_SELECT and n < len(dataset) else "")
    )
dataset = dataset.map(formatting_prompts_func, batched=True)

# ---------- 训练参数（5090：大 batch，不保存 checkpoint）----------
per_device_batch = 8 if device == "cuda" else 2
gradient_accumulation_steps = 8
warmup_steps = 2
num_train_epochs = 3
use_gradient_checkpointing = True
optim_name = "adamw_8bit" if use_4bit else "adamw_torch"

sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=2e-4,
    fp16=(device == "cuda" and not use_bf16),
    bf16=(device == "cuda" and use_bf16),
    logging_steps=1,
    optim=optim_name,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=seed,
    report_to="none",
    save_strategy="no",
    gradient_checkpointing=use_gradient_checkpointing,
    dataset_text_field="text",
    max_length=max_seq_length,
    packing=False,
    dataset_num_proc=2,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_args,
)

# ---------- 训练 ----------
print("开始训练医疗 LoRA (5090 版)...")
trainer.train()
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA 已保存到: {LORA_SAVE_DIR}")

# ---------- 简单推理示例 ----------
def generate_medical_response(question, model=None, tokenizer=None):
    m = model or trainer.model
    tok = tokenizer or getattr(trainer, "processing_class", None)
    m.eval()
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = getattr(tok, "eos_token", "<|endoftext|>")
    prompt = medical_prompt.format(question, "")
    enc = tok(prompt, return_tensors="pt").to(m.device)
    gen_kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None) or eos_id
    with torch.no_grad():
        out = m.generate(
            **gen_kw,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.25,
            do_sample=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    gen = out[0][enc["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


for q in ["我最近总是感觉头晕，应该怎么办？", "感冒发烧应该吃什么药？"]:
    print("\n问题:", q)
    print("回答:", generate_medical_response(q))
