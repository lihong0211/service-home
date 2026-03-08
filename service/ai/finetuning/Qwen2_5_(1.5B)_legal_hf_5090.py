#!/usr/bin/env python
# coding: utf-8
"""
Qwen2.5-1.5B 法律咨询微调 - RTX 5090（32GB）版本。
不做 4bit 量化，全量 bf16 + 更大 batch，适合 5090/4090/3090 等大显存显卡。

数据来源同 Qwen2_5_(1.5B)_legal_hf.py：
- dataset/【数据集】legal/qa_corpus.json
- dataset/【数据集】legal/kg_crime.json
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
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

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
    BASE_MODEL_PATH = os.path.join(
        _SCRIPT_DIR, "models", "Qwen", "Qwen2.5-1.5B-Instruct"
    )
_LOCAL_MODEL = os.path.isdir(BASE_MODEL_PATH)

# 5090 版单独目录，便于与 4bit 版区分
LEGAL_RUN_NAME = "Qwen2.5-1.5B-Instruct-legal-5090"
_RUN_PARENT = get_run_parent_dir(get_finetuning_root(), model_name=LEGAL_RUN_NAME)
os.makedirs(_RUN_PARENT, exist_ok=True)
LORA_SAVE_DIR = str(get_lora_dir(_RUN_PARENT))
OUTPUT_DIR = str(get_outputs_hf_dir(_RUN_PARENT))
print(f"本次运行目录: {_RUN_PARENT} -> lora: {LORA_SAVE_DIR}, outputs_hf: {OUTPUT_DIR}")

# ---------- 设备与精度（5090：全量 bf16，不量化）----------
if torch.cuda.is_available():
    device = "cuda"
    use_4bit = False  # 5090 32GB 不量化
    use_bf16 = torch.cuda.is_bf16_supported()
else:
    device = "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "cpu"
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

# ---------- 加载模型（全量 bf16/fp16，无量化）----------
dtype = torch.bfloat16 if (device == "cuda" and use_bf16) else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=dtype,
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


MAX_TRAIN_SAMPLES = 50000
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

# ---------- 训练参数（5090 大 batch，无梯度检查点可提速）----------
# 5090 32GB：1.5B 全量 + LoRA 可轻松跑 per_device 16~32
per_device_batch = 16 if device == "cuda" else 4
gradient_accumulation_steps = 4  # effective batch = 16*4 = 64
# 显存仍紧张时可开 gradient_checkpointing 或减小 per_device_batch
use_gradient_checkpointing = False  # 5090 关掉可提速

sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=(device == "cuda" and not use_bf16),
    bf16=(device == "cuda" and use_bf16),
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=seed,
    report_to="none",
    gradient_checkpointing=use_gradient_checkpointing,
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
print("开始训练法律咨询 LoRA (5090 版)...")
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
    gen = out[0][enc["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


for q in ["公司不发工资怎么办？", "交通事故对方全责怎么索赔？"]:
    print("\n问题:", q)
    print("回答:", generate_legal_response(q))
