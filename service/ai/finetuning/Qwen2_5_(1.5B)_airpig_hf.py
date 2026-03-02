#!/usr/bin/env python
# coding: utf-8
"""
Qwen2.5-1.5B 空气小猪客服微调 - 使用空气小猪产品概念 QA 数据集训练 LoRA。
数据来源：dataset/【数据集】空气小猪/qa_train.json（Alpaca 风格 instruction/input/output）
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

from service.ai.finetuning.dataset_airpig import load_airpig_data
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

# 空气小猪 LoRA 单独目录：lora/{日期}_Qwen2.5-1.5B-Instruct-airpig
AIRPIG_RUN_NAME = "Qwen2.5-1.5B-Instruct-airpig"
_RUN_PARENT = get_run_parent_dir(get_finetuning_root(), model_name=AIRPIG_RUN_NAME)
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

# ---------- 数据：用与推理完全一致的 chat 格式训练（train=inference 格式对齐）----------
# 产品定义 system 与 finetuning.py 里 AIRPIG_SYSTEM_PROMPT 保持一致
AIRPIG_SYSTEM = (
    "你是空气小猪的客服助手。空气小猪是一款以即时通讯为核心的语言环境产品，"
    "帮助用户把日常聊天内容转换成自己正在学习的目标语言并朗读，形成长期外语学习环境。"
    "请仅根据上述产品定义回答用户问题。"
)


def formatting_prompts_func(examples):
    """
    将每条 QA 转为 Qwen2.5 chat 格式：
    <|im_start|>system ... <|im_end|>
    <|im_start|>user 问题 <|im_end|>
    <|im_start|>assistant 回答 <|im_end|>
    训练格式与 finetuning.py 推理格式完全一致，LoRA 学到的分布才能在推理时正确触发。
    """
    texts = []
    for q, a in zip(examples["input"], examples["output"]):
        messages = [
            {"role": "system", "content": AIRPIG_SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # 兜底：手动拼 Qwen2.5 格式
            text = (
                f"<|im_start|>system\n{AIRPIG_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n{a}<|im_end|>\n"
            )
        texts.append(text + EOS_TOKEN)
    return {"text": texts}


# 数据量：优先用 qa_augmented.json（扩增后 200+ 条），否则退回 qa_train.json（15 条）
# 扩增方法：python dataset/【数据集】空气小猪/augment.py
import importlib.util as _ilu
from pathlib import Path as _P
_AUGMENTED = _P(__file__).resolve().parents[3] / "dataset" / "【数据集】空气小猪" / "qa_augmented.json"
_SRC_FILE = str(_AUGMENTED) if _AUGMENTED.exists() else None
if _SRC_FILE:
    print(f"使用扩增数据集：{_AUGMENTED.name}")
else:
    print("未找到 qa_augmented.json，使用原始 qa_train.json（仅 15 条，效果有限）")

MAX_TRAIN_SAMPLES = None  # None 表示全量
AIRPIG_EPOCHS = 10        # 小数据时建议 10～15；200+ 条时可降到 3～5
_seed = seed

dataset = load_airpig_data(json_path=_SRC_FILE, max_question_len=500, max_answer_len=1200)
print(f"训练样本数: {len(dataset)}")
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
    num_train_epochs=AIRPIG_EPOCHS,
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
print("开始训练空气小猪客服 LoRA...")
trainer.train()
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA 已保存到: {LORA_SAVE_DIR}")

# ---------- 简单推理示例（与训练格式一致：chat template + system）----------
def generate_airpig_response(question, model=None, tokenizer=None):
    m = model or trainer.model
    tok = tokenizer or getattr(trainer, "processing_class", None)
    messages = [
        {"role": "system", "content": AIRPIG_SYSTEM},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = (
            f"<|im_start|>system\n{AIRPIG_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(m.device)
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


for q in ["空气小猪是什么？", "空气小猪和翻译软件有什么区别？"]:
    print("\n问题:", q)
    print("回答:", generate_airpig_response(q))
