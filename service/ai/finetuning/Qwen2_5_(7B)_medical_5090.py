#!/usr/bin/env python
# coding: utf-8
"""
Qwen2.5-1.5B 医疗微调 - RTX 5090（32GB）版本，基于 Unsloth。
5090：更大 batch、不保存训练过程 checkpoint，LoRA 存到 lora/{日期}_Qwen2.5-1.5B-Instruct-medical-5090。
"""
import os
import sys

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
_bootstrap_root = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)

import torch

# 无外网时 Unsloth 会请求 huggingface.co 做统计检查导致报错，先 patch 再 import
try:
    import unsloth.models._utils as _u

    _u.get_statistics = lambda *a, **k: None
except Exception:
    pass


# 兼容云服务器：transformers 在 tokenizer 里把 json.load 的 dict 当 .model_type 用会报错，运行时 patch json.load
def _patch_tokenizer_config_json():
    import json as _json_mod

    if getattr(_json_mod, "_config_wrap_patch", False):
        return
    _orig = _json_mod.load

    class _ConfigWrap:
        __slots__ = ("_d",)

        def __init__(self, d):
            self._d = d

        def __getattr__(self, k):
            return self._d.get(k)

        def get(self, k, default=None):
            return self._d.get(k, default)

    def _patched(*a, **k):
        data = _orig(*a, **k)
        if (
            isinstance(data, dict)
            and "model_type" in data
            and "transformers_version" in data
        ):
            return _ConfigWrap(data)
        return data

    _json_mod.load = _patched
    _json_mod._config_wrap_patch = True


try:
    _patch_tokenizer_config_json()
except Exception:
    pass
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import pandas as pd
from datasets import Dataset

from service.ai.finetuning.paths import (
    get_finetuning_root,
    get_run_parent_dir,
    get_lora_dir,
    get_outputs_hf_dir,
)

# ---------- 路径 ----------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen2.5-1.5B-Instruct")
if not os.path.isdir(BASE_MODEL_PATH):
    BASE_MODEL_PATH = os.path.join(
        _SCRIPT_DIR, "models", "Qwen", "Qwen2.5-1.5B-Instruct"
    )

MEDICAL_DATA_DIR = os.path.join(_PROJECT_ROOT, "dataset", "【数据集】中文医疗数据")
if not os.path.isdir(MEDICAL_DATA_DIR):
    MEDICAL_DATA_DIR = os.path.join(_SCRIPT_DIR, "【数据集】中文医疗数据")

RUN_NAME = "Qwen2.5-1.5B-Instruct-medical-5090"
_RUN_PARENT = get_run_parent_dir(get_finetuning_root(), model_name=RUN_NAME)
os.makedirs(_RUN_PARENT, exist_ok=True)
LORA_SAVE_DIR = str(get_lora_dir(_RUN_PARENT))
OUTPUT_DIR = str(get_outputs_hf_dir(_RUN_PARENT))
print(f"本次运行目录: {_RUN_PARENT} -> lora: {LORA_SAVE_DIR}, outputs_hf: {OUTPUT_DIR}")

# ---------- 模型参数 ----------
max_seq_length = 2048
dtype = None
load_in_4bit = os.environ.get("USE_4BIT", "1").strip() not in (
    "0",
    "false",
    "no",
)  # 默认 4bit，5090 可设 USE_4BIT=0 试全量
_local_model = os.path.isdir(BASE_MODEL_PATH)
if _local_model:
    print(f"使用本地基座: {BASE_MODEL_PATH}")
else:
    print(f"将从 HF 加载基座: {BASE_MODEL_PATH}（若无缓存会较慢）")

print("正在加载基座模型（Unsloth）...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    local_files_only=_local_model,
)
print("基座加载完成，正在添加 LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("LoRA 添加完成。")

# ---------- 数据 ----------
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""
EOS_TOKEN = tokenizer.eos_token


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
        "IM_内科": "内科",
        "Surgical_外科": "外科",
        "Pediatric_儿科": "儿科",
        "Oncology_肿瘤科": "肿瘤科",
        "OAGD_妇产科": "妇产科",
        "Andriatria_男科": "男科",
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
                    for k in ("question", "问题", "ask"):
                        if k in row and pd.notna(row.get(k)):
                            q = str(row[k]).strip()
                            break
                    a = None
                    for k in ("answer", "回答", "response"):
                        if k in row and pd.notna(row.get(k)):
                            a = str(row[k]).strip()
                            break
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
    print(f"成功处理 {len(data)} 条数据")
    return Dataset.from_list(data)


def formatting_prompts_func(examples):
    texts = []
    for i, o in zip(examples["input"], examples["output"]):
        texts.append(medical_prompt.format(i, o) + EOS_TOKEN)
    return {"text": texts}


# 先跑少量数据看效果，改大或设为 None 表示全量
MAX_TRAIN_SAMPLES = None
_seed = 3407

dataset = load_medical_data(MEDICAL_DATA_DIR)
if MAX_TRAIN_SAMPLES is not None:
    n = min(MAX_TRAIN_SAMPLES, len(dataset))
    dataset = dataset.shuffle(seed=_seed).select(range(n))
    print(f"截取 {n} 条用于训练（先跑看效果）")
dataset = dataset.map(formatting_prompts_func, batched=True)

# ---------- 训练参数（5090：4bit 下可开大 batch 拉高显存，全量用 USE_4BIT=0）---------
# 想多占显存：per_device 调大（如 32/64）或运行时 USE_4BIT=0 全精度
per_device_batch = 32
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# ---------- 训练 ----------
print("开始训练医疗 LoRA (1.5B 5090 版)...")
trainer.train()
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA 已保存到: {LORA_SAVE_DIR}")


# ---------- 推理示例 ----------
def generate_medical_response(question):
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([medical_prompt.format(question, "")], return_tensors="pt").to(
        model.device
    )
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


for q in ["我最近总是感觉头晕，应该怎么办？", "感冒发烧应该吃什么药？"]:
    print("\n" + "=" * 50)
    print("问题:", q)
    print("回答:", generate_medical_response(q))
