#!/usr/bin/env python
# coding: utf-8
"""
Qwen2.5-1.5B 医疗微调 - Hugging Face 原生版（无 Unsloth）
"""

import os
import sys

# 直接运行脚本时项目根不在 PYTHONPATH，先加入以便 import service.ai.finetuning.paths
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

# ---------- 路径配置 ----------
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
# 项目根：脚本在 service/ai/finetuning/ 下
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen2.5-1.5B-Instruct")
if not os.path.isdir(BASE_MODEL_PATH):
    BASE_MODEL_PATH = os.path.join(
        _SCRIPT_DIR, "models", "Qwen", "Qwen2.5-1.5B-Instruct"
    )
_LOCAL_MODEL = os.path.isdir(
    BASE_MODEL_PATH
)  # 本地目录则 from_pretrained 用 local_files_only
MEDICAL_DATA_DIR = os.path.join(_SCRIPT_DIR, "【数据集】中文医疗数据")

# 输出目录：父目录 = 日期+模型名，其下 lora/ 与 outputs_hf/
from service.ai.finetuning.paths import (
    get_finetuning_root,
    get_run_parent_dir,
    get_lora_dir,
    get_outputs_hf_dir,
)

_RUN_PARENT = get_run_parent_dir(
    get_finetuning_root(), model_name="Qwen2.5-1.5B-Instruct"
)
_LORA_DIR = get_lora_dir(_RUN_PARENT)
os.makedirs(_RUN_PARENT, exist_ok=True)
os.makedirs(_LORA_DIR, exist_ok=True)  # 确保始终生成在 lora 文件夹下
LORA_SAVE_DIR = str(_LORA_DIR)
OUTPUT_DIR = str(get_outputs_hf_dir(_RUN_PARENT))
print(f"本次运行目录: {_RUN_PARENT} -> lora: {LORA_SAVE_DIR}, outputs_hf: {OUTPUT_DIR}")

# ---------- 设备与精度 ----------
if torch.cuda.is_available():
    device = "cuda"
    use_4bit = True  # 显存紧张时用 4bit
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
    use_4bit = False  # Mac 上通常不用 4bit
else:
    device = "cpu"
    use_4bit = False

print(f"使用设备: {device}, 4bit 量化: {use_4bit}")

max_seq_length = 2048
seed = 3407

# ---------- 加载 tokenizer ----------
# 本地加载时用 PreTrainedTokenizerFast(tokenizer_file=...) 绕过 transformers 对 tokenizer_config 的 dict.model_type 兼容性 bug
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
    # MPS/CPU 不能用 device_map="auto"（会把部分参数卸载到 meta device，与 MPS backward 不兼容）
    # 18GB 内存：用 float16 加载（7B≈14GB），base 参数冻结，只有 LoRA adapter 产生梯度
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


# MAX_TRAIN_SAMPLES: 本地调试/限量用，设为 None 则使用全量数据
# 取子集时先打乱再取前 N 条，避免全是第一个科室（当前顺序下先男科）
MAX_TRAIN_SAMPLES = 10000
SHUFFLE_BEFORE_SELECT = True  # 均匀取样，False是全男科数据
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

# ---------- 训练参数 -----------
# 全量训练：加大 batch/accumulation 减少步数，减少 epoch 缩短总时间（显存/内存允许下）
# 1.5B 较轻，MPS/18GB 可试 batch=2；CUDA 可用更大 batch
_use_full_data = MAX_TRAIN_SAMPLES is None
batch_size = 2 if device in ("mps", "cpu") else 4
grad_accum = 16 if _use_full_data else 8
num_epochs = 2 if _use_full_data else 1
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    warmup_steps=5 if _use_full_data else 2,
    num_train_epochs=num_epochs,
    learning_rate=2e-4,
    # MPS 不支持 fp16 GradScaler，也不支持 bf16；统一用 fp32，CUDA 时可开精度
    fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
    bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
    logging_steps=1,
    optim="adamw_8bit" if use_4bit else "adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=seed,
    report_to="none",
    # MPS 上 gradient_checkpointing 与 meta device 冲突，须关闭
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
print("开始训练...")
trainer.train()
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA 已保存到: {LORA_SAVE_DIR}")


# ---------- 简单推理示例 ----------
def generate_medical_response(question, model=None, tokenizer=None):
    m = model or trainer.model
    tok = tokenizer or getattr(trainer, "processing_class", None)
    prompt = medical_prompt.format(question, "")
    enc = tok(prompt, return_tensors="pt").to(m.device)
    # Qwen2 不用 token_type_ids，generate 会报错，只传 input_ids/attention_mask
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


for q in ["我最近总是感觉头晕，应该怎么办？", "感冒发烧应该吃什么药？"]:
    print("\n问题:", q)
    print("回答:", generate_medical_response(q))

# =============================================================================
# 国内云 GPU 主机推荐（跑 Unsloth 原版脚本、大 batch、更快训练）
# =============================================================================
# - AutoDL (autodl.com)：按量/包时，RTX 3090/4090 等，镜像多，适合算法与微调
# - 恒源云 (gpushare.com)：按量 GPU，价格透明，有预装 PyTorch 镜像
# - 阿里云 PAI-DSW / 腾讯云 GPU 实例：适合企业、需备案与实名
# - 矩池云 (matpool.com)、智星云：学生/个人有优惠
# - 使用方式：上传代码与数据，选择 PyTorch + CUDA 镜像，pip install unsloth，运行
#   Qwen2_5_(7B)_医疗微调.py（非本 _hf 版）
# =============================================================================
