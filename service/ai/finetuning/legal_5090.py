#!/usr/bin/env python
# coding: utf-8
"""
法律咨询 LoRA 微调 - Qwen2.5-1.5B-Instruct + DISC-Law-SFT（RTX 5090）。
使用 Unsloth 加速：自动 Flash Attention + 融合内核，比原生 transformers 快 2-5x，显存减半。

基座：Qwen2.5-1.5B-Instruct（指令对齐版）
数据：
  - dataset/【数据集】legal_hq/DISC-Law-SFT/DISC-Law-SFT-Pair-QA-released.jsonl   (79,692条)
  - dataset/【数据集】legal_hq/DISC-Law-SFT/DISC-Law-SFT-Triplet-QA-released.jsonl (23,331条)
格式：Qwen chat template（apply_chat_template）
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


# ── 修复 transformers tokenizer bug（在任何 import 之前执行）─────────────────────
# 部分版本的 transformers tokenization_utils_base.py 里有：
#   if _is_local and _config.model_type not in [...]
# 当 _config 是 dict 时报 AttributeError。修复：加 getattr 保护。
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
    """把 SFTTrainer 里 eos_token 词表校验的 raise 降级为 pass（Qwen2+unsloth 兼容）。"""
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
        # 用括号计数定位完整的 raise ValueError(...) 块并替换为 pass
        idx = src.find(marker)
        start = src.rfind("raise ValueError(", 0, idx)
        if start == -1:
            print("[startup] 未找到 raise ValueError，跳过")
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
        old_block = src[start:end]
        new_block = "pass  # eos_token check skipped (patched for Qwen2+unsloth)"
        with open(spec.origin, "w", encoding="utf-8") as f:
            f.write(src.replace(old_block, new_block, 1))
        print("[startup] SFTTrainer eos_token check 已跳过")
    except Exception as e:
        print(f"[startup] SFTTrainer patch 失败: {e}")


_fix_sft_trainer_eos_check()

import torch
from datasets import Dataset

# Unsloth 在无外网时上报统计会崩溃，patch 掉 snapshot_download 拦截网络调用
import unsloth.models._utils as _unsloth_utils

_unsloth_utils.snapshot_download = lambda *a, **kw: None

# trl 必须在 unsloth 完成 patch 之后 import，否则拿到的是未被 patch 的原始版本
from unsloth import FastLanguageModel  # noqa: E402 - unsloth patches trl here
from trl import SFTTrainer, SFTConfig  # noqa: E402 - import after unsloth patch

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

LEGAL_RUN_NAME = "Qwen2.5-1.5B-Instruct-legal-disc-5090"
_RUN_PARENT = get_run_parent_dir(get_finetuning_root(), model_name=LEGAL_RUN_NAME)
os.makedirs(_RUN_PARENT, exist_ok=True)
LORA_SAVE_DIR = str(get_lora_dir(_RUN_PARENT))
OUTPUT_DIR = str(get_outputs_hf_dir(_RUN_PARENT))
print(f"基座: {BASE_MODEL_PATH}")
print(f"LoRA 保存: {LORA_SAVE_DIR}")

max_seq_length = 1024  # 法律问答平均 400-600 token，1024 足够，降低 attention 计算量
seed = 3407

# ── Unsloth：模型 + Tokenizer ──────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=None,  # 自动检测：5090 走 bf16
    load_in_4bit=False,  # 32GB 显存无需量化
    local_files_only=True,  # 无外网，禁止联网校验
)
print(f"设备: {next(model.parameters()).device}")

# unsloth 加载 Qwen2 时会把 eos_token 设成占位符 <EOS_TOKEN>。
# 直接赋属性对 Fast Tokenizer 后端不够，必须用 add_special_tokens 才能真正覆盖。
tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

# ── LoRA（通过 unsloth 接口）────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
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
    use_gradient_checkpointing="unsloth",  # unsloth 优化版，比标准 GC 更省显存
    random_state=seed,
    use_rslora=False,
)
model.print_trainable_parameters()

# ── System Prompt ──────────────────────────────────────────────────────────────
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


MAX_TRAIN_SAMPLES = None  # 试跑；正式全量训练设为 None

raw_data = load_disc_law_sft(max_input_len=800, max_output_len=1000)
if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < len(raw_data):
    random.seed(seed)
    raw_data = random.sample(raw_data, MAX_TRAIN_SAMPLES)
print(f"总计: {len(raw_data):,} 条")


# ── 格式化：Qwen chat template ─────────────────────────────────────────────────
def format_sample(item):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": item["answer"]},
    ]
    # apply_chat_template 已在 assistant 末尾添加 <|im_end|>，无需手动追加 eos
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
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    num_train_epochs=1,  # 试跑；正式全量改为 2-3
    learning_rate=1e-4,
    warmup_ratio=0.03,
    bf16=True,
    fp16=False,
    logging_steps=10,
    optim="adamw_8bit",  # unsloth 推荐
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
print("开始训练法律 LoRA (Unsloth + DISC-Law-SFT)...")
trainer.train()
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA 已保存到: {LORA_SAVE_DIR}")


# ── 推理示例 ──────────────────────────────────────────────────────────────────
def generate_legal_response(question, model=None, tokenizer=None):
    m = model or trainer.model
    tok = tokenizer or getattr(trainer, "processing_class", None)
    FastLanguageModel.for_inference(m)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tok(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(
            **enc,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
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
