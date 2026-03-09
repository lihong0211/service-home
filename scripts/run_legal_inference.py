#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行验证法律 LoRA 答案：加载基座 + 法律 LoRA，对传入的问题生成回答。

用法:
  python scripts/run_legal_inference.py "公司不发工资怎么办？"
  python scripts/run_legal_inference.py "问题1" "问题2"
  python scripts/run_legal_inference.py   # 交互模式，逐行输入问题
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen" / "Qwen2.5-1.5B-Instruct"
LORA_LEGAL_5090 = "Qwen2.5-1.5B-Instruct-legal-5090"


def get_latest_legal_lora_dir():
    """取法律 LoRA 目录：优先 lora/legal（与 finetuning 服务一致），否则取最新 dated 目录。"""
    lora_base = PROJECT_ROOT / "lora"
    if not lora_base.is_dir():
        return None
    fixed = lora_base / "legal"
    if fixed.is_dir() and (fixed / "adapter_config.json").exists():
        return fixed
    prefix = "_" + LORA_LEGAL_5090
    candidates = [d for d in lora_base.iterdir() if d.is_dir() and d.name.endswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.name, reverse=True)
    return candidates[0]


def main():
    if not BASE_MODEL_PATH.is_dir():
        print(f"基座模型不存在: {BASE_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    lora_dir = get_latest_legal_lora_dir()
    if lora_dir is None:
        print("未找到法律 LoRA 目录 (lora/legal 或 lora/*_Qwen2.5-1.5B-Instruct-legal-5090)", file=sys.stderr)
        sys.exit(1)

    questions = [q.strip() for q in sys.argv[1:] if q.strip()]
    if not questions:
        print("用法: python scripts/run_legal_inference.py \"你的问题\"")
        print("或直接运行进入交互模式，逐行输入问题（空行退出）")
        print("输入问题:")
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                break
            questions.append(line)

    if not questions:
        sys.exit(0)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    tokenizer_path = lora_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
        if getattr(tokenizer, "eos_token", None) is None:
            tokenizer.eos_token = "<|endoftext|>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(lora_dir), trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", "<|endoftext|>")

    print(f"加载基座: {BASE_MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    print(f"加载 LoRA: {lora_dir} ...")
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model.eval()

    legal_prompt = """你是一个专业的法律咨询助手。请根据用户的问题提供专业、准确的法律建议。

### 问题：
{}

### 回答：
{}"""

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id

    for q in questions:
        prompt = legal_prompt.format(q, "")
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            out = model.generate(
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
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
        print(f"\n问题: {q}")
        print(f"回答: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()
