"""
微调模型聊天服务：加载 Qwen2.5-1.5B + LoRA 医疗适配器，提供与 /ai/chat 兼容的对话接口。
供前端通过 POST /ai/finetuning/chat 调用。
可通过环境变量覆盖：FINETUNING_BASE_MODEL、FINETUNING_LORA_PATH、FINETUNING_MODEL_NAME（如 7B 则设为 Qwen2.5-7B-Instruct）。
"""

import os
import json
import threading
from queue import Queue
from pathlib import Path

# 懒加载：首次调用时再 import 大依赖
_model_tokenizer_lock = threading.Lock()
_model_tokenizer = None  # (lora_model, tokenizer)
_base_model_tokenizer = None  # (base_model, tokenizer)，用于对比输出


# 默认使用 1.5B + 医疗 LoRA 作为前端问答基础（轻量、响应快）
_DEFAULT_MODEL_NAME = "Qwen2.5-1.5B-Instruct"


def _get_paths():
    """解析 base 模型与 LoRA 适配器路径。

    - Base 模型：项目根 / models / Qwen / {FINETUNING_MODEL_NAME}，默认 1.5B（可用 FINETUNING_BASE_MODEL 覆盖完整路径）
    - LoRA 适配器：环境变量 FINETUNING_LORA_PATH，或根目录 lora/ 下最新一次 {日期}_{模型名}（由 FINETUNING_MODEL_NAME 决定），
      若不存在则回退到同目录 lora_model_medical_hf（兼容旧结构）。
    """
    from service.ai.finetuning.paths import get_finetuning_root, get_latest_lora_dir, get_project_root

    project_root = get_project_root()
    file_path = Path(__file__).resolve()
    model_name = os.environ.get("FINETUNING_MODEL_NAME", _DEFAULT_MODEL_NAME)
    base = os.environ.get("FINETUNING_BASE_MODEL") or str(project_root / "models" / "Qwen" / model_name)
    lora_env = os.environ.get("FINETUNING_LORA_PATH")
    if lora_env:
        lora = lora_env
    else:
        latest = get_latest_lora_dir(get_finetuning_root(), model_name=model_name)
        legacy = file_path.parent / "lora_model_medical_hf"
        lora = str(latest) if (latest and latest.is_dir()) else str(legacy)
    return base, lora


# Qwen2.5 标准 chat_template（无 tools），用于 tokenizer 从 tokenizer_file 加载时的兜底
_QWEN25_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "<|im_start|>system\n{{ messages[0]['content'] }}<|im_end|>\n{% else %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'system' and not loop.first %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _load_tokenizer(base_path):
    """加载 tokenizer，若遇 'dict' has no attribute 'model_type' 则用 tokenizer_file + 手动 chat_template 兜底。"""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    except AttributeError as e:
        if "model_type" not in str(e):
            raise
    tokenizer_file = os.path.join(base_path, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f"tokenizer.json 不存在: {tokenizer_file}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.eos_token = "<|im_end|>"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", "<|im_end|>")
    tokenizer.chat_template = _QWEN25_CHAT_TEMPLATE
    return tokenizer


def get_model():
    """懒加载：加载 base 模型 + LoRA 适配器，返回 (model, tokenizer)。"""
    global _model_tokenizer
    with _model_tokenizer_lock:
        if _model_tokenizer is not None:
            return _model_tokenizer

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_path, lora_path = _get_paths()
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"LoRA 适配器目录不存在: {lora_path}")
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Base 模型目录不存在: {base_path}")

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float16

        # tokenizer 从 base 加载；遇 dict.model_type 错误时用 tokenizer_file + chat_template 兜底
        tokenizer = _load_tokenizer(base_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
        model = model.to(device)
        model.eval()

        _model_tokenizer = (model, tokenizer)
        return _model_tokenizer


def get_base_model():
    """懒加载：仅加载 base 模型（无 LoRA），返回 (model, tokenizer)。用于与 LoRA 版对比输出。"""
    global _base_model_tokenizer
    with _model_tokenizer_lock:
        if _base_model_tokenizer is not None:
            return _base_model_tokenizer

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_path, lora_path = _get_paths()
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Base 模型目录不存在: {base_path}")

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float16

        tokenizer = _load_tokenizer(base_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = model.to(device)
        model.eval()

        _base_model_tokenizer = (model, tokenizer)
        return _base_model_tokenizer


def _run_one_model(model, tokenizer, prompt, gen_kw):
    """对单个模型做一次 generate，返回解码后的文本。"""
    import torch
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "token_type_ids"}
    with torch.no_grad():
        out = model.generate(**enc, **gen_kw)
    input_len = enc["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _messages_to_input(messages, tokenizer):
    """将 OpenAI 风格 messages 转为模型输入。Qwen2.5 使用 apply_chat_template。"""
    if not messages or not isinstance(messages, list):
        raise ValueError("messages 必须为非空列表")
    # 只保留 role/content，且 content 为字符串（本接口不做多模态）
    clean = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        if role == "system":
            role = "system"
        elif role in ("user", "assistant"):
            pass
        else:
            role = "user"
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                x.get("text", str(x)) for x in content if isinstance(x, dict) and "text" in x
            ) or str(content)
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        clean.append({"role": role, "content": content})
    if not clean:
        raise ValueError("messages 内容为空")
    # add_generation_prompt=True 会在末尾加 assistant 的 prompt
    text = tokenizer.apply_chat_template(
        clean,
        tokenize=False,
        add_generation_prompt=True,
        tokenizer=tokenizer,
    )
    return text


def _gen_kwargs(request_options, tokenizer):
    """从请求 options 里取出 generate 参数。"""
    opts = request_options or {}
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id")
    return {
        "max_new_tokens": int(opts.get("num_predict", opts.get("max_new_tokens", 512))),
        "temperature": float(opts.get("temperature", 0.7)),
        "top_p": float(opts.get("top_p", opts.get("top_p", 0.9))),
        "repetition_penalty": float(opts.get("repeat_penalty", 1.1)),
        "do_sample": True,
        "pad_token_id": pad_id,
    }


def chat(messages, stream=False, options=None):
    """
    使用微调模型生成回复。

    :param messages: [{"role":"user"|"assistant"|"system","content":"..."}, ...]
    :param stream: 是否流式返回
    :param options: 可选 {"temperature", "num_predict", "top_p", "repeat_penalty"}
    :return: stream=False 时返回完整 content 字符串；stream=True 时返回生成器，yield 文本块。
    """
    model, tokenizer = get_model()
    prompt = _messages_to_input(messages, tokenizer)
    gen_kw = _gen_kwargs(options, tokenizer)

    import torch
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "token_type_ids"}

    if stream:
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kw["streamer"] = streamer
        gen_kw["max_new_tokens"] = gen_kw.get("max_new_tokens", 512)

        def generate_in_thread():
            with torch.no_grad():
                model.generate(**enc, **gen_kw)

        t = threading.Thread(target=generate_in_thread)
        t.start()
        for piece in streamer:
            yield piece
        t.join()
        return
    text = _run_one_model(model, tokenizer, prompt, gen_kw)
    yield text


def chat_sync(messages, options=None):
    """非流式：返回完整回复字符串（仅 LoRA 模型）。"""
    gen = chat(messages, stream=False, options=options)
    return next(gen, "")


def chat_sync_compare(messages, options=None):
    """
    非流式：同时用 base（1.5B 原版）与 base+LoRA（lora/20260224_Qwen2.5-1.5B-Instruct）各生成一次，返回两段回复便于对比。
    :return: {"base": "基座回复", "lora": "LoRA 回复"}
    """
    lora_model, tokenizer = get_model()
    base_model, _ = get_base_model()
    prompt = _messages_to_input(messages, tokenizer)
    gen_kw = _gen_kwargs(options, tokenizer)
    base_text = _run_one_model(base_model, tokenizer, prompt, gen_kw)
    lora_text = _run_one_model(lora_model, tokenizer, prompt, gen_kw)
    return {"base": base_text, "lora": lora_text}


def chat_stream(messages, options=None):
    """流式：返回生成器，逐个 yield 文本块（仅 LoRA）。"""
    return chat(messages, stream=True, options=options)


def chat_stream_compare(messages, options=None):
    """
    流式：base 与 LoRA 两路并行生成，按到达顺序 yield (source, chunk)，source 为 "base" 或 "lora"。
    前端可同时往两栏追加内容。
    """
    import torch
    from transformers import TextIteratorStreamer

    base_model, tokenizer = get_base_model()
    lora_model, _ = get_model()
    prompt = _messages_to_input(messages, tokenizer)
    gen_kw = _gen_kwargs(options, tokenizer)
    gen_kw["max_new_tokens"] = gen_kw.get("max_new_tokens", 512)

    def make_enc(device):
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
        return enc

    enc_base = make_enc(base_model.device)
    enc_lora = make_enc(lora_model.device)
    base_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    lora_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    shared = Queue()

    def run_base():
        with torch.no_grad():
            base_model.generate(**enc_base, streamer=base_streamer, **gen_kw)

    def run_lora():
        with torch.no_grad():
            lora_model.generate(**enc_lora, streamer=lora_streamer, **gen_kw)

    def consume_base():
        for chunk in base_streamer:
            shared.put(("base", chunk))
        shared.put(("base", None))

    def consume_lora():
        for chunk in lora_streamer:
            shared.put(("lora", chunk))
        shared.put(("lora", None))

    t_base = threading.Thread(target=run_base)
    t_lora = threading.Thread(target=run_lora)
    t_cbase = threading.Thread(target=consume_base)
    t_clora = threading.Thread(target=consume_lora)
    t_base.start()
    t_lora.start()
    t_cbase.start()
    t_clora.start()

    base_done = lora_done = False
    while not (base_done and lora_done):
        source, chunk = shared.get()
        if chunk is None:
            if source == "base":
                base_done = True
            else:
                lora_done = True
            yield (source, None)
            continue
        yield (source, chunk)

    t_base.join()
    t_lora.join()
    t_cbase.join()
    t_clora.join()


def finetuning_chat_api():
    """Flask 视图：POST /ai/finetuning/chat。请求体与 /ai/chat 兼容，返回 JSON 或 SSE。"""
    from flask import request, Response, jsonify, stream_with_context

    data = request.get_json(silent=True) or {}
    if not data and request.get_data():
        return jsonify({"code": 400, "msg": "Invalid JSON or body too large"}), 400

    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"code": 400, "msg": "Missing or invalid messages"}), 400

    stream = data.get("stream", False)
    compare = data.get("compare", False)
    options = data.get("options", {})

    try:
        if stream:
            if compare:
                # 同时流式输出 base 与 lora 两路，事件格式 {"source": "base"|"lora", "content": "..."} 或 {"source": "...", "done": true}
                def generate_compare():
                    try:
                        for source, chunk in chat_stream_compare(messages, options=options):
                            if chunk is None:
                                out = {"source": source, "done": True}
                            else:
                                out = {"source": source, "content": chunk}
                            yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"

                gen = generate_compare()
            else:
                def generate():
                    try:
                        for chunk in chat_stream(messages, options=options):
                            out = {"message": {"content": chunk}, "response": chunk}
                            yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"

                gen = generate()

            return Response(
                stream_with_context(gen),
                mimetype="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
            )
        # 非流式：同时返回基座与 LoRA 两种回复，便于对比
        result = chat_sync_compare(messages, options=options)
        return jsonify({
            "code": 0,
            "message": {
                "role": "assistant",
                "content": result.get("lora") or "",
                "base_content": result.get("base") or "",
                "lora_content": result.get("lora") or "",
            },
            "base": result.get("base") or "",
            "lora": result.get("lora") or "",
        })
    except FileNotFoundError as e:
        return jsonify({"code": 404, "msg": str(e)}), 404
    except ValueError as e:
        return jsonify({"code": 400, "msg": str(e)}), 400
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500
