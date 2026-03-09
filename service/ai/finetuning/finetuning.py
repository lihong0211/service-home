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
_model_tokenizer = {}  # lora_type -> (lora_model, tokenizer)
_base_model_tokenizer = None  # (base_model, tokenizer)，用于对比输出

# 默认使用 1.5B + 医疗 LoRA 作为前端问答基础（轻量、响应快）
_DEFAULT_MODEL_NAME = "Qwen2.5-1.5B-Instruct"

# 前端选择的 LoRA 类型：medical=医疗问诊，legal=法律咨询，airpig=空气小猪客服
# 指定目录名时用 lora/{目录名}（如 medical、legal 无日期前缀）；None 时用 get_latest_lora_dir 取最新
LORA_TYPE_DIRS = {
    "medical": "medical",  # lora/medical
    "legal": "legal",  # lora/legal
    "airpig": None,
}
# 当 LORA_TYPE_DIRS[type] 为 None 时，用此 model_name 找最新 lora 目录（目录名后缀匹配）
LORA_TYPE_MODEL_NAME = {
    "medical": "Qwen2.5-1.5B-Instruct-medical-5090",
    "legal": "Qwen2.5-1.5B-Instruct-legal-5090",
    "airpig": None,
}


def _get_paths(lora_type="medical"):
    """解析 base 模型与 LoRA 适配器路径。

    - lora_type: "medical" | "legal" | "airpig"，决定使用哪一版 LoRA。
    - Base 模型：项目根 / models / Qwen / {FINETUNING_MODEL_NAME}
    - LoRA：LORA_TYPE_DIRS[lora_type] 指定目录名时用 lora/{目录名}，否则用 get_latest_lora_dir。
    """
    from service.ai.finetuning.paths import (
        get_finetuning_root,
        get_latest_lora_dir,
        get_project_root,
    )

    project_root = get_project_root()
    file_path = Path(__file__).resolve()
    model_name = os.environ.get("FINETUNING_MODEL_NAME", _DEFAULT_MODEL_NAME)
    base = os.environ.get("FINETUNING_BASE_MODEL") or str(
        project_root / "models" / "Qwen" / model_name
    )
    lora_env = os.environ.get("FINETUNING_LORA_PATH")
    if lora_env:
        lora = lora_env
    else:
        dir_name = LORA_TYPE_DIRS.get(lora_type) if lora_type else None
        if dir_name:
            lora = str(project_root / "lora" / dir_name)
        else:
            # medical / airpig 等：按类型取对应 model_name 找最新 lora 目录
            lookup_name = LORA_TYPE_MODEL_NAME.get(lora_type, _DEFAULT_MODEL_NAME)
            latest = get_latest_lora_dir(get_finetuning_root(), model_name=lookup_name)
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


def get_model(lora_type="medical"):
    """懒加载：按 lora_type 加载 base + 对应 LoRA，返回 (model, tokenizer)。"""
    global _model_tokenizer
    lora_type = lora_type or "medical"
    with _model_tokenizer_lock:
        if lora_type in _model_tokenizer:
            return _model_tokenizer[lora_type]

        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        base_path, lora_path = _get_paths(lora_type)
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

        _model_tokenizer[lora_type] = (model, tokenizer)
        return _model_tokenizer[lora_type]


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


# 空气小猪推理时强制注入的系统提示，避免基座/LoRA 把「空气小猪」联想成阿里云、空调、游戏等
AIRPIG_SYSTEM_PROMPT = (
    "你是空气小猪的客服助手。空气小猪是一款以即时通讯为核心的语言环境产品，"
    "帮助用户把日常聊天内容转换成自己正在学习的目标语言并朗读，形成长期外语学习环境。"
    "重要：空气小猪不是阿里云/分布式文件系统，不是空调/制冷/环保设备，不是游戏或手游。"
    "请仅根据上述产品定义回答。若问「解决什么核心问题」，应回答：学外语无法进入日常生活、输入输出割裂、材料与生活无关难以坚持等；"
    "空气小猪通过重用已有聊天内容，为用户建立长期、低成本、真实相关的外语环境。"
)


# 法律/医疗 LoRA 训练时用的 prompt，推理必须一致，否则答非所问
LEGAL_PROMPT_TEMPLATE = """你是一个专业的法律咨询助手。请根据用户的问题提供专业、准确的法律建议。

### 问题：
{}

### 回答：
"""

MEDICAL_PROMPT_TEMPLATE = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
"""

# 法律 LoRA 有时会漂移到医疗话术，检测到下列片段时截断，保留前半段法律相关回复
LEGAL_DRIFT_PHRASES = (
    "其实得了", "并不可怕", "战胜病魔", "对症治疗", "缓解病情", "患者要相信",
    "外科疾病", "及时发现症状", "康复的几率",
)


def _truncate_legal_drift(text):
    """若法律回复中出现典型医疗漂移用语，截断到该处之前（保留最后一个完整句）。"""
    if not text or not isinstance(text, str):
        return text
    first = len(text)
    for phrase in LEGAL_DRIFT_PHRASES:
        i = text.find(phrase)
        if i != -1 and i < first:
            first = i
    if first >= len(text):
        return text
    out = text[:first].rstrip()
    # 尽量在句号处截断
    last_period = out.rfind("。")
    if last_period != -1:
        out = out[: last_period + 1]
    return out.strip() or text[:first].strip()


def _prompt_for_sft_lora(lora_type, messages):
    """legal/medical 推理时用与训练一致的 prompt 格式，避免与 apply_chat_template 不一致导致答非所问。"""
    if lora_type not in ("legal", "medical"):
        return None
    # 取最后一条用户内容作为问题（或拼接所有 user 内容）
    parts = [m.get("content", "") or "" for m in messages if (m.get("role") or "").strip().lower() == "user"]
    question = " ".join(parts).strip() or (messages[-1].get("content") if messages else "") or ""
    if lora_type == "legal":
        return LEGAL_PROMPT_TEMPLATE.format(question)
    if lora_type == "medical":
        return MEDICAL_PROMPT_TEMPLATE.format(question)
    return None


def _inject_airpig_system(messages):
    """当使用空气小猪 LoRA 时，在 messages 前注入产品定义，避免模型幻觉成游戏等。"""
    if not messages or not isinstance(messages, list):
        return messages
    inject = {"role": "system", "content": AIRPIG_SYSTEM_PROMPT}
    first = (messages[0].get("role") or "").strip().lower()
    if first == "system":
        # 已有 system 时插在原有 system 之后、user 之前，保证产品定义优先
        return [messages[0], inject] + list(messages[1:])
    return [inject] + list(messages)


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
                x.get("text", str(x))
                for x in content
                if isinstance(x, dict) and "text" in x
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
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(
        tokenizer, "eos_token_id"
    )
    return {
        "max_new_tokens": int(opts.get("num_predict", opts.get("max_new_tokens", 512))),
        "temperature": float(opts.get("temperature", 0.7)),
        "top_p": float(opts.get("top_p", opts.get("top_p", 0.9))),
        "repetition_penalty": float(opts.get("repeat_penalty", 1.1)),
        "do_sample": True,
        "pad_token_id": pad_id,
    }


def chat(messages, stream=False, options=None, lora_type="medical"):
    """
    使用微调模型生成回复。

    :param messages: [{"role":"user"|"assistant"|"system","content":"..."}, ...]
    :param stream: 是否流式返回
    :param options: 可选 {"temperature", "num_predict", "top_p", "repeat_penalty"}
    :param lora_type: "medical" | "legal" | "airpig"，选择医疗、法律或空气小猪客服 LoRA
    :return: stream=False 时返回完整 content 字符串；stream=True 时返回生成器，yield 文本块。
    """
    if lora_type == "airpig":
        messages = _inject_airpig_system(messages)
    model, tokenizer = get_model(lora_type=lora_type)
    prompt = _prompt_for_sft_lora(lora_type, messages)
    if prompt is None:
        prompt = _messages_to_input(messages, tokenizer)
    gen_kw = _gen_kwargs(options, tokenizer)

    import torch

    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items() if k != "token_type_ids"}

    if stream:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
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
    if lora_type == "legal":
        text = _truncate_legal_drift(text)
    yield text


def chat_sync(messages, options=None):
    """非流式：返回完整回复字符串（仅 LoRA 模型）。"""
    gen = chat(messages, stream=False, options=options)
    return next(gen, "")


def chat_sync_compare(messages, options=None, lora_type="medical"):
    """
    非流式：同时用 base（1.5B 原版）与 base+LoRA（医疗/法律等）各生成一次，返回两段回复便于对比。
    :param lora_type: "medical" | "legal" | "airpig"
    :return: {"base": "基座回复", "lora": "LoRA 回复"}
    """
    if lora_type == "airpig":
        messages = _inject_airpig_system(messages)
    lora_model, tokenizer = get_model(lora_type=lora_type)
    base_model, _ = get_base_model()
    prompt = _prompt_for_sft_lora(lora_type, messages)
    if prompt is None:
        prompt = _messages_to_input(messages, tokenizer)
    gen_kw = _gen_kwargs(options, tokenizer)
    base_text = _run_one_model(base_model, tokenizer, prompt, gen_kw)
    lora_text = _run_one_model(lora_model, tokenizer, prompt, gen_kw)
    if lora_type == "legal":
        lora_text = _truncate_legal_drift(lora_text)
    return {"base": base_text, "lora": lora_text}


def chat_stream(messages, options=None, lora_type="medical"):
    """流式：返回生成器，逐个 yield 文本块（仅 LoRA）。"""
    return chat(messages, stream=True, options=options, lora_type=lora_type)


def chat_stream_compare(messages, options=None, lora_type="medical"):
    """
    流式：base 与 LoRA 两路并行生成，按到达顺序 yield (source, chunk)，source 为 "base" 或 "lora"。
    前端可同时往两栏追加内容。
    :param lora_type: "medical" | "legal" | "airpig"
    """
    import torch
    from transformers import TextIteratorStreamer

    if lora_type == "airpig":
        messages = _inject_airpig_system(messages)
    base_model, tokenizer = get_base_model()
    lora_model, _ = get_model(lora_type=lora_type)
    prompt = _prompt_for_sft_lora(lora_type, messages)
    if prompt is None:
        prompt = _messages_to_input(messages, tokenizer)
    gen_kw = _gen_kwargs(options, tokenizer)
    gen_kw["max_new_tokens"] = gen_kw.get("max_new_tokens", 512)

    def make_enc(device):
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
        return enc

    enc_base = make_enc(base_model.device)
    enc_lora = make_enc(lora_model.device)
    base_streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    lora_streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
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


# 前端「选择 LORA 模型」列表：id 用于请求体 lora/model_type，name/tag 用于展示
LORA_OPTIONS = [
    {"id": "medical", "name": "医疗问诊 LoRA", "tag": "医疗"},
    {"id": "legal", "name": "法律咨询 LoRA", "tag": "法律"},
    {"id": "airpig", "name": "空气小猪客服助手", "tag": "客服"},
]


def list_lora_options_api():
    """GET /ai/finetuning/lora-options：返回可选 LoRA 列表，供前端选择模型。"""
    from flask import jsonify

    return jsonify({"code": 0, "msg": "ok", "data": LORA_OPTIONS})


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
    # 前端选择 LoRA：medical=医疗问诊，legal=法律咨询，airpig=空气小猪客服
    lora_type = (
        (data.get("lora") or data.get("model_type") or "medical").strip().lower()
    )
    if lora_type not in LORA_TYPE_DIRS:
        lora_type = "medical"

    try:
        if stream:
            if compare:
                # 同时流式输出 base 与 lora 两路，事件格式 {"source": "base"|"lora", "content": "..."} 或 {"source": "...", "done": true}
                def generate_compare():
                    try:
                        for source, chunk in chat_stream_compare(
                            messages, options=options, lora_type=lora_type
                        ):
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
                        for chunk in chat_stream(
                            messages, options=options, lora_type=lora_type
                        ):
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
        result = chat_sync_compare(messages, options=options, lora_type=lora_type)
        _, lora_path_used = _get_paths(lora_type)
        return jsonify(
            {
                "code": 0,
                "lora_type": lora_type,
                "lora_path_used": lora_path_used,
                "message": {
                    "role": "assistant",
                    "content": result.get("lora") or "",
                    "base_content": result.get("base") or "",
                    "lora_content": result.get("lora") or "",
                },
                "base": result.get("base") or "",
                "lora": result.get("lora") or "",
            }
        )
    except FileNotFoundError as e:
        return jsonify({"code": 404, "msg": str(e)}), 404
    except ValueError as e:
        return jsonify({"code": 400, "msg": str(e)}), 400
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500
