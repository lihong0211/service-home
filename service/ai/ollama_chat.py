"""
Ollama 聊天服务 - 封装本地 Ollama API，支持流式输出
"""

import json
import re
import requests
from flask import request, jsonify, Response, stream_with_context

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "my-deepseek-r1-1.5"
# 有图时可选：ocr 文字识别 / vl 视觉理解（多图、描述等）
OCR_MODEL = "deepseek-ocr:latest"
VL_MODEL = "qwen3-vl:2b"

# OCR 结果开头常见 LaTeX/模板噪声（如 <\begin、\begin{...}），识别后去掉
OCR_STRIP_PREFIX = re.compile(
    r"^[\s]*(?:<\\begin|\\\\begin|\\begin(?:\{[^}]*\})?)[\s]*",
    re.IGNORECASE,
)


def _dedupe_vision_content(text):
    """识图结果去重：合并连续重复的同一行，避免模型反复输出同一句话。"""
    if not text or not text.strip():
        return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    deduped = []
    for ln in lines:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)
    return "\n".join(deduped)


def chat():
    """Ollama 聊天接口

    请求格式：
    {
        "messages": [{"role": "user", "content": "你好", "images": ["<base64>"]}],
        "images": ["<base64>"],   // 可选，与 messages 二选一或合并到最后一条 user
        "model": "模型名，可选",
        "vision": "ocr" | "vl",  // 有图且未指定 model 时：ocr=文字识别，vl=视觉理解(默认 vl)
        "stream": true,
        "options": {"temperature": 0.45, "num_predict": 2048}
    }

    响应格式：
    - stream=false: {"code": 0, "message": {"role": "assistant", "content": "..."}}
    - stream=true: SSE 流，每行 data: {"message": {"content": "增量", "thinking": "..."}, "done": false}
    """
    data = request.get_json(silent=True) or {}
    if not data and request.get_data():
        return jsonify({"code": 400, "msg": "Invalid JSON or body too large"}), 400

    messages = data.get("messages")
    # 兼容顶层 image（单图）或 images（多图）
    top_level_images = data.get("images")
    if top_level_images is None and data.get("image") is not None:
        top_level_images = (
            [data["image"]] if isinstance(data["image"], str) else data["image"]
        )
    if top_level_images is not None and not isinstance(top_level_images, list):
        top_level_images = [top_level_images] if top_level_images else []

    model = data.get("model") or DEFAULT_MODEL
    stream = data.get("stream", False)
    options = data.get("options", {"temperature": 0.45, "num_predict": 2048})

    if not messages or not isinstance(messages, list):
        return jsonify({"code": 400, "msg": "Missing or invalid messages"}), 400

    # 深拷贝并统一 message 内 image(s) 为 images 数组
    messages = []
    for m in data.get("messages", []):
        msg = dict(m)
        imgs = msg.get("images")
        if imgs is None and msg.get("image") is not None:
            imgs = [msg["image"]] if isinstance(msg["image"], str) else msg["image"]
        if imgs is not None:
            msg["images"] = imgs if isinstance(imgs, list) else [imgs]
        messages.append(msg)

    # 顶层 images 合并到最后一条 user 消息（支持前端只传 images 不塞进 message）
    if top_level_images:
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            msg = messages[last_user_idx]
            msg["images"] = list(top_level_images) + list(msg.get("images") or [])
        else:
            messages.append(
                {"role": "user", "content": "", "images": list(top_level_images)}
            )

    # 有图且未显式指定 model 时：按 vision 选 OCR 或 VL
    has_images = bool(top_level_images) or any(m.get("images") for m in messages)
    if has_images and not data.get("model"):
        vision = (data.get("vision") or "vl").lower()
        model = VL_MODEL if vision == "vl" else OCR_MODEL

    # 识图请求（OCR/VL）：结束后让 Ollama 卸载模型以释放显存
    is_vision = has_images or "ocr" in model.lower() or "vl" in model.lower()
    keep_alive = 0 if is_vision else None
    is_ocr = "ocr" in model.lower()  # 仅 OCR 做 LaTeX 清洗
    # 识图时提高 repeat_penalty，减少同一句话反复输出
    if is_vision and isinstance(options, dict):
        options = dict(options)
        options.setdefault("repeat_penalty", 1.25)
    try:
        if stream:
            return _stream_chat(model, messages, options, keep_alive, is_ocr, is_vision)
        return _sync_chat(model, messages, options, keep_alive, is_ocr, is_vision)
    except requests.exceptions.ConnectionError:
        return jsonify({"code": 503, "msg": "Ollama service not running"}), 503
    except requests.exceptions.Timeout:
        return jsonify({"code": 504, "msg": "Ollama request timeout"}), 504
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def ocr_chat():
    """专用 OCR 接口：固定使用 OCR_MODEL，请求格式同 chat（需带图）。"""
    data = request.get_json(silent=True) or {}
    if not data and request.get_data():
        return jsonify({"code": 400, "msg": "Invalid JSON or body too large"}), 400

    messages = data.get("messages")
    top_level_images = data.get("images")
    if top_level_images is None and data.get("image") is not None:
        top_level_images = (
            [data["image"]] if isinstance(data["image"], str) else data["image"]
        )
    if top_level_images is not None and not isinstance(top_level_images, list):
        top_level_images = [top_level_images] if top_level_images else []

    stream = data.get("stream", False)
    options = data.get("options", {"temperature": 0.45, "num_predict": 2048})
    if isinstance(options, dict):
        options = dict(options)
        options.setdefault("repeat_penalty", 1.25)

    if not messages or not isinstance(messages, list):
        return jsonify({"code": 400, "msg": "Missing or invalid messages"}), 400

    messages = []
    for m in data.get("messages", []):
        msg = dict(m)
        imgs = msg.get("images")
        if imgs is None and msg.get("image") is not None:
            imgs = [msg["image"]] if isinstance(msg["image"], str) else msg["image"]
        if imgs is not None:
            msg["images"] = imgs if isinstance(imgs, list) else [imgs]
        messages.append(msg)

    if top_level_images:
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            msg = messages[last_user_idx]
            msg["images"] = list(top_level_images) + list(msg.get("images") or [])
        else:
            messages.append(
                {"role": "user", "content": "", "images": list(top_level_images)}
            )

    has_images = bool(top_level_images) or any(m.get("images") for m in messages)
    if not has_images:
        return (
            jsonify(
                {
                    "code": 400,
                    "msg": "OCR requires images in messages or top-level images",
                }
            ),
            400,
        )

    try:
        if stream:
            return _stream_chat(
                OCR_MODEL, messages, options, keep_alive=0, is_ocr=True, is_vision=True
            )
        return _sync_chat(
            OCR_MODEL, messages, options, keep_alive=0, is_ocr=True, is_vision=True
        )
    except requests.exceptions.ConnectionError:
        return jsonify({"code": 503, "msg": "Ollama service not running"}), 503
    except requests.exceptions.Timeout:
        return jsonify({"code": 504, "msg": "Ollama request timeout"}), 504
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def _sync_chat(
    model, messages, options, keep_alive=None, is_ocr=False, is_vision=False
):
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    message = result.get("message", {})
    if message.get("content"):
        message = dict(message)
        if is_ocr:
            message["content"] = OCR_STRIP_PREFIX.sub("", message["content"]).strip()
        if is_vision:
            message["content"] = _dedupe_vision_content(message["content"])
    return Response(
        json.dumps({"code": 0, "message": message}, ensure_ascii=False),
        mimetype="application/json; charset=utf-8",
    )


def _stream_chat(
    model, messages, options, keep_alive=None, is_ocr=False, is_vision=False
):
    def generate():
        try:
            body = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": options,
            }
            if keep_alive is not None:
                body["keep_alive"] = keep_alive
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=body,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"
            first_content_chunk = True
            # 流式识图去重：累积全文并按行去重，只下发「相对上一帧新增」的非重复部分
            accumulated = ""
            prev_deduped = ""
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    chunk = json.loads(line)
                    msg = chunk.get("message", {})
                    out = dict(chunk)
                    content = msg.get("content", "")
                    thinking = msg.get("thinking", "")
                    if is_ocr and first_content_chunk and (content or thinking):
                        if content:
                            content = OCR_STRIP_PREFIX.sub("", content).lstrip()
                        first_content_chunk = False
                    if is_vision and content:
                        accumulated += content
                        deduped = _dedupe_vision_content(accumulated)
                        content = deduped[len(prev_deduped) :]
                        prev_deduped = deduped
                    out.setdefault("response", content)
                    out.setdefault("thinking", thinking)
                    yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
