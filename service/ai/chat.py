"""
Ollama 聊天服务 - 封装本地 Ollama API，支持流式输出
"""

import json
import re
import requests
from flask import request, jsonify, Response, stream_with_context

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "my-deepseek-r1-1.5"
OCR_MODEL = "deepseek-ocr:latest"
VL_MODEL = "qwen3-vl:2b"

# OCR 结果开头常见 LaTeX/模板噪声（如 <\begin、\begin{...}），识别后去掉
OCR_STRIP_PREFIX = re.compile(
    r"^[\s]*(?:<\\begin|\\\\begin|\\begin(?:\{[^}]*\})?)[\s]*",
    re.IGNORECASE,
)


def _collapse_repeated_phrase(s):
    """若整段是同一短语重复多次，只保留一次。如 'ABABAB' -> 'AB'"""
    if not s or len(s) < 2:
        return s
    n = len(s)
    for period in range(1, n // 2 + 1):
        if n % period != 0:
            continue
        if s == s[:period] * (n // period):
            return s[:period]
    return s


def _dedupe_vision_content(text):
    """识图结果去重：合并连续重复的同一行 + 行内重复短语。"""
    if not text or not text.strip():
        return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    deduped = []
    for ln in lines:
        ln = _collapse_repeated_phrase(ln)
        if not ln:
            continue
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)
    return "\n".join(deduped)


def chat():
    """Ollama 对话接口（纯文本/对话，不处理图片）

    请求：{ "messages": [...], "model": "可选", "stream": true, "options": {} }
    响应：stream=false 时 { "code": 0, "message": {...} }；stream=true 时 SSE
    """
    data = request.get_json(silent=True) or {}
    if not data and request.get_data():
        return jsonify({"code": 400, "msg": "Invalid JSON or body too large"}), 400

    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"code": 400, "msg": "Missing or invalid messages"}), 400

    model = data.get("model") or DEFAULT_MODEL
    stream = data.get("stream", False)
    options = data.get("options", {"temperature": 0.45, "num_predict": 2048})

    try:
        if stream:
            return _stream_chat(
                model, messages, options, keep_alive=None, is_ocr=False, is_vision=False
            )
        return _sync_chat(
            model, messages, options, keep_alive=None, is_ocr=False, is_vision=False
        )
    except requests.exceptions.ConnectionError:
        return jsonify({"code": 503, "msg": "Ollama service not running"}), 503
    except requests.exceptions.Timeout:
        return jsonify({"code": 504, "msg": "Ollama request timeout"}), 504
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def ocr_chat():
    """专用 OCR 接口：固定 OCR_MODEL，只收图。请求：{ "images": ["<base64>"] 或 "image": "<base64>", "stream": true, "options": {} }"""
    data = request.get_json(silent=True) or {}
    if not data and request.get_data():
        return jsonify({"code": 400, "msg": "Invalid JSON or body too large"}), 400

    images = data.get("images")
    if images is None and data.get("image") is not None:
        images = [data["image"]] if isinstance(data["image"], str) else data["image"]
    if not images:
        return jsonify({"code": 400, "msg": "Missing images or image"}), 400
    if not isinstance(images, list):
        images = [images]

    stream = data.get("stream", False)
    options = data.get(
        "options", {"temperature": 0.45, "num_predict": 1024, "repeat_penalty": 1.35}
    )
    if isinstance(options, dict):
        options = dict(options)
        options.setdefault("repeat_penalty", 1.35)
        options.setdefault("num_predict", 1024)

    messages = [{"role": "user", "content": "识别图中文字", "images": list(images)}]

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
