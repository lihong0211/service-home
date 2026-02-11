"""
语音转文字 (STT) 服务 - 基于 faster-whisper
支持文件上传或 base64 音频，返回转录文本与可选时间戳
实时：流式 SSE 按段返回 / WebSocket 持续对讲
"""

import base64
import json
import os
import tempfile
from flask import request, jsonify, Response, stream_with_context

# 懒加载，首次请求时再加载模型
_whisper_model = None

# 默认配置（Mac 可改为 device="cpu", compute_type="int8"）
STT_CONFIG = {
    "model_size": os.environ.get(
        "STT_MODEL", "base"
    ),  # tiny, base, small, medium, large-v2, large-v3
    "device": os.environ.get("STT_DEVICE", "cpu"),
    "compute_type": os.environ.get(
        "STT_COMPUTE_TYPE", "int8"
    ),  # CPU 用 int8，GPU 用 float16
    "language": None,  # None=自动检测，或 "zh"、"en" 等
    "vad_filter": True,
    "beam_size": 5,
}


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            STT_CONFIG["model_size"],
            device=STT_CONFIG["device"],
            compute_type=STT_CONFIG["compute_type"],
        )
    return _whisper_model


def transcribe():
    """
    语音转文字接口

    请求方式一：multipart/form-data，字段名 file 或 audio
    请求方式二：application/json，body: { "audio_base64": "<base64>", "language": "zh" 可选 }

    响应：{ "code": 0, "text": "完整文本", "language": "zh", "segments": [{ "start", "end", "text" }] }
    """
    audio_path = None
    language = request.form.get("language") or (
        request.get_json(silent=True) or {}
    ).get("language")
    language = language or STT_CONFIG["language"]

    try:
        # 1. 文件上传
        if request.files:
            f = request.files.get("file") or request.files.get("audio")
            if not f or not f.filename:
                return (
                    jsonify({"code": 400, "msg": "Missing file or audio in form"}),
                    400,
                )
            suffix = os.path.splitext(f.filename)[1] or ".webm"
            fd, audio_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            f.save(audio_path)
        # 2. base64
        elif request.is_json:
            data = request.get_json()
            b64 = data.get("audio_base64") or data.get("audio")
            if not b64:
                raise ValueError("Missing audio_base64 or audio in body")
            raw = base64.b64decode(b64)
            suffix = ".wav"
            if isinstance(b64, str) and b64.startswith("data:"):
                # data:audio/webm;base64,xxx
                raw = base64.b64decode(b64.split(",", 1)[-1])
                if "webm" in (data.get("audio_base64") or "")[:50]:
                    suffix = ".webm"
            fd, audio_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(audio_path, "wb") as out:
                out.write(raw)
        else:
            raise ValueError("Send multipart file or JSON with audio_base64")

        model = _get_model()
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=STT_CONFIG["vad_filter"],
            beam_size=STT_CONFIG["beam_size"],
        )
        segments_list = [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments_iter
        ]
        text = "".join(s["text"] for s in segments_list).strip()

        lang = getattr(info, "language", None)
        lang_prob = getattr(info, "language_probability", None)
        return jsonify(
            {
                "code": 0,
                "text": text,
                "language": lang,
                "language_probability": lang_prob,
                "segments": segments_list,
            }
        )
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception:
                pass


def _get_audio_path_from_request():
    """从请求中解析出临时音频文件路径，调用方负责删除。"""
    if request.files:
        f = request.files.get("file") or request.files.get("audio")
        if not f or not f.filename:
            return None, "Missing file or audio"
        suffix = os.path.splitext(f.filename)[1] or ".webm"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        f.save(path)
        return path, None
    if request.is_json:
        data = request.get_json()
        b64 = data.get("audio_base64") or data.get("audio")
        if not b64:
            return None, "Missing audio_base64 or audio"
        raw = base64.b64decode(
            b64.split(",", 1)[-1]
            if isinstance(b64, str) and b64.startswith("data:")
            else b64
        )
        suffix = ".webm" if (isinstance(b64, str) and "webm" in b64[:80]) else ".wav"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(path, "wb") as out:
            out.write(raw)
        return path, None
    return None, "Send multipart file or JSON with audio_base64"


def transcribe_stream():
    """
    实时流式转录：一次上传一段音频，按识别出的片段 SSE 推送。
    请求同 transcribe（file 或 audio_base64），响应为 SSE：
    data: {"text": "...", "start": 0.0, "end": 1.2}
    data: [DONE]
    """
    language = (
        request.form.get("language")
        or (request.get_json(silent=True) or {}).get("language")
        or STT_CONFIG["language"]
    )

    audio_path, err = _get_audio_path_from_request()
    if err:
        raise ValueError(err)

    def generate():
        try:
            model = _get_model()
            segments_iter, info = model.transcribe(
                audio_path,
                language=language,
                vad_filter=STT_CONFIG["vad_filter"],
                beam_size=STT_CONFIG["beam_size"],
            )
            lang = getattr(info, "language", None)
            yield f"data: {json.dumps({'language': lang}, ensure_ascii=False)}\n\n"
            for s in segments_iter:
                yield f"data: {json.dumps({'text': s.text, 'start': s.start, 'end': s.end}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def register_stt_ws(sock):
    """注册 WebSocket：/ai/stt/live，客户端持续发音频 chunk（base64），服务端按段回传文本。"""

    @sock.route("/ai/stt/live")
    def live(ws):
        language = None
        while True:
            msg = ws.receive()
            if msg is None:
                break
            # 支持 JSON { "audio_base64": "...", "language": "zh" } 或 纯 base64 字符串
            if isinstance(msg, str) and msg.strip().startswith("{"):
                try:
                    data = json.loads(msg)
                    b64 = data.get("audio_base64") or data.get("audio")
                    language = (
                        data.get("language") or language or STT_CONFIG["language"]
                    )
                except Exception:
                    ws.send(json.dumps({"error": "Invalid JSON"}))
                    continue
            else:
                b64 = msg
            if not b64:
                ws.send(json.dumps({"error": "Missing audio"}))
                continue
            audio_path = None
            try:
                raw = base64.b64decode(
                    b64.split(",", 1)[-1]
                    if isinstance(b64, str) and "base64," in b64
                    else b64
                )
                fd, audio_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                with open(audio_path, "wb") as f:
                    f.write(raw)
                model = _get_model()
                segments_iter, _ = model.transcribe(
                    audio_path,
                    language=language,
                    vad_filter=STT_CONFIG["vad_filter"],
                    beam_size=3,
                )
                text = "".join(s.text for s in segments_iter).strip()
                if text:
                    ws.send(json.dumps({"text": text}, ensure_ascii=False))
            except Exception as e:
                ws.send(json.dumps({"error": str(e)}))
            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass
