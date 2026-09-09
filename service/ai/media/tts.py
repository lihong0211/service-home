"""
文字转语音 (TTS) 服务 - 基于 Edge-TTS（免费、免 key，质量好）
生成音频文件并返回给前端播放/下载
"""

import asyncio
import os
import tempfile
from io import BytesIO

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

# 默认中文女声
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "zh-CN-XiaoxiaoNeural")


async def _generate_mp3(text: str, voice: str, path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)


async def speech(request: Request):
    """
    文字转语音接口

    请求：application/json，body: { "text": "要读的文字", "voice": "zh-CN-..." 可选 }
    响应：audio/mpeg 文件（默认女声，可直接播放或下载）
    """
    content_type = (request.headers.get("content-type") or "").lower()
    data = {}
    form = {}
    if "application/json" in content_type:
        try:
            data = await request.json()
        except Exception:
            data = {}
    else:
        try:
            form = await request.form()
        except Exception:
            form = {}

    text = data.get("text") or form.get("text")
    voice = data.get("voice") or form.get("voice") or DEFAULT_VOICE

    if not text or not str(text).strip():
        raise ValueError("Missing text")

    text = str(text).strip()
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        await _generate_mp3(text, voice, tmp)
        with open(tmp, "rb") as f:
            buf = BytesIO(f.read())
        try:
            os.unlink(tmp)
        except Exception:
            pass
        buf.seek(0)
        # inline 播放；前端可自行决定是否下载
        return StreamingResponse(
            buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speech.mp3"'},
        )
    except Exception as e:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return JSONResponse(
            content={"code": 500, "msg": str(e)},
            status_code=500,
        )
