"""
文字转语音 (TTS) 服务 - 基于 Edge-TTS（免费、免 key，质量好）
生成音频文件并返回给前端播放/下载
"""

import asyncio
import os
import tempfile
from io import BytesIO

from flask import request, send_file, jsonify

# 默认中文女声
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "zh-CN-XiaoxiaoNeural")


async def _generate_mp3(text: str, voice: str, path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)


def speech():
    """
    文字转语音接口

    请求：application/json，body: { "text": "要读的文字", "voice": "zh-CN-..." 可选 }
    响应：audio/mpeg 文件（默认女声，可直接播放或下载）
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text") or (request.form and request.form.get("text"))
    voice = data.get("voice") or request.form.get("voice") or DEFAULT_VOICE

    if not text or not str(text).strip():
        return jsonify({"code": 400, "msg": "Missing text"}), 400

    text = str(text).strip()
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_generate_mp3(text, voice, tmp))
        finally:
            loop.close()
        with open(tmp, "rb") as f:
            buf = BytesIO(f.read())
        try:
            os.unlink(tmp)
        except Exception:
            pass
        buf.seek(0)
        return send_file(
            buf,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="speech.mp3",
        )
    except Exception as e:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return jsonify({"code": 500, "msg": str(e)}), 500
