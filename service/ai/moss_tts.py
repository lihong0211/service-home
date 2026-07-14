"""
MOSS-TTS-Nano 直接集成（ONNX 后端）
在 service-home 进程内运行，无需独立子服务。
首次请求时自动加载模型（~400MB，从 HuggingFace 缓存下载）。
"""
from __future__ import annotations

import os
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

import anyio
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── 单例 ──────────────────────────────────────────────────────────────────────
_runtime = None
_runtime_error: Optional[str] = None
_init_lock = threading.Lock()


def _get_runtime():
    """懒加载 OnnxTtsRuntime 单例（线程安全，自动下载 ONNX 模型）"""
    global _runtime, _runtime_error
    if _runtime is not None:
        return _runtime
    if _runtime_error is not None:
        raise RuntimeError(_runtime_error)
    with _init_lock:
        if _runtime is not None:
            return _runtime
        if _runtime_error is not None:
            raise RuntimeError(_runtime_error)
        try:
            from onnx_tts_runtime import OnnxTtsRuntime  # pip install -e MOSS-TTS-Nano/
            _runtime = OnnxTtsRuntime(thread_count=os.cpu_count() or 4)
        except ImportError:
            _runtime_error = (
                "MOSS-TTS-Nano 未安装。请运行：\n"
                "  git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git\n"
                "  pip install -e MOSS-TTS-Nano/ --no-deps\n"
                "  pip install torchaudio"
            )
            raise RuntimeError(_runtime_error)
        except Exception as e:
            _runtime_error = f"MOSS-TTS-Nano 初始化失败: {e}"
            raise RuntimeError(_runtime_error)
    return _runtime


def preload() -> None:
    """在 FastAPI lifespan 后台预热（首次调用会下载 ONNX 模型，约 400MB）"""
    try:
        _get_runtime()
        print("[MOSS-TTS] ONNX 模型加载完成", flush=True)
    except Exception as e:
        print(f"[MOSS-TTS] 预热失败（将在首次请求时重试）: {e}", flush=True)


# ── 推理（同步，跑在线程池里） ────────────────────────────────────────────────

def _do_synthesize(text: str, prompt_audio_path: Optional[str]) -> bytes:
    import wave
    import numpy as np

    runtime = _get_runtime()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        result = runtime.synthesize(
            text=text,
            prompt_audio_path=prompt_audio_path,
            output_audio_path=out_path,
            enable_wetext=False,       # 跳过 WeTextProcessing / pynini
            enable_normalize_tts_text=True,
        )
        # 优先读文件（已完整写入）
        wav_path = result.get("audio_path")
        if wav_path and Path(wav_path).exists():
            return Path(wav_path).read_bytes()

        # fallback：从 waveform numpy 构造 WAV
        waveform: np.ndarray = result["waveform"]
        sample_rate: int = result["sample_rate"]
        buf = BytesIO()
        n_channels = 2 if (waveform.ndim == 2 and waveform.shape[0] == 2) else 1
        if n_channels == 2:
            pcm = waveform.T.reshape(-1)
        else:
            pcm = waveform.reshape(-1)
        pcm16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        buf.seek(0)
        return buf.read()


# ── FastAPI 路由处理函数 ───────────────────────────────────────────────────────

async def moss_tts_speech_api(request: Request):
    """
    POST /ai/moss-tts/speech
    multipart/form-data 或 application/json:
      - text: str           必填
      - prompt_audio: file  可选，WAV 参考音频（声音克隆）
    返回 audio/wav
    """
    content_type = (request.headers.get("content-type") or "").lower()
    text: Optional[str] = None
    prompt_audio_bytes: Optional[bytes] = None
    prompt_audio_filename = "reference.wav"

    if "multipart/form-data" in content_type:
        form = await request.form()
        text = form.get("text")
        audio_file = form.get("prompt_audio")
        if audio_file and hasattr(audio_file, "read"):
            prompt_audio_bytes = await audio_file.read()
            prompt_audio_filename = getattr(audio_file, "filename", None) or "reference.wav"
    else:
        try:
            data = await request.json()
        except Exception:
            data = {}
        text = data.get("text")

    if not text or not str(text).strip():
        return JSONResponse(content={"code": 400, "msg": "缺少 text 参数"}, status_code=400)

    text = str(text).strip()
    prompt_tmp: Optional[str] = None

    try:
        if prompt_audio_bytes:
            suffix = Path(prompt_audio_filename).suffix or ".wav"
            fd, prompt_tmp = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            Path(prompt_tmp).write_bytes(prompt_audio_bytes)

        # CPU 推理在线程池执行，不阻塞事件循环
        wav_bytes = await anyio.to_thread.run_sync(
            lambda: _do_synthesize(text, prompt_tmp),
            cancellable=True,
        )

        return StreamingResponse(
            BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="speech.wav"'},
        )
    except RuntimeError as e:
        return JSONResponse(content={"code": 503, "msg": str(e)}, status_code=503)
    except Exception as e:
        return JSONResponse(content={"code": 500, "msg": str(e)}, status_code=500)
    finally:
        if prompt_tmp:
            try:
                os.unlink(prompt_tmp)
            except Exception:
                pass


async def moss_tts_status_api(request: Request):
    """GET /ai/moss-tts/status"""
    if _runtime is not None:
        return {"online": True, "backend": "onnx", "msg": "模型已加载，就绪"}
    if _runtime_error is not None:
        return {"online": False, "backend": "onnx", "msg": _runtime_error}
    return {
        "online": False,
        "backend": "onnx",
        "msg": "模型尚未加载（首次请求时自动触发，约需 30-60s）",
    }
