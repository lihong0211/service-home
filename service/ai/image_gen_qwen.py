"""
文生图 - Qwen-Image-2512 + Wuli-Art Turbo LoRA（2 步出图）
基于 DiffSynth-Engine。仅保留代码，不对外提供 HTTP 接口。
"""

import math
import os
import tempfile
from io import BytesIO

import torch
from flask import request, send_file, jsonify

from diffsynth_engine import (
    fetch_model,
    QwenImagePipeline,
    QwenImagePipelineConfig,
)

_image_pipe_qwen = None


def _get_pipeline():
    global _image_pipe_qwen
    if _image_pipe_qwen is None:
        use_cuda = torch.cuda.is_available()
        device = os.environ.get("IMAGE_DEVICE", "cuda" if use_cuda else "cpu")
        offload = (
            None if device == "cpu" else os.environ.get("IMAGE_OFFLOAD", "cpu_offload")
        )
        config = QwenImagePipelineConfig.basic_config(
            model_path=fetch_model(
                "Qwen/Qwen-Image-2512", path="transformer/*.safetensors"
            ),
            encoder_path=fetch_model(
                "Qwen/Qwen-Image-2512", path="text_encoder/*.safetensors"
            ),
            vae_path=fetch_model("Qwen/Qwen-Image-2512", path="vae/*.safetensors"),
            device=device,
            offload_mode=offload,
        )
        _image_pipe_qwen = QwenImagePipeline.from_pretrained(config)
        _image_pipe_qwen.load_lora(
            path=fetch_model(
                "Wuli-Art/Qwen-Image-2512-Turbo-LoRA-2-Steps",
                path="Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors",
            ),
            scale=1.0,
            fused=True,
        )
        _image_pipe_qwen.apply_scheduler_config(
            {
                "exponential_shift_mu": math.log(2.5),
                "use_dynamic_shifting": True,
                "shift_terminal": 0.7155,
            }
        )
    return _image_pipe_qwen


def generate_qwen():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt") or (request.form and request.form.get("prompt"))
    if not prompt or not str(prompt).strip():
        return jsonify({"code": 400, "msg": "Missing prompt"}), 400

    prompt = str(prompt).strip()
    width = _int_or_default(
        data.get("width"),
        request.form.get("width") if request.form else None,
        1024,
        512,
        2048,
    )
    height = _int_or_default(
        data.get("height"),
        request.form.get("height") if request.form else None,
        1024,
        512,
        2048,
    )
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None
    num_inference_steps = _int_or_default(
        data.get("num_inference_steps"), None, 2, 1, 50
    )
    cfg_scale = _float_or_default(data.get("cfg_scale"), None, 1.0)

    tmp = None
    try:
        pipe = _get_pipeline()
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        out = pipe(
            prompt=prompt,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            width=width,
            height=height,
        )
        out.save(tmp)
        with open(tmp, "rb") as f:
            buf = BytesIO(f.read())
        try:
            os.unlink(tmp)
        except Exception:
            pass
        buf.seek(0)
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=False,
            download_name="image.png",
        )
    except Exception as e:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return jsonify({"code": 500, "msg": str(e)}), 500


def _int_or_default(val, form_val, default, min_val, max_val):
    v = val if val is not None else form_val
    if v is None:
        return default
    try:
        v = int(v)
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return default


def _float_or_default(val, form_val, default):
    v = val if val is not None else form_val
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
