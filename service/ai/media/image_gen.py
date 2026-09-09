"""
文生图 (Text-to-Image) 服务 - Stable Diffusion XL Base 1.0
基于 diffusers，固定使用 MPS（Apple Silicon）。
"""

import os
import tempfile
from io import BytesIO

import torch
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from utils.http_body import read_json_optional, read_form_optional, is_json_request

_image_pipe = None
DEVICE = torch.device("mps")
TORCH_DTYPE = torch.float32


def _get_pipeline():
    global _image_pipe
    if _image_pipe is None:
        from diffusers import StableDiffusionXLPipeline

        _image_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True,
        )
        _image_pipe.enable_attention_slicing()
        _image_pipe = _image_pipe.to(DEVICE)
    return _image_pipe


async def generate(request: Request):
    """
    文生图接口：prompt 必填，其余可选；返回 image/png。
    """
    if is_json_request(request):
        data = await read_json_optional(request) or {}
        form = {}
    else:
        form = await read_form_optional(request)
        data = {}

    prompt = data.get("prompt") or form.get("prompt")
    if not prompt or not str(prompt).strip():
        return ({"code": 400, "msg": "Missing prompt"}, 400)

    prompt = str(prompt).strip()
    seed = data.get("seed") if data else form.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None
    tmp = None
    try:
        pipe = _get_pipeline()
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        gen = pipe(
            prompt=prompt,
            height=96,
            width=96,
            num_inference_steps=25,
            guidance_scale=7.0,
            generator=(
                torch.Generator(device=DEVICE).manual_seed(seed)
                if seed is not None
                else None
            ),
        )
        image = gen.images[0]
        image.save(tmp)
        with open(tmp, "rb") as f:
            buf = BytesIO(f.read())
        try:
            os.unlink(tmp)
        except Exception:
            pass
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": 'inline; filename="image.png"'},
        )
    except Exception as e:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return JSONResponse(content={"code": 500, "msg": str(e)}, status_code=500)
