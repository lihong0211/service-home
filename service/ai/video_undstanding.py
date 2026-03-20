"""
视频理解 - Qwen2-VL-2B，适配 Mac MPS。
支持 HTTP 接口：上传视频 + 问题，返回回答。
"""

import base64
import os
import tempfile

import av
import numpy as np
import torch
from fastapi import Request
from PIL import Image

from qwen_vl_utils import process_vision_info

from utils.http_body import is_json_request, read_json_optional, read_form_optional, write_upload_to_disk

# 模型路径，可用环境变量 VIDEO_MODEL_PATH 覆盖
MODEL_PATH = os.environ.get("VIDEO_MODEL_PATH", "models/Qwen/Qwen2-VL-2B-Instruct")
DEVICE = "mps"

_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=None,
        )
        _model = _model.to(DEVICE)
        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return _model, _processor


def extract_video_frames(video_path, num_frames=6):
    """从视频中均匀采样 num_frames 帧，返回 PIL Image 列表。"""
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    if total_frames == 0:
        frames_list = []
        for frame in container.decode(video=0):
            frames_list.append(frame.to_ndarray(format="rgb24"))
        total_frames = len(frames_list)
        container.close()
        container = av.open(video_path)
    else:
        frames_list = None

    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    if frames_list is not None:
        for idx in indices:
            frames.append(Image.fromarray(frames_list[idx]))
    else:
        all_frames = []
        for frame in container.decode(video=0):
            all_frames.append(frame.to_ndarray(format="rgb24"))
        for idx in indices:
            if idx < len(all_frames):
                frames.append(Image.fromarray(all_frames[idx]))
    container.close()
    return frames


def _analyze_video(video_path, question, num_frames=6):
    """内部：根据视频路径和问题生成回答。"""
    model, processor = _get_model()
    frames = extract_video_frames(video_path, num_frames)

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": frame} for frame in frames],
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated_ids, generated_ids_trimmed
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    return answer


async def video_understand(request: Request):
    """
    视频理解接口

    请求方式一：multipart/form-data，字段 video 或 file，question（必填），num_frames（可选，默认 6）
    请求方式二：application/json，body: { "video_base64": "<base64>", "question": "问题", "num_frames": 6 可选 }

    响应：{ "code": 0, "answer": "回答文本" }
    """
    video_path = None
    try:
        if is_json_request(request):
            body = await read_json_optional(request) or {}
            form = {}
        else:
            form = await read_form_optional(request)
            body = {}

        question = body.get("question") or form.get("question")
        if not question or not str(question).strip():
            raise ValueError("Missing question")
        question = str(question).strip()

        num_frames = 6
        nf = body.get("num_frames") if body else None
        if nf is None and form:
            nf = form.get("num_frames")
        if nf is not None:
            try:
                num_frames = max(1, min(32, int(nf)))
            except (TypeError, ValueError):
                pass

        if form:
            f = form.get("video") or form.get("file")
            if not f or not getattr(f, "filename", None):
                raise ValueError("Missing video or file in form")
            suffix = os.path.splitext(f.filename)[1] or ".mp4"
            fd, video_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            await write_upload_to_disk(f, video_path)
        elif body:
            b64 = body.get("video_base64") or body.get("video")
            if not b64:
                raise ValueError("Missing video_base64 or video in body")
            raw = base64.b64decode(
                b64.split(",", 1)[-1] if isinstance(b64, str) and "base64," in b64 else b64
            )
            fd, video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            with open(video_path, "wb") as out:
                out.write(raw)
        else:
            raise ValueError("Send multipart video/file or JSON with video_base64")

        answer = _analyze_video(video_path, question, num_frames=num_frames)
        return {"code": 0, "answer": answer}
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception:
                pass


# CLI：直接运行脚本时用本地视频测试
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "car.mp4")
    if not os.path.exists(video_path):
        print(f"请放置测试视频 car.mp4 到: {script_dir}")
    else:
        print("加载模型并分析视频...")
        a = _analyze_video(video_path, "Describe this video in detail.", num_frames=6)
        print("回答:", a)
