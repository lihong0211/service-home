# utils/http_body.py
"""FastAPI/Starlette 请求体与查询参数解析（替代 Flask request）。"""
from __future__ import annotations

from fastapi import Request


def _content_type(request: Request) -> str:
    return (request.headers.get("content-type") or "").lower()


def is_json_request(request: Request) -> bool:
    return "application/json" in _content_type(request)


async def read_json_optional(request: Request) -> dict:
    if not is_json_request(request):
        return {}
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


async def read_json_or_empty(request: Request) -> dict:
    """与 read_json_optional 相同，语义上用于 POST JSON。"""
    return await read_json_optional(request)


async def read_form_optional(request: Request):
    """multipart 或 x-www-form-urlencoded；失败返回空 dict-like 不可用，用 try/except 外层处理。"""
    try:
        return await request.form()
    except Exception:
        return {}


def query_dict(request: Request) -> dict:
    return dict(request.query_params)


async def write_upload_to_disk(upload, path: str, chunk_size: int = 1024 * 1024) -> None:
    """将 Starlette UploadFile 写入磁盘。"""
    with open(path, "wb") as out:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)


def collect_upload_files_from_form(form) -> list:
    """从 FormData 收集上传文件（兼容 file / file[] / file[0] / files）。"""
    out: list = []
    try:
        keys = list(form.keys())
    except Exception:
        keys = []
    for key in keys:
        if key in ("file", "file[]", "files") or (
            key.startswith("file[") and key.endswith("]")
        ):
            for v in form.getlist(key):
                if hasattr(v, "read"):
                    out.append(v)
    if not out:
        v = form.get("file")
        if v is not None and hasattr(v, "read"):
            out.append(v)
    return out


async def read_json_or_form_fields(request: Request) -> dict:
    """
    优先 JSON body；否则解析 form，仅保留非文件字段为 dict（值取单值时 str）。
    """
    data = await read_json_optional(request)
    if data:
        return data
    form = await read_form_optional(request)
    if not form:
        return {}
    out: dict = {}
    for k in form.keys():
        v = form.get(k)
        if hasattr(v, "read"):
            continue
        out[k] = v
    return out
