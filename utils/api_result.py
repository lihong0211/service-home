# utils/api_result.py
"""将业务层返回值规范为 Starlette/FastAPI Response（替代 compat.to_fastapi_response）。"""
from __future__ import annotations

from typing import Any

from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.responses import Response


def normalize_api_result(result: Any) -> Response:
    if result is None:
        return JSONResponse(content={"code": 500, "msg": "empty response"}, status_code=500)

    if isinstance(result, Response):
        return result

    if isinstance(result, tuple):
        if len(result) == 2:
            body, status_code = result
            headers = None
        elif len(result) == 3:
            body, status_code, headers = result
            headers = dict(headers) if headers else {}
        else:
            body, status_code = result[0], 500
            headers = None

        h = dict(headers or {})

        if isinstance(body, dict):
            return JSONResponse(content=body, status_code=int(status_code), headers=h)
        if isinstance(body, (StreamingResponse, FileResponse, Response)):
            for k, v in h.items():
                body.headers[k] = v
            if int(status_code) != getattr(body, "status_code", 200):
                body.status_code = int(status_code)
            return body
        return JSONResponse(
            content={"code": 500, "msg": "invalid tuple response body"},
            status_code=500,
        )

    if isinstance(result, dict):
        return JSONResponse(content=result, status_code=200)

    return JSONResponse(content={"code": 500, "msg": "unsupported response type"}, status_code=500)
