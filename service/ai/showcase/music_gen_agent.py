"""
AI 音乐生成 Agent - 需要 Suno API Key
环境变量：
  SUNO_API_KEY - Suno AI 的 API Key（从 https://suno.com 获取）
  SUNO_API_BASE - API 基础地址（默认使用非官方代理，建议自行部署）

备选方案：
  可使用 MusicGen (facebook/musicgen-small) 本地模型，但需要 GPU 和约 2GB 显存
  pip install audiocraft transformers

配置步骤：
1. 注册 Suno AI 账号：https://suno.com
2. 获取 API Key 或使用第三方 Suno API 代理（如 suno-api）
3. 将 SUNO_API_KEY 写入 .env 文件

备注：Suno 官方目前无正式公开 API，建议关注官方动态
"""
import os
import json
import uuid
import asyncio
from fastapi import Request
from fastapi.responses import StreamingResponse

SUNO_API_KEY = os.getenv("SUNO_API_KEY", "")
SUNO_API_BASE = os.getenv("SUNO_API_BASE", "https://api.suno.ai/v1")

_task_store: dict = {}


async def music_generate_api(request: Request):
    if not SUNO_API_KEY:
        return {"code": 500, "msg": "未配置 SUNO_API_KEY，请在 .env 文件中设置"}

    body = await request.json()
    prompt = body.get("prompt", "")
    style = body.get("style", "pop")
    duration = body.get("duration", 30)

    if not prompt:
        return {"code": 400, "msg": "请描述想要的音乐"}

    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{SUNO_API_BASE}/generate",
            headers={"Authorization": f"Bearer {SUNO_API_KEY}", "Content-Type": "application/json"},
            json={"prompt": f"{prompt}, {style} style", "duration": duration, "make_instrumental": False},
        )
        if resp.status_code != 200:
            return {"code": 500, "msg": f"Suno API 错误: {resp.text}"}
        data = resp.json()
        task_id = data.get("id", str(uuid.uuid4()))
        _task_store[task_id] = {"status": "pending", "data": data}

    return {"code": 0, "msg": "success", "data": {"task_id": task_id, "status": "pending"}}


async def music_status_api(request: Request):
    task_id = request.query_params.get("task_id", "")
    if not task_id:
        return {"code": 400, "msg": "缺少 task_id"}

    if not SUNO_API_KEY:
        return {"code": 500, "msg": "未配置 SUNO_API_KEY"}

    import httpx
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{SUNO_API_BASE}/status/{task_id}",
            headers={"Authorization": f"Bearer {SUNO_API_KEY}"},
        )
        if resp.status_code != 200:
            return {"code": 500, "msg": f"查询失败: {resp.text}"}
        data = resp.json()

    return {
        "code": 0, "msg": "success",
        "data": {
            "task_id": task_id,
            "status": data.get("status", "pending"),
            "audio_url": data.get("audio_url"),
            "title": data.get("title", ""),
            "duration": data.get("duration"),
        },
    }
