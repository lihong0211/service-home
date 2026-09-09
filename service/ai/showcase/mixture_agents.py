#!/usr/bin/env python3
"""Mixture of Agents：并发调用多个 Ollama 模型，DashScope 聚合最优答案。"""

from __future__ import annotations

import os
import json
import asyncio

import anyio.to_thread
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

from utils.http_body import read_json_optional
from config.ai import DEFAULT_CHAT_MODEL
from service.ai._dashscope_common import get_dashscope_client

_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_AGGREGATE_MODEL = DEFAULT_CHAT_MODEL

_client = get_dashscope_client(timeout=60.0)

_SENTINEL = object()


async def _iter_sync_stream(stream):
    """把同步 SDK 的 stream 迭代逐个 chunk 挪到线程池执行，避免阻塞事件循环。"""
    it = iter(stream)
    while True:
        chunk = await anyio.to_thread.run_sync(next, it, _SENTINEL)
        if chunk is _SENTINEL:
            return
        yield chunk


async def list_models_api(request: Request):
    try:
        async with httpx.AsyncClient(timeout=5) as http:
            resp = await http.get(f"{_OLLAMA_BASE}/api/tags")
            data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"code": 0, "msg": "success", "data": models}
    except Exception as e:
        return {"code": 500, "msg": f"无法连接 Ollama: {e}"}


async def _call_ollama_model(model: str, question: str) -> tuple[str, str]:
    """调用单个 Ollama 模型，返回 (model_name, reply)。"""
    try:
        async with httpx.AsyncClient(timeout=60) as http:
            resp = await http.post(
                f"{_OLLAMA_BASE}/api/generate",
                json={"model": model, "prompt": question, "stream": False},
            )
            data = resp.json()
            return model, data.get("response", "（无响应）")
    except Exception as e:
        return model, f"（调用失败: {e}）"


async def mixture_chat_api(request: Request):
    body = await read_json_optional(request) or {}
    question = (body.get("question") or "").strip()
    models: list[str] = body.get("models") or []

    if not question:
        return {"code": 400, "msg": "Missing question"}
    if not models:
        return {"code": 400, "msg": "Missing models"}

    async def _stream():
        # 1. 并发调用所有模型
        tasks = [_call_ollama_model(m, question) for m in models]
        results = await asyncio.gather(*tasks)
        model_answers = [{"model": m, "answer": a} for m, a in results]

        # 2. 先把各模型原始回答发给前端
        yield f"data: {json.dumps({'type': 'models', 'data': model_answers}, ensure_ascii=False)}\n\n"

        # 3. 用 DashScope 聚合
        answers_text = "\n\n".join(
            f"【{r['model']}】\n{r['answer']}" for r in model_answers
        )
        aggregate_prompt = (
            f"以下是多个 AI 模型对同一问题的回答，请综合各模型的优点，给出一个最准确、最全面的最终答案。\n\n"
            f"问题：{question}\n\n"
            f"各模型回答：\n{answers_text}\n\n"
            f"综合最优答案："
        )

        stream = await anyio.to_thread.run_sync(
            lambda: _client.chat.completions.create(
                model=_AGGREGATE_MODEL,
                messages=[{"role": "user", "content": aggregate_prompt}],
                temperature=0.3,
                max_tokens=1024,
                stream=True,
            )
        )

        async for chunk in _iter_sync_stream(stream):
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield f"data: {json.dumps({'type': 'aggregate', 'response': delta.content}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
