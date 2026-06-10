#!/usr/bin/env python3
"""对话持久记忆：对话时注入历史记忆，对话后异步提取新记忆写库。"""

from __future__ import annotations

import os
import json
import asyncio

from fastapi import Request
from fastapi.responses import StreamingResponse
from openai import OpenAI
from app.database import db as _db
from utils.http_body import read_json_optional

_MODEL_CHAT = os.getenv("MEMORY_CHAT_MODEL", "qwen-turbo")
_MODEL_EXTRACT = "qwen-turbo"

_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0,
)


def _get_db():
    return _db.session


def _load_memories(user_id: str, limit: int = 20) -> list[dict]:
    from model.ai.chat_memory import ChatMemory
    db = _get_db()
    rows = (
        db.query(ChatMemory)
        .filter(ChatMemory.user_id == user_id)
        .order_by(ChatMemory.id.desc())
        .limit(limit)
        .all()
    )
    return [{"id": r.id, "content": r.content, "type": r.memory_type} for r in reversed(rows)]


def _save_memories(user_id: str, contents: list[str], memory_type: str = "fact"):
    """用独立 session 写记忆（在 asyncio.to_thread 里调用，ContextVar 不可用）。"""
    from model.ai.chat_memory import ChatMemory
    from app.database import SessionLocal
    with SessionLocal() as session:
        for content in contents:
            content = content.strip()
            if content:
                session.add(ChatMemory(user_id=user_id, content=content, memory_type=memory_type))
        session.commit()


def _extract_memories_sync(user_message: str, assistant_reply: str) -> list[str]:
    """从一轮对话中提取值得记住的信息。"""
    prompt = (
        "请从以下对话中提取用户透露的重要个人信息（姓名、职业、喜好、目标等）。\n"
        "每条记忆单独一行，不重要则返回空。只输出记忆内容，不要解释。\n\n"
        f"用户说：{user_message}\n"
        f"助手回复：{assistant_reply}\n\n"
        "提取的记忆（每行一条，没有则输出 NONE）："
    )
    try:
        resp = _client.chat.completions.create(
            model=_MODEL_EXTRACT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        text = resp.choices[0].message.content or ""
        if text.strip().upper() == "NONE":
            return []
        return [line.strip("- •·\t") for line in text.strip().splitlines() if line.strip() and line.strip().upper() != "NONE"]
    except Exception:
        return []


async def memory_chat_api(request: Request):
    body = await read_json_optional(request) or {}
    user_id = (body.get("user_id") or "").strip()
    message = (body.get("message") or "").strip()

    if not user_id or not message:
        return {"code": 400, "msg": "Missing user_id or message"}

    memories = _load_memories(user_id)
    memory_text = "\n".join(f"- {m['content']}" for m in memories) if memories else "（暂无记忆）"

    system_prompt = (
        "你是一个有记忆的智能助手。以下是关于该用户的已知信息：\n"
        f"{memory_text}\n\n"
        "请结合用户信息进行个性化回答。"
    )

    full_reply_parts: list[str] = []

    async def _stream():
        stream = _client.chat.completions.create(
            model=_MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_reply_parts.append(delta.content)
                yield f"data: {json.dumps({'response': delta.content}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

        # 对话结束后异步提取记忆
        full_reply = "".join(full_reply_parts)
        new_memories = await asyncio.to_thread(_extract_memories_sync, message, full_reply)
        if new_memories:
            await asyncio.to_thread(_save_memories, user_id, new_memories)

    return StreamingResponse(_stream(), media_type="text/event-stream")


async def list_memories_api(request: Request):
    user_id = (request.query_params.get("user_id") or "").strip()
    if not user_id:
        return {"code": 400, "msg": "Missing user_id"}
    memories = _load_memories(user_id, limit=50)
    return {"code": 0, "msg": "success", "data": memories}


async def clear_memories_api(request: Request):
    body = await read_json_optional(request) or {}
    user_id = (body.get("user_id") or "").strip()
    if not user_id:
        return {"code": 400, "msg": "Missing user_id"}
    from model.ai.chat_memory import ChatMemory
    db = _get_db()
    db.query(ChatMemory).filter(ChatMemory.user_id == user_id).delete()
    db.commit()
    return {"code": 0, "msg": "success"}
