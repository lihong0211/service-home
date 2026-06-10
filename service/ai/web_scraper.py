#!/usr/bin/env python3
"""网页智能提取 Agent：newspaper3k 抓取正文，LLM 按 schema 结构化提取。"""

from __future__ import annotations

import os
import json

from fastapi import Request
from openai import OpenAI

from utils.http_body import read_json_optional

_MODEL = "qwen-turbo"
_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0,
)


def _scrape_url(url: str) -> dict:
    from newspaper import Article
    import asyncio

    article = Article(url, language="zh")
    article.download()
    article.parse()

    return {
        "title": article.title or "",
        "text": article.text or "",
        "authors": article.authors or [],
        "publish_date": str(article.publish_date) if article.publish_date else "",
        "top_image": article.top_image or "",
    }


async def web_scrape_extract_api(request: Request):
    import asyncio
    body = await read_json_optional(request) or {}
    url = (body.get("url") or "").strip()
    schema: dict[str, str] = body.get("schema") or {}

    if not url:
        return {"code": 400, "msg": "Missing url"}

    try:
        page = await asyncio.to_thread(_scrape_url, url)
    except Exception as e:
        return {"code": 500, "msg": f"网页抓取失败: {e}"}

    text = page["text"][:4000] if page["text"] else ""
    word_count = len(page["text"].split()) if page["text"] else 0

    if not schema:
        # 默认提取
        prompt = (
            f"请从以下网页内容中提取结构化信息，以 JSON 格式返回，包含以下字段：\n"
            f"title（标题）、summary（摘要100字）、keywords（关键词列表）、main_topic（主要主题）\n\n"
            f"网页内容：\n{text}\n\n只返回 JSON，不要其他内容："
        )
    else:
        schema_desc = "\n".join(f"- {k}: {v}" for k, v in schema.items())
        prompt = (
            f"请从以下网页内容中提取以下字段的信息，以 JSON 格式返回：\n{schema_desc}\n\n"
            f"网页内容：\n{text}\n\n只返回 JSON，不要其他内容："
        )

    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or ""
        start = raw.find("{")
        end = raw.rfind("}") + 1
        extracted = json.loads(raw[start:end]) if start >= 0 and end > start else {"raw": raw}
    except Exception as e:
        extracted = {"error": str(e)}

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "url": url,
            "title": page["title"],
            "word_count": word_count,
            "extracted": extracted,
        },
    }
