#!/usr/bin/env python3
"""AI 新闻摘要 Agent：RSS 抓取 → LLM 摘要 → 今日要闻总结。"""

from __future__ import annotations

import os
import asyncio
from datetime import datetime, timezone, timedelta

from fastapi import Request
from openai import OpenAI

_MODEL = "qwen-turbo"
_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0,
)

RSS_SOURCES = [
    {"name": "Hacker News", "url": "https://hnrss.org/frontpage"},
    {"name": "MIT Tech Review", "url": "https://www.technologyreview.com/feed/"},
    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
    {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
    {"name": "36kr AI", "url": "https://36kr.com/feed"},
]

_24H = timedelta(hours=24)


def _fetch_feed(source: dict) -> list[dict]:
    import feedparser
    import time
    feed = feedparser.parse(source["url"])
    now = datetime.now(timezone.utc)
    articles = []
    for entry in feed.entries[:10]:
        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        # 宽松过滤：没有时间戳也收录
        if published and (now - published) > _24H * 3:
            continue
        articles.append({
            "title": getattr(entry, "title", ""),
            "url": getattr(entry, "link", ""),
            "source": source["name"],
            "published_at": published.isoformat() if published else "",
            "summary": "",
        })
    return articles


def _summarize_article(title: str, url: str) -> str:
    prompt = f"请用50字以内的中文概括以下文章的主要内容。只输出摘要，不要解释。\n文章标题：{title}\n文章链接：{url}"
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return title


async def fetch_articles_api(request: Request):
    # 并发抓取所有 RSS
    all_articles: list[dict] = []
    feeds = await asyncio.gather(
        *[asyncio.to_thread(_fetch_feed, s) for s in RSS_SOURCES],
        return_exceptions=True,
    )
    for result in feeds:
        if isinstance(result, list):
            all_articles.extend(result)

    # 限制30篇，并发生成摘要
    all_articles = all_articles[:30]
    summaries = await asyncio.gather(
        *[asyncio.to_thread(_summarize_article, a["title"], a["url"]) for a in all_articles],
        return_exceptions=True,
    )
    for i, s in enumerate(summaries):
        if isinstance(s, str):
            all_articles[i]["summary"] = s

    return {"code": 0, "msg": "success", "data": all_articles}


async def news_summary_api(request: Request):
    # 先获取文章列表
    feeds = await asyncio.gather(
        *[asyncio.to_thread(_fetch_feed, s) for s in RSS_SOURCES],
        return_exceptions=True,
    )
    all_articles: list[dict] = []
    for result in feeds:
        if isinstance(result, list):
            all_articles.extend(result)
    all_articles = all_articles[:20]

    if not all_articles:
        return {"code": 200, "msg": "success", "data": {
            "summary": "暂无最新新闻",
            "article_count": 0,
            "generated_at": datetime.now().isoformat(),
        }}

    titles = "\n".join(f"- {a['title']} ({a['source']})" for a in all_articles)
    prompt = (
        f"以下是今日科技/AI 领域的最新新闻标题，请用200字以内的中文生成今日要闻总结，突出最重要的趋势和事件。\n\n"
        f"{titles}\n\n今日要闻总结："
    )

    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        summary = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        summary = f"摘要生成失败: {e}"

    return {"code": 0, "msg": "success", "data": {
        "summary": summary,
        "article_count": len(all_articles),
        "generated_at": datetime.now().isoformat(),
    }}
