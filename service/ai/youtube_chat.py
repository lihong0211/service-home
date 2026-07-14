#!/usr/bin/env python3
"""YouTube 视频聊天：提取字幕 → 向量索引 → 流式 RAG 问答。"""

from __future__ import annotations

import os
import re
import time
import uuid
import json
import asyncio

from fastapi import Request
from fastapi.responses import StreamingResponse

from utils.http_body import read_json_optional
from config.ai import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL
from service.ai._dashscope_common import get_dashscope_client

_MODEL = DEFAULT_CHAT_MODEL
_DIMENSION = int(os.getenv("VECTOR_DB_DIMENSION", "1024"))
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 100
_MAX_CHUNKS = 200
_EMBED_BATCH_SIZE = 10

_client = get_dashscope_client(timeout=60.0)


def _extract_video_id(url: str) -> str:
    url = url.strip()
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError(f"无法解析 YouTube 视频 ID: {url}")


def _collection_name(video_id: str) -> str:
    return f"youtube_{video_id}"


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    url = (os.getenv("QDRANT_URL") or "http://localhost:6333").strip()
    api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
    return QdrantClient(url=url, api_key=api_key)


def _ensure_collection(qdrant, name: str):
    from qdrant_client.http.models import Distance, VectorParams
    if not qdrant.collection_exists(collection_name=name):
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=_DIMENSION, distance=Distance.COSINE),
        )


def _make_yt_api():
    """创建 YouTubeTranscriptApi 实例，优先使用系统代理。"""
    import requests
    from youtube_transcript_api import YouTubeTranscriptApi

    proxy = (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
        or ""
    ).strip()

    session = requests.Session()
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}

    return YouTubeTranscriptApi(http_client=session)


def _fetch_transcript(video_id: str) -> tuple[list, str]:
    """获取字幕，返回 (segments, language)。v1.x 使用实例方法。"""
    api = _make_yt_api()

    # 优先中文，其次英文
    preferred_groups = [
        (["zh-Hans", "zh-CN", "zh"], "zh"),
        (["en", "en-US", "en-GB"], "en"),
    ]
    last_err = None
    for langs, lang_label in preferred_groups:
        try:
            transcript = api.fetch(video_id, languages=langs)
            return list(transcript), lang_label
        except Exception as e:
            last_err = e

    # 兜底：列出所有可用字幕，取第一个
    try:
        tl = api.list(video_id)
        for t in tl:
            fetched = t.fetch()
            return list(fetched), t.language_code
    except Exception:
        pass

    raise ValueError(f"未找到可用字幕: {last_err}")


def _segments_to_chunks(segments: list) -> list[str]:
    """将字幕片段合并为文本块。"""
    full_text = " ".join(
        (s.text if hasattr(s, "text") else s.get("text", ""))
        for s in segments
    ).strip()

    chunks = []
    start = 0
    while start < len(full_text) and len(chunks) < _MAX_CHUNKS:
        end = start + _CHUNK_SIZE
        chunks.append(full_text[start:end])
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    results: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i: i + _EMBED_BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = _client.embeddings.create(
                    model=DEFAULT_EMBEDDING_MODEL,
                    input=batch,
                    dimensions=_DIMENSION,
                    encoding_format="float",
                )
                results.extend([item.embedding for item in resp.data])
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise e
    return results


def _get_embedding(text: str) -> list[float]:
    return _get_embeddings_batch([text])[0]


async def youtube_index_api(request: Request):
    body = await read_json_optional(request) or {}
    video_url = (body.get("video_url") or "").strip()

    if not video_url:
        return {"code": 400, "msg": "Missing video_url"}

    try:
        video_id = _extract_video_id(video_url)
    except ValueError as e:
        return {"code": 400, "msg": str(e)}

    collection = _collection_name(video_id)

    # 获取字幕
    try:
        segments, language = await asyncio.to_thread(_fetch_transcript, video_id)
    except ValueError as e:
        return {"code": 400, "msg": str(e)}
    except Exception as e:
        return {"code": 500, "msg": f"字幕提取失败: {e}"}

    if not segments:
        return {"code": 400, "msg": "字幕内容为空"}

    chunks = _segments_to_chunks(segments)
    if not chunks:
        return {"code": 400, "msg": "字幕内容太短，无法索引"}

    # 初始化 Qdrant（重建）
    try:
        qdrant = _get_qdrant_client()
        if qdrant.collection_exists(collection_name=collection):
            qdrant.delete_collection(collection)
        _ensure_collection(qdrant, collection)
    except Exception as e:
        return {"code": 500, "msg": f"向量库初始化失败: {e}"}

    # Embedding
    try:
        embeddings = await asyncio.to_thread(_get_embeddings_batch, chunks)
    except Exception as e:
        return {"code": 500, "msg": f"向量化失败: {e}"}

    # 写入 Qdrant
    from qdrant_client.http.models import PointStruct
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": chunks[i], "chunk_index": i},
        )
        for i in range(len(chunks))
        if i < len(embeddings) and embeddings[i]
    ]

    if not points:
        return {"code": 500, "msg": "向量化失败，请检查 DASHSCOPE_API_KEY"}

    try:
        batch_size = 50
        for i in range(0, len(points), batch_size):
            qdrant.upsert(collection_name=collection, points=points[i: i + batch_size])
    except Exception as e:
        return {"code": 500, "msg": f"写入向量库失败: {e}"}

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "index_id": video_id,
            "video_id": video_id,
            "segment_count": len(points),
            "language": language,
        },
    }


async def youtube_ask_api(request: Request):
    body = await read_json_optional(request) or {}
    index_id = (body.get("index_id") or "").strip()
    question = (body.get("question") or "").strip()

    if not index_id or not question:
        return {"code": 400, "msg": "Missing index_id or question"}

    collection = _collection_name(index_id)

    async def _stream():
        try:
            qdrant = _get_qdrant_client()
            if not qdrant.collection_exists(collection_name=collection):
                yield f"data: {json.dumps({'response': '索引不存在，请先对视频进行索引'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            query_vec = await asyncio.to_thread(_get_embedding, question)
            if hasattr(qdrant, "query_points"):
                qresp = qdrant.query_points(
                    collection_name=collection,
                    query=query_vec,
                    limit=5,
                    with_payload=True,
                )
                hits = getattr(qresp, "points", None) or []
            else:
                hits = qdrant.search(
                    collection_name=collection,
                    query_vector=query_vec,
                    limit=5,
                    with_payload=True,
                ) or []

            context_parts = [
                (getattr(h, "payload", None) or {}).get("text", "")
                for h in hits
            ]
            context_parts = [t for t in context_parts if t]

            if not context_parts:
                yield f"data: {json.dumps({'response': '未在视频字幕中找到相关内容。'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            context = "\n\n---\n\n".join(context_parts)
            prompt = (
                "你是一个视频内容助手，请根据以下视频字幕内容回答问题。\n\n"
                f"字幕内容：\n{context}\n\n"
                f"问题：{question}\n\n"
                "请直接回答："
            )

            stream = _client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield f"data: {json.dumps({'response': delta.content}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
