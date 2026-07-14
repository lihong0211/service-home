#!/usr/bin/env python3
"""GitHub 仓库聊天：下载代码文件 → 向量索引 → 流式 RAG 问答。"""

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

_client = get_dashscope_client(timeout=60.0)

# 支持索引的文件扩展名
_ALLOWED_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".go", ".java", ".rs", ".cpp", ".c", ".h"}
_MAX_FILE_SIZE = 100 * 1024  # 100KB
_MAX_FILES = 50   # 限制文件数，降低索引时间
_CHUNK_SIZE = 1500  # chars per chunk
_CHUNK_OVERLAP = 150
_MAX_CHUNKS_TOTAL = 150  # 最多 150 个 chunk，约 6 次批量 embedding
_EMBED_BATCH_SIZE = 10  # DashScope text-embedding-v4 每批最多 10 条


def _parse_owner_repo(repo_url: str) -> tuple[str, str]:
    """从 GitHub URL 解析 owner/repo。"""
    url = repo_url.strip().rstrip("/")
    # 支持 https://github.com/owner/repo 或 owner/repo 格式
    m = re.search(r"github\.com[:/]([^/]+)/([^/\s#?]+)", url)
    if m:
        return m.group(1), m.group(2).removesuffix(".git")
    # 短格式 owner/repo
    parts = url.split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1].removesuffix(".git")
    raise ValueError(f"无法解析 GitHub URL: {repo_url}")


def _collection_name(index_id: str) -> str:
    return f"github_{index_id}"


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


def _get_embedding(text: str) -> list[float]:
    return _get_embeddings_batch([text])[0]


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """批量 embedding，每批最多 _EMBED_BATCH_SIZE 条。"""
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


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _download_github_files(owner: str, repo: str, token: str | None) -> list[dict]:
    """通过 ZIP 下载仓库（单次请求，不受 API 速率限制），返回 [{path, content}] 列表。"""
    import io
    import zipfile
    import urllib.request

    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
    headers = {"User-Agent": "github-chat-indexer"}
    if token:
        headers["Authorization"] = f"token {token}"

    # 尝试 main 分支，失败则尝试 master
    for branch in ("main", "master"):
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        req = urllib.request.Request(zip_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                zip_data = resp.read()
            break
        except Exception:
            continue
    else:
        raise ValueError(f"无法下载仓库 {owner}/{repo}，请确认仓库存在且可公开访问")

    files = []
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for name in zf.namelist():
            if len(files) >= _MAX_FILES:
                break
            # ZIP 内路径格式：{repo}-{branch}/path/to/file
            parts = name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            rel_path = parts[1]
            ext = os.path.splitext(rel_path)[1].lower()
            if ext not in _ALLOWED_EXTS:
                continue
            info = zf.getinfo(name)
            if info.file_size > _MAX_FILE_SIZE or info.file_size == 0:
                continue
            try:
                content = zf.read(name).decode("utf-8", errors="ignore").strip()
                if content:
                    files.append({"path": rel_path, "content": content})
            except Exception:
                pass
    return files


async def github_index_api(request: Request):
    body = await read_json_optional(request) or {}
    repo_url = (body.get("repo_url") or "").strip()
    token = (body.get("github_token") or body.get("token") or "").strip() or None

    if not repo_url:
        return {"code": 400, "msg": "Missing repo_url"}

    try:
        owner, repo = _parse_owner_repo(repo_url)
    except ValueError as e:
        return {"code": 400, "msg": str(e)}

    # 使用稳定的 index_id（owner_repo），同一仓库重复索引会覆盖
    safe_owner = re.sub(r"[^a-zA-Z0-9_-]", "_", owner)
    safe_repo = re.sub(r"[^a-zA-Z0-9_-]", "_", repo)
    index_id = f"{safe_owner}_{safe_repo}"
    collection = _collection_name(index_id)

    # 下载文件
    try:
        files = await asyncio.to_thread(_download_github_files, owner, repo, token)
    except ValueError as e:
        return {"code": 400, "msg": str(e)}
    except Exception as e:
        return {"code": 500, "msg": f"下载仓库文件失败: {e}"}

    if not files:
        return {"code": 400, "msg": "仓库中未找到可索引的代码文件（支持 .py/.js/.ts/.md 等）"}

    # 初始化 Qdrant collection（已存在则先删除重建）
    try:
        qdrant = _get_qdrant_client()
        if qdrant.collection_exists(collection_name=collection):
            qdrant.delete_collection(collection)
        _ensure_collection(qdrant, collection)
    except Exception as e:
        return {"code": 500, "msg": f"向量库初始化失败: {e}"}

    # 分段：收集所有 (path, chunk_index, text) 并限制总数
    all_items: list[tuple[str, int, str]] = []
    for file_info in files:
        if len(all_items) >= _MAX_CHUNKS_TOTAL:
            break
        path = file_info["path"]
        content = file_info["content"]
        chunks = _chunk_text(content)
        for i, chunk in enumerate(chunks):
            if len(all_items) >= _MAX_CHUNKS_TOTAL:
                break
            all_items.append((path, i, f"File: {path}\n\n{chunk}"))

    if not all_items:
        return {"code": 400, "msg": "仓库中未找到可分段的内容"}

    # 批量 embedding（一次 API 调用处理多条，大幅减少请求次数）
    texts = [item[2] for item in all_items]
    try:
        embeddings = await asyncio.to_thread(_get_embeddings_batch, texts)
    except Exception as e:
        return {"code": 500, "msg": f"向量化失败: {e}"}

    # 组装 Qdrant points
    from qdrant_client.http.models import PointStruct
    qdrant_points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[j],
            payload={"path": all_items[j][0], "text": all_items[j][2], "chunk_index": all_items[j][1]},
        )
        for j in range(len(all_items))
        if j < len(embeddings) and embeddings[j]
    ]

    if not qdrant_points:
        return {"code": 500, "msg": "向量化失败，请检查 DASHSCOPE_API_KEY"}

    # 批量写入 Qdrant
    try:
        write_batch = 50
        for i in range(0, len(qdrant_points), write_batch):
            qdrant.upsert(collection_name=collection, points=qdrant_points[i: i + write_batch])
    except Exception as e:
        return {"code": 500, "msg": f"写入向量库失败: {e}"}

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "index_id": index_id,
            "file_count": len(files),
            "chunk_count": len(qdrant_points),
            "owner": owner,
            "repo": repo,
        },
    }


async def github_ask_api(request: Request):
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
                yield f"data: {json.dumps({'response': '索引不存在，请先对仓库进行索引'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 向量检索（兼容 qdrant-client >= 1.17 使用 query_points，旧版用 search）
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

            context_parts = []
            for hit in hits:
                payload = getattr(hit, "payload", None) or {}
                text = payload.get("text", "")
                if text:
                    context_parts.append(text)

            if not context_parts:
                yield f"data: {json.dumps({'response': '未在仓库中找到相关代码片段。'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            context = "\n\n---\n\n".join(context_parts)
            prompt = (
                f"你是一个代码助手，请基于以下代码片段回答问题。\n\n"
                f"代码片段：\n{context}\n\n"
                f"问题：{question}\n\n"
                f"请直接回答："
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
