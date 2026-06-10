#!/usr/bin/env python3
"""
Elasticsearch BM25 检索 + MySQL 同步（向量库文档）。

目标：
- 将 MySQL 的 vector_db_document 同步到 ES（用于 BM25/关键词检索）
- 提供 BM25 搜索函数，供 RAG 做 hybrid（Dense + BM25 + rerank）

设计约束：
- ES 作为可选组件：未配置 ES 时不影响主流程
- 同步与检索尽量 best-effort：失败返回可读错误，不抛出到主链路
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any, Optional

from fastapi import Request

from utils.http_body import read_json_optional
from model.ai import VectorDb, VectorDbDocument


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


ES_ENABLED = _env_bool("ES_BM25_ENABLED", False)
ES_URL = (os.getenv("ES_URL") or "").strip()
ES_API_KEY = (os.getenv("ES_API_KEY") or "").strip()
ES_USERNAME = (os.getenv("ES_USERNAME") or "").strip()
ES_PASSWORD = (os.getenv("ES_PASSWORD") or "").strip()
ES_INDEX = (os.getenv("ES_BM25_INDEX") or "service_home_vector_db_docs").strip()
ES_TEXT_ANALYZER = (os.getenv("ES_BM25_TEXT_ANALYZER") or "standard").strip()
ES_TEXT_SEARCH_ANALYZER = (os.getenv("ES_BM25_TEXT_SEARCH_ANALYZER") or ES_TEXT_ANALYZER).strip()


def _es_available() -> bool:
    if not ES_ENABLED:
        return False
    return bool(ES_URL)


def _get_es_client():
    """
    延迟导入 elasticsearch，避免未安装依赖时影响非 ES 功能。
    """
    Elasticsearch = getattr(importlib.import_module("elasticsearch"), "Elasticsearch")

    kwargs: dict[str, Any] = {"hosts": [ES_URL]}
    if ES_API_KEY:
        kwargs["api_key"] = ES_API_KEY
    elif ES_USERNAME and ES_PASSWORD:
        kwargs["basic_auth"] = (ES_USERNAME, ES_PASSWORD)
    return Elasticsearch(**kwargs)


def _doc_es_id(vector_db_id: int, doc_id: str) -> str:
    # 保证跨库 doc_id 不冲突
    return f"{int(vector_db_id)}::{str(doc_id)}"


def ensure_index() -> dict:
    """
    确保 ES 索引存在。映射面向：
    - BM25：text 字段
    - 过滤：vector_db_id / category / metadata.*
    """
    if not _es_available():
        return {"enabled": False, "reason": "ES_BM25_ENABLED=0 或未配置 ES_URL"}
    es = _get_es_client()
    if es.indices.exists(index=ES_INDEX):
        return {"enabled": True, "index": ES_INDEX, "created": False}

    body = {
        "settings": {
            "number_of_shards": int(os.getenv("ES_BM25_SHARDS", "1")),
            "number_of_replicas": int(os.getenv("ES_BM25_REPLICAS", "0")),
            "analysis": {},
        },
        "mappings": {
            "dynamic": True,
            "properties": {
                "vector_db_id": {"type": "integer"},
                "db_name": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "category": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": ES_TEXT_ANALYZER,
                    "search_analyzer": ES_TEXT_SEARCH_ANALYZER,
                },
                # metadata 允许动态扩展；常用精确过滤场景建议写入 keyword 子字段或显式字段
                "metadata": {"type": "object", "dynamic": True},
                "create_at": {"type": "date"},
                "update_at": {"type": "date"},
            }
        },
    }
    es.indices.create(index=ES_INDEX, body=body)
    return {"enabled": True, "index": ES_INDEX, "created": True}


def upsert_one_from_mysql(vector_db_id: int, doc_id: str) -> dict:
    """
    从 MySQL 读取一条文档并 upsert 到 ES。
    """
    if not _es_available():
        return {"ok": False, "skipped": True, "reason": "ES not enabled"}
    ensure_index()
    row = VectorDbDocument.select_one_by({"vector_db_id": int(vector_db_id), "doc_id": str(doc_id)})
    if not row:
        return {"ok": False, "skipped": True, "reason": "doc not found in mysql"}
    vdb = VectorDb.get_by_id(int(vector_db_id))
    db_name = vdb.name if vdb else None
    payload = {
        "vector_db_id": int(vector_db_id),
        "db_name": db_name,
        "doc_id": str(row.doc_id),
        "category": (row.category or "").strip() or None,
        "text": row.text,
        "metadata": getattr(row, "document_metadata", None),
        "create_at": row.create_at.isoformat() if getattr(row, "create_at", None) else None,
        "update_at": row.update_at.isoformat() if getattr(row, "update_at", None) else None,
    }
    es = _get_es_client()
    es.index(index=ES_INDEX, id=_doc_es_id(vector_db_id, doc_id), document=payload, refresh=False)
    return {"ok": True}


def delete_one(vector_db_id: int, doc_id: str) -> dict:
    if not _es_available():
        return {"ok": False, "skipped": True, "reason": "ES not enabled"}
    ensure_index()
    es = _get_es_client()
    try:
        es.delete(index=ES_INDEX, id=_doc_es_id(vector_db_id, doc_id), refresh=False)
    except Exception as e:
        # 不存在也算成功
        msg = str(e)
        if "not_found" in msg.lower() or "404" in msg:
            return {"ok": True, "deleted": False}
        return {"ok": False, "error": msg}
    return {"ok": True, "deleted": True}


def sync_vector_db(vector_db_id: int, batch_size: int = 200, refresh: bool = True) -> dict:
    """
    全量同步一个向量库的文档到 ES（覆盖/更新，不做“删差集”）。
    - 对删除同步：建议用 delete_one（在文档删除接口里调用）或单独做一次“重建索引”。
    """
    if not _es_available():
        return {"ok": False, "skipped": True, "reason": "ES not enabled"}
    ensure_index()
    bulk = getattr(importlib.import_module("elasticsearch.helpers"), "bulk")

    vdb = VectorDb.get_by_id(int(vector_db_id))
    if not vdb:
        return {"ok": False, "error": "vector_db not found"}
    db_name = vdb.name

    # 分批按自增 id 扫描（BaseModel.builder_query 支持 gt/gte，但这里更简单地用 offset 分页）
    total = VectorDbDocument.count({"vector_db_id": int(vector_db_id)})
    es = _get_es_client()
    synced = 0
    t0 = time.time()
    page = 0
    while synced < total:
        rows = VectorDbDocument.select_by(
            {
                "vector_db_id": int(vector_db_id),
                "order_by": [{"col": "id", "sort": "asc"}],
            }
        )
        # 上面 select_by 会一次性全取；为了避免大库内存占用，改用 builder_query 分页
        # 这里保守实现：若数据量较大，建议后续改为基于 builder_query().limit().offset()
        break

    # 改用真正分页（直接操作 SQLAlchemy Query）
    from app.app import db as _db

    q = _db.session.query(VectorDbDocument).where(
        VectorDbDocument.vector_db_id == int(vector_db_id),
        VectorDbDocument.deleted_at.is_(None),
    ).order_by(VectorDbDocument.id.asc())
    offset = 0
    while True:
        batch = q.limit(int(batch_size)).offset(int(offset)).all()
        if not batch:
            break
        actions = []
        for row in batch:
            doc_id = str(row.doc_id)
            actions.append(
                {
                    "_op_type": "index",
                    "_index": ES_INDEX,
                    "_id": _doc_es_id(vector_db_id, doc_id),
                    "vector_db_id": int(vector_db_id),
                    "db_name": db_name,
                    "doc_id": doc_id,
                    "category": (row.category or "").strip() or None,
                    "text": row.text,
                    "metadata": getattr(row, "document_metadata", None),
                    "create_at": row.create_at.isoformat() if getattr(row, "create_at", None) else None,
                    "update_at": row.update_at.isoformat() if getattr(row, "update_at", None) else None,
                }
            )
        bulk(es, actions, refresh=False, raise_on_error=False)
        synced += len(batch)
        offset += len(batch)

    if refresh:
        try:
            es.indices.refresh(index=ES_INDEX)
        except Exception:
            pass
    return {"ok": True, "vector_db_id": int(vector_db_id), "db_name": db_name, "synced": synced, "took_s": round(time.time() - t0, 3)}


def bm25_search(
    vector_db_id: int,
    query: str,
    top_k: int = 5,
    category: str | None = None,
    metadata_filter: dict | None = None,
) -> dict:
    if not _es_available():
        return {"ok": False, "error": "ES not enabled (set ES_BM25_ENABLED=1 and ES_URL)"}
    ensure_index()
    es = _get_es_client()
    q = (query or "").strip()
    if not q:
        return {"ok": True, "hits": []}

    must: list[dict[str, Any]] = [
        {"term": {"vector_db_id": int(vector_db_id)}},
        {
            "match": {
                "text": {
                    "query": q,
                    "operator": "or",
                }
            }
        },
    ]
    if category is not None:
        if category == "":
            must.append({"bool": {"must_not": [{"exists": {"field": "category"}}]}})
        else:
            must.append({"term": {"category": str(category)}})

    if isinstance(metadata_filter, dict) and metadata_filter:
        for k_meta, v_meta in metadata_filter.items():
            if k_meta is None:
                continue
            must.append({"term": {f"metadata.{str(k_meta)}": v_meta}})

    body = {
        "size": max(1, min(int(top_k or 5), 50)),
        "query": {"bool": {"must": must}},
        "_source": ["vector_db_id", "db_name", "doc_id", "category", "text", "metadata"],
    }
    resp = es.search(index=ES_INDEX, body=body)
    hits = (resp.get("hits") or {}).get("hits") or []
    out = []
    for i, h in enumerate(hits):
        src = h.get("_source") or {}
        out.append(
            {
                "rank": i + 1,
                "score": float(h.get("_score") or 0.0),
                "doc": {
                    "id": src.get("doc_id"),
                    "text": src.get("text"),
                    "category": src.get("category"),
                    "metadata": src.get("metadata"),
                },
            }
        )
    return {"ok": True, "hits": out}


async def bm25_sync_api(request: Request):
    """
    POST body:
    - db_id/vector_db_id 或 db_name
    - batch_size（可选）
    """
    body = await read_json_optional(request) or {}
    db_id = body.get("db_id") or body.get("vector_db_id")
    db_name = (body.get("db_name") or body.get("name") or "").strip() or None
    batch_size = body.get("batch_size", 200)
    if not db_id and not db_name:
        raise ValueError("请提供 db_id/vector_db_id 或 db_name")
    if db_id is None:
        row = VectorDb.select_one_by({"name": db_name})
        if not row:
            raise FileNotFoundError("向量库不存在")
        db_id = row.id
    try:
        db_id = int(db_id)
    except (TypeError, ValueError):
        raise ValueError("db_id 必须为数字")
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        batch_size = 200

    res = sync_vector_db(vector_db_id=db_id, batch_size=max(10, min(batch_size, 2000)))
    return {"code": 0, "msg": "ok", "data": res}

