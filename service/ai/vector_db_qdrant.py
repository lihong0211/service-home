#!/usr/bin/env python3
"""
Qdrant 向量库实现（替代 FAISS + 本地文件）。

- 每个 db_name 对应一个 Qdrant collection。
- Qdrant 的 point id 仅支持 uint64 / UUID；业务 doc_id 任意字符串 → 稳定映射为 UUID5，payload 存真实 doc_id。
- 与 MySQL（vector_db / vector_db_document / vector_db_category）双写；rebuild / sync-from-disk 用于修复不一致。

环境变量：见模块顶部注释（与 vector_db.py 的 DASHSCOPE / VECTOR_DB_DIMENSION 对齐）。
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Optional

from openai import OpenAI

from fastapi import Request
from utils.http_body import query_dict, read_json_optional


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


DIMENSION = int(os.getenv("VECTOR_DB_DIMENSION", "1024"))

# 召回最佳实践开关（默认开启；可用环境变量显式关闭）
DEFAULT_USE_MMR = _env_bool("VECTOR_DB_USE_MMR_DEFAULT", True)
DEFAULT_ENABLE_HYBRID = _env_bool("VECTOR_DB_ENABLE_HYBRID_DEFAULT", True)

# 与 vector_db.py 一致：供 RAG chat 使用 DashScope OpenAI 兼容接口
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30.0,
)

_embedding_client = client

# 业务 doc_id → Qdrant 合法 point id（UUID 字符串）
_POINT_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_uuid(doc_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, str(doc_id).strip()))


DB_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _collection_prefix() -> str:
    return os.getenv("QDRANT_COLLECTION_PREFIX", "vdb_")


def _collection_name(db_name: str) -> str:
    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    return f"{_collection_prefix()}{db_name}"


def _storage_root() -> str:
    """兼容旧版「磁盘向量库」路径；sync-from-disk 仍可读 metadata.json。"""
    return os.path.abspath(
        os.getenv("VECTOR_DB_STORAGE", os.path.join(os.getcwd(), "data", "vector_dbs"))
    )


def _ensure_storage() -> str:
    root = _storage_root()
    os.makedirs(root, exist_ok=True)
    return root


def get_embedding(text: str, max_retries: int = 3) -> list[float]:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            completion = _embedding_client.embeddings.create(
                model="text-embedding-v4",
                input=text,
                dimensions=DIMENSION,
                encoding_format="float",
            )
            return completion.data[0].embedding
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    assert last_exc is not None
    raise last_exc


def _get_qdrant_client():
    global _qdrant_client
    try:
        _qdrant_client  # type: ignore[name-defined]
    except NameError:
        _qdrant_client = None  # type: ignore[name-defined]
    if _qdrant_client is not None:  # type: ignore[name-defined]
        return _qdrant_client  # type: ignore[name-defined]

    from qdrant_client import QdrantClient

    url = (os.getenv("QDRANT_URL") or "http://localhost:6333").strip()
    api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
    prefer_grpc = _env_bool("QDRANT_PREFER_GRPC", False)
    grpc_url = (os.getenv("QDRANT_GRPC_URL") or "").strip() or None
    if prefer_grpc and grpc_url:
        _qdrant_client = QdrantClient(url=grpc_url, api_key=api_key, prefer_grpc=True)  # type: ignore[name-defined]
    else:
        _qdrant_client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)  # type: ignore[name-defined]
    return _qdrant_client  # type: ignore[name-defined]


def _collection_physical_exists(collection_name: str) -> bool:
    client = _get_qdrant_client()
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name=collection_name))
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def _ensure_collection(db_name: str) -> str:
    from qdrant_client.http.models import Distance, VectorParams

    client = _get_qdrant_client()
    name = _collection_name(db_name)
    if hasattr(client, "collection_exists") and client.collection_exists(collection_name=name):
        info = client.get_collection(name)
        vectors_cfg = getattr(info.config.params, "vectors", None)
        size = getattr(vectors_cfg, "size", None) if vectors_cfg is not None else None
        if size is not None and int(size) != int(DIMENSION):
            if _env_bool("QDRANT_RECREATE_COLLECTION_ON_DIM_MISMATCH", False):
                client.delete_collection(name)
            else:
                raise ValueError(
                    f"Qdrant collection 维度不匹配: {name} size={size} != {DIMENSION}. "
                    f"如确需重建，请设置 QDRANT_RECREATE_COLLECTION_ON_DIM_MISMATCH=1"
                )
        _ensure_payload_indexes(name)
        return name
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
    )
    _ensure_payload_indexes(name)
    return name


def _ensure_payload_indexes(collection_name: str) -> None:
    """
    为 payload 字段创建索引，以支持过滤与关键词兜底检索。
    - category: keyword
    - text: full-text（若客户端/服务端不支持则忽略）
    """
    client = _get_qdrant_client()
    if not hasattr(client, "create_payload_index"):
        return
    try:
        client.create_payload_index(collection_name=collection_name, field_name="category", field_schema="keyword")
    except Exception:
        pass
    # full-text 索引：不同版本 qdrant-client 参数差异较大，失败时直接忽略，不影响主流程
    try:
        client.create_payload_index(collection_name=collection_name, field_name="text", field_schema="text")
    except Exception:
        pass


def _normalize_documents(documents: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for i, doc in enumerate(documents or []):
        if not isinstance(doc, dict):
            continue
        text = doc.get("text") or doc.get("content") or ""
        text = str(text).strip()
        if not text:
            continue
        doc_id = str(doc.get("id", f"doc_{i}")).strip()
        if not doc_id:
            doc_id = f"doc_{i}"
        category = (doc.get("category") or "").strip() or None
        item: dict[str, Any] = {"id": doc_id, "text": text, "category": category}
        if "metadata" in doc and doc["metadata"] is not None:
            item["metadata"] = doc["metadata"] if isinstance(doc["metadata"], dict) else {}
        normalized.append(item)
    return normalized


def _upsert_points(db_name: str, docs: list[dict], batch_size: int = 64) -> int:
    from qdrant_client.http.models import PointStruct

    if not docs:
        return 0
    client = _get_qdrant_client()
    collection = _ensure_collection(db_name)
    total = 0
    batch: list[PointStruct] = []
    for doc in docs:
        vec = get_embedding(doc["text"])
        payload: dict[str, Any] = {
            "doc_id": str(doc["id"]),
            "text": doc["text"],
            "category": doc.get("category"),
        }
        if "metadata" in doc and isinstance(doc["metadata"], dict):
            payload["metadata"] = doc["metadata"]
        pid = _point_uuid(str(doc["id"]))
        batch.append(PointStruct(id=pid, vector=vec, payload=payload))
        if len(batch) >= batch_size:
            client.upsert(collection_name=collection, points=batch, wait=True)
            total += len(batch)
            batch = []
    if batch:
        client.upsert(collection_name=collection, points=batch, wait=True)
        total += len(batch)
    return total


def _delete_points_by_doc_ids(db_name: str, doc_ids: list[str]) -> int:
    from qdrant_client.http.models import PointIdsList

    if not doc_ids:
        return 0
    client = _get_qdrant_client()
    collection = _ensure_collection(db_name)
    ids = [_point_uuid(str(x).strip()) for x in doc_ids if str(x).strip()]
    if not ids:
        return 0
    client.delete(collection_name=collection, points_selector=PointIdsList(points=ids), wait=True)
    return len(ids)


def delete_vector_db_collection(db_name: str) -> None:
    client = _get_qdrant_client()
    name = _collection_name(db_name)
    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass


def _rename_qdrant_collection(old_db_name: str, new_db_name: str) -> None:
    """库改名时复制向量数据到新 collection 并删除旧 collection。"""
    if old_db_name == new_db_name:
        return
    from qdrant_client.http.models import PointStruct

    client = _get_qdrant_client()
    old_c = _collection_name(old_db_name)
    new_c = _collection_name(new_db_name)
    if not _collection_physical_exists(old_c):
        _ensure_collection(new_db_name)
        return
    _ensure_collection(new_db_name)
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=old_c,
            limit=128,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break
        points = [
            PointStruct(id=r.id, vector=r.vector, payload=r.payload or {})
            for r in records
        ]
        client.upsert(collection_name=new_c, points=points, wait=True)
        if next_offset is None:
            break
    try:
        client.delete_collection(collection_name=old_c)
    except Exception:
        pass


def search_in_db(
    db_name: str,
    query: str,
    top_k: int = 3,
    category: str | None = None,
    metadata_filter: dict | None = None,
    enable_hybrid: bool | None = None,
    use_mmr: bool | None = None,
    mmr_lambda: float = 0.5,
    candidate_k: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    from qdrant_client.http.models import FieldCondition, Filter, IsEmptyCondition, MatchValue, PayloadField

    if not (query or "").strip():
        return []
    cname = _collection_name(db_name)
    if not _collection_physical_exists(cname):
        return []
    client = _get_qdrant_client()
    collection = cname
    k = max(1, min(int(top_k or 3), 100))
    if enable_hybrid is None:
        enable_hybrid = DEFAULT_ENABLE_HYBRID
    if use_mmr is None:
        use_mmr = DEFAULT_USE_MMR
    fetch_k = int(candidate_k or (max(k, min(50, k * 5)) if (use_mmr or enable_hybrid) else k))
    fetch_k = max(k, min(fetch_k, 200))

    must: list[Any] = []
    if category is not None:
        if category == "":
            must.append(IsEmptyCondition(is_empty=PayloadField(key="category")))
        else:
            must.append(FieldCondition(key="category", match=MatchValue(value=str(category))))

    if isinstance(metadata_filter, dict) and metadata_filter:
        for k_meta, v_meta in metadata_filter.items():
            if k_meta is None:
                continue
            key = f"metadata.{str(k_meta)}"
            must.append(FieldCondition(key=key, match=MatchValue(value=v_meta)))

    q_filter = Filter(must=must) if must else None
    query_vec = get_embedding(query)

    def _run_dense(limit: int, with_vectors: bool = False):
        # qdrant-client >= 1.17 移除 search，统一走 query_points；旧版仍可能仅有 search
        if hasattr(client, "query_points"):
            qresp = client.query_points(
                collection_name=collection,
                query=query_vec,
                limit=limit,
                with_payload=True,
                with_vectors=with_vectors,
                query_filter=q_filter,
                score_threshold=score_threshold,
            )
            return getattr(qresp, "points", None) or []
        return (
            client.search(
                collection_name=collection,
                query_vector=query_vec,
                limit=limit,
                with_payload=True,
                with_vectors=with_vectors,
                query_filter=q_filter,
                score_threshold=score_threshold,
            )
            or []
        )

    def _payload_to_doc(hit) -> dict:
        payload = getattr(hit, "payload", None) or {}
        doc = {
            "id": payload.get("doc_id") or str(getattr(hit, "id", "")),
            "text": payload.get("text"),
            "category": payload.get("category"),
        }
        if "metadata" in payload:
            doc["metadata"] = payload.get("metadata")
        return doc

    def _hit_score(hit) -> float:
        return float(getattr(hit, "score", 0.0))

    def _hit_vector(hit) -> Optional[list[float]]:
        v = getattr(hit, "vector", None)
        if v is None:
            return None
        # qdrant-client 可能返回 list 或 dict（多向量场景）；此处只取单向量
        if isinstance(v, dict):
            # 取第一个向量值
            for _, vv in v.items():
                if isinstance(vv, list):
                    return vv
            return None
        return v if isinstance(v, list) else None

    def _mmr_select(hits: list, limit: int) -> list:
        # cosine 相似度：score 越大越相似。MMR 需要候选向量；拿不到就退化为原排序
        import math

        vecs: list[Optional[list[float]]] = [_hit_vector(h) for h in hits]
        if not any(vecs):
            return hits[:limit]

        def cos(a: list[float], b: list[float]) -> float:
            # 避免重复归一化成本：embedding 通常已近似归一化，这里做安全归一化
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return sum(x * y for x, y in zip(a, b)) / (na * nb)

        selected: list[int] = []
        candidates = list(range(len(hits)))
        qv = query_vec

        def rel_score(i: int) -> float:
            v = vecs[i]
            return _hit_score(hits[i]) if v is None else cos(qv, v)

        while candidates and len(selected) < limit:
            best_i: Optional[int] = None
            best_score = -1e9
            for i in candidates:
                rel = rel_score(i)
                if not selected:
                    mmr = rel
                else:
                    max_div = -1.0
                    vi = vecs[i]
                    if vi is None:
                        max_div = max(_hit_score(hits[j]) for j in selected)
                    else:
                        for j in selected:
                            vj = vecs[j]
                            if vj is None:
                                continue
                            max_div = max(max_div, cos(vi, vj))
                    mmr = float(mmr_lambda) * rel - (1.0 - float(mmr_lambda)) * max_div
                if mmr > best_score:
                    best_score = mmr
                    best_i = i
            if best_i is None:
                break
            selected.append(best_i)
            candidates = [x for x in candidates if x != best_i]
        return [hits[i] for i in selected]

    dense_hits = _run_dense(fetch_k, with_vectors=bool(use_mmr))

    merged_hits = list(dense_hits)

    # Hybrid keyword 兜底：基于 text 的 full-text 匹配（若不支持则自动跳过）
    if enable_hybrid:
        try:
            from qdrant_client.http.models import MatchText

            kw_filter = Filter(
                must=[*(must or []), FieldCondition(key="text", match=MatchText(text=str(query).strip()))]
            )
            if hasattr(client, "query_points"):
                kw_resp = client.query_points(
                    collection_name=collection,
                    query=query_vec,
                    limit=min(fetch_k, 50),
                    with_payload=True,
                    query_filter=kw_filter,
                    score_threshold=score_threshold,
                )
                kw_hits = getattr(kw_resp, "points", None) or []
            else:
                kw_hits = client.search(
                    collection_name=collection,
                    query_vector=query_vec,
                    limit=min(fetch_k, 50),
                    with_payload=True,
                    query_filter=kw_filter,
                    score_threshold=score_threshold,
                ) or []
            merged_hits.extend(kw_hits)
        except Exception:
            pass

    # 按 doc_id 去重并保留更高 score
    best_by_doc: dict[str, Any] = {}
    for h in merged_hits:
        doc = _payload_to_doc(h)
        did = str(doc.get("id") or "")
        if not did:
            continue
        if did not in best_by_doc or _hit_score(h) > _hit_score(best_by_doc[did]):
            best_by_doc[did] = h
    dedup_hits = sorted(best_by_doc.values(), key=_hit_score, reverse=True)

    final_hits = dedup_hits[:fetch_k]
    if use_mmr and len(final_hits) > k:
        final_hits = _mmr_select(final_hits, k)
    else:
        final_hits = final_hits[:k]

    results: list[dict] = []
    for h in final_hits:
        score = _hit_score(h)
        distance = max(0.0, 1.0 - score)
        results.append(
            {
                "doc": _payload_to_doc(h),
                "score": score,
                "distance": distance,
                "rank": len(results) + 1,
            }
        )
    return results


# -----------------------------
# MySQL
# -----------------------------


def list_vector_dbs_from_mysql() -> list[dict]:
    from model.ai import VectorDb

    rows = VectorDb.select_by({"order_by": [{"col": "id", "sort": "desc"}]})
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": (r.description or "").strip() or None,
            "create_at": r.create_at.isoformat() if r.create_at else None,
            "update_at": r.update_at.isoformat() if r.update_at else None,
        }
        for r in rows
    ]


def list_vector_dbs() -> list[str]:
    """列出向量库名：以 MySQL 为准（与旧版「仅磁盘 list」相比更贴近管理端）。"""
    return [x["name"] for x in list_vector_dbs_from_mysql()]


def _save_documents_to_mysql(vector_db_id: int, documents: list[dict]) -> None:
    from model.ai import VectorDbDocument

    VectorDbDocument.force_delete({"vector_db_id": vector_db_id})
    if not documents:
        return
    for doc in documents:
        row = {
            "vector_db_id": vector_db_id,
            "doc_id": str(doc.get("id", "")),
            "text": doc.get("text", ""),
            "category": (doc.get("category") or "").strip() or None,
        }
        if "metadata" in doc and doc["metadata"] is not None and isinstance(doc["metadata"], dict):
            row["document_metadata"] = doc["metadata"]
        VectorDbDocument.insert(row)


def _append_documents_to_mysql(vector_db_id: int, documents: list[dict]) -> None:
    from model.ai import VectorDbDocument

    for doc in documents:
        row = {
            "vector_db_id": vector_db_id,
            "doc_id": str(doc.get("id", "")),
            "text": doc.get("text", ""),
            "category": (doc.get("category") or "").strip() or None,
        }
        if "metadata" in doc and doc["metadata"] is not None and isinstance(doc["metadata"], dict):
            row["document_metadata"] = doc["metadata"]
        VectorDbDocument.insert(row)


def _sync_categories_from_documents(vector_db_id: int, documents: list[dict]) -> None:
    from model.ai import VectorDbCategory

    names = set()
    for doc in documents or []:
        cat = (doc.get("category") or "").strip()
        if cat:
            names.add(cat)
    if not names:
        return
    existing = {
        (r.name or "").strip()
        for r in VectorDbCategory.select_by({"vector_db_id": vector_db_id})
    }
    for i, name in enumerate(sorted(names)):
        if name not in existing:
            VectorDbCategory.insert({"vector_db_id": vector_db_id, "name": name, "sort_order": i})
            existing.add(name)


def list_documents(db_id: int | None = None, db_name: str | None = None) -> list[dict]:
    from model.ai import VectorDb, VectorDbDocument

    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 db_id 或 db_name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    rows = VectorDbDocument.select_by({"vector_db_id": row.id, "order_by": [{"col": "id", "sort": "asc"}]})
    return [
        {
            "id": r.id,
            "vector_db_id": r.vector_db_id,
            "doc_id": r.doc_id,
            "text": r.text,
            "category": (r.category or "").strip() or None,
            "metadata": getattr(r, "document_metadata", None),
            "create_at": r.create_at.isoformat() if r.create_at else None,
        }
        for r in rows
    ]


def list_documents_paginated(
    db_id: int | None = None,
    db_name: str | None = None,
    page: int = 1,
    page_size: int = 20,
    category: str | None = None,
) -> dict:
    from model.ai import VectorDb, VectorDbDocument

    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 db_id 或 db_name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    criterion = {"vector_db_id": row.id, "order_by": [{"col": "id", "sort": "asc"}]}

    def _build_query():
        q = VectorDbDocument.builder_query(criterion)
        if category is not None:
            if category == "":
                q = q.where(VectorDbDocument.category.is_(None))
            else:
                q = q.where(VectorDbDocument.category == category)
        return q

    total = _build_query().with_entities(VectorDbDocument.id).count()
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size
    rows = _build_query().limit(page_size).offset(offset).all()
    list_ = [
        {
            "id": r.id,
            "vector_db_id": r.vector_db_id,
            "doc_id": r.doc_id,
            "text": r.text,
            "category": (r.category or "").strip() or None,
            "metadata": getattr(r, "document_metadata", None),
            "create_at": r.create_at.isoformat() if r.create_at else None,
        }
        for r in rows
    ]
    return {"list": list_, "total": total, "page": page, "page_size": page_size}


def get_vector_db_detail(db_id: int | None = None, db_name: str | None = None, with_documents: bool = False) -> dict:
    from model.ai import VectorDb

    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 id 或 name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    out = {
        "id": row.id,
        "name": row.name,
        "description": (row.description or "").strip() or None,
        "create_at": row.create_at.isoformat() if row.create_at else None,
        "update_at": row.update_at.isoformat() if row.update_at else None,
        "documents": [],
    }
    if with_documents:
        doc_list = list_documents(db_id=row.id)
        out["documents"] = [
            {"id": r["doc_id"], "text": r["text"], "category": r.get("category")}
            for r in doc_list
        ]
    return out


def list_categories(db_id: int | None = None, db_name: str | None = None) -> list[dict]:
    from model.ai import VectorDb, VectorDbCategory

    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 db_id 或 db_name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    rows = VectorDbCategory.select_by({
        "vector_db_id": row.id,
        "order_by": [{"col": "sort_order", "sort": "asc"}, {"col": "id", "sort": "asc"}],
    })
    return [
        {
            "id": r.id,
            "vector_db_id": r.vector_db_id,
            "name": r.name,
            "sort_order": r.sort_order if r.sort_order is not None else 0,
            "create_at": r.create_at.isoformat() if r.create_at else None,
        }
        for r in rows
    ]


def add_category(
    db_id: int | None = None,
    db_name: str | None = None,
    name: str | None = None,
    sort_order: int | None = None,
) -> dict:
    from model.ai import VectorDb, VectorDbCategory

    if not (name or "").strip():
        raise ValueError("分类名称不能为空")
    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 db_id 或 db_name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    name = str(name).strip()
    cat_id = VectorDbCategory.insert({
        "vector_db_id": row.id,
        "name": name,
        "sort_order": sort_order if sort_order is not None else 0,
    })
    return {"id": cat_id, "vector_db_id": row.id, "name": name, "sort_order": sort_order or 0}


def update_category(category_id: int, name: str | None = None, sort_order: int | None = None) -> dict:
    from model.ai import VectorDbCategory

    row = VectorDbCategory.get_by_id(category_id)
    if not row:
        raise FileNotFoundError("分类不存在")
    update_data = {"id": category_id}
    if name is not None:
        update_data["name"] = str(name).strip()
    if sort_order is not None:
        update_data["sort_order"] = int(sort_order)
    if len(update_data) > 1:
        VectorDbCategory.update(update_data)
    row = VectorDbCategory.get_by_id(category_id)
    return {"id": row.id, "vector_db_id": row.vector_db_id, "name": row.name, "sort_order": row.sort_order or 0}


def delete_category(category_id: int) -> dict:
    from model.ai import VectorDbCategory

    row = VectorDbCategory.get_by_id(category_id)
    if not row:
        raise FileNotFoundError("分类不存在")
    VectorDbCategory.force_delete({"id": category_id})
    return {"id": category_id}


def load_vector_db(db_name: str) -> dict:
    """供知识库增量向量化：返回 { metadata: [{id,text,category,...}] }，来源 MySQL。"""
    from model.ai import VectorDb

    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    row = VectorDb.select_one_by({"name": db_name})
    if not row:
        raise FileNotFoundError(f"向量库不存在: {db_name}")
    doc_list = list_documents(db_id=row.id)
    metadata = []
    for r in doc_list:
        item = {
            "id": str(r["doc_id"]),
            "text": r["text"],
            "category": r.get("category"),
        }
        meta = r.get("metadata")
        if isinstance(meta, dict):
            item["metadata"] = meta
        metadata.append(item)
    return {"index": None, "metadata": metadata}


def _create_empty_vector_db_on_disk(db_name: str) -> str:
    """兼容 knowledge 模块命名：仅确保 Qdrant collection 存在。"""
    _ensure_collection(db_name)
    return f"qdrant://{_collection_name(db_name)}"


def _delete_vector_db_from_disk(db_name: str) -> None:
    delete_vector_db_collection(db_name)


def create_vector_db(db_name: str, documents: list[dict] | None = None) -> dict:
    """
    创建或全量替换向量数据。
    - documents 非空：删除并重建 collection 后 upsert（去除已删文档的残留 point）。
    - documents 为空：仅创建空 collection。
    """
    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    docs = _normalize_documents(documents or [])
    cname = _collection_name(db_name)
    path = f"qdrant://{cname}"
    if not docs:
        _ensure_collection(db_name)
        return {"count": 0, "path": path, "collection": cname, "documents": []}
    client = _get_qdrant_client()
    try:
        client.delete_collection(collection_name=cname)
    except Exception:
        pass
    _ensure_collection(db_name)
    n = _upsert_points(db_name, docs)
    return {"count": n, "path": path, "collection": cname, "documents": docs}


def sync_vector_db_from_disk(db_name: str, description: str | None = None) -> dict:
    """读取旧版 FAISS 目录下的 metadata.json，写入 MySQL 并 upsert 到 Qdrant。"""
    from model.ai import VectorDb

    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    meta_path = os.path.join(_storage_root(), db_name, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"磁盘上不存在该向量库: {db_name}")
    with open(meta_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    if not documents:
        raise ValueError("磁盘上文档列表为空")
    docs = _normalize_documents(documents)
    if not docs:
        raise ValueError("没有有效的文档")
    row = VectorDb.select_one_by({"name": db_name})
    if row:
        vector_db_id = row.id
        create_vector_db(db_name, docs)
        _save_documents_to_mysql(vector_db_id, docs)
        _sync_categories_from_documents(vector_db_id, docs)
        if description is not None:
            VectorDb.update({"id": vector_db_id, "description": (description or "").strip() or None})
        return {"id": vector_db_id, "name": db_name, "count": len(docs), "synced": "qdrant_and_documents"}
    row_id = VectorDb.insert({"name": db_name, "description": (description or "").strip() or None})
    create_vector_db(db_name, docs)
    _save_documents_to_mysql(row_id, docs)
    _sync_categories_from_documents(row_id, docs)
    return {"id": row_id, "name": db_name, "count": len(docs), "synced": "db_qdrant_and_documents"}


def _rebuild_vector_db_index(vector_db_id: int, documents: list[dict]) -> None:
    from model.ai import VectorDb

    row = VectorDb.get_by_id(vector_db_id)
    if not row:
        raise FileNotFoundError("向量库不存在")
    db_name = row.name
    if not documents:
        delete_vector_db_collection(db_name)
        _ensure_collection(db_name)
        _save_documents_to_mysql(vector_db_id, [])
        return
    docs = _normalize_documents(documents)
    if not docs:
        raise ValueError("没有有效的文档")
    create_vector_db(db_name, docs)
    _save_documents_to_mysql(vector_db_id, docs)
    _sync_categories_from_documents(vector_db_id, docs)


def append_documents_batch(vector_db_id: int, documents: list[dict]) -> int:
    from model.ai import VectorDb

    row = VectorDb.get_by_id(vector_db_id)
    if not row:
        raise FileNotFoundError("向量库不存在")
    db_name = row.name
    db = load_vector_db(db_name)
    existing_ids = {str(d.get("id", "")) for d in (db.get("metadata") or [])}
    normalized = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("id", f"doc_{i}")).strip()
        if doc_id in existing_ids:
            continue
        text = (doc.get("text") or doc.get("content") or "").strip()
        if not text:
            continue
        category = (doc.get("category") or "").strip() or None
        item = {"id": doc_id, "text": text, "category": category}
        if "metadata" in doc and doc["metadata"] is not None:
            item["metadata"] = doc["metadata"] if isinstance(doc["metadata"], dict) else {}
        normalized.append(item)
    if not normalized:
        return 0
    _ensure_collection(db_name)
    n = _upsert_points(db_name, normalized)
    _append_documents_to_mysql(vector_db_id, normalized)
    _sync_categories_from_documents(vector_db_id, normalized)
    return n


def delete_vector_db_by_id(db_id: int) -> str:
    from model.ai import VectorDb, VectorDbDocument, VectorDbCategory

    row = VectorDb.get_by_id(db_id)
    if not row:
        raise FileNotFoundError("向量库不存在")
    name = row.name
    VectorDbCategory.force_delete({"vector_db_id": db_id})
    VectorDbDocument.force_delete({"vector_db_id": db_id})
    VectorDb.force_delete({"id": db_id})
    delete_vector_db_collection(name)
    return name


def add_single_document(
    db_id: int | None = None,
    db_name: str | None = None,
    doc_id: str | None = None,
    text: str | None = None,
    category: str | None = None,
) -> dict:
    from model.ai import VectorDb, VectorDbDocument

    if not (text or "").strip():
        raise ValueError("text 必填")
    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 db_id 或 db_name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    vector_db_id = row.id
    name = row.name
    text = str(text).strip()
    cat = (category or "").strip() or None
    db = load_vector_db(name)
    metadata_store = list(db["metadata"])
    existing_ids = {str(d.get("id", "")) for d in metadata_store}
    insert_mysql_after = False
    if (doc_id or "").strip():
        new_id = str(doc_id).strip()
        if new_id in existing_ids:
            raise ValueError(f"doc_id 已存在: {new_id}，请换一个或使用更新接口")
        insert_mysql_after = True
    else:
        new_row_id = VectorDbDocument.insert({
            "vector_db_id": vector_db_id,
            "doc_id": "_",
            "text": text,
            "category": cat,
        })
        new_id = str(new_row_id)
        VectorDbDocument.update({"id": new_row_id, "doc_id": new_id})
    _ensure_collection(name)
    doc_dict = {"id": new_id, "text": text, "category": cat}
    _upsert_points(name, [doc_dict])
    if insert_mysql_after:
        VectorDbDocument.insert({
            "vector_db_id": vector_db_id,
            "doc_id": str(new_id),
            "text": text,
            "category": cat,
        })
    if cat:
        _sync_categories_from_documents(vector_db_id, [{"category": cat}])
    return {"doc_id": str(new_id), "db_name": name, "total": len(metadata_store) + 1}


def update_single_document(
    db_id: int | None = None,
    db_name: str | None = None,
    doc_id: str | None = None,
    index: int | None = None,
    text: str | None = None,
    category: str | None = None,
) -> dict:
    from model.ai import VectorDb, VectorDbDocument

    if not (text or "").strip():
        raise ValueError("text 必填")
    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 db_id 或 db_name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    vector_db_id = row.id
    name = row.name
    metadata_store = [m for m in (load_vector_db(name).get("metadata") or [])]
    idx = None
    if doc_id is not None and str(doc_id).strip():
        for i, doc in enumerate(metadata_store):
            if str(doc.get("id", "")) == str(doc_id):
                idx = i
                break
        if idx is None:
            raise ValueError(f"文档不存在: doc_id={doc_id}")
    elif index is not None and 0 <= index < len(metadata_store):
        idx = index
        doc_id = metadata_store[idx].get("id", f"doc_{index}")
    else:
        raise ValueError("请提供 doc_id 或有效的 index（0-based）")
    text = str(text).strip()
    cat = (category or "").strip() or None
    doc_id = str(doc_id)
    _ensure_collection(name)
    doc_dict = {"id": doc_id, "text": text, "category": cat}
    if "metadata" in metadata_store[idx] and isinstance(metadata_store[idx].get("metadata"), dict):
        doc_dict["metadata"] = metadata_store[idx]["metadata"]
    _upsert_points(name, [doc_dict])
    doc_rows = VectorDbDocument.select_by({"vector_db_id": vector_db_id, "doc_id": str(doc_id)})
    if doc_rows:
        upd = {"id": doc_rows[0].id, "text": text, "category": cat}
        if "metadata" in doc_dict:
            upd["document_metadata"] = doc_dict["metadata"]
        VectorDbDocument.update(upd)
    else:
        row_ins = {
            "vector_db_id": vector_db_id,
            "doc_id": str(doc_id),
            "text": text,
            "category": cat,
        }
        if "metadata" in doc_dict:
            row_ins["document_metadata"] = doc_dict["metadata"]
        VectorDbDocument.insert(row_ins)
    if cat:
        _sync_categories_from_documents(vector_db_id, [{"category": cat}])
    return {"doc_id": str(doc_id), "db_name": name}


def delete_single_document(
    db_id: int | None = None,
    db_name: str | None = None,
    doc_id: str | None = None,
) -> dict:
    from model.ai import VectorDb, VectorDbDocument

    if not (doc_id or "").strip():
        raise ValueError("请提供 doc_id")
    doc_id = str(doc_id).strip()
    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 db_id 或 db_name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    vector_db_id = row.id
    name = row.name
    metadata_store = [m for m in (load_vector_db(name).get("metadata") or [])]
    found = False
    for doc in metadata_store:
        if str(doc.get("id", "")) == doc_id:
            found = True
            break
    if not found:
        raise FileNotFoundError(f"文档不存在: doc_id={doc_id}")
    _delete_points_by_doc_ids(name, [doc_id])
    VectorDbDocument.force_delete({"vector_db_id": vector_db_id, "doc_id": doc_id})
    return {"doc_id": doc_id, "db_name": name, "total": len(metadata_store) - 1}


def rebuild_vector_db_from_mysql(db_id: int | None = None, db_name: str | None = None) -> dict:
    from model.ai import VectorDb

    if db_id is not None:
        row = VectorDb.get_by_id(db_id)
    elif db_name:
        row = VectorDb.select_one_by({"name": db_name})
    else:
        raise ValueError("请提供 db_id 或 db_name")
    if not row:
        raise FileNotFoundError("向量库不存在")
    doc_list = list_documents(db_id=row.id)
    if not doc_list:
        raise ValueError("该库下没有文档，无法重建")
    documents = [
        {"id": r["doc_id"], "text": r["text"], "category": r.get("category")}
        for r in doc_list
    ]
    for r in doc_list:
        meta = r.get("metadata")
        if isinstance(meta, dict):
            for i, d in enumerate(documents):
                if str(d["id"]) == str(r["doc_id"]):
                    documents[i]["metadata"] = meta
                    break
    out = create_vector_db(row.name, documents)
    _save_documents_to_mysql(row.id, out.get("documents") or [])
    _sync_categories_from_documents(row.id, out.get("documents") or [])
    return {"id": row.id, "name": row.name, "count": out["count"]}


# ---------- HTTP ----------


async def list_api(request: Request):
    items = list_vector_dbs_from_mysql()
    return {"code": 0, "msg": "ok", "data": {"list": items, "names": [x["name"] for x in items]}}


async def create_api(request: Request):
    from model.ai import VectorDb

    data = await read_json_optional(request) or {}
    name = (data.get("name") or data.get("db") or "").strip()
    if not name:
        raise ValueError("缺少参数 name 或 db")
    if not DB_NAME_PATTERN.match(name):
        raise ValueError("库名仅允许 a-zA-Z0-9_-")
    documents = data.get("documents") if data.get("documents") is not None else []
    description = (data.get("description") or "").strip() or None
    if VectorDb.select_one_by({"name": name}):
        raise ValueError(f"库名已存在: {name}")
    row_id = VectorDb.insert({"name": name, "description": description})
    try:
        out = create_vector_db(name, documents)
        docs = out.get("documents") or []
        _save_documents_to_mysql(row_id, docs)
        if docs:
            _sync_categories_from_documents(row_id, docs)
        return {
            "code": 0,
            "msg": "ok",
            "data": {
                "id": row_id,
                "name": name,
                "description": description,
                "count": out["count"],
                "path": out["path"],
                "collection": out.get("collection"),
            },
        }
    except Exception as e_vec:
        VectorDb.force_delete({"id": row_id})
        _delete_vector_db_from_disk(name)
        raise e_vec


async def detail_api(request: Request):
    db_id = query_dict(request).get("id")
    db_name = query_dict(request).get("name")
    with_documents = query_dict(request).get("with_documents", "0") in ("1", "true", "yes")
    if not db_id and not db_name:
        raise ValueError("请提供 id 或 name")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("id 必须为数字")
    else:
        db_id = None
    detail = get_vector_db_detail(db_id=db_id, db_name=db_name, with_documents=with_documents)
    return {"code": 0, "msg": "ok", "data": detail}


async def update_api(request: Request):
    from model.ai import VectorDb

    data = await read_json_optional(request) or {}
    db_id = data.get("id")
    if db_id is None:
        raise ValueError("缺少参数 id")
    try:
        db_id = int(db_id)
    except (TypeError, ValueError):
        raise ValueError("id 必须为数字")
    row = VectorDb.get_by_id(db_id)
    if not row:
        raise FileNotFoundError("向量库不存在")
    name = row.name
    documents = data.get("documents", [])
    if not documents:
        raise ValueError("documents 不能为空")
    description = data.get("description")
    if description is not None:
        description = str(description).strip() or None
    out = create_vector_db(name, documents)
    docs = out.get("documents") or []
    _save_documents_to_mysql(db_id, docs)
    _sync_categories_from_documents(db_id, docs)
    update_data = {"id": db_id}
    if description is not None:
        update_data["description"] = description
    if len(update_data) > 1:
        VectorDb.update(update_data)
    return {"code": 0, "msg": "ok", "data": {"id": db_id, "name": name, "count": out["count"]}}


async def update_meta_api(request: Request):
    from model.ai import VectorDb

    data = await read_json_optional(request) or {}
    db_id = data.get("id")
    if db_id is None:
        raise ValueError("缺少参数 id")
    try:
        db_id = int(db_id)
    except (TypeError, ValueError):
        raise ValueError("id 必须为数字")
    row = VectorDb.get_by_id(db_id)
    if not row:
        raise FileNotFoundError("向量库不存在")
    old_name = row.name
    update_data = {"id": db_id}
    if data.get("description") is not None:
        desc = (data.get("description") or "").strip() or None
        update_data["description"] = desc
    new_name = None
    if data.get("name") is not None:
        name = (str(data.get("name") or "").strip())
        if not name:
            raise ValueError("name 不能为空")
        if not DB_NAME_PATTERN.match(name):
            raise ValueError("库名仅允许 a-zA-Z0-9_-")
        other = VectorDb.select_one_by({"name": name})
        if other and other.id != db_id:
            raise ValueError(f"库名已存在: {name}")
        update_data["name"] = name
        new_name = name
    if len(update_data) > 1:
        VectorDb.update(update_data)
    if new_name is not None and new_name != old_name:
        _rename_qdrant_collection(old_name, new_name)
    row = VectorDb.get_by_id(db_id)
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "id": row.id,
            "name": row.name,
            "description": (row.description or "").strip() or None,
        },
    }


async def delete_api(request: Request):
    data = await read_json_optional(request) or {}
    db_id = data.get("id")
    if db_id is None:
        raise ValueError("缺少参数 id")
    try:
        db_id = int(db_id)
    except (TypeError, ValueError):
        raise ValueError("id 必须为数字")
    name = delete_vector_db_by_id(db_id)
    return {"code": 0, "msg": "ok", "data": {"id": db_id, "name": name}}


async def sync_from_disk_api(request: Request):
    data = await read_json_optional(request) or {}
    name = (data.get("name") or data.get("db") or "").strip()
    if not name:
        raise ValueError("缺少参数 name 或 db")
    if not DB_NAME_PATTERN.match(name):
        raise ValueError("库名仅允许 a-zA-Z0-9_-")
    description = (data.get("description") or "").strip() or None
    out = sync_vector_db_from_disk(name, description=description)
    return {"code": 0, "msg": "ok", "data": out}


async def rebuild_api(request: Request):
    data = await read_json_optional(request) or {}
    db_id = data.get("id")
    db_name = (data.get("name") or data.get("db") or "").strip() or None
    if not db_id and not db_name:
        raise ValueError("请提供 id 或 name/db")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("id 必须为数字")
    else:
        db_id = None
    out = rebuild_vector_db_from_mysql(db_id=db_id, db_name=db_name)
    return {"code": 0, "msg": "ok", "data": out}


async def documents_api(request: Request):
    db_id = query_dict(request).get("db_id")
    db_name = (query_dict(request).get("db_name") or "").strip() or None
    if not db_id and not db_name:
        raise ValueError("请提供 db_id 或 db_name")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("db_id 必须为数字")
    else:
        db_id = None
    try:
        page = int(query_dict(request).get("page") or 1)
        page_size = int(query_dict(request).get("page_size") or 20)
    except (TypeError, ValueError):
        page, page_size = 1, 20
    category = query_dict(request).get("category")
    out = list_documents_paginated(db_id=db_id, db_name=db_name, page=page, page_size=page_size, category=category)
    return {"code": 0, "msg": "ok", "data": out}


async def document_add_api(request: Request):
    data = await read_json_optional(request) or {}
    db_id = data.get("db_id")
    db_name = (data.get("db_name") or "").strip() or None
    text = data.get("text")
    doc_id = data.get("doc_id")
    if doc_id is not None:
        doc_id = str(doc_id).strip() or None
    category = data.get("category")
    if not text or not str(text).strip():
        raise ValueError("缺少参数 text")
    if not db_id and not db_name:
        raise ValueError("请提供 db_id 或 db_name")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("db_id 必须为数字")
    else:
        db_id = None
    out = add_single_document(
        db_id=db_id,
        db_name=db_name,
        doc_id=doc_id,
        text=str(text).strip(),
        category=(category or "").strip() or None if category is not None else None,
    )
    return {"code": 0, "msg": "ok", "data": out}


async def document_update_api(request: Request):
    data = await read_json_optional(request) or {}
    db_id = data.get("db_id")
    db_name = (data.get("db_name") or "").strip() or None
    doc_id = data.get("doc_id")
    if doc_id is not None:
        doc_id = str(doc_id).strip() or None
    index = data.get("index")
    if index is not None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = None
    text = data.get("text")
    category = data.get("category")
    if not doc_id and index is None:
        raise ValueError("请提供 doc_id 或 index")
    if not text or not str(text).strip():
        raise ValueError("缺少参数 text")
    if not db_id and not db_name:
        raise ValueError("请提供 db_id 或 db_name")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("db_id 必须为数字")
    else:
        db_id = None
    out = update_single_document(
        db_id=db_id,
        db_name=db_name,
        doc_id=doc_id,
        index=index,
        text=str(text).strip(),
        category=(category or "").strip() or None if category is not None else None,
    )
    return {"code": 0, "msg": "ok", "data": out}


async def document_delete_api(request: Request):
    data = await read_json_optional(request) or {}
    db_id = data.get("db_id")
    db_name = (data.get("db_name") or "").strip() or None
    doc_id = data.get("doc_id")
    if doc_id is None:
        raise ValueError("请提供 doc_id")
    doc_id = str(doc_id).strip()
    if not doc_id:
        raise ValueError("doc_id 不能为空")
    if not db_id and not db_name:
        raise ValueError("请提供 db_id 或 db_name")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("db_id 必须为数字")
    else:
        db_id = None
    out = delete_single_document(db_id=db_id, db_name=db_name, doc_id=doc_id)
    return {"code": 0, "msg": "ok", "data": out}


async def categories_api(request: Request):
    db_id = query_dict(request).get("db_id")
    db_name = (query_dict(request).get("db_name") or "").strip() or None
    if not db_id and not db_name:
        raise ValueError("请提供 db_id 或 db_name")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("db_id 必须为数字")
    else:
        db_id = None
    items = list_categories(db_id=db_id, db_name=db_name)
    return {"code": 0, "msg": "ok", "data": {"list": items}}


async def category_add_api(request: Request):
    data = await read_json_optional(request) or {}
    db_id = data.get("db_id")
    db_name = (data.get("db_name") or "").strip() or None
    name = (data.get("name") or "").strip()
    sort_order = data.get("sort_order")
    if not name:
        raise ValueError("缺少参数 name")
    if not db_id and not db_name:
        raise ValueError("请提供 db_id 或 db_name")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("db_id 必须为数字")
    else:
        db_id = None
    if sort_order is not None:
        try:
            sort_order = int(sort_order)
        except (TypeError, ValueError):
            sort_order = 0
    out = add_category(db_id=db_id, db_name=db_name, name=name, sort_order=sort_order)
    return {"code": 0, "msg": "ok", "data": out}


async def category_update_api(request: Request):
    data = await read_json_optional(request) or {}
    category_id = data.get("id")
    if category_id is None:
        raise ValueError("缺少参数 id")
    try:
        category_id = int(category_id)
    except (TypeError, ValueError):
        raise ValueError("id 必须为数字")
    name = data.get("name")
    if name is not None:
        name = str(name).strip()
    sort_order = data.get("sort_order")
    out = update_category(category_id, name=name, sort_order=sort_order)
    return {"code": 0, "msg": "ok", "data": out}


async def category_delete_api(request: Request):
    data = await read_json_optional(request) or {}
    category_id = data.get("id")
    if category_id is None:
        raise ValueError("缺少参数 id")
    try:
        category_id = int(category_id)
    except (TypeError, ValueError):
        raise ValueError("id 必须为数字")
    out = delete_category(category_id)
    return {"code": 0, "msg": "ok", "data": out}


async def search_api(request: Request):
    data = await read_json_optional(request) or {}
    db_id = data.get("db_id")
    db_name = (data.get("db_name") or data.get("name") or "").strip()
    query = (data.get("query") or data.get("question") or "").strip()
    if not query:
        raise ValueError("请提供 query")
    if not db_id and not db_name:
        raise ValueError("请提供 db_id 或 db_name")
    if db_id is not None:
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            raise ValueError("db_id 必须为数字")
        from model.ai import VectorDb
        row = VectorDb.get_by_id(db_id)
        if not row:
            raise FileNotFoundError("向量库不存在")
        db_name = row.name
    try:
        top_k = int(data.get("top_k") or 3)
    except (TypeError, ValueError):
        top_k = 3
    category = data.get("category")
    metadata = data.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else None
    results = search_in_db(db_name, query, top_k=top_k, category=category, metadata_filter=metadata)
    return {"code": 0, "msg": "ok", "data": {"results": results}}
