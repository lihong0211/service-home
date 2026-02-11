#!/usr/bin/env python3
"""
向量库模块：FAISS 向量库的创建、加载、文档/分类 CRUD、检索，以及 /ai/vector-db/* HTTP 接口。
- 与 MySQL vector_db / vector_db_document / vector_db_category 同步
- 供知识库模块与 RAG 调用
"""

import os
import re
import time
import json
from openai import OpenAI
from flask import request, jsonify

# faiss/numpy 延后到首次使用时 import，避免进程启动即加载、CTRL+C 退出时析构顺序导致 segfault

# 超时 30s，避免 search 一直无响应；失败时尽快返回错误
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30.0,
)

# 向量库存储根目录，可通过环境变量 VECTOR_DB_STORAGE 覆盖
def _storage_root():
    return os.path.abspath(
        os.getenv("VECTOR_DB_STORAGE", os.path.join(os.getcwd(), "data", "vector_dbs"))
    )

DIMENSION = 1024
VECTORS_FILENAME = "vectors.npy"  # 与 index 同目录，单条更新时只改一行，避免全量 re-embed
# 内存缓存：db_name -> {"index": faiss.Index, "metadata": list}
_db_cache = {}

# 库名只允许字母、数字、下划线、短横线，避免路径问题
DB_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def get_embedding(text: str, max_retries: int = 3) -> list:
    """调用 API 生成向量，失败时重试。"""
    for attempt in range(max_retries):
        try:
            completion = client.embeddings.create(
                model="text-embedding-v4",
                input=text,
                dimensions=DIMENSION,
                encoding_format="float",
            )
            return completion.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                raise e


def _db_path(db_name: str) -> str:
    """某向量库的磁盘目录。"""
    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    return os.path.join(_storage_root(), db_name)


def _ensure_storage():
    root = _storage_root()
    os.makedirs(root, exist_ok=True)
    return root


def _create_empty_vector_db_on_disk(db_name: str) -> str:
    """在磁盘上创建空向量库（空 index + 空 metadata + 空 vectors.npy），返回 path_dir。"""
    import numpy as np
    import faiss
    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    path_dir = _db_path(db_name)
    _ensure_storage()
    os.makedirs(path_dir, exist_ok=True)
    index_flat = faiss.IndexFlatL2(DIMENSION)
    index = faiss.IndexIDMap(index_flat)
    index_path = os.path.join(path_dir, "index.faiss")
    meta_path = os.path.join(path_dir, "metadata.json")
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    np.save(os.path.join(path_dir, VECTORS_FILENAME), np.zeros((0, DIMENSION), dtype=np.float32))
    _db_cache[db_name] = {"index": index, "metadata": []}
    return path_dir


def create_vector_db(db_name: str, documents: list[dict] = None) -> dict:
    """
    创建并持久化命名向量库。documents 可为空，此时只建空库（仅名字+描述，编辑页再加文档）。
    :param db_name: 向量库名称（仅允许字母、数字、下划线、短横线）
    :param documents: 可选，列表每项 {"id": str, "text": str, "category": str 可选}；不传或 [] 则建空库
    :return: {"count": 文档数, "path": 存储路径, "documents": 文档列表}
    """
    import numpy as np
    import faiss
    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    if documents is None:
        documents = []
    if not isinstance(documents, list):
        raise ValueError("documents 必须为列表")

    if not documents:
        path_dir = _create_empty_vector_db_on_disk(db_name)
        return {"count": 0, "path": path_dir, "documents": []}

    # 规范化文档：text 必填；id、category、metadata 可选。
    normalized = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            continue
        text = doc.get("text") or doc.get("content") or ""
        text = str(text).strip()
        if not text:
            continue
        category = (doc.get("category") or "").strip() or None
        item = {
            "id": doc.get("id", f"doc_{i}"),
            "text": text,
            "category": category,
        }
        if "metadata" in doc and doc["metadata"] is not None:
            item["metadata"] = doc["metadata"] if isinstance(doc["metadata"], dict) else {}
        normalized.append(item)

    if not normalized:
        raise ValueError("没有有效的文档（需包含 text 或 content）")

    metadata_store = []
    vectors_list = []
    vector_ids = []

    for i, doc in enumerate(normalized):
        try:
            vector = get_embedding(doc["text"])
            vectors_list.append(vector)
            metadata_store.append(doc)
            vector_ids.append(i)
        except Exception:
            continue

    if not vectors_list:
        raise ValueError("所有文档生成向量均失败")

    vectors_np = np.array(vectors_list).astype("float32")
    vector_ids_np = np.array(vector_ids)
    index_flat = faiss.IndexFlatL2(DIMENSION)
    index = faiss.IndexIDMap(index_flat)
    index.add_with_ids(vectors_np, vector_ids_np)

    # 持久化
    path_dir = _db_path(db_name)
    _ensure_storage()
    os.makedirs(path_dir, exist_ok=True)
    index_path = os.path.join(path_dir, "index.faiss")
    meta_path = os.path.join(path_dir, "metadata.json")
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(path_dir, VECTORS_FILENAME), vectors_np)

    # 更新缓存
    _db_cache[db_name] = {"index": index, "metadata": metadata_store}

    return {"count": len(metadata_store), "path": path_dir, "documents": metadata_store}


def load_vector_db(db_name: str) -> dict:
    """加载命名向量库（优先内存缓存）。返回 {"index": faiss.Index, "metadata": list}"""
    import faiss
    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    if db_name in _db_cache:
        return _db_cache[db_name]
    path_dir = _db_path(db_name)
    index_path = os.path.join(path_dir, "index.faiss")
    meta_path = os.path.join(path_dir, "metadata.json")
    if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(f"向量库不存在: {db_name}")
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    _db_cache[db_name] = {"index": index, "metadata": metadata}
    return _db_cache[db_name]


def sync_vector_db_from_disk(db_name: str, description: str = None) -> dict:
    """
    将已在磁盘存在的向量库同步到 MySQL（补写 vector_db、vector_db_document）。
    用于：创建时向量生成成功但 MySQL 写入失败，或仅从磁盘建了库的情况。
    """
    if not DB_NAME_PATTERN.match(db_name):
        raise ValueError(f"无效的向量库名: {db_name}")
    path_dir = _db_path(db_name)
    meta_path = os.path.join(path_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"磁盘上不存在该向量库: {db_name}")
    with open(meta_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    if not documents:
        raise ValueError("磁盘上文档列表为空")
    from model.ai import VectorDb
    row = VectorDb.select_one_by({"name": db_name})
    if row:
        vector_db_id = row.id
        _save_documents_to_mysql(vector_db_id, documents)
        _sync_categories_from_documents(vector_db_id, documents)
        if description is not None:
            VectorDb.update({"id": vector_db_id, "description": (description or "").strip() or None})
        return {"id": vector_db_id, "name": db_name, "count": len(documents), "synced": "documents"}
    row_id = VectorDb.insert({"name": db_name, "description": (description or "").strip() or None})
    _save_documents_to_mysql(row_id, documents)
    _sync_categories_from_documents(row_id, documents)
    return {"id": row_id, "name": db_name, "count": len(documents), "synced": "db_and_documents"}


def _rebuild_vector_db_index(vector_db_id: int, documents: list[dict]) -> None:
    """用新文档列表重建已有向量库的索引与 MySQL 文档表（覆盖磁盘与 DB）。"""
    import numpy as np
    import faiss
    from model.ai import VectorDb
    row = VectorDb.get_by_id(vector_db_id)
    if not row:
        raise FileNotFoundError("向量库不存在")
    db_name = row.name
    if not documents:
        path_dir = _db_path(db_name)
        _ensure_storage()
        os.makedirs(path_dir, exist_ok=True)
        index_flat = faiss.IndexFlatL2(DIMENSION)
        index = faiss.IndexIDMap(index_flat)
        faiss.write_index(index, os.path.join(path_dir, "index.faiss"))
        with open(os.path.join(path_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        np.save(os.path.join(path_dir, VECTORS_FILENAME), np.zeros((0, DIMENSION), dtype=np.float32))
        _save_documents_to_mysql(vector_db_id, [])
        if db_name in _db_cache:
            del _db_cache[db_name]
        _db_cache[db_name] = {"index": index, "metadata": []}
        return
    normalized = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            continue
        text = (doc.get("text") or doc.get("content") or "").strip()
        if not text:
            continue
        category = (doc.get("category") or "").strip() or None
        item = {
            "id": doc.get("id", f"doc_{i}"),
            "text": text,
            "category": category,
        }
        if "metadata" in doc and doc["metadata"] is not None:
            item["metadata"] = doc["metadata"] if isinstance(doc["metadata"], dict) else {}
        normalized.append(item)
    metadata_store = []
    vectors_list = []
    vector_ids = []
    for i, doc in enumerate(normalized):
        try:
            vector = get_embedding(doc["text"])
            vectors_list.append(vector)
            metadata_store.append(doc)
            vector_ids.append(i)
        except Exception:
            continue
    if not vectors_list:
        raise ValueError("所有文档生成向量均失败")
    vectors_np = np.array(vectors_list).astype("float32")
    vector_ids_np = np.array(vector_ids)
    index_flat = faiss.IndexFlatL2(DIMENSION)
    index = faiss.IndexIDMap(index_flat)
    index.add_with_ids(vectors_np, vector_ids_np)
    path_dir = _db_path(db_name)
    _ensure_storage()
    os.makedirs(path_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(path_dir, "index.faiss"))
    with open(os.path.join(path_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(path_dir, VECTORS_FILENAME), vectors_np)
    _save_documents_to_mysql(vector_db_id, metadata_store)
    _sync_categories_from_documents(vector_db_id, metadata_store)
    _db_cache[db_name] = {"index": index, "metadata": metadata_store}


def append_documents_batch(vector_db_id: int, documents: list[dict]) -> int:
    """
    向已有向量库批量追加文档（只对新文档调 embedding，省 token）。
    要求 documents 中每条 doc["id"] 在库中尚不存在；重复 id 会跳过。
    返回实际追加的条数。
    """
    import faiss
    import numpy as np
    from model.ai import VectorDb
    row = VectorDb.get_by_id(vector_db_id)
    if not row:
        raise FileNotFoundError("向量库不存在")
    db_name = row.name
    db = load_vector_db(db_name)
    faiss_index = db["index"]
    metadata_store = list(db["metadata"])
    existing_ids = {str(d.get("id", "")) for d in metadata_store}
    ntotal = faiss_index.ntotal
    normalized = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("id", f"doc_{ntotal + i}"))
        if doc_id in existing_ids:
            continue
        text = (doc.get("text") or doc.get("content") or "").strip()
        if not text:
            continue
        category = (doc.get("category") or "").strip() or None
        item = {
            "id": doc_id,
            "text": text,
            "category": category,
        }
        if "metadata" in doc and doc["metadata"] is not None:
            item["metadata"] = doc["metadata"] if isinstance(doc["metadata"], dict) else {}
        normalized.append(item)
    if not normalized:
        return 0
    to_add = []
    for doc in normalized:
        try:
            vec = get_embedding(doc["text"])
            to_add.append((doc, vec))
        except Exception:
            continue
    if not to_add:
        return 0
    new_vectors = np.array([v for _, v in to_add], dtype=np.float32)
    new_docs = [d for d, _ in to_add]
    new_ids = np.arange(ntotal, ntotal + len(new_vectors), dtype=np.int64)
    faiss_index.add_with_ids(new_vectors, new_ids)
    metadata_store.extend(new_docs)
    path_dir = _db_path(db_name)
    _ensure_storage()
    os.makedirs(path_dir, exist_ok=True)
    faiss.write_index(faiss_index, os.path.join(path_dir, "index.faiss"))
    with open(os.path.join(path_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)
    vectors_path = os.path.join(path_dir, VECTORS_FILENAME)
    if ntotal > 0 and os.path.isfile(vectors_path):
        v = np.load(vectors_path).astype(np.float32)
        v = np.vstack((v, new_vectors))
        np.save(vectors_path, v)
    else:
        np.save(vectors_path, new_vectors)
    _append_documents_to_mysql(vector_db_id, new_docs)
    _sync_categories_from_documents(vector_db_id, new_docs)
    _db_cache[db_name] = {"index": faiss_index, "metadata": metadata_store}
    return len(new_docs)


def list_vector_dbs() -> list[str]:
    """列出已存在的向量库名称（仅磁盘）。"""
    root = _storage_root()
    if not os.path.isdir(root):
        return []
    names = []
    for name in os.listdir(root):
        if not DB_NAME_PATTERN.match(name):
            continue
        path_dir = os.path.join(root, name)
        if os.path.isdir(path_dir) and os.path.isfile(os.path.join(path_dir, "index.faiss")):
            names.append(name)
    return sorted(names)


def _delete_vector_db_from_disk(db_name: str) -> None:
    """删除某向量库的磁盘目录并清除缓存。"""
    if db_name in _db_cache:
        del _db_cache[db_name]
    path_dir = _db_path(db_name)
    if not os.path.isdir(path_dir):
        return
    import shutil
    shutil.rmtree(path_dir, ignore_errors=True)


def list_vector_dbs_from_mysql() -> list[dict]:
    """从 MySQL 列出向量库列表。返回 [{"id", "name", "description", "create_at", "update_at"}, ...]"""
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


def _save_documents_to_mysql(vector_db_id: int, documents: list[dict]) -> None:
    """将文档列表写入 vector_db_document 表（先删后插）。支持 doc.metadata 写入 metadata 列。"""
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
    """向某向量库追加文档（仅插入，不删原有）。支持 doc.metadata 写入 metadata 列。"""
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
    """根据文档中的 category 字段，将不存在的分类名写入 vector_db_category（不删已有分类）。"""
    from model.ai import VectorDbCategory
    names = set()
    for doc in documents:
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
            VectorDbCategory.insert({
                "vector_db_id": vector_db_id,
                "name": name,
                "sort_order": i,
            })
            existing.add(name)


def list_documents(db_id: int = None, db_name: str = None) -> list[dict]:
    """从 MySQL 查询某向量库下的文档 item 列表（不分页）。"""
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
    db_id: int = None,
    db_name: str = None,
    page: int = 1,
    page_size: int = 20,
    category: str = None,
) -> dict:
    """分页查询某向量库下的文档，可按 category 筛选。category 为空字符串时筛「未分类」。"""
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


def get_vector_db_detail(db_id: int = None, db_name: str = None, with_documents: bool = False) -> dict:
    """获取向量库详情。with_documents=False 时不拉文档列表（编辑页用分页接口拉取）。"""
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


def list_categories(db_id: int = None, db_name: str = None) -> list[dict]:
    """查询某向量库下的分类列表（用于新增/编辑文档时选择）。"""
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


def add_category(db_id: int = None, db_name: str = None, name: str = None, sort_order: int = None) -> dict:
    """为某向量库新增一个分类。"""
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


def update_category(category_id: int, name: str = None, sort_order: int = None) -> dict:
    """更新分类名称或排序。"""
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
    """删除一个分类（不影响已存文档的 category 字段）。"""
    from model.ai import VectorDbCategory
    row = VectorDbCategory.get_by_id(category_id)
    if not row:
        raise FileNotFoundError("分类不存在")
    VectorDbCategory.force_delete({"id": category_id})
    return {"id": category_id}


def delete_vector_db_by_id(db_id: int) -> str:
    """根据 ID 删除向量库：删分类、文档表、库记录、磁盘目录。返回被删的 name。"""
    from model.ai import VectorDb, VectorDbDocument, VectorDbCategory
    row = VectorDb.get_by_id(db_id)
    if not row:
        raise FileNotFoundError("向量库不存在")
    name = row.name
    VectorDbCategory.force_delete({"vector_db_id": db_id})
    VectorDbDocument.force_delete({"vector_db_id": db_id})
    VectorDb.force_delete({"id": db_id})
    _delete_vector_db_from_disk(name)
    return name


def add_single_document(
    db_id: int = None,
    db_name: str = None,
    doc_id: str = None,
    text: str = None,
    category: str = None,
) -> dict:
    """
    向已有向量库追加一条文档：生成向量并追加到索引、落盘、写入 MySQL。
    doc_id 可选：不传则用数据库自增 id 作为 doc_id；传了则用传入值（须不与已有重复）。
    """
    import numpy as np
    import faiss
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
    faiss_index = db["index"]
    metadata_store = list(db["metadata"])
    ntotal = faiss_index.ntotal
    existing_ids = {str(doc.get("id", "")) for doc in metadata_store}
    insert_mysql_after = False
    if (doc_id or "").strip():
        new_id = str(doc_id).strip()
        if new_id in existing_ids:
            raise ValueError(f"doc_id 已存在: {new_id}，请换一个或使用更新接口")
        insert_mysql_after = True
    else:
        # 用数据库自增 id：先插一行占位，取 id 作为 doc_id，再更新该行 doc_id
        new_row_id = VectorDbDocument.insert({
            "vector_db_id": vector_db_id,
            "doc_id": "_",
            "text": text,
            "category": cat,
        })
        new_id = str(new_row_id)
        VectorDbDocument.update({"id": new_row_id, "doc_id": new_id})
    new_embedding = get_embedding(text)
    new_vector = np.array([new_embedding], dtype=np.float32)
    new_idx = ntotal
    if ntotal == 0:
        index_flat = faiss.IndexFlatL2(DIMENSION)
        new_index = faiss.IndexIDMap(index_flat)
        new_index.add_with_ids(new_vector, np.array([0], dtype=np.int64))
        metadata_store = [{"id": str(new_id), "text": text, "category": cat}]
    else:
        # 直接追加，避免 reconstruct（部分 faiss 读回的索引不支持 reconstruct）
        faiss_index.add_with_ids(new_vector, np.array([ntotal], dtype=np.int64))
        new_index = faiss_index
        metadata_store.append({"id": str(new_id), "text": text, "category": cat})
    path_dir = _db_path(name)
    _ensure_storage()
    os.makedirs(path_dir, exist_ok=True)
    index_path = os.path.join(path_dir, "index.faiss")
    meta_path = os.path.join(path_dir, "metadata.json")
    vectors_path = os.path.join(path_dir, VECTORS_FILENAME)
    if ntotal > 0:
        if os.path.isfile(vectors_path):
            v = np.load(vectors_path).astype(np.float32)
            v = np.vstack((v, new_vector[0]))
            np.save(vectors_path, v)
        else:
            # 旧库无 vectors.npy：一次性 re-embed 全部并写入，后续追加/更新可复用
            arr = np.zeros((len(metadata_store), DIMENSION), dtype=np.float32)
            for j in range(len(metadata_store) - 1):
                arr[j] = np.array(get_embedding(metadata_store[j]["text"]), dtype=np.float32)
            arr[-1] = new_vector[0]
            np.save(vectors_path, arr)
    else:
        np.save(vectors_path, new_vector)
    faiss.write_index(new_index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)
    _db_cache[name] = {"index": new_index, "metadata": metadata_store}
    if insert_mysql_after:
        VectorDbDocument.insert({
            "vector_db_id": vector_db_id,
            "doc_id": str(new_id),
            "text": text,
            "category": cat,
        })
    if cat:
        _sync_categories_from_documents(vector_db_id, [{"category": cat}])
    return {"doc_id": str(new_id), "db_name": name, "total": len(metadata_store)}


def update_single_document(
    db_id: int = None,
    db_name: str = None,
    doc_id: str = None,
    index: int = None,
    text: str = None,
    category: str = None,
) -> dict:
    """
    单条文档更新：只更新指定文档的文本、分类，并重新生成该条向量、写回索引与 MySQL。
    通过 doc_id 或 index（0-based 下标）定位文档，二者至少传一个；若都传则优先 doc_id。
    """
    import numpy as np
    import faiss
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
    db = load_vector_db(name)
    faiss_index = db["index"]
    metadata_store = list(db["metadata"])
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
    new_embedding = np.array(get_embedding(text), dtype=np.float32)
    ntotal = faiss_index.ntotal
    if ntotal == 0:
        raise ValueError("向量库索引为空")
    path_dir = _db_path(name)
    vectors_path = os.path.join(path_dir, VECTORS_FILENAME)
    if os.path.isfile(vectors_path):
        # 只改一行，不 re-embed 其他文档
        vectors_np = np.load(vectors_path).astype(np.float32)
        if vectors_np.shape[0] != ntotal or vectors_np.shape[1] != DIMENSION:
            vectors_np = np.zeros((ntotal, DIMENSION), dtype=np.float32)
            for j in range(ntotal):
                vectors_np[j] = new_embedding if j == idx else np.array(get_embedding(metadata_store[j]["text"]), dtype=np.float32)
        else:
            vectors_np[idx] = new_embedding
        np.save(vectors_path, vectors_np)
    else:
        # 旧库无 vectors.npy：全量 re-embed 一次并落盘，下次可只改一行
        vectors_np = np.zeros((ntotal, DIMENSION), dtype=np.float32)
        for j in range(ntotal):
            vectors_np[j] = new_embedding if j == idx else np.array(get_embedding(metadata_store[j]["text"]), dtype=np.float32)
        np.save(vectors_path, vectors_np)
    vector_ids_np = np.arange(ntotal, dtype=np.int64)
    index_flat = faiss.IndexFlatL2(DIMENSION)
    new_index = faiss.IndexIDMap(index_flat)
    new_index.add_with_ids(vectors_np, vector_ids_np)
    metadata_store[idx] = {"id": str(doc_id), "text": text, "category": cat}
    _ensure_storage()
    os.makedirs(path_dir, exist_ok=True)
    index_path = os.path.join(path_dir, "index.faiss")
    meta_path = os.path.join(path_dir, "metadata.json")
    faiss.write_index(new_index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)
    _db_cache[name] = {"index": new_index, "metadata": metadata_store}
    # 更新 MySQL 中该文档（有则更新，无则插入以保持同步）
    doc_rows = VectorDbDocument.select_by({"vector_db_id": vector_db_id, "doc_id": str(doc_id)})
    if doc_rows:
        VectorDbDocument.update({
            "id": doc_rows[0].id,
            "text": text,
            "category": cat,
        })
    else:
        VectorDbDocument.insert({
            "vector_db_id": vector_db_id,
            "doc_id": str(doc_id),
            "text": text,
            "category": cat,
        })
    if cat:
        _sync_categories_from_documents(vector_db_id, [{"category": cat}])
    return {"doc_id": str(doc_id), "db_name": name}


def delete_single_document(db_id: int = None, db_name: str = None, doc_id: str = None) -> dict:
    """删除某向量库下的一条文档（磁盘 index/metadata/vectors.npy + MySQL）。"""
    import numpy as np
    import faiss
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
    db = load_vector_db(name)
    metadata_store = list(db["metadata"])
    idx = None
    for i, doc in enumerate(metadata_store):
        if str(doc.get("id", "")) == doc_id:
            idx = i
            break
    if idx is None:
        raise FileNotFoundError(f"文档不存在: doc_id={doc_id}")
    new_metadata = [m for i, m in enumerate(metadata_store) if i != idx]
    path_dir = _db_path(name)
    vectors_path = os.path.join(path_dir, VECTORS_FILENAME)
    if len(metadata_store) == 1:
        # 删成空库
        index_flat = faiss.IndexFlatL2(DIMENSION)
        new_index = faiss.IndexIDMap(index_flat)
        np.save(vectors_path, np.zeros((0, DIMENSION), dtype=np.float32))
    else:
        if os.path.isfile(vectors_path):
            vectors_np = np.load(vectors_path).astype(np.float32)
            vectors_np = np.delete(vectors_np, idx, axis=0)
            np.save(vectors_path, vectors_np)
        else:
            # 旧库无 vectors.npy：用剩余文档 re-embed 生成向量并落盘
            vectors_list = [
                np.array(get_embedding(m["text"]), dtype=np.float32)
                for m in new_metadata
            ]
            vectors_np = np.array(vectors_list).astype(np.float32)
            np.save(vectors_path, vectors_np)
        vector_ids_np = np.arange(len(new_metadata), dtype=np.int64)
        index_flat = faiss.IndexFlatL2(DIMENSION)
        new_index = faiss.IndexIDMap(index_flat)
        new_index.add_with_ids(vectors_np, vector_ids_np)
    index_path = os.path.join(path_dir, "index.faiss")
    meta_path = os.path.join(path_dir, "metadata.json")
    faiss.write_index(new_index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=2)
    _db_cache[name] = {"index": new_index, "metadata": new_metadata}
    VectorDbDocument.force_delete({"vector_db_id": vector_db_id, "doc_id": doc_id})
    return {"doc_id": doc_id, "db_name": name, "total": len(new_metadata)}


def rebuild_vector_db_from_mysql(db_id: int = None, db_name: str = None) -> dict:
    """根据 MySQL 中该库的文档列表重新生成向量并写回磁盘（index/metadata/vectors.npy），保证与 MySQL 顺序一致。"""
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
    out = create_vector_db(row.name, documents)
    _save_documents_to_mysql(row.id, out.get("documents") or [])
    _sync_categories_from_documents(row.id, out.get("documents") or [])
    return {"id": row.id, "name": row.name, "count": out["count"]}


def search_in_db(db_name: str, query: str, top_k: int = 3) -> list[dict]:
    """
    在指定向量库中检索。
    :return: [{"doc": {...}, "distance": float, "rank": int}, ...]
    """
    import numpy as np
    db = load_vector_db(db_name)
    index = db["index"]
    metadata = db["metadata"]
    if index.ntotal == 0:
        return []
    k = min(top_k, index.ntotal)
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype("float32")
    distances, retrieved_ids = index.search(query_vector, k)
    results = []
    n_meta = len(metadata)
    for i in range(k):
        doc_id = int(retrieved_ids[0][i])
        if doc_id == -1 or doc_id < 0 or doc_id >= n_meta:
            continue
        doc = metadata[doc_id]
        results.append({
            "doc": doc,
            "distance": float(distances[0][i]),
            "rank": len(results) + 1,
        })
    return results


# ---------- HTTP 接口（/ai/vector-db/*） ----------

def list_api():
    """GET/POST /ai/vector-db/list — 从 MySQL 列出向量库。"""
    items = list_vector_dbs_from_mysql()
    return jsonify({"code": 0, "msg": "ok", "data": {"list": items, "names": [x["name"] for x in items]}})


def create_api():
    """POST /ai/vector-db — 创建向量库。body: name, description?, documents?"""
    from model.ai import VectorDb
    data = request.get_json(silent=True) or {}
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
        row = VectorDb.get_by_id(row_id)
        return jsonify({
            "code": 0,
            "msg": "ok",
            "data": {"id": row_id, "name": name, "description": description, "count": out["count"], "path": out["path"]},
        })
    except Exception as e_vec:
        VectorDb.force_delete({"id": row_id})
        _delete_vector_db_from_disk(name)
        raise e_vec


def detail_api():
    """GET /ai/vector-db/detail — 向量库详情。query: id 或 name, with_documents?"""
    db_id = request.args.get("id")
    db_name = request.args.get("name")
    with_documents = request.args.get("with_documents", "0") in ("1", "true", "yes")
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
    return jsonify({"code": 0, "msg": "ok", "data": detail})


def update_api():
    """POST /ai/vector-db/update — 全量替换文档并重建。body: id, documents, description?"""
    from model.ai import VectorDb
    data = request.get_json() or {}
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
    return jsonify({"code": 0, "msg": "ok", "data": {"id": db_id, "name": name, "count": out["count"]}})


def update_meta_api():
    """POST /ai/vector-db/update-meta — 仅更新向量库元信息（说明、名称），不动文档与索引。body: id, description?, name?"""
    from model.ai import VectorDb
    data = request.get_json() or {}
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
    update_data = {"id": db_id}
    if data.get("description") is not None:
        desc = (data.get("description") or "").strip() or None
        update_data["description"] = desc
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
    if len(update_data) > 1:
        VectorDb.update(update_data)
    row = VectorDb.get_by_id(db_id)
    return jsonify({
        "code": 0,
        "msg": "ok",
        "data": {
            "id": row.id,
            "name": row.name,
            "description": (row.description or "").strip() or None,
        },
    })


def delete_api():
    """POST /ai/vector-db/delete — 删除向量库。body: id"""
    data = request.get_json() or {}
    db_id = data.get("id")
    if db_id is None:
        raise ValueError("缺少参数 id")
    try:
        db_id = int(db_id)
    except (TypeError, ValueError):
        raise ValueError("id 必须为数字")
    name = delete_vector_db_by_id(db_id)
    return jsonify({"code": 0, "msg": "ok", "data": {"id": db_id, "name": name}})


def sync_from_disk_api():
    """POST /ai/vector-db/sync-from-disk — 磁盘库同步到 MySQL。body: name 或 db, description?"""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or data.get("db") or "").strip()
    if not name:
        raise ValueError("缺少参数 name 或 db")
    if not DB_NAME_PATTERN.match(name):
        raise ValueError("库名仅允许 a-zA-Z0-9_-")
    description = (data.get("description") or "").strip() or None
    out = sync_vector_db_from_disk(name, description=description)
    return jsonify({"code": 0, "msg": "ok", "data": out})


def rebuild_api():
    """POST /ai/vector-db/rebuild — 按 MySQL 文档重建向量索引。body: id 或 name/db"""
    data = request.get_json(silent=True) or {}
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
    return jsonify({"code": 0, "msg": "ok", "data": out})


def documents_api():
    """GET /ai/vector-db/documents — 分页查文档。query: db_id 或 db_name, page, page_size, category?"""
    db_id = request.args.get("db_id")
    db_name = (request.args.get("db_name") or "").strip() or None
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
        page = int(request.args.get("page") or 1)
        page_size = int(request.args.get("page_size") or 20)
    except (TypeError, ValueError):
        page, page_size = 1, 20
    category = request.args.get("category")
    out = list_documents_paginated(db_id=db_id, db_name=db_name, page=page, page_size=page_size, category=category)
    return jsonify({"code": 0, "msg": "ok", "data": out})


def document_add_api():
    """POST /ai/vector-db/document/add — 追加一条文档。body: db_id 或 db_name, text, doc_id?, category?"""
    data = request.get_json(silent=True) or {}
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
    out = add_single_document(db_id=db_id, db_name=db_name, doc_id=doc_id, text=str(text).strip(), category=(category or "").strip() or None if category is not None else None)
    return jsonify({"code": 0, "msg": "ok", "data": out})


def document_update_api():
    """POST /ai/vector-db/document/update — 更新单条文档。body: db_id 或 db_name, doc_id 或 index, text, category?"""
    data = request.get_json() or {}
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
    out = update_single_document(db_id=db_id, db_name=db_name, doc_id=doc_id, index=index, text=str(text).strip(), category=(category or "").strip() or None if category is not None else None)
    return jsonify({"code": 0, "msg": "ok", "data": out})


def document_delete_api():
    """POST /ai/vector-db/document/delete — 删除单条文档。body: db_id 或 db_name, doc_id"""
    data = request.get_json() or {}
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
    return jsonify({"code": 0, "msg": "ok", "data": out})


def categories_api():
    """GET /ai/vector-db/categories — 分类列表。query: db_id 或 db_name"""
    db_id = request.args.get("db_id")
    db_name = (request.args.get("db_name") or "").strip() or None
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
    return jsonify({"code": 0, "msg": "ok", "data": {"list": items}})


def category_add_api():
    """POST /ai/vector-db/category/add — 新增分类。body: db_id 或 db_name, name, sort_order?"""
    data = request.get_json() or {}
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
    return jsonify({"code": 0, "msg": "ok", "data": out})


def category_update_api():
    """POST /ai/vector-db/category/update — 更新分类。body: id, name?, sort_order?"""
    data = request.get_json() or {}
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
    return jsonify({"code": 0, "msg": "ok", "data": out})


def category_delete_api():
    """POST /ai/vector-db/category/delete — 删除分类。body: id"""
    data = request.get_json() or {}
    category_id = data.get("id")
    if category_id is None:
        raise ValueError("缺少参数 id")
    try:
        category_id = int(category_id)
    except (TypeError, ValueError):
        raise ValueError("id 必须为数字")
    out = delete_category(category_id)
    return jsonify({"code": 0, "msg": "ok", "data": out})


def search_api():
    """POST /ai/vector-db/search — 向量检索。body: db_id 或 db_name, query, top_k?"""
    data = request.get_json(silent=True) or {}
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
    else:
        db_name = db_name or None
    try:
        top_k = int(data.get("top_k") or 3)
    except (TypeError, ValueError):
        top_k = 3
    results = search_in_db(db_name, query, top_k=top_k)
    return jsonify({"code": 0, "msg": "ok", "data": {"results": results}})
