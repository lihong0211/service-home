#!/usr/bin/env python3
"""
RAG 模块：基于知识库的检索与问答。
- 选定知识库后检索相关文档
- 可选：Query 改写（CASEA）、Rerank（DashScope）
- 返回前后状态供前端展示
"""

import logging

import anyio.from_thread
from fastapi import Request

from service.ai.vector_db_qdrant import client, search_in_db
from utils.http_body import read_json_optional
from service.ai.rag_enhance import query_rewrite, rerank_documents, build_rag_answer_prompt
from service.ai import bm25_es
from service.ai._dashscope_common import call_openai_chat_with_retry
from config.ai import DEFAULT_CHAT_MODEL, DEFAULT_RERANK_MODEL
from model.ai import VectorDb, KnowledgeBase

logger = logging.getLogger(__name__)


def _results_to_sources(results: list, use_relevance_score: bool = False) -> list:
    """将检索/rerank 结果转为 sources 列表。"""
    sources = []
    for r in results:
        doc = r.get("doc") if isinstance(r.get("doc"), dict) else r
        text = (doc.get("text") or "").strip()
        sources.append({
            "doc_id": doc.get("id"),
            "text": text[:200] + ("..." if len(text) > 200 else ""),
            "category": doc.get("category"),
            "rank": r.get("rank"),
            "distance": r.get("distance") if not use_relevance_score else None,
            "relevance_score": r.get("relevance_score") if use_relevance_score else None,
        })
    return sources


def _merge_dense_and_bm25(dense: list[dict], bm25_hits: list[dict], top_k: int) -> list[dict]:
    """
    简单融合：对 doc_id 去重，优先保留 dense 的 distance/score，同时保留 bm25_score 供调试。
    当前实现用于“BM25 兜底召回”，不是严格的融合打分（后续可升级为 RRF/加权）。
    """
    if not bm25_hits:
        return dense[:top_k]
    best: dict[str, dict] = {}
    # 先放 dense
    for r in dense or []:
        doc = r.get("doc") if isinstance(r.get("doc"), dict) else r
        did = str((doc or {}).get("id") or "")
        if did:
            best[did] = r
    # 再融合 bm25
    for h in bm25_hits:
        doc = h.get("doc") if isinstance(h.get("doc"), dict) else None
        did = str((doc or {}).get("id") or "")
        if not did:
            continue
        if did in best:
            # 保留 dense 结果，但附加 bm25_score
            best[did]["bm25_score"] = h.get("score")
            continue
        best[did] = {
            "doc": doc,
            "score": None,
            "distance": None,
            "bm25_score": h.get("score"),
            "rank": 0,
        }
    merged = list(best.values())
    # 排序：优先 dense 的 rank，其次 bm25_score
    def _key(x: dict):
        r = x.get("rank")
        if isinstance(r, int) and r > 0:
            return (0, r, 0.0)
        s = x.get("bm25_score")
        try:
            s = float(s)
        except Exception:
            s = 0.0
        return (1, 10**9, -s)

    merged.sort(key=_key)
    # 重置 rank
    out = []
    for i, x in enumerate(merged[: max(1, top_k)]):
        xx = dict(x)
        xx["rank"] = i + 1
        out.append(xx)
    return out


def rag_chat(
    kb_id: int = None,
    kb_name: str = None,
    question: str = None,
    top_k: int = 5,
    model: str = DEFAULT_CHAT_MODEL,
    enable_query_rewrite: bool = False,
    enable_rerank: bool = False,
    enable_hybrid: bool = True,
    enable_bm25: bool = True,
    enable_mmr: bool = True,
    mmr_lambda: float = 0.5,
    score_threshold: float | None = None,
    category: str | None = None,
    metadata_filter: dict | None = None,
    conversation_history: str = "",
    query_rewrite_model: str = DEFAULT_CHAT_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> dict:
    """
    基于知识库的 RAG 问答：可选 Query 改写、Rerank，再检索与生成答案。
    :return: answer, sources, model, 以及 query_rewrite / rerank 状态（供前端展示前后对比）
    """
    if not (question or "").strip():
        raise ValueError("请提供 question")
    
    # 解析知识库/向量库：支持知识库 ID 和向量库 ID
    row = None
    if kb_id is not None:
        # 先尝试作为知识库 ID 查询
        kb = KnowledgeBase.get_by_id(kb_id)
        if kb:
            # 如果知识库有 vector_db_id，用该 ID 查向量库
            if kb.vector_db_id:
                row = VectorDb.get_by_id(kb.vector_db_id)
            # 否则用 kb_{kb_id} 作为向量库名称查询
            if not row:
                vec_name = f"kb_{kb_id}"
                row = VectorDb.select_one_by({"name": vec_name})
        # 如果没查到知识库，尝试作为向量库 ID 查询
        if not row:
            row = VectorDb.get_by_id(kb_id)
    elif kb_name:
        # 先尝试作为向量库名称查询
        row = VectorDb.select_one_by({"name": kb_name})
        # 如果没查到，尝试作为知识库名称查询
        if not row:
            kb = KnowledgeBase.select_one_by({"name": kb_name})
            if kb:
                if kb.vector_db_id:
                    row = VectorDb.get_by_id(kb.vector_db_id)
                if not row:
                    vec_name = f"kb_{kb.id}"
                    row = VectorDb.select_one_by({"name": vec_name})
    else:
        raise ValueError("请提供 kb_id 或 kb_name")
    
    if not row:
        raise FileNotFoundError("知识库或向量库不存在")
    name = row.name

    # ---------- 1. Query 改写（可选） ----------
    search_query = question.strip()
    query_rewrite_state = None
    if enable_query_rewrite:
        qr = query_rewrite(
            query=search_query,
            conversation_history=conversation_history,
            model=query_rewrite_model,
        )
        search_query = (qr.get("rewritten_query") or search_query).strip()
        query_rewrite_state = {
            "original_query": qr.get("original_query"),
            "rewritten_query": qr.get("rewritten_query"),
            "query_type": qr.get("query_type"),
            "confidence": qr.get("confidence"),
        }

    # ---------- 2. 检索（Rerank 时多召一些再精排） ----------
    retrieve_k = min(20, top_k * 2) if enable_rerank else top_k
    results = search_in_db(
        name,
        search_query,
        top_k=retrieve_k,
        category=category,
        metadata_filter=metadata_filter,
        enable_hybrid=enable_hybrid,
        use_mmr=enable_mmr,
        mmr_lambda=mmr_lambda,
        candidate_k=min(80, max(retrieve_k, retrieve_k * 5)) if (enable_mmr or enable_hybrid) else None,
        score_threshold=score_threshold,
    )
    # BM25 兜底召回（可选）：从 ES 做关键词检索，合并到 dense 结果中
    if enable_bm25:
        try:
            bm25_res = bm25_es.bm25_search(
                vector_db_id=int(row.id),
                query=search_query,
                top_k=min(50, max(retrieve_k, retrieve_k * 2)),
                category=category,
                metadata_filter=metadata_filter,
            )
            if bm25_res.get("ok") and bm25_res.get("hits"):
                results = _merge_dense_and_bm25(results, bm25_res["hits"], retrieve_k)
        except Exception as e:
            # ES 作为可选组件：失败不影响主流程，但需要留痕方便排查
            logger.warning("BM25 兜底召回失败，跳过: %s", e)
    if not results:
        rewritten_query = query_rewrite_state.get("rewritten_query") if query_rewrite_state else None
        return {
            "answer": "未检索到相关文档，无法基于当前库回答。",
            "sources": [],
            "full_contexts": [],
            "model": model,
            "rewritten_query": rewritten_query,
            "before": [],
        }

    # ---------- 3. Rerank（可选）：启用时 before=检索结果，results=重排后；不启用时 before=[] ----------
    before_list = []
    if enable_rerank and results:
        rr = rerank_documents(
            query=search_query,
            documents=results,
            top_n=top_k,
            model=rerank_model,
        )
        before_list = _results_to_sources(rr.get("before", results), use_relevance_score=False)
        if rr.get("after"):
            results = rr["after"]

    # ---------- 4. 组 context 与 sources ----------
    context_parts = []
    for r in results:
        doc = r.get("doc") if isinstance(r.get("doc"), dict) else r
        context_parts.append(doc.get("text", ""))
    sources = _results_to_sources(results, use_relevance_score=enable_rerank)
    context = "\n\n---\n\n".join(context_parts)
    prompt = build_rag_answer_prompt(question, context)
    try:
        resp = call_openai_chat_with_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("RAG 生成回答失败: %s", e)
        answer = f"大模型调用失败: {e}"

    # 启用改写时返回改写后的 query（顶层便于前端展示）
    rewritten_query = None
    if query_rewrite_state:
        rewritten_query = query_rewrite_state.get("rewritten_query")

    out = {
        "answer": answer,
        "sources": sources,
        # sources[].text 是给前端展示用的 200 字截断摘要；full_contexts 是未截断的完整片段，
        # 供 RAGAS 评测（faithfulness/context_precision 等）使用——截断文本会让评测判断失真。
        "full_contexts": context_parts,
        "model": model,
        "rewritten_query": rewritten_query,
        "before": before_list,
    }
    return out


def rag_ask_api(request: Request):
    """
    基于知识库的 RAG 问答。
    POST body:
      - knowledge_base_id / kb_id 或 knowledge_base_name / kb_name
      - question / query
      - top_k, model
      - enable_query_rewrite (bool): 是否启用 Query 改写，返回 query_rewrite 前后状态
      - enable_rerank (bool): 是否启用 Rerank，返回 rerank 前后状态
      - conversation_history (str): 对话历史，供 Query 改写使用
    """
    data = anyio.from_thread.run(read_json_optional, request) or {}
    kb_id = data.get("knowledge_base_id") or data.get("kb_id") or data.get("db_id")
    kb_name = (
        data.get("knowledge_base_name")
        or data.get("kb_name")
        or data.get("db_name")
        or data.get("db")
        or data.get("name")
        or data.get("kb")
        or ""
    ).strip()
    question = (data.get("question") or data.get("query") or "").strip()
    if not question:
        raise ValueError("请提供 question 或 query")
    if not kb_id and not kb_name:
        raise ValueError("请提供 knowledge_base_id 或 knowledge_base_name")
    if kb_id is not None:
        try:
            kb_id = int(kb_id)
        except (TypeError, ValueError):
            raise ValueError("knowledge_base_id 必须为数字")
    else:
        kb_id = None
    top_k = data.get("top_k", 5)
    try:
        top_k = max(1, min(20, int(top_k)))
    except (TypeError, ValueError):
        top_k = 5
    model = (data.get("model") or DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
    enable_query_rewrite = bool(data.get("enable_query_rewrite", False))
    enable_rerank = bool(data.get("enable_rerank", False))
    enable_hybrid = bool(data.get("enable_hybrid", True))
    enable_bm25 = bool(data.get("enable_bm25", True))
    enable_mmr = bool(data.get("enable_mmr", True))
    mmr_lambda = data.get("mmr_lambda", 0.5)
    try:
        mmr_lambda = float(mmr_lambda)
    except (TypeError, ValueError):
        mmr_lambda = 0.5
    mmr_lambda = max(0.0, min(1.0, mmr_lambda))
    score_threshold = data.get("score_threshold")
    if score_threshold is not None:
        try:
            score_threshold = float(score_threshold)
        except (TypeError, ValueError):
            score_threshold = None
    category = data.get("category")
    metadata_filter = data.get("metadata")
    metadata_filter = metadata_filter if isinstance(metadata_filter, dict) else None
    conversation_history = (data.get("conversation_history") or "").strip()
    if isinstance(data.get("conversation_history"), list):
        conversation_history = "\n".join(
            str(x) for x in data["conversation_history"]
        ).strip()
    out = rag_chat(
        kb_id=kb_id,
        kb_name=kb_name or None,
        question=question,
        top_k=top_k,
        model=model,
        enable_query_rewrite=enable_query_rewrite,
        enable_rerank=enable_rerank,
        enable_hybrid=enable_hybrid,
        enable_bm25=enable_bm25,
        enable_mmr=enable_mmr,
        mmr_lambda=mmr_lambda,
        score_threshold=score_threshold,
        category=category,
        metadata_filter=metadata_filter,
        conversation_history=conversation_history,
    )
    return {"code": 0, "msg": "ok", "data": out}


def rag_search_api(request: Request):
    """
    在指定知识库中做向量检索（不调用大模型）。
    POST body: { "knowledge_base_id"/"kb_name" 或 "knowledge_base_name", "query", "top_k": 3,
                 "enable_query_rewrite": bool, "enable_rerank": bool, "conversation_history": str }
    """
    from service.ai.rag_enhance import query_rewrite, rerank_documents
    data = anyio.from_thread.run(read_json_optional, request) or {}
    kb_id = data.get("knowledge_base_id") or data.get("kb_id") or data.get("db_id")
    kb_name = (
        data.get("knowledge_base_name")
        or data.get("kb_name")
        or data.get("db_name")
        or data.get("db")
        or data.get("name")
        or ""
    ).strip()
    query = (data.get("query") or "").strip()
    if not kb_id and not kb_name:
        raise ValueError("缺少参数 knowledge_base_id 或 knowledge_base_name")
    
    row = None
    if kb_id is not None:
        try:
            kb_id = int(kb_id)
        except (TypeError, ValueError):
            raise ValueError("knowledge_base_id 必须为数字")
        # 先尝试作为知识库 ID 查询
        kb = KnowledgeBase.get_by_id(kb_id)
        if kb:
            if kb.vector_db_id:
                row = VectorDb.get_by_id(kb.vector_db_id)
            if not row:
                vec_name = f"kb_{kb_id}"
                row = VectorDb.select_one_by({"name": vec_name})
        # 如果没查到知识库，尝试作为向量库 ID 查询
        if not row:
            row = VectorDb.get_by_id(kb_id)
    elif kb_name:
        # 先尝试作为向量库名称查询
        row = VectorDb.select_one_by({"name": kb_name})
        # 如果没查到，尝试作为知识库名称查询
        if not row:
            kb = KnowledgeBase.select_one_by({"name": kb_name})
            if kb:
                if kb.vector_db_id:
                    row = VectorDb.get_by_id(kb.vector_db_id)
                if not row:
                    vec_name = f"kb_{kb.id}"
                    row = VectorDb.select_one_by({"name": vec_name})
    
    if not row:
        raise FileNotFoundError("知识库或向量库不存在")
    kb_name = row.name
    
    if not query:
        raise ValueError("缺少参数 query")
    top_k = data.get("top_k", 3)
    try:
        top_k = max(1, min(20, int(top_k)))
    except (TypeError, ValueError):
        top_k = 3
    enable_query_rewrite = bool(data.get("enable_query_rewrite", False))
    enable_rerank = bool(data.get("enable_rerank", False))
    enable_hybrid = bool(data.get("enable_hybrid", True))
    enable_bm25 = bool(data.get("enable_bm25", True))
    enable_mmr = bool(data.get("enable_mmr", True))
    mmr_lambda = data.get("mmr_lambda", 0.5)
    try:
        mmr_lambda = float(mmr_lambda)
    except (TypeError, ValueError):
        mmr_lambda = 0.5
    mmr_lambda = max(0.0, min(1.0, mmr_lambda))
    score_threshold = data.get("score_threshold")
    if score_threshold is not None:
        try:
            score_threshold = float(score_threshold)
        except (TypeError, ValueError):
            score_threshold = None
    category = data.get("category")
    metadata_filter = data.get("metadata")
    metadata_filter = metadata_filter if isinstance(metadata_filter, dict) else None
    conversation_history = (data.get("conversation_history") or "").strip()
    if isinstance(data.get("conversation_history"), list):
        conversation_history = "\n".join(str(x) for x in data["conversation_history"]).strip()
    
    # Query 改写（可选）
    search_query = query
    query_rewrite_state = None
    rewritten_query = None
    if enable_query_rewrite:
        qr = query_rewrite(query=search_query, conversation_history=conversation_history)
        search_query = (qr.get("rewritten_query") or search_query).strip()
        query_rewrite_state = {
            "original_query": qr.get("original_query"),
            "rewritten_query": qr.get("rewritten_query"),
            "query_type": qr.get("query_type"),
            "confidence": qr.get("confidence"),
        }
        rewritten_query = query_rewrite_state.get("rewritten_query")
    
    # 检索
    retrieve_k = min(20, top_k * 2) if enable_rerank else top_k
    try:
        results = search_in_db(
            kb_name,
            search_query,
            top_k=retrieve_k,
            category=category,
            metadata_filter=metadata_filter,
            enable_hybrid=enable_hybrid,
            use_mmr=enable_mmr,
            mmr_lambda=mmr_lambda,
            candidate_k=min(80, max(retrieve_k, retrieve_k * 5)) if (enable_mmr or enable_hybrid) else None,
            score_threshold=score_threshold,
        )
        if enable_bm25:
            try:
                vdb_row = VectorDb.select_one_by({"name": kb_name})
                if vdb_row:
                    bm25_res = bm25_es.bm25_search(
                        vector_db_id=int(vdb_row.id),
                        query=search_query,
                        top_k=min(50, max(retrieve_k, retrieve_k * 2)),
                        category=category,
                        metadata_filter=metadata_filter,
                    )
                    if bm25_res.get("ok") and bm25_res.get("hits"):
                        results = _merge_dense_and_bm25(results, bm25_res["hits"], retrieve_k)
            except Exception:
                pass
    except Exception as e:
        err_msg = str(e)
        if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
            return (
                {"code": 504, "msg": "检索超时，请稍后重试", "detail": err_msg},
                504,
            )
        raise
    
    # Rerank（可选）：启用时 before=检索结果，results=重排后；不启用时 before=[]
    before_list = []
    if enable_rerank and results:
        rr = rerank_documents(query=search_query, documents=results, top_n=top_k)
        before_list = [
            {"rank": x.get("rank", i + 1), "distance": x.get("distance"), "doc": x.get("doc")}
            for i, x in enumerate(rr.get("before", results))
        ]
        if rr.get("after"):
            results = rr["after"]
    
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "knowledge_base": kb_name,
            "query": query,
            "rewritten_query": rewritten_query,
            "before": before_list,
            "results": [
                {"rank": r.get("rank", i + 1), "distance": r.get("distance"), "relevance_score": r.get("relevance_score"), "doc": r.get("doc")}
                for i, r in enumerate(results)
            ],
        },
    }
