#!/usr/bin/env python3
"""
RAG 模块：基于知识库的检索与问答。
- 选定知识库后检索相关文档
- 可选：Query 改写（CASEA）、Rerank（DashScope）
- 返回前后状态供前端展示
"""

from flask import request, jsonify

from service.ai.vector_db import client, search_in_db
from service.ai.rag_enhance import query_rewrite, rerank_documents
from model.ai import VectorDb, KnowledgeBase


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


def rag_chat(
    kb_id: int = None,
    kb_name: str = None,
    question: str = None,
    top_k: int = 5,
    model: str = "qwen-turbo",
    enable_query_rewrite: bool = False,
    enable_rerank: bool = False,
    conversation_history: str = "",
    query_rewrite_model: str = "qwen-turbo",
    rerank_model: str = "qwen3-rerank",
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
    results = search_in_db(name, search_query, top_k=retrieve_k)
    if not results:
        rewritten_query = query_rewrite_state.get("rewritten_query") if query_rewrite_state else None
        return {
            "answer": "未检索到相关文档，无法基于当前库回答。",
            "sources": [],
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
    prompt = f"""基于以下参考资料回答问题。若资料中无相关内容，请说明无法从资料中得出答案。

参考资料：
{context}

问题：{question.strip()}

请直接给出答案（可简要说明依据的段落或页码）："""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        answer = f"大模型调用失败: {e}"

    # 启用改写时返回改写后的 query（顶层便于前端展示）
    rewritten_query = None
    if query_rewrite_state:
        rewritten_query = query_rewrite_state.get("rewritten_query")

    out = {
        "answer": answer,
        "sources": sources,
        "model": model,
        "rewritten_query": rewritten_query,
        "before": before_list,
    }
    return out


def rag_ask_api():
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
    data = request.get_json() or {}
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
    model = (data.get("model") or "qwen-turbo").strip() or "qwen-turbo"
    enable_query_rewrite = bool(data.get("enable_query_rewrite", False))
    enable_rerank = bool(data.get("enable_rerank", False))
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
        conversation_history=conversation_history,
    )
    return jsonify({"code": 0, "msg": "ok", "data": out})


def rag_search_api():
    """
    在指定知识库中做向量检索（不调用大模型）。
    POST body: { "knowledge_base_id"/"kb_name" 或 "knowledge_base_name", "query", "top_k": 3, 
                 "enable_query_rewrite": bool, "enable_rerank": bool, "conversation_history": str }
    """
    from service.ai.rag_enhance import query_rewrite, rerank_documents
    data = request.get_json() or {}
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
        results = search_in_db(kb_name, search_query, top_k=retrieve_k)
    except Exception as e:
        err_msg = str(e)
        if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
            return (
                jsonify(
                    {"code": 504, "msg": "检索超时，请稍后重试", "detail": err_msg}
                ),
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
    
    return jsonify({
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
    })
