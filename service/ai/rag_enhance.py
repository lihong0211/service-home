#!/usr/bin/env python3
"""
RAG 增强：Query 改写（CASEA）与 Rerank（DashScope）。
- 供 rag.py 调用，可选开启
- 返回前后状态供前端展示
"""

import os
import json

import dashscope

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def _get_completion(prompt: str, model: str = "qwen-turbo") -> str:
    """调用 dashscope 生成文本。"""
    messages = [{"role": "user", "content": prompt}]
    resp = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        temperature=0,
    )
    if not resp or not resp.output or not resp.output.choices:
        return ""
    return (resp.output.choices[0].message.content or "").strip()


def query_rewrite(
    query: str,
    conversation_history: str = "",
    context_info: str = "",
    model: str = "qwen-turbo",
) -> dict:
    """
    自动识别 Query 类型并改写（CASEA-Query 改写）。
    :return: {
        "original_query": str,
        "rewritten_query": str,
        "query_type": str,
        "confidence": float,
        "raw_analysis": dict | None,
    }
    """
    instruction = """
你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含"还有"、"其他"等需要上下文理解的词汇
2. 对比型 - 包含"哪个"、"比较"、"更"、"哪个更好"、"哪个更"等比较词汇
3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
4. 多意图型 - 包含多个独立问题，用"、"或"？"分隔
5. 反问型 - 包含"不会"、"难道"等反问语气
6. 无需改写 - 问题已完整清晰

请返回JSON格式（仅一行，不要换行）：
{"query_type": "类型名", "rewritten_query": "改写后的查询", "confidence": 0.0-1.0}
说明：若不需改写，rewritten_query 为原问题；多意图型可合并为一个完整问句或保留主要意图。
"""
    prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history or "无"}

### 上下文信息 ###
{context_info or "无"}

### 原始查询 ###
{query}

### 分析结果（仅返回一行JSON） ###
"""
    try:
        raw = _get_completion(prompt, model=model)
        # 尝试从返回中提取 JSON
        if raw:
            for start in ("{", "```json", "```"):
                idx = raw.find(start)
                if idx >= 0:
                    if start.startswith("```"):
                        idx = raw.find("{", idx)
                    end = raw.rfind("}") + 1
                    if end > idx:
                        obj = json.loads(raw[idx:end])
                        qtype = obj.get("query_type") or "未知"
                        rewritten = (obj.get("rewritten_query") or query).strip() or query
                        conf = float(obj.get("confidence", 0.5))
                        return {
                            "original_query": query,
                            "rewritten_query": rewritten,
                            "query_type": qtype,
                            "confidence": conf,
                            "raw_analysis": obj,
                        }
        return {
            "original_query": query,
            "rewritten_query": query,
            "query_type": "未知",
            "confidence": 0.5,
            "raw_analysis": None,
        }
    except Exception:
        return {
            "original_query": query,
            "rewritten_query": query,
            "query_type": "改写失败",
            "confidence": 0,
            "raw_analysis": None,
        }


def rerank_documents(
    query: str,
    documents: list[dict],
    top_n: int = None,
    model: str = "qwen3-rerank",
    text_key: str = "text",
) -> dict:
    """
    对检索结果做 Rerank（DashScope 文本排序）。
    :param query: 查询文本
    :param documents: 列表，每项为 dict，需含 text_key 字段（或 "text"）
    :param top_n: 返回前 N 条，默认全部
    :return: {
        "before": [{"doc": {...}, "rank": 1, "distance": ...}, ...],  # 原始顺序
        "after": [{"doc": {...}, "rank": 1, "relevance_score": ...}, ...],  # 重排后
        "model": str,
    }
    """
    if not documents:
        return {"before": [], "after": [], "model": model}
    texts = []
    for d in documents:
        doc = d.get("doc") if isinstance(d.get("doc"), dict) else d
        t = (doc.get(text_key) or doc.get("text") or "").strip()
        if not t:
            t = str(doc)[:4000]
        texts.append(t[:4000])
    try:
        resp = dashscope.TextReRank.call(
            model=model,
            query=query,
            documents=texts,
            top_n=top_n or len(texts),
            return_documents=True,
        )
    except Exception as e:
        return {
            "before": documents,
            "after": documents,
            "model": model,
            "error": str(e),
        }
    if not getattr(resp, "output", None) or not getattr(resp.output, "results", None):
        return {"before": documents, "after": documents, "model": model}
    results = resp.output.results
    after = []
    for i, r in enumerate(results):
        idx = getattr(r, "index", i)
        score = getattr(r, "relevance_score", 0.0)
        if idx < len(documents):
            item = documents[idx].copy()
            doc = item.get("doc")
            if isinstance(doc, dict):
                item = {"doc": doc, "rank": i + 1, "relevance_score": score}
            else:
                item["rank"] = i + 1
                item["relevance_score"] = score
            after.append(item)
    return {
        "before": [
            {"doc": (x.get("doc") or x), "rank": x.get("rank", i + 1), "distance": x.get("distance")}
            for i, x in enumerate(documents)
        ],
        "after": after,
        "model": model,
    }
