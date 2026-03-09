"""
迪士尼客服助手 - RAG 直连实现（带流式过程）

拆分为两个可见步骤，给前端 3D 动画提供过程感：
  Step 1 → retrieval : 向量检索（1 次 embedding）
  Step 2 → generation: 生成回答（1 次 chat/completions）

.stream() 逐步 yield，.invoke() 复用缓存，彻底杜绝双重调用。
"""

import os

DISNEY_KNOWLEDGE_BASE_NAME = "disney_knowledge_base"
_MODEL = "qwen-turbo"


def _extract_question(input_data: dict) -> str:
    """从 messages 中取最后一条 user 内容作为问题。"""
    for m in reversed(input_data.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _resolve_vector_db():
    """将知识库名称解析为向量库行记录。"""
    from model.ai import VectorDb, KnowledgeBase
    row = VectorDb.select_one_by({"name": DISNEY_KNOWLEDGE_BASE_NAME})
    if not row:
        kb = KnowledgeBase.select_one_by({"name": DISNEY_KNOWLEDGE_BASE_NAME})
        if kb:
            if kb.vector_db_id:
                row = VectorDb.get_by_id(kb.vector_db_id)
            if not row:
                row = VectorDb.select_one_by({"name": f"kb_{kb.id}"})
    return row


def _build_sources(results: list) -> tuple[list, list]:
    """从检索结果构造 context_parts 和 sources 两份数据。"""
    context_parts, sources = [], []
    for r in results:
        doc = r.get("doc") if isinstance(r.get("doc"), dict) else r
        text = (doc.get("text") or "").strip()
        context_parts.append(text)
        sources.append({
            "doc_id": doc.get("id"),
            "text": text[:200] + ("..." if len(text) > 200 else ""),
            "category": doc.get("category"),
            "distance": r.get("distance"),
        })
    return context_parts, sources


def _make_final_state(query: str, response: str, sources: list) -> dict:
    """
    构造与 langchain 一致的 finalState 结构，供 stream/invoke 统一返回。
    包含 query、response、sources、messages，便于前端与 LangGraph 的 state 格式一致。
    """
    return {
        "query": query or "",
        "response": response or "",
        "sources": sources if isinstance(sources, list) else [],
        "messages": [
            {"role": "user", "content": query or ""},
            {"role": "assistant", "content": response or ""},
        ],
    }


class _DisneyRagAgent:
    """
    两步流式 RAG Agent：
      stream() → yield retrieval step → yield generation step
      invoke() → 直接返回 stream() 已缓存的结果，不重复调用 API
    """

    def __init__(self):
        self._cached: dict | None = None

    # ------------------------------------------------------------------
    # 流式接口：供 agent.py hasattr(agent, "stream") 分支使用
    # ------------------------------------------------------------------
    def stream(self, input_data: dict, config: dict = None):
        question = _extract_question(input_data)
        if not question:
            result = _make_final_state("", "请提供您的问题。", [])
            self._cached = result
            yield {"generation": result}
            return

        try:
            from service.ai.vector_db import search_in_db, client

            # ── Step 1: 向量检索（step 输出与 langchain 的 node output 一致：含 query/sources 等）──
            row = _resolve_vector_db()
            if not row:
                result = _make_final_state(question, "迪士尼知识库暂未就绪，请稍后再试或联系工作人员。", [])
                self._cached = result
                yield {"retrieval": {"query": question, "hits": 0, "sources": []}}
                yield {"generation": result}
                return

            results = search_in_db(row.name, question, top_k=5)
            context_parts, sources = _build_sources(results)
            yield {"retrieval": {"query": question, "hits": len(results), "sources": sources}}

            # ── Step 2: 大模型生成（generation 步输出完整 state，与 invoke 返回的 finalState 一致）──
            if not results:
                result = _make_final_state(
                    question,
                    "未能在迪士尼知识库中找到相关答案，请换一种方式描述您的问题，或咨询现场工作人员。",
                    [],
                )
                self._cached = result
                yield {"generation": result}
                return

            context = "\n\n---\n\n".join(context_parts)
            prompt = (
                f"基于以下参考资料回答问题。若资料中无相关内容，请说明无法从资料中得出答案。\n\n"
                f"参考资料：\n{context}\n\n"
                f"问题：{question}\n\n"
                f"请直接给出答案（可简要说明依据的段落）："
            )
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            answer = (resp.choices[0].message.content or "").strip()
            result = _make_final_state(question, answer, sources)
            self._cached = result
            yield {"generation": result}

        except FileNotFoundError:
            result = _make_final_state(question or "", "迪士尼知识库暂未就绪，请稍后再试或联系工作人员。", [])
            self._cached = result
            yield {"generation": result}
        except Exception as e:
            result = _make_final_state(question or "", f"查询出错：{e}，请稍后再试。", [])
            self._cached = result
            yield {"generation": result}

    # ------------------------------------------------------------------
    # 同步接口：agent.py stream 结束后调 invoke() 取最终 state
    # 复用缓存，不重复调用 API
    # ------------------------------------------------------------------
    def invoke(self, input_data: dict, config: dict = None) -> dict:
        if self._cached is not None:
            result = self._cached
            self._cached = None
            return result

        # 兜底：未经 stream() 直接调用 invoke() 时走完整 RAG，返回与 langchain 一致的 finalState 结构
        question = _extract_question(input_data)
        if not question:
            return _make_final_state("", "请提供您的问题。", [])
        try:
            from service.ai.rag import rag_chat
            out = rag_chat(
                kb_name=DISNEY_KNOWLEDGE_BASE_NAME,
                question=question,
                top_k=5,
                model=_MODEL,
                enable_query_rewrite=False,
                enable_rerank=False,
            )
            answer = (out.get("answer") or "").strip() or "未能找到相关答案。"
            return _make_final_state(question, answer, out.get("sources", []))
        except FileNotFoundError:
            return _make_final_state(question, "迪士尼知识库暂未就绪，请稍后再试或联系工作人员。", [])
        except Exception as e:
            return _make_final_state(question, f"查询出错：{e}，请稍后再试。", [])


def create_fund_qa_agent() -> _DisneyRagAgent:
    """创建迪士尼客服 Agent（沿用 fund_qa 入口名以兼容现有路由）。"""
    return _DisneyRagAgent()
