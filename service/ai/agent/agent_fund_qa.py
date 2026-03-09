"""
迪士尼客服助手 - 直接 RAG 实现

不使用 ReAct Agent，直接调用知识库 RAG，每次问答只需：
  1 次 embedding（向量检索）+ 1 次 chat/completions（生成回答）
"""

import os

DISNEY_KNOWLEDGE_BASE_NAME = "disney_knowledge_base"

_API_KEY = os.environ.get("DASHSCOPE_API_KEY")


def _query_disney_rag(question: str, top_k: int = 5) -> dict:
    """调用 RAG 模块，在迪士尼知识库中检索并生成答案。"""
    from service.ai.rag import rag_chat
    return rag_chat(
        kb_name=DISNEY_KNOWLEDGE_BASE_NAME,
        question=question.strip(),
        top_k=top_k,
        model="qwen-turbo",
        enable_query_rewrite=False,
        enable_rerank=False,
    )


class _DisneyRagAgent:
    """
    轻量包装：对外暴露 .invoke(input_data, config) 接口，与 agent.py 兼容。
    没有 .stream 属性，agent.py 会走 else 分支，只调用一次 invoke。
    """

    def invoke(self, input_data: dict, config: dict = None) -> dict:
        # 从 messages 中取最后一条 user 消息作为问题
        messages = input_data.get("messages") or []
        question = ""
        for m in reversed(messages):
            role = m.get("role", "") if isinstance(m, dict) else ""
            if role == "user":
                question = (m.get("content") or "").strip()
                break

        if not question:
            return {"response": "请提供您的问题。", "sources": []}

        try:
            out = _query_disney_rag(question)
            answer = (out.get("answer") or "").strip()
            if not answer:
                answer = "未能在迪士尼知识库中找到相关答案，请换一种方式描述您的问题，或咨询现场工作人员。"
            return {"response": answer, "sources": out.get("sources", [])}
        except FileNotFoundError:
            return {"response": "迪士尼知识库暂未就绪，请稍后再试或联系工作人员。", "sources": []}
        except Exception as e:
            return {"response": f"查询出错：{e}，请稍后再试。", "sources": []}


def create_fund_qa_agent() -> _DisneyRagAgent:
    """创建迪士尼客服 Agent（沿用 fund_qa 入口名以兼容现有路由）。"""
    return _DisneyRagAgent()
