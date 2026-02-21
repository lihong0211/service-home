"""
反应式智能体 - 迪士尼客服助手

使用 LangGraph 的 ReAct Agent 实现的迪士尼客服，可回答关于迪士尼乐园、电影、角色、
门票、园区信息等各种问题，中途会调用知识库「disney_knowledge_base」检索资料。
"""

import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 知识库名称（与项目内创建的知识库一致）
DISNEY_KNOWLEDGE_BASE_NAME = "disney_knowledge_base"

_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
_LLM = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
    max_tokens=1500,
)


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


@tool
def query_disney_knowledge_base(question: str) -> str:
    """在迪士尼知识库中查询并获取答案。用于回答关于迪士尼乐园、电影、角色、门票、园区、酒店、交通等问题。
    question: 用户的问题，例如：上海迪士尼开门时间、有哪些游乐项目、门票价格、如何购票等。"""
    if not (question or "").strip():
        return "请提供要查询的问题。"
    try:
        out = _query_disney_rag(question.strip(), top_k=5)
        answer = (out.get("answer") or "").strip()
        if not answer:
            return "未能在迪士尼知识库中找到相关答案，请换一种方式描述您的问题，或咨询现场工作人员。"
        return answer
    except FileNotFoundError as e:
        return "迪士尼知识库暂未就绪，请稍后再试或联系工作人员。"
    except Exception as e:
        return f"查询知识库时出错：{e}。请换一种方式提问或稍后再试。"


@tool
def search_disney_knowledge_base(query: str, top_k: int = 5) -> str:
    """在迪士尼知识库中检索与关键词/问题相关的资料片段（不生成答案，仅返回检索到的原文）。
    适用于需要多段资料综合回答、或需要引用原文时。
    query: 检索关键词或问题；top_k: 返回最多几条资料，默认 5。"""
    if not (query or "").strip():
        return "请提供检索关键词或问题。"
    try:
        from service.ai.vector_db import search_in_db
        from model.ai import VectorDb, KnowledgeBase

        # 解析知识库名称到向量库名称（与 rag.py 一致）
        row = VectorDb.select_one_by({"name": DISNEY_KNOWLEDGE_BASE_NAME})
        if not row:
            kb = KnowledgeBase.select_one_by({"name": DISNEY_KNOWLEDGE_BASE_NAME})
            if kb:
                if kb.vector_db_id:
                    row = VectorDb.get_by_id(kb.vector_db_id)
                if not row:
                    row = VectorDb.select_one_by({"name": f"kb_{kb.id}"})
        if not row:
            return "迪士尼知识库暂未就绪。"
        results = search_in_db(row.name, query.strip(), top_k=max(1, min(10, int(top_k))))
        if not results:
            return "未检索到与「{}」相关的资料。可尝试其他关键词。".format(query.strip()[:50])
        parts = []
        for i, r in enumerate(results, 1):
            doc = r.get("doc") if isinstance(r.get("doc"), dict) else r
            text = (doc.get("text") or "").strip()
            if text:
                parts.append("[{}] {}".format(i, text))
        return "\n\n---\n\n".join(parts) if parts else "未检索到有效资料。"
    except FileNotFoundError:
        return "迪士尼知识库暂未就绪。"
    except Exception as e:
        return "检索时出错：{}。".format(e)


def create_fund_qa_agent():
    """创建迪士尼客服 Agent（沿用 fund_qa 入口名以兼容现有路由）。"""
    tools = [query_disney_knowledge_base, search_disney_knowledge_base]
    return create_react_agent(
        model=_LLM,
        tools=tools,
        prompt="""你是迪士尼客服助手，专门回答用户关于迪士尼的各种问题，包括但不限于：
- 迪士尼乐园（上海、香港等）的开放时间、门票、交通、园区与游乐项目
- 迪士尼电影、角色、周边与活动
- 购票方式、年卡、尊享卡、酒店与餐饮

请优先使用「迪士尼知识库」工具（query_disney_knowledge_base 或 search_disney_knowledge_base）获取准确信息后再回答。
若知识库中无相关内容，可礼貌说明并建议用户换一种问法或联系现场/官方渠道。
回答请友好、简洁、有条理。""",
    )
