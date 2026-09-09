"""2. 并行执行 - 多分支汇聚（使用 Send 或顺序模拟）。"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

from service.ai.langchain.common import _call_llm, _format_history_context

class ParallelState(TypedDict):
    input_text: str
    analyses: Annotated[list, operator.add]  # 并行节点用 append 合并
    final_result: str
    response: str  # 供前端对话区展示，与 final_result 一致或为可读摘要


def _sentiment_analysis(state: ParallelState) -> dict:
    print("🔵 情感分析中...")
    text = state.get("input_text", "")
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = f"对以下文本做情感分析，只返回：positive / negative / neutral 之一。\n\n文本：{text}"
    if hist_ctx:
        prompt = f"【上文参考】\n{hist_ctx}\n\n{prompt}"
    result = _call_llm(
        prompt,
        system="你是情感分析专家，只输出 positive、negative 或 neutral。",
    )
    sentiment = result.strip().lower().split()[0] if result else "neutral"
    print(f"   情感: {sentiment}")
    return {"analyses": [("sentiment", sentiment)]}


def _keyword_extraction(state: ParallelState) -> dict:
    print("🟢 关键词提取中...")
    text = state.get("input_text", "")
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = f"从以下文本中提取3-5个关键词，以英文逗号分隔，只返回关键词列表，不要其他内容。\n\n文本：{text}"
    if hist_ctx:
        prompt = f"【上文参考】\n{hist_ctx}\n\n{prompt}"
    result = _call_llm(prompt, system="你是关键词提取专家。")
    keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
    print(f"   关键词: {keywords}")
    return {"analyses": [("keywords", keywords)]}


def _text_summary(state: ParallelState) -> dict:
    print("🟠 文本摘要中...")
    text = state.get("input_text", "")
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = f"用一句话（不超过30字）概括以下文本的核心内容：\n\n{text}"
    if hist_ctx:
        prompt = f"【上文参考】\n{hist_ctx}\n\n{prompt}"
    summary = _call_llm(prompt, system="你是专业的文本摘要助手。")
    print(f"   摘要: {summary}")
    return {"analyses": [("summary", summary)]}


def _aggregate_results(state: ParallelState) -> dict:
    print("📊 聚合所有分析结果")
    analyses = state.get("analyses") or []
    analysis_dict = dict(analyses) if analyses else {}
    final_result = f"综合结果：{analysis_dict}"
    # 供前端对话区展示的可读文案（关键词/情感/摘要一行一条）
    def _fmt(v):
        return ", ".join(v) if isinstance(v, (list, tuple)) else str(v)

    parts = []
    if "keywords" in analysis_dict:
        parts.append(f"关键词：{_fmt(analysis_dict['keywords'])}")
    if "sentiment" in analysis_dict:
        parts.append(f"情感：{_fmt(analysis_dict['sentiment'])}")
    if "summary" in analysis_dict:
        parts.append(f"摘要：{_fmt(analysis_dict['summary'])}")
    response = "\n".join(parts) if parts else final_result
    return {"final_result": final_result, "response": response}


def build_parallel_graph():
    """
    并行执行图：入口分发到 sentiment / keywords / summary，再汇聚到 aggregate。
    若当前环境不支持 Send，则用顺序边模拟（三节点依次执行后到 aggregate）。
    """
    builder = StateGraph(ParallelState)
    # 三个并行节点都调用 LLM，各自独立加重试；聚合节点是纯逻辑，不需要
    builder.add_node("sentiment", _sentiment_analysis, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("keywords", _keyword_extraction, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("summary", _text_summary, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("aggregate", _aggregate_results)

    try:
        from langgraph.types import Send

        def _dispatch(state: ParallelState):
            return [Send("sentiment", state), Send("keywords", state), Send("summary", state)]

        builder.add_node("dispatch", lambda s: s)  # 透传 state
        builder.set_entry_point("dispatch")
        # path_map 让 get_graph() 能静态解析出所有可能目标，
        # 实际路由仍由 Send 对象决定，两者互不干扰。
        builder.add_conditional_edges(
            "dispatch",
            _dispatch,
            {"sentiment": "sentiment", "keywords": "keywords", "summary": "summary"},
        )
        builder.add_edge("sentiment", "aggregate")
        builder.add_edge("keywords", "aggregate")
        builder.add_edge("summary", "aggregate")
    except ImportError:
        # 无 Send 时：顺序执行三节点再聚合
        builder.set_entry_point("sentiment")
        builder.add_edge("sentiment", "keywords")
        builder.add_edge("keywords", "summary")
        builder.add_edge("summary", "aggregate")

    builder.add_edge("aggregate", END)
    return builder.compile()


def demo_parallel():
    """演示并行（或顺序模拟）流程图。"""
    graph = build_parallel_graph()
    print("📊 **并行执行流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: dispatch → sentiment/keywords/summary → aggregate → END)")
    print()
    out = graph.invoke({"input_text": "示例文本", "analyses": [], "final_result": "", "response": ""})
    print("final_result:", out.get("final_result", "")[:80])
    return graph
