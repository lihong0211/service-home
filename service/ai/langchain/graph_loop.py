"""1. 循环与分支 - 基础功能（think/decide 循环）。"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

from service.ai.langchain.common import _call_llm, _format_history_context

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str
    iteration: int
    query: str  # 用户问题，供 _think 做针对性多轮推理；前端 input 合并时会带入
    response: str  # 最终回答，供前端展示（由 respond 节点填充）


def _think(state: AgentState) -> dict:
    iteration = state["iteration"]
    print(f"🤔 思考中... (第{iteration}轮)")
    prior = "；".join(state["messages"][-3:]) if state["messages"] else "无"
    user_query = (state.get("query") or "").strip()
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = (
        f"这是第 {iteration + 1} 轮思考。"
        f"前几轮已得到：{prior}。"
        "请用一句话给出新的思考或推进，不超过50字。"
    )
    if user_query:
        prompt = f"用户问题：{user_query}\n\n{prompt}"
    if hist_ctx:
        prompt = f"【多轮上文】\n{hist_ctx}\n\n{prompt}"
    thought = _call_llm(prompt, system="你是一个逻辑推理助手，围绕用户问题做多轮思考，每轮产生新的思考进展。")
    print(f"   💡 {thought}")
    return {
        "messages": [thought],
        "iteration": iteration + 1,
    }


def _decide(state: AgentState) -> dict:
    if state["iteration"] < 3:
        print("🔄 需要继续思考，进入循环")
        return {"next_step": "think"}
    print("✅ 思考完成，进入回答")
    return {"next_step": "respond"}


def _loop_respond(state: AgentState) -> dict:
    """根据多轮思考结果 + 用户问题，用 LLM 总结成最终回答。天气/股票等查数请走 router 图。"""
    query = (state.get("query") or "").strip()
    messages = state.get("messages") or []
    prior = "；".join(messages[-5:]) if messages else "无"
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = f"用户问题：{query}\n\n多轮思考要点：{prior}\n\n请用 2～4 句话给出直接、可操作的回答或结论，不要复述思考过程。"
    if hist_ctx:
        prompt = f"【多轮上文】\n{hist_ctx}\n\n{prompt}"
    response = _call_llm(prompt, system="你是助手，根据上述思考给出简洁结论或建议。")
    print(f"📢 最终回答: {(response or '')[:80]}...")
    return {"response": response or "暂无结论，请补充问题或换种问法。"}


def build_loop_graph():
    """
    创建带循环的图：think → decide → (think | respond) → respond → END。
    【模块 3／8：重试机制】think/respond 会调 LLM（外部网络请求），挂 retry_policy；decide 是纯
    逻辑判断、没有外部调用，不需要重试。全项目统一策略：只给"会打外部请求的节点"加重试，纯
    逻辑节点不加——重试解决的是网络抖动/限流这类瞬时故障，纯逻辑节点不存在这类故障源，加了
    也没意义。完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 「重试机制」一节。
    """
    builder = StateGraph(AgentState)
    builder.add_node("think", _think, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("decide", _decide)
    builder.add_node("respond", _loop_respond, retry_policy=RetryPolicy(max_attempts=3))
    builder.set_entry_point("think")
    builder.add_edge("think", "decide")
    builder.add_conditional_edges(
        "decide",
        lambda s: s["next_step"],
        {"think": "think", "respond": "respond"},
    )
    builder.add_edge("respond", END)
    return builder.compile()


def demo_loop():
    """演示循环流程图并打印 ASCII 图。"""
    graph = build_loop_graph()
    print("📊 **循环流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: think → decide → think 或 END)")
    print()
    # 执行一轮演示
    out = graph.invoke(
        {"messages": [], "next_step": "", "iteration": 0, "query": "示例问题", "response": ""}
    )
    print("最终状态 iteration:", out.get("iteration"), "response:", (out.get("response") or "")[:60])
    return graph
