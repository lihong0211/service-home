"""
混合型智能体 - 财富管理投顾AI助手

基于 LangGraph 实现的混合型智能体，结合反应式架构的即时响应能力和深思熟虑架构的长期规划能力，
通过协调层动态切换处理模式，提供智能化财富管理咨询服务。

三层架构：
1. 底层（反应式）：即时响应客户查询，提供快速反馈
2. 中层（协调）：评估任务类型和优先级，动态选择处理模式
3. 顶层（深思熟虑）：进行复杂的投资分析和长期财务规划
"""

import json
from typing import Any, Dict, Literal, Optional, TypedDict

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from service.ai.langchain import graph_to_schema, run_graph_stream_and_collect
from config.ai import DASHSCOPE_BASE_URL, DEFAULT_CHAT_MODEL, dashscope_api_key

_LLM = ChatOpenAI(
    model=DEFAULT_CHAT_MODEL,
    openai_api_key=dashscope_api_key(),
    openai_api_base=DASHSCOPE_BASE_URL,
    temperature=0.7,
    max_tokens=2000,
)


class WealthAdvisorState(TypedDict):
    user_query: str
    customer_profile: Optional[Dict[str, Any]]
    query_type: Optional[Literal["emergency", "informational", "analytical"]]
    processing_mode: Optional[Literal["reactive", "deliberative"]]
    market_data: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    current_phase: Optional[str]
    error: Optional[str]


_ASSESS_PROMPT = """你是一个财富管理投顾AI助手的协调层。评估用户查询，确定类型和处理模式。

用户查询: {user_query}

请以JSON返回：
- query_type: "emergency"|"informational"|"analytical"
- processing_mode: "reactive"|"deliberative"
- reasoning: 决策理由
"""

_REACTIVE_PROMPT = """你是财富管理投顾AI助手，提供快速准确的响应。

用户查询: {user_query}
客户信息: {customer_profile}

请直接、简洁地回答。"""

_DATA_PROMPT = """你是数据收集模块。用户查询: {user_query}，客户信息: {customer_profile}
请收集深入分析所需的市场和财务数据，以JSON返回：required_data_types, collected_data（可模拟）。"""

_ANALYSIS_PROMPT = """你是分析引擎。用户查询: {user_query}，客户信息: {customer_profile}，市场数据: {market_data}
请提供全面的投资分析（JSON）：市场状况、组合分析、个性化建议、风险评估、预期回报。"""

_RECOMMEND_PROMPT = """你是财富管理投顾。根据分析结果为客户准备最终建议。

用户查询: {user_query}
客户信息: {customer_profile}
分析结果: {analysis_results}

请提供自然语言的投资建议：总体策略、具体步骤、资产配置、风险管理、时间框架、预期收益。"""


def _assess_query(state: WealthAdvisorState) -> WealthAdvisorState:
    try:
        chain = ChatPromptTemplate.from_template(_ASSESS_PROMPT) | _LLM | JsonOutputParser()
        result = chain.invoke({"user_query": state["user_query"]})
        mode = result.get("processing_mode") or "reactive"
        if mode not in ("reactive", "deliberative"):
            mode = "reactive"
        qtype = result.get("query_type") or "emergency"
        if qtype not in ("emergency", "informational", "analytical"):
            qtype = "emergency"
        return {**state, "query_type": qtype, "processing_mode": mode}
    except Exception as e:
        return {**state, "error": str(e), "final_response": "评估查询时发生错误。"}


def _reactive_processing(state: WealthAdvisorState) -> WealthAdvisorState:
    try:
        chain = ChatPromptTemplate.from_template(_REACTIVE_PROMPT) | _LLM | StrOutputParser()
        resp = chain.invoke({
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile") or {}, ensure_ascii=False),
        })
        return {**state, "final_response": resp}
    except Exception as e:
        return {**state, "error": str(e), "final_response": "处理时发生错误。"}


def _collect_data(state: WealthAdvisorState) -> WealthAdvisorState:
    try:
        chain = ChatPromptTemplate.from_template(_DATA_PROMPT) | _LLM | JsonOutputParser()
        result = chain.invoke({
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile") or {}, ensure_ascii=False, indent=2),
        })
        return {**state, "market_data": result.get("collected_data", {}), "current_phase": "analyze"}
    except Exception as e:
        return {**state, "error": str(e)}


def _analyze_data(state: WealthAdvisorState) -> WealthAdvisorState:
    if not state.get("market_data"):
        return {**state, "error": "分析阶段缺少市场数据"}
    try:
        chain = ChatPromptTemplate.from_template(_ANALYSIS_PROMPT) | _LLM | JsonOutputParser()
        result = chain.invoke({
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile") or {}, ensure_ascii=False, indent=2),
            "market_data": json.dumps(state.get("market_data") or {}, ensure_ascii=False, indent=2),
        })
        return {**state, "analysis_results": result, "current_phase": "recommend"}
    except Exception as e:
        return {**state, "error": str(e)}


def _generate_recommendations(state: WealthAdvisorState) -> WealthAdvisorState:
    if not state.get("analysis_results"):
        return {**state, "error": "建议生成阶段缺少分析结果"}
    try:
        chain = ChatPromptTemplate.from_template(_RECOMMEND_PROMPT) | _LLM | StrOutputParser()
        result = chain.invoke({
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile") or {}, ensure_ascii=False, indent=2),
            "analysis_results": json.dumps(state.get("analysis_results") or {}, ensure_ascii=False, indent=2),
        })
        return {**state, "final_response": result, "current_phase": "respond"}
    except Exception as e:
        return {**state, "error": str(e)}


def _respond(state: WealthAdvisorState) -> WealthAdvisorState:
    if not state.get("final_response"):
        return {**state, "final_response": "无法生成响应，请稍后重试。", "error": state.get("error", "未知错误")}
    return state


# 节点 id -> 前端展示（供 LangGraph 图谱可视化）
WEALTH_NODE_DISPLAY = {
    "__start__": {"name": "用户输入", "type": "input", "icon": "📝", "description": "用户咨询与画像"},
    "__end__": {"name": "响应输出", "type": "output", "icon": "📢", "description": "投顾建议"},
    "assess": {"name": "协调评估", "type": "llm", "icon": "🎯", "description": "评估查询类型与处理模式"},
    "reactive": {"name": "即时响应", "type": "llm", "icon": "⚡", "description": "快速回答简单咨询"},
    "collect_data": {"name": "数据收集", "type": "tool", "icon": "📊", "description": "收集市场与财务数据"},
    "analyze": {"name": "分析引擎", "type": "llm", "icon": "🔍", "description": "投资与组合分析"},
    "recommend": {"name": "建议生成", "type": "llm", "icon": "💡", "description": "生成个性化建议"},
    "respond": {"name": "最终响应", "type": "process", "icon": "📤", "description": "输出投顾建议"},
}


def create_wealth_advisor_workflow():
    """创建财富顾问混合智能体工作流"""
    workflow = StateGraph(WealthAdvisorState)
    workflow.add_node("assess", _assess_query)
    workflow.add_node("reactive", _reactive_processing)
    workflow.add_node("collect_data", _collect_data)
    workflow.add_node("analyze", _analyze_data)
    workflow.add_node("recommend", _generate_recommendations)
    workflow.add_node("respond", _respond)
    workflow.set_entry_point("assess")
    workflow.add_conditional_edges(
        "assess",
        lambda x: "reactive" if x.get("processing_mode") == "reactive" else "collect_data",
        {"reactive": "reactive", "collect_data": "collect_data"},
    )
    workflow.add_edge("reactive", "respond")
    workflow.add_edge("collect_data", "analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", "respond")
    workflow.add_edge("respond", END)
    return workflow.compile()


def get_wealth_advisor_graph_schema() -> dict:
    """返回财富顾问智能体 LangGraph 图谱信息，供前端展示（nodes + edges）。"""
    graph = create_wealth_advisor_workflow()
    schema = graph_to_schema(graph, node_display=WEALTH_NODE_DISPLAY)
    schema.setdefault("executionOrder", [])
    return schema


def run_wealth_advisor_and_collect_steps(input_state: dict) -> dict:
    """
    执行财富顾问智能体并收集每步信息，供前端按执行顺序与节奏展示。
    返回：graphData（nodes, edges, executionOrder）、steps、finalState、executionOrder。
    """
    graph = create_wealth_advisor_workflow()
    schema = graph_to_schema(graph, node_display=WEALTH_NODE_DISPLAY)
    run_result = run_graph_stream_and_collect(graph, input_state)
    return {
        "graphData": {
            "nodes": schema["nodes"],
            "edges": schema["edges"],
            "executionOrder": run_result["executionOrder"],
        },
        "steps": run_result["steps"],
        "finalState": run_result["finalState"],
        "executionOrder": run_result["executionOrder"],
    }
