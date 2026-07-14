"""
深思熟虑型智能体 - 智能投研助手

基于 LangGraph 实现的深思熟虑型智能体，适用于投资研究场景，能够整合数据，
进行多步骤分析和推理，生成投资观点和研究报告。

核心流程：
1. 感知：收集市场数据和信息
2. 建模：构建内部世界模型，理解市场状态
3. 推理：生成多个候选分析方案并模拟结果
4. 决策：选择最优投资观点并形成报告
5. 报告：生成完整研究报告
"""

import json
from typing import Any, Dict, List, Literal, Optional, TypedDict

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


class ResearchAgentState(TypedDict):
    research_topic: str
    industry_focus: str
    time_horizon: str
    perception_data: Optional[Dict[str, Any]]
    world_model: Optional[Dict[str, Any]]
    reasoning_plans: Optional[List[Dict[str, Any]]]
    selected_plan: Optional[Dict[str, Any]]
    final_report: Optional[str]
    current_phase: Optional[str]
    error: Optional[str]


_PERCEPTION_PROMPT = """你是一个专业的投资研究分析师，请收集和整理关于以下研究主题的市场数据和信息：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

请从以下几个方面进行市场感知：
1. 市场概况和最新动态
2. 关键经济和市场指标
3. 近期重要新闻（至少3条）
4. 行业趋势分析（至少针对3个细分领域）

根据你的专业知识和经验，提供尽可能详细和准确的信息。

请以JSON格式输出，包含以下字段：
- market_overview: 字符串，市场概况
- key_indicators: 对象，键为指标名称，值为指标值和简要解释
- recent_news: 数组，每项为一条重要新闻
- industry_trends: 对象，键为细分领域，值为趋势分析
"""

_MODELING_PROMPT = """你是一个资深投资策略师，请根据以下市场数据和信息，构建市场内部模型，进行深度分析：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场数据和信息:
{perception_data}

请构建一个全面的市场内部模型，包括：
1. 当前市场状态评估
2. 经济周期判断
3. 主要风险因素（至少3个）
4. 潜在机会领域（至少3个）
5. 市场情绪分析

请以JSON格式输出，包含以下字段：
- market_state: 字符串，市场状态
- economic_cycle: 字符串，经济周期
- risk_factors: 数组，风险因素列表
- opportunity_areas: 数组，机会领域列表
- market_sentiment: 字符串，市场情绪
"""

_REASONING_PROMPT = """你是一个战略投资顾问，请根据以下市场模型，生成3个不同的投资分析方案：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场内部模型:
{world_model}

请为每个方案提供：plan_id, hypothesis, analysis_approach, expected_outcome, confidence_level, pros, cons。
请以JSON数组格式输出。
"""

_DECISION_PROMPT = """你是一个投资决策委员会主席，请评估以下候选分析方案，选择最优方案并形成投资决策。

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场内部模型:
{world_model}

候选分析方案:
{reasoning_plans}

请以JSON格式输出，包含以下字段：
- selected_plan_id: 字符串
- investment_thesis: 字符串
- supporting_evidence: 数组
- risk_assessment: 字符串
- recommendation: 字符串
- timeframe: 字符串
"""

_REPORT_PROMPT = """你是一个专业的投资研究报告撰写人，请根据以下信息生成一份完整的投资研究报告：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场数据和信息:
{perception_data}

市场内部模型:
{world_model}

选定的投资决策:
{selected_plan}

请生成一份结构完整、逻辑清晰的投研报告，包括：报告标题和摘要、市场和行业背景、核心投资观点、详细分析论证、风险因素、投资建议、时间框架和预期回报。
"""


def _research_perception(state: ResearchAgentState) -> ResearchAgentState:
    try:
        chain = ChatPromptTemplate.from_template(_PERCEPTION_PROMPT) | _LLM | JsonOutputParser()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
        })
        return {**state, "perception_data": result, "current_phase": "modeling"}
    except Exception as e:
        return {**state, "error": str(e), "current_phase": "perception"}


def _research_modeling(state: ResearchAgentState) -> ResearchAgentState:
    if not state.get("perception_data"):
        return {**state, "error": "建模阶段缺少感知数据", "current_phase": "perception"}
    try:
        chain = ChatPromptTemplate.from_template(_MODELING_PROMPT) | _LLM | JsonOutputParser()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "perception_data": json.dumps(state["perception_data"], ensure_ascii=False, indent=2),
        })
        return {**state, "world_model": result, "current_phase": "reasoning"}
    except Exception as e:
        return {**state, "error": str(e), "current_phase": "modeling"}


def _research_reasoning(state: ResearchAgentState) -> ResearchAgentState:
    if not state.get("world_model"):
        return {**state, "error": "推理阶段缺少世界模型", "current_phase": "modeling"}
    try:
        chain = ChatPromptTemplate.from_template(_REASONING_PROMPT) | _LLM | JsonOutputParser()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "world_model": json.dumps(state["world_model"], ensure_ascii=False, indent=2),
        })
        return {**state, "reasoning_plans": result, "current_phase": "decision"}
    except Exception as e:
        return {**state, "error": str(e), "current_phase": "reasoning"}


def _research_decision(state: ResearchAgentState) -> ResearchAgentState:
    if not state.get("reasoning_plans"):
        return {**state, "error": "决策阶段缺少候选方案", "current_phase": "reasoning"}
    try:
        chain = ChatPromptTemplate.from_template(_DECISION_PROMPT) | _LLM | JsonOutputParser()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "world_model": json.dumps(state["world_model"], ensure_ascii=False, indent=2),
            "reasoning_plans": json.dumps(state["reasoning_plans"], ensure_ascii=False, indent=2),
        })
        return {**state, "selected_plan": result, "current_phase": "report"}
    except Exception as e:
        return {**state, "error": str(e), "current_phase": "decision"}


def _research_report(state: ResearchAgentState) -> ResearchAgentState:
    if not state.get("selected_plan"):
        return {**state, "error": "报告阶段缺少选定方案", "current_phase": "decision"}
    try:
        chain = ChatPromptTemplate.from_template(_REPORT_PROMPT) | _LLM | StrOutputParser()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "perception_data": json.dumps(state["perception_data"], ensure_ascii=False, indent=2),
            "world_model": json.dumps(state["world_model"], ensure_ascii=False, indent=2),
            "selected_plan": json.dumps(state["selected_plan"], ensure_ascii=False, indent=2),
        })
        return {**state, "final_report": result, "current_phase": "completed"}
    except Exception as e:
        return {**state, "error": str(e), "current_phase": "report"}


# 节点 id -> 前端展示（供 LangGraph 图谱可视化）
RESEARCH_NODE_DISPLAY = {
    "__start__": {"name": "用户输入", "type": "input", "icon": "📝", "description": "研究主题与参数"},
    "__end__": {"name": "报告输出", "type": "output", "icon": "📢", "description": "投研报告"},
    "perception": {"name": "感知", "type": "llm", "icon": "🔍", "description": "收集市场数据与信息"},
    "modeling": {"name": "建模", "type": "llm", "icon": "📐", "description": "构建市场内部模型"},
    "reasoning": {"name": "推理", "type": "llm", "icon": "🤔", "description": "生成候选分析方案"},
    "decision": {"name": "决策", "type": "condition", "icon": "🎯", "description": "选择最优方案"},
    "report": {"name": "报告", "type": "llm", "icon": "📄", "description": "生成投研报告"},
}


def create_research_agent_workflow():
    """创建深思熟虑型研究智能体工作流图"""
    workflow = StateGraph(ResearchAgentState)
    workflow.add_node("perception", _research_perception)
    workflow.add_node("modeling", _research_modeling)
    workflow.add_node("reasoning", _research_reasoning)
    workflow.add_node("decision", _research_decision)
    workflow.add_node("report", _research_report)
    workflow.set_entry_point("perception")
    workflow.add_edge("perception", "modeling")
    workflow.add_edge("modeling", "reasoning")
    workflow.add_edge("reasoning", "decision")
    workflow.add_edge("decision", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def get_research_agent_graph_schema() -> dict:
    """返回研究智能体 LangGraph 图谱信息，供前端展示（nodes + edges）。"""
    graph = create_research_agent_workflow()
    schema = graph_to_schema(graph, node_display=RESEARCH_NODE_DISPLAY)
    schema.setdefault("executionOrder", [])
    return schema


def run_research_agent_and_collect_steps(input_state: dict) -> dict:
    """
    执行研究智能体并收集每步信息，供前端按执行顺序与节奏展示。
    返回：graphData（nodes, edges, executionOrder）、steps、finalState、executionOrder。
    """
    graph = create_research_agent_workflow()
    schema = graph_to_schema(graph, node_display=RESEARCH_NODE_DISPLAY)
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
