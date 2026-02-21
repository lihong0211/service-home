"""
智能体系统 - 统一管理3个智能体，提供前端接口

包含的智能体：
1. research_agent - 深思熟虑型：智能投研助手
2. fund_qa_agent - 反应式：迪士尼客服助手（使用知识库 disney_knowledge_base）
3. wealth_advisor_agent - 混合型：财富管理投顾AI助手
"""

from __future__ import annotations

import time
from typing import Optional

from service.ai.agent.agent_fund_qa import create_fund_qa_agent
from service.ai.agent.agent_research import (
    create_research_agent_workflow,
    RESEARCH_NODE_DISPLAY,
)
from service.ai.agent.agent_wealth_advisor import (
    create_wealth_advisor_workflow,
    WEALTH_NODE_DISPLAY,
)
from service.ai.langchain import graph_to_schema

# 智能体构建器映射
AGENT_BUILDERS = {
    "research_agent": create_research_agent_workflow,
    "fund_qa_agent": create_fund_qa_agent,
    "wealth_advisor_agent": create_wealth_advisor_workflow,
}

# 智能体元信息（名称、描述、类型）
AGENT_META = {
    "research_agent": {
        "name": "智能投研助手",
        "description": "深思熟虑型智能体，适用于投资研究场景，多步骤分析和推理，生成投资观点和研究报告。",
        "type": "deliberative",
        "icon": "📊",
    },
    "fund_qa_agent": {
        "name": "迪士尼客服助手",
        "description": "反应式智能体，回答关于迪士尼乐园、电影、角色、门票、园区等问题，使用知识库 disney_knowledge_base 检索。",
        "type": "reactive",
        "icon": "🏰",
    },
    "wealth_advisor_agent": {
        "name": "财富管理投顾助手",
        "description": "混合型智能体，结合反应式与深思熟虑，提供财富管理咨询服务。",
        "type": "hybrid",
        "icon": "💰",
    },
}

# 默认输入（用于演示）
DEFAULT_INPUTS = {
    "research_agent": {
        "research_topic": "新能源汽车行业投资机会",
        "industry_focus": "电动汽车制造、电池技术",
        "time_horizon": "中期",
        "perception_data": None,
        "world_model": None,
        "reasoning_plans": None,
        "selected_plan": None,
        "final_report": None,
        "current_phase": "perception",
        "error": None,
    },
    "fund_qa_agent": {
        "messages": [{"role": "user", "content": "上海迪士尼乐园的开放时间是多少？"}]
    },
    "wealth_advisor_agent": {
        "user_query": "根据当前市场情况，我应该如何调整投资组合？",
        "customer_profile": None,
        "query_type": None,
        "processing_mode": None,
        "market_data": None,
        "analysis_results": None,
        "final_response": None,
        "current_phase": None,
        "error": None,
    },
}


def list_agents():
    """返回所有可用的智能体列表（含元信息）。"""
    return {aid: {"id": aid, **meta} for aid, meta in AGENT_META.items()}


def get_agent_schema(agent_id: str) -> Optional[dict]:
    """
    获取智能体的图结构（GraphData），供前端 3D 可视化。
    对于 StateGraph 类型的智能体，从编译后的图动态生成；
    对于非 StateGraph 类型（如 fund_qa_agent），返回简化结构。
    """
    if agent_id not in AGENT_BUILDERS:
        return None

    builder_fn = AGENT_BUILDERS[agent_id]
    agent = builder_fn()

    # 检查是否为 StateGraph（有 get_graph 方法）
    if hasattr(agent, "get_graph"):
        node_display = None
        if agent_id == "research_agent":
            node_display = RESEARCH_NODE_DISPLAY
        elif agent_id == "wealth_advisor_agent":
            node_display = WEALTH_NODE_DISPLAY
        schema = graph_to_schema(agent, node_display=node_display)
        schema["executionOrder"] = []  # 真实顺序由 POST /run 返回
        return schema

    # 对于非 StateGraph 类型（如 fund_qa_agent），返回简化结构
    meta = AGENT_META.get(agent_id, {})
    return {
        "nodes": [
            {
                "id": "input",
                "name": "用户输入",
                "type": "input",
                "icon": "📝",
                "description": "接收用户查询",
            },
            {
                "id": "agent",
                "name": meta.get("name", agent_id),
                "type": "process",
                "icon": meta.get("icon", "🤖"),
                "description": meta.get("description", ""),
            },
            {
                "id": "output",
                "name": "输出",
                "type": "output",
                "icon": "📢",
                "description": "返回结果",
            },
        ],
        "edges": [
            {"source": "input", "target": "agent", "type": "normal"},
            {"source": "agent", "target": "output", "type": "normal"},
        ],
        "executionOrder": [],
    }


def run_agent_and_collect_steps(agent_id: str, input_data: Optional[dict] = None):
    """
    执行指定智能体，收集每一步的执行信息，供前端按真实执行顺序驱动 3D 动画。
    返回：{
        "agentMeta": { id, name, description, type, icon },
        "graphData": { nodes, edges, executionOrder },
        "steps": [ { "nodeId", "status": "end", "duration_ms", "output" }, ... ],
        "finalState": { ... },
        "executionOrder": [ "node1", "node2", ... ]
    }
    """
    if agent_id not in AGENT_BUILDERS:
        return {
            "error": f"未知智能体: {agent_id}",
            "allowed": list(AGENT_BUILDERS.keys()),
        }

    builder_fn = AGENT_BUILDERS[agent_id]
    agent = builder_fn()
    meta = AGENT_META.get(agent_id, {})

    # 准备输入
    if input_data is None:
        input_data = DEFAULT_INPUTS.get(agent_id, {})

    steps = []
    execution_order = []
    t0 = time.perf_counter()

    try:
        # 检查是否为 StateGraph
        if hasattr(agent, "stream"):
            # StateGraph 类型：使用 stream 收集步骤
            for step in agent.stream(input_data):
                for node_id, output in step.items():
                    t1 = time.perf_counter()
                    duration_ms = round((t1 - t0) * 1000)
                    t0 = t1
                    steps.append(
                        {
                            "nodeId": node_id,
                            "status": "end",
                            "duration_ms": duration_ms,
                            "output": output,
                        }
                    )
                    execution_order.append(node_id)
            # 获取最终状态
            final_state = agent.invoke(input_data)
        else:
            # 非 StateGraph 类型（如 fund_qa_agent）：直接 invoke
            config = {"configurable": {"thread_id": f"{agent_id}-{int(time.time())}"}}
            t_start = time.perf_counter()
            final_state = agent.invoke(input_data, config)
            duration_ms = round((time.perf_counter() - t_start) * 1000)
            steps.append(
                {
                    "nodeId": "agent",
                    "status": "end",
                    "duration_ms": duration_ms,
                    "output": final_state,
                }
            )
            execution_order = ["agent"]

        # 获取图结构
        graph_data = get_agent_schema(agent_id) or {
            "nodes": [],
            "edges": [],
            "executionOrder": [],
        }
        if execution_order:
            graph_data["executionOrder"] = execution_order

        return {
            "agentMeta": {
                "id": agent_id,
                **meta,
            },
            "graphData": graph_data,
            "steps": steps,
            "finalState": final_state,
            "executionOrder": execution_order,
        }
    except Exception as e:
        return {"error": str(e)}
