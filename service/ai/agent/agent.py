"""
智能体系统 - 统一管理3个智能体，提供前端接口

包含的智能体：
1. research_agent - 深思熟虑型：智能投研助手
2. fund_qa_agent - 反应式：迪士尼客服助手（使用知识库 disney_knowledge_base）
3. wealth_advisor_agent - 混合型：财富管理投顾AI助手
"""

from __future__ import annotations

import time
from typing import Any, Optional

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

# 迪士尼客服等 ReAct 类 agent 的节点展示名（供前端 step 展示）
FUND_QA_NODE_LABELS = {
    "agent": "推理",
    "tools": "知识库检索",
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


def _lc_message_to_dict(msg: Any) -> dict:
    """将 LangChain HumanMessage/AIMessage 等转为可 JSON 序列化的 dict。"""
    role = getattr(msg, "type", None) or getattr(msg, "type_", "user")
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text", c.get("content", str(c))))
            else:
                parts.append(str(c))
        content = "\n".join(parts) if parts else ""
    return {"role": role, "content": content if isinstance(content, str) else str(content)}


def _ensure_json_serializable(obj: Any) -> Any:
    """
    递归将 state/output 中的 LangChain Message 等转为可 JSON 序列化的结构。
    避免 Flask jsonify 报错：Object of type HumanMessage is not JSON serializable。
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "type") and hasattr(obj, "content"):
        return _lc_message_to_dict(obj)
    if getattr(obj, "__class__", None) and getattr(obj.__class__, "__name__", "").endswith("Message"):
        return _lc_message_to_dict(obj)
    if isinstance(obj, dict):
        return {k: _ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_json_serializable(x) for x in obj]
    try:
        import json as _json
        _json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _extract_response_from_messages(messages: list) -> str:
    """从 ReAct/LCEL 的 messages 中取最后一条 assistant 的 content，供前端 finalState.response。支持 dict 或 LangChain Message。"""
    if not messages:
        return ""
    for m in reversed(messages):
        if isinstance(m, dict):
            role = (m.get("role") or "").lower()
            content = m.get("content")
        elif hasattr(m, "type") and hasattr(m, "content"):
            t = getattr(m, "type", "") or getattr(m, "type_", "")
            role = "assistant" if t == "ai" else ("user" if t == "human" else t)
            content = getattr(m, "content", "")
        else:
            continue
        if role != "assistant":
            continue
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = (part.get("text") or "").strip()
                    if text:
                        return text
    return ""


def _enrich_step_for_agent(agent_id: str, step_index: int, node_id: str, step_payload: dict) -> dict:
    """为前端补充 stepIndex、label 等，与 langchain run 的 step 格式一致。"""
    out = dict(step_payload)
    out["stepIndex"] = step_index
    if agent_id == "fund_qa_agent":
        out["label"] = FUND_QA_NODE_LABELS.get(node_id, node_id)
    return out


def run_agent_and_collect_steps(agent_id: str, input_data: Optional[dict] = None):
    """
    执行指定智能体，收集每一步的执行信息，供前端按真实执行顺序驱动 3D 动画。
    返回：{
        "agentMeta": { id, name, description, type, icon },
        "graphData": { nodes, edges, executionOrder },
        "steps": [ { "stepIndex", "nodeId", "status", "duration_ms", "output", "label"? }, ... ],
        "finalState": { ... , "response"? 供迪士尼等展示最终回答 },
        "executionOrder": [ "node1", "node2", ... ],
        "totalSteps": N
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
    step_index = 0
    t0 = time.perf_counter()

    try:
        # 检查是否为 StateGraph（含 ReAct 等有 stream 的 agent）
        if hasattr(agent, "stream"):
            config = {"configurable": {"thread_id": f"{agent_id}-{int(time.time())}"}}
            for step in agent.stream(input_data, config=config):
                for node_id, output in step.items():
                    t1 = time.perf_counter()
                    duration_ms = round((t1 - t0) * 1000)
                    t0 = t1
                    raw = {
                        "nodeId": node_id,
                        "status": "end",
                        "duration_ms": duration_ms,
                        "output": _ensure_json_serializable(output),
                    }
                    steps.append(_enrich_step_for_agent(agent_id, step_index, node_id, raw))
                    execution_order.append(node_id)
                    step_index += 1
            final_state = agent.invoke(input_data, config=config)
        else:
            config = {"configurable": {"thread_id": f"{agent_id}-{int(time.time())}"}}
            t_start = time.perf_counter()
            final_state = agent.invoke(input_data, config)
            duration_ms = round((time.perf_counter() - t_start) * 1000)
            raw = {
                "nodeId": "agent",
                "status": "end",
                "duration_ms": duration_ms,
                "output": _ensure_json_serializable(final_state),
            }
            steps.append(_enrich_step_for_agent(agent_id, 0, "agent", raw))
            execution_order = ["agent"]
            step_index = 1

        # 迪士尼等 ReAct：finalState 中补充 response 供前端展示
        if agent_id == "fund_qa_agent" and isinstance(final_state, dict):
            msgs = final_state.get("messages") or final_state.get("message") or []
            if not isinstance(final_state.get("response"), str):
                final_state = dict(final_state)
                final_state["response"] = _extract_response_from_messages(msgs)

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
            "finalState": _ensure_json_serializable(final_state),
            "executionOrder": execution_order,
            "totalSteps": len(steps),
        }
    except Exception as e:
        return {"error": str(e)}


def run_agent_stream_yield_events(agent_id: str, input_data: Optional[dict] = None):
    """
    执行智能体并逐步 yield 事件，与 langchain run_graph_stream_yield_events 对齐。
    yield: ("step", { stepIndex, nodeId, status, duration_ms, output, label? }) 或 ("done", { finalState, steps, totalSteps, executionOrder })。
    供 POST /ai/agent/run 的 stream=true 时 SSE 使用。
    """
    if agent_id not in AGENT_BUILDERS:
        yield ("error", {"error": f"未知智能体: {agent_id}"})
        return

    builder_fn = AGENT_BUILDERS[agent_id]
    agent = builder_fn()
    meta = AGENT_META.get(agent_id, {})
    if input_data is None:
        input_data = DEFAULT_INPUTS.get(agent_id, {})

    steps = []
    execution_order = []
    step_index = 0
    t0 = time.perf_counter()

    try:
        if hasattr(agent, "stream"):
            config = {"configurable": {"thread_id": f"{agent_id}-{int(time.time())}"}}
            for step in agent.stream(input_data, config=config):
                for node_id, output in step.items():
                    t1 = time.perf_counter()
                    duration_ms = round((t1 - t0) * 1000)
                    t0 = t1
                    raw = {
                        "nodeId": node_id,
                        "status": "end",
                        "duration_ms": duration_ms,
                        "output": _ensure_json_serializable(output),
                    }
                    step_payload = _enrich_step_for_agent(agent_id, step_index, node_id, raw)
                    steps.append(step_payload)
                    execution_order.append(node_id)
                    step_index += 1
                    yield ("step", step_payload)
            final_state = agent.invoke(input_data, config=config)
        else:
            config = {"configurable": {"thread_id": f"{agent_id}-{int(time.time())}"}}
            t_start = time.perf_counter()
            final_state = agent.invoke(input_data, config=config)
            duration_ms = round((time.perf_counter() - t_start) * 1000)
            raw = {"nodeId": "agent", "status": "end", "duration_ms": duration_ms, "output": _ensure_json_serializable(final_state)}
            step_payload = _enrich_step_for_agent(agent_id, 0, "agent", raw)
            steps.append(step_payload)
            execution_order = ["agent"]
            step_index = 1
            yield ("step", step_payload)

        if agent_id == "fund_qa_agent" and isinstance(final_state, dict):
            msgs = final_state.get("messages") or final_state.get("message") or []
            if not isinstance(final_state.get("response"), str):
                final_state = dict(final_state)
                final_state["response"] = _extract_response_from_messages(msgs)

        graph_data = get_agent_schema(agent_id) or {"nodes": [], "edges": [], "executionOrder": []}
        if execution_order:
            graph_data["executionOrder"] = execution_order

        yield ("done", {
            "agentMeta": {"id": agent_id, **meta},
            "graphData": graph_data,
            "finalState": _ensure_json_serializable(final_state),
            "steps": steps,
            "executionOrder": execution_order,
            "totalSteps": len(steps),
        })
    except Exception as e:
        yield ("error", {"error": str(e)})


# ---------------------------------------------------------------------------
# Flask 视图：供 routes/ai 注册 GET/POST
# ---------------------------------------------------------------------------


def agent_list_api():
    """GET /ai/agent/list 返回所有可用的智能体列表（含元信息）。"""
    from flask import jsonify

    agents = list_agents()
    return jsonify({"code": 0, "msg": "ok", "data": agents})


def agent_schema_api():
    """GET /ai/agent/schema?agent_id=research_agent 返回智能体的图结构，供前端 3D 可视化。"""
    from flask import request, jsonify

    agent_id = request.args.get("agent_id") or "research_agent"
    schema = get_agent_schema(agent_id)
    if schema is None:
        return (
            jsonify(
                {
                    "code": 400,
                    "msg": f"未知智能体: {agent_id}",
                    "data": {"allowed": list(list_agents().keys())},
                }
            ),
            400,
        )
    return jsonify({"code": 0, "msg": "ok", "data": schema})


def agent_run_api():
    """
    POST /ai/agent/run 执行智能体并返回步骤与最终状态，供前端按真实执行顺序驱动 3D 动画。
    与 langchain run 对齐：支持 stream=true 时走 SSE，先发 init（graphData），再逐步 step，最后 done（含 finalState、steps、totalSteps）。
    """
    from flask import request, jsonify, Response, stream_with_context

    body = request.get_json() or {}
    agent_id = body.get("agent_id") or "research_agent"
    input_data = body.get("input")
    stream = body.get("stream", False)

    if stream:
        import json as _json
        meta = AGENT_META.get(agent_id, {})
        graph_data = get_agent_schema(agent_id) or {"nodes": [], "edges": [], "executionOrder": []}

        def gen():
            try:
                yield f"data: {_json.dumps({'type': 'init', 'graphData': graph_data, 'agentMeta': {'id': agent_id, **meta}}, ensure_ascii=False)}\n\n"
                for event_type, payload in run_agent_stream_yield_events(agent_id, input_data):
                    if event_type == "step":
                        yield f"data: {_json.dumps({'type': 'step', 'step': payload}, ensure_ascii=False)}\n\n"
                    elif event_type == "done":
                        yield f"data: {_json.dumps({'type': 'done', **payload}, ensure_ascii=False)}\n\n"
                    elif event_type == "error":
                        yield f"data: {_json.dumps({'type': 'error', 'error': payload.get('error', '')}, ensure_ascii=False)}\n\n"
                        return
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {_json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(gen()),
            mimetype="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    out = run_agent_and_collect_steps(agent_id, input_data)
    if out.get("error"):
        return jsonify({"code": 400, "msg": out["error"], "data": out}), 400
    return jsonify({"code": 0, "msg": "ok", "data": out})
