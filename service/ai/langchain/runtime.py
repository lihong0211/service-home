"""
通用图运行时：从编译后的图动态生成前端 schema、执行 stream 并收集步骤，供 loop/parallel/
router/hitl 共用；也是 agent_research.py / agent_wealth_advisor.py / agent.py 等外部
模块直接复用的通用工具（graph_to_schema / run_graph_stream_and_collect）。
GRAPH_BUILDERS 是所有可执行图的注册表，是本文件反过来 import 各 graph_*.py 的原因。
"""

import time

from service.ai.langchain.graph_hitl import _HITL_GRAPH
from service.ai.langchain.graph_loop import build_loop_graph
from service.ai.langchain.graph_parallel import build_parallel_graph
from service.ai.langchain.graph_router import build_router_graph

# 节点图标：由后端返回给前端 graphData.nodes[].icon，前端据此展示；可在此修改
NODE_ICONS = {
    "think": "🤔",
    "decide": "🎯",
    "process": "⚙️",
    "analyze": "🔍",
    "generate": "✨",
    "classify": "🎯",
    "aggregate": "📊",
    "weather": "☀️",
    "news": "📰",
    "chat": "💭",
    "sentiment": "😊",   # 情感分析
    "keywords": "🏷️",   # 关键词
    "summary": "📝",    # 摘要
    "review": "👤",
    "dispatch": "📤",
    "respond": "📢",
    "insight": "🧩",   # 子图组合（router 内嵌 parallel 子图）
}

# 节点 id -> 前端展示（可选覆盖），未列出的用 raw_id、type=process
NODE_DISPLAY = {
    "__start__": {"name": "用户输入", "type": "input", "icon": "📝", "description": "入口"},
    "__end__": {"name": "输出", "type": "output", "icon": "📢", "description": "出口"},
    "classify": {"name": "意图分类", "type": "llm", "description": "分析用户意图"},
    "weather": {"name": "天气", "type": "tool", "description": "天气查询"},
    "news": {"name": "新闻", "type": "tool", "description": "新闻摘要"},
    "chat": {"name": "闲聊", "type": "llm", "description": "通用对话"},
    "insight": {"name": "多维分析（子图）", "type": "tool", "description": "调用 parallel 子图：情感/关键词/摘要"},
    "think": {"name": "思考", "type": "llm", "description": "迭代思考"},
    "decide": {"name": "决策", "type": "condition", "description": "是否继续"},
    "sentiment": {"name": "情感分析", "type": "llm", "description": "情感分析"},
    "keywords": {"name": "关键词", "type": "tool", "description": "关键词提取"},
    "summary": {"name": "摘要", "type": "llm", "description": "文本摘要"},
    "aggregate": {"name": "聚合", "type": "process", "description": "汇总结果"},
    "analyze": {"name": "AI 分析", "type": "llm", "description": "生成建议"},
    "review": {"name": "人工审核", "type": "condition", "description": "人工确认"},
    "process": {"name": "处理反馈", "type": "process", "description": "应用反馈"},
    "respond": {"name": "最终回答", "type": "output", "icon": "📢", "description": "根据思考生成或调用接口返回结果"},
}


def _node_id_for_schema(raw_id: str) -> str:
    """将图内部节点 id 转为前端 schema 的 id（__start__ -> input, __end__ -> output）。"""
    if raw_id == "__start__":
        return "input"
    if raw_id == "__end__":
        return "output"
    return raw_id


def graph_to_schema(compiled_graph, node_display: dict | None = None, node_icons: dict | None = None) -> dict:
    """
    从 LangGraph 编译后的图动态生成前端 GraphData 格式：nodes + edges。
    使用 get_graph() 的 nodes/edges，不手写结构。
    node_display / node_icons 可选，供其他模块（如 agent_research、agent_wealth_advisor）传入自定义展示信息。
    """
    raw = compiled_graph.get_graph()
    display_map = node_display if node_display is not None else NODE_DISPLAY
    icons_map = node_icons if node_icons is not None else NODE_ICONS
    nodes_out = []
    # 节点：raw.nodes 为 dict[id -> Node]
    for raw_id in raw.nodes:
        display = display_map.get(raw_id, {})
        schema_id = _node_id_for_schema(raw_id)
        name = display.get("name") or raw_id
        node_type = display.get("type") or "process"
        icon = display.get("icon") or icons_map.get(raw_id, "🔹")
        desc = display.get("description") or ""
        nodes_out.append({
            "id": schema_id,
            "name": name,
            "type": node_type,
            "icon": icon,
            "description": desc,
        })
    # 边：raw.edges 为 list[Edge(source, target, data, conditional)]
    edges_out = []
    for e in raw.edges:
        src = getattr(e, "source", None) or (e.get("source") if isinstance(e, dict) else None)
        tgt = getattr(e, "target", None) or (e.get("target") if isinstance(e, dict) else None)
        src = _node_id_for_schema(src) if src else src
        tgt = _node_id_for_schema(tgt) if tgt else tgt
        conditional = getattr(e, "conditional", None)
        if conditional is None and isinstance(e, dict):
            conditional = e.get("conditional", False)
        edge_type = "conditional" if conditional else "normal"
        edges_out.append({"source": src, "target": tgt, "type": edge_type})
    return {"nodes": nodes_out, "edges": edges_out}

def _merge_state_update(current: dict, update: dict) -> dict:
    """按 LangGraph 的 reducer 语义合并一次节点输出到当前 state（messages 用 add，其余覆盖）。"""
    if not update or not isinstance(update, dict):
        return current
    out = dict(current)
    for key, value in update.items():
        if key == "messages":
            existing = out.get("messages") or []
            add = value if isinstance(value, list) else [value]
            out["messages"] = existing + add
        elif key == "analyses" and isinstance(value, list):
            existing = out.get("analyses") or []
            out["analyses"] = existing + value
        else:
            out[key] = value
    return out


def run_graph_stream_and_collect(graph, state: dict, config: dict | None = None):
    """
    执行图 stream 一次，收集每一步的 nodeId、耗时、输出，并从各步输出合并出最终 state。
    不再二次 invoke，避免流程跑两遍、最终回答提前打印。
    config：{"configurable": {"thread_id": "..."}}，【上下文管理】/【人工处理】依赖 checkpointer
    做跨请求状态持久化的图（router/hitl）靠它认出"这是同一个会话"；不传则每次都是全新会话
    （等价于旧行为），loop/parallel 没接 checkpointer，传不传都不影响它们。
    返回：{"steps": [...], "finalState": {...}, "executionOrder": [...], "totalSteps": N}。
    前端进度条应用：当前步 = stepIndex+1，总步数 = totalSteps，进度 = (stepIndex+1)/totalSteps*100%。
    勿用 finalState.iteration 当作总步数（iteration 仅表示“思考轮数”，如 loop 里为 3）。
    """
    steps = []
    execution_order = []
    t0 = time.perf_counter()
    step_index = 0
    current_state = dict(state)
    # stream_mode="updates"：并行节点（如 parallel 的 sentiment/keywords/summary）会分别 yield，前端才能逐步展示，不会「从开始直接跳到结束」
    for step in graph.stream(state, config=config, stream_mode="updates"):
        for node_id, output in step.items():
            t1 = time.perf_counter()
            duration_ms = round((t1 - t0) * 1000)
            t0 = t1
            step_payload = {
                "stepIndex": step_index,
                "nodeId": node_id,
                "status": "end",
                "duration_ms": duration_ms,
                "output": output,
            }
            step_payload.update(_enrich_step_for_frontend(node_id, output if isinstance(output, dict) else {}, current_state))
            steps.append(step_payload)
            execution_order.append(node_id)
            current_state = _merge_state_update(current_state, output if isinstance(output, dict) else {})
            step_index += 1
    return {
        "steps": steps,
        "finalState": current_state,
        "executionOrder": execution_order,
        "totalSteps": len(steps),
    }


def _enrich_step_for_frontend(node_id: str, output: dict, current_state: dict) -> dict:
    """
    为前端展示补充 step 的易用字段：loop 图用 iteration/thought/label，parallel 图用 label。
    """
    extra = {}
    # parallel 图：每步给中文 label，便于时间线展示
    if node_id == "dispatch":
        extra["label"] = "分发"
    elif node_id == "sentiment":
        extra["label"] = "情感分析"
    elif node_id == "keywords":
        extra["label"] = "关键词提取"
    elif node_id == "summary":
        extra["label"] = "摘要"
    elif node_id == "aggregate":
        extra["label"] = "聚合"
    # loop 图
    elif node_id == "think" and isinstance(output, dict):
        msgs = output.get("messages")
        extra["iteration"] = output.get("iteration", current_state.get("iteration", 0))
        extra["thought"] = (msgs[-1] if isinstance(msgs, list) and msgs else msgs) or ""
        extra["label"] = f"第{extra['iteration']}轮思考"
    elif node_id == "decide" and isinstance(output, dict):
        next_step = output.get("next_step", "")
        extra["nextStep"] = next_step
        extra["label"] = "继续思考" if next_step == "think" else "进入回答"
    elif node_id == "respond" and isinstance(output, dict):
        extra["response"] = output.get("response", "")
        extra["label"] = "最终回答"
    # router 图：weather/news/chat/insight 节点也带 response，前端可直接从 step 或 finalState 取
    elif node_id in ("weather", "news", "chat", "insight") and isinstance(output, dict):
        extra["response"] = output.get("response", "")
        extra["label"] = {"weather": "天气", "news": "新闻", "chat": "闲聊", "insight": "多维分析（子图）"}.get(node_id, node_id)
    return extra


def run_graph_stream_yield_events(graph, state: dict, config: dict | None = None):
    """
    执行图 stream，每完成一步 yield 一个 step 事件，最后 yield 一个 done 事件。
    供 SSE 流式接口使用：前端先按步更新流程动画，收到 done 后再展示回答，避免「回答比流程快」。
    loop 图每步会带 iteration/thought/nextStep/response/label 等字段，便于前端展示「第 N 轮思考」。
    config 用途见 run_graph_stream_and_collect 的说明（【上下文管理】跨请求会话识别）。

    stream_mode=["updates","custom"]：graph.stream() 因此按 (mode, payload) 元组产出。
    "custom" 来自节点内 get_stream_writer()（目前是 chat 节点的 DashScope 逐 token 转发），
    只有 router 图的 chat 分支会产出 token 事件，其余图/节点没有 writer 调用，不受影响。
    yield: ("step", {...}) | ("token", {"nodeId","content"}) | ("done", {...})。
    """
    steps = []
    execution_order = []
    t0 = time.perf_counter()
    step_index = 0
    current_state = dict(state)
    for mode, payload in graph.stream(state, config=config, stream_mode=["updates", "custom"]):
        if mode == "custom":
            yield ("token", payload)
            continue
        for node_id, output in payload.items():
            t1 = time.perf_counter()
            duration_ms = round((t1 - t0) * 1000)
            t0 = t1
            step_payload = {
                "stepIndex": step_index,
                "nodeId": node_id,
                "status": "end",
                "duration_ms": duration_ms,
                "output": output,
            }
            extra = _enrich_step_for_frontend(node_id, output if isinstance(output, dict) else {}, current_state)
            step_payload.update(extra)
            steps.append(step_payload)
            execution_order.append(node_id)
            current_state = _merge_state_update(current_state, output if isinstance(output, dict) else {})
            step_index += 1
            yield ("step", step_payload)
    yield ("done", {
        "finalState": current_state,
        "totalSteps": len(steps),
        "executionOrder": execution_order,
        "steps": steps,
    })


GRAPH_BUILDERS = {
    "router": build_router_graph,
    "loop": build_loop_graph,
    "parallel": build_parallel_graph,
    # hitl 需要跨请求维持 interrupt 暂停状态，复用模块级单例，而非每次重新 build
    "hitl": lambda: _HITL_GRAPH,
}

DEFAULT_INPUTS = {
    "router": {
        "query": "今天天气怎么样？", "intent": "", "response": "", "user_id": "demo-user",
        # 【上下文管理】messages/context_summary 显式给空默认值（而不是依赖 LangGraph 隐式默认），
        # 避免走 weather/news/insight 等从不碰这两个字段的分支时，channel 因从未被赋值而行为不确定。
        "messages": [], "context_summary": "",
    },
    "loop": {"messages": [], "next_step": "", "iteration": 0, "query": "", "response": ""},
    "parallel": {"input_text": "示例文本", "analyses": [], "final_result": "", "response": ""},
    "hitl": {"query": "", "suggestion": "", "decision": "", "response": ""},
}

def get_graph_schema(name: str) -> dict | None:
    """从编译后的图动态生成前端 GraphData（nodes + edges），不手写结构。"""
    builder_fn = GRAPH_BUILDERS.get(name)
    if not builder_fn:
        return None
    graph = builder_fn()
    schema = graph_to_schema(graph)
    schema["executionOrder"] = []  # 真实顺序由 POST /run 返回
    return schema


def list_graph_names():
    """返回可用的图名称列表。"""
    return list(GRAPH_BUILDERS.keys())
