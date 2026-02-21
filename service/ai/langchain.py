# service/ai/langchain.py
"""
LangGraph 核心功能可视化演示

演示内容：
1. 循环与分支 - 基础功能（think/decide 循环）
2. 并行执行 - 多分支汇聚（情感/关键词/摘要 → 聚合）
3. 状态管理 - MemorySaver 持久化与恢复
4. 条件路由 - 意图识别与多路分发
5. 人机交互节点 - AI 建议 → 人工审核 → 处理反馈
6. 实时执行监控 - stream 可视化与简单仪表盘
"""

from __future__ import annotations

import operator
import time
from datetime import datetime
from typing import Annotated, TypedDict

# LangGraph 图与状态
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# 1. 循环与分支 - 基础功能
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str
    iteration: int


def _think(state: AgentState) -> dict:
    print(f"🤔 思考中... (第{state['iteration']}轮)")
    return {
        "messages": [f"思考轮次：{state['iteration']}"],
        "iteration": state["iteration"] + 1,
    }


def _decide(state: AgentState) -> dict:
    if state["iteration"] < 3:
        print("🔄 需要继续思考，进入循环")
        return {"next_step": "think"}
    print("✅ 思考完成，结束")
    return {"next_step": END}


def build_loop_graph():
    """创建带循环的图：think → decide → (think | END)。"""
    builder = StateGraph(AgentState)
    builder.add_node("think", _think)
    builder.add_node("decide", _decide)
    builder.set_entry_point("think")
    builder.add_edge("think", "decide")
    builder.add_conditional_edges(
        "decide",
        lambda s: s["next_step"],
        {"think": "think", END: END},
    )
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
        {"messages": [], "next_step": "", "iteration": 0}
    )
    print("最终状态 iteration:", out.get("iteration"), "messages 数量:", len(out.get("messages", [])))
    return graph


# ---------------------------------------------------------------------------
# 2. 并行执行 - 多分支汇聚（使用 Send 或顺序模拟）
# ---------------------------------------------------------------------------


class ParallelState(TypedDict):
    input_text: str
    analyses: Annotated[list, operator.add]  # 并行节点用 append 合并
    final_result: str


def _sentiment_analysis(state: ParallelState) -> dict:
    print("🔵 情感分析中...")
    return {"analyses": [("sentiment", "positive")]}


def _keyword_extraction(state: ParallelState) -> dict:
    print("🟢 关键词提取中...")
    return {"analyses": [("keywords", ["AI", "LangGraph"])]}


def _text_summary(state: ParallelState) -> dict:
    print("🟠 文本摘要中...")
    return {"analyses": [("summary", "这是摘要")]}


def _aggregate_results(state: ParallelState) -> dict:
    print("📊 聚合所有分析结果")
    analyses = dict(state["analyses"]) if state.get("analyses") else {}
    return {"final_result": f"综合结果：{analyses}"}


def build_parallel_graph():
    """
    并行执行图：入口分发到 sentiment / keywords / summary，再汇聚到 aggregate。
    若当前环境不支持 Send，则用顺序边模拟（三节点依次执行后到 aggregate）。
    """
    builder = StateGraph(ParallelState)
    builder.add_node("sentiment", _sentiment_analysis)
    builder.add_node("keywords", _keyword_extraction)
    builder.add_node("summary", _text_summary)
    builder.add_node("aggregate", _aggregate_results)

    try:
        from langgraph.types import Send

        def _dispatch(state: ParallelState):
            return [Send("sentiment", state), Send("keywords", state), Send("summary", state)]

        builder.add_node("dispatch", lambda s: s)  # 透传 state
        builder.set_entry_point("dispatch")
        builder.add_conditional_edges("dispatch", _dispatch)
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
    out = graph.invoke({"input_text": "示例文本", "analyses": [], "final_result": ""})
    print("final_result:", out.get("final_result", "")[:80])
    return graph


# ---------------------------------------------------------------------------
# 3. 状态管理 - MemorySaver 持久化
# ---------------------------------------------------------------------------


class ConversationState(TypedDict):
    messages: list
    context: dict
    user_info: dict
    tokens_used: int


def _process_message(state: ConversationState) -> dict:
    new_message = f"处理消息 #{len(state['messages']) + 1}"
    print(f"💬 {new_message}")
    return {
        "messages": state["messages"] + [new_message],
        "tokens_used": state.get("tokens_used", 0) + 10,
    }


def build_state_mgmt_graph():
    """带 checkpoint 的图，用于演示状态恢复。"""
    builder = StateGraph(ConversationState)
    builder.add_node("process", _process_message)
    builder.set_entry_point("process")
    builder.add_edge("process", END)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def demo_state_management():
    """演示状态管理：同一 thread_id 下两次 invoke 会累积 messages。"""
    graph = build_state_mgmt_graph()
    print("📊 **状态管理演示**")
    config = {"configurable": {"thread_id": "demo-thread-1"}}
    initial = {"messages": [], "context": {}, "user_info": {}, "tokens_used": 0}
    out1 = graph.invoke(initial, config)
    print("第一次执行 messages:", out1.get("messages"), "tokens_used:", out1.get("tokens_used"))
    out2 = graph.invoke(initial, config)
    print("第二次执行（带历史）messages:", out2.get("messages"), "tokens_used:", out2.get("tokens_used"))
    return graph


# ---------------------------------------------------------------------------
# 4. 条件路由 - 意图识别与多路分发
# ---------------------------------------------------------------------------


class RouterState(TypedDict):
    query: str
    intent: str
    response: str


def _classify_intent(state: RouterState) -> dict:
    query = (state.get("query") or "").lower()
    if "天气" in query:
        intent = "weather"
    elif "股票" in query:
        intent = "stock"
    elif "新闻" in query:
        intent = "news"
    else:
        intent = "chat"
    print(f"🎯 意图识别: {intent}")
    return {"intent": intent}


def _weather_handler(state: RouterState) -> dict:
    return {"response": "☀️ 今天天气晴朗，25度"}


def _stock_handler(state: RouterState) -> dict:
    return {"response": "📈 股市上涨0.5%"}


def _news_handler(state: RouterState) -> dict:
    return {"response": "📰 今日头条：AI新突破"}


def _chat_handler(state: RouterState) -> dict:
    return {"response": "💭 你好，我是AI助手"}


def build_router_graph():
    """条件路由：classify → weather | stock | news | chat → END。"""
    builder = StateGraph(RouterState)
    builder.add_node("classify", _classify_intent)
    builder.add_node("weather", _weather_handler)
    builder.add_node("stock", _stock_handler)
    builder.add_node("news", _news_handler)
    builder.add_node("chat", _chat_handler)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        lambda s: s["intent"],
        {"weather": "weather", "stock": "stock", "news": "news", "chat": "chat"},
    )
    for name in ["weather", "stock", "news", "chat"]:
        builder.add_edge(name, END)

    return builder.compile()


def demo_router():
    """演示条件路由并打印 ASCII 图。"""
    graph = build_router_graph()
    print("📊 **智能路由流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: classify → weather|stock|news|chat → END)")
    print()
    for q in ["今天天气怎么样？", "有什么新闻？", "随便聊聊"]:
        out = graph.invoke({"query": q, "intent": "", "response": ""})
        print(f"  query={q!r} → response={out.get('response', '')}")
    return graph


# ---------------------------------------------------------------------------
# 5. 人机交互节点（人工审核用 mock，避免阻塞服务）
# ---------------------------------------------------------------------------


class HumanInLoopState(TypedDict):
    task: str
    ai_suggestion: str
    human_feedback: str
    final_output: str


def _ai_analyze(state: HumanInLoopState) -> dict:
    print("🤖 AI分析中...")
    time.sleep(0.2)
    suggestion = f"建议方案：处理 {state.get('task', '')}"
    print(f"💡 AI建议：{suggestion}")
    return {"ai_suggestion": suggestion}


def _human_review(state: HumanInLoopState) -> dict:
    """模拟人工审核；生产环境可改为 interrupt + 外部输入。"""
    print("\n👤 === 等待人工审核（此处用 mock）===")
    print(f"AI建议：{state.get('ai_suggestion', '')}")
    feedback = "approve"  # 可改为从请求/队列读取
    return {"human_feedback": feedback}


def _process_feedback(state: HumanInLoopState) -> dict:
    if state.get("human_feedback") == "approve":
        return {"final_output": state.get("ai_suggestion", "")}
    return {"final_output": "已根据人工反馈修改"}


def build_human_loop_graph():
    """人机协作：analyze → review → process → END。"""
    builder = StateGraph(HumanInLoopState)
    builder.add_node("analyze", _ai_analyze)
    builder.add_node("review", _human_review)
    builder.add_node("process", _process_feedback)
    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "review")
    builder.add_edge("review", "process")
    builder.add_edge("process", END)
    return builder.compile()


def demo_human_loop():
    """演示人机协作流程图。"""
    graph = build_human_loop_graph()
    print("📊 **人机协作流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: analyze → review → process → END)")
    out = graph.invoke({"task": "审核工单", "ai_suggestion": "", "human_feedback": "", "final_output": ""})
    print("final_output:", out.get("final_output"))
    return graph


# ---------------------------------------------------------------------------
# 6. 实时执行监控 - stream 可视化
# ---------------------------------------------------------------------------

NODE_ICONS = {
    "think": "🤔",
    "decide": "🎯",
    "process": "⚙️",
    "analyze": "🔍",
    "generate": "✨",
    "classify": "🎯",
    "aggregate": "📊",
    "weather": "☀️",
    "stock": "📈",
    "news": "📰",
    "chat": "💭",
    "sentiment": "🔵",
    "keywords": "🟢",
    "summary": "🟠",
    "review": "👤",
    "dispatch": "📤",
}

# 节点 id -> 前端展示（可选覆盖），未列出的用 raw_id、type=process
NODE_DISPLAY = {
    "__start__": {"name": "用户输入", "type": "input", "icon": "📝", "description": "入口"},
    "__end__": {"name": "输出", "type": "output", "icon": "📢", "description": "出口"},
    "classify": {"name": "意图分类", "type": "llm", "description": "分析用户意图"},
    "weather": {"name": "天气", "type": "tool", "description": "天气查询"},
    "stock": {"name": "股票", "type": "tool", "description": "股票信息"},
    "news": {"name": "新闻", "type": "tool", "description": "新闻摘要"},
    "chat": {"name": "闲聊", "type": "llm", "description": "通用对话"},
    "think": {"name": "思考", "type": "llm", "description": "迭代思考"},
    "decide": {"name": "决策", "type": "condition", "description": "是否继续"},
    "sentiment": {"name": "情感分析", "type": "llm", "description": "情感分析"},
    "keywords": {"name": "关键词", "type": "tool", "description": "关键词提取"},
    "summary": {"name": "摘要", "type": "llm", "description": "文本摘要"},
    "aggregate": {"name": "聚合", "type": "process", "description": "汇总结果"},
    "analyze": {"name": "AI 分析", "type": "llm", "description": "生成建议"},
    "review": {"name": "人工审核", "type": "condition", "description": "人工确认"},
    "process": {"name": "处理反馈", "type": "process", "description": "应用反馈"},
}


def visualize_execution(graph, inputs: dict, sleep_sec: float = 0.3):
    """按 stream 步进打印每个节点的执行与状态更新。"""
    print("🎬 **执行开始**")
    print("=" * 50)
    for step in graph.stream(inputs):
        for node_name, node_output in step.items():
            ts = datetime.now().strftime("%H:%M:%S")
            icon = NODE_ICONS.get(node_name, "🔹")
            print(f"[{ts}] {icon} 节点: {node_name}")
            print(f"   📦 状态更新: {node_output}")
            print("-" * 30)
            if sleep_sec:
                time.sleep(sleep_sec)
    print("=" * 50)
    print("✅ **执行完成**")


def get_node_color(status: str) -> str:
    """按状态返回终端颜色码（可选，用于高级可视化）。"""
    colors = {"active": "\033[92m", "completed": "\033[94m", "error": "\033[91m", "waiting": "\033[93m"}
    return colors.get(status, "\033[0m")


class LangGraphDashboard:
    """简单内存仪表盘：记录执行路径与节点状态。"""

    def __init__(self):
        self.nodes_status: dict = {}
        self.execution_path: list = []

    def update(self, node_name: str, status: str, data=None):
        self.nodes_status[node_name] = {
            "status": status,
            "timestamp": datetime.now(),
            "data": data,
        }
        self.execution_path.append(node_name)

    def render(self, clear: bool = False):
        if clear:
            print("\033c", end="")
        print("╔════════════════════════════════╗")
        print("║   LangGraph 实时执行仪表盘     ║")
        print("╚════════════════════════════════╝")
        print("\n📈 执行路径:", " → ".join(self.execution_path))
        print("\n📊 节点状态:")
        for node, info in self.nodes_status.items():
            icon = "✅" if info["status"] == "completed" else "⏳"
            print(f"  {icon} {node}: {info['status']}")
        print()


# ---------------------------------------------------------------------------
# 前端 3D 可视化对接：从编译后的图动态生成 schema（供 React+Three.js 使用）
# ---------------------------------------------------------------------------

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


def run_graph_stream_and_collect(graph, state: dict):
    """
    执行图 stream，收集每一步的 nodeId、耗时、输出，供前端按真实执行顺序驱动可视化。
    返回：{"steps": [...], "finalState": {...}, "executionOrder": [...]}
    """
    steps = []
    execution_order = []
    t0 = time.perf_counter()
    for step in graph.stream(state):
        for node_id, output in step.items():
            t1 = time.perf_counter()
            duration_ms = round((t1 - t0) * 1000)
            t0 = t1
            steps.append({
                "nodeId": node_id,
                "status": "end",
                "duration_ms": duration_ms,
                "output": output,
            })
            execution_order.append(node_id)
    final_state = graph.invoke(state)
    return {
        "steps": steps,
        "finalState": final_state,
        "executionOrder": execution_order,
    }


GRAPH_BUILDERS = {
    "router": build_router_graph,
    "loop": build_loop_graph,
    "parallel": build_parallel_graph,
    "human_loop": build_human_loop_graph,
}

DEFAULT_INPUTS = {
    "router": {"query": "今天天气怎么样？", "intent": "", "response": ""},
    "loop": {"messages": [], "next_step": "", "iteration": 0},
    "parallel": {"input_text": "示例文本", "analyses": [], "final_result": ""},
    "human_loop": {"task": "审核工单", "ai_suggestion": "", "human_feedback": "", "final_output": ""},
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


def run_graph_and_collect_steps(graph_name: str, input_state: dict | None = None):
    """
    执行指定图，收集每一步的 nodeId、耗时、输出，供前端按真实执行顺序与节奏驱动 3D 动画。
    返回：{
        "graphData": { nodes, edges, executionOrder },
        "steps": [ { "nodeId", "status": "end", "duration_ms", "output" }, ... ],
        "finalState": { ... },
        "executionOrder": [ "classify", "weather", ... ]
    }
    """
    builder_fn = GRAPH_BUILDERS.get(graph_name)
    if not builder_fn:
        return {"error": f"未知图: {graph_name}", "allowed": list(GRAPH_BUILDERS.keys())}
    graph = builder_fn()
    state = input_state if input_state is not None else DEFAULT_INPUTS.get(graph_name, {})
    try:
        run_result = run_graph_stream_and_collect(graph, state)
    except Exception as e:
        return {"error": str(e)}
    schema = graph_to_schema(graph)
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


# ---------------------------------------------------------------------------
# 入口：运行全部演示
# ---------------------------------------------------------------------------


def run_all_demos():
    """依次运行所有 LangGraph 功能演示。"""
    print("\n" + "=" * 60)
    print("  LangGraph 核心功能可视化演示")
    print("=" * 60 + "\n")

    demo_loop()
    print()

    demo_parallel()
    print()

    demo_state_management()
    print()

    demo_router()
    print()

    demo_human_loop()
    print()

    # 用路由图做一次 stream 可视化
    router_graph = build_router_graph()
    print("📊 **实时执行监控示例（条件路由）**")
    visualize_execution(router_graph, {"query": "今天天气怎么样？", "intent": "", "response": ""}, sleep_sec=0.2)
    print()

    print("📊 **功能对比表**")
    print("| 功能       | 适用场景           | 复杂度 |")
    print("|------------|--------------------|--------|")
    print("| 循环       | 迭代优化、多轮对话 | ⭐⭐    |")
    print("| 并行       | 批量处理、多任务   | ⭐⭐⭐   |")
    print("| 条件路由   | 智能客服、分类器   | ⭐⭐    |")
    print("| 人机交互   | 审核流程、人工介入 | ⭐     |")
    print("| 状态管理   | 长对话、工作流     | ⭐⭐⭐   |")


if __name__ == "__main__":
    run_all_demos()
