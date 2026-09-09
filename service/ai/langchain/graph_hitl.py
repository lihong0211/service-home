"""
5. 人机交互节点 - interrupt() 暂停 → 人工审核/编辑 → 处理反馈
同时也是【模块 7／8：人工处理】的完整实现，以及【模块 6／8：权限控制】confirm 档的
落地机制——两个模块在这里是同一套代码，不是巧合：生产系统里"危险动作要不要执行"这个
权限判断，落到实现上往往就是"暂停下来等人工确认"，见下面 _hitl_review 的 interrupt()。
"""

import os
import sqlite3
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from service.ai.langchain.common import _call_llm
from service.ai.langchain.permission import check_node_permission

class HitlState(TypedDict):
    query: str
    suggestion: str  # AI 生成的建议，供人工审核
    decision: str  # approved / rejected，记录人工决定
    response: str


def _hitl_analyze(state: HitlState) -> dict:
    """AI 根据用户请求生成一条具体行动建议，等待人工审核。"""
    query = (state.get("query") or "").strip()
    suggestion = _call_llm(
        f"用户请求：{query}\n\n请给出你建议采取的具体行动方案，用一句话说清楚要做什么，不超过40字。",
        system="你是一个行动建议助手，只输出具体、可执行的建议本身，不要解释。",
    )
    print(f"✨ AI 建议: {suggestion}")
    return {"suggestion": suggestion or "（未能生成建议，请重新提问）"}


def _hitl_review(state: HitlState) -> Command[Literal["process", "__end__"]]:
    """
    暂停图执行，把 AI 建议交给人工审核。
    resume 传入 True/False 表示批准/拒绝；传入非空字符串表示"批准并采用编辑后的建议"。
    """
    decision = interrupt({
        "question": "是否批准以下 AI 建议？可直接批准/拒绝，或提交编辑后的文本作为最终建议。",
        "suggestion": state.get("suggestion", ""),
    })
    print(f"👤 人工审核结果: {decision!r}")
    if isinstance(decision, str) and decision.strip():
        # 人工编辑过建议内容：视为批准，并采用编辑后的文本
        return Command(goto="process", update={"decision": "approved", "suggestion": decision.strip()})
    if decision:
        return Command(goto="process", update={"decision": "approved"})
    return Command(
        goto=END,
        update={"decision": "rejected", "response": "已拒绝该建议，未执行任何操作。"},
    )


def _hitl_process(state: HitlState) -> dict:
    """
    人工批准（或编辑）后执行，生成最终回复。
    【权限控制】这是全项目唯一 confirm 档节点的真正执行点：断言一下它的权限档位配置没被
    改错——图结构已经保证了只有 review 节点批准（即 interrupt() 收到批准结果）才能路由到
    这里（见 _hitl_review），这行断言是双保险，防止未来有人改动 NODE_PERMISSIONS 却忘了
    同步图结构，导致"标着 confirm 却其实没人审核"这种权限声明和实际行为不一致的情况。
    """
    assert check_node_permission("process") == "confirm", "process 节点权限档位配置有误，应为 confirm"
    suggestion = state.get("suggestion", "")
    response = f"✅ 已按审核通过的建议执行：{suggestion}"
    print(f"⚙️ {response}")
    return {"response": response}


def _get_hitl_checkpointer() -> SqliteSaver:
    """
    HITL 用共享 SQLite 文件做 checkpointer，而不是 MemorySaver。
    原因：生产部署是多 worker 进程（UVICORN_WORKERS>1），每个 worker 是独立进程、
    独立内存；interrupt 命中后的第一次请求和 resume 请求很可能被负载均衡到不同 worker，
    MemorySaver 存在各自进程内存里，跨进程互相看不到，resume 时会因为"找不到暂停状态"
    而从头重新执行整个图，人工审核形同虚设。SQLite 文件所有 worker 共享同一份，可跨进程恢复。
    """
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "checkpoints")
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(db_dir, "hitl.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # 幂等，首次调用建表
    return saver


def build_hitl_graph(checkpointer=None):
    """人机交互图：analyze → review(interrupt) → process | END。需要 checkpointer 才能跨请求暂停/恢复。"""
    builder = StateGraph(HitlState)
    builder.add_node("analyze", _hitl_analyze, retry_policy=RetryPolicy(max_attempts=3))
    # destinations 告诉 get_graph() 这个 Command 节点可能跳去哪些目标，
    # 纯供静态可视化用，实际路由仍由 _hitl_review 运行时返回的 Command(goto=...) 决定。
    builder.add_node("review", _hitl_review, destinations=("process", END))
    builder.add_node("process", _hitl_process)
    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "review")
    builder.add_edge("process", END)
    return builder.compile(checkpointer=checkpointer or _get_hitl_checkpointer())


# HITL 图依赖 checkpointer 维持跨请求（analyze→interrupt / resume→process）的暂停状态，
# 必须是同一个编译后的图对象和同一个 checkpointer 实例，因此在模块级构建一次并复用，
# 而不是像 router/loop/parallel 那样每次请求都重新 build。
_HITL_GRAPH = build_hitl_graph()


def demo_hitl():
    """演示 HITL 流程：第一次调用命中 interrupt 暂停，第二次带 resume 决定继续。"""
    graph = _HITL_GRAPH
    print("📊 **人机交互流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: analyze → review →[interrupt]→ process | END)")
    print()
    config = {"configurable": {"thread_id": "demo-hitl-1"}}
    out = graph.invoke({"query": "帮我优化这段文案", "suggestion": "", "decision": "", "response": ""}, config)
    print("命中 interrupt:", out.get("__interrupt__"))
    resumed = graph.invoke(Command(resume=True), config)
    print("恢复后 response:", resumed.get("response"))
    return graph


def run_hitl_graph(input_state: dict | None, thread_id: str, resume=None) -> dict:
    """
    执行/恢复 HITL 图，供 HTTP 层调用。
    首次调用不传 resume：命中 interrupt 后返回 waitingForInput=True + interrupt payload。
    第二次调用带上同一个 thread_id + resume（人工决定），从暂停点继续执行到 END。
    """
    graph = _HITL_GRAPH
    config = {"configurable": {"thread_id": thread_id}}
    if resume is not None:
        run_input = Command(resume=resume)
    else:
        # 惰性 import：runtime.py 反过来在模块加载期 import 本模块的 _HITL_GRAPH（组装
        # GRAPH_BUILDERS），放到函数体内、延迟到真正调用时才 import，避免两边循环引用。
        from service.ai.langchain.runtime import DEFAULT_INPUTS

        default = DEFAULT_INPUTS.get("hitl", {})
        run_input = {**default, **(input_state or {})}
    result = graph.invoke(run_input, config=config)
    interrupts = result.get("__interrupt__")
    if interrupts:
        first = interrupts[0]
        payload = getattr(first, "value", first)
        return {
            "threadId": thread_id,
            "waitingForInput": True,
            "interrupt": payload,
            "finalState": {k: v for k, v in result.items() if k != "__interrupt__"},
        }
    return {
        "threadId": thread_id,
        "waitingForInput": False,
        "finalState": result,
        "response": result.get("response", ""),
    }

