"""3. 状态管理 - MemorySaver 持久化。"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

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
