"""
【模块 6／8：权限控制】节点级三档权限：readonly（自由执行）/ confirm（需人工审核才能执行，
复用 graph_hitl.py 的 interrupt() 机制）/ forbidden（直接拒绝，本项目暂无这一档的真实节点，
机制已就绪，留给未来接入真正有副作用的节点，比如"发送邮件""删除文件"）。
完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 「权限控制」一节。

本项目目前 weather/news/chat/insight/think/decide/respond/sentiment/keywords/summary/
aggregate 这些节点全部只读——只调用 LLM 或查询外部只读 API，没有任何写操作/副作用，天然
readonly，不需要挡。唯一有真实副作用的是 HITL 的 process 节点（代表"真正执行一个动作"），
所以它是全项目唯一标为 confirm 的节点，且这一档不是摆设——review 节点的 interrupt() 就是它
的强制执行点，process 不经过人工批准的 review 节点，图结构上根本走不到（见 build_hitl_graph）。
"""

NODE_PERMISSIONS: dict[str, str] = {
    "classify": "readonly", "weather": "readonly", "news": "readonly",
    "chat": "readonly", "insight": "readonly",
    "think": "readonly", "decide": "readonly", "respond": "readonly",
    "sentiment": "readonly", "keywords": "readonly", "summary": "readonly", "aggregate": "readonly",
    "analyze": "readonly",  # HITL：只生成建议文本，不执行任何操作，本身只读
    "process": "confirm",   # HITL：真正执行建议的节点，必须经过人工审核（interrupt）才能到达
}


def check_node_permission(node_id: str) -> str:
    """
    【权限控制】查询某节点的权限档位。未登记的新节点默认按 "confirm"（需要确认）处理而不是
    放行——这是"失败关闭"（fail closed）的安全姿态：新接入一个不认识的、可能有副作用的节点时，
    宁可多一次人工确认，也不能悄悄放行。
    """
    return NODE_PERMISSIONS.get(node_id, "confirm")
