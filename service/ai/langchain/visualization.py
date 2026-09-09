"""6. 实时执行监控 - stream 可视化与简单仪表盘（终端 demo 工具，供 demos.py 使用）。"""

import time
from datetime import datetime

from service.ai.langchain.runtime import NODE_ICONS

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

