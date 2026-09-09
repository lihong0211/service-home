"""入口：运行全部 LangGraph 功能演示。"""

from service.ai.langchain.graph_hitl import demo_hitl
from service.ai.langchain.graph_loop import demo_loop
from service.ai.langchain.graph_parallel import demo_parallel
from service.ai.langchain.graph_router import build_router_graph, demo_memory, demo_router
from service.ai.langchain.graph_state_mgmt import demo_state_management
from service.ai.langchain.visualization import visualize_execution

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

    demo_memory()
    print()

    demo_hitl()
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
    print("| 状态管理   | 长对话、工作流     | ⭐⭐⭐   |")
