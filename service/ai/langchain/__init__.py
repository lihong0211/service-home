"""
LangGraph 核心功能可视化演示（包）。

对外公开的名字保持与拆分前的 service/ai/langchain.py 一致，供以下外部调用方直接
`from service.ai.langchain import xxx` 使用，无需改动：
  - service/ai/showcase/agents/agent.py, agent_research.py, agent_wealth_advisor.py:
      graph_to_schema, run_graph_stream_and_collect
  - service/ai/review.py: run_hitl_graph
  - routes/ai.py: langgraph_graph_api, langgraph_run_api, trace_feedback_api,
      trace_bad_cases_api, trace_list_api, trace_copy_api, saga_demo_api
"""

from service.ai.langchain.runtime import graph_to_schema, run_graph_stream_and_collect
from service.ai.langchain.graph_hitl import run_hitl_graph
from service.ai.langchain.saga import saga_demo_api
from service.ai.langchain.views import (
    langgraph_graph_api,
    langgraph_run_api,
    trace_bad_cases_api,
    trace_copy_api,
    trace_feedback_api,
    trace_list_api,
)

__all__ = [
    "graph_to_schema",
    "run_graph_stream_and_collect",
    "run_hitl_graph",
    "saga_demo_api",
    "langgraph_graph_api",
    "langgraph_run_api",
    "trace_bad_cases_api",
    "trace_copy_api",
    "trace_feedback_api",
    "trace_list_api",
]
