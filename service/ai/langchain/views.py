"""HTTP 视图：供 routes/ai.py 注册 /ai/langgraph/* 端点（图运行、回流机制三个端点）。"""

import json
import time

import anyio.from_thread
from fastapi import Request
from fastapi.responses import StreamingResponse

from utils.http_body import query_dict, read_json_optional

from service.ai.langchain.runtime import (
    DEFAULT_INPUTS,
    GRAPH_BUILDERS,
    get_graph_schema,
    graph_to_schema,
    list_graph_names,
    run_graph_stream_and_collect,
    run_graph_stream_yield_events,
)
from service.ai.langchain.trace import (
    _persist_trace,
    list_bad_cases,
    list_traces,
    mark_trace_copied,
    submit_trace_feedback,
)
from service.ai.langchain.graph_hitl import run_hitl_graph

def trace_feedback_api(request: Request):
    """POST /ai/langgraph/trace/feedback  body: {traceId, rating: good|bad, note?}"""
    body = anyio.from_thread.run(read_json_optional, request) or {}
    trace_id = body.get("traceId") or body.get("trace_id")
    rating = body.get("rating")
    note = body.get("note", "")
    if not trace_id or rating not in ("good", "bad"):
        return ({"code": 400, "msg": "缺少参数: traceId / rating(good|bad)", "data": None}, 400)
    try:
        ok = submit_trace_feedback(int(trace_id), rating, note)
    except Exception as e:
        return ({"code": 400, "msg": str(e), "data": None}, 400)
    if not ok:
        return ({"code": 404, "msg": f"未找到 trace: {trace_id}", "data": None}, 404)
    return {"code": 0, "msg": "ok", "data": {"traceId": trace_id, "rating": rating}}


def trace_copy_api(request: Request):
    """POST /ai/langgraph/trace/copy  body: {traceId}  隐式反馈：用户复制了这次回答"""
    body = anyio.from_thread.run(read_json_optional, request) or {}
    trace_id = body.get("traceId") or body.get("trace_id")
    if not trace_id:
        return ({"code": 400, "msg": "缺少参数: traceId", "data": None}, 400)
    ok = mark_trace_copied(int(trace_id))
    if not ok:
        return ({"code": 404, "msg": f"未找到 trace: {trace_id}", "data": None}, 404)
    return {"code": 0, "msg": "ok", "data": {"traceId": trace_id}}


def trace_list_api(request: Request):
    """
    GET /ai/langgraph/trace/list?graph=router&status=success&limit=50
    【可观测性】全量 trace 列表，不限 status/feedback，支撑排查慢请求/看整体执行情况。
    """
    q = query_dict(request)
    graph_name = q.get("graph")
    status = q.get("status")
    limit = int(q.get("limit") or 50)
    traces = list_traces(graph_name, status, limit)
    return {"code": 0, "msg": "ok", "data": {"traces": traces, "total": len(traces)}}


def trace_bad_cases_api(request: Request):
    """GET /ai/langgraph/trace/bad-cases?graph=router&limit=50  返回 bad case 候选池，供人工复核/回流。"""
    q = query_dict(request)
    graph_name = q.get("graph")
    limit = int(q.get("limit") or 50)
    cases = list_bad_cases(graph_name, limit)
    return {"code": 0, "msg": "ok", "data": {"cases": cases, "total": len(cases)}}


def run_graph_and_collect_steps(graph_name: str, input_state: dict | None = None, thread_id: str | None = None):
    """
    执行指定图，收集每一步的 nodeId、耗时、输出，供前端按真实执行顺序与节奏驱动 3D 动画。
    返回：{
        "graphData": { nodes, edges, executionOrder },
        "steps": [ { "nodeId", "status": "end", "duration_ms", "output" }, ... ],
        "finalState": { ... },
        "executionOrder": [ "classify", "weather", ... ]
    }
    前端传入的 input 会与当前图的默认 state 合并，避免切图后残留字段导致缺键报错（如 loop 下误传 query 等）。
    【上下文管理】生产用法：router 图传 thread_id 即可让服务端跨请求持久化对话（见
    build_router_graph 的 checkpointer），不需要再传 history；thread_id 缺省或图未接
    checkpointer（loop/parallel）时退回旧行为——input 里带 history 仍然兼容。
    """
    builder_fn = GRAPH_BUILDERS.get(graph_name)
    if not builder_fn:
        return {"error": f"未知图: {graph_name}", "allowed": list(GRAPH_BUILDERS.keys())}
    graph = builder_fn()
    default = DEFAULT_INPUTS.get(graph_name, {})
    if input_state:
        state = {**default, **input_state}
    else:
        state = default.copy()
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    trace_state = {**state, "thread_id": thread_id} if thread_id else state  # 只为落 trace 用，不进图执行
    t_start = time.perf_counter()
    try:
        run_result = run_graph_stream_and_collect(graph, state, config)
    except Exception as e:
        _persist_trace(graph_name, trace_state, None, round((time.perf_counter() - t_start) * 1000), error=str(e))
        return {"error": str(e)}
    trace_id = _persist_trace(graph_name, trace_state, run_result, round((time.perf_counter() - t_start) * 1000))
    schema = graph_to_schema(graph)
    steps = run_result["steps"]
    execution_order = run_result["executionOrder"]
    total_steps = run_result.get("totalSteps", len(steps))
    nodes = schema["nodes"]
    total_nodes = len(nodes)
    # 执行监控用：总节点数、已完成步数、进度百分比；对话历史用 finalState.response
    completed_steps = len(steps)
    execution_progress = round((completed_steps / total_nodes * 100), 1) if total_nodes else 0
    return {
        "graphData": {
            "nodes": nodes,
            "edges": schema["edges"],
            "executionOrder": execution_order,
        },
        "steps": steps,
        "finalState": run_result["finalState"],
        "executionOrder": execution_order,
        "totalSteps": total_steps,
        "totalNodes": total_nodes,
        "completedSteps": completed_steps,
        "executionProgress": execution_progress,
        "response": run_result["finalState"].get("response", ""),
        # 【回流机制】前端要靠这个 id 调 /ai/langgraph/trace/feedback 给这次回答打反馈
        "traceId": trace_id,
    }


# ---------------------------------------------------------------------------
# HTTP 视图：供 routes/ai 注册 GET/POST
# ---------------------------------------------------------------------------


def langgraph_graph_api(request: Request):
    """GET /ai/langgraph/graph?name=router 返回图结构，供前端 3D 可视化（GraphData）。"""
    q = query_dict(request)
    name = q.get("name") or "router"
    schema = get_graph_schema(name)
    if schema is None:
        return (
            {
                "code": 400,
                "msg": f"未知图: {name}",
                "data": {"allowed": list_graph_names()},
            },
            400,
        )
    return {"code": 0, "msg": "ok", "data": schema}


def langgraph_run_api(request: Request):
    """
    POST /ai/langgraph/run 执行图并返回步骤与最终状态，供前端按真实执行顺序驱动 3D 动画。

    非流式（默认）：响应体为 JSON，结构为：
      { "code": 0, "msg": "ok", "data": {
          "graphData": { "nodes", "edges", "executionOrder" },
          "steps": [ { "stepIndex", "nodeId", "output", "response?", "label?" }, ... ],
          "finalState": { "query", "intent", "response", ... },
          "totalNodes": 7, "completedSteps": 2, "executionProgress": 28.6,
          "response": "最终回复正文（与 finalState.response 一致，便于直接展示对话）"
        }}
    前端「执行监控」建议：总节点 = data.totalNodes，已完成 = data.completedSteps，执行进度 = data.executionProgress%；
    对话历史：取 data.response 或 data.finalState.response 展示。

    流式（body.stream=true）：SSE，先 type=init（含 graphData、totalNodes），中间穿插 type=step 与 type=token
    （router 图 chat 节点的 LLM 逐 token 增量，{nodeId:"chat", content:"..."}，可用于打字机效果），
    最后 type=done（含 finalState、steps、totalNodes、completedSteps、executionProgress、response）；
    前端按 step 播动画、按 token 拼字，收到 done 后以 response 为准做最终展示。
    """
    body = anyio.from_thread.run(read_json_optional, request) or {}
    graph_name = body.get("graph") or "router"
    stream = body.get("stream", False)
    input_state = body.get("input")
    if input_state is not None and not isinstance(input_state, dict):
        input_state = None
    if input_state is None:
        input_state = {}
    top_query = body.get("query")
    if top_query and (not input_state.get("query")):
        input_state = {**input_state, "query": top_query}
    if graph_name == "parallel" and (top_query or input_state.get("query")) and not input_state.get("input_text"):
        input_state = {**input_state, "input_text": (top_query or input_state.get("query", "")).strip() or "示例文本"}

    # 【上下文管理】生产用法：router 图传 threadId，服务端就用 checkpointer 跨请求记住对话（见
    # build_router_graph）；不传则每次都是全新会话（等价于旧行为，仍兼容 input.history 直传）。
    # loop/parallel 没接 checkpointer，传了也不影响它们，无需按图名特判。
    thread_id = body.get("threadId") or body.get("thread_id")

    if graph_name == "hitl":
        # hitl 走独立的暂停/恢复流程（interrupt），不复用 router/loop/parallel 的无状态 stream 收集逻辑。
        # 首次请求：body 传 {graph:"hitl", threadId, query}；命中 interrupt 后返回 waitingForInput=True + interrupt。
        # 第二次请求：body 传 {graph:"hitl", threadId（同一个）, resume: true/false/"编辑后的文本"}。
        thread_id = body.get("threadId") or body.get("thread_id") or "hitl-default"
        resume_value = body.get("resume")
        try:
            out = run_hitl_graph(input_state, thread_id, resume=resume_value)
        except Exception as e:
            return ({"code": 400, "msg": str(e), "data": {}}, 400)
        if resume_value is None and out.get("waitingForInput"):
            # 首次调用命中 interrupt：落一条待审核记录到集中的"人工审核"列表页，同 text2sql
            # 的 text2sql_hitl_api 是同一套接入方式；resume_value is None 表示这是首次调用，
            # 不是 resume 请求，避免 resume 时重复创建。
            from service.ai.review import create_review_task
            interrupt = out.get("interrupt") or {}
            create_review_task(
                source="hitl",
                thread_id=thread_id,
                question=interrupt.get("question", ""),
                content=interrupt.get("suggestion", ""),
            )
        return {"code": 0, "msg": "ok", "data": out}

    builder_fn = GRAPH_BUILDERS.get(graph_name)
    if not builder_fn:
        return (
            {"code": 400, "msg": f"未知图: {graph_name}", "data": {"allowed": list(GRAPH_BUILDERS.keys())}},
            400,
        )
    graph = builder_fn()
    default = DEFAULT_INPUTS.get(graph_name, {})
    state = {**default, **input_state} if input_state else default.copy()
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    trace_state = {**state, "thread_id": thread_id} if thread_id else state  # 只为落 trace 用

    if stream:
        schema = graph_to_schema(graph)
        total_nodes = len(schema["nodes"])
        def gen():
            t_start = time.perf_counter()
            try:
                # 先发 graphData，方便前端画图
                yield f"data: {json.dumps({'type': 'init', 'graphData': {'nodes': schema['nodes'], 'edges': schema['edges']}, 'totalNodes': total_nodes}, ensure_ascii=False)}\n\n"
                for event_type, payload in run_graph_stream_yield_events(graph, state, config):
                    if event_type == "step":
                        yield f"data: {json.dumps({'type': 'step', 'step': payload}, ensure_ascii=False)}\n\n"
                    elif event_type == "token":
                        # LLM 逐 token 增量（目前仅 chat 节点会产出），前端可用来做打字机效果；
                        # 该节点完成后仍会有一次 step 事件带上拼接好的完整 output，token 只是过程量。
                        yield f"data: {json.dumps({'type': 'token', **payload}, ensure_ascii=False)}\n\n"
                    else:
                        # done：补充执行监控与对话用字段，便于前端显示进度和 finalState.response
                        trace_id = _persist_trace(graph_name, trace_state, payload, round((time.perf_counter() - t_start) * 1000))
                        steps_list = payload.get("steps", [])
                        completed = len(steps_list)
                        progress = round((completed / total_nodes * 100), 1) if total_nodes else 0
                        done_data = {
                            **payload,
                            "totalNodes": total_nodes,
                            "completedSteps": completed,
                            "executionProgress": progress,
                            "response": (payload.get("finalState") or {}).get("response", ""),
                            "traceId": trace_id,
                        }
                        yield f"data: {json.dumps({'type': 'done', **done_data}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                _persist_trace(graph_name, trace_state, None, round((time.perf_counter() - t_start) * 1000), error=str(e))
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        return StreamingResponse(
            gen(),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    out = run_graph_and_collect_steps(graph_name, input_state, thread_id)
    if out.get("error"):
        return ({"code": 400, "msg": out["error"], "data": out}, 400)
    return {"code": 0, "msg": "ok", "data": out}
