"""
【人工审核】集中审核列表接口：所有走 interrupt() 暂停的业务（text2sql 写操作、HITL 演示）
命中 interrupt 时都在这里落一条 AgentReviewTask 记录（见 create_review_task），前端"人工审核"
页面查询 /ai/review/list 展示，通过/拒绝按钮打 /ai/review/{id}/approve|reject，按 source
分发去调用各自的 run_text2sql_graph / run_hitl_graph 完成 resume——原来分散在 Text2SQL.tsx、
AgentHitl.tsx 里各自的"需要人工审核"卡片改成只在这一个列表页操作。
"""
import json
from datetime import datetime

import anyio
from fastapi import Request

from app.database import db
from model.ai.agent_review_task import AgentReviewTask
from utils.http_body import read_json_optional


def create_review_task(source: str, thread_id: str, question: str, content: str, warnings: list | None = None) -> int:
    """interrupt() 命中时调用，落一条待审核记录。source: text2sql / hitl。"""
    task = AgentReviewTask(
        source=source,
        thread_id=thread_id,
        question=question,
        content=content,
        warnings=json.dumps(warnings, ensure_ascii=False) if warnings else None,
        status="pending",
    )
    db.session.add(task)
    db.session.commit()
    return task.id


def _serialize(task: AgentReviewTask) -> dict:
    return {
        "id": task.id,
        "source": task.source,
        "threadId": task.thread_id,
        "question": task.question,
        "content": task.content,
        "warnings": json.loads(task.warnings) if task.warnings else [],
        "status": task.status,
        "result": json.loads(task.result) if task.result else None,
        "createdAt": task.create_at.isoformat() if task.create_at else None,
        "resolvedAt": task.resolved_at.isoformat() if task.resolved_at else None,
    }


def list_review_tasks_api(request: Request):
    """GET /ai/review/list?status=pending&limit=50"""
    status = request.query_params.get("status")
    limit = request.query_params.get("limit", "50")
    try:
        limit = max(1, min(200, int(limit)))
    except (TypeError, ValueError):
        limit = 50
    q = AgentReviewTask.query.order_by(AgentReviewTask.id.desc())
    if status:
        q = q.filter(AgentReviewTask.status == status)
    tasks = q.limit(limit).all()
    return {"code": 0, "msg": "ok", "data": {"tasks": [_serialize(t) for t in tasks], "total": len(tasks)}}


def _resume_by_source(task: AgentReviewTask, resume_value) -> dict:
    # 延迟 import：text2sql.py / langchain.py 都会 import service/ai/review.py 来落审核记录，
    # 模块级直接 import 会互相循环 import，只能在真正 resume 的时候才 import。
    if task.source == "text2sql":
        from service.ai.text2sql import run_text2sql_graph
        return run_text2sql_graph(None, task.thread_id, resume=resume_value)
    if task.source == "hitl":
        from service.ai.langchain import run_hitl_graph
        return run_hitl_graph(None, task.thread_id, resume=resume_value)
    raise ValueError(f"未知审核来源: {task.source}")


def approve_review_task_api(request: Request, task_id: int):
    """POST /ai/review/{task_id}/approve  body: { content?: 编辑后的内容，不传则原样批准 }"""
    data = anyio.from_thread.run(read_json_optional, request) or {}
    edited = (data.get("content") or "").strip()
    task = AgentReviewTask.query.get(task_id)
    if not task:
        return ({"code": 404, "msg": "审核记录不存在", "data": None}, 404)
    if task.status != "pending":
        return ({"code": 400, "msg": f"该记录已是 {task.status} 状态，不能重复审批", "data": None}, 400)
    resume_value = edited if edited else True
    try:
        out = _resume_by_source(task, resume_value)
    except Exception as e:
        return ({"code": 400, "msg": str(e), "data": None}, 400)
    task.status = "approved"
    task.resolved_at = datetime.now()
    task.result = json.dumps(out, ensure_ascii=False, default=str)
    db.session.commit()
    return {"code": 0, "msg": "ok", "data": _serialize(task)}


def reject_review_task_api(request: Request, task_id: int):
    """POST /ai/review/{task_id}/reject"""
    task = AgentReviewTask.query.get(task_id)
    if not task:
        return ({"code": 404, "msg": "审核记录不存在", "data": None}, 404)
    if task.status != "pending":
        return ({"code": 400, "msg": f"该记录已是 {task.status} 状态，不能重复审批", "data": None}, 400)
    try:
        out = _resume_by_source(task, False)
    except Exception as e:
        return ({"code": 400, "msg": str(e), "data": None}, 400)
    task.status = "rejected"
    task.resolved_at = datetime.now()
    task.result = json.dumps(out, ensure_ascii=False, default=str)
    db.session.commit()
    return {"code": 0, "msg": "ok", "data": _serialize(task)}
