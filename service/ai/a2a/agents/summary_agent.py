"""
SummaryAgent — 标准 A2A 实现
输入：Message.parts 中包含 DataPart（document artifact 的 data）
输出：Task.artifacts[0] 包含 DataPart（summary 数据）
接口：
  GET  /.well-known/agent.json
  POST /tasks/send
  GET  /tasks/{task_id}
  POST /tasks/sendSubscribe
"""
from __future__ import annotations

import json
import re
import uuid
import sys
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import (
    A2AArtifact, AgentCapabilities, AgentCard, AgentSkill,
    DataPart, Message, SendMessageRequest, Task, TaskStatus,
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TextPart,
)

app = FastAPI(title="SummaryAgent")

_BASE_URL = "http://127.0.0.1:8003"

AGENT_CARD = AgentCard(
    name="SummaryAgent",
    description="根据文章正文生成摘要",
    url=_BASE_URL,
    version="1.0",
    skills=[
        AgentSkill(
            id="generate_summary",
            name="生成摘要",
            description="输入正文 DataPart，输出摘要文本与要点（JSON）",
            inputModes=["data"],
            outputModes=["data"],
        )
    ],
    capabilities=AgentCapabilities(streaming=True, stateTransitionHistory=True),
    authentication={"schemes": ["apiKey"]},
)

_tasks: dict[str, Task] = {}


def _extract_document(message: Message) -> dict | None:
    """从 Message.parts 中提取 document DataPart 数据"""
    for part in message.parts:
        if hasattr(part, "type") and part.type == "data":
            data = part.data or {}
            if "paragraphs" in data or "title" in data:
                return data
    return None


_OLLAMA_URL = "http://localhost:11434"
_LLM_MODEL = "my-deepseek-r1-1.5"
_THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _call_llm(prompt: str) -> str:
    body = {
        "model": _LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.5, "num_predict": 1024},
    }
    resp = requests.post(f"{_OLLAMA_URL}/api/chat", json=body, timeout=180)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    return _THINKING_RE.sub("", content).strip()


def _generate_summary(doc: dict) -> dict:
    """调用 LLM 根据正文生成摘要与核心要点"""
    title = doc.get("title", "未命名")
    # 截取正文（避免超过 token 限制）
    full_text = "\n\n".join(
        f"[{p['heading']}]\n{p['text']}"
        for p in doc.get("paragraphs", [])
        if p.get("heading") or p.get("text")
    )[:3000]

    # 生成摘要段落
    summary_prompt = (
        f"请为以下文章「{title}」写一段 100~200 字的摘要，概括文章主旨与核心内容。\n\n"
        f"{full_text}\n\n"
        "直接输出摘要段落，不要标题，不要额外说明。"
    )
    # 生成核心要点
    points_prompt = (
        f"请从以下文章「{title}」中提炼 4~6 条核心要点，每条一行，以短横线开头。\n\n"
        f"{full_text}\n\n"
        "直接输出要点列表，不要额外说明。"
    )

    try:
        summary_text = _call_llm(summary_prompt)
    except Exception:
        summary_text = f"本文围绕《{title}》展开，深入探讨了相关核心内容。"

    try:
        points_raw = _call_llm(points_prompt)
        key_points = [
            re.sub(r"^[-•*·\d\.\s]+", "", line).strip()
            for line in points_raw.splitlines()
            if line.strip() and re.match(r"^[-•*·\d]", line.strip())
        ][:6]
        if not key_points:
            key_points = [p.get("heading", "") for p in doc.get("paragraphs", []) if p.get("heading")][:5]
    except Exception:
        key_points = [p.get("heading", "") for p in doc.get("paragraphs", []) if p.get("heading")][:5]

    return {
        "title": title,
        "summary": summary_text,
        "key_points": key_points,
    }


def _process(message: Message) -> Task:
    task_id = message.taskId or str(uuid.uuid4())
    ctx_id = message.contextId or str(uuid.uuid4())
    message.taskId = task_id
    message.contextId = ctx_id

    task = Task(id=task_id, contextId=ctx_id,
                status=TaskStatus(state="submitted"), history=[message])
    _tasks[task_id] = task

    doc = _extract_document(message)
    if doc is None:
        task.status = TaskStatus(state="failed")
        _tasks[task_id] = task
        return task

    task.status = TaskStatus(state="working")
    _tasks[task_id] = task

    summary = _generate_summary(doc)
    artifact = A2AArtifact(
        name="summary",
        description="文章摘要",
        parts=[DataPart(data=summary)],
        lastChunk=True,
    )
    agent_msg = Message(
        role="agent",
        parts=[TextPart(text=summary["summary"])],
        taskId=task_id, contextId=ctx_id,
    )
    task.status = TaskStatus(state="completed", message=agent_msg)
    task.artifacts = [artifact]
    task.history = [message, agent_msg]
    _tasks[task_id] = task
    return task


@app.get("/.well-known/agent.json")
async def get_agent_card():
    return AGENT_CARD.model_dump()


@app.post("/tasks/send")
async def tasks_send(req: SendMessageRequest):
    return _process(req.message).model_dump()


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.model_dump()


@app.post("/tasks/sendSubscribe")
async def tasks_send_subscribe(req: SendMessageRequest):
    def stream():
        task_id = req.message.taskId or str(uuid.uuid4())
        ctx_id = req.message.contextId or str(uuid.uuid4())
        req.message.taskId = task_id
        req.message.contextId = ctx_id

        def emit(obj) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        task = Task(id=task_id, contextId=ctx_id,
                    status=TaskStatus(state="submitted"), history=[req.message])
        _tasks[task_id] = task
        yield emit(TaskStatusUpdateEvent(task=task).model_dump())

        doc = _extract_document(req.message)
        if doc is None:
            task.status = TaskStatus(state="failed")
            _tasks[task_id] = task
            yield emit(TaskStatusUpdateEvent(task=task).model_dump())
            return

        task.status = TaskStatus(state="working")
        _tasks[task_id] = task
        yield emit(TaskStatusUpdateEvent(task=task).model_dump())

        summary = _generate_summary(doc)
        artifact = A2AArtifact(
            name="summary", parts=[DataPart(data=summary)], lastChunk=True)
        task.artifacts = [artifact]
        _tasks[task_id] = task
        yield emit(TaskArtifactUpdateEvent(
            taskId=task_id, contextId=ctx_id, artifact=artifact).model_dump())

        agent_msg = Message(
            role="agent",
            parts=[TextPart(text=summary["summary"])],
            taskId=task_id, contextId=ctx_id,
        )
        task.status = TaskStatus(state="completed", message=agent_msg)
        task.history = [req.message, agent_msg]
        _tasks[task_id] = task
        yield emit(TaskStatusUpdateEvent(task=task).model_dump())

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
