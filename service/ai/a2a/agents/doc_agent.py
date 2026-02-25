"""
DocAgent — 标准 A2A 实现
输入：Message.parts 中包含 DataPart（outline artifact 的 data）
输出：Task.artifacts[0] 包含 DataPart（document 数据）
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

app = FastAPI(title="DocAgent")

_BASE_URL = "http://127.0.0.1:8002"

AGENT_CARD = AgentCard(
    name="DocAgent",
    description="根据文章大纲生成正文内容",
    url=_BASE_URL,
    version="1.0",
    skills=[
        AgentSkill(
            id="generate_document",
            name="生成正文",
            description="输入大纲 DataPart，输出各章节正文（JSON）",
            inputModes=["data"],
            outputModes=["data"],
        )
    ],
    capabilities=AgentCapabilities(streaming=True, stateTransitionHistory=True),
    authentication={"schemes": ["apiKey"]},
)

_tasks: dict[str, Task] = {}


def _extract_outline(message: Message) -> dict | None:
    """从 Message.parts 中提取 outline DataPart 数据"""
    for part in message.parts:
        if hasattr(part, "type") and part.type == "data":
            data = part.data or {}
            if "sections" in data or "topic" in data:
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
        "options": {"temperature": 0.7, "num_predict": 4096},
    }
    resp = requests.post(f"{_OLLAMA_URL}/api/chat", json=body, timeout=300)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    return _THINKING_RE.sub("", content).strip()


def _generate_document(outline: dict) -> dict:
    """调用 LLM 根据大纲逐章节生成正文"""
    topic = outline.get("topic", "未命名")
    sections = outline.get("sections", [])
    paragraphs = []

    for sec in sections:
        heading = sec.get("title", "")
        key_points = sec.get("key_points", [])
        points_str = "\n".join(f"- {p}" for p in key_points)
        prompt = (
            f"请为文章「{topic}」的章节「{heading}」撰写一段正文。\n\n"
            f"该章节需要涵盖以下要点：\n{points_str}\n\n"
            "要求：\n"
            "- 用流畅的中文段落写作，内容具体、充实\n"
            "- 不少于 150 字\n"
            "- 直接输出正文段落，不要标题，不要编号"
        )
        try:
            text = _call_llm(prompt)
            if text:
                paragraphs.append({"heading": heading, "text": text})
            else:
                raise ValueError("empty response")
        except Exception:
            paragraphs.append({
                "heading": heading,
                "text": "\n".join(f"- {p}" for p in key_points),
            })

    return {"title": topic, "paragraphs": paragraphs}


def _process(message: Message) -> Task:
    task_id = message.taskId or str(uuid.uuid4())
    ctx_id = message.contextId or str(uuid.uuid4())
    message.taskId = task_id
    message.contextId = ctx_id

    task = Task(id=task_id, contextId=ctx_id,
                status=TaskStatus(state="submitted"), history=[message])
    _tasks[task_id] = task

    outline = _extract_outline(message)
    if outline is None:
        task.status = TaskStatus(state="failed")
        _tasks[task_id] = task
        return task

    task.status = TaskStatus(state="working")
    _tasks[task_id] = task

    doc = _generate_document(outline)
    artifact = A2AArtifact(
        name="document",
        description="文章正文",
        parts=[DataPart(data=doc)],
        lastChunk=True,
    )
    agent_msg = Message(
        role="agent",
        parts=[TextPart(text=f"正文生成完成，共 {len(doc['paragraphs'])} 节。")],
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

        outline = _extract_outline(req.message)
        if outline is None:
            task.status = TaskStatus(state="failed")
            _tasks[task_id] = task
            yield emit(TaskStatusUpdateEvent(task=task).model_dump())
            return

        task.status = TaskStatus(state="working")
        _tasks[task_id] = task
        yield emit(TaskStatusUpdateEvent(task=task).model_dump())

        doc = _generate_document(outline)
        artifact = A2AArtifact(
            name="document", parts=[DataPart(data=doc)], lastChunk=True)
        task.artifacts = [artifact]
        _tasks[task_id] = task
        yield emit(TaskArtifactUpdateEvent(
            taskId=task_id, contextId=ctx_id, artifact=artifact).model_dump())

        agent_msg = Message(
            role="agent",
            parts=[TextPart(text=f"正文生成完成，共 {len(doc['paragraphs'])} 节。")],
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
    uvicorn.run(app, host="0.0.0.0", port=8002)
