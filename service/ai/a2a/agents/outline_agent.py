"""
OutlineAgent — 标准 A2A 实现
接口：
  GET  /.well-known/agent.json     智能体名片（能力发现）
  POST /tasks/send                 发送消息，同步返回 Task（含 artifacts）
  GET  /tasks/{task_id}            查询任务状态（轮询）
  POST /tasks/sendSubscribe        发送消息，SSE 流式推送状态与工件
"""
from __future__ import annotations

import json
import uuid
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import (
    A2AArtifact, AgentCapabilities, AgentCard, AgentSkill,
    DataPart, Message, SendMessageRequest, Task, TaskStatus,
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TextPart,
)

app = FastAPI(title="OutlineAgent")

_BASE_URL = "http://127.0.0.1:8001"

AGENT_CARD = AgentCard(
    name="OutlineAgent",
    description="根据用户给定的主题生成结构化文章大纲",
    url=_BASE_URL,
    version="1.0",
    skills=[
        AgentSkill(
            id="generate_outline",
            name="生成大纲",
            description="输入主题文本，输出 JSON 格式的文章大纲",
            inputModes=["text"],
            outputModes=["data"],
            examples=["A2A 协议简介", "新能源汽车的未来"],
        )
    ],
    capabilities=AgentCapabilities(streaming=True, stateTransitionHistory=True),
    authentication={"schemes": ["apiKey"]},
)

# 内存任务存储（生产建议换为 Redis/DB）
_tasks: dict[str, Task] = {}


def _extract_topic(message: Message) -> str:
    """从 Message.parts 中提取主题文本"""
    for part in message.parts:
        if hasattr(part, "type"):
            if part.type == "text":
                return part.text.strip()
            if part.type == "data":
                data = part.data or {}
                if data.get("topic"):
                    return str(data["topic"])
    return ""


def _generate_outline(topic: str) -> dict:
    """根据主题生成大纲（可替换为 LLM 调用）"""
    return {
        "topic": topic,
        "sections": [
            {"title": "引言", "key_points": [f"介绍{topic}的背景与意义"]},
            {"title": "核心内容", "key_points": ["要点一", "要点二", "要点三"]},
            {"title": "总结", "key_points": ["回顾与展望"]},
        ],
    }


def _process(message: Message) -> Task:
    """同步处理消息，走完完整任务生命周期，返回 completed/failed Task"""
    task_id = message.taskId or str(uuid.uuid4())
    ctx_id = message.contextId or str(uuid.uuid4())
    message.taskId = task_id
    message.contextId = ctx_id

    task = Task(id=task_id, contextId=ctx_id,
                status=TaskStatus(state="submitted"), history=[message])
    _tasks[task_id] = task

    topic = _extract_topic(message)
    if not topic:
        task.status = TaskStatus(state="failed")
        _tasks[task_id] = task
        return task

    task.status = TaskStatus(state="working")
    _tasks[task_id] = task

    outline = _generate_outline(topic)
    artifact = A2AArtifact(
        name="outline",
        description="文章大纲",
        parts=[DataPart(data=outline)],
        lastChunk=True,
    )
    agent_msg = Message(
        role="agent",
        parts=[TextPart(text=f"已生成《{topic}》大纲，共 {len(outline['sections'])} 节。")],
        taskId=task_id, contextId=ctx_id,
    )
    task.status = TaskStatus(state="completed", message=agent_msg)
    task.artifacts = [artifact]
    task.history = [message, agent_msg]
    _tasks[task_id] = task
    return task


# ---------- 标准 A2A 接口 ----------

@app.get("/.well-known/agent.json")
async def get_agent_card():
    return AGENT_CARD.model_dump()


@app.post("/tasks/send")
async def tasks_send(req: SendMessageRequest):
    """发送消息，同步返回 Task（含 artifacts）"""
    return _process(req.message).model_dump()


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """查询任务状态（供客户端轮询）"""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.model_dump()


@app.post("/tasks/sendSubscribe")
async def tasks_send_subscribe(req: SendMessageRequest):
    """发送消息，SSE 流式推送：submitted → working → artifact → completed"""
    def stream():
        task_id = req.message.taskId or str(uuid.uuid4())
        ctx_id = req.message.contextId or str(uuid.uuid4())
        req.message.taskId = task_id
        req.message.contextId = ctx_id

        def emit(obj) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        # submitted
        task = Task(id=task_id, contextId=ctx_id,
                    status=TaskStatus(state="submitted"), history=[req.message])
        _tasks[task_id] = task
        yield emit(TaskStatusUpdateEvent(task=task).model_dump())

        topic = _extract_topic(req.message)
        if not topic:
            task.status = TaskStatus(state="failed")
            _tasks[task_id] = task
            yield emit(TaskStatusUpdateEvent(task=task).model_dump())
            return

        # working
        task.status = TaskStatus(state="working")
        _tasks[task_id] = task
        yield emit(TaskStatusUpdateEvent(task=task).model_dump())

        # artifact
        outline = _generate_outline(topic)
        artifact = A2AArtifact(
            name="outline", parts=[DataPart(data=outline)], lastChunk=True)
        task.artifacts = [artifact]
        _tasks[task_id] = task
        yield emit(TaskArtifactUpdateEvent(
            taskId=task_id, contextId=ctx_id, artifact=artifact).model_dump())

        # completed
        agent_msg = Message(
            role="agent",
            parts=[TextPart(text=f"大纲生成完成，共 {len(outline['sections'])} 节。")],
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
