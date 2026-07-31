"""
StyleAgent — 标准 A2A 实现，唯一职责是在生成大纲前询问写作风格（真实触发 input-required）
输入：Message.parts 中的 TextPart（主题文本）
首次调用：不产出 artifact，直接返回 input-required，附带提问文本
续接调用（Message.taskId 指向一个 input-required 状态的任务）：解析用户回答，
  输出 Task.artifacts[0] 包含 DataPart（style 数据）；回答"跳过"/空 则视为默认风格
接口：
  GET  /.well-known/agent.json
  POST /tasks/send
  GET  /tasks/{task_id}
"""
from __future__ import annotations

import uuid
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import (
    A2AArtifact, AgentCapabilities, AgentCard, AgentSkill,
    DataPart, Message, SendMessageRequest, Task, TaskStatus, TextPart,
)

app = FastAPI(title="StyleAgent")

_BASE_URL = "http://127.0.0.1:8004"

AGENT_CARD = AgentCard(
    name="StyleAgent",
    description="在生成大纲前询问用户期望的写作风格与目标读者，建议作为内容生成链的第一步执行；用户可跳过使用默认风格",
    url=_BASE_URL,
    version="1.0",
    skills=[
        AgentSkill(
            id="ask_style",
            name="询问写作风格",
            description="输入主题文本，先向用户提问写作风格偏好，得到回答后输出风格设定（JSON）",
            inputModes=["text"],
            outputModes=["data"],
            examples=["科普向", "技术向、面向工程师", "跳过"],
        )
    ],
    capabilities=AgentCapabilities(streaming=False, stateTransitionHistory=True),
    authentication={"schemes": ["apiKey"]},
)

# 内存任务存储（生产建议换为 Redis/DB）
_tasks: dict[str, Task] = {}

_SKIP_WORDS = ("跳过", "默认", "不需要", "无", "skip", "default")


def _extract_text(message: Message) -> str:
    for part in message.parts:
        if hasattr(part, "type") and part.type == "text":
            return part.text.strip()
    return ""


def _parse_style_answer(answer: str) -> dict:
    if not answer or any(w in answer for w in _SKIP_WORDS):
        return {"style": "默认", "skipped": True}
    return {"style": answer, "skipped": False}


def _process(message: Message) -> Task:
    # 续接：taskId 指向一个正在等待用户回答的任务
    if message.taskId and message.taskId in _tasks:
        existing = _tasks[message.taskId]
        if existing.status.state == "input-required":
            answer = _extract_text(message)
            style = _parse_style_answer(answer)
            artifact = A2AArtifact(
                name="style", description="写作风格设定",
                parts=[DataPart(data=style)], lastChunk=True,
            )
            agent_msg = Message(
                role="agent",
                parts=[TextPart(text=f"已采用风格：{style['style']}")],
                taskId=existing.id, contextId=existing.contextId,
            )
            existing.status = TaskStatus(state="completed", message=agent_msg)
            existing.artifacts = [artifact]
            existing.history = (existing.history or []) + [message, agent_msg]
            _tasks[existing.id] = existing
            return existing

    # 首次调用：提出风格问题，进入 input-required，等待用户续接回答
    task_id = message.taskId or str(uuid.uuid4())
    ctx_id = message.contextId or str(uuid.uuid4())
    topic_hint = _extract_text(message)
    question = (
        f"关于「{topic_hint}」这篇文章，希望使用什么写作风格？（如：科普向、技术向、正式、轻松幽默）\n"
        "不需要指定可直接回复「跳过」。"
        if topic_hint else
        "希望这篇文章使用什么写作风格？不需要指定可直接回复「跳过」。"
    )
    ask_msg = Message(role="agent", parts=[TextPart(text=question)], taskId=task_id, contextId=ctx_id)
    task = Task(id=task_id, contextId=ctx_id,
                status=TaskStatus(state="input-required", message=ask_msg), history=[message])
    _tasks[task_id] = task
    return task


# ---------- 标准 A2A 接口 ----------

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
