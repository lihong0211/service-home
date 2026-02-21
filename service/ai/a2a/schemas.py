"""
A2A 协议标准类型定义
基于 A2A Protocol Specification (HTTP / JSON-RPC 2.0 / SSE)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ---------- Part（零件）：消息或工件的最小内容单位 ----------

class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] | None = None


class FilePart(BaseModel):
    type: Literal["file"] = "file"
    file: dict[str, Any]        # {name, mimeType, bytes(base64) 或 uri}
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


Part = Union[TextPart, FilePart, DataPart]


# ---------- Message（消息）----------

class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[Part]
    messageId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    taskId: str | None = None
    contextId: str | None = None
    metadata: dict[str, Any] | None = None


# ---------- Artifact（工件）----------

class A2AArtifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None
    description: str | None = None
    parts: list[Part]
    index: int = 0
    append: bool | None = None
    lastChunk: bool | None = None
    metadata: dict[str, Any] | None = None


# ---------- Task（任务）生命周期 ----------

TaskState = Literal["submitted", "working", "input-required", "completed", "failed", "canceled"]


class TaskStatus(BaseModel):
    state: TaskState
    message: Message | None = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contextId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    artifacts: list[A2AArtifact] | None = None
    history: list[Message] | None = None
    metadata: dict[str, Any] | None = None


# ---------- Agent Card（智能体名片）----------

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str | None = None
    inputModes: list[str] = ["text"]
    outputModes: list[str] = ["data"]
    tags: list[str] | None = None
    examples: list[str] | None = None


class AgentCapabilities(BaseModel):
    streaming: bool = True
    pushNotifications: bool = False
    stateTransitionHistory: bool = True


class AgentCard(BaseModel):
    name: str
    description: str | None = None
    url: str
    version: str = "1.0"
    skills: list[AgentSkill] = []
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    authentication: dict[str, Any] | None = None
    defaultInputModes: list[str] = ["text"]
    defaultOutputModes: list[str] = ["data"]


# ---------- API 请求体 ----------

class SendMessageRequest(BaseModel):
    message: Message


# ---------- SSE 事件 ----------

class TaskStatusUpdateEvent(BaseModel):
    """SSE event: 任务状态变更（submitted → working → completed/failed）"""
    task: Task


class TaskArtifactUpdateEvent(BaseModel):
    """SSE event: 工件分块推送（流式输出时逐块）"""
    taskId: str
    contextId: str
    artifact: A2AArtifact


# ---------- 编排器：调用链步骤（供前端展示） ----------

class ChainStep(BaseModel):
    step_index: int
    agent_name: str
    agent_version: str
    status: TaskState | Literal["running"] = "submitted"
    input_summary: str = ""
    output_summary: str = ""
    started_at: str = ""
    ended_at: str = ""
    error_message: str | None = None


class OrchestrationResult(BaseModel):
    """编排结果：调用链 + 每步 Task（含 artifacts），供前端展示"""
    chain: list[ChainStep]
    tasks: list[Task]           # 每步的完整 Task（含 artifacts）
    final_task: Task            # 最后一步的 Task
