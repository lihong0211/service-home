"""
A2A 内容生成链编排器（Client Agent / Host Agent）

标准 A2A 流程：
  1. 能力发现：GET /.well-known/agent.json（service/ai/a2a/registry.py）
  2. 动态路由：每一步由 Host Agent（service/ai/a2a/host_agent.py）结合目标 +
     已发现 Agent 的能力 + 已完成进度，决定下一步调用谁，而不是写死的顺序
  3. 发起任务：POST /tasks/send，body = {message: {role:"user", parts:[...]}}
  4. 从 Task.artifacts[0].parts[0].data 取出工件数据，传给下一个 Agent
  5. 重复直到 Host Agent 判断完成（DONE），或某步返回 input-required 需要
     人工补充信息（暂停并可通过 /ai/a2a/chain/resume 续跑）

单步调用失败时按 _RETRY_ATTEMPTS 重试，仍失败则该步标记 failed，但已完成
步骤的产出会保留在结果里，不会整链丢失。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Generator

import requests

from fastapi import Request
from fastapi.responses import StreamingResponse

from utils.http_body import read_json_optional

from service.ai.a2a.schemas import (
    ChainStep, DataPart, Message, OrchestrationResult,
    SendMessageRequest, Task, TaskStatus, TextPart,
)
from service.ai.a2a.host_agent import decide_next_agent, MAX_STEPS
from service.ai.a2a.registry import agent_url, discover as registry_discover

_RETRY_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 2
_SEND_TIMEOUT_SECONDS = 600


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ---------- A2A 客户端工具函数 ----------

def _send_message(agent_name: str, message: Message) -> Task:
    """向远程 Agent 发送消息（POST /tasks/send），返回 Task 对象。"""
    req = SendMessageRequest(message=message)
    r = requests.post(
        f"{agent_url(agent_name)}/tasks/send",
        json=req.model_dump(),
        timeout=_SEND_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    return Task(**r.json())


def _send_message_with_retry(agent_name: str, message: Message) -> Task:
    """网络错误/超时时按退避间隔重试；重试耗尽后抛出最后一次异常，由调用方降级处理。"""
    last_error: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return _send_message(agent_name, message)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error


def _extract_data_from_task(task: Task) -> dict | None:
    """从 Task.artifacts 中提取第一个 DataPart 的数据"""
    if not task.artifacts:
        return None
    for artifact in task.artifacts:
        for part in artifact.parts:
            if hasattr(part, "type") and part.type == "data":
                return part.data
    return None


def _extract_prompt(task: Task) -> str:
    """input-required 状态下，从 Task.status.message 里取出需要用户补充的提示文本"""
    if task.status.message:
        for part in task.status.message.parts:
            if hasattr(part, "type") and part.type == "text":
                return part.text
    return "该步骤需要补充信息才能继续"


def _short_summary(data: dict | None, max_len: int = 80) -> str:
    if not data:
        return ""
    if data.get("topic"):
        return f"topic: {data['topic']}"[:max_len]
    if data.get("title"):
        return f"title: {data['title']}"[:max_len]
    s = str(data)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def _failed_task(ctx_id: str, error: str) -> Task:
    return Task(
        contextId=ctx_id,
        status=TaskStatus(state="failed"),
        metadata={"error": error},
    )


# ---------- 人机协作：暂停会话（input-required）----------

@dataclass
class PausedSession:
    context_id: str
    topic: str
    chain: list[ChainStep]
    tasks: list[Task]
    completed_agents: list[str]
    latest_data: dict | None
    paused_agent: str
    paused_task_id: str


# 进程内存储：contextId -> 暂停时的链路状态（生产建议换为 Redis/DB）
_paused_sessions: dict[str, PausedSession] = {}


# ---------- 编排链：动态路由 + 重试 + 暂停恢复 ----------

class ChainRun:
    """
    一次链路执行的状态载体。sync（run_chain/resume_chain）与 SSE（stream_chain）
    两种对外形态共用 run_events() 产出的事件序列，只是消费方式不同，避免同一套
    路由/重试/暂停逻辑写两份。
    """

    def __init__(self, topic: str, paused: PausedSession | None = None):
        self.topic = topic
        if paused:
            self.context_id = paused.context_id
            self.chain = paused.chain
            self.tasks = paused.tasks
            self.completed_agents = paused.completed_agents
            self.latest_data = paused.latest_data
            self._resume_agent: str | None = paused.paused_agent
            self._resume_task_id: str | None = paused.paused_task_id
        else:
            self.context_id = str(uuid.uuid4())
            self.chain = []
            self.tasks = []
            self.completed_agents = []
            self.latest_data = None
            self._resume_agent = None
            self._resume_task_id = None
        self._resume_answer: str = ""
        self.final_task: Task | None = None

    @classmethod
    def from_paused(cls, paused: PausedSession, answer: str) -> "ChainRun":
        run = cls(paused.topic, paused=paused)
        run._resume_answer = answer
        return run

    def run_events(self):
        if self._resume_agent:
            parts = [TextPart(text=self._resume_answer)]
            if self.latest_data:
                parts.append(DataPart(data=self.latest_data))
            ok = yield from self._call_step(
                self._resume_agent, parts, is_resume=True, resume_task_id=self._resume_task_id)
            if not ok:
                return

        while len(self.completed_agents) < MAX_STEPS:
            next_agent = decide_next_agent(self.topic, self.completed_agents, self.latest_data)
            if next_agent == "DONE":
                break
            # 主题文本每一步都带上（不止第一步）——中间可能插入不产出 "topic" 字段的
            # 步骤（如 StyleAgent 只产出风格数据），后续 Agent 仍需要拿到原始主题。
            parts = [TextPart(text=self.topic)]
            if self.latest_data:
                parts.append(DataPart(data=self.latest_data))
            ok = yield from self._call_step(next_agent, parts)
            if not ok:
                return

        self.final_task = self.tasks[-1] if self.tasks else _failed_task(self.context_id, "未完成任何步骤")
        yield {
            "type": "chain_done",
            "context_id": self.context_id,
            "chain": [s.model_dump() for s in self.chain],
            "data": self.latest_data,
        }

    def _call_step(self, agent_name: str, parts: list, is_resume: bool = False, resume_task_id: str | None = None):
        """执行单步调用。返回 True 表示继续主循环，False 表示链路已结束（完成/暂停/失败）。"""
        step_index = len(self.chain) + 1
        yield {"type": "step_start", "step": step_index, "agent": agent_name}

        step = ChainStep(
            step_index=step_index, agent_name=agent_name, agent_version="1.0",
            status="submitted",
            input_summary=("补充信息续跑" if is_resume else (_short_summary(self.latest_data) or f"topic: {self.topic}")),
            started_at=_now(),
        )
        self.chain.append(step)

        try:
            if not is_resume:
                registry_discover(agent_name)  # 能力发现兼在线检查
            # 续接时带上原 taskId，让 Agent 能识别这是对同一个 input-required
            # 任务的回答，而不是一次新的调用（标准 A2A 任务续接语义）。
            msg = Message(role="user", parts=parts, contextId=self.context_id, taskId=resume_task_id)
            task = _send_message_with_retry(agent_name, msg)
        except Exception as e:
            step.status, step.ended_at, step.error_message = "failed", _now(), str(e)
            self.final_task = _failed_task(self.context_id, str(e))
            yield {"type": "chain_error", "step": step_index, "agent": agent_name, "error": str(e)}
            return False

        data = _extract_data_from_task(task)
        step.status, step.ended_at = task.status.state, _now()
        self.tasks.append(task)

        if task.status.state == "input-required":
            step.output_summary = "等待补充信息"
            _paused_sessions[self.context_id] = PausedSession(
                context_id=self.context_id, topic=self.topic, chain=self.chain,
                tasks=self.tasks, completed_agents=self.completed_agents,
                latest_data=self.latest_data, paused_agent=agent_name,
                paused_task_id=task.id,
            )
            self.final_task = task
            yield {"type": "chain_paused", "step": step_index, "agent": agent_name,
                   "context_id": self.context_id, "message": _extract_prompt(task)}
            return False

        if task.status.state != "completed" or data is None:
            step.status = "failed"
            self.final_task = task
            yield {"type": "chain_error", "step": step_index, "agent": agent_name, "error": "no artifact"}
            return False

        step.output_summary = _short_summary(data) or agent_name
        self.completed_agents.append(agent_name)
        self.latest_data = data
        yield {"type": "step_done", "step": step_index, "agent": agent_name,
               "status": step.status, "data": data}
        return True


def run_chain(topic: str) -> OrchestrationResult:
    """执行动态路由链（Host Agent 逐步决定下一个 Agent），直到完成或暂停/失败。"""
    run = ChainRun(topic)
    for _ in run.run_events():
        pass
    return OrchestrationResult(chain=run.chain, tasks=run.tasks, final_task=run.final_task)


def resume_chain(context_id: str, answer: str) -> OrchestrationResult:
    """为处于 input-required 状态的会话补充信息并继续执行。"""
    paused = _paused_sessions.pop(context_id, None)
    if paused is None:
        raise KeyError(f"未找到待续处理的会话: {context_id}")
    run = ChainRun.from_paused(paused, answer)
    for _ in run.run_events():
        pass
    return OrchestrationResult(chain=run.chain, tasks=run.tasks, final_task=run.final_task)


# ---------- SSE 流式链路 ----------

def stream_chain(topic: str) -> Generator[str, None, None]:
    """
    SSE 生成器：每完成一个 Agent 步骤就推送一条事件。

    事件格式（每条均为标准 SSE `data: <json>\\n\\n`）：
      step_start   — 某步骤开始执行
      step_done    — 某步骤完成，携带该步骤的 artifact data
      chain_paused — 某步骤返回 input-required，链路暂停，携带 context_id 供
                     POST /ai/a2a/chain/resume 续跑
      chain_done   — 全链完成，携带 chain 摘要 + final artifact data
      chain_error  — 某步骤失败，携带错误信息
    """
    def emit(payload: dict) -> str:
        # 内部事件字典用 "type" 字段区分种类，SSE 线上协议字段名是 "event"
        # （前端 a2a.ts 按 event.event 分发）——这里做一次映射，两边各自保持习惯命名。
        wire = dict(payload)
        wire["event"] = wire.pop("type")
        return f"data: {json.dumps(wire, ensure_ascii=False)}\n\n"

    run = ChainRun(topic)
    for event in run.run_events():
        yield emit(event)
    yield "data: [DONE]\n\n"


# ---------- 供路由层使用 ----------

def get_result_for_frontend(topic: str) -> dict:
    """
    执行链，把结果转换为前端可展示的 JSON。
    chain：调用链（每步 agent、状态、时间）
    tasks：每步的完整 Task（含 artifacts），前端可展示每步输出
    final_task：最后一步的 Task
    """
    result = run_chain(topic)
    return result.model_dump()


# ---------------------------------------------------------------------------
# HTTP 视图：供 routes/ai 注册 POST
# ---------------------------------------------------------------------------


async def a2a_chain_api(request: Request):
    """POST /ai/a2a/chain 执行 A2A 内容生成链，返回 chain + artifacts + final_artifact，供前端展示。"""
    body = (await read_json_optional(request)) or {}
    topic = body.get("topic", "").strip()
    if not topic:
        return ({"code": 400, "msg": "缺少参数: topic", "data": None}, 400)
    try:
        data = get_result_for_frontend(topic)
        return {"code": 0, "msg": "ok", "data": data}
    except Exception as e:
        return ({"code": 500, "msg": str(e), "data": None}, 500)


async def a2a_chain_resume_api(request: Request):
    """POST /ai/a2a/chain/resume  为处于 input-required 状态的链路补充信息并继续执行。"""
    body = (await read_json_optional(request)) or {}
    context_id = body.get("context_id", "").strip()
    answer = body.get("answer", "").strip()
    if not context_id or not answer:
        return ({"code": 400, "msg": "缺少参数: context_id / answer", "data": None}, 400)
    try:
        result = resume_chain(context_id, answer)
        return {"code": 0, "msg": "ok", "data": result.model_dump()}
    except KeyError as e:
        return ({"code": 404, "msg": e.args[0] if e.args else str(e), "data": None}, 404)
    except Exception as e:
        return ({"code": 500, "msg": str(e), "data": None}, 500)


async def a2a_chain_stream_api(request: Request):
    """POST /ai/a2a/chain/stream  SSE 流式接口：每完成一个 Agent 步骤推送一条事件。"""
    body = (await read_json_optional(request)) or {}
    topic = body.get("topic", "").strip()
    if not topic:
        return ({"code": 400, "msg": "缺少参数: topic", "data": None}, 400)
    return StreamingResponse(
        stream_chain(topic),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


if __name__ == "__main__":
    import json as _json
    out = get_result_for_frontend("A2A 协议简介")
    print(_json.dumps(out, ensure_ascii=False, indent=2))
