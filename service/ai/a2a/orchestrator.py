"""
A2A 内容生成链编排器（Client Agent）

标准 A2A 流程：
  1. 能力发现：GET /.well-known/agent.json
  2. 发起任务：POST /tasks/send，body = {message: {role:"user", parts:[...]}}
  3. 从 Task.artifacts[0].parts[0].data 取出工件数据
  4. 构造下一条 Message，把工件数据放入 DataPart，发给下一个 Agent
  5. 重复直到链结束
"""
from __future__ import annotations

import uuid
from datetime import datetime

import requests

from service.ai.a2a.schemas import (
    A2AArtifact, ChainStep, DataPart, Message, OrchestrationResult,
    SendMessageRequest, Task, TaskStatus, TextPart,
)

# 三个能力 Agent 的 base URL（可通过配置覆盖）
AGENT_URLS = {
    "OutlineAgent": "http://127.0.0.1:8001",
    "DocAgent": "http://127.0.0.1:8002",
    "SummaryAgent": "http://127.0.0.1:8003",
}


# ---------- A2A 客户端工具函数 ----------

def _discover(agent_name: str) -> dict:
    """能力发现：读取远程 Agent 的智能体名片"""
    base = AGENT_URLS[agent_name]
    r = requests.get(f"{base}/.well-known/agent.json", timeout=5)
    r.raise_for_status()
    return r.json()


def _send_message(agent_name: str, message: Message) -> Task:
    """
    向远程 Agent 发送消息（POST /tasks/send），返回 Task 对象。
    这是 A2A 协议里的 SendMessage 请求。
    """
    base = AGENT_URLS[agent_name]
    req = SendMessageRequest(message=message)
    r = requests.post(
        f"{base}/tasks/send",
        json=req.model_dump(),
        timeout=30,
    )
    r.raise_for_status()
    return Task(**r.json())


def _poll_task(agent_name: str, task_id: str) -> Task:
    """轮询任务状态（GET /tasks/{task_id}），用于异步场景"""
    base = AGENT_URLS[agent_name]
    r = requests.get(f"{base}/tasks/{task_id}", timeout=10)
    r.raise_for_status()
    return Task(**r.json())


def _extract_data_from_task(task: Task) -> dict | None:
    """从 Task.artifacts 中提取第一个 DataPart 的数据"""
    if not task.artifacts:
        return None
    for artifact in task.artifacts:
        for part in artifact.parts:
            if hasattr(part, "type") and part.type == "data":
                return part.data
    return None


def _short_summary(data: dict | None, max_len: int = 80) -> str:
    if not data:
        return ""
    if data.get("topic"):
        return f"topic: {data['topic']}"[:max_len]
    if data.get("title"):
        return f"title: {data['title']}"[:max_len]
    s = str(data)
    return s[:max_len] + ("..." if len(s) > max_len else "")


# ---------- 编排链 ----------

def run_chain(topic: str) -> OrchestrationResult:
    """
    执行 OutlineAgent → DocAgent → SummaryAgent 链。

    每步：
      - 构造 Message(role="user", parts=[TextPart 或 DataPart])
      - POST /tasks/send → 得到 Task
      - 从 Task.artifacts[0].parts[0].data 提取工件数据
      - 传给下一个 Agent
    """
    chain: list[ChainStep] = []
    tasks: list[Task] = []
    ctx_id = str(uuid.uuid4())     # 整条链共享一个 contextId

    # ── Step 1: OutlineAgent ──────────────────────────────────────────
    step = ChainStep(
        step_index=1, agent_name="OutlineAgent", agent_version="1.0",
        status="submitted",
        input_summary=f"topic: {topic}",
        started_at=datetime.utcnow().isoformat() + "Z",
    )
    chain.append(step)

    try:
        # 能力发现（可选，确认 agent 在线）
        _discover("OutlineAgent")

        # 构造标准 Message：用户提需求，parts 里放 TextPart
        msg = Message(
            role="user",
            parts=[TextPart(text=topic)],
            contextId=ctx_id,
        )
        task = _send_message("OutlineAgent", msg)

        step.status = task.status.state
        step.ended_at = datetime.utcnow().isoformat() + "Z"
        outline_data = _extract_data_from_task(task)
        step.output_summary = _short_summary(outline_data) or "大纲"
        tasks.append(task)
    except Exception as e:
        step.status = "failed"
        step.ended_at = datetime.utcnow().isoformat() + "Z"
        step.error_message = str(e)
        return OrchestrationResult(chain=chain, tasks=tasks,
                                   final_task=_failed_task(ctx_id, str(e)))

    if task.status.state != "completed" or not outline_data:
        step.status = "failed"
        return OrchestrationResult(chain=chain, tasks=tasks, final_task=task)

    # ── Step 2: DocAgent ──────────────────────────────────────────────
    step = ChainStep(
        step_index=2, agent_name="DocAgent", agent_version="1.0",
        status="submitted",
        input_summary=_short_summary(outline_data),
        started_at=datetime.utcnow().isoformat() + "Z",
    )
    chain.append(step)

    try:
        _discover("DocAgent")

        # 把上一步的 Artifact 数据放入 DataPart，传给下一个 Agent
        msg = Message(
            role="user",
            parts=[
                TextPart(text="请根据以下大纲生成文章正文"),
                DataPart(data=outline_data),
            ],
            contextId=ctx_id,
        )
        task = _send_message("DocAgent", msg)

        step.status = task.status.state
        step.ended_at = datetime.utcnow().isoformat() + "Z"
        doc_data = _extract_data_from_task(task)
        step.output_summary = _short_summary(doc_data) or "正文"
        tasks.append(task)
    except Exception as e:
        step.status = "failed"
        step.ended_at = datetime.utcnow().isoformat() + "Z"
        step.error_message = str(e)
        return OrchestrationResult(chain=chain, tasks=tasks,
                                   final_task=_failed_task(ctx_id, str(e)))

    if task.status.state != "completed" or not doc_data:
        step.status = "failed"
        return OrchestrationResult(chain=chain, tasks=tasks, final_task=task)

    # ── Step 3: SummaryAgent ──────────────────────────────────────────
    step = ChainStep(
        step_index=3, agent_name="SummaryAgent", agent_version="1.0",
        status="submitted",
        input_summary=_short_summary(doc_data),
        started_at=datetime.utcnow().isoformat() + "Z",
    )
    chain.append(step)

    try:
        _discover("SummaryAgent")

        msg = Message(
            role="user",
            parts=[
                TextPart(text="请根据以下正文生成摘要"),
                DataPart(data=doc_data),
            ],
            contextId=ctx_id,
        )
        task = _send_message("SummaryAgent", msg)

        step.status = task.status.state
        step.ended_at = datetime.utcnow().isoformat() + "Z"
        summary_data = _extract_data_from_task(task)
        step.output_summary = _short_summary(summary_data) or "摘要"
        tasks.append(task)
    except Exception as e:
        step.status = "failed"
        step.ended_at = datetime.utcnow().isoformat() + "Z"
        step.error_message = str(e)
        return OrchestrationResult(chain=chain, tasks=tasks,
                                   final_task=_failed_task(ctx_id, str(e)))

    return OrchestrationResult(chain=chain, tasks=tasks, final_task=task)


def _failed_task(ctx_id: str, error: str) -> Task:
    return Task(
        contextId=ctx_id,
        status=TaskStatus(state="failed"),
        metadata={"error": error},
    )


# ---------- 供 Flask 路由使用 ----------

def get_result_for_frontend(topic: str) -> dict:
    """
    执行链，把结果转换为前端可展示的 JSON。
    chain：调用链（每步 agent、状态、时间）
    tasks：每步的完整 Task（含 artifacts），前端可展示每步输出
    final_task：最后一步的 Task
    """
    result = run_chain(topic)
    return result.model_dump()


if __name__ == "__main__":
    import json
    out = get_result_for_frontend("A2A 协议简介")
    print(json.dumps(out, ensure_ascii=False, indent=2))
