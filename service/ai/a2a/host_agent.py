"""
Host Agent：多智能体动态路由。

不再走写死的 Outline→Doc→Summary 顺序，每一步都让 LLM 结合
「目标 + 已发现 Agent 的能力描述 + 已完成进度」决定下一步调用哪个 Agent，
直到判断任务完成（DONE）。

LLM 不可用 / 输出无法解析 / 选择了已完成过的 Agent 时，退化为按
_FALLBACK_ORDER 依次选择下一个未完成的 Agent —— 保证模型不稳定时
行为仍然可预测，不会卡死或乱跳。
"""
from __future__ import annotations

import re

import requests

from service.ai.a2a.registry import discover_all

_OLLAMA_URL = "http://localhost:11434"
_LLM_MODEL = "deepseek-r1:1.5b"
_THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

MAX_STEPS = 5
_FALLBACK_ORDER = ["StyleAgent", "OutlineAgent", "DocAgent", "SummaryAgent"]


def _call_llm(prompt: str) -> str:
    body = {
        "model": _LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 200},
    }
    resp = requests.post(f"{_OLLAMA_URL}/api/chat", json=body, timeout=60)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    return _THINKING_RE.sub("", content).strip()


def _short_summary(data: dict | None, max_len: int = 80) -> str:
    if not data:
        return "无"
    if data.get("topic"):
        return f"topic: {data['topic']}"[:max_len]
    if data.get("title"):
        return f"title: {data['title']}"[:max_len]
    s = str(data)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def _fallback_next(completed_agents: list[str]) -> str:
    for name in _FALLBACK_ORDER:
        if name not in completed_agents:
            return name
    return "DONE"


def decide_next_agent(topic: str, completed_agents: list[str], latest_data: dict | None) -> str:
    """决定下一步调用的 Agent 名，或 "DONE"。"""
    available = discover_all()
    if not available:
        return _fallback_next(completed_agents)

    candidates = [name for name in available if name not in completed_agents]
    if not candidates:
        return "DONE"

    skills_desc = "\n".join(
        f"- {name}：{card.description or ''}" for name, card in available.items()
    )
    prompt = (
        "你是多智能体编排器（Host Agent），需要决定内容生成链的下一步。\n\n"
        f"目标：围绕主题「{topic}」，依次调用合适的 Agent，最终产出完整文章（大纲→正文→摘要）。\n\n"
        f"可用 Agent：\n{skills_desc}\n\n"
        f"已完成步骤：{'、'.join(completed_agents) if completed_agents else '无'}\n"
        f"最近产出：{_short_summary(latest_data)}\n\n"
        f"请从【{'、'.join(candidates)}、DONE】中选择下一步，只输出这一个词，不要输出其它任何内容。"
    )

    try:
        raw = _call_llm(prompt)
    except Exception:
        return _fallback_next(completed_agents)

    choice = raw.strip()
    if not choice:
        return _fallback_next(completed_agents)
    if "DONE" in choice and not any(name in choice for name in candidates):
        return "DONE"
    for name in candidates:
        if name in choice:
            return name
    return _fallback_next(completed_agents)
