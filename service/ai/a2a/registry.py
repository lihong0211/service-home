"""
Agent 注册表：集中管理已知 Agent 的 base URL，并通过
GET /.well-known/agent.json 做能力发现（AgentCard 带 skills 描述），
供 Host Agent 做路由决策使用。

新增 Agent 只需在 AGENT_BASE_URLS 里加一行，无需改动路由/编排逻辑。
"""
from __future__ import annotations

import requests

from service.ai.a2a.schemas import AgentCard

AGENT_BASE_URLS = {
    "StyleAgent": "http://127.0.0.1:8004",
    "OutlineAgent": "http://127.0.0.1:8001",
    "DocAgent": "http://127.0.0.1:8002",
    "SummaryAgent": "http://127.0.0.1:8003",
}

_card_cache: dict[str, AgentCard] = {}


def agent_url(agent_name: str) -> str:
    return AGENT_BASE_URLS[agent_name]


def discover(agent_name: str, force: bool = False) -> AgentCard:
    """拉取并缓存指定 Agent 的名片；force=True 时跳过缓存重新拉取。"""
    if not force and agent_name in _card_cache:
        return _card_cache[agent_name]
    r = requests.get(f"{agent_url(agent_name)}/.well-known/agent.json", timeout=5)
    r.raise_for_status()
    card = AgentCard(**r.json())
    _card_cache[agent_name] = card
    return card


def discover_all() -> dict[str, AgentCard]:
    """发现所有已知 Agent 的名片；单个 Agent 不可达时跳过，不影响其它。"""
    cards: dict[str, AgentCard] = {}
    for name in AGENT_BASE_URLS:
        try:
            cards[name] = discover(name)
        except Exception:
            continue
    return cards
