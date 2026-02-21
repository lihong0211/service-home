"""
MCP PPT 助手：通过 MCP 接入 ChatPPT 官方服务（YOOTeam/ChatPPT-MCP）
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, List, Optional

import dashscope
from qwen_agent.agents import Assistant

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

DEFAULT_CHATPPT_MCP_URL = "http://mcp.yoo-ai.com/mcp"
YOO_API_KEY = os.getenv("YOO_API_KEY")


ROLE = "role"
CONTENT = "content"
NAME = "name"
ASSISTANT = "assistant"
FUNCTION = "function"
USER = "user"

_bot: Optional[Any] = None


def _build_chatppt_mcp_config() -> Optional[dict]:
    url = f"{DEFAULT_CHATPPT_MCP_URL}?key={YOO_API_KEY}"

    return {
        "mcpServers": {
            "chatppt": {
                "type": "streamable-http",
                "url": url,
                "sse_read_timeout": 300,
            }
        }
    }


def init_agent_service():
    mcp_config = _build_chatppt_mcp_config()
    llm_cfg = {
        "model": "qwen-max",
        "timeout": 60,
        "retry_count": 3,
    }
    system = (
        "你是 PPT 汇报助手，已接入 ChatPPT MCP 服务。"
        "你可以根据用户主题或需求生成 PPT，也支持上传文档自动生成演示文稿、在线编辑与下载。"
        "当用户说「基于本月销售数据做个汇报PPT」或类似需求时，使用 MCP 提供的工具生成或优化 PPT，并告知用户如何查看或下载。"
    )
    tools = [mcp_config]
    bot = Assistant(
        llm=llm_cfg,
        name="PPT 汇报助手",
        description="接入 ChatPPT MCP，根据主题或文档生成、编辑与下载 PPT。可接入活字格、阿里云百炼等。",
        system_message=system,
        function_list=tools,
    )
    return bot


def _get_bot() -> Any:
    global _bot
    if _bot is None:
        _bot = init_agent_service()
    return _bot


def _message_to_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        role = msg.get(ROLE, "")
        content = msg.get(CONTENT, "")
        name = msg.get(NAME)
        fn_call = msg.get("function_call")
    else:
        role = getattr(msg, ROLE, "")
        content = getattr(msg, CONTENT, "")
        name = getattr(msg, NAME, None)
        fn_call = getattr(msg, "function_call", None)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                text_parts.append(item.text or "")
            else:
                text_parts.append(str(item))
        content = " ".join(text_parts)
    if content is None:
        content = ""
    out = {ROLE: role, CONTENT: content}
    if name:
        out[NAME] = name
    if fn_call:
        if hasattr(fn_call, "name"):
            out["function_call"] = {
                "name": fn_call.name,
                "arguments": getattr(fn_call, "arguments", "{}"),
            }
        else:
            out["function_call"] = (
                fn_call
                if isinstance(fn_call, dict)
                else {"name": "", "arguments": "{}"}
            )
    return out


def get_mcp_ppt_info() -> dict:
    """返回 PPT 助手元信息与 MCP 插件列表。未配置或连接失败时返回配置说明。"""
    try:
        bot = _get_bot()
        plugins = (
            list(bot.function_map.keys()) if getattr(bot, "function_map", None) else []
        )
        return {
            "name": getattr(bot, "name", "PPT 汇报助手"),
            "description": getattr(
                bot,
                "description",
                "接入 ChatPPT MCP，根据主题或文档生成、编辑与下载 PPT。",
            ),
            "plugins": plugins,
            "mcp_server": "chatppt",
        }
    except ValueError as e:
        return {
            "name": "PPT 汇报助手",
            "description": "需配置 ChatPPT MCP 后使用。",
            "plugins": [],
            "mcp_server": None,
            "config_required": True,
            "config_hint": str(e),
        }
    except Exception as e:
        return {
            "name": "PPT 汇报助手",
            "description": "需配置 ChatPPT MCP 后使用。",
            "plugins": [],
            "mcp_server": None,
            "config_required": True,
            "config_hint": str(e),
        }


def run_mcp_ppt_chat_stream(messages: List[dict]) -> Iterator[str]:
    """流式返回与 ChatPPT MCP 的对话步骤（NDJSON）。"""

    def send(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    try:
        bot = _get_bot()
    except ValueError as e:
        yield send({"event": "error", "data": {"message": str(e)}})
        return

    run_messages = [dict(m) for m in messages]
    try:
        for response in bot.run(run_messages):
            if not response:
                continue
            current_messages = [_message_to_dict(m) for m in response]
            yield send({"event": "step", "data": current_messages})
    except Exception as e:
        yield send({"event": "error", "data": {"message": str(e)}})


def run_mcp_ppt_chat(
    messages: List[dict],
    model: str = "qwen-turbo",
    system_message: Optional[str] = None,
) -> dict:
    """
    执行与 ChatPPT MCP 的对话，收集全部步骤并返回。
    model / system_message 当前由 Assistant 初始化固定，保留参数以兼容路由。
    """
    try:
        bot = _get_bot()
    except ValueError as e:
        return {
            "error": str(e),
            "reply_messages": [],
            "steps": [],
            "history": [],
            "final_answer": "",
        }

    run_messages = [dict(m) for m in messages]
    steps = []
    final_answer = ""
    history = []
    try:
        for response in bot.run(run_messages):
            if not response:
                continue
            current = [_message_to_dict(m) for m in response]
            steps.append({"type": "step", "data": current})
            history = current
        for msg in reversed(history or []):
            if msg.get(ROLE) == ASSISTANT and msg.get(CONTENT):
                final_answer = (msg.get(CONTENT) or "").strip()
                break
    except Exception as e:
        return {
            "error": str(e),
            "reply_messages": [],
            "steps": steps,
            "history": history,
            "final_answer": final_answer,
        }

    reply_messages = [m for m in history if m.get(ROLE) in (ASSISTANT, USER)]
    return {
        "reply_messages": reply_messages,
        "steps": steps,
        "history": history,
        "final_answer": final_answer,
    }


# ---------- Flask 视图（供 routes 直接注册）----------

def mcp_ppt_info_api():
    from flask import jsonify
    try:
        data = get_mcp_ppt_info()
        return jsonify({"code": 0, "msg": "ok", "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def mcp_ppt_chat_api():
    from flask import request, jsonify
    body = request.get_json() or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"code": 400, "msg": "请提供 messages", "data": None}), 400
    try:
        out = run_mcp_ppt_chat(
            messages,
            model=body.get("model", "qwen-turbo"),
            system_message=body.get("system_message"),
        )
        if out.get("error"):
            return jsonify({"code": 500, "msg": out["error"], "data": out}), 500
        return jsonify({"code": 0, "msg": "ok", "data": out})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def mcp_ppt_chat_stream_api():
    from flask import request, jsonify, Response, stream_with_context
    body = request.get_json() or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"code": 400, "msg": "请提供 messages", "data": None}), 400
    try:
        def generate():
            for line in run_mcp_ppt_chat_stream(messages):
                yield line
        return Response(
            stream_with_context(generate()),
            mimetype="application/x-ndjson",
            headers={"X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500
