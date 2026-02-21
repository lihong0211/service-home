"""
MCP STT 助手：通过 MCP 接入 DashScope 千问语音识别（SpeechToText）

配置（环境变量）：
- MCP_STT_URL: MCP 服务地址，默认百炼 SpeechToText SSE
- MCP_STT_API_KEY 或 DASHSCOPE_API_KEY: Bearer 鉴权，用于 headers.Authorization
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, List, Optional

import dashscope
from qwen_agent.agents import Assistant

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

MCP_STT_URL = os.getenv(
    "MCP_STT_URL",
    "https://dashscope.aliyuncs.com/api/v1/mcps/SpeechToText/sse",
).strip()
MCP_STT_API_KEY = os.getenv("MCP_STT_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""

ROLE = "role"
CONTENT = "content"
NAME = "name"
ASSISTANT = "assistant"
USER = "user"

_bot: Optional[Any] = None


def _build_stt_mcp_config() -> Optional[dict]:
    """构建 STT MCP 配置。如果 URL 为空则返回 None。"""
    if not MCP_STT_URL or not MCP_STT_URL.strip():
        return None
    cfg = {
        "mcpServers": {
            "stt": {
                "type": "streamable-http",
                "url": MCP_STT_URL.strip(),
                "sse_read_timeout": 120,
            }
        }
    }
    if MCP_STT_API_KEY and MCP_STT_API_KEY.strip():
        cfg["mcpServers"]["stt"]["headers"] = {
            "Authorization": f"Bearer {MCP_STT_API_KEY.strip()}"
        }
    return cfg


def init_agent_service():
    mcp_config = _build_stt_mcp_config()
    if not mcp_config:
        raise ValueError("MCP_STT_URL 未配置，请设置环境变量 MCP_STT_URL 或使用默认值")
    if not MCP_STT_API_KEY:
        raise ValueError("MCP_STT_API_KEY 或 DASHSCOPE_API_KEY 未配置，请设置环境变量")

    llm_cfg = {
        "model": "qwen-max",
        "timeout": 60,
        "retry_count": 5,
    }
    system = (
        "你是语音识别助手，已接入千问 STT MCP 服务。"
        "根据用户需求将语音转为文字，或推荐合适的识别参数、语言设置。"
    )
    tools = [mcp_config]
    try:
        bot = Assistant(
            llm=llm_cfg,
            name="STT 语音识别助手",
            description="接入 DashScope 千问 STT MCP，支持语音转文字。",
            system_message=system,
            function_list=tools,
        )
        return bot
    except Exception as e:
        error_msg = str(e)
        import traceback
        full_traceback = traceback.format_exc()

        if "TaskGroup" in error_msg or "sub-exception" in error_msg:
            detailed_error = error_msg
            if "sub-exception" in full_traceback:
                lines = full_traceback.split('\n')
                for i, line in enumerate(lines):
                    if "Exception" in line or "Error" in line:
                        detailed_error = line.strip()
                        break

            raise ValueError(
                f"MCP 服务连接失败。\n"
                f"URL: {MCP_STT_URL}\n"
                f"API_KEY: {'已设置（长度: ' + str(len(MCP_STT_API_KEY)) + '）' if MCP_STT_API_KEY else '未设置'}\n"
                f"错误详情: {detailed_error}\n"
                f"请检查：\n"
                f"1. API_KEY 是否有效（可在 https://dashscope.console.aliyun.com/ 查看）\n"
                f"2. URL 是否正确（当前: {MCP_STT_URL}）\n"
                f"3. 网络连接是否正常"
            )
        raise ValueError(f"初始化 STT MCP 失败: {error_msg}\n完整错误: {full_traceback}")


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


def get_mcp_stt_info() -> dict:
    """返回 STT 助手元信息与 MCP 插件列表。未配置或连接失败时返回配置说明。"""
    url_status = "已配置" if MCP_STT_URL else "未配置"
    api_key_status = "已配置" if MCP_STT_API_KEY else "未配置"

    if not MCP_STT_URL or not MCP_STT_API_KEY:
        return {
            "name": "STT 语音识别助手",
            "description": "需配置 STT MCP 后使用。",
            "plugins": [],
            "mcp_server": None,
            "config_required": True,
            "config_hint": f"配置不完整（当前状态: URL={url_status}, API_KEY={api_key_status}）。请设置 MCP_STT_URL 和 DASHSCOPE_API_KEY（或 MCP_STT_API_KEY）",
            "config_status": {
                "url_configured": bool(MCP_STT_URL),
                "api_key_configured": bool(MCP_STT_API_KEY),
                "url_example": "https://dashscope.aliyuncs.com/api/v1/mcps/SpeechToText/sse",
            },
        }

    try:
        import threading
        import queue

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def get_bot_info():
            try:
                bot = _get_bot()
                plugins = (
                    list(bot.function_map.keys()) if getattr(bot, "function_map", None) else []
                )
                result_queue.put({
                    "name": getattr(bot, "name", "STT 语音识别助手"),
                    "description": getattr(
                        bot, "description", "接入 DashScope 千问 STT MCP，支持语音转文字。"
                    ),
                    "plugins": plugins,
                    "mcp_server": "stt",
                })
            except Exception as e:
                exception_queue.put(e)

        thread = threading.Thread(target=get_bot_info, daemon=True)
        thread.start()
        thread.join(timeout=3)

        if thread.is_alive():
            raise TimeoutError("MCP 服务连接超时（3秒）")

        if not exception_queue.empty():
            raise exception_queue.get()

        if not result_queue.empty():
            return result_queue.get()
        else:
            raise Exception("无法获取 MCP 服务信息")

    except (TimeoutError, ValueError) as e:
        error_msg = str(e)
        url_status = "已配置" if MCP_STT_URL else "未配置"
        api_key_status = "已配置" if MCP_STT_API_KEY else "未配置"

        if "超时" in error_msg or isinstance(e, TimeoutError):
            hint = f"MCP 服务连接超时（当前状态: URL={url_status}, API_KEY={api_key_status}）。可能原因：\n1. MCP 服务暂时不可用\n2. 网络连接问题\n3. URL 或 API_KEY 配置错误"
        elif "未配置" in error_msg:
            hint = (
                f"{error_msg}（当前状态: URL={url_status}, API_KEY={api_key_status}）"
            )
        elif "连接失败" in error_msg or "TaskGroup" in error_msg:
            hint = f"MCP 服务连接失败（当前状态: URL={url_status}, API_KEY={api_key_status}）。请检查：\n1. MCP_STT_URL 是否正确（默认: https://dashscope.aliyuncs.com/api/v1/mcps/SpeechToText/sse）\n2. DASHSCOPE_API_KEY 或 MCP_STT_API_KEY 是否已设置且有效"
        else:
            hint = (
                f"{error_msg}（当前状态: URL={url_status}, API_KEY={api_key_status}）"
            )
        return {
            "name": "STT 语音识别助手",
            "description": "需配置 STT MCP 后使用。",
            "plugins": [],
            "mcp_server": None,
            "config_required": True,
            "config_hint": hint,
            "config_status": {
                "url_configured": bool(MCP_STT_URL),
                "api_key_configured": bool(MCP_STT_API_KEY),
                "url_example": "https://dashscope.aliyuncs.com/api/v1/mcps/SpeechToText/sse",
            },
        }
    except Exception as e:
        error_msg = str(e)
        url_status = "已配置" if MCP_STT_URL else "未配置"
        api_key_status = "已配置" if MCP_STT_API_KEY else "未配置"

        if "超时" in error_msg or "timeout" in error_msg.lower():
            hint = f"MCP 服务连接超时（当前状态: URL={url_status}, API_KEY={api_key_status}）。可能原因：\n1. MCP 服务暂时不可用\n2. 网络连接问题\n3. URL 或 API_KEY 配置错误"
        elif "TaskGroup" in error_msg or "sub-exception" in error_msg:
            hint = f"MCP 服务连接失败（当前状态: URL={url_status}, API_KEY={api_key_status}）。请检查：\n1. MCP_STT_URL 是否正确\n2. DASHSCOPE_API_KEY 或 MCP_STT_API_KEY 是否已设置且有效\n3. 网络连接是否正常"
        else:
            hint = f"初始化失败: {error_msg}（当前状态: URL={url_status}, API_KEY={api_key_status}）"
        return {
            "name": "STT 语音识别助手",
            "description": "需配置 STT MCP 后使用。",
            "plugins": [],
            "mcp_server": None,
            "config_required": True,
            "config_hint": hint,
            "config_status": {
                "url_configured": bool(MCP_STT_URL),
                "api_key_configured": bool(MCP_STT_API_KEY),
                "url_example": "https://dashscope.aliyuncs.com/api/v1/mcps/SpeechToText/sse",
            },
        }


def run_mcp_stt_chat_stream(messages: List[dict]) -> Iterator[str]:
    def send(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    try:
        bot = _get_bot()
    except (ValueError, Exception) as e:
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


def run_mcp_stt_chat(
    messages: List[dict],
    model: str = "qwen-turbo",
    system_message: Optional[str] = None,
) -> dict:
    try:
        bot = _get_bot()
    except (ValueError, Exception) as e:
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


def mcp_stt_info_api():
    from flask import jsonify

    try:
        data = get_mcp_stt_info()
        return jsonify({"code": 0, "msg": "ok", "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def mcp_stt_chat_api():
    from flask import request, jsonify

    body = request.get_json() or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"code": 400, "msg": "请提供 messages", "data": None}), 400
    try:
        out = run_mcp_stt_chat(
            messages,
            model=body.get("model", "qwen-turbo"),
            system_message=body.get("system_message"),
        )
        if out.get("error"):
            return jsonify({"code": 500, "msg": out["error"], "data": out}), 500
        return jsonify({"code": 0, "msg": "ok", "data": out})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def mcp_stt_chat_stream_api():
    from flask import request, jsonify, Response, stream_with_context

    body = request.get_json() or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"code": 400, "msg": "请提供 messages", "data": None}), 400
    try:

        def generate():
            for line in run_mcp_stt_chat_stream(messages):
                yield line

        return Response(
            stream_with_context(generate()),
            mimetype="application/x-ndjson",
            headers={"X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500
