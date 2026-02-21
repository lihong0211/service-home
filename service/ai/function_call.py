"""
Function Calling：天气查询等工具调用，多轮对话直到模型返回最终文本。
"""

import json
import os
import dashscope
import requests

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
_GAODE_API_KEY = os.getenv("AMAP_MAPS_API_KEY")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 北京",
                },
                "adcode": {
                    "type": "string",
                    "description": "The city code, e.g. 110000 (北京)",
                },
            },
            "required": ["location"],
        },
    },
}


def get_weather_from_gaode(location: str, adcode: str = None):
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {"key": _GAODE_API_KEY, "city": adcode or location, "extensions": "base"}
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def _run_tool(name: str, arguments: dict):
    """执行工具并返回可序列化为字符串的结果。"""
    if name == "get_current_weather":
        location = arguments.get("location", "")
        adcode = arguments.get("adcode")
        out = get_weather_from_gaode(location, adcode)
        return json.dumps(out, ensure_ascii=False)
    return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)


def run_function_calling_chat(
    messages: list,
    model: str = "qwen-turbo",
    system_message: str = None,
    max_iterations: int = 5,
):
    """
    多轮对话 + 工具调用：有 tool_calls 就执行工具、写回，再请求下一轮，直到模型返回最终 content。
    """
    if system_message:
        messages = [{"role": "system", "content": system_message}] + list(messages)
    current = [dict(m) for m in messages]
    final_content = ""

    for _ in range(max_iterations):
        resp = dashscope.Generation.call(
            model=model,
            messages=current,
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        )
        if getattr(resp, "status_code", 0) != 200:
            return {"error": getattr(resp, "message", None) or getattr(resp, "code", "请求失败")}
        if not resp or not getattr(resp, "output", None) or not getattr(resp.output, "choices", None):
            return {"error": getattr(resp, "message", None) or "服务返回异常"}
        msg = resp.output.choices[0].message
        content = getattr(msg, "content", None) or ""
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            final_content = (content or "").strip()
            break

        # 有 tool_calls：先追加 assistant 消息，再逐条执行工具并追加 function 消息，下一轮继续请求
        tcs_for_history = []
        function_messages = []
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
            if isinstance(fn, dict):
                fn_name = fn.get("name", "")
                fn_args_str = fn.get("arguments", "{}")
            else:
                fn_name = getattr(fn, "name", "")
                fn_args_str = getattr(fn, "arguments", "{}")
            try:
                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else (fn_args_str or {})
            except Exception:
                fn_args = {}
            result = _run_tool(fn_name, fn_args)
            tcs_for_history.append(
                tc if isinstance(tc, dict) else {
                    "id": getattr(tc, "id", ""),
                    "type": "function",
                    "function": {"name": fn_name, "arguments": fn_args_str},
                }
            )
            function_messages.append({"role": "function", "name": fn_name, "content": result})
        current.append({"role": "assistant", "content": content or "", "tool_calls": tcs_for_history})
        current.extend(function_messages)

    return final_content


def get_function_calling_info():
    fn = WEATHER_TOOL["function"]
    return {
        "name": "Function Calling",
        "description": "天气查询小助手",
        "tools": [
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        ],
    }


# ---------- 两个 HTTP 接口（给 routes 注册）----------


def function_calling_info_api():
    from flask import jsonify

    try:
        return jsonify({"code": 0, "msg": "ok", "data": get_function_calling_info()})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def function_calling_chat_api():
    from flask import request, jsonify

    body = request.get_json() or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"code": 400, "msg": "请提供 messages"}), 400
    try:
        out = run_function_calling_chat(
            messages,
            model=body.get("model", "qwen-turbo"),
            system_message=body.get("system_message"),
            max_iterations=5,
        )
        if isinstance(out, dict) and out.get("error"):
            return jsonify({"code": 500, "msg": out["error"]}), 500
        return jsonify({"code": 0, "msg": "ok", "data": out})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500
