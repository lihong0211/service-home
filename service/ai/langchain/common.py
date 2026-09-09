"""公共 LLM 调用 helper（Qwen via Dashscope）+ 高德地图 helper + 通用历史格式化。"""

import json
import logging
import os

import dashscope
import requests

from config.ai import DEFAULT_CHAT_MODEL

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
_GAODE_API_KEY = os.getenv("AMAP_MAPS_API_KEY")
logger = logging.getLogger(__name__)

# 多轮对话：保留的轮数/条数上限（真正意义的长对话）
MAX_HISTORY_MESSAGES = 50   # router 闲聊等传入 LLM 的最近消息条数（约 25 轮）
MAX_HISTORY_TURNS_CONTEXT = 20  # 拼进 prompt 的「上文」最近轮数（think/respond/parallel）


def _call_llm_messages(messages: list, model: str = DEFAULT_CHAT_MODEL) -> str:
    """多轮对话：messages 为 [{"role":"system"|"user"|"assistant", "content": "..."}, ...]，返回最后一轮 assistant 回复。"""
    if not messages:
        return ""
    resp = dashscope.Generation.call(model=model, messages=messages)
    if getattr(resp, "status_code", None) != 200:
        return ""
    output = getattr(resp, "output", None)
    if not output:
        return ""
    text = getattr(output, "text", None)
    if text is not None and str(text).strip():
        return str(text).strip()
    try:
        choices = getattr(output, "choices", None) or []
        if choices and len(choices) > 0:
            msg = getattr(choices[0], "message", None)
            if msg:
                content = getattr(msg, "content", None)
                if content is not None:
                    return str(content).strip()
    except Exception:
        pass
    return ""


def _call_llm_messages_stream(messages: list, model: str = DEFAULT_CHAT_MODEL):
    """
    与 _call_llm_messages 等价，但用 DashScope 原生流式接口（stream=True, incremental_output=True）逐段 yield 增量文本。
    DashScope 不是 LangChain 聊天模型，接不上 LangGraph 的 stream_mode="messages"，所以用「与任何 LLM 一起使用」
    的模式：节点内部自己消费这个生成器、通过 get_stream_writer() 转发，而不是让 LangGraph 直接理解 DashScope 的输出。
    """
    if not messages:
        return
    responses = dashscope.Generation.call(
        model=model,
        messages=messages,
        stream=True,
        incremental_output=True,
    )
    for resp in responses:
        if getattr(resp, "status_code", None) != 200:
            continue
        output = getattr(resp, "output", None)
        if not output:
            continue
        text = getattr(output, "text", None)
        if text:
            yield str(text)
            continue
        choices = getattr(output, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            content = getattr(msg, "content", None) if msg else None
            if content:
                yield str(content)


def _call_llm(prompt: str, system: str = "你是一个专业的AI助手，请简洁准确地回答。", model: str = DEFAULT_CHAT_MODEL) -> str:
    """调用 Qwen 大模型，返回纯文本结果。兼容 output.choices 与 output.text 两种返回格式。"""
    resp = dashscope.Generation.call(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    if getattr(resp, "status_code", None) == 200:
        output = getattr(resp, "output", None)
        if not output:
            return ""
        # Dashscope 可能返回 output.text（choices 为 null）或 output.choices[0].message.content
        text = getattr(output, "text", None)
        if text is not None and str(text).strip():
            return str(text).strip()
        try:
            choices = getattr(output, "choices", None) or []
            if choices and len(choices) > 0:
                msg = getattr(choices[0], "message", None)
                if msg:
                    content = getattr(msg, "content", None)
                    if content is not None:
                        return str(content).strip()
        except Exception as e:
            print(f"   [LLM] 解析 choices 失败: {e}")
    code = getattr(resp, "code", "")
    msg = getattr(resp, "message", "") or getattr(resp, "msg", "")
    status = getattr(resp, "status_code", "")
    print(f"   [LLM Error] status={status} code={code} message={msg}")
    return ""


def _gaode_geocode_adcode(city: str) -> str | None:
    """
    用高德地理编码 API 把城市名转成市级 adcode。
    geocode 返回的是区级 adcode（如 110101），截成前 4 位 + '00' 得市级（110100）。
    """
    try:
        r = requests.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params={"key": _GAODE_API_KEY, "address": city, "output": "JSON"},
            timeout=10,
        )
        data = r.json()
        geocodes = data.get("geocodes") or []
        if geocodes:
            adcode = geocodes[0].get("adcode", "")
            if adcode and len(adcode) == 6:
                return adcode[:4] + "00"  # 区级 → 市级
            return adcode or None
    except Exception:
        pass
    return None



def _get_gaode_weather(adcode: str) -> dict:
    """用 adcode 查高德实时天气。"""
    if not _GAODE_API_KEY:
        return {"error": "未配置 AMAP_MAPS_API_KEY"}
    try:
        r = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={"key": _GAODE_API_KEY, "city": adcode, "extensions": "base"},
            timeout=10,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        data = r.json()
        print(f"   高德天气原始响应: {json.dumps(data, ensure_ascii=False)[:300]}")
        return data
    except Exception as e:
        return {"error": str(e)}

def _format_history_context(history: list, max_turns: int | None = None, max_chars_per_msg: int = 300) -> str:
    """把 history 格式化为「上文」文本，供 loop/parallel 节点拼进 prompt（简单截断，非生产级——
    生产级的滑动窗口+摘要压缩实现见上面的 _build_context_window，router 图 chat 节点专用）。
    默认保留最近 MAX_HISTORY_TURNS_CONTEXT 轮。"""
    if not history or not isinstance(history, list):
        return ""
    turns = max_turns if max_turns is not None else MAX_HISTORY_TURNS_CONTEXT
    lines = []
    for h in history[-turns * 2 :]:
        role = (h.get("role") or "").lower()
        content = (h.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        lines.append(f"{'用户' if role == 'user' else '助手'}: {content[:max_chars_per_msg]}")
    return "\n".join(lines) if lines else ""
