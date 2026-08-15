# service/ai/langchain.py
"""
LangGraph 核心功能可视化演示

演示内容：
1. 循环与分支 - 基础功能（think/decide 循环）
2. 并行执行 - 多分支汇聚（情感/关键词/摘要 → 聚合）
3. 状态管理 - MemorySaver 持久化与恢复
4. 条件路由 - 意图识别与多路分发
5. 人机交互节点 - AI 建议 → 人工审核 → 处理反馈
6. 实时执行监控 - stream 可视化与简单仪表盘
"""

from __future__ import annotations

import json
import logging
import operator
import os
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Annotated, Literal, TypedDict

import dashscope
import requests

import anyio.from_thread
from fastapi import Request
from fastapi.responses import StreamingResponse

from utils.http_body import query_dict, read_json_optional
from config.ai import DEFAULT_CHAT_MODEL

# LangGraph 图与状态
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, RetryPolicy, interrupt

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
_GAODE_API_KEY = os.getenv("AMAP_MAPS_API_KEY")
logger = logging.getLogger(__name__)

# 多轮对话：保留的轮数/条数上限（真正意义的长对话）
MAX_HISTORY_MESSAGES = 50   # router 闲聊等传入 LLM 的最近消息条数（约 25 轮）
MAX_HISTORY_TURNS_CONTEXT = 20  # 拼进 prompt 的「上文」最近轮数（think/respond/parallel）


# ---------------------------------------------------------------------------
# 公共 LLM 调用 helper（Qwen via Dashscope）
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 1. 循环与分支 - 基础功能
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str
    iteration: int
    query: str  # 用户问题，供 _think 做针对性多轮推理；前端 input 合并时会带入
    response: str  # 最终回答，供前端展示（由 respond 节点填充）


# ---------------------------------------------------------------------------
# 【模块 1／8：上下文管理】生产级实现（router 图 chat 节点专用，见 _chat_handler）。
# 完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 「上下文管理」一节。
# 核心是三件事，缺一不可：
#   1. 按真实 token 预算做滑动窗口（_count_tokens/_build_context_window），不是拍脑袋的
#      消息条数或字符数——短消息浪费预算、长消息可能悄悄超预算，条数/字符数都不可靠。
#   2. 滑出窗口的旧对话增量压缩成摘要（_summarize_messages），不是直接丢弃——否则对话
#      早期说过的事就是真的没了，用户体验上表现为"AI 突然失忆"。
#   3. 用 checkpointer 让服务端持有会话状态（见 build_router_graph 的 checkpointer 参数），
#      不是靠前端每次把完整历史重新传一遍——那样带宽浪费、且没有服务端统一压缩的落脚点。
# loop/parallel 两个图是单轮演示图（没有真正的多轮对话场景），仍用下面这个轻量的
# _format_history_context 做简单截断，没有接入 token 预算/摘要压缩，这是有意的范围收窄。
# ---------------------------------------------------------------------------

MAX_CONTEXT_TOKENS = 2000  # chat 节点历史对话的 token 预算（不含 system prompt 和当前这句问题本身）

_TOKENIZER = None  # 惰性初始化的 DashScope 本地分词器，模块内只建一次


def _count_tokens(text: str) -> int:
    """
    【上下文管理】用 DashScope 本地分词器精确计 token 数——完全离线、不发网络请求，
    且和线上实际调用的模型（DEFAULT_CHAT_MODEL）分词规则一致，比按字符数估算准确得多。
    分词器初始化/编码失败时降级为「中文约2字/token」的粗略估算，不让主流程因此报错。
    """
    global _TOKENIZER
    if not text:
        return 0
    try:
        if _TOKENIZER is None:
            _TOKENIZER = dashscope.get_tokenizer(DEFAULT_CHAT_MODEL)
        return len(_TOKENIZER.encode(text))
    except Exception:
        logger.exception("token 计数失败，降级为字符数估算")
        return len(text) // 2


def _summarize_messages(messages: list, prior_summary: str) -> str:
    """
    【上下文管理】增量摘要压缩：只摘要"这一轮新滑出窗口"的那一小段旧对话，叠加在已有摘要之上，
    而不是每次把全部历史重新摘要一遍——否则对话轮数一多，摘要本身也会变成新的延迟/成本瓶颈。
    摘要调用失败时保留旧摘要、不中断主流程：上下文退化但不影响当前这轮的正常回答。
    """
    if not messages:
        return prior_summary
    text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}" for m in messages
    )
    prompt = (
        (f"已有摘要：{prior_summary}\n\n" if prior_summary else "")
        + "以下是新滑出对话窗口、即将不再原样保留的历史内容，请把它和已有摘要合并，"
        "输出一段更新后的摘要（保留关键事实/结论/用户偏好，控制在150字以内，只输出摘要正文）：\n\n"
        + text
    )
    try:
        return _call_llm(prompt, system="你是对话历史摘要助手，只输出摘要正文，不要解释、不要标题。").strip()
    except Exception:
        logger.exception("历史摘要压缩失败，本轮保留旧摘要")
        return prior_summary


def _build_context_window(
    messages: list, prior_summary: str, max_tokens: int = MAX_CONTEXT_TOKENS
) -> tuple[str, list]:
    """
    【上下文管理】滑动窗口核心函数：从最新消息往前数，累计 token 数不超过 max_tokens 的部分
    整段保留原文（recent，进 prompt 时逐字还原，模型能看到原始措辞）；一旦超出预算，更早的
    部分整体压缩进摘要（summary），而不是简单截断丢弃——这是和旧版 _format_history_context
    （拍脑袋数轮次/数字符）最本质的区别。
    返回 (更新后的摘要, 保留的原文消息列表)；调用方把两者一起拼进最终 prompt。
    """
    if not messages:
        return prior_summary, []
    kept: list = []
    used_tokens = 0
    cut_index = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = _count_tokens(messages[i].get("content", ""))
        if used_tokens + msg_tokens > max_tokens and kept:
            cut_index = i + 1
            break
        used_tokens += msg_tokens
        kept.append(messages[i])
        cut_index = i
    kept.reverse()
    to_summarize = messages[:cut_index]
    if not to_summarize:
        return prior_summary, kept
    return _summarize_messages(to_summarize, prior_summary), kept


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


def _think(state: AgentState) -> dict:
    iteration = state["iteration"]
    print(f"🤔 思考中... (第{iteration}轮)")
    prior = "；".join(state["messages"][-3:]) if state["messages"] else "无"
    user_query = (state.get("query") or "").strip()
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = (
        f"这是第 {iteration + 1} 轮思考。"
        f"前几轮已得到：{prior}。"
        "请用一句话给出新的思考或推进，不超过50字。"
    )
    if user_query:
        prompt = f"用户问题：{user_query}\n\n{prompt}"
    if hist_ctx:
        prompt = f"【多轮上文】\n{hist_ctx}\n\n{prompt}"
    thought = _call_llm(prompt, system="你是一个逻辑推理助手，围绕用户问题做多轮思考，每轮产生新的思考进展。")
    print(f"   💡 {thought}")
    return {
        "messages": [thought],
        "iteration": iteration + 1,
    }


def _decide(state: AgentState) -> dict:
    if state["iteration"] < 3:
        print("🔄 需要继续思考，进入循环")
        return {"next_step": "think"}
    print("✅ 思考完成，进入回答")
    return {"next_step": "respond"}


def _loop_respond(state: AgentState) -> dict:
    """根据多轮思考结果 + 用户问题，用 LLM 总结成最终回答。天气/股票等查数请走 router 图。"""
    query = (state.get("query") or "").strip()
    messages = state.get("messages") or []
    prior = "；".join(messages[-5:]) if messages else "无"
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = f"用户问题：{query}\n\n多轮思考要点：{prior}\n\n请用 2～4 句话给出直接、可操作的回答或结论，不要复述思考过程。"
    if hist_ctx:
        prompt = f"【多轮上文】\n{hist_ctx}\n\n{prompt}"
    response = _call_llm(prompt, system="你是助手，根据上述思考给出简洁结论或建议。")
    print(f"📢 最终回答: {(response or '')[:80]}...")
    return {"response": response or "暂无结论，请补充问题或换种问法。"}


def build_loop_graph():
    """
    创建带循环的图：think → decide → (think | respond) → respond → END。
    【模块 3／8：重试机制】think/respond 会调 LLM（外部网络请求），挂 retry_policy；decide 是纯
    逻辑判断、没有外部调用，不需要重试。全项目统一策略：只给"会打外部请求的节点"加重试，纯
    逻辑节点不加——重试解决的是网络抖动/限流这类瞬时故障，纯逻辑节点不存在这类故障源，加了
    也没意义。完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 「重试机制」一节。
    """
    builder = StateGraph(AgentState)
    builder.add_node("think", _think, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("decide", _decide)
    builder.add_node("respond", _loop_respond, retry_policy=RetryPolicy(max_attempts=3))
    builder.set_entry_point("think")
    builder.add_edge("think", "decide")
    builder.add_conditional_edges(
        "decide",
        lambda s: s["next_step"],
        {"think": "think", "respond": "respond"},
    )
    builder.add_edge("respond", END)
    return builder.compile()


def demo_loop():
    """演示循环流程图并打印 ASCII 图。"""
    graph = build_loop_graph()
    print("📊 **循环流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: think → decide → think 或 END)")
    print()
    # 执行一轮演示
    out = graph.invoke(
        {"messages": [], "next_step": "", "iteration": 0, "query": "示例问题", "response": ""}
    )
    print("最终状态 iteration:", out.get("iteration"), "response:", (out.get("response") or "")[:60])
    return graph


# ---------------------------------------------------------------------------
# 2. 并行执行 - 多分支汇聚（使用 Send 或顺序模拟）
# ---------------------------------------------------------------------------


class ParallelState(TypedDict):
    input_text: str
    analyses: Annotated[list, operator.add]  # 并行节点用 append 合并
    final_result: str
    response: str  # 供前端对话区展示，与 final_result 一致或为可读摘要


def _sentiment_analysis(state: ParallelState) -> dict:
    print("🔵 情感分析中...")
    text = state.get("input_text", "")
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = f"对以下文本做情感分析，只返回：positive / negative / neutral 之一。\n\n文本：{text}"
    if hist_ctx:
        prompt = f"【上文参考】\n{hist_ctx}\n\n{prompt}"
    result = _call_llm(
        prompt,
        system="你是情感分析专家，只输出 positive、negative 或 neutral。",
    )
    sentiment = result.strip().lower().split()[0] if result else "neutral"
    print(f"   情感: {sentiment}")
    return {"analyses": [("sentiment", sentiment)]}


def _keyword_extraction(state: ParallelState) -> dict:
    print("🟢 关键词提取中...")
    text = state.get("input_text", "")
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = f"从以下文本中提取3-5个关键词，以英文逗号分隔，只返回关键词列表，不要其他内容。\n\n文本：{text}"
    if hist_ctx:
        prompt = f"【上文参考】\n{hist_ctx}\n\n{prompt}"
    result = _call_llm(prompt, system="你是关键词提取专家。")
    keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
    print(f"   关键词: {keywords}")
    return {"analyses": [("keywords", keywords)]}


def _text_summary(state: ParallelState) -> dict:
    print("🟠 文本摘要中...")
    text = state.get("input_text", "")
    hist_ctx = _format_history_context(state.get("history") or [])
    prompt = f"用一句话（不超过30字）概括以下文本的核心内容：\n\n{text}"
    if hist_ctx:
        prompt = f"【上文参考】\n{hist_ctx}\n\n{prompt}"
    summary = _call_llm(prompt, system="你是专业的文本摘要助手。")
    print(f"   摘要: {summary}")
    return {"analyses": [("summary", summary)]}


def _aggregate_results(state: ParallelState) -> dict:
    print("📊 聚合所有分析结果")
    analyses = state.get("analyses") or []
    analysis_dict = dict(analyses) if analyses else {}
    final_result = f"综合结果：{analysis_dict}"
    # 供前端对话区展示的可读文案（关键词/情感/摘要一行一条）
    def _fmt(v):
        return ", ".join(v) if isinstance(v, (list, tuple)) else str(v)

    parts = []
    if "keywords" in analysis_dict:
        parts.append(f"关键词：{_fmt(analysis_dict['keywords'])}")
    if "sentiment" in analysis_dict:
        parts.append(f"情感：{_fmt(analysis_dict['sentiment'])}")
    if "summary" in analysis_dict:
        parts.append(f"摘要：{_fmt(analysis_dict['summary'])}")
    response = "\n".join(parts) if parts else final_result
    return {"final_result": final_result, "response": response}


def build_parallel_graph():
    """
    并行执行图：入口分发到 sentiment / keywords / summary，再汇聚到 aggregate。
    若当前环境不支持 Send，则用顺序边模拟（三节点依次执行后到 aggregate）。
    """
    builder = StateGraph(ParallelState)
    # 三个并行节点都调用 LLM，各自独立加重试；聚合节点是纯逻辑，不需要
    builder.add_node("sentiment", _sentiment_analysis, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("keywords", _keyword_extraction, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("summary", _text_summary, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("aggregate", _aggregate_results)

    try:
        from langgraph.types import Send

        def _dispatch(state: ParallelState):
            return [Send("sentiment", state), Send("keywords", state), Send("summary", state)]

        builder.add_node("dispatch", lambda s: s)  # 透传 state
        builder.set_entry_point("dispatch")
        # path_map 让 get_graph() 能静态解析出所有可能目标，
        # 实际路由仍由 Send 对象决定，两者互不干扰。
        builder.add_conditional_edges(
            "dispatch",
            _dispatch,
            {"sentiment": "sentiment", "keywords": "keywords", "summary": "summary"},
        )
        builder.add_edge("sentiment", "aggregate")
        builder.add_edge("keywords", "aggregate")
        builder.add_edge("summary", "aggregate")
    except ImportError:
        # 无 Send 时：顺序执行三节点再聚合
        builder.set_entry_point("sentiment")
        builder.add_edge("sentiment", "keywords")
        builder.add_edge("keywords", "summary")
        builder.add_edge("summary", "aggregate")

    builder.add_edge("aggregate", END)
    return builder.compile()


def demo_parallel():
    """演示并行（或顺序模拟）流程图。"""
    graph = build_parallel_graph()
    print("📊 **并行执行流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: dispatch → sentiment/keywords/summary → aggregate → END)")
    print()
    out = graph.invoke({"input_text": "示例文本", "analyses": [], "final_result": "", "response": ""})
    print("final_result:", out.get("final_result", "")[:80])
    return graph


# ---------------------------------------------------------------------------
# 3. 状态管理 - MemorySaver 持久化
# ---------------------------------------------------------------------------


class ConversationState(TypedDict):
    messages: list
    context: dict
    user_info: dict
    tokens_used: int


def _process_message(state: ConversationState) -> dict:
    new_message = f"处理消息 #{len(state['messages']) + 1}"
    print(f"💬 {new_message}")
    return {
        "messages": state["messages"] + [new_message],
        "tokens_used": state.get("tokens_used", 0) + 10,
    }


def build_state_mgmt_graph():
    """带 checkpoint 的图，用于演示状态恢复。"""
    builder = StateGraph(ConversationState)
    builder.add_node("process", _process_message)
    builder.set_entry_point("process")
    builder.add_edge("process", END)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def demo_state_management():
    """演示状态管理：同一 thread_id 下两次 invoke 会累积 messages。"""
    graph = build_state_mgmt_graph()
    print("📊 **状态管理演示**")
    config = {"configurable": {"thread_id": "demo-thread-1"}}
    initial = {"messages": [], "context": {}, "user_info": {}, "tokens_used": 0}
    out1 = graph.invoke(initial, config)
    print("第一次执行 messages:", out1.get("messages"), "tokens_used:", out1.get("tokens_used"))
    out2 = graph.invoke(initial, config)
    print("第二次执行（带历史）messages:", out2.get("messages"), "tokens_used:", out2.get("tokens_used"))
    return graph


# ---------------------------------------------------------------------------
# 4. 条件路由 - 意图识别与多路分发（chat 分支接入长期记忆 Store）
#
# 【模块 2／8：记忆管理】以下到 _maybe_write_long_term_memory 为止，是"跨会话记住关于
# 这个用户的事实/偏好"的完整实现——注意这和【模块 1／8：上下文管理】管的不是一回事：
# 上下文管理记的是"这一次会话聊了什么"（哪怕不带任何长期记忆，单次对话内也要连贯）；
# 记忆管理记的是"这个人是谁、喜欢什么"，换一个全新会话、隔了很久也该记得。完整设计说明
# 见 service/ai/AGENT_ARCHITECTURE.md 「记忆管理」一节。
# ---------------------------------------------------------------------------


def _memory_embed(texts: list[str]) -> list[list[float]]:
    """【记忆管理】长期记忆的语义索引向量化。复用 vector_db_qdrant 现成的 DashScope embedding，不重新实现一套。"""
    from service.ai.vector_db_qdrant import get_embedding

    return [get_embedding(t) for t in texts]


# 【记忆管理】长期记忆 Store：按 (user_id, "memories") 命名空间存事实型记忆，语义检索用同一套 embedding。
# 注意：InMemoryStore 是进程内存，和 HITL 最初用 MemorySaver 一样的问题——多 worker 部署下
# 各进程互不可见，记忆无法跨进程共享。这里保留 InMemoryStore（文档也是这么示例的，定位是
# demo/单进程场景），生产要用需要换成 DB-backed store（如自建基于 MySQL 的 BaseStore 实现）。
_LONG_TERM_STORE = InMemoryStore(index={"embed": _memory_embed, "dims": int(os.getenv("VECTOR_DB_DIMENSION", "1024"))})


class RouterState(TypedDict):
    query: str
    intent: str
    response: str
    # 【模块 1／8：上下文管理】以下两个字段配合 build_router_graph() 的 checkpointer 使用：
    # 有 checkpointer + 请求带 thread_id 时，LangGraph 会在同一 thread 的历次调用之间自动
    # 持久化并回填这两个字段，服务端因此"记住"了对话，不需要前端每次把全部历史重新传一遍。
    messages: Annotated[list, operator.add]  # 原始对话轮次，只增不减（reducer=append）
    context_summary: str  # 滑出窗口的旧对话被压缩成的摘要，每轮整体覆盖（reducer=最后写入生效）


def _classify_intent(state: RouterState) -> dict:
    query = (state.get("query") or "").strip()
    q = query.lower()
    if "天气" in q:
        intent = "weather"
    elif "新闻" in q:
        intent = "news"
    elif "分析" in q:
        intent = "insight"
    else:
        intent = "chat"
    print(f"🎯 意图识别: {intent}")
    return {"intent": intent}


def _analyze_via_subgraph(state: RouterState) -> dict:
    """
    子图组合演示：router 与 parallel 状态模式不同（query vs input_text），
    所以在节点函数内调用子图（而不是把 parallel 直接 add_node 挂进来）——
    先把 RouterState 转换成子图的 ParallelState，invoke 子图，再把子图输出转换回 RouterState.response。
    """
    query = (state.get("query") or "").strip()
    subgraph = build_parallel_graph()
    sub_output = subgraph.invoke({"input_text": query, "analyses": [], "final_result": "", "response": ""})
    return {"response": f"🧩 [子图: parallel] \n{sub_output.get('response', '')}"}


def _weather_handler(state: RouterState) -> dict:
    query = state.get("query", "")

    # LLM 从 query 提取城市名；没有城市则返回空字符串
    city = _call_llm(
        f"从下面这句话中提取城市名，只返回城市名本身（例如：上海）。如果没有提到城市，返回空字符串。\n\n句子：{query}",
        system="你只能输出城市名或空字符串，不要输出任何其他内容。",
    ).strip()

    print(f"🌤️ LLM 提取城市: {city!r}")

    if not city:
        return {"response": "请告诉我你想查哪个城市的天气，例如：上海今天天气怎么样？"}

    adcode = _gaode_geocode_adcode(city) or ""
    print(f"   高德 geocode: {city} → adcode: {adcode}")

    if not adcode:
        return {"response": f"未能识别城市「{city}」，请换个写法试试，例如直接写城市名：上海、北京。"}

    data = _get_gaode_weather(adcode)
    lives = data.get("lives") or []
    if data.get("status") == "1" and lives:
        live = lives[0]
        response = (
            f"☀️ {live.get('city', city)} 实时天气：{live.get('weather', '')}，"
            f"气温 {live.get('temperature', '')}°C，"
            f"湿度 {live.get('humidity', '')}%，"
            f"风向 {live.get('winddirection', '')} {live.get('windpower', '')} 级，"
            f"更新时间 {live.get('reporttime', '')}"
        )
    elif data.get("error"):
        response = f"天气查询失败：{data['error']}"
    else:
        response = f"天气查询失败（adcode={adcode} 未匹配）：{json.dumps(data, ensure_ascii=False)}"
    return {"response": response}


def _news_handler(state: RouterState) -> dict:
    """根据用户 query 选择信源：若问 AI/科技 则用科技 RSS 并用 LLM 筛选与问题相关的条目，否则用综合要闻。"""
    query = (state.get("query") or "").strip().lower()
    # 用户是否在问 AI/科技/互联网 等垂直领域新闻
    _tech_keywords = ("ai", "人工智能", "科技", "互联网", "技术", "大模型", "机器学习", "深度学习", "chatgpt", "gpt", "算法", "智能")
    want_tech = any(k in query for k in _tech_keywords)
    try:
        import feedparser
        if want_tech:
            # 科技/创投类 RSS，便于筛出与 query 相关的
            feed = feedparser.parse("https://36kr.com/feed", request_headers={"User-Agent": "Mozilla/5.0"})
            feed_label = "科技/AI"
        else:
            feed = feedparser.parse("https://rss.sina.com.cn/news/china/focus15.xml")
            feed_label = "今日要闻"
        entries = getattr(feed, "entries", None) or []
        if not entries:
            response = "📰 新闻获取失败（RSS 暂无条目）"
        else:
            candidates = [(getattr(e, "title", "") or "").strip() for e in entries[:15] if getattr(e, "title", None)]
            if want_tech and candidates and query:
                prompt = (
                    f"用户问题：{state.get('query', '')}\n\n"
                    "以下是一条条新闻标题，请只保留与用户问题**直接相关**的（如问 AI 就只保留 AI/人工智能/大模型等），按相关度排序，最多 5 条，每行一条，格式仅输出：\n• 标题1\n• 标题2\n不要解释、不要其他文字。"
                    + "\n".join(candidates)
                )
                filtered = _call_llm(prompt, system="你只输出筛选后的新闻列表，每行以 • 开头，不要其他内容。")
                if filtered and "•" in filtered:
                    response = f"📰 {feed_label}（与您问题相关）：\n" + filtered.strip()
                else:
                    top5 = [f"• {t}" for t in candidates[:5]]
                    response = f"📰 {feed_label}：\n" + "\n".join(top5)
            else:
                top5 = [f"• {t}" for t in candidates[:5]]
                response = f"📰 {feed_label}：\n" + "\n".join(top5)
    except Exception as e:
        response = f"📰 新闻获取失败：{e}"
    return {"response": response}


def _search_long_term_memory(store: BaseStore, user_id: str, query: str, limit: int = 3) -> list[str]:
    """语义检索该用户过去留下的事实型记忆，取 Top-K 拼进 prompt。"""
    if not query:
        return []
    try:
        hits = store.search((user_id, "memories"), query=query, limit=limit)
    except Exception as e:
        print(f"   [长期记忆] 检索失败: {e}")
        return []
    return [h.value.get("fact", "") for h in hits if h.value.get("fact")]


def _maybe_write_long_term_memory(store: BaseStore, user_id: str, query: str) -> None:
    """
    热路径写记忆：每轮对话后让 LLM 判断这句话里有没有值得长期记住的用户事实/偏好，
    有就写一条新记忆（集合式，而非维护单一概要），没有则什么都不做。
    """
    if not query:
        return
    extracted = _call_llm(
        f"用户说：{query}\n\n"
        "如果这句话包含值得长期记住的、关于用户本人的事实或偏好（例如姓名、职业、喜好、忌口、习惯性约束），"
        "用一句话提炼（不超过30字）；如果没有这类信息，只回复：无",
        system="你只输出提炼后的一句话事实，或「无」，不要解释、不要标点以外的其他内容。",
    ).strip()
    if not extracted or extracted in ("无", "无。", "没有", "没有。"):
        return
    try:
        store.put((user_id, "memories"), str(uuid.uuid4()), {"fact": extracted, "source_query": query})
        print(f"🧠 已写入长期记忆 [{user_id}]: {extracted}")
    except Exception as e:
        print(f"   [长期记忆] 写入失败: {e}")


def _chat_handler(state: RouterState, store: BaseStore) -> dict:
    """
    router 图的闲聊节点，是【模块 1／8：上下文管理】和【模块 2／8：记忆管理】的交汇点：
    - 记忆管理（跨会话，"记住关于这个人的事实"）：走 store，见 _search_long_term_memory /
      _maybe_write_long_term_memory，namespace 按 user_id 隔离，语义检索命中才拼进 prompt。
    - 上下文管理（同一会话内，"记住这次聊了什么"）：走 checkpointer 持久化的 state["messages"]，
      配合 _build_context_window 做 token 预算滑动窗口 + 旧对话摘要压缩，见 build_router_graph。
    两者互补但不能互相替代：记忆管理丢了不影响"这句话在说什么"，上下文管理丢了对话直接断片。
    """
    query = (state.get("query") or "").strip()
    user_id = (state.get("user_id") or "demo-user").strip() or "demo-user"

    # 【上下文管理】优先用服务端持久化的原始对话（有 checkpointer + thread_id 时，同一 thread
    # 的历次调用会自动带回）；只有服务端完全没有持久化记录时（没传 thread_id，或该 thread 第
    # 一次进入）才退回旧版"前端直传 history"的兼容路径，避免服务端/前端两个信源同时生效打架。
    persisted_messages = state.get("messages") or []
    if not persisted_messages:
        legacy_history = state.get("history") or []
        persisted_messages = [
            {"role": (h.get("role") or "").lower(), "content": (h.get("content") or "").strip()}
            for h in legacy_history
            if (h.get("role") or "").lower() in ("user", "assistant") and (h.get("content") or "").strip()
        ]

    memories = _search_long_term_memory(store, user_id, query)
    mem_ctx = ("\n【关于该用户的长期记忆】\n" + "\n".join(f"- {f}" for f in memories)) if memories else ""

    # 【上下文管理】核心调用：token 预算滑动窗口 + 旧对话增量摘要压缩（见函数定义处的详细说明）
    context_summary, recent_messages = _build_context_window(persisted_messages, state.get("context_summary", ""))

    system_prompt = "你是一个友善的AI助手，用中文简洁地回答，并结合完整上文语境进行多轮对话。" + mem_ctx
    if context_summary:
        system_prompt += f"\n【更早对话的摘要】{context_summary}"
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_messages)
    messages.append({"role": "user", "content": query})

    # DashScope 原生流式 + get_stream_writer()：非流式调用（invoke / stream_mode="updates"）下
    # writer 是no-op，本函数退化成「攒完整段再返回」，行为和之前的 _call_llm_messages 完全一致；
    # SSE 且 stream_mode 含 "custom" 时，前端能拿到逐 token 的打字机效果。
    writer = get_stream_writer()
    parts = []
    for delta in _call_llm_messages_stream(messages):
        parts.append(delta)
        writer({"nodeId": "chat", "content": delta})
    response = "".join(parts).strip()

    _maybe_write_long_term_memory(store, user_id, query)
    return {
        "response": f"💭 {response}",
        # 【上下文管理】写回这一轮的原始问答 + 可能刚更新的摘要；RouterState 里
        # messages 是 append 语义、context_summary 是覆盖语义，checkpointer 落盘后
        # 下一轮同一 thread_id 的请求会自动带回，不需要前端重新传 history。
        "messages": [{"role": "user", "content": query}, {"role": "assistant", "content": response}],
        "context_summary": context_summary,
    }


def _get_context_checkpointer() -> SqliteSaver:
    """
    【上下文管理】router 图用的共享 checkpointer——和 HITL 的 _get_hitl_checkpointer 是同一个模式，
    必须用共享文件的 SqliteSaver，不能用 MemorySaver：生产按 UVICORN_WORKERS>1 起多进程，
    MemorySaver 是进程内存，同一个 thread_id 的两次请求如果被负载均衡到不同 worker，第二个
    worker 根本看不到第一个 worker 存的对话历史，"服务端记住上下文"这件事直接失效。
    """
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "checkpoints")
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(db_dir, "context.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # 幂等，首次调用建表
    return saver


# 模块级单例：多个 worker 各自进程内只需建一次连接；router 图仍是"每次请求重新 build StateGraph
# 对象"（见 GRAPH_BUILDERS），但共享同一个 checkpointer 实例——状态落在 SQLite 文件里，
# 和 StateGraph 对象本身是否每次重建无关，这点和 HITL 把整个图都做成单例不同，没必要照搬。
_CONTEXT_CHECKPOINTER = _get_context_checkpointer()


def build_router_graph():
    """
    条件路由：classify → weather | news | chat | insight → END。
    chat 节点接了两套持久化：长期记忆 Store（跨会话，_LONG_TERM_STORE）+ 上下文 checkpointer
    （同一会话内，_CONTEXT_CHECKPOINTER）——分别对应【记忆管理】和【上下文管理】两个模块，
    见 _chat_handler 开头的说明。
    insight 是子图组合演示——内部调用 parallel 图（状态模式不同，节点内 invoke + 手动转换 state）。
    weather/news/chat/insight 都会打外部网络请求（高德/RSS/DashScope），加重试；classify 纯字符串匹配，不需要。
    """
    builder = StateGraph(RouterState)
    builder.add_node("classify", _classify_intent)
    builder.add_node("weather", _weather_handler, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("news", _news_handler, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("chat", _chat_handler, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("insight", _analyze_via_subgraph, retry_policy=RetryPolicy(max_attempts=3))

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        lambda s: s["intent"],
        {"weather": "weather", "news": "news", "chat": "chat", "insight": "insight"},
    )
    for name in ["weather", "news", "chat", "insight"]:
        builder.add_edge(name, END)

    return builder.compile(store=_LONG_TERM_STORE, checkpointer=_CONTEXT_CHECKPOINTER)


def demo_router():
    """演示条件路由并打印 ASCII 图。"""
    graph = build_router_graph()
    print("📊 **智能路由流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: classify → weather|news|chat → END)")
    print()
    for q in ["今天天气怎么样？", "有什么新闻？", "随便聊聊", "帮我分析一下：这家餐厅服务差但菜很好吃"]:
        out = graph.invoke({"query": q, "intent": "", "response": ""})
        print(f"  query={q!r} → response={out.get('response', '')}")
    return graph


def demo_memory():
    """演示长期记忆：第一轮告诉 AI 一个事实，第二轮换个问法验证跨轮次能语义召回。"""
    graph = build_router_graph()
    print("📊 **长期记忆 Store 演示**")
    user_id = "demo-memory-user"
    out1 = graph.invoke({"query": "我对海鲜过敏，以后推荐吃的东西要避开", "intent": "", "response": "", "user_id": user_id})
    print("第一轮 response:", out1.get("response"))
    out2 = graph.invoke({"query": "今晚吃什么好，给点建议", "intent": "", "response": "", "user_id": user_id})
    print("第二轮（应体现出对海鲜过敏的记忆）response:", out2.get("response"))
    return graph


# ---------------------------------------------------------------------------
# 【模块 6／8：权限控制】节点级三档权限：readonly（自由执行）/ confirm（需人工审核才能执行，
# 复用下面 5 的 interrupt() 机制）/ forbidden（直接拒绝，本项目暂无这一档的真实节点，
# 机制已就绪，留给未来接入真正有副作用的节点，比如"发送邮件""删除文件"）。
# 完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 「权限控制」一节。
#
# 本项目目前 weather/news/chat/insight/think/decide/respond/sentiment/keywords/summary/
# aggregate 这些节点全部只读——只调用 LLM 或查询外部只读 API，没有任何写操作/副作用，天然
# readonly，不需要挡。唯一有真实副作用的是 HITL 的 process 节点（代表"真正执行一个动作"），
# 所以它是全项目唯一标为 confirm 的节点，且这一档不是摆设——review 节点的 interrupt() 就是它
# 的强制执行点，process 不经过人工批准的 review 节点，图结构上根本走不到（见 build_hitl_graph）。
# ---------------------------------------------------------------------------

NODE_PERMISSIONS: dict[str, str] = {
    "classify": "readonly", "weather": "readonly", "news": "readonly",
    "chat": "readonly", "insight": "readonly",
    "think": "readonly", "decide": "readonly", "respond": "readonly",
    "sentiment": "readonly", "keywords": "readonly", "summary": "readonly", "aggregate": "readonly",
    "analyze": "readonly",  # HITL：只生成建议文本，不执行任何操作，本身只读
    "process": "confirm",   # HITL：真正执行建议的节点，必须经过人工审核（interrupt）才能到达
}


def check_node_permission(node_id: str) -> str:
    """
    【权限控制】查询某节点的权限档位。未登记的新节点默认按 "confirm"（需要确认）处理而不是
    放行——这是"失败关闭"（fail closed）的安全姿态：新接入一个不认识的、可能有副作用的节点时，
    宁可多一次人工确认，也不能悄悄放行。
    """
    return NODE_PERMISSIONS.get(node_id, "confirm")


# ---------------------------------------------------------------------------
# 【模块 8／8：副作用回滚】补偿事务（Saga Pattern）——权限控制挡的是"执行前要不要批准"，
# 这里管的是"万一已经执行了、后面某一步又失败了，怎么把已经产生的副作用撤销掉"，两者互补。
# 完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 「副作用回滚」一节。
#
# 核心思路：每个可能有副作用的正向动作（execute）都配一个补偿动作（undo），Saga 编排器按
# 顺序执行；任意一步失败，就把已经成功的步骤按**逆序**依次 undo，undo 拿的是对应 execute
# 当初返回的结果（比如新插入行的主键），而不是重新猜状态。
#
# 下面 AgentBooking（真实 MySQL 表）+ create_booking/charge_payment 两步是唯一一个真的会
# 产生外部（数据库）副作用的演示场景，用来验证"回滚真的发生了"，不是纸面上的接口设计——
# 项目里其余节点全部只读，没有真实副作用可回滚，硬造 undo 接口给它们只是摆设，没有意义。
# ---------------------------------------------------------------------------


class SagaStep:
    """一步 Saga：execute 是正向动作，undo 是对应的补偿动作，undo 接收 execute 的返回值。"""

    def __init__(self, name: str, execute, undo):
        self.name = name
        self.execute = execute
        self.undo = undo


def run_saga(steps: list) -> dict:
    """
    Saga 编排器：按顺序跑 steps；任意一步抛异常，立刻停止往后执行，把已完成的步骤按逆序
    依次调用 undo 补偿。某一步的 undo 本身也失败时不会中断其余补偿（尽量撤销更多），但会
    记进日志——这种情况在真实系统里代表"需要人工介入核对账目"，补偿机制不是万能的。
    返回 {"status": "committed"|"rolled_back", "completed": [...], "compensated": [...], "error": str|None}
    """
    completed = []  # [(step, execute返回值)]
    for step in steps:
        try:
            result = step.execute()
            completed.append((step, result))
            print(f"✅ Saga step 执行成功: {step.name}")
        except Exception as e:
            print(f"❌ Saga step 失败: {step.name}，错误: {e}，开始逆序回滚已完成步骤")
            compensated = []
            for done_step, done_result in reversed(completed):
                try:
                    done_step.undo(done_result)
                    compensated.append(done_step.name)
                    print(f"↩️ 已回滚: {done_step.name}")
                except Exception:
                    logger.exception(f"补偿失败: {done_step.name}，需要人工介入核对")
            return {
                "status": "rolled_back",
                "completed": [s.name for s, _ in completed],
                "compensated": compensated,
                "error": str(e),
            }
    return {"status": "committed", "completed": [s.name for s, _ in completed], "compensated": [], "error": None}


def _create_booking_step(thread_id: str, item: str, amount: int) -> SagaStep:
    """execute 真实插入一行 AgentBooking，返回主键；undo 按主键删除同一行——一一对应。"""

    def execute():
        from app.database import SessionLocal
        from model.ai.agent_booking import AgentBooking

        session = SessionLocal()
        try:
            booking = AgentBooking(thread_id=thread_id, item=item, amount=amount, status="pending")
            session.add(booking)
            session.commit()
            session.refresh(booking)
            return booking.id
        finally:
            session.close()

    def undo(booking_id):
        from app.database import SessionLocal
        from model.ai.agent_booking import AgentBooking

        session = SessionLocal()
        try:
            row = session.query(AgentBooking).filter(AgentBooking.id == booking_id).first()
            if row:
                session.delete(row)
                session.commit()
        finally:
            session.close()

    return SagaStep("create_booking", execute, undo)


def _charge_payment_step(should_fail: bool) -> SagaStep:
    """
    演示用扣款步骤：should_fail=True 时故意抛异常，模拟支付渠道失败，触发 run_saga 逆序回滚。
    undo 是 no-op——因为这一步从未真正提交成功过（execute 直接抛异常），没有可撤销的状态；
    真实接入支付渠道时，这里的 undo 才需要真的调 stripe.refunds.create 之类的退款接口。
    """

    def execute():
        if should_fail:
            raise RuntimeError("支付渠道返回失败（演示用：故意触发，用来验证补偿回滚）")
        return "charged"

    def undo(_result):
        pass

    return SagaStep("charge_payment", execute, undo)


def saga_demo_api(request: Request):
    """
    POST /ai/langgraph/saga-demo  body: {item?, amount?, failPayment?(默认true), threadId?}
    【副作用回滚】演示端点：两步 Saga——create_booking（真实写 MySQL）→ charge_payment。
    failPayment=true（默认）：第二步故意失败，触发回滚，返回后查 agent_booking 表应该
    查不到这次生成的行；failPayment=false：两步都成功，booking 行保留。
    """
    body = anyio.from_thread.run(read_json_optional, request) or {}
    item = (body.get("item") or "测试预订").strip()
    amount = int(body.get("amount") or 100)
    fail_payment = bool(body.get("failPayment", True))
    thread_id = body.get("threadId") or str(uuid.uuid4())
    steps = [
        _create_booking_step(thread_id, item, amount),
        _charge_payment_step(fail_payment),
    ]
    result = run_saga(steps)
    return {"code": 0, "msg": "ok", "data": {**result, "threadId": thread_id}}


# ---------------------------------------------------------------------------
# 5. 人机交互节点 - interrupt() 暂停 → 人工审核/编辑 → 处理反馈
# 同时也是【模块 7／8：人工处理】的完整实现，以及【模块 6／8：权限控制】confirm 档的
# 落地机制——两个模块在这里是同一套代码，不是巧合：生产系统里"危险动作要不要执行"这个
# 权限判断，落到实现上往往就是"暂停下来等人工确认"，见下面 _hitl_review 的 interrupt()。
# ---------------------------------------------------------------------------


class HitlState(TypedDict):
    query: str
    suggestion: str  # AI 生成的建议，供人工审核
    decision: str  # approved / rejected，记录人工决定
    response: str


def _hitl_analyze(state: HitlState) -> dict:
    """AI 根据用户请求生成一条具体行动建议，等待人工审核。"""
    query = (state.get("query") or "").strip()
    suggestion = _call_llm(
        f"用户请求：{query}\n\n请给出你建议采取的具体行动方案，用一句话说清楚要做什么，不超过40字。",
        system="你是一个行动建议助手，只输出具体、可执行的建议本身，不要解释。",
    )
    print(f"✨ AI 建议: {suggestion}")
    return {"suggestion": suggestion or "（未能生成建议，请重新提问）"}


def _hitl_review(state: HitlState) -> Command[Literal["process", "__end__"]]:
    """
    暂停图执行，把 AI 建议交给人工审核。
    resume 传入 True/False 表示批准/拒绝；传入非空字符串表示"批准并采用编辑后的建议"。
    """
    decision = interrupt({
        "question": "是否批准以下 AI 建议？可直接批准/拒绝，或提交编辑后的文本作为最终建议。",
        "suggestion": state.get("suggestion", ""),
    })
    print(f"👤 人工审核结果: {decision!r}")
    if isinstance(decision, str) and decision.strip():
        # 人工编辑过建议内容：视为批准，并采用编辑后的文本
        return Command(goto="process", update={"decision": "approved", "suggestion": decision.strip()})
    if decision:
        return Command(goto="process", update={"decision": "approved"})
    return Command(
        goto=END,
        update={"decision": "rejected", "response": "已拒绝该建议，未执行任何操作。"},
    )


def _hitl_process(state: HitlState) -> dict:
    """
    人工批准（或编辑）后执行，生成最终回复。
    【权限控制】这是全项目唯一 confirm 档节点的真正执行点：断言一下它的权限档位配置没被
    改错——图结构已经保证了只有 review 节点批准（即 interrupt() 收到批准结果）才能路由到
    这里（见 _hitl_review），这行断言是双保险，防止未来有人改动 NODE_PERMISSIONS 却忘了
    同步图结构，导致"标着 confirm 却其实没人审核"这种权限声明和实际行为不一致的情况。
    """
    assert check_node_permission("process") == "confirm", "process 节点权限档位配置有误，应为 confirm"
    suggestion = state.get("suggestion", "")
    response = f"✅ 已按审核通过的建议执行：{suggestion}"
    print(f"⚙️ {response}")
    return {"response": response}


def _get_hitl_checkpointer() -> SqliteSaver:
    """
    HITL 用共享 SQLite 文件做 checkpointer，而不是 MemorySaver。
    原因：生产部署是多 worker 进程（UVICORN_WORKERS>1），每个 worker 是独立进程、
    独立内存；interrupt 命中后的第一次请求和 resume 请求很可能被负载均衡到不同 worker，
    MemorySaver 存在各自进程内存里，跨进程互相看不到，resume 时会因为"找不到暂停状态"
    而从头重新执行整个图，人工审核形同虚设。SQLite 文件所有 worker 共享同一份，可跨进程恢复。
    """
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "checkpoints")
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(db_dir, "hitl.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # 幂等，首次调用建表
    return saver


def build_hitl_graph(checkpointer=None):
    """人机交互图：analyze → review(interrupt) → process | END。需要 checkpointer 才能跨请求暂停/恢复。"""
    builder = StateGraph(HitlState)
    builder.add_node("analyze", _hitl_analyze, retry_policy=RetryPolicy(max_attempts=3))
    # destinations 告诉 get_graph() 这个 Command 节点可能跳去哪些目标，
    # 纯供静态可视化用，实际路由仍由 _hitl_review 运行时返回的 Command(goto=...) 决定。
    builder.add_node("review", _hitl_review, destinations=("process", END))
    builder.add_node("process", _hitl_process)
    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "review")
    builder.add_edge("process", END)
    return builder.compile(checkpointer=checkpointer or _get_hitl_checkpointer())


# HITL 图依赖 checkpointer 维持跨请求（analyze→interrupt / resume→process）的暂停状态，
# 必须是同一个编译后的图对象和同一个 checkpointer 实例，因此在模块级构建一次并复用，
# 而不是像 router/loop/parallel 那样每次请求都重新 build。
_HITL_GRAPH = build_hitl_graph()


def demo_hitl():
    """演示 HITL 流程：第一次调用命中 interrupt 暂停，第二次带 resume 决定继续。"""
    graph = _HITL_GRAPH
    print("📊 **人机交互流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: analyze → review →[interrupt]→ process | END)")
    print()
    config = {"configurable": {"thread_id": "demo-hitl-1"}}
    out = graph.invoke({"query": "帮我优化这段文案", "suggestion": "", "decision": "", "response": ""}, config)
    print("命中 interrupt:", out.get("__interrupt__"))
    resumed = graph.invoke(Command(resume=True), config)
    print("恢复后 response:", resumed.get("response"))
    return graph


def run_hitl_graph(input_state: dict | None, thread_id: str, resume=None) -> dict:
    """
    执行/恢复 HITL 图，供 HTTP 层调用。
    首次调用不传 resume：命中 interrupt 后返回 waitingForInput=True + interrupt payload。
    第二次调用带上同一个 thread_id + resume（人工决定），从暂停点继续执行到 END。
    """
    graph = _HITL_GRAPH
    config = {"configurable": {"thread_id": thread_id}}
    if resume is not None:
        run_input = Command(resume=resume)
    else:
        default = DEFAULT_INPUTS.get("hitl", {})
        run_input = {**default, **(input_state or {})}
    result = graph.invoke(run_input, config=config)
    interrupts = result.get("__interrupt__")
    if interrupts:
        first = interrupts[0]
        payload = getattr(first, "value", first)
        return {
            "threadId": thread_id,
            "waitingForInput": True,
            "interrupt": payload,
            "finalState": {k: v for k, v in result.items() if k != "__interrupt__"},
        }
    return {
        "threadId": thread_id,
        "waitingForInput": False,
        "finalState": result,
        "response": result.get("response", ""),
    }


# ---------------------------------------------------------------------------
# 6. 实时执行监控 - stream 可视化
# ---------------------------------------------------------------------------

# 节点图标：由后端返回给前端 graphData.nodes[].icon，前端据此展示；可在此修改
NODE_ICONS = {
    "think": "🤔",
    "decide": "🎯",
    "process": "⚙️",
    "analyze": "🔍",
    "generate": "✨",
    "classify": "🎯",
    "aggregate": "📊",
    "weather": "☀️",
    "news": "📰",
    "chat": "💭",
    "sentiment": "😊",   # 情感分析
    "keywords": "🏷️",   # 关键词
    "summary": "📝",    # 摘要
    "review": "👤",
    "dispatch": "📤",
    "respond": "📢",
    "insight": "🧩",   # 子图组合（router 内嵌 parallel 子图）
}

# 节点 id -> 前端展示（可选覆盖），未列出的用 raw_id、type=process
NODE_DISPLAY = {
    "__start__": {"name": "用户输入", "type": "input", "icon": "📝", "description": "入口"},
    "__end__": {"name": "输出", "type": "output", "icon": "📢", "description": "出口"},
    "classify": {"name": "意图分类", "type": "llm", "description": "分析用户意图"},
    "weather": {"name": "天气", "type": "tool", "description": "天气查询"},
    "news": {"name": "新闻", "type": "tool", "description": "新闻摘要"},
    "chat": {"name": "闲聊", "type": "llm", "description": "通用对话"},
    "insight": {"name": "多维分析（子图）", "type": "tool", "description": "调用 parallel 子图：情感/关键词/摘要"},
    "think": {"name": "思考", "type": "llm", "description": "迭代思考"},
    "decide": {"name": "决策", "type": "condition", "description": "是否继续"},
    "sentiment": {"name": "情感分析", "type": "llm", "description": "情感分析"},
    "keywords": {"name": "关键词", "type": "tool", "description": "关键词提取"},
    "summary": {"name": "摘要", "type": "llm", "description": "文本摘要"},
    "aggregate": {"name": "聚合", "type": "process", "description": "汇总结果"},
    "analyze": {"name": "AI 分析", "type": "llm", "description": "生成建议"},
    "review": {"name": "人工审核", "type": "condition", "description": "人工确认"},
    "process": {"name": "处理反馈", "type": "process", "description": "应用反馈"},
    "respond": {"name": "最终回答", "type": "output", "icon": "📢", "description": "根据思考生成或调用接口返回结果"},
}


def visualize_execution(graph, inputs: dict, sleep_sec: float = 0.3):
    """按 stream 步进打印每个节点的执行与状态更新。"""
    print("🎬 **执行开始**")
    print("=" * 50)
    for step in graph.stream(inputs):
        for node_name, node_output in step.items():
            ts = datetime.now().strftime("%H:%M:%S")
            icon = NODE_ICONS.get(node_name, "🔹")
            print(f"[{ts}] {icon} 节点: {node_name}")
            print(f"   📦 状态更新: {node_output}")
            print("-" * 30)
            if sleep_sec:
                time.sleep(sleep_sec)
    print("=" * 50)
    print("✅ **执行完成**")


def get_node_color(status: str) -> str:
    """按状态返回终端颜色码（可选，用于高级可视化）。"""
    colors = {"active": "\033[92m", "completed": "\033[94m", "error": "\033[91m", "waiting": "\033[93m"}
    return colors.get(status, "\033[0m")


class LangGraphDashboard:
    """简单内存仪表盘：记录执行路径与节点状态。"""

    def __init__(self):
        self.nodes_status: dict = {}
        self.execution_path: list = []

    def update(self, node_name: str, status: str, data=None):
        self.nodes_status[node_name] = {
            "status": status,
            "timestamp": datetime.now(),
            "data": data,
        }
        self.execution_path.append(node_name)

    def render(self, clear: bool = False):
        if clear:
            print("\033c", end="")
        print("╔════════════════════════════════╗")
        print("║   LangGraph 实时执行仪表盘     ║")
        print("╚════════════════════════════════╝")
        print("\n📈 执行路径:", " → ".join(self.execution_path))
        print("\n📊 节点状态:")
        for node, info in self.nodes_status.items():
            icon = "✅" if info["status"] == "completed" else "⏳"
            print(f"  {icon} {node}: {info['status']}")
        print()


# ---------------------------------------------------------------------------
# 前端 3D 可视化对接：从编译后的图动态生成 schema（供 React+Three.js 使用）
# ---------------------------------------------------------------------------

def _node_id_for_schema(raw_id: str) -> str:
    """将图内部节点 id 转为前端 schema 的 id（__start__ -> input, __end__ -> output）。"""
    if raw_id == "__start__":
        return "input"
    if raw_id == "__end__":
        return "output"
    return raw_id


def graph_to_schema(compiled_graph, node_display: dict | None = None, node_icons: dict | None = None) -> dict:
    """
    从 LangGraph 编译后的图动态生成前端 GraphData 格式：nodes + edges。
    使用 get_graph() 的 nodes/edges，不手写结构。
    node_display / node_icons 可选，供其他模块（如 agent_research、agent_wealth_advisor）传入自定义展示信息。
    """
    raw = compiled_graph.get_graph()
    display_map = node_display if node_display is not None else NODE_DISPLAY
    icons_map = node_icons if node_icons is not None else NODE_ICONS
    nodes_out = []
    # 节点：raw.nodes 为 dict[id -> Node]
    for raw_id in raw.nodes:
        display = display_map.get(raw_id, {})
        schema_id = _node_id_for_schema(raw_id)
        name = display.get("name") or raw_id
        node_type = display.get("type") or "process"
        icon = display.get("icon") or icons_map.get(raw_id, "🔹")
        desc = display.get("description") or ""
        nodes_out.append({
            "id": schema_id,
            "name": name,
            "type": node_type,
            "icon": icon,
            "description": desc,
        })
    # 边：raw.edges 为 list[Edge(source, target, data, conditional)]
    edges_out = []
    for e in raw.edges:
        src = getattr(e, "source", None) or (e.get("source") if isinstance(e, dict) else None)
        tgt = getattr(e, "target", None) or (e.get("target") if isinstance(e, dict) else None)
        src = _node_id_for_schema(src) if src else src
        tgt = _node_id_for_schema(tgt) if tgt else tgt
        conditional = getattr(e, "conditional", None)
        if conditional is None and isinstance(e, dict):
            conditional = e.get("conditional", False)
        edge_type = "conditional" if conditional else "normal"
        edges_out.append({"source": src, "target": tgt, "type": edge_type})
    return {"nodes": nodes_out, "edges": edges_out}


def _merge_state_update(current: dict, update: dict) -> dict:
    """按 LangGraph 的 reducer 语义合并一次节点输出到当前 state（messages 用 add，其余覆盖）。"""
    if not update or not isinstance(update, dict):
        return current
    out = dict(current)
    for key, value in update.items():
        if key == "messages":
            existing = out.get("messages") or []
            add = value if isinstance(value, list) else [value]
            out["messages"] = existing + add
        elif key == "analyses" and isinstance(value, list):
            existing = out.get("analyses") or []
            out["analyses"] = existing + value
        else:
            out[key] = value
    return out


def run_graph_stream_and_collect(graph, state: dict, config: dict | None = None):
    """
    执行图 stream 一次，收集每一步的 nodeId、耗时、输出，并从各步输出合并出最终 state。
    不再二次 invoke，避免流程跑两遍、最终回答提前打印。
    config：{"configurable": {"thread_id": "..."}}，【上下文管理】/【人工处理】依赖 checkpointer
    做跨请求状态持久化的图（router/hitl）靠它认出"这是同一个会话"；不传则每次都是全新会话
    （等价于旧行为），loop/parallel 没接 checkpointer，传不传都不影响它们。
    返回：{"steps": [...], "finalState": {...}, "executionOrder": [...], "totalSteps": N}。
    前端进度条应用：当前步 = stepIndex+1，总步数 = totalSteps，进度 = (stepIndex+1)/totalSteps*100%。
    勿用 finalState.iteration 当作总步数（iteration 仅表示“思考轮数”，如 loop 里为 3）。
    """
    steps = []
    execution_order = []
    t0 = time.perf_counter()
    step_index = 0
    current_state = dict(state)
    # stream_mode="updates"：并行节点（如 parallel 的 sentiment/keywords/summary）会分别 yield，前端才能逐步展示，不会「从开始直接跳到结束」
    for step in graph.stream(state, config=config, stream_mode="updates"):
        for node_id, output in step.items():
            t1 = time.perf_counter()
            duration_ms = round((t1 - t0) * 1000)
            t0 = t1
            step_payload = {
                "stepIndex": step_index,
                "nodeId": node_id,
                "status": "end",
                "duration_ms": duration_ms,
                "output": output,
            }
            step_payload.update(_enrich_step_for_frontend(node_id, output if isinstance(output, dict) else {}, current_state))
            steps.append(step_payload)
            execution_order.append(node_id)
            current_state = _merge_state_update(current_state, output if isinstance(output, dict) else {})
            step_index += 1
    return {
        "steps": steps,
        "finalState": current_state,
        "executionOrder": execution_order,
        "totalSteps": len(steps),
    }


def _enrich_step_for_frontend(node_id: str, output: dict, current_state: dict) -> dict:
    """
    为前端展示补充 step 的易用字段：loop 图用 iteration/thought/label，parallel 图用 label。
    """
    extra = {}
    # parallel 图：每步给中文 label，便于时间线展示
    if node_id == "dispatch":
        extra["label"] = "分发"
    elif node_id == "sentiment":
        extra["label"] = "情感分析"
    elif node_id == "keywords":
        extra["label"] = "关键词提取"
    elif node_id == "summary":
        extra["label"] = "摘要"
    elif node_id == "aggregate":
        extra["label"] = "聚合"
    # loop 图
    elif node_id == "think" and isinstance(output, dict):
        msgs = output.get("messages")
        extra["iteration"] = output.get("iteration", current_state.get("iteration", 0))
        extra["thought"] = (msgs[-1] if isinstance(msgs, list) and msgs else msgs) or ""
        extra["label"] = f"第{extra['iteration']}轮思考"
    elif node_id == "decide" and isinstance(output, dict):
        next_step = output.get("next_step", "")
        extra["nextStep"] = next_step
        extra["label"] = "继续思考" if next_step == "think" else "进入回答"
    elif node_id == "respond" and isinstance(output, dict):
        extra["response"] = output.get("response", "")
        extra["label"] = "最终回答"
    # router 图：weather/news/chat/insight 节点也带 response，前端可直接从 step 或 finalState 取
    elif node_id in ("weather", "news", "chat", "insight") and isinstance(output, dict):
        extra["response"] = output.get("response", "")
        extra["label"] = {"weather": "天气", "news": "新闻", "chat": "闲聊", "insight": "多维分析（子图）"}.get(node_id, node_id)
    return extra


def run_graph_stream_yield_events(graph, state: dict, config: dict | None = None):
    """
    执行图 stream，每完成一步 yield 一个 step 事件，最后 yield 一个 done 事件。
    供 SSE 流式接口使用：前端先按步更新流程动画，收到 done 后再展示回答，避免「回答比流程快」。
    loop 图每步会带 iteration/thought/nextStep/response/label 等字段，便于前端展示「第 N 轮思考」。
    config 用途见 run_graph_stream_and_collect 的说明（【上下文管理】跨请求会话识别）。

    stream_mode=["updates","custom"]：graph.stream() 因此按 (mode, payload) 元组产出。
    "custom" 来自节点内 get_stream_writer()（目前是 chat 节点的 DashScope 逐 token 转发），
    只有 router 图的 chat 分支会产出 token 事件，其余图/节点没有 writer 调用，不受影响。
    yield: ("step", {...}) | ("token", {"nodeId","content"}) | ("done", {...})。
    """
    steps = []
    execution_order = []
    t0 = time.perf_counter()
    step_index = 0
    current_state = dict(state)
    for mode, payload in graph.stream(state, config=config, stream_mode=["updates", "custom"]):
        if mode == "custom":
            yield ("token", payload)
            continue
        for node_id, output in payload.items():
            t1 = time.perf_counter()
            duration_ms = round((t1 - t0) * 1000)
            t0 = t1
            step_payload = {
                "stepIndex": step_index,
                "nodeId": node_id,
                "status": "end",
                "duration_ms": duration_ms,
                "output": output,
            }
            extra = _enrich_step_for_frontend(node_id, output if isinstance(output, dict) else {}, current_state)
            step_payload.update(extra)
            steps.append(step_payload)
            execution_order.append(node_id)
            current_state = _merge_state_update(current_state, output if isinstance(output, dict) else {})
            step_index += 1
            yield ("step", step_payload)
    yield ("done", {
        "finalState": current_state,
        "totalSteps": len(steps),
        "executionOrder": execution_order,
        "steps": steps,
    })


GRAPH_BUILDERS = {
    "router": build_router_graph,
    "loop": build_loop_graph,
    "parallel": build_parallel_graph,
    # hitl 需要跨请求维持 interrupt 暂停状态，复用模块级单例，而非每次重新 build
    "hitl": lambda: _HITL_GRAPH,
}

DEFAULT_INPUTS = {
    "router": {
        "query": "今天天气怎么样？", "intent": "", "response": "", "user_id": "demo-user",
        # 【上下文管理】messages/context_summary 显式给空默认值（而不是依赖 LangGraph 隐式默认），
        # 避免走 weather/news/insight 等从不碰这两个字段的分支时，channel 因从未被赋值而行为不确定。
        "messages": [], "context_summary": "",
    },
    "loop": {"messages": [], "next_step": "", "iteration": 0, "query": "", "response": ""},
    "parallel": {"input_text": "示例文本", "analyses": [], "final_result": "", "response": ""},
    "hitl": {"query": "", "suggestion": "", "decision": "", "response": ""},
}

def get_graph_schema(name: str) -> dict | None:
    """从编译后的图动态生成前端 GraphData（nodes + edges），不手写结构。"""
    builder_fn = GRAPH_BUILDERS.get(name)
    if not builder_fn:
        return None
    graph = builder_fn()
    schema = graph_to_schema(graph)
    schema["executionOrder"] = []  # 真实顺序由 POST /run 返回
    return schema


def list_graph_names():
    """返回可用的图名称列表。"""
    return list(GRAPH_BUILDERS.keys())


# ---------------------------------------------------------------------------
# 【模块 4／8：可观测性】+【模块 5／8：回流机制】共用同一张表 model/ai/agent_trace.py：
# _persist_trace 是可观测性的采集入口（每次图运行落一条，记录耗时/步骤/成败）；
# submit_trace_feedback/list_bad_cases 是回流机制的采集+挖掘入口（人工/用户标注 好评/差评，
# 差评连同天然的 status=error 一起构成 bad case 池，供后续回流进知识库/prompt/训练集——
# 这三处回流目的地本身是人工/离线流程，本项目只负责把"哪些是 bad case"这一步做实）。
# 完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 对应两节。
# ---------------------------------------------------------------------------


def _persist_trace(
    graph_name: str,
    state: dict,
    run_result: dict | None,
    duration_ms: int,
    error: str | None = None,
) -> None:
    """
    【可观测性】把一次图执行（invoke 或 stream 一轮）落一条 trace 记录，供排查"哪一步慢/哪一步错"
    和后续 bad case 回流分析用。用独立 session（SessionLocal），不读请求级 db.session：SSE 分支
    落库发生在 StreamingResponse 对象返回之后，此时 _ai_route 已经 clear_request_session，
    db.session 会抛 RuntimeError。落库失败只记日志，不抛出，不影响主流程。
    """
    from app.database import SessionLocal
    from model.ai.agent_trace import AgentTrace

    steps = (run_result or {}).get("steps", [])
    final_state = (run_result or {}).get("finalState", {}) or {}
    session = SessionLocal()
    try:
        session.add(AgentTrace(
            graph_name=graph_name,
            thread_id=state.get("threadId") or state.get("thread_id"),
            user_id=state.get("user_id"),
            input_summary=(state.get("query") or state.get("input_text") or "")[:2000],
            output_summary=(final_state.get("response") or "")[:2000],
            status="error" if error else "success",
            error_message=error,
            total_steps=len(steps),
            duration_ms=duration_ms,
            steps_detail=json.dumps(
                [{"nodeId": s.get("nodeId"), "duration_ms": s.get("duration_ms")} for s in steps],
                ensure_ascii=False,
            ),
        ))
        session.commit()
    except Exception:
        logger.exception("agent trace 落库失败，不影响主流程")
        session.rollback()
    finally:
        session.close()


def submit_trace_feedback(trace_id: int, rating: str, note: str = "") -> bool:
    """
    【回流机制】采集入口：给一条已存在的 trace 打显式反馈（good/bad）。这是飞轮的第一步——
    没有反馈信号，后面"哪些是 bad case"就无从谈起。rating 只接受 good/bad，其他值直接拒绝
    （不做静默纠正，调用方传错参数应该显式失败，而不是被悄悄改成别的值）。
    返回 True/False 表示是否成功命中并更新了一条记录。
    """
    if rating not in ("good", "bad"):
        raise ValueError(f"rating 只能是 good/bad，收到: {rating!r}")
    from app.database import SessionLocal
    from model.ai.agent_trace import AgentTrace

    session = SessionLocal()
    try:
        row = session.query(AgentTrace).filter(AgentTrace.id == trace_id).first()
        if row is None:
            return False
        row.feedback = rating
        row.feedback_note = note[:2000] if note else None
        session.commit()
        return True
    except Exception:
        logger.exception("trace 反馈写入失败")
        session.rollback()
        return False
    finally:
        session.close()


def list_bad_cases(graph_name: str | None = None, limit: int = 50) -> list[dict]:
    """
    【回流机制】挖掘入口：拉取 bad case 候选池——status='error'（系统自己判定的失败）或
    feedback='bad'（人工/用户标注的不满意）任一命中即算。这是回流的第二步：从原始信号里
    筛出"值得回流"的样本。真正"回流去哪"（知识库补全 / prompt 新增约束 / 攒训练数据）是
    后续人工或离线批处理的事，不在这个函数职责内——这里只负责把候选池准确地找出来。
    """
    from app.database import SessionLocal
    from model.ai.agent_trace import AgentTrace

    session = SessionLocal()
    try:
        query = session.query(AgentTrace).filter(
            (AgentTrace.status == "error") | (AgentTrace.feedback == "bad")
        )
        if graph_name:
            query = query.filter(AgentTrace.graph_name == graph_name)
        rows = query.order_by(AgentTrace.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "graph_name": r.graph_name,
                "thread_id": r.thread_id,
                "user_id": r.user_id,
                "input_summary": r.input_summary,
                "output_summary": r.output_summary,
                "status": r.status,
                "error_message": r.error_message,
                "feedback": r.feedback,
                "feedback_note": r.feedback_note,
                "created_at": r.create_at.isoformat() if r.create_at else None,
            }
            for r in rows
        ]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# HTTP 视图：供 routes/ai.py 注册 【回流机制】的两个端点
# ---------------------------------------------------------------------------


def trace_feedback_api(request: Request):
    """POST /ai/langgraph/trace/feedback  body: {traceId, rating: good|bad, note?}"""
    body = anyio.from_thread.run(read_json_optional, request) or {}
    trace_id = body.get("traceId") or body.get("trace_id")
    rating = body.get("rating")
    note = body.get("note", "")
    if not trace_id or rating not in ("good", "bad"):
        return ({"code": 400, "msg": "缺少参数: traceId / rating(good|bad)", "data": None}, 400)
    try:
        ok = submit_trace_feedback(int(trace_id), rating, note)
    except Exception as e:
        return ({"code": 400, "msg": str(e), "data": None}, 400)
    if not ok:
        return ({"code": 404, "msg": f"未找到 trace: {trace_id}", "data": None}, 404)
    return {"code": 0, "msg": "ok", "data": {"traceId": trace_id, "rating": rating}}


def trace_bad_cases_api(request: Request):
    """GET /ai/langgraph/trace/bad-cases?graph=router&limit=50  返回 bad case 候选池，供人工复核/回流。"""
    q = query_dict(request)
    graph_name = q.get("graph")
    limit = int(q.get("limit") or 50)
    cases = list_bad_cases(graph_name, limit)
    return {"code": 0, "msg": "ok", "data": {"cases": cases, "total": len(cases)}}


def run_graph_and_collect_steps(graph_name: str, input_state: dict | None = None, thread_id: str | None = None):
    """
    执行指定图，收集每一步的 nodeId、耗时、输出，供前端按真实执行顺序与节奏驱动 3D 动画。
    返回：{
        "graphData": { nodes, edges, executionOrder },
        "steps": [ { "nodeId", "status": "end", "duration_ms", "output" }, ... ],
        "finalState": { ... },
        "executionOrder": [ "classify", "weather", ... ]
    }
    前端传入的 input 会与当前图的默认 state 合并，避免切图后残留字段导致缺键报错（如 loop 下误传 query 等）。
    【上下文管理】生产用法：router 图传 thread_id 即可让服务端跨请求持久化对话（见
    build_router_graph 的 checkpointer），不需要再传 history；thread_id 缺省或图未接
    checkpointer（loop/parallel）时退回旧行为——input 里带 history 仍然兼容。
    """
    builder_fn = GRAPH_BUILDERS.get(graph_name)
    if not builder_fn:
        return {"error": f"未知图: {graph_name}", "allowed": list(GRAPH_BUILDERS.keys())}
    graph = builder_fn()
    default = DEFAULT_INPUTS.get(graph_name, {})
    if input_state:
        state = {**default, **input_state}
    else:
        state = default.copy()
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    trace_state = {**state, "thread_id": thread_id} if thread_id else state  # 只为落 trace 用，不进图执行
    t_start = time.perf_counter()
    try:
        run_result = run_graph_stream_and_collect(graph, state, config)
    except Exception as e:
        _persist_trace(graph_name, trace_state, None, round((time.perf_counter() - t_start) * 1000), error=str(e))
        return {"error": str(e)}
    _persist_trace(graph_name, trace_state, run_result, round((time.perf_counter() - t_start) * 1000))
    schema = graph_to_schema(graph)
    steps = run_result["steps"]
    execution_order = run_result["executionOrder"]
    total_steps = run_result.get("totalSteps", len(steps))
    nodes = schema["nodes"]
    total_nodes = len(nodes)
    # 执行监控用：总节点数、已完成步数、进度百分比；对话历史用 finalState.response
    completed_steps = len(steps)
    execution_progress = round((completed_steps / total_nodes * 100), 1) if total_nodes else 0
    return {
        "graphData": {
            "nodes": nodes,
            "edges": schema["edges"],
            "executionOrder": execution_order,
        },
        "steps": steps,
        "finalState": run_result["finalState"],
        "executionOrder": execution_order,
        "totalSteps": total_steps,
        "totalNodes": total_nodes,
        "completedSteps": completed_steps,
        "executionProgress": execution_progress,
        "response": run_result["finalState"].get("response", ""),
    }


# ---------------------------------------------------------------------------
# HTTP 视图：供 routes/ai 注册 GET/POST
# ---------------------------------------------------------------------------


def langgraph_graph_api(request: Request):
    """GET /ai/langgraph/graph?name=router 返回图结构，供前端 3D 可视化（GraphData）。"""
    q = query_dict(request)
    name = q.get("name") or "router"
    schema = get_graph_schema(name)
    if schema is None:
        return (
            {
                "code": 400,
                "msg": f"未知图: {name}",
                "data": {"allowed": list_graph_names()},
            },
            400,
        )
    return {"code": 0, "msg": "ok", "data": schema}


def langgraph_run_api(request: Request):
    """
    POST /ai/langgraph/run 执行图并返回步骤与最终状态，供前端按真实执行顺序驱动 3D 动画。

    非流式（默认）：响应体为 JSON，结构为：
      { "code": 0, "msg": "ok", "data": {
          "graphData": { "nodes", "edges", "executionOrder" },
          "steps": [ { "stepIndex", "nodeId", "output", "response?", "label?" }, ... ],
          "finalState": { "query", "intent", "response", ... },
          "totalNodes": 7, "completedSteps": 2, "executionProgress": 28.6,
          "response": "最终回复正文（与 finalState.response 一致，便于直接展示对话）"
        }}
    前端「执行监控」建议：总节点 = data.totalNodes，已完成 = data.completedSteps，执行进度 = data.executionProgress%；
    对话历史：取 data.response 或 data.finalState.response 展示。

    流式（body.stream=true）：SSE，先 type=init（含 graphData、totalNodes），中间穿插 type=step 与 type=token
    （router 图 chat 节点的 LLM 逐 token 增量，{nodeId:"chat", content:"..."}，可用于打字机效果），
    最后 type=done（含 finalState、steps、totalNodes、completedSteps、executionProgress、response）；
    前端按 step 播动画、按 token 拼字，收到 done 后以 response 为准做最终展示。
    """
    body = anyio.from_thread.run(read_json_optional, request) or {}
    graph_name = body.get("graph") or "router"
    stream = body.get("stream", False)
    input_state = body.get("input")
    if input_state is not None and not isinstance(input_state, dict):
        input_state = None
    if input_state is None:
        input_state = {}
    top_query = body.get("query")
    if top_query and (not input_state.get("query")):
        input_state = {**input_state, "query": top_query}
    if graph_name == "parallel" and (top_query or input_state.get("query")) and not input_state.get("input_text"):
        input_state = {**input_state, "input_text": (top_query or input_state.get("query", "")).strip() or "示例文本"}

    # 【上下文管理】生产用法：router 图传 threadId，服务端就用 checkpointer 跨请求记住对话（见
    # build_router_graph）；不传则每次都是全新会话（等价于旧行为，仍兼容 input.history 直传）。
    # loop/parallel 没接 checkpointer，传了也不影响它们，无需按图名特判。
    thread_id = body.get("threadId") or body.get("thread_id")

    if graph_name == "hitl":
        # hitl 走独立的暂停/恢复流程（interrupt），不复用 router/loop/parallel 的无状态 stream 收集逻辑。
        # 首次请求：body 传 {graph:"hitl", threadId, query}；命中 interrupt 后返回 waitingForInput=True + interrupt。
        # 第二次请求：body 传 {graph:"hitl", threadId（同一个）, resume: true/false/"编辑后的文本"}。
        thread_id = body.get("threadId") or body.get("thread_id") or "hitl-default"
        resume_value = body.get("resume")
        try:
            out = run_hitl_graph(input_state, thread_id, resume=resume_value)
        except Exception as e:
            return ({"code": 400, "msg": str(e), "data": {}}, 400)
        return {"code": 0, "msg": "ok", "data": out}

    builder_fn = GRAPH_BUILDERS.get(graph_name)
    if not builder_fn:
        return (
            {"code": 400, "msg": f"未知图: {graph_name}", "data": {"allowed": list(GRAPH_BUILDERS.keys())}},
            400,
        )
    graph = builder_fn()
    default = DEFAULT_INPUTS.get(graph_name, {})
    state = {**default, **input_state} if input_state else default.copy()
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    trace_state = {**state, "thread_id": thread_id} if thread_id else state  # 只为落 trace 用

    if stream:
        schema = graph_to_schema(graph)
        total_nodes = len(schema["nodes"])
        def gen():
            t_start = time.perf_counter()
            try:
                # 先发 graphData，方便前端画图
                yield f"data: {json.dumps({'type': 'init', 'graphData': {'nodes': schema['nodes'], 'edges': schema['edges']}, 'totalNodes': total_nodes}, ensure_ascii=False)}\n\n"
                for event_type, payload in run_graph_stream_yield_events(graph, state, config):
                    if event_type == "step":
                        yield f"data: {json.dumps({'type': 'step', 'step': payload}, ensure_ascii=False)}\n\n"
                    elif event_type == "token":
                        # LLM 逐 token 增量（目前仅 chat 节点会产出），前端可用来做打字机效果；
                        # 该节点完成后仍会有一次 step 事件带上拼接好的完整 output，token 只是过程量。
                        yield f"data: {json.dumps({'type': 'token', **payload}, ensure_ascii=False)}\n\n"
                    else:
                        # done：补充执行监控与对话用字段，便于前端显示进度和 finalState.response
                        _persist_trace(graph_name, trace_state, payload, round((time.perf_counter() - t_start) * 1000))
                        steps_list = payload.get("steps", [])
                        completed = len(steps_list)
                        progress = round((completed / total_nodes * 100), 1) if total_nodes else 0
                        done_data = {
                            **payload,
                            "totalNodes": total_nodes,
                            "completedSteps": completed,
                            "executionProgress": progress,
                            "response": (payload.get("finalState") or {}).get("response", ""),
                        }
                        yield f"data: {json.dumps({'type': 'done', **done_data}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                _persist_trace(graph_name, trace_state, None, round((time.perf_counter() - t_start) * 1000), error=str(e))
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        return StreamingResponse(
            gen(),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    out = run_graph_and_collect_steps(graph_name, input_state, thread_id)
    if out.get("error"):
        return ({"code": 400, "msg": out["error"], "data": out}, 400)
    return {"code": 0, "msg": "ok", "data": out}


# ---------------------------------------------------------------------------
# 入口：运行全部演示
# ---------------------------------------------------------------------------


def run_all_demos():
    """依次运行所有 LangGraph 功能演示。"""
    print("\n" + "=" * 60)
    print("  LangGraph 核心功能可视化演示")
    print("=" * 60 + "\n")

    demo_loop()
    print()

    demo_parallel()
    print()

    demo_state_management()
    print()

    demo_router()
    print()

    demo_memory()
    print()

    demo_hitl()
    print()

    # 用路由图做一次 stream 可视化
    router_graph = build_router_graph()
    print("📊 **实时执行监控示例（条件路由）**")
    visualize_execution(router_graph, {"query": "今天天气怎么样？", "intent": "", "response": ""}, sleep_sec=0.2)
    print()

    print("📊 **功能对比表**")
    print("| 功能       | 适用场景           | 复杂度 |")
    print("|------------|--------------------|--------|")
    print("| 循环       | 迭代优化、多轮对话 | ⭐⭐    |")
    print("| 并行       | 批量处理、多任务   | ⭐⭐⭐   |")
    print("| 条件路由   | 智能客服、分类器   | ⭐⭐    |")
    print("| 状态管理   | 长对话、工作流     | ⭐⭐⭐   |")


if __name__ == "__main__":
    run_all_demos()
