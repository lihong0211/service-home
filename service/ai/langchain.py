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
import operator
import os
import time
from datetime import datetime
from typing import Annotated, TypedDict

import dashscope
import requests

# LangGraph 图与状态
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
_GAODE_API_KEY = os.getenv("AMAP_MAPS_API_KEY")

# 多轮对话：保留的轮数/条数上限（真正意义的长对话）
MAX_HISTORY_MESSAGES = 50   # router 闲聊等传入 LLM 的最近消息条数（约 25 轮）
MAX_HISTORY_TURNS_CONTEXT = 20  # 拼进 prompt 的「上文」最近轮数（think/respond/parallel）


# ---------------------------------------------------------------------------
# 公共 LLM 调用 helper（Qwen via Dashscope）
# ---------------------------------------------------------------------------

def _call_llm_messages(messages: list, model: str = "qwen-turbo") -> str:
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


def _call_llm(prompt: str, system: str = "你是一个专业的AI助手，请简洁准确地回答。", model: str = "qwen-turbo") -> str:
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


def _format_history_context(history: list, max_turns: int | None = None, max_chars_per_msg: int = 300) -> str:
    """把 history 格式化为「上文」文本，供各节点拼进 prompt。默认保留最近 MAX_HISTORY_TURNS_CONTEXT 轮。"""
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
    """创建带循环的图：think → decide → (think | respond) → respond → END。"""
    builder = StateGraph(AgentState)
    builder.add_node("think", _think)
    builder.add_node("decide", _decide)
    builder.add_node("respond", _loop_respond)
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
    builder.add_node("sentiment", _sentiment_analysis)
    builder.add_node("keywords", _keyword_extraction)
    builder.add_node("summary", _text_summary)
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
# 4. 条件路由 - 意图识别与多路分发
# ---------------------------------------------------------------------------


class RouterState(TypedDict):
    query: str
    intent: str
    response: str


def _classify_intent(state: RouterState) -> dict:
    query = (state.get("query") or "").strip()
    q = query.lower()
    if "天气" in q:
        intent = "weather"
    elif any(k in q for k in ("股票", "a股", "股市", "行情", "大盘", "涨停", "跌停", "沪指", "深指")):
        intent = "stock"
    elif "新闻" in q:
        intent = "news"
    else:
        intent = "chat"
    print(f"🎯 意图识别: {intent}")
    return {"intent": intent}


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


def _stock_handler(state: RouterState) -> dict:
    query = state.get("query", "")
    symbol = _call_llm(
        f"从这句话中提取股票代码或公司名称（A股六位代码优先），只返回股票代码或名称，识别不到则返回贵州茅台：{query}",
        system="只输出股票代码或股票名称，不要其他内容。",
    ).strip()
    print(f"📈 查询股票: {symbol}")
    # 东方财富接口经代理常被断开，请求时临时绕过代理
    _saved_proxy = {
        k: os.environ.pop(k, None)
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
    }
    try:
        import akshare as ak
        if symbol.isdigit() and len(symbol) == 6:
            # 直接用六位代码
            df = ak.stock_individual_info_em(symbol=symbol)
            info = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            response = f"📈 股票 {symbol}：{json.dumps(info, ensure_ascii=False)[:200]}"
        else:
            # 按名称在实时行情里搜索
            spot_df = ak.stock_zh_a_spot_em()
            matched = spot_df[spot_df["名称"].str.contains(symbol, na=False)]
            if matched.empty:
                matched = spot_df[spot_df["代码"].str.contains(symbol, na=False)]
            if not matched.empty:
                row = matched.iloc[0]
                response = (
                    f"📈 {row.get('名称', '')}（{row.get('代码', '')}）"
                    f"最新价：{row.get('最新价', '')}，"
                    f"涨跌幅：{row.get('涨跌幅', '')}%，"
                    f"成交量：{row.get('成交量', '')}手"
                )
            else:
                response = f"未找到匹配股票：{symbol}"
    except Exception as e:
        response = f"股票查询失败：{e}"
    finally:
        for k, v in _saved_proxy.items():
            if v is not None:
                os.environ[k] = v
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


def _chat_handler(state: RouterState) -> dict:
    query = (state.get("query") or "").strip()
    history = state.get("history") or []
    if history and isinstance(history, list) and len(history) > 0:
        # 多轮：保留最近 MAX_HISTORY_MESSAGES 条，拼成 messages 传入模型
        recent = history[-MAX_HISTORY_MESSAGES:]
        messages = [{"role": "system", "content": "你是一个友善的AI助手，用中文简洁地回答，并结合完整上文语境进行多轮对话。"}]
        for h in recent:
            role = (h.get("role") or "user").lower()
            if role not in ("user", "assistant"):
                continue
            content = (h.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})
        response = _call_llm_messages(messages)
    else:
        response = _call_llm(query, system="你是一个友善的AI助手，用中文简洁地回答用户问题。")
    return {"response": f"💭 {response}"}


def build_router_graph():
    """条件路由：classify → weather | stock | news | chat → END。"""
    builder = StateGraph(RouterState)
    builder.add_node("classify", _classify_intent)
    builder.add_node("weather", _weather_handler)
    builder.add_node("stock", _stock_handler)
    builder.add_node("news", _news_handler)
    builder.add_node("chat", _chat_handler)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        lambda s: s["intent"],
        {"weather": "weather", "stock": "stock", "news": "news", "chat": "chat"},
    )
    for name in ["weather", "stock", "news", "chat"]:
        builder.add_edge(name, END)

    return builder.compile()


def demo_router():
    """演示条件路由并打印 ASCII 图。"""
    graph = build_router_graph()
    print("📊 **智能路由流程图**")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("  (图结构: classify → weather|stock|news|chat → END)")
    print()
    for q in ["今天天气怎么样？", "有什么新闻？", "随便聊聊"]:
        out = graph.invoke({"query": q, "intent": "", "response": ""})
        print(f"  query={q!r} → response={out.get('response', '')}")
    return graph


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
    "stock": "📈",
    "news": "📰",
    "chat": "💭",
    "sentiment": "😊",   # 情感分析
    "keywords": "🏷️",   # 关键词
    "summary": "📝",    # 摘要
    "review": "👤",
    "dispatch": "📤",
    "respond": "📢",
}

# 节点 id -> 前端展示（可选覆盖），未列出的用 raw_id、type=process
NODE_DISPLAY = {
    "__start__": {"name": "用户输入", "type": "input", "icon": "📝", "description": "入口"},
    "__end__": {"name": "输出", "type": "output", "icon": "📢", "description": "出口"},
    "classify": {"name": "意图分类", "type": "llm", "description": "分析用户意图"},
    "weather": {"name": "天气", "type": "tool", "description": "天气查询"},
    "stock": {"name": "股票", "type": "tool", "description": "股票信息"},
    "news": {"name": "新闻", "type": "tool", "description": "新闻摘要"},
    "chat": {"name": "闲聊", "type": "llm", "description": "通用对话"},
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


def run_graph_stream_and_collect(graph, state: dict):
    """
    执行图 stream 一次，收集每一步的 nodeId、耗时、输出，并从各步输出合并出最终 state。
    不再二次 invoke，避免流程跑两遍、最终回答提前打印。
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
    for step in graph.stream(state, stream_mode="updates"):
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
    return extra


def run_graph_stream_yield_events(graph, state: dict):
    """
    执行图 stream，每完成一步 yield 一个 step 事件，最后 yield 一个 done 事件。
    供 SSE 流式接口使用：前端先按步更新流程动画，收到 done 后再展示回答，避免「回答比流程快」。
    loop 图每步会带 iteration/thought/nextStep/response/label 等字段，便于前端展示「第 N 轮思考」。
    yield: ("step", { stepIndex, nodeId, status, duration_ms, output, ...enrich }) 或 ("done", ...)。
    """
    steps = []
    execution_order = []
    t0 = time.perf_counter()
    step_index = 0
    current_state = dict(state)
    for step in graph.stream(state, stream_mode="updates"):
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
}

DEFAULT_INPUTS = {
    "router": {"query": "今天天气怎么样？", "intent": "", "response": ""},
    "loop": {"messages": [], "next_step": "", "iteration": 0, "query": "", "response": ""},
    "parallel": {"input_text": "示例文本", "analyses": [], "final_result": "", "response": ""},
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


def run_graph_and_collect_steps(graph_name: str, input_state: dict | None = None):
    """
    执行指定图，收集每一步的 nodeId、耗时、输出，供前端按真实执行顺序与节奏驱动 3D 动画。
    返回：{
        "graphData": { nodes, edges, executionOrder },
        "steps": [ { "nodeId", "status": "end", "duration_ms", "output" }, ... ],
        "finalState": { ... },
        "executionOrder": [ "classify", "weather", ... ]
    }
    前端传入的 input 会与当前图的默认 state 合并，避免切图后残留字段导致缺键报错（如 loop 下误传 query 等）。
    多轮对话：input 中可带 history: [{role:"user", content:"..."}, {role:"assistant", content:"..."}, ...]，
    router/loop/parallel 均会使用该上文语境。
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
    try:
        run_result = run_graph_stream_and_collect(graph, state)
    except Exception as e:
        return {"error": str(e)}
    schema = graph_to_schema(graph)
    steps = run_result["steps"]
    execution_order = run_result["executionOrder"]
    total = run_result.get("totalSteps", len(steps))
    return {
        "graphData": {
            "nodes": schema["nodes"],
            "edges": schema["edges"],
            "executionOrder": execution_order,
        },
        "steps": steps,
        "finalState": run_result["finalState"],
        "executionOrder": execution_order,
        "totalSteps": total,
    }


# ---------------------------------------------------------------------------
# Flask 视图：供 routes/ai 注册 GET/POST
# ---------------------------------------------------------------------------


def langgraph_graph_api():
    """GET /ai/langgraph/graph?name=router 返回图结构，供前端 3D 可视化（GraphData）。"""
    from flask import request, jsonify

    name = request.args.get("name") or "router"
    schema = get_graph_schema(name)
    if schema is None:
        return (
            jsonify(
                {
                    "code": 400,
                    "msg": f"未知图: {name}",
                    "data": {"allowed": list_graph_names()},
                }
            ),
            400,
        )
    return jsonify({"code": 0, "msg": "ok", "data": schema})


def langgraph_run_api():
    """
    POST /ai/langgraph/run 执行图并返回步骤与最终状态，供前端按真实执行顺序驱动 3D 动画。

    router / loop / parallel 均存在「先答案、后流程」问题：非流式时一次返回 steps+finalState，
    前端若先渲染 finalState.response 再播步骤动画，就会看到答案比流程快。

    解决：请求体传 "stream": true，改为 SSE 流式：
    - 先依次推送 type=step（每步 nodeId、duration_ms、output 等）
    - 最后推送 type=done（含 finalState、totalSteps、steps）
    前端应：按 step 更新流程动画，仅在收到 type=done 后再展示 finalState.response。
    """
    from flask import request, jsonify, Response, stream_with_context

    body = request.get_json() or {}
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

    builder_fn = GRAPH_BUILDERS.get(graph_name)
    if not builder_fn:
        return jsonify({"code": 400, "msg": f"未知图: {graph_name}", "data": {"allowed": list(GRAPH_BUILDERS.keys())}}), 400
    graph = builder_fn()
    default = DEFAULT_INPUTS.get(graph_name, {})
    state = {**default, **input_state} if input_state else default.copy()

    if stream:
        schema = graph_to_schema(graph)
        def gen():
            try:
                # 先发 graphData，方便前端画图
                yield f"data: {json.dumps({'type': 'init', 'graphData': {'nodes': schema['nodes'], 'edges': schema['edges']}}, ensure_ascii=False)}\n\n"
                for event_type, payload in run_graph_stream_yield_events(graph, state):
                    if event_type == "step":
                        yield f"data: {json.dumps({'type': 'step', 'step': payload}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'done', **payload}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        return Response(
            stream_with_context(gen()),
            mimetype="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    out = run_graph_and_collect_steps(graph_name, input_state)
    if out.get("error"):
        return jsonify({"code": 400, "msg": out["error"], "data": out}), 400
    return jsonify({"code": 0, "msg": "ok", "data": out})


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
