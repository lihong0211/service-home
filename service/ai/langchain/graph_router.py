"""
4. 条件路由 - 意图识别与多路分发（chat 分支接入长期记忆 Store）

【模块 1／8：上下文管理】+【模块 2／8：记忆管理】的生产级实现，完整设计说明见
service/ai/AGENT_ARCHITECTURE.md 对应两节。
"""

import json
import logging
import operator
import os
import sqlite3
import uuid
from typing import Annotated, TypedDict

import dashscope

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import RetryPolicy

from config.ai import DEFAULT_CHAT_MODEL
from service.ai.langchain.common import (
    _call_llm,
    _call_llm_messages_stream,
    _gaode_geocode_adcode,
    _get_gaode_weather,
)
from service.ai.langchain.graph_parallel import build_parallel_graph

logger = logging.getLogger(__name__)

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

def _memory_embed(texts: list[str]) -> list[list[float]]:
    """【记忆管理】长期记忆的语义索引向量化。复用 vector_db_qdrant 现成的 DashScope embedding，不重新实现一套。"""
    from service.ai.rag.vector_db_qdrant import get_embedding

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
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "checkpoints")
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
