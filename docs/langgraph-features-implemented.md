# LangGraph 特性落地记录（service/ai/langchain.py）

本文档记录在 [service/ai/langchain.py](../service/ai/langchain.py) 里新增的 5 项 LangGraph 特性：每项对应
`docs/langgraph/` 下哪篇官方文档、用了哪些 API、接入了哪个图/节点、怎么通过 HTTP 验证。按实现优先级排列。

实现前的基线（已有能力，不在本次范围内）：`StateGraph` + 条件路由、`Send` 动态并行、`MemorySaver`
（仅演示用）、`stream_mode="updates"`、`graph.get_graph()` 动态生成前端 schema。

环境：`langgraph==1.0.8`（`langgraph-checkpoint==4.0.0`）。这个版本号很关键——第 4 节会解释为什么
"超时"没有实现。

---

## 1. 人机交互（HITL）：真正的 `interrupt()`

**对应文档**：[langgraph-interrupts.md](langgraph/langgraph-interrupts.md)

模块顶部注释早就写了"5. 人机交互节点 - AI 建议 → 人工审核 → 处理反馈"，但改造前 `GRAPH_BUILDERS` 里根本
没有这张图——纯粹是空承诺。现在补上了一个真正基于 `interrupt()` 的暂停/恢复流程。

**用到的 API**：`langgraph.types.interrupt`、`langgraph.types.Command`、`checkpointer`（必需，见下）。

**图结构**：`analyze`（AI 生成建议）→ `review`（`interrupt()` 暂停，等待人工）→ `process`（人工批准/编辑后执行）
或直接 `END`（拒绝）。

```python
# service/ai/langchain.py:749
def _hitl_review(state: HitlState) -> Command[Literal["process", "__end__"]]:
    decision = interrupt({
        "question": "是否批准以下 AI 建议？可直接批准/拒绝，或提交编辑后的文本作为最终建议。",
        "suggestion": state.get("suggestion", ""),
    })
    if isinstance(decision, str) and decision.strip():
        return Command(goto="process", update={"decision": "approved", "suggestion": decision.strip()})
    if decision:
        return Command(goto="process", update={"decision": "approved"})
    return Command(goto=END, update={"decision": "rejected", "response": "已拒绝该建议，未执行任何操作。"})
```

**踩到的坑（多 worker 下 `MemorySaver` 会直接把 HITL 做废）**：`interrupt()` 依赖 checkpointer 跨请求保存
"暂停在哪"这件事。最初用 `MemorySaver`（进程内存）验证时，单 worker 一切正常；切到生产配置的
`UVICORN_WORKERS=2` 后，"提交建议"和"恢复决定"这两次 HTTP 请求如果被负载均衡到不同 worker 进程，第二个
worker 的 `MemorySaver` 里根本没有那次暂停的记录，图会从头重新执行，人工审核形同虚设。改用共享文件的
`SqliteSaver` 后，两次请求无论落在哪个 worker 都读写同一份 SQLite 文件，问题消失。

```python
# service/ai/langchain.py:778
def _get_hitl_checkpointer() -> SqliteSaver:
    db_dir = ...  # <repo>/data/checkpoints/
    conn = sqlite3.connect(os.path.join(db_dir, "hitl.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # 幂等，首次调用建表
    return saver

def build_hitl_graph(checkpointer=None):
    ...
    return builder.compile(checkpointer=checkpointer or _get_hitl_checkpointer())

# 模块级单例：必须是同一个编译后的图 + 同一个 checkpointer 实例，
# 不能像 router/loop/parallel 那样每次请求都重新 build
_HITL_GRAPH = build_hitl_graph()
```

**HTTP 接口**：`POST /ai/langgraph/run`，`graph:"hitl"`。第一次请求触发 interrupt，第二次带上同一个
`threadId` + `resume` 完成恢复。

```bash
# 第一次：提交请求，命中 interrupt
curl -X POST http://localhost:3000/ai/langgraph/run -H "Content-Type: application/json" \
  -d '{"graph":"hitl","threadId":"demo-1","query":"帮我给客户回一封确认周三开会的邮件"}'
# → {"data": {"waitingForInput": true, "interrupt": {"suggestion": "..."}, "threadId": "demo-1"}}

# 第二次：批准（resume=true）/ 拒绝（resume=false）/ 编辑后采用（resume="改写后的文本"）
curl -X POST http://localhost:3000/ai/langgraph/run -H "Content-Type: application/json" \
  -d '{"graph":"hitl","threadId":"demo-1","resume":true}'
# → {"data": {"waitingForInput": false, "response": "✅ 已按审核通过的建议执行：..."}}
```

三条路径（批准/拒绝/编辑）以及 2-worker 部署下的跨进程恢复，均已用 curl 实测通过。

---

## 2. 长期记忆 Store（跨会话语义记忆）

**对应文档**：[langgraph-memory.md](langgraph/langgraph-memory.md)、[langgraph-persistence.md](langgraph/langgraph-persistence.md)（"内存存储"一节）

`MemorySaver`/`SqliteSaver` 这类 checkpointer 只解决"同一个 thread 内记住什么"（短期记忆）。这里加的是
**长期记忆**：同一个 `user_id`，换一个全新的 thread/会话，AI 依然记得之前说过的事实或偏好——用的是
`BaseStore`，一个和 checkpointer 平行的、按 `namespace` 组织的独立存储。

**用到的 API**：`langgraph.store.memory.InMemoryStore`（带语义索引）、在节点函数签名里声明
`store: BaseStore` 参数（LangGraph 自动注入，不用手动传）、`store.search(namespace, query=...)`（语义检索）、
`store.put(namespace, key, value)`（写入）。

```python
# service/ai/langchain.py:474
def _memory_embed(texts: list[str]) -> list[list[float]]:
    from service.ai.vector_db_qdrant import get_embedding  # 复用现成的 DashScope embedding
    return [get_embedding(t) for t in texts]

_LONG_TERM_STORE = InMemoryStore(index={"embed": _memory_embed, "dims": 1024})

# service/ai/langchain.py:633
def _chat_handler(state: RouterState, store: BaseStore) -> dict:  # store 由 LangGraph 自动注入
    user_id = state.get("user_id") or "demo-user"
    memories = _search_long_term_memory(store, user_id, query)     # namespace=(user_id, "memories")
    ...
    _maybe_write_long_term_memory(store, user_id, query)            # 热路径写入，见下

def build_router_graph():
    ...
    return builder.compile(store=_LONG_TERM_STORE)
```

采用的是文档里"记忆类型对照表"中的**语义记忆·集合**模式（每条事实是独立文档，而不是维护一份不断合并的
用户画像），写入方式是**热路径写入**：每轮对话后用 LLM 判断这句话里有没有"值得长期记住的事实/偏好"，有就
`store.put` 一条，没有就跳过。

```python
# service/ai/langchain.py:611
def _maybe_write_long_term_memory(store: BaseStore, user_id: str, query: str) -> None:
    extracted = _call_llm(f"用户说：{query}\n\n如果这句话包含...用一句话提炼；如果没有，只回复：无", ...)
    if extracted and extracted not in ("无", "无。", "没有", "没有。"):
        store.put((user_id, "memories"), str(uuid.uuid4()), {"fact": extracted, "source_query": query})
```

**验证**（两次完全独立的 HTTP 请求，中间没有任何共享 thread/session）：

```bash
curl -X POST .../ai/langgraph/run -d '{"graph":"router","input":{"query":"我对海鲜过敏，以后推荐吃的东西要避开","user_id":"u1"}}'
curl -X POST .../ai/langgraph/run -d '{"graph":"router","input":{"query":"提醒一下，我之前跟你说过我对什么过敏？","user_id":"u1"}}'
# → "你之前跟我说过你对海鲜过敏，我会注意避开海鲜相关的内容。"
```

namespace 隔离也做了针对性验证：用 4 个全新 `user_id` 问一个模型**猜不出来**的问题（"我养的宠物叫什么名字"），
全部正确回答"你没告诉过我"——排除了"其实是 LLM 瞎编巧合撞对"的可能性，证明检索确实按 `user_id` 隔离。

**已知限制（没有隐藏）**：`InMemoryStore` 和 HITL 最初的 `MemorySaver` 是同一类问题——进程内存，多 worker
部署下各进程互不可见，记忆没法跨进程共享。这里**没有**像 HITL 那样换成持久化实现，原因是官方目前只提供
`langgraph-store-postgres`，而本项目数据库是 MySQL，要做对应就得自己实现一套 `BaseStore` 接口（`put`/`get`/
`search`，其中 `search` 还要支持向量相似度），工作量和这次任务的"打通功能演示"目标不成比例，所以先保留
`InMemoryStore`（也是官方文档示例本身的选择），把这个限制记录在这里，作为后续可选项。

---

## 3. LLM token 级流式（自定义流 + `get_stream_writer`）

**对应文档**：[langgraph-streaming.md](langgraph/langgraph-streaming.md)（"与任何LLM一起使用"一节）

这里**没有**用文档里最常见的 `stream_mode="messages"`——那个模式是 LangGraph 自动埋点 LangChain 聊天模型
对象的输出，而本项目用的是原始 DashScope SDK（`dashscope.Generation.call`），不是 LangChain 的
`ChatOpenAI`/`ChatTongyi`。文档专门有一节讲这种情况怎么办：`stream_mode="custom"` + 节点内手动
`get_stream_writer()` 转发，对接**任意**自带流式接口的客户端。

**用到的 API**：`langgraph.config.get_stream_writer`、`graph.stream(state, stream_mode=["updates","custom"])`。

```python
# service/ai/langchain.py:82
def _call_llm_messages_stream(messages: list, model: str = DEFAULT_CHAT_MODEL):
    """DashScope 原生流式（stream=True, incremental_output=True），逐段 yield 增量文本。"""
    responses = dashscope.Generation.call(model=model, messages=messages, stream=True, incremental_output=True)
    for resp in responses:
        ...
        yield str(text)  # 或 choices[0].message.content

# service/ai/langchain.py:633（_chat_handler 内）
writer = get_stream_writer()
parts = []
for delta in _call_llm_messages_stream(messages):
    parts.append(delta)
    writer({"nodeId": "chat", "content": delta})
response = "".join(parts).strip()
```

`get_stream_writer()` 在非流式调用（`.invoke()` 或 `stream_mode="updates"`）下是安全的 no-op——已经用一个
最小 toy graph 验证过：`.invoke()` 正常返回，`stream_mode="updates"` 拿不到任何 custom 事件；只有
`stream_mode` 显式包含 `"custom"` 时才会收到。这意味着 `_chat_handler` 不用为"是否在流式请求里"写任何分支，
一套代码两种模式都对。

SSE 层新增了 `type=token` 事件（在 `type=step` 之间穿插），前端可以用来做打字机效果；节点整体完成后仍会有
一次 `type=step` 带上拼接好的完整文本，`token` 只是过程量，最终以 `step`/`done` 里的 `response` 为准。

```python
# service/ai/langchain.py:1108
for mode, payload in graph.stream(state, stream_mode=["updates", "custom"]):
    if mode == "custom":
        yield ("token", payload)
        continue
    for node_id, output in payload.items():
        ...
        yield ("step", step_payload)
```

**验证**（`stream:true`，SSE）：

```bash
curl -N -X POST .../ai/langgraph/run -d '{"graph":"router","stream":true,"input":{"query":"用一句话介绍一下杭州"}}'
```

```
data: {"type": "step", "step": {"nodeId": "classify", ...}}
data: {"type": "token", "nodeId": "chat", "content": "杭州"}
data: {"type": "token", "nodeId": "chat", "content": "是"}
data: {"type": "token", "nodeId": "chat", "content": "浙江省"}
...
data: {"type": "step", "step": {"nodeId": "chat", "output": {"response": "💭 杭州是浙江省的省会，..."}}}
data: {"type": "done", ...}
```

拼接所有 `token.content` 与最终 `step.output.response` 完全一致；非流式 JSON 请求（`stream` 不传/`false`）
行为不受影响，同样验证过。

---

## 4. 容错：`RetryPolicy`（节点级重试）

**对应文档**：[langgraph-fault-tolerance.md](langgraph/langgraph-fault-tolerance.md)

**用到的 API**：`langgraph.types.RetryPolicy`，`add_node(..., retry_policy=RetryPolicy(max_attempts=3))`。

给所有会发外部网络请求的节点加了重试：router 的 `weather`（高德 API）/`news`（RSS + LLM）/`chat`（DashScope）/
`insight`（内部走子图，同样会打 LLM）、loop 的 `think`/`respond`、parallel 的 `sentiment`/`keywords`/`summary`、
hitl 的 `analyze`。纯逻辑节点（`classify`/`decide`/`aggregate`/`dispatch`）没有外部调用，没有加。

```python
# service/ai/langchain.py:674
builder.add_node("weather", _weather_handler, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("news", _news_handler, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("chat", _chat_handler, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("insight", _analyze_via_subgraph, retry_policy=RetryPolicy(max_attempts=3))
```

HITL 的 `review` 节点**没有**加 `retry_policy`——它内部调用 `interrupt()`，文档明确说 `interrupt()` 走的是
`GraphBubbleUp` 机制而不是普通异常，给一个包含 `interrupt()` 的节点配重试语义上很奇怪（会不会重复弹出审核？
副作用是否幂等？），所以刻意跳过。

**用一个独立 toy graph 验证过重试机制本身确实生效**（节点前两次抛异常，第三次成功）：

```python
def flaky(state):
    _counter['n'] += 1
    if _counter['n'] < 3:
        raise ConnectionError('模拟网络抖动')
    return {'tries': _counter['n']}

builder.add_node('flaky', flaky, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01))
# g.invoke(...) → {'tries': 3}，实际尝试次数 3（重试了 2 次）
```

**没有实现超时（`TimeoutPolicy`）**——原本计划里这部分也在"生产级"范围内，但文档第一行就写着
"（需要 `langgraph>=1.2`，目前处于 alpha 阶段）"。本项目当前锁定的是 `langgraph==1.0.8`，
`from langgraph.types import TimeoutPolicy` 直接 `ImportError`，`add_node()` 的签名里也确实没有 `timeout=`
参数（不是被 guard 掉，是根本不存在）。要用这个特性得把 `langgraph` 升到 1.2+（还是 alpha），但这个包是
`service/ai/` 下十几个模块的共享依赖（agent、text2sql、function_call 等都在用），为了一个 demo 模块的 alpha
特性去动一个被广泛依赖的核心包版本，风险和收益不成比例，所以没有做。等 `langgraph>=1.2` 转正后可以补上：
`add_node("chat", _chat_handler, timeout=TimeoutPolicy(run_timeout=30))`。

`错误处理`（`error_handler=`、`NodeError`）同样需要 `langgraph>=1.2`，同样未实现，原因相同。

---

## 5. 子图组合（不同状态模式）

**对应文档**：[langgraph-subgraphs.md](langgraph/langgraph-subgraphs.md)（"在节点内调用子图"一节）

router 图新增一个 `insight` 意图（关键词"分析"触发），内部**调用 `parallel` 图作为子图**。这是子图组合的
两种模式之一：因为 `RouterState`（`query`/`intent`/`response`）和 `ParallelState`（`input_text`/`analyses`/
`final_result`/`response`）状态键不同，不能直接 `add_node(subgraph)` 共享通道，只能在节点函数内部手动
`invoke` 子图、手动做两边状态的转换。

```python
# service/ai/langchain.py:509
def _analyze_via_subgraph(state: RouterState) -> dict:
    query = (state.get("query") or "").strip()
    subgraph = build_parallel_graph()
    sub_output = subgraph.invoke({"input_text": query, "analyses": [], "final_result": "", "response": ""})
    return {"response": f"🧩 [子图: parallel] \n{sub_output.get('response', '')}"}

# service/ai/langchain.py:674
builder.add_node("insight", _analyze_via_subgraph, retry_policy=RetryPolicy(max_attempts=3))
builder.add_conditional_edges("classify", lambda s: s["intent"],
    {"weather": "weather", "news": "news", "chat": "chat", "insight": "insight"})
```

**验证**：

```bash
curl -X POST .../ai/langgraph/run -d '{"graph":"router","input":{"query":"帮我分析一下：这家餐厅服务差但菜很好吃"}}'
```

`classify` 正确识别 `intent=insight`，`insight` 节点内部跑完 `parallel` 子图的
`sentiment`/`keywords`/`summary` → `aggregate` 全流程，结果正确映射回 `RouterState.response`：

```
🧩 [子图: parallel]
关键词：restaurant, service, food, poor, good
情感：neutral
摘要：餐厅服务差但菜品美味。
```

前端 `GET /ai/langgraph/graph?name=router` 的 schema 里也能看到 `insight` 节点（图标 🧩，类型 tool），
`classify → insight → output` 的边。

---

## 快速验证清单

```bash
# 1. HITL：提交 → 批准/拒绝/编辑
curl -X POST http://localhost:3000/ai/langgraph/run -d '{"graph":"hitl","threadId":"t1","query":"..."}'
curl -X POST http://localhost:3000/ai/langgraph/run -d '{"graph":"hitl","threadId":"t1","resume":true}'

# 2. 长期记忆：同一 user_id 跨请求召回
curl -X POST http://localhost:3000/ai/langgraph/run -d '{"graph":"router","input":{"query":"我叫小明","user_id":"u1"}}'
curl -X POST http://localhost:3000/ai/langgraph/run -d '{"graph":"router","input":{"query":"我叫什么名字？","user_id":"u1"}}'

# 3. token 流式
curl -N -X POST http://localhost:3000/ai/langgraph/run -d '{"graph":"router","stream":true,"input":{"query":"..."}}'

# 4. 重试：无独立 HTTP 触发方式（网络抖动不可控），已用 toy graph 单测验证机制本身

# 5. 子图：insight 意图
curl -X POST http://localhost:3000/ai/langgraph/run -d '{"graph":"router","input":{"query":"帮我分析一下：..."}}'
```

## 涉及的文件

| 文件 | 改动 |
|---|---|
| [service/ai/langchain.py](../service/ai/langchain.py) | 全部 5 项特性的实现 |
| [requirements.txt](../requirements.txt) | 新增 `langgraph-checkpoint-sqlite` |
| [.gitignore](../.gitignore) | 新增 `data/checkpoints/`（HITL 的 SQLite 运行时文件，不提交） |
| `data/checkpoints/hitl.sqlite` | 运行时生成，HITL 的共享 checkpointer 文件 |
