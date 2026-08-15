# Agent 生产化架构说明

本文档描述 [`service/ai/langchain.py`](./langchain.py)（LangGraph 演示服务，`router`/`loop`/`parallel`/`hitl` 四张图）里
"一个生产级 Agent 应该具备的 8 个能力模块"分别是怎么实现的、代码在哪、已知边界在哪。

配套建的数据表：[`model/ai/agent_trace.py`](../../model/ai/agent_trace.py)（可观测性/回流）、
[`model/ai/agent_booking.py`](../../model/ai/agent_booking.py)（副作用回滚演示）。
路由注册：[`routes/ai.py`](../../routes/ai.py) 的 `/ai/langgraph/*` 一段。

代码里每处相关实现都用 `【模块 N／8：xxx】` 的注释标出，搜这个关键字能快速定位。

| # | 模块 | 生产化程度 | 核心代码 |
|---|---|---|---|
| 1 | 上下文管理 | ✅ 生产级（仅 router 图 chat 节点） | `_build_context_window` / `_chat_handler` / `build_router_graph` |
| 2 | 记忆管理 | ✅ 已实现（单进程 store，生产要换 DB-backed） | `_search_long_term_memory` / `_maybe_write_long_term_memory` |
| 3 | 重试机制 | ✅ 已实现 | 各 `add_node(..., retry_policy=RetryPolicy(...))` |
| 4 | 可观测性 | ✅ 已实现 | `_persist_trace` / `model/ai/agent_trace.py` |
| 5 | 回流机制 | ✅ 采集+挖掘已实现（回流目的地是人工/离线流程） | `submit_trace_feedback` / `list_bad_cases` |
| 6 | 权限控制 | ✅ 机制已实现（forbidden 档暂无真实节点） | `NODE_PERMISSIONS` / `check_node_permission` |
| 7 | 人工处理 | ✅ 已实现 | `_hitl_review` 的 `interrupt()` |
| 8 | 副作用回滚 | ✅ 补偿事务（Saga）已实现，真实 DB 副作用验证过 | `run_saga` / `SagaStep` |

---

## 1. 上下文管理

**解决什么问题**：同一次会话内，模型要"记得"前面聊过什么，又不能无限把全部历史堆进 prompt——
会超出模型上下文窗口、拖慢响应、推高成本。

**现在怎么做的**（[langchain.py:219-296](./langchain.py#L219)，节点见 [`_chat_handler`](./langchain.py#L745)）：

1. **真实 token 预算，不是拍脑袋的条数/字符数**——[`_count_tokens`](./langchain.py#L224) 用
   `dashscope.get_tokenizer(DEFAULT_CHAT_MODEL)` 本地精确分词（离线、和线上模型分词规则一致），
   `MAX_CONTEXT_TOKENS = 2000` 是历史对话的预算上限。
2. **超出预算的旧对话不是丢弃，是增量摘要压缩**——[`_build_context_window`](./langchain.py#L266)
   从最新往前数，凑够 token 预算内的整段保留原文，更早的部分调用
   [`_summarize_messages`](./langchain.py#L242) 压成一段摘要，叠加在已有摘要之上（不是每轮
   重新摘要全部历史）。
3. **服务端持有会话状态，不靠前端重传全部历史**——[`build_router_graph`](./langchain.py#L824)
   给 router 图挂了 [`_CONTEXT_CHECKPOINTER`](./langchain.py#L821)（`SqliteSaver`，与 HITL 用的
   是同一个"多 worker 部署必须用共享文件、不能用 MemorySaver"的模式，见
   [`_get_context_checkpointer`](./langchain.py#L803) 的注释）。`RouterState` 新增
   `messages`（append 语义）/`context_summary`（覆盖语义）两个 checkpointer 持久化字段。

**调用方式变化**：

```bash
# 生产用法：只传 threadId，服务端自动记住上下文
curl -X POST http://localhost:3000/ai/langgraph/run \
  -d '{"graph":"router","threadId":"user-42-session-1","input":{"query":"我叫小明"}}'
curl -X POST http://localhost:3000/ai/langgraph/run \
  -d '{"graph":"router","threadId":"user-42-session-1","input":{"query":"我叫什么名字？"}}'
# 第二轮应正确回答"小明"，且请求体里完全没有传 history
```

旧调用方式（`input.history` 直传全部历史、不传 `threadId`）仍然兼容——`_chat_handler` 里
"服务端没有持久化记录时才退回 `history` 兼容路径"，两个信源不会同时生效打架。

**已知边界（诚实写明，没有藏）**：

- 只覆盖 router 图的 `chat` 节点。`loop`/`parallel` 是单轮演示图（无真正多轮对话场景），仍用
  原来的 [`_format_history_context`](./langchain.py#L296)（简单截断），没有接 token 预算/摘要/
  checkpointer——这是有意的范围收窄，不是遗漏。
- `weather`/`news`/`insight` 三个 router 分支不产出 `messages`，走这几个分支时上下文不会累积
  （符合直觉：查天气不需要记住"这是第几轮对话"）。

---

## 2. 记忆管理

**解决什么问题**：跨会话记住"这个用户是谁、有什么偏好"——和上下文管理的区别是，这个信息哪怕
换一个全新会话、隔了很久，也该被记得。

**现在怎么做的**（[langchain.py:564-593](./langchain.py#L564)，节点内调用见 `_chat_handler`）：

- `InMemoryStore`（[`_LONG_TERM_STORE`](./langchain.py#L592)）按 `(user_id, "memories")`
  namespace 隔离，语义检索用 DashScope embedding（[`_memory_embed`](./langchain.py#L581)，复用
  `vector_db_qdrant.get_embedding`，没有另起一套）。
- 写入是"热路径写入"：每轮对话后让 LLM 判断这句话有没有值得记住的事实/偏好，有就
  `store.put` 一条（[`_maybe_write_long_term_memory`](./langchain.py#L723)），集合式存储（每条
  独立事实），不是维护一份不断合并的用户画像。
- 读取是语义检索 Top-K（[`_search_long_term_memory`](./langchain.py#L711)），拼进 system prompt。

**已知边界**：`InMemoryStore` 是进程内存，多 worker 部署下各进程互不可见，记忆不能跨进程共享
（和 HITL 最初踩的坑是同一类问题，见下面第 7 节）。这里**没有**像 HITL 那样换成持久化实现，
原因是官方目前只提供 `langgraph-store-postgres`，本项目数据库是 MySQL，要用得自己实现一套
`BaseStore` 接口（含向量相似度检索），工作量较大，留作后续可选项。

---

## 3. 重试机制

**解决什么问题**：网络抖动、第三方 API 限流这类瞬时故障，不应该让整次请求直接失败。

**现在怎么做的**：`langgraph.types.RetryPolicy(max_attempts=3)`，挂在每一个**会发外部网络请求**
的节点上——router 的 `weather`（高德 API）/`news`（RSS+LLM）/`chat`（DashScope）/`insight`
（内部子图，同样打 LLM），loop 的 `think`/`respond`（见 [`build_loop_graph`](./langchain.py#L358)
的模块注释），parallel 的 `sentiment`/`keywords`/`summary`，HITL 的 `analyze`。

**统一策略**：纯逻辑节点（`classify`/`decide`/`aggregate`/`dispatch`）不加重试——重试解决的是
"这次调用本身没问题、只是网络抖了一下"，纯逻辑节点没有这类故障源，加了没有意义。

**已知边界**：HITL 的 `review` 节点**没有**加重试——它内部调用 `interrupt()`，走的是
`GraphBubbleUp` 机制而不是普通异常，给一个包含 `interrupt()` 的节点配重试语义上很奇怪（会不会
重复弹出审核？副作用是否幂等？），刻意跳过。没有实现节点级超时（`TimeoutPolicy`）——该特性需要
`langgraph>=1.2`（目前 alpha），本项目锁定 `langgraph==1.0.8`，为了一个 demo 特性升级被十几个
模块共享依赖的核心包，风险收益不成比例。详见 [`docs/langgraph-features-implemented.md`](../../docs/langgraph-features-implemented.md) 第 4 节。

---

## 4. 可观测性

**解决什么问题**：线上出问题（慢、错、答非所问）时，得有地方能查"这次请求具体经过了哪些步骤、
每步耗时多少、最后是不是失败了"，不能只靠猜。

**现在怎么做的**（[langchain.py:1526-1545](./langchain.py#L1526)，表结构见
[`model/ai/agent_trace.py`](../../model/ai/agent_trace.py)）：

- [`_persist_trace`](./langchain.py#L1526) 挂在两个出口：`run_graph_and_collect_steps`（非流式）
  和 `langgraph_run_api` 的 SSE `gen()`（流式，done/error 分支都覆盖）——**一次图执行落一条记录**，
  不是每个节点一条，避免表膨胀；`steps_detail` 字段里仍保留每步 `nodeId`+耗时的 JSON，需要展开
  排查时能查到节点粒度。
- 用独立 `SessionLocal()` 写库，不借用请求级 `db.session`：SSE 那条路径落库时机在
  `StreamingResponse` 对象返回之后，此时 `_ai_route` 已经清了请求级 session，借用会直接抛
  `RuntimeError`（这是趟过的一个坑，见函数内注释）。

**怎么验证**：

```bash
curl -X POST http://localhost:3000/ai/langgraph/run \
  -d '{"graph":"router","input":{"query":"用一句话介绍杭州"}}'
# 然后查 MySQL：SELECT * FROM agent_trace ORDER BY id DESC LIMIT 1;
```

---

## 5. 回流机制

**解决什么问题**：Agent 上线不是终点，得把线上真实交互数据回流回去，形成"用得越多效果越好"的
正循环，而不是每次出问题都靠人肉巡查才发现。

**现在怎么做的**（[langchain.py:1569-1665](./langchain.py#L1569)）：

- **采集**：[`submit_trace_feedback`](./langchain.py#L1569) 给一条已存在的 trace 打显式反馈
  （`good`/`bad` + 备注），对应 `agent_trace` 表新增的 `feedback`/`feedback_note` 字段。
- **挖掘**：[`list_bad_cases`](./langchain.py#L1598) 拉取 bad case 候选池——`status='error'`
  （系统自己判定失败）或 `feedback='bad'`（人工/用户标注不满意）任一命中即算。
- **端点**：`POST /ai/langgraph/trace/feedback`、`GET /ai/langgraph/trace/bad-cases?graph=router`
  （见 [`trace_feedback_api`](./langchain.py#L1641)/[`trace_bad_cases_api`](./langchain.py#L1658)，
  注册在 [`routes/ai.py`](../../routes/ai.py)）。

**怎么验证**：

```bash
# 1. 跑一次得到 traceId（查 agent_trace 表最新一行的 id）
# 2. 打反馈
curl -X POST http://localhost:3000/ai/langgraph/trace/feedback \
  -d '{"traceId": 1, "rating": "bad", "note": "答非所问"}'
# 3. 查 bad case 池
curl "http://localhost:3000/ai/langgraph/trace/bad-cases?graph=router"
```

**诚实的范围边界**：这里只做实了飞轮的"采集"和"挖掘"两步。"回流去哪"——把 bad case 对应的
正确知识补进知识库、把典型坏案例提炼成新的 prompt 约束、把人工修正的高质量答案攒成训练/评估
集——这三处目的地本身是人工判断或离线批处理的工作，不是这几个函数的职责，本次没有实现自动化
接入。`list_bad_cases` 的返回结构已经设计成可以直接喂给人工复核界面或离线脚本。

---

## 6. 权限控制

**解决什么问题**：不是所有节点都能自由执行——有副作用、不可逆的操作，得有一道机制拦下来，而
不是模型说执行就执行。

**现在怎么做的**（[langchain.py:892-902](./langchain.py#L892)）：

三档权限，[`NODE_PERMISSIONS`](./langchain.py#L892) 是全项目节点到档位的映射表：

- `readonly`（自由执行）：本项目里 `weather`/`news`/`chat`/`insight`/`think`/`decide`/
  `respond`/`sentiment`/`keywords`/`summary`/`aggregate`/`analyze` 全部属于这档——它们只调用
  LLM 或查询只读外部 API，没有任何写操作，天然可以自由执行。
- `confirm`（需人工审核才能执行）：全项目唯一真实节点是 HITL 的 `process`
  （见 [`_hitl_process`](./langchain.py#L1085)）——这不是摆设：图结构上 `process` 只能从
  `review` 节点的 `interrupt()` 批准之后到达，`_hitl_process` 里还加了一行运行时断言
  （`check_node_permission("process") == "confirm"`）防止未来权限表和图结构改动不同步。
- `forbidden`（直接拒绝）：本项目暂无这一档的真实节点——现有节点全部只读或已被 confirm 挡住，
  没有"一定要拒绝"的场景。机制已经就绪（[`check_node_permission`](./langchain.py#L902) 对未
  登记节点默认返回 `confirm` 而不是 `readonly`，是"失败关闭"的安全姿态），留给未来接入真正
  有副作用的节点（如"发送邮件""删除文件"）时直接用。

**怎么验证**：

```python
from service.ai.langchain import check_node_permission
assert check_node_permission("process") == "confirm"
assert check_node_permission("weather") == "readonly"
assert check_node_permission("some_未来才会有的危险节点") == "confirm"  # 默认失败关闭
```

---

## 7. 人工处理

**解决什么问题**：AI 生成的建议在真正产生影响之前，要能暂停下来，让人工审核、编辑或拒绝。

**现在怎么做的**（[langchain.py:1046-1135](./langchain.py#L1046)）：

`analyze`（AI 生成建议）→ `review`（[`interrupt()`](./langchain.py#L1064) 真正暂停，等待人工）
→ `process`（人工批准/编辑后执行）或直接 `END`（拒绝）。三条路径（批准/拒绝/编辑）都已用 curl
实测通过，详见 [`docs/langgraph-features-implemented.md`](../../docs/langgraph-features-implemented.md) 第 1 节。

**关键实现细节**：跨请求维持"暂停在哪"这件事依赖 checkpointer；`MemorySaver`（进程内存）在
`UVICORN_WORKERS>1` 部署下会直接把 HITL 做废——第一次请求和 resume 请求如果被负载均衡到不同
worker，第二个 worker 根本看不到那次暂停记录，图会从头重跑，人工审核形同虚设。改用共享文件
的 `SqliteSaver`（[`_get_hitl_checkpointer`](./langchain.py#L1100)）后，所有 worker 读写同一份
文件，问题消失——这也是【模块 1／8：上下文管理】的 `_CONTEXT_CHECKPOINTER` 照搬同一模式的
原因。

**怎么验证**：

```bash
curl -X POST http://localhost:3000/ai/langgraph/run \
  -d '{"graph":"hitl","threadId":"t1","query":"帮我给客户回一封确认周三开会的邮件"}'
# → {"waitingForInput": true, "interrupt": {"suggestion": "..."}}
curl -X POST http://localhost:3000/ai/langgraph/run \
  -d '{"graph":"hitl","threadId":"t1","resume":true}'
# → {"waitingForInput": false, "response": "✅ 已按审核通过的建议执行：..."}
```

---

## 8. 副作用回滚

**解决什么问题**：Agent 执行产生的副作用（改数据库、发邮件、扣款）如果中途某一步失败，已经
发生的副作用不能就这么留在半成品状态——得有机制把它撤销掉，而不是靠 Agent"自觉"。

**权限控制 vs 副作用回滚**：两者互补、管的不是同一时刻。第 6 节的权限控制挡在**执行前**（要不
要批准这个动作）；这一节管**执行后**（万一已经执行了、后面又失败了，怎么撤销已经产生的影响）。

**业界三种主流思路，本项目选的是第一种**：

1. **补偿事务（Saga Pattern）**——每个正向动作配一个补偿动作，失败后逆序执行补偿。✅ 本项目已实现。
2. **状态快照与恢复**——执行前整体打快照，失败就整体回滚到快照点。适合文件系统/文档类场景，本
   项目节点没有直接操作文件系统的场景，未实现。
3. **两阶段提交**——计划阶段不产生真实副作用，确认后才提交。本项目的权限控制 `confirm` 档
   （interrupt 暂停等审批）本质上已经是这个思路的一种落地，未额外重复实现。

**现在怎么做的**（[langchain.py:912-1035](./langchain.py#L912)）：

- [`SagaStep`](./langchain.py#L926)：一步的 `execute`（正向动作）+`undo`（补偿动作，接收
  `execute` 的返回值，据此精确定位要撤销什么，而不是重新猜状态）。
- [`run_saga`](./langchain.py#L935)：编排器，顺序执行 steps；任意一步抛异常，立刻停止并把已
  完成的步骤按**逆序**依次调用 `undo`。某一步补偿本身失败不会中断其余补偿（尽量撤销更多），
  但会记日志——这种情况代表"补偿机制兜不住了，需要人工介入核对"，如实记录而不是假装成功。
- **真实副作用演示场景**（[`_create_booking_step`](./langchain.py#L967) +
  [`_charge_payment_step`](./langchain.py#L1000)，端点 [`saga_demo_api`](./langchain.py#L1018)）：
  `create_booking` 真实写入一行 [`AgentBooking`](../../model/ai/agent_booking.py) 记录（MySQL）；
  `charge_payment` 可配置故意失败（模拟支付渠道拒绝）。这是全项目唯一一个真的有外部（数据库）
  副作用、可以拿来验证"回滚真的发生了"的场景——其余节点全部只读，给它们编 `undo` 接口没有
  意义，纯粹是摆设。

**怎么验证**（已实测，两条路径都验证过）：

```bash
# 场景1：支付失败 → 应触发回滚，booking 行被真实删除
curl -X POST http://localhost:3000/ai/langgraph/saga-demo \
  -d '{"item":"演示预订","amount":9900,"failPayment":true}'
# → {"status":"rolled_back","completed":["create_booking"],"compensated":["create_booking"],...}
# 查表：SELECT * FROM agent_booking WHERE item='演示预订';  → 空，行已被 undo 删除

# 场景2：支付成功 → 两步都提交，booking 行保留
curl -X POST http://localhost:3000/ai/langgraph/saga-demo \
  -d '{"item":"演示预订","amount":9900,"failPayment":false}'
# → {"status":"committed","completed":["create_booking","charge_payment"],"compensated":[],...}
# 查表：能查到这一行，status=pending
```

**诚实的范围边界**：只做了 Saga 这一种策略，且演示场景是特意设计的（预订+扣款），不是接入了
真实的支付/邮件服务。`charge_payment` 的 `undo` 目前是 no-op——因为这一步在 demo 里从未真正
提交成功过（`execute` 直接抛异常），没有可撤销的状态；真实接入支付渠道后，这里需要换成真的
调用退款 API（如 `stripe.refunds.create`）。补偿本身失败（比如退款接口也调不通）时目前只记日
志、不重试，生产环境这里通常需要死信队列/人工工单兜底，本项目未实现到这一层。
