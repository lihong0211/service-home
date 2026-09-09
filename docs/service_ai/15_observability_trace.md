# 可观测性 + 回流机制层

> 文件：`model/ai/agent_trace.py`（trace 数据模型）、`service/ai/langchain.py`（LangGraph 图的写入点 + 全部 HTTP 视图函数）、`service/ai/rag.py`（RAG 问答的写入点）
> 生成日期：2026-08-18

---

## 第一部分：背景演进

**为什么 LLM 应用需要专门的可观测性体系**

传统 Web 服务的 APM（Application Performance Monitoring，如日志埋点、Sentry、SkyWalking）关注的是"请求有没有 5xx、接口耗时多少、哪个函数抛了异常"——这套体系建立在"输入输出结构固定、正确性可以用状态码/异常类型判断"的假设上。LLM 应用打破了这个假设：

- **没有异常也可能是错的**：一次 RAG 问答可以正常返回 200，答案却是"看似合理但答非所问"的幻觉，传统 APM 的异常监控完全捕捉不到这类问题。
- **"对不对"依赖人工/语义判断**，不是布尔值。需要把每次交互的输入、检索片段、生成结果都留痕，供事后人工复核或用另一个 LLM 当裁判来评判。
- **多步骤执行**：Agent/LangGraph 这类系统一次请求内部会经过多个节点（分类、检索、生成、工具调用……），排查"哪一步慢/哪一步错"需要保留每一步的中间状态，而不只是首尾的输入输出。

**核心概念：trace/span 与回流机制**

- **trace**：一次完整交互（一次 RAG 问答，或一次 LangGraph 图的 invoke/stream）的记录，是本模块的最小追踪单元。行业标准方案（如 OpenTelemetry）里 trace 由多个 **span**（每个节点/每次 LLM 调用是一个 span）组成树状结构；本模块做了简化，一个 trace 只用一个 JSON 字段（`steps_detail`）平铺记录各步骤的耗时，没有实现真正的树状 span 嵌套。
- **回流机制**：指把"这次交互是好是坏"的反馈信号采集下来，反哺后续迭代——具体到本模块，是把 `status='error'`（系统判定失败）或 `feedback='bad'`（人工/用户标注差评）的记录汇总成 **bad case 候选池**，供人工复核后决定"回流去哪"：可能是补充知识库文档、修正 prompt 约束，或积累成微调数据集。本模块只负责采集信号和挖掘候选池这两步，不负责回流目的地本身的落地（那是后续人工或离线批处理的工作）。

**演进脉络：从无观测到全链路 trace**

行业里 LLM 可观测性经历了从"业务日志里顺手打几行 print/logger"到专门工具链的演进：LangSmith（LangChain 官方，与 LangGraph 深度绑定）、Langfuse（开源、自托管）、OpenTelemetry GenAI 语义约定（厂商中立标准）等工具相继出现，共同点是把 trace/span、token 用量、评测分数、用户反馈整合进统一的可视化平台，支持按 trace 下钻查看每个节点的 prompt/completion。

本项目走的是相反方向：不引入外部平台，而是在现有 MySQL 里加一张 `agent_trace` 表，用几个纯函数完成"写入-查询-反馈"闭环。这是"重"（接入 LangSmith/Langfuse，功能完整但引入新依赖、新服务）与"轻"（自建，功能覆盖当前实际需要即可）之间的权衡，第二部分的对比表格会展开讲。

**本模块定位**

`model/ai/agent_trace.py` + `service/ai/langchain.py` 里以 `_persist_trace`/`submit_trace_feedback`/`list_traces`/`list_bad_cases` 为核心的一组函数，是本项目"可观测性"与"回流机制"两个需求共用的同一套基础设施：LangGraph 演示图（router/loop/parallel/hitl）最先接入，RAG 问答（`service/ai/rag.py` 的 `_persist_rag_trace`）在 `7482df9` 补上——这是目前唯一真正在线上被使用的问答链路，之前完全没有可观测性覆盖。`docs/service_ai/14_rag_eval_ragas.md` 描述的 RAGAS 批量评测目前与本模块**尚未打通**（评测结果不落库、trace 数据也未反哺评测），这是第五部分要展开的一条演进建议。

---

## 第二部分：架构剖析

**整体分层**

```
model/ai/agent_trace.py          ← 数据模型：一张表 AgentTrace，同时承载可观测性字段
                                    （status/duration_ms/steps_detail）和回流字段
                                    （feedback/feedback_note/copied）

service/ai/langchain.py          ← LangGraph 图的写入点 + 全部 5 个 HTTP 视图函数
  ├── _persist_trace()             写入点（非流式/流式 done 分支 / 异常分支复用同一个函数）
  ├── submit_trace_feedback()      回流采集：显式反馈 good/bad
  ├── mark_trace_copied()          回流采集：隐式反馈（答案被复制）
  ├── list_traces()                可观测性查询：全量列表
  ├── list_bad_cases()             回流挖掘：bad case 候选池
  ├── trace_feedback_api / trace_copy_api / trace_list_api / trace_bad_cases_api
  │                                 上述函数对应的 HTTP 视图

service/ai/rag.py                ← RAG 问答的写入点（独立实现，同一张表，graph_name="rag"）
  └── _persist_rag_trace()

routes/ai.py                     ← 路由注册（`_ai_route` 包装，见 CLAUDE.md 路由注册规范）
  ├── POST /ai/langgraph/trace/feedback   → trace_feedback_api
  ├── POST /ai/langgraph/trace/copy       → trace_copy_api
  ├── GET  /ai/langgraph/trace/bad-cases  → trace_bad_cases_api
  └── GET  /ai/langgraph/trace/list       → trace_list_api
```

写入点没有做成公共装饰器/中间件，而是 `langchain.py` 和 `rag.py` 各自维护一份逻辑相近但独立的 `_persist_*` 函数——这是"表结构统一、写入逻辑各自贴合调用方数据形状"的取舍，第三部分的设计取舍会展开。

**核心数据流：一次请求如何被记录成 trace，直到回流**

以 LangGraph 非流式调用为例（`langgraph_run_api` → `run_graph_and_collect_steps`，`service/ai/langchain.py:1766`）：

```
1. 前端 POST /ai/langgraph/run { graph: "router", input: {...}, threadId }
2. run_graph_and_collect_steps() 记录 t_start，执行图（run_graph_stream_and_collect）
3a. 执行成功 → _persist_trace(graph_name, trace_state, run_result, duration_ms)
      写入 AgentTrace 行：status="success"，total_steps/steps_detail 来自 run_result["steps"]
      返回新行 id 作为 trace_id
3b. 执行抛异常 → _persist_trace(..., run_result=None, error=str(e))
      写入 status="error"，error_message=异常信息，trace_id 同样返回
4. 响应体里带上 traceId（非流式在 finalState 旁；流式在 SSE type=done 事件里）
5. 前端展示回答，用户可选择：
   a) 点赞/点踩 → POST /ai/langgraph/trace/feedback { traceId, rating, note }
      → submit_trace_feedback() 更新 feedback/feedback_note 字段（显式反馈）
   b) 复制回答 → POST /ai/langgraph/trace/copy { traceId }
      → mark_trace_copied() 更新 copied=True（隐式反馈，见下方"回流机制"细节）
6. 【可观测性】随时可查 GET /ai/langgraph/trace/list?graph=router&status=error
      → list_traces() 返回全量记录（不限 status/feedback），排查慢请求/看整体健康度
7. 【回流机制】GET /ai/langgraph/trace/bad-cases?graph=router
      → list_bad_cases() 只挑 status='error' 或 feedback='bad' 的行，作为候选池
      → 人工从候选池里复核，决定回流去处（补知识库 / 改 prompt / 攒训练数据）——
        这最后一步本模块不实现，是候选池产出之后的人工/离线流程
```

RAG 问答（`rag_ask_api`，`service/ai/rag.py:301`）走的是并行独立的一条路径：同样在成功/异常两个出口调用 `_persist_rag_trace`，`graph_name="rag"` 与 LangGraph 演示图的 `router`/`loop`/`parallel`/`hitl` 共用同一张表、按 `graph_name` 筛选区分，但 RAG 侧目前**没有** `total_steps`/`steps_detail`（RAG 是"检索+生成"两步固定流程，没有图执行的多节点步骤概念），也因此 RAG 的反馈/复制回流复用的是 LangGraph 那套已有的 `/ai/langgraph/trace/*` 端点，而不是单独开一套 `/ai/rag/trace/*`。

"回流机制"具体所指：本模块里的回流不是自动化的模型再训练管道，而是**信号采集 + 候选池挖掘**这两步——`feedback`（显式，用户/人工主动打分）和 `copied`（隐式，答案被复制这个自然动作，作者认为比主动点踩更真实，因为大部分不满意的用户是沉默流失、不会主动点踩）共同构成"这条回答质量如何"的信号源；`status='error'` 与 `feedback='bad'` 的并集构成 bad case 候选池，`list_bad_cases()` 把这个池子暴露出来，供人工在下游做知识库补全、prompt 调整或训练数据积累，这三个真正的"回流目的地"都在本模块职责之外。

**关键设计原则**

1. **一张表承载两个职责**：可观测性字段（`status`/`duration_ms`/`steps_detail`/`total_steps`）和回流字段（`feedback`/`feedback_note`/`copied`）放在同一张 `agent_trace` 表里，而不是拆成 trace 表 + feedback 表两张表——因为二者共享同一个主键生命周期（一条交互记录），拆表需要额外一次 JOIN 才能拿到完整视图，收益不明显。
2. **独立 Session，不复用请求级 `db.session`**：写入点统一用 `SessionLocal()` 开自己的会话，而不是走 `app/database.py` 的请求作用域 `db.session`。原因见下方"关键实现细节"。
3. **失败不影响主流程**：所有写入函数（`_persist_trace`/`_persist_rag_trace`）在 `except` 分支只 `logger.exception` + `session.rollback()`，不重新抛出——trace 落库失败不应该导致用户拿不到问答结果。

**与行业标准方案对比**

| 维度 | 本项目自建（agent_trace 表） | LangSmith | Langfuse | OpenTelemetry (GenAI 语义约定) |
|---|---|---|---|---|
| 部署成本 | 零额外部署，复用现有 MySQL | SaaS，需要账号+网络出站 | 可自托管（Docker）或 SaaS | 需要自建/托管 Collector + 后端存储（Jaeger/Tempo 等） |
| span 粒度 | 无真正 span 树，`steps_detail` 是平铺 JSON 数组 | 完整 trace/span 树，逐 LLM 调用级别，含 token 用量、prompt 快照 | 同 LangSmith，另支持 session/user 维度分组 | 标准化 span 模型，跨语言/跨厂商互通 |
| 查询/可视化 | 手写 SQL 或调用现成的 4 个 REST 接口，无内置 UI | 内置 Web UI，支持按 trace 下钻、对比 | 内置 Web UI + 开源可自建看板 | 需自行接入 Grafana/Jaeger UI |
| 反馈/回流集成 | 内建 `feedback`/`copied` 字段 + `list_bad_cases()`，与业务表同库 | 内建 Feedback API，可关联标注队列 | 内建 Score/Annotation Queue | 无内建反馈概念，需自行扩展 |
| 迁移/绑定成本 | 无厂商绑定，随存量业务表迁移 | 绑定 LangChain 生态与 SaaS 账号 | 弱绑定，开源自托管可控 | 标准协议，理论上厂商无关 |

**选型建议**：当前项目体量下（单体 FastAPI 服务、trace 写入量不大、主要消费者是开发者自己排查问题），自建轻量表是合理选择——不需要为了几个查询接口引入新的部署单元或 SaaS 依赖，且能直接复用现有的 MySQL 运维能力。触发切换的信号是：① 需要跨多个微服务串联同一条链路（分布式 trace，本模块目前无 trace_id 跨服务传递能力）；② 需要逐 LLM 调用的 token/成本级别追踪；③ 团队规模扩大到需要非工程角色也能用可视化界面复核 bad case，而不是要求会写 SQL 或调接口——这种情况下 Langfuse（自托管、开源、有 UI）是比 LangSmith 更贴近本项目风格的下一步选项。

---

## 第三部分：代码实现深度解析

**核心函数/类清单**

| 名称 | 位置 | 职责 |
|---|---|---|
| `AgentTrace` | `model/ai/agent_trace.py:6` | ORM 模型，一张表承载可观测性字段（status/duration_ms/steps_detail）和回流字段（feedback/feedback_note/copied） |
| `_persist_trace()` | `service/ai/langchain.py:1526` | LangGraph 图执行的写入点，非流式/流式/异常三个出口共用 |
| `_persist_rag_trace()` | `service/ai/rag.py:267` | RAG 问答的写入点，独立实现，`graph_name="rag"` |
| `submit_trace_feedback()` | `service/ai/langchain.py:1576` | 回流采集：写入显式反馈（good/bad + note），拒绝非法 rating 值 |
| `mark_trace_copied()` | `service/ai/langchain.py:1605` | 回流采集：写入隐式反馈（copied=True） |
| `list_traces()` | `service/ai/langchain.py:1630` | 可观测性查询：全量列表，按 graph_name/status 过滤 |
| `list_bad_cases()` | `service/ai/langchain.py:1671` | 回流挖掘：`status='error'` 或 `feedback='bad'` 的候选池 |
| `trace_feedback_api` / `trace_copy_api` / `trace_list_api` / `trace_bad_cases_api` | `service/ai/langchain.py:1715~1763` | 对应上述函数的 HTTP 视图，供 `routes/ai.py` 注册 |

**关键实现细节**

1. **独立 `SessionLocal()` 而非请求级 `db.session`**：`_persist_trace` 的注释明确解释了原因——SSE 流式分支的落库发生在 `StreamingResponse` 对象**返回之后**（生成器 `gen()` 内部延迟执行），此时 `_ai_route` 已经调用过 `clear_request_session()`，如果这时访问 `db.session`（依赖请求作用域 ContextVar）会抛 `RuntimeError`。为了让同一个 `_persist_trace` 函数在流式/非流式两种场景下行为一致，统一改用不依赖请求生命周期的 `SessionLocal()` 开独立会话，并在 `finally` 里显式 `session.close()`。
2. **`input_summary`/`output_summary` 截断上限设为 15000 字符**：`AgentTrace.input_summary`/`output_summary` 是 `Text` 列，MySQL `TEXT` 类型上限是 65535 字节，按 utf8mb4 最坏情况每字符 4 字节换算，安全上限约 16383 字符。两处写入点（`_persist_trace` 和 `_persist_rag_trace`）都统一截断到 15000，而不是更保守的旧值（代码注释里提到之前是 2000），是为了避免长回答被过度截断丢失信息，同时留出安全余量。
3. **`submit_trace_feedback` 对非法 `rating` 显式抛异常，不做静默纠正**：`rating not in ("good", "bad")` 时直接 `raise ValueError`，而不是把非法值悄悄改成某个默认值或忽略。调用方（`trace_feedback_api`）捕获后转换成 400 响应。这是"调用方传错参数应该显式失败"的设计取舍，避免脏数据静默进表。
4. **流式分支里 `trace_id` 只在 `type=done` 事件里补发**：`langgraph_run_api` 的 SSE 生成器（`service/ai/langchain.py:1926` 起）在 `step`/`token` 事件时不知道最终结果，只有等到图执行完毕、拿到 `run_result` 后才调用 `_persist_trace` 拿到 `trace_id`，随 `done` 事件一起下发给前端。这意味着前端只有在收到 `done` 之后才能对这次回答调用反馈接口，中间过程无法预先提交反馈。

**设计决策与取舍**

1. **为什么自建轻量表，而不用 LangSmith/Langfuse 等现成可观测性平台**：一是部署成本——项目目前是单体 FastAPI + MySQL 的架构风格（见 CLAUDE.md 的"轻量、自建路由层"定位），接入外部 SaaS（LangSmith）意味着引入新的网络出站依赖和账号体系，自托管 Langfuse 则要多起一个服务；二是当前需求边界明确——只需要"记下每次交互 + 支持打反馈 + 能查出 bad case"，这几个需求用一张表 + 几个纯函数就能覆盖，没有必要为潜在的高级功能（token 级别追踪、可视化 UI）预先付出复杂度。这是"最小可行可观测性"而非"功能完备的可观测性平台"的取舍，代价在第五部分"局限"里展开。
2. **`_persist_trace`（LangGraph）与 `_persist_rag_trace`（RAG）是两份独立实现，而不是抽出公共写入函数**：两者数据形状不同——`_persist_trace` 需要处理 `steps`/`finalState` 这类图执行特有的结构，`_persist_rag_trace` 只有 question/answer 这种更简单的输入输出。共用点仅是"截断到 15000、失败不抛出、返回 id"这几行逻辑，抽公共函数收益有限，保持两处独立实现代码定位更直接（各自模块内一眼可见完整逻辑），符合"不为单次复用做抽象"的取舍。
3. **`list_traces`/`list_bad_cases` 直接用 `session.query()` 拼 SQL 过滤条件，不做分页游标**：当前只支持 `limit`（默认 50）+ `order_by(id.desc())`，没有 offset/cursor 分页。这是因为当前使用场景是"看最近一段时间的记录"而非"翻页浏览全部历史"，随着数据量增长这会成为技术债务（见第五部分）。
4. **隐式反馈（`copied`）与显式反馈（`feedback`）分成两个独立字段和两个独立接口，而不是合并成一种反馈类型**：因为二者语义不冲突且都可能同时存在——一条回答完全可能既被复制过（`copied=True`）又被打了差评（`feedback='bad'`，比如"复制走去别处验证发现是错的"），保留两个独立信号比强行合并成一个枚举更能保留原始信息，`list_bad_cases` 的过滤条件也因此只用 `status`/`feedback`，不涉及 `copied`（复制本身不代表内容有问题，只是"被使用过"的信号，不构成 bad case 判定依据）。

---

## 第四部分：应用场景与实战

**核心使用场景**

1. **排查慢请求/失败请求**：调用 `GET /ai/langgraph/trace/list?graph=router&status=error` 或不带 `status` 看全量记录，配合 `duration_ms`/`steps_detail` 定位是哪个节点慢、哪次调用报错。
2. **支撑 bad case 人工复核**：`GET /ai/langgraph/trace/bad-cases?graph=rag` 拉取候选池，作为 `docs/service_ai/14_rag_eval_ragas.md` 描述的测试集人工审核/修正的一个潜在数据来源——不过目前两个模块尚未打通，候选池目前只能人工另行摘录进测试集，不是自动化流程（见第五部分）。
3. **RAG 问答的可观测性回填**：`_persist_rag_trace` 是 `7482df9` commit 补上的能力，让此前完全没有可观测性覆盖的线上主链路（RAG 问答）也能查到执行记录、打反馈。

**快速上手**

查询最近的 trace 列表（可选 `graph`/`status`/`limit`）：

```bash
curl "http://localhost:3000/ai/langgraph/trace/list?graph=router&limit=20"
# 返回: { "code": 0, "msg": "ok", "data": { "traces": [...], "total": 20 } }
```

对一次 RAG 问答结果打反馈（`traceId` 来自 `rag_ask_api` 返回体里的 `data.traceId`）：

```bash
curl -X POST http://localhost:3000/ai/langgraph/trace/feedback \
  -H "Content-Type: application/json" \
  -d '{"traceId": 1, "rating": "bad", "note": "检索到的片段跟问题不相关"}'
# 返回: { "code": 0, "msg": "ok", "data": { "traceId": 1, "rating": "bad" } }
```

拉取 bad case 候选池供人工复核：

```bash
curl "http://localhost:3000/ai/langgraph/trace/bad-cases?graph=rag&limit=50"
# 返回: { "code": 0, "msg": "ok", "data": { "cases": [...], "total": N } }
```

**常见问题排查**

- **`traceId` 返回 `null`**：说明该次 `_persist_trace`/`_persist_rag_trace` 落库失败（`logger.exception` 会记录具体原因，去日志找"trace 落库失败"字样），常见原因是数据库连接异常，但主流程（问答本身）不受影响，只是这次交互没能落 trace。
- **`trace/feedback` 返回 404**：说明传入的 `traceId` 在 `agent_trace` 表里不存在，检查是否传错了 id 或者对应记录本身落库失败（见上一条）。
- **`trace/feedback` 返回 400 且 msg 是 `rating 只能是 good/bad`**：`rating` 字段拼写错误或传了非 `good`/`bad` 的值，这是 `submit_trace_feedback` 的显式校验，不会静默接受。
- **想查某个具体请求但不知道 `traceId`**：先按时间倒序查 `trace/list`（默认按 `id desc` 排），结合 `input_summary`/`created_at` 定位到具体记录再拿 `id`。

---

## 第五部分：评估与展望

**优势**

- 零额外部署成本，复用现有 MySQL 与 SQLAlchemy 技术栈，不引入新的运维单元。
- 可观测性与回流机制共用同一张表、同一套写入路径，两个需求（"排查问题"和"挖掘 bad case"）不需要维护两套独立基础设施。
- 显式反馈（`feedback`）与隐式反馈（`copied`）并存，隐式信号弥补了"大部分不满意用户不会主动点踩"这个显式反馈的天然盲区。
- 失败隔离到位：trace 落库失败不影响主流程返回结果，符合"可观测性本身不应该成为新的故障点"的原则。

**局限与技术债务**

- **无分页游标**：`list_traces`/`list_bad_cases` 只支持 `limit`，数据量增长后（表持续追加、无归档/清理机制）查询会退化为"总是看最新 N 条"，无法翻页查历史区间，也没有按时间范围过滤的参数。
- **无真正的 span 树**：`steps_detail` 是平铺 JSON 数组，无法表达节点间的父子/并行关系，多分支图（如 `parallel`）的执行结构在数据层面被拍平，只能靠前端按顺序猜测层级。
- **存储压力未做预案**：`input_summary`/`output_summary` 截断到 15000 字符仍然是较大的 `TEXT` 字段，随着调用量增长，表体积会线性上升，目前没有归档、冷热分离或 TTL 清理策略。
- **无分布式 trace 能力**：`thread_id` 只在单次图执行内串联，跨服务（比如本项目 `service/ai/a2a/` 的多进程 A2A 子 agent）调用链目前不会被同一个 trace 记录关联起来。
- **与 RAGAS 评测（`14_rag_eval_ragas.md`）尚未打通**：批量回归评测的结果不落库，也不会跟 trace 里的真实线上反馈关联，两套数据目前是孤立的，14 号文档的"行业前沿"部分也提到了这一点。

**演进建议**

- 优先解决存储压力：给 `agent_trace` 加基于时间的归档策略（比如按月分表或转存冷库），并在查询接口补上时间范围参数。
- 补一层轻量分页（游标或 offset），支撑未来的可视化列表页翻页浏览。
- 探索把 RAGAS 批量评测结果与 trace 关联：比如评测时把结果写回对应的 trace 行，或者反过来用 `list_bad_cases` 的候选池自动生成/校准 RAGAS 测试集的 `ground_truth`，形成"线上反馈 → 测试集 → 回归评测 → 优化 → 再验证"的闭环，而不是两条独立数据链路。
- 若团队规模扩大到需要非工程角色参与 bad case 复核，评估自托管 Langfuse（开源、有 UI、风格上比 LangSmith 更贴近本项目"自建可控"的取向）替代当前的纯接口查询方式。

**行业前沿**

LLM 可观测性领域仍在快速演进：OpenTelemetry 的 GenAI 语义约定（semantic conventions for generative AI systems）正在试图把 trace/span 的字段标准化，减少厂商锁定；LangSmith/Langfuse 等工具持续加强"评测结果反哺 trace"的闭环能力（对应本项目当前缺失的"评测与 trace 打通"这块）；此外，随着多 Agent 协作（本项目的 A2A 模块）成为更常见的架构模式，跨进程/跨服务的分布式 trace 传播（而不只是单进程内的 thread_id）会是接下来行业和本项目都需要面对的问题。

---

## 变更记录

- 2026-08-18 新建文档，覆盖近期新增的可观测性/trace 能力。
