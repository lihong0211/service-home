# LangGraph 多 Agent 工作流层

> 覆盖代码：`service/ai/showcase/agents/agent.py`、`agent_doctor.py`、`agent_fund_qa.py`、`agent_research.py`、`agent_wealth_advisor.py`（辅以 `service/ai/langchain.py` 的图结构提取工具）
> 全量重写日期：2026-08-18

---

## 第一部分：背景演进

**从单轮调用到图编排**

单次 LLM 调用（prompt → completion）只能解决"一问一答"型任务。稍复杂一点的场景——"先检索资料、再分析、再决策、再写报告"——需要把多次 LLM 调用和确定性代码逻辑串起来，这就是 Agent 编排要解决的问题。行业里大致经历了三个阶段：

1. **Prompt Chaining**：把上一步输出手工拼接进下一步的 prompt，代码里是一串顺序函数调用，没有统一的状态容器，分支和重试都要手写 if/else。
2. **ReAct / Function Calling / AgentExecutor**：LLM 在推理（Reason）和行动（Act，即工具调用）之间交替，框架（如 LangChain 的 `AgentExecutor`）封装了这个循环，但循环内部是隐式的黑盒——调试时很难知道"现在到第几步、状态是什么"。
3. **LangGraph StateGraph（本模块采用）**：把执行过程显式建模为**有向图**：节点（Node）是处理步骤，边（Edge）是流转关系（含条件路由），一个 `TypedDict` 定义的 State 在图中流动，每个节点读一部分状态、写一部分状态。图结构本身可以被程序内省（`compiled_graph.get_graph()`），可以逐节点 `stream()` 观察执行过程，也可以接 `checkpointer` 做跨请求的状态持久化。

**核心概念（3 个）**

- **State（状态）**：`TypedDict` 定义的字典，是图中唯一的数据载体。节点函数的输入输出都是它的子集或全量。本模块四个 Agent 各自定义了互不共享的 State（`ResearchAgentState`、`WealthAdvisorState`、`DoctorState`，`fund_qa_agent` 甚至没有 TypedDict，只用普通 dict）。
- **Node / Conditional Edge**：Node 是一个 `(state) -> dict` 函数，代表一次 LLM 调用或一段确定性逻辑；Conditional Edge 是一个 `(state) -> str` 路由函数，根据状态决定下一个节点 id，是分支逻辑的唯一载体（本模块没有用到工具调用型的 Tool Node）。
- **Checkpointer（状态持久化）**：LangGraph 提供的可插拔存储，以 `thread_id` 为键保存/恢复某次执行的完整 State。本模块只有 `agent_doctor.py` 使用了它（`MemorySaver`），用来实现"同一 `session_id` 多轮问诊、状态跨请求累积"；其余三个 Agent 每次请求都是一次性执行，不保留历史。

**本模块的定位**

`service/ai/showcase/agents/` 目录下有两条**互不相通**的产品线：

- `agent.py` 是一个**通用注册表 + 执行/可视化服务**，统一管理 `research_agent`（深思熟虑型）、`fund_qa_agent`（反应式）、`wealth_advisor_agent`（混合型）三个 Agent，对外暴露 `/ai/agent/list`、`/ai/agent/schema`、`/ai/agent/run` 三个通用接口，服务于前端的 3D 执行动画可视化。
- `agent_doctor.py` 是一个**独立的多轮问诊 Agent**，走自己的一套接口（`/ai/doctor/chat`、`/ai/doctor/session/{session_id}`），不注册进 `agent.py` 的 `AGENT_BUILDERS`，也不参与 `list_agents()`/`get_agent_schema()`，是本目录里唯一真正用到跨请求状态持久化的 Agent。

---

## 第二部分：架构剖析

### 分层关系：没有共享基类

`agent.py` **不是**一个抽象基类或框架层，四个 Agent 之间也不共享图结构或 State 定义。`agent.py` 只提供三样东西：

1. `AGENT_BUILDERS: dict[str, Callable]`——agent_id 到"构建函数"的映射（`create_research_agent_workflow`、`create_fund_qa_agent`、`create_wealth_advisor_workflow`）；
2. `AGENT_META` / `DEFAULT_INPUTS`——展示元信息和演示用默认输入；
3. 一组**鸭子类型**的通用执行/可视化函数（`get_agent_schema`、`run_agent_and_collect_steps`、`run_agent_stream_yield_events`），靠 `hasattr(agent, "get_graph")` / `hasattr(agent, "stream")` 判断一个 Agent 实例是真正的 LangGraph `CompiledStateGraph` 还是自定义类，而不是靠继承关系。

这意味着 `fund_qa_agent`（`agent_fund_qa.py` 里的 `_DisneyRagAgent` 类）完全不是 LangGraph 对象，只是手写了 `.stream()` / `.invoke()` 两个方法来"伪装"成一个 Agent，以便复用 `agent.py` 里的通用执行逻辑。`agent_doctor.py` 则连伪装都没做——它是真正的 `StateGraph`，但因为不在 `AGENT_BUILDERS` 里，完全绕开了 `agent.py` 的执行/可视化通用代码，自己实现了一套 `chat()` / `get_session_info()`。

### 核心数据流：一次执行的完整路径

以非流式 `POST /ai/agent/run`（`agent_id=research_agent`）为例：

```
路由层：routes/ai.py → _ai_route(router, "/ai/agent/run", agent_run_api, ["POST"])
  → agent_run_api(request)
      body = anyio.from_thread.run(read_json_optional, request)
      agent_id, input_data, stream = body["agent_id"], body["input"], body["stream"]
      stream=False → run_agent_and_collect_steps(agent_id, input_data)
          builder_fn = AGENT_BUILDERS[agent_id]          # 每次请求重新构建
          agent = builder_fn()                            # create_research_agent_workflow() → workflow.compile()
          if hasattr(agent, "stream"):                     # StateGraph 分支
              for step in agent.stream(input_data, config):
                  # step = {"perception": {...partial state...}}
                  for node_id, output in step.items():
                      steps.append({stepIndex, nodeId, duration_ms,
                                     output: _ensure_json_serializable(output)})
              final_state = agent.invoke(input_data, config=config)   # ← 第二次完整执行，见决策1
          graph_data = get_agent_schema(agent_id)          # graph_to_schema(agent, RESEARCH_NODE_DISPLAY)
          return {agentMeta, graphData, steps, finalState, executionOrder, totalSteps}
  → normalize_api_result(...) → JSONResponse({"code":0,"msg":"ok","data": ...})
```

`stream=True` 时路径相同，只是 `agent_run_api` 用 `StreamingResponse` 包一层 SSE：先发 `type=init`（`graphData` + `agentMeta`），再逐条转发 `run_agent_stream_yield_events` 产出的 `("step", payload)` 事件（`type=step`），最后发一条 `type=done`（含 `finalState`/`steps`/`totalSteps`），以 `data: [DONE]\n\n` 结束；异常时发 `type=error` 并终止。

医生智能体走的是完全独立的一条路径（`POST /ai/doctor/chat` → `doctor_chat_api` → `chat(session_id, message)` → `get_doctor_graph()` 单例 → `graph.invoke({"messages":[HumanMessage(...)]}, config={"configurable":{"thread_id": session_id}})`），不经过 `agent.py` 的任何一行代码。

### 关键设计原则

- **注册表模式**：新增一个 StateGraph/伪 Agent 只需要在 `AGENT_BUILDERS`/`AGENT_META`/`DEFAULT_INPUTS` 三个字典里各加一行，`list_agents`/`get_agent_schema`/`run_agent_and_collect_steps` 都不用改。
- **鸭子类型而非接口约束**：用 `hasattr` 判断能力而不是要求继承某个基类或实现某个 Protocol，换来的是可以把非 LangGraph 对象（`_DisneyRagAgent`）也接入同一套执行/可视化代码，代价是契约是隐式的（详见第三部分决策 2）。
- **图结构自省，不手工维护**：`graph_to_schema()`（`service/ai/langchain.py`）直接读 `compiled_graph.get_graph()` 的 `nodes`/`edges`，节点展示名/图标/类型通过 `node_display` 字典（如 `RESEARCH_NODE_DISPLAY`、`WEALTH_NODE_DISPLAY`）覆盖，图拓扑变化时前端可视化自动跟随，不需要手写 nodes/edges JSON——但这条能力只对真 `StateGraph`（research/wealth_advisor）生效，`fund_qa_agent` 在 `get_agent_schema()` 里是硬编码的四节点结构。

### 与行业标准方案对比

| 维度 | 本项目（LangGraph StateGraph，手写节点） | LangChain `AgentExecutor` | AutoGen | CrewAI |
|------|---------------------------------------|---------------------------|---------|--------|
| 编排方式 | 显式有向图，节点/边/条件路由代码可读 | 隐式 ReAct 循环，封装在框架内部 | 多 Agent 对话驱动，靠消息在 Agent 间流转 | 角色（Role）+ 任务（Task）驱动，框架内部调度 |
| 状态管理 | `TypedDict` State + 节点局部读写；仅 `agent_doctor` 接了 `MemorySaver` 做跨请求持久化 | 依赖 `AgentExecutor` 内部的 `intermediate_steps`，手工管理 | 对话历史即状态，无结构化 State | 任务上下文对象，结构较松散 |
| 可观测性 | `graph.stream(..., stream_mode="updates")` 原生逐节点产出，天然适合前端做步骤动画 | 需要注册回调（`Callback Handler`）才能拿到中间步骤 | 需解析 Agent 间的对话日志 | 有限，主要靠日志 |
| 多 Agent 协作 | 本模块未使用（每个图内部都是单一决策者），理论上可用子图节点组合 | 弱，一个 Executor 通常对应一个 Agent | 原生支持，多 Agent 自由对话是核心卖点 | 原生支持，角色分工是核心卖点 |
| 生产级特性（HITL、超时、并行节点） | 本模块未启用（无 `interrupt`、无超时保护），LangGraph 本身支持但代码里没用到 | 弱 | 部分支持 | 部分支持 |
| 学习曲线 | 中，需理解图/状态/reducer 概念 | 低，开箱即用 | 低，但多 Agent 对话的可控性差 | 低，但角色抽象在简单任务上是过度设计 |

**选型建议**：当任务是**单一决策者、多步骤、需要给前端展示"正在执行到第几步"**（如本模块的投研报告生成、财富咨询、多轮问诊）时，LangGraph StateGraph 的显式图 + 原生 `stream()` 是合适的选择；如果任务本质是"多个专门化角色互相讨论/委派"（如多 Agent 代码评审、多角色头脑风暴），AutoGen/CrewAI 的对话驱动模型更省代码。不建议在本模块的场景下引入多 Agent 框架——四个 Agent 目前都是单图单决策者，没有 Agent 间协作的需求。

---

## 第三部分：代码实现深度解析

### 核心函数/类清单

| 符号 | 文件 | 作用 |
|------|------|------|
| `AGENT_BUILDERS` / `AGENT_META` / `DEFAULT_INPUTS` | agent.py | 三张登记表：agent_id → 构建函数 / 展示元信息 / 演示默认输入 |
| `get_agent_schema(agent_id)` | agent.py | 按 `hasattr(agent,"get_graph")` 分支：真 StateGraph 走 `graph_to_schema`，`fund_qa_agent` 走硬编码四节点结构，其余走通用三节点结构 |
| `run_agent_and_collect_steps(agent_id, input_data=None)` / `run_agent_stream_yield_events(...)` | agent.py | 同步/生成器两个版本的统一执行入口，`stream()` 收集步骤 + `invoke()` 取终态（双重执行，见决策1） |
| `graph_to_schema(compiled_graph, node_display=None, node_icons=None)` | langchain.py | 读 `compiled_graph.get_graph()` 动态生成 `{nodes, edges}`，供三个 StateGraph Agent 复用 |
| `create_research_agent_workflow()` 及 `_research_perception/_modeling/_reasoning/_decision/_report` | agent_research.py | 5 节点线性 StateGraph：感知→建模→推理→决策→报告 |
| `create_wealth_advisor_workflow()` 及 `_assess_query/_reactive_processing/_collect_data/_analyze_data/_generate_recommendations/_respond` | agent_wealth_advisor.py | 6 节点 StateGraph，`assess` 后按 `processing_mode` 条件路由到反应式或深思熟虑式分支 |
| `_DisneyRagAgent`（`.stream()` / `.invoke()`） | agent_fund_qa.py | 非 StateGraph 的两步 RAG："伪装"出 `stream`/`invoke` 接口以复用 `agent.py` 的通用执行代码 |
| `get_doctor_graph()` / `_build_graph()` / `_extract_info/_check_completeness/_ask_questions/_generate_assessment` / `chat(session_id, message)` / `get_session_info(session_id)` | agent_doctor.py | 多轮问诊 StateGraph + `MemorySaver`，独立于 `agent.py` 的整套执行入口 |

### 关键实现细节

**1. 四种 Agent 范式的实际代码差异**

- **反应式（`fund_qa_agent`）**：`_DisneyRagAgent` 完全没有用 `StateGraph`，`.stream()` 是一个手写生成器，固定 yield 两次——`{"retrieval": {...}}` 再 `{"generation": {...}}`——`.invoke()` 优先读 `self._cached`（`.stream()` 跑完后缓存的最终结果），只有在没经过 `.stream()` 直接调 `.invoke()` 时才会走 `service.ai.rag.rag_chat()` 兜底，这是为了避免"检索+生成"被重复调用两次 LLM/embedding API（详见决策1）。
- **深思熟虑型（`research_agent`）**：`create_research_agent_workflow()` 是纯线性图，`perception → modeling → reasoning → decision → report`，五条边全部是 `add_edge`（无条件路由）。每个节点函数在依赖的上游字段缺失时（如 `_research_modeling` 检查 `state.get("perception_data")`），不会中断图执行，而是把 `error` 写进 state、`current_phase` 回退，图仍会往下走到 `report` 节点——即**没有错误短路机制**，最终报告可能是基于部分缺失数据生成的。
- **混合型（`wealth_advisor_agent`）**：`create_wealth_advisor_workflow()` 在 `assess` 节点后用 `add_conditional_edges` 做唯一一次分支：`lambda x: "reactive" if x.get("processing_mode") == "reactive" else "collect_data"`。`_assess_query` 节点通过 LLM 输出 JSON 判定 `query_type`/`processing_mode`，两个字段做了枚举兜底（不在合法值内则分别兜底为 `"reactive"`/`"emergency"`）。`reactive` 分支只有一步（`_reactive_processing`）直达 `respond`；`deliberative` 分支要走 `collect_data → analyze → recommend` 三步才到 `respond`。
- **多轮问诊型（`doctor_agent`）**：见下一小节。

**2. `DoctorState` 的多轮状态管理**

```python
class DoctorState(TypedDict):
    messages: Annotated[list, add_messages]   # 唯一使用 reducer 的字段：自动追加而非覆盖
    patient_info: Dict[str, Any]              # 逐轮由 LLM 抽取合并
    collection_phase: str                     # "collecting" | "completed"
    turn_count: int
    assessment: Optional[str]
```

- 字段按重要性分三档：`_CRITICAL_FIELDS`（7 个：age/gender/chief_complaint/symptom_onset/symptom_duration/severity/accompanying_symptoms）、`_IMPORTANT_FIELDS`（2 个：past_medical_history/current_medications）、`_OPTIONAL_FIELDS`（5 个）。`_is_filled()` 用 `_EMPTY_VALUES = {"", "none", "null", "未知", "不详", "无", "没有"}` 过滤掉 LLM 抽取出的占位性回答。
- `_extract_info` 节点每轮只从**最新一条 `HumanMessage`**中抽取"本轮新增/更新"的字段（prompt 明确要求"没有新信息则返回空对象 `{}}`"），失败时 `try/except: pass`——抽取失败不会中断问诊流程，只是这一轮不更新 `patient_info`。
- `_check_completeness` 是条件路由函数，判定逻辑在 `_is_info_sufficient()`：`critical 全部已填 且 important 至少 1 个已填`，或 `turn_count >= 10`（硬性轮次上限，防止患者一直不给关键信息导致无限问诊）。
- 状态持久化靠 `MemorySaver()` + `config={"configurable":{"thread_id": session_id}}`；`get_doctor_graph()` 是模块级单例（`_doctor_graph` 全局变量，`_build_graph()` 只执行一次），所有 `session_id` 共享同一个编译后的图对象，但各自的 checkpoint 状态互相隔离。
- `chat()` 会先 `graph.get_state(config)` 检查 `collection_phase` 是否已是 `"completed"`，若是则直接返回固定提示（"本次问诊已完成…如需重新问诊，请使用新的会话 ID"），**不再调用 `graph.invoke()`**，避免误操作覆盖已生成的诊断报告。
- `completion_pct`（信息完整度百分比）不是图状态的一部分，是 `get_session_info()` 读取时用 `_calc_completion_pct()` 现算的（`已填字段数 / 14 个字段总数`）。

### 设计决策与取舍

**决策 1：`stream()` + `invoke()` 双重执行，且已有的单次执行方案未被采用**

`agent.py` 的 `run_agent_and_collect_steps`/`run_agent_stream_yield_events` 为了同时拿到"逐步骤输出"（给前端做动画）和"完整终态"，先 `for step in agent.stream(input_data, config)` 收集每步，再额外调一次 `agent.invoke(input_data, config=config)` 拿终态——**同一个输入被完整执行两遍**，有副作用的节点（如 `fund_qa_agent` 的向量检索/LLM 生成）会被调用两次。`agent_fund_qa.py` 用 `self._cached` 做了针对性规避（`.invoke()` 优先读 `.stream()` 缓存的结果），但这只是给单个 Agent 打了补丁。

更值得注意的是：`agent_research.py` 和 `agent_wealth_advisor.py` 里其实已经各自写好了单次执行的替代方案——`run_research_agent_and_collect_steps()` / `run_wealth_advisor_and_collect_steps()`，内部调用 `service/ai/langchain.py` 的 `run_graph_stream_and_collect()`（该函数文档明确写着"不再二次 invoke，避免流程跑两遍"，靠 `_merge_state_update()` 从每步的增量输出手动合并出终态）。但这两个函数**没有被 `agent.py` 或任何路由调用**，是死代码——`/ai/agent/run` 走的仍是 `agent.py` 里那份会双重执行的实现。

**决策 2：鸭子类型接口而非共享基类**

`agent.py` 用 `hasattr(agent, "get_graph")` 和 `hasattr(agent, "stream")` 判断 Agent 类型，而不是定义一个 `Protocol`/抽象基类强制约束。好处是 `_DisneyRagAgent` 这种手写类可以零成本接入通用执行代码；代价是接口契约完全隐式——`_DisneyRagAgent.stream()` 的 yield 顺序、`.invoke()` 的缓存复用逻辑，都是靠阅读 `agent.py` 里 `hasattr` 分支的调用方式反推出来的，没有任何类型系统或运行时检查保证新增的 Agent 会正确实现这套隐式协议。

**决策 3：`agent_doctor.py` 完全独立于 `agent.py` 的注册表**

医生智能体没有注册进 `AGENT_BUILDERS`，因此没有 `/ai/agent/schema?agent_id=doctor_agent` 可用，前端也拿不到它的图结构做 3D 可视化。原因大概率是它的输入模型（`session_id` + 单条 `message`，多轮累积）和其余三个 Agent 的"一次性完整输入 → 一次性完整输出"模型（`DEFAULT_INPUTS` 那种一次性传全部参数）不兼容，硬塞进统一接口会让 `run_agent_and_collect_steps` 的语义变得混乱。代价是医生智能体重复实现了一遍 config/thread_id 构造、错误处理，且无法复用 `graph_to_schema` 之类的可视化能力。

**决策 4：State 更新语义在同目录内不统一**

`research_agent`/`wealth_advisor_agent` 的节点函数普遍返回 `{**state, "some_field": ...}`（整份 state 覆盖式返回），而 `DoctorState.messages` 字段声明了 `Annotated[list, add_messages]` reducer，节点只返回 `{"messages": [AIMessage(...)]}` 增量、由 LangGraph 自动合并追加。两种更新模型混用在同一个目录下，读代码时容易搞错某个节点返回的到底是全量状态还是增量。

---

## 第四部分：应用场景与实战

### 核心使用场景

- **投资研究自动化**（`research_agent`）：输入 `research_topic`/`industry_focus`/`time_horizon`，五步线性图输出完整投研报告（`final_report`）。
- **知识库客服**（`fund_qa_agent`）：基于 `disney_knowledge_base` 向量库的两步 RAG（检索 + 生成），走 `service.ai.vector_db_qdrant.search_in_db`。
- **财富管理咨询**（`wealth_advisor_agent`）：按查询复杂度动态选择"即时响应"或"数据收集→分析→建议"深度路径。
- **多轮问诊**（`agent_doctor`）：通过 `/ai/doctor/chat` 用同一 `session_id` 连续对话，直到信息充分自动切换到诊断报告生成（仅限研究/演示用途，不能替代真实医疗诊断）。
- **前端 3D 执行动画**：`/ai/agent/schema` 拿图结构、`/ai/agent/run`（`stream=true`）拿逐步骤 SSE 事件，驱动可视化。

### 快速上手

**环境依赖**

```bash
pip install langgraph langchain langchain-openai langchain-community
```

`.env` 中需要 `DASHSCOPE_API_KEY`。注意：`agent_research.py`/`agent_wealth_advisor.py` 通过 `config.ai.dashscope_api_key()` 读取，而 `agent_doctor.py` 是 `os.environ.get("DASHSCOPE_API_KEY")` 直接读——两者理论上应指向同一个值，但走的是两条不同的读取路径，`config.ai` 里如果加了额外的校验/兜底逻辑，`agent_doctor.py` 不会受益。

**示例 1：查看 Agent 列表并执行投研 Agent**

```python
from service.ai.showcase.agents.agent import list_agents, run_agent_and_collect_steps

agents = list_agents()
# {"research_agent": {"name": "智能投研助手", "type": "deliberative", ...}, ...}

result = run_agent_and_collect_steps("research_agent", {
    "research_topic": "新能源汽车行业投资机会",
    "industry_focus": "电动汽车制造、电池技术",
    "time_horizon": "中期",
    "perception_data": None, "world_model": None, "reasoning_plans": None,
    "selected_plan": None, "final_report": None, "current_phase": "perception", "error": None,
})
for step in result["steps"]:
    print(f"[{step['nodeId']}] 耗时 {step['duration_ms']}ms")
print(result["finalState"]["final_report"])
```

**示例 2：curl 调用通用执行接口（非流式）**

```bash
curl -X POST http://localhost:3000/ai/agent/run \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "wealth_advisor_agent", "input": {"user_query": "我应该如何调整投资组合？", "customer_profile": null}}'
```

**示例 3：医生智能体多轮问诊**

```bash
curl -X POST http://localhost:3000/ai/doctor/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我头疼两天了，挺严重的"}'
# → 返回 session_id，用同一 session_id 继续对话直到 phase 变为 completed

curl http://localhost:3000/ai/doctor/session/<session_id>
# → 查看 patient_info、completion_pct、当前 phase
```

### 常见问题排查

- **`agent_id` 传 `doctor_agent` 到 `/ai/agent/run` 报"未知智能体"**：医生智能体不在 `AGENT_BUILDERS` 里，必须用 `/ai/doctor/chat`，不能走通用 Agent 接口。
- **`fund_qa_agent` 返回"迪士尼知识库暂未就绪"**：`_resolve_vector_db()` 在 `VectorDb`/`KnowledgeBase` 表里找不到 `disney_knowledge_base`，需要先通过知识库管理接口创建并完成向量化。
- **同一 `session_id` 反复调 `/ai/doctor/chat` 但一直拿不到诊断报告**：检查 `turn_count`/`patient_info` 是否卡在缺 critical 字段，`_is_info_sufficient` 要求全部 7 个 critical 字段都填了才会（连同 ≥1 个 important 字段）进入诊断阶段，否则要问满 10 轮才会强制触发。
- **`doctor/chat` 对已完成的会话调用不生效**：这是预期行为（决策见上），需要换一个新的 `session_id`（不传则自动 `uuid4()`）重新开始。
- **`stream=true` 的 SSE 客户端要能处理 `type=error`**：`agent_run_api` 出错时会发 `data: {"type":"error",...}` 后直接结束流，不会再有 `[DONE]`，前端需要显式监听这个类型，不能只等 `[DONE]`。
- **投研/财富顾问 Agent 的 JSON 解析节点偶发失败**：`JsonOutputParser` 依赖 LLM 严格输出 JSON，失败会被节点内的 `try/except` 捕获写进 `state["error"]`，但图**不会中断**，后续节点若发现依赖字段缺失只是再次记录错误、`current_phase` 回退，最终仍会跑到 `report`/`respond` 节点——排查时要看 `finalState.error` 而不是只看有没有报错。

---

## 第五部分：评估与展望

### 优势

- LangGraph 的显式图结构让四种 Agent 范式（反应式/深思熟虑/混合/多轮持久化）的执行过程完全透明，`stream(stream_mode="updates")` 天然支持逐节点观察，配合 `graph_to_schema` 可以零手工维护地驱动前端可视化。
- `agent.py` 的注册表模式（`AGENT_BUILDERS`/`AGENT_META`/`DEFAULT_INPUTS`）让"新增一个 StateGraph Agent"的成本很低，鸭子类型接口也证明了非 LangGraph 的实现（`_DisneyRagAgent`）可以低成本接入同一套执行代码。
- `agent_doctor.py` 展示了 `MemorySaver` + `thread_id` 做跨请求状态持久化的完整可用实现（含"已完成会话拒绝继续"这类边界处理），是目录里唯一的生产可参考范式。

### 局限与技术债务

- **双重执行是当前最大的性能/正确性风险**：`agent.py` 的通用执行入口对每次 `/ai/agent/run` 都跑两遍图，而修复方案（`run_graph_stream_and_collect`）已经写好在 `service/ai/langchain.py` 并被 `agent_research.py`/`agent_wealth_advisor.py` 各自封装了一层（`run_research_agent_and_collect_steps`/`run_wealth_advisor_and_collect_steps`），却完全没接入实际调用链——这是"修复已存在但未生效"的典型技术债务。
- **医生智能体游离在通用体系之外**：无法复用 `graph_to_schema`/`get_agent_schema` 做可视化，也不在 `list_agents()` 里，前端要单独适配一套接口和数据结构。
- **StateGraph 节点没有错误短路机制**：`research_agent`/`wealth_advisor_agent` 的节点在依赖缺失时只是记录 `error` 字段，图仍会走到底，可能产出基于部分缺失数据拼出来的报告，而不是在上游失败时提前终止。
- **无执行超时保护**：投研 Agent 五步 LLM 调用、财富顾问最多四步，都没有 `asyncio.wait_for` 之类的超时包装，单个节点 LLM 响应慢会拖垮整个 HTTP 请求。
- **State 更新语义不统一**（全量覆盖 vs `add_messages` 增量 reducer）、**`DASHSCOPE_API_KEY` 读取路径不统一**（`config.ai.dashscope_api_key()` vs 直接 `os.environ.get`），都是可读性/一致性层面的小债务。
- Agent 执行结果不落库，前端刷新页面后历史执行记录（`steps`/`finalState`）全部丢失。

### 演进建议

- **短期**：把 `run_agent_and_collect_steps`/`run_agent_stream_yield_events` 切换到已经写好的单次执行路径（复用 `run_graph_stream_and_collect` 的模式），消灭双重执行；给节点执行加超时保护。
- **中期**：把 `agent_doctor.py` 接入统一注册表（哪怕只接 `get_agent_schema`，不强求 `DEFAULT_INPUTS` 语义一致），或者至少把它现在重复实现的 config/thread_id 构造逻辑抽出来复用；Agent 执行结果落库，支持历史记录查询。
- **长期**：为深思熟虑型/混合型 Agent 的节点补上错误短路（`add_conditional_edges` 判断 `state.get("error")` 直接跳到 END 或错误节点）；统一 State 更新语义（要么全部用 reducer，要么明确约定"必须整份返回"）。

### 行业前沿

- **LangGraph Platform / `interrupt()`**：官方托管版本原生支持人工介入（Human-in-the-loop，本仓库 `service/ai/text2sql.py` 的 HITL 已经在用类似机制）、并行节点执行、生产级 checkpointing（Postgres/Redis 后端），可以直接替代本模块手写的 `MemorySaver`（仅内存、不跨进程）。
- **OpenAI Agents SDK / Swarm**：用 handoff 机制在多个专门化 Agent 间传递控制权，比本模块"单图单决策者"的模式更适合真正需要多 Agent 协作的场景。
- **自适应 RAG / 动态图**：根据问题复杂度决定走直接回答、单轮检索还是多轮检索，比本模块 `fund_qa_agent` 固定的"检索→生成"两步流程更省资源，也更适合和 `research_agent`/`wealth_advisor_agent` 这类深思熟虑型图融合。

---

## 变更记录

2026-08-18 全量重写。与旧版本的关键差异：

- 明确指出 `agent.py` 与四个领域 Agent 之间**没有共享基类/图结构**，是"注册表 + 鸭子类型执行代码"的关系，而非旧版描述的统一框架层。
- 新增：`agent_research.py`/`agent_wealth_advisor.py` 中已实现但**从未被调用**的单次执行方案（`run_research_agent_and_collect_steps`/`run_wealth_advisor_and_collect_steps`，基于 `run_graph_stream_and_collect`），指出这是修复"双重执行"问题的死代码，旧版文档未发现此事实。
- 新增：`DoctorState` 的 `add_messages` reducer 与 `research_agent`/`wealth_advisor_agent` 的全量覆盖式状态更新之间的语义不一致，作为设计决策取舍单独列出。
- 新增：`DASHSCOPE_API_KEY` 在 `agent_doctor.py`（直接 `os.environ.get`）与其余模块（`config.ai.dashscope_api_key()`）读取路径不一致的问题排查项。
- 补充 StateGraph 节点"无错误短路机制"的具体代码证据（`_research_modeling` 等节点在依赖缺失时仍继续执行而非中断）。
- 六个二级标题按最新写作规范调整为：背景演进、架构剖析、代码实现深度解析、应用场景与实战、评估与展望、变更记录。
