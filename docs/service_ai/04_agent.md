# LangGraph 多 Agent 工作流层

> 文件：`service/ai/agent/`（agent.py、agent_research.py、agent_fund_qa.py、agent_wealth_advisor.py）、`service/ai/langchain.py`  
> 生成日期：2026-02-26

---

## 第一部分：技术背景与演进

**问题背景**

单轮 LLM 调用适合简单问答，但遇到"先分析市场、再查财务数据、再生成报告"这类多步骤任务时力不从心。Agent 的核心思想是让 LLM 具备**规划和行动能力**：自主决定做什么（推理）、调用哪些工具（行动）、根据反馈调整计划（迭代），最终完成复杂目标。

**核心概念**

- **Agent**：具备感知→规划→行动循环的 LLM 驱动程序单元，比单次 LLM 调用多了状态管理和工具调用能力。
- **StateGraph（LangGraph）**：将 Agent 的执行过程建模为有向图，每个节点是一个处理步骤，边是条件路由。状态（State）在图中流动，节点只修改状态的特定字段。
- **三种 Agent 类型**：反应式（直接响应，单步）、深思熟虑型（多步规划后执行）、混合型（两者结合）。

**演进脉络**

| 阶段 | 方案 | 特点 |
|------|------|------|
| ReAct（2022） | LLM 交替推理（Reason）和行动（Act） | 简单有效，但无状态管理，多步任务容易丢失上下文 |
| Function Calling | LLM 结构化输出工具调用参数 | OpenAI 标准化，工具调用更可靠 |
| LangChain AgentExecutor | 封装 ReAct 循环，工具生态丰富 | 调试困难，执行流程不透明 |
| **LangGraph（2024）** | 图结构显式编排，状态可检查 | 流程透明，支持 stream 逐步观察，适合复杂工作流 |
| Multi-Agent（2024+） | 多个专门化 Agent 协作 | 分工明确，适合大型复杂任务 |

**本模块的定位**

`agent/` 目录实现三个专业领域 Agent，`agent.py` 提供统一管理入口和 Flask API，`langchain.py` 提供从 LangGraph 编译图中动态提取节点/边结构的工具函数（用于前端 3D 可视化）。三个 Agent 覆盖了反应式、深思熟虑、混合三种范式，是 LangGraph 三种设计模式的示范实现。

---

## 第二部分：架构剖析

**三个 Agent 设计对比**

| Agent | 类型 | 技术特点 | 应用场景 |
|-------|------|---------|---------|
| `research_agent` | 深思熟虑型 | 多节点 StateGraph，感知→推理→规划→执行→报告 | 投资研究，需要多步分析 |
| `fund_qa_agent` | 反应式 | 单步，知识库检索 + LLM 回答 | 迪士尼客服，需要快速响应 |
| `wealth_advisor_agent` | 混合型 | StateGraph，根据查询类型动态路由 | 财富管理咨询，兼顾速度和深度 |

**核心执行流程（以 `research_agent` 为例）**

```
输入：{research_topic, industry_focus, time_horizon}
      │
      ▼ perception 节点（感知）
        → 收集市场数据、行业背景
      │
      ▼ world_model 节点（世界模型构建）
        → 构建行业认知框架
      │
      ▼ reasoning 节点（推理规划）
        → 生成多个分析方案
      │
      ▼ plan_selection 节点（方案选择）
        → 选取最优分析路径
      │
      ▼ execution 节点（执行）
        → 深度分析、数据处理
      │
      ▼ report 节点（生成报告）
        → 输出最终投研报告
      │
输出：{final_report, ...完整 State}
```

**流程可视化机制**

`langchain.py` 的 `graph_to_schema()` 从编译后的 `StateGraph` 对象动态提取节点和边：

```python
# 自动从图结构生成前端所需的 JSON
schema = graph_to_schema(agent, node_display=RESEARCH_NODE_DISPLAY)
# 返回 {nodes: [{id, name, type, icon, description}], edges: [{source, target, type}]}
```

Agent 内部拓扑变化后，前端 3D 可视化自动跟随，无需手动维护图描述。

**`run_agent_and_collect_steps` 执行模式**

```python
# StateGraph 类型：stream 收集步骤 + invoke 获取最终状态
for step in agent.stream(input_data):
    for node_id, output in step.items():
        steps.append({"nodeId": node_id, "duration_ms": ..., "output": output})
final_state = agent.invoke(input_data)   # ← 第二次完整执行

# 非 StateGraph：直接 invoke
final_state = agent.invoke(input_data, config)
```

**关键设计原则**

- **统一注册表**：`AGENT_BUILDERS` 字典将 agent_id 映射到构建函数，新增 Agent 只需添加一行注册，`list_agents`/`get_agent_schema`/`run_agent` 接口无需修改。
- **元信息驱动**：`AGENT_META` 存储每个 Agent 的名称、描述、类型、图标，前端展示完全由元信息驱动，Agent 内部逻辑与展示解耦。

**与行业标准方案对比**

| 维度 | LangGraph（本项目） | LangChain AgentExecutor | AutoGen | CrewAI |
|------|--------------------|-----------------------|---------|--------|
| 流程控制 | 显式图结构，节点/边清晰 | 隐式循环，难以控制 | 对话驱动 | 角色驱动 |
| 可观测性 | 原生 stream，逐节点输出 | 需回调 | 需日志解析 | 有限 |
| 状态管理 | StateGraph 内置 | 手工管理 | 对话历史 | 任务上下文 |
| 多 Agent 协作 | 支持（图中节点可调其他图） | 有限 | 原生支持 | 原生支持 |
| 学习曲线 | 中（需理解图概念） | 低 | 低 | 低 |
| **选型建议** | 工作流明确、需可视化调试 | 快速原型 | 多 Agent 自由对话 | 角色扮演型任务 |

---

## 第三部分：代码实现深度解析

**核心函数清单**

| 函数 | 文件 | 作用 |
|------|------|------|
| `list_agents()` | agent.py | 返回所有 Agent 元信息 |
| `get_agent_schema(agent_id)` | agent.py | 动态提取图结构，返回前端可视化所需 JSON |
| `run_agent_and_collect_steps(agent_id, input_data)` | agent.py | 执行 Agent，收集每步输出，供前端驱动 3D 动画 |
| `graph_to_schema(agent, node_display)` | langchain.py | 从编译后的 LangGraph 提取节点/边结构 |
| `create_research_agent_workflow()` | agent_research.py | 构建投研 Agent 的 StateGraph |
| `create_fund_qa_agent()` | agent_fund_qa.py | 构建知识库问答 Agent（非 StateGraph） |
| `create_wealth_advisor_workflow()` | agent_wealth_advisor.py | 构建财富管理 Agent 的 StateGraph |

**设计决策与取舍**

**决策 1：stream + invoke 双重执行**  
原因：`stream()` 逐步收集每个节点的执行信息（用于前端动画），但无法直接获取最终完整状态，需要再 `invoke()` 一次。  
代价：整个工作流执行两遍，延迟翻倍，有副作用的节点（写数据库、调外部 API）会被执行两次。  
适用范围：当前场景是演示/可视化，可接受重复执行；生产环境中有副作用的 Agent 不能这样用。  
演进方案：利用 LangGraph 的 `checkpointer` 机制，`stream()` 结束后从 checkpoint 读取最终状态，消除双重执行。

**决策 2：非 StateGraph Agent 的简化结构**  
`fund_qa_agent` 是基于 LangChain `create_react_agent` 构建的，不是 `StateGraph`，通过 `hasattr(agent, "get_graph")` 判断类型：
```python
if hasattr(agent, "get_graph"):    # StateGraph
    schema = graph_to_schema(agent, ...)
else:                              # 非 StateGraph，返回简化三节点结构
    return {"nodes": [input, agent, output], "edges": [...]}
```
这保证了统一的接口，不同类型的 Agent 对外表现一致。

**决策 3：默认输入（`DEFAULT_INPUTS`）**  
每个 Agent 预置了一组默认输入，供演示时直接运行：
```python
DEFAULT_INPUTS = {
    "research_agent": {"research_topic": "新能源汽车行业投资机会", ...},
    "fund_qa_agent": {"messages": [{"role": "user", "content": "上海迪士尼的开放时间"}]},
    ...
}
```
调用方不传 `input_data` 时自动使用默认值，降低演示门槛。

---

## 第四部分：应用场景与实战

**使用场景**

- 投资研究自动化：输入研究主题，Agent 多步分析后输出完整投研报告
- 领域知识问答：基于知识库的客服 Agent，比直接 RAG 更能处理多轮追问
- 前端 3D 可视化：通过 `/ai/agent/schema` 和 `/ai/agent/run` 驱动 Agent 执行动画

**环境依赖**

```bash
pip install langgraph langchain langchain-openai langchain-community
export DASHSCOPE_API_KEY=sk-xxx
```

**代码示例**

```python
from service.ai.agent.agent import run_agent_and_collect_steps, list_agents

# 查看所有 Agent
agents = list_agents()
# {"research_agent": {"name": "智能投研助手", "type": "deliberative", ...}, ...}

# 执行投研 Agent
result = run_agent_and_collect_steps("research_agent", {
    "research_topic": "新能源汽车",
    "industry_focus": "电动汽车、电池",
    "time_horizon": "中期",
})
for step in result["steps"]:
    print(f"[{step['nodeId']}] 耗时 {step['duration_ms']}ms")
print(result["finalState"]["final_report"])
```

**常见问题**

- **`stream()` 后 `invoke()` 结果不一致**：LLM 有随机性，两次执行结果可能不同。演示场景可接受；生产环境建议固定 `temperature=0`。
- **`get_graph` 抛出异常**：部分 LangGraph 版本的 `get_graph()` 接口有变化，`graph_to_schema` 应做 try-except 保护。
- **Agent 执行超时**：投研 Agent 多步执行可能超过 Flask 默认请求超时，需要在 Nginx/uWSGI 层调高超时配置，或将执行改为异步任务。

---

## 第五部分：优缺点评估与未来展望

**优势**

- LangGraph StateGraph 让工作流节点/边完全透明，执行过程可逐步观察
- 统一注册表设计，扩展新 Agent 成本低
- 动态图结构提取，前端可视化无需手工维护
- 三种 Agent 类型覆盖不同业务场景，互为参考实现

**已知局限**

- 双重执行（stream + invoke）性能浪费，有副作用时存在风险
- Agent 执行结果不持久化，刷新页面后历史执行记录丢失
- 工具调用硬编码在每个 Agent 内，工具无法跨 Agent 复用
- 无执行超时保护，长时间运行的 Agent 可能阻塞 HTTP 连接

**演进建议**

- 短期：利用 `checkpointer` 消除双重执行；为 Agent 执行增加超时保护（`asyncio.wait_for`）
- 中期：将 Agent 执行结果写入数据库，支持历史记录查询和执行状态轮询（异步化）
- 长期：构建工具注册中心，工具可跨 Agent 共享；支持用户自定义配置 Agent 的工具集和工作流节点

**行业前沿**

- **LangGraph Platform**：LangGraph 云服务版，内置 checkpointing、人工介入（Human-in-the-loop）、并行节点执行等生产级特性
- **OpenAI Swarm / Agents SDK**：轻量级多 Agent 框架，通过 handoff 机制在 Agent 间传递控制权，适合大型多 Agent 协作
- **自适应 RAG Agent**：Agent 根据问题复杂度动态决定是直接回答、单轮检索还是多轮检索，比固定流程更高效
