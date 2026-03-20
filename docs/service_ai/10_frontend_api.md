# 前端对接说明（Agent / LangGraph）

> 面向前端开发：智能体执行、LangGraph 图执行的接口约定、请求/响应格式、SSE 流式协议及使用示例。  
> 后端入口：`routes/ai.py`；实现：`service/ai/agent/agent.py`、`service/ai/langchain.py`  
> 生成日期：2026-02-26 | 更新：2026-03-11（LangGraph 多轮对话 history、执行监控字段 totalNodes/completedSteps/executionProgress）

---

## 一、通用约定

### 1.1 响应结构

| 场景       | HTTP | 响应体 |
|------------|------|--------|
| 成功       | 200  | `{ "code": 0, "msg": "ok", "data": { ... } }` |
| 业务错误   | 400  | `{ "code": 400, "msg": "错误说明", "data": { ... } }` |
| 未找到资源 | 404  | body 中 `code: 404` |
| 服务器错误 | 500  | `{ "code": 500, "msg": "..." }` |

### 1.2 请求头与 Base URL

- POST 且传 JSON 时：`Content-Type: application/json`
- 文档中的路径为相对路径，前端需拼接实际 Base URL（如 `https://your-api.com` 或同源省略）

---

## 二、Agent 智能体接口

用于「智能投研助手」「迪士尼客服助手」「财富管理投顾」等场景，支持 3D 流程可视化：先取图结构，再执行并按步骤驱动动画，最后展示最终状态（含 `response`）。

### 2.1 获取智能体列表

```http
GET /ai/agent/list
```

**响应示例**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "research_agent": {
      "id": "research_agent",
      "name": "智能投研助手",
      "description": "深思熟虑型智能体，适用于投资研究场景...",
      "type": "deliberative",
      "icon": "📊"
    },
    "fund_qa_agent": {
      "id": "fund_qa_agent",
      "name": "迪士尼客服助手",
      "description": "反应式智能体，回答关于迪士尼乐园、电影、角色...",
      "type": "reactive",
      "icon": "🏰"
    },
    "wealth_advisor_agent": {
      "id": "wealth_advisor_agent",
      "name": "财富管理投顾助手",
      "description": "混合型智能体...",
      "type": "hybrid",
      "icon": "💰"
    }
  }
}
```

- `data` 的 key 即 `agent_id`，用于后续 schema、run。
- `type`：`deliberative` | `reactive` | `hybrid`，可用于展示或筛选。

---

### 2.2 获取智能体图结构（3D 可视化）

```http
GET /ai/agent/schema?agent_id=fund_qa_agent
```

| 参数       | 类型   | 说明                     |
|------------|--------|--------------------------|
| `agent_id` | string | 必填，如 `fund_qa_agent` |

**响应示例（迪士尼客服）**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "nodes": [
      { "id": "input", "name": "用户输入", "type": "input", "icon": "📝", "description": "接收用户查询" },
      { "id": "retrieval", "name": "知识库检索", "type": "process", "icon": "🔍", "description": "向量检索迪士尼知识库" },
      { "id": "generation", "name": "生成回答", "type": "process", "icon": "🏰", "description": "基于检索结果生成答案" },
      { "id": "output", "name": "输出", "type": "output", "icon": "📢", "description": "返回最终答案" }
    ],
    "edges": [
      { "source": "input", "target": "retrieval", "type": "normal" },
      { "source": "retrieval", "target": "generation", "type": "normal" },
      { "source": "generation", "target": "output", "type": "normal" }
    ],
    "executionOrder": []
  }
}
```

- GET 时 `executionOrder` 为空；真实执行顺序由 **POST /ai/agent/run** 的 `data.executionOrder` / `data.steps` 提供。
- 前端可用 `nodes` + `edges` 画图，用 `steps[].nodeId` 或 `executionOrder` 高亮当前/已执行节点。

---

### 2.3 执行智能体（非流式）

```http
POST /ai/agent/run
Content-Type: application/json
```

**请求体**

| 字段       | 类型    | 必填 | 说明 |
|------------|---------|------|------|
| `agent_id` | string  | 否   | 默认 `research_agent` |
| `input`    | object  | 否   | 与具体 Agent 对应，见下表 |
| `stream`   | boolean | 否   | 默认 `false`；为 `true` 时走 SSE，见 2.4 |

**各 Agent 推荐 `input`**

| agent_id            | 推荐 input |
|---------------------|------------|
| fund_qa_agent       | `{ "messages": [ { "role": "user", "content": "上海迪士尼乐园的开放时间是多少？" } ] }` |
| research_agent      | `{ "research_topic": "新能源汽车行业投资机会", "industry_focus": "电动汽车制造", "time_horizon": "中期", ... }` |
| wealth_advisor_agent| `{ "user_query": "根据当前市场情况，我应该如何调整投资组合？", ... }` |

**响应示例（非流式）**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "agentMeta": {
      "id": "fund_qa_agent",
      "name": "迪士尼客服助手",
      "description": "...",
      "type": "reactive",
      "icon": "🏰"
    },
    "graphData": {
      "nodes": [ ... ],
      "edges": [ ... ],
      "executionOrder": [ "retrieval", "generation" ]
    },
    "steps": [
      {
        "stepIndex": 0,
        "nodeId": "retrieval",
        "status": "end",
        "duration_ms": 120,
        "output": { "query": "...", "hits": 5, "sources": [ ... ] },
        "label": "知识库检索"
      },
      {
        "stepIndex": 1,
        "nodeId": "generation",
        "status": "end",
        "duration_ms": 800,
        "output": { "query": "...", "response": "...", "sources": [ ... ], "messages": [ ... ] },
        "label": "生成回答"
      }
    ],
    "finalState": {
      "query": "上海迪士尼乐园的开放时间是多少？",
      "response": "上海迪士尼乐园的开放时间为...",
      "sources": [ { "doc_id": "...", "text": "...", "category": "...", "distance": ... } ],
      "messages": [
        { "role": "user", "content": "上海迪士尼乐园的开放时间是多少？" },
        { "role": "assistant", "content": "上海迪士尼乐园的开放时间为..." }
      ]
    },
    "executionOrder": [ "retrieval", "generation" ],
    "totalSteps": 2
  }
}
```

**前端使用建议**

- **最终回答**：用 `data.finalState.response` 展示；迪士尼等还可展示 `data.finalState.sources` 作为引用。
- **步骤动画**：用 `data.steps` 按 `stepIndex` 顺序驱动；`data.steps[].label` 作步骤说明。
- **进度**：总步数 `data.totalSteps`，当前步序号为 `stepIndex + 1`，进度 = `(stepIndex + 1) / totalSteps * 100%`。

---

### 2.4 执行智能体（流式 SSE）

请求同上，**`stream: true`**。响应为 **SSE（text/event-stream）**，不再返回 JSON body。

**SSE 事件顺序**

1. **init**（一条）  
   - 含 `graphData`、`agentMeta`，用于先画图再收步骤。

2. **step**（多条）  
   - 每执行完一个节点一条，结构与上面单条 `steps[]` 一致（含 `stepIndex`、`nodeId`、`status`、`duration_ms`、`output`、`label` 等）。

3. **done**（一条）  
   - 含 `finalState`、`steps`、`executionOrder`、`totalSteps`、`graphData`、`agentMeta`。

4. **error**（出错时）  
   - 含 `error` 文案。

5. **结束**  
   - 最后一条为 `data: [DONE]\n\n`。

**SSE 数据格式示例**

```text
data: {"type":"init","graphData":{...},"agentMeta":{...}}

data: {"type":"step","step":{"stepIndex":0,"nodeId":"retrieval","status":"end","duration_ms":120,"output":{...},"label":"知识库检索"}}

data: {"type":"step","step":{"stepIndex":1,"nodeId":"generation","status":"end","duration_ms":800,"output":{...},"label":"生成回答"}}

data: {"type":"done","finalState":{...},"steps":[...],"executionOrder":[...],"totalSteps":2,"graphData":{...},"agentMeta":{...}}

data: [DONE]
```

**前端建议**

- 收到 `init` 后渲染/更新 3D 图（nodes/edges）。
- 每收到一条 `step` 更新当前步骤高亮与进度。
- **仅在收到 `done` 后再展示 `finalState.response`**，避免「先出答案、后出步骤」的体验。
- 因需 POST + body，不能用标准 `EventSource`，请用 `fetch` + `ReadableStream` 按行解析 `data:`（见下文示例）。

---

### 2.5 医生智能体接口（多轮问诊）

医生智能体使用**独立接口**，不通过 `/ai/agent/list` 与 `/ai/agent/run`。多轮对话由同一 `session_id` 标识，后端用 MemorySaver 持久化问诊状态。

#### 发送消息（多轮对话）

```http
POST /ai/doctor/chat
Content-Type: application/json
```

**请求体**

| 字段         | 类型   | 必填 | 说明 |
|--------------|--------|------|------|
| `session_id` | string | 否   | 会话 ID，不传则服务端自动生成；同一问诊全程使用同一值 |
| `message`    | string | 是   | 患者本轮发送的消息 |

**响应示例**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "session_id": "uuid-xxx",
    "reply": "医生回复内容（引导问题或诊断报告）",
    "phase": "collecting",
    "patient_info": { "age": "32岁", "chief_complaint": "头痛", ... },
    "completion_pct": 45,
    "assessment": null
  }
}
```

- `phase`：`collecting` 表示仍在采集信息，`completed` 表示已生成诊断报告。
- `assessment`：当 `phase === "completed"` 时有值，为完整诊断分析报告文本。
- 若该会话已完成问诊，再次发送消息会返回提示「本次问诊已完成」，并带 `assessment`，不会覆盖原报告。

#### 获取会话状态

```http
GET /ai/doctor/session/<session_id>
```

**响应示例**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "session_id": "uuid-xxx",
    "patient_info": { ... },
    "phase": "collecting",
    "turn_count": 3,
    "completion_pct": 40,
    "assessment": null
  }
}
```

- 会话不存在或未开始时返回 404，`msg` 为错误说明。

---

## 三、LangGraph 图执行接口

用于「循环/并行/路由」等演示图的 3D 可视化，数据形态与 Agent 对齐：图结构 + 步骤 + 最终状态。

### 3.1 获取图结构

```http
GET /ai/langgraph/graph?name=loop
```

| 参数 | 类型   | 说明 |
|------|--------|------|
| `name` | string | 图名称：`router` \| `loop` \| `parallel`，缺省一般为 `router` |

**响应示例**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "nodes": [
      { "id": "input", "name": "入口", "type": "input", "icon": "📥", "description": "" },
      { "id": "think", "name": "思考", "type": "llm", "icon": "🤔", "description": "" },
      { "id": "decide", "name": "决策", "type": "condition", "icon": "🔀", "description": "" },
      { "id": "respond", "name": "最终回答", "type": "output", "icon": "📢", "description": "" }
    ],
    "edges": [ ... ],
    "executionOrder": []
  }
}
```

---

### 3.2 执行图（非流式）

```http
POST /ai/langgraph/run
Content-Type: application/json
```

**请求体**

| 字段     | 类型    | 必填 | 说明 |
|----------|---------|------|------|
| `graph`  | string  | 否   | `router` \| `loop` \| `parallel`，默认 `router` |
| `query`  | string  | 否   | 若传则合并进 `input.query`（如 router/loop 的查询） |
| `input`  | object  | 否   | 图状态初始值，见下表 |
| `stream` | boolean | 否   | 默认 `false`；为 `true` 时走 SSE |

**各图推荐 `input`**

| graph     | 推荐 input |
|-----------|------------|
| router    | `{ "query": "今天天气怎么样？", "intent": "", "response": "" }` |
| loop      | `{ "query": "示例问题", "messages": [], "next_step": "", "iteration": 0, "response": "" }` |
| parallel  | `{ "input_text": "示例文本", "analyses": [], "final_result": "", "response": "" }`；若只传 `query` 会映射到 `input_text` |

**多轮对话**：`input` 中可附加 `history` 字段，传入历史对话轮次，router/loop/parallel 三种图均会将上文拼入 Prompt，实现多轮记忆：

```json
{
  "graph": "router",
  "input": {
    "query": "那它的营业时间呢？",
    "history": [
      { "role": "user",      "content": "上海迪士尼乐园在哪里？" },
      { "role": "assistant", "content": "上海迪士尼乐园位于浦东新区..." }
    ]
  }
}
```

- `history` 为可选；不传时等同于单轮。
- 后端最多取最近 50 条消息（约 25 轮）传入 LLM，拼入 Prompt 的上文默认取最近 20 轮。

**响应示例（非流式）**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "graphData": { "nodes": [ ... ], "edges": [ ... ], "executionOrder": [ "think", "decide", "think", "decide", "respond" ] },
    "steps": [
      { "stepIndex": 0, "nodeId": "think", "status": "end", "duration_ms": 200, "output": { ... }, "iteration": 1, "thought": "...", "label": "第1轮思考" },
      { "stepIndex": 1, "nodeId": "decide", "status": "end", "duration_ms": 5, "output": { ... }, "nextStep": "think", "label": "继续思考" },
      { "stepIndex": 4, "nodeId": "respond", "status": "end", "duration_ms": 150, "output": { "response": "..." }, "response": "...", "label": "最终回答" }
    ],
    "finalState": { "messages": [ ... ], "iteration": 3, "query": "...", "response": "最终回答内容..." },
    "executionOrder": [ "think", "decide", "think", "decide", "respond" ],
    "totalSteps": 5,
    "totalNodes": 4,
    "completedSteps": 5,
    "executionProgress": 125.0,
    "response": "最终回答内容..."
  }
}
```

- 最终回答：`data.finalState.response` 或顶层 `data.response`（两者一致，`response` 是便捷字段）。
- 执行监控：`data.totalNodes`（图中节点总数）、`data.completedSteps`（完成步数）、`data.executionProgress`（完成百分比，step/node 比，loop 类图可能超过 100%）。
- 步骤：`data.steps` 含 `stepIndex`、`nodeId`、`label`；loop 图还有 `iteration`、`thought`、`nextStep` 等。

---

### 3.3 执行图（流式 SSE）

与 Agent 流式类似：**`stream: true`** 时响应为 SSE。

**事件顺序**：init → step（多条）→ done → 结束 `data: [DONE]\n\n`；出错时发 **error**。

- **init**：`{ "type": "init", "graphData": { "nodes", "edges" }, "totalNodes": 4 }`
- **step**：`{ "type": "step", "step": { stepIndex, nodeId, status, duration_ms, output, label, ... } }`
- **done**：`{ "type": "done", "finalState", "steps", "executionOrder", "totalSteps", "totalNodes", "completedSteps", "executionProgress", "response" }`
- **error**：`{ "type": "error", "error": "..." }`

**前端建议**：与 Agent 一致——init 画图（此时可用 `totalNodes` 初始化进度条），step 更新步骤高亮，**done 后再展示 `response`**（`finalState.response` 的快捷引用）。

---

## 四、数据字段速查

### 4.1 图结构 `graphData`

| 字段            | 类型  | 说明 |
|-----------------|-------|------|
| `nodes`         | array | `{ id, name, type, icon, description }` |
| `edges`         | array | `{ source, target, type: "normal" \| "conditional" }` |
| `executionOrder`| array | 本次执行的节点 id 顺序（run 后才有） |

### 4.2 单步 `step`

| 字段         | 类型   | 说明 |
|--------------|--------|------|
| `stepIndex`  | number | 从 0 开始的步序号 |
| `nodeId`     | string | 节点 id，与 `graphData.nodes[].id` 对应 |
| `status`     | string | 当前固定为 `"end"` |
| `duration_ms`| number | 本步耗时（毫秒） |
| `output`     | object | 本步输出（状态片段或结果） |
| `label`      | string | 前端展示用（如「知识库检索」「生成回答」「第1轮思考」「最终回答」） |
| `iteration`  | number | （loop 图）思考轮次 |
| `thought`    | string | （loop 图）本轮思考内容 |
| `nextStep`   | string | （loop 图）下一节点 |
| `response`   | string | （respond 节点）最终回答片段 |

### 4.3 最终状态 `finalState`

- **Agent（迪士尼 fund_qa_agent）**：与 LangGraph 对齐，含  
  `query`、`response`、`sources`、`messages`（`[{role, content}, ...]`）。  
  展示最终回答用 `finalState.response`，引用来源用 `finalState.sources`。
- **其他 Agent / LangGraph**：含图定义的 state 字段，如 `response`、`query`、`iteration`、`messages` 等；展示最终回答用 `finalState.response`。

### 4.4 进度与顺序

- **总步数**：`totalSteps`（与 `steps.length` 一致）。
- **当前步**：第 n 步对应 `stepIndex === n - 1`。
- **进度（基于步骤）**：`(stepIndex + 1) / totalSteps`。
- **执行监控（LangGraph 专用）**：`totalNodes`（图节点总数）、`completedSteps`（已完成步数，含循环节点多次执行）、`executionProgress`（completedSteps / totalNodes × 100，loop 图可超过 100%，建议以 `min(executionProgress, 100)` 展示）。

---

## 五、前端使用示例

### 5.1 非流式执行 Agent

```javascript
const runAgent = async (agentId, input) => {
  const res = await fetch('/ai/agent/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agent_id: agentId, input, stream: false }),
  });
  const json = await res.json();
  if (json.code !== 0) throw new Error(json.msg || json.data?.error);
  const { finalState, steps, totalSteps } = json.data;
  console.log('最终回答:', finalState.response);
  console.log('步骤数:', totalSteps);
  steps.forEach((s) => console.log(`[${s.stepIndex}] ${s.label}`, s.nodeId));
  return json.data;
};
```

### 5.2 流式执行 Agent（SSE 解析）

```javascript
const runAgentStream = async (agentId, input, onInit, onStep, onDone, onError) => {
  const res = await fetch('/ai/agent/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agent_id: agentId, input, stream: true }),
  });
  if (!res.ok) throw new Error(res.statusText);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') return;

      try {
        const event = JSON.parse(data);
        switch (event.type) {
          case 'init':
            onInit?.(event.graphData, event.agentMeta);
            break;
          case 'step':
            onStep?.(event.step);
            break;
          case 'done':
            onDone?.(event);
            break;
          case 'error':
            onError?.(event.error);
            break;
        }
      } catch (e) {
        // 忽略解析异常（如空行、注释）
      }
    }
  }
};

// 使用示例
await runAgentStream(
  'fund_qa_agent',
  { messages: [{ role: 'user', content: '上海迪士尼开放时间？' }] },
  (graphData) => { /* 渲染 3D 图 */ },
  (step) => { /* 高亮 step.nodeId，更新进度 step.stepIndex / step 总数 */ },
  (payload) => { /* 展示 payload.finalState.response */ },
  (err) => { console.error(err); }
);
```

### 5.3 获取图结构并执行 LangGraph（非流式）

```javascript
const runGraph = async (graphName, queryOrInput) => {
  const graphRes = await fetch(`/ai/langgraph/graph?name=${graphName}`);
  const graphJson = await graphRes.json();
  if (graphJson.code !== 0) throw new Error(graphJson.msg);
  const graphData = graphJson.data;

  const runRes = await fetch('/ai/langgraph/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      graph: graphName,
      query: typeof queryOrInput === 'string' ? queryOrInput : undefined,
      input: typeof queryOrInput === 'object' ? queryOrInput : undefined,
      stream: false,
    }),
  });
  const runJson = await runRes.json();
  if (runJson.code !== 0) throw new Error(runJson.msg || runJson.data?.error);
  return { graphData, runData: runJson.data };
};
```

---

## 六、错误与边界

- **未知 agent_id / graph name**：GET schema 或 POST run 返回 400，`data.allowed` 会列出合法取值。
- **run 执行异常**：非流式时 `data.error` 有文案；流式时发 `type: "error"` 的 SSE 事件。
- **流式断开**：前端应处理连接中断，可提示「连接中断，请重试」并可选重发非流式请求作保底。

---

## 七、接口一览

| 能力           | 方法 | 路径                  | 状态       | 步骤     | 流式           |
|----------------|------|-----------------------|------------|----------|----------------|
| Agent 列表     | GET  | /ai/agent/list        | -          | -        | -              |
| Agent 图结构   | GET  | /ai/agent/schema      | -          | -        | -              |
| Agent 执行     | POST | /ai/agent/run         | finalState | steps    | stream: true → SSE |
| LangGraph 图结构 | GET  | /ai/langgraph/graph   | -          | -        | -              |
| LangGraph 执行 | POST | /ai/langgraph/run     | finalState + 执行监控字段 | steps | stream: true → SSE |
| 医生智能体对话 | POST | /ai/doctor/chat       | phase/assessment | -    | -（多轮会话）  |
| 医生会话状态   | GET  | /ai/doctor/session/:id | phase/patient_info | -   | -              |

前端可统一：**用 graphData 画图 → 用 steps 驱动步骤/进度 → 用 finalState.response（或顶层 `response`）展示最终回答**；流式时先处理 step 再在 done 后展示 response，保证「先步骤、后答案」的体验。LangGraph 额外提供 `totalNodes`/`completedSteps`/`executionProgress` 用于执行监控仪表盘。

---

## 变更记录

| 日期 | 变更说明 |
|------|----------|
| 2026-03-11 | 新增医生智能体接口（2.5）；补充 LangGraph 执行监控字段（totalNodes/completedSteps/executionProgress/response）；新增多轮对话 history 支持说明；更新接口一览表与数据字段速查。 |
