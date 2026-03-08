# 前端对接文档（Agent / LangGraph 状态与步骤）

> 面向前端开发：Agent 智能体、LangGraph 图执行的接口约定、请求/响应格式、SSE 流式协议。  
> 对应后端：`routes/ai.py`、`service/ai/agent/agent.py`、`service/ai/langchain.py`

---

## 一、通用约定

### 1.1 响应结构

- 成功：`{ "code": 0, "msg": "ok", "data": { ... } }`
- 业务错误（如参数错误、未找到）：`{ "code": 400, "msg": "错误说明", "data": { ... } }`，HTTP 400
- 服务器错误：`{ "code": 500, "msg": "..." }`，HTTP 500
- 未找到资源：HTTP 404，body 中 `code: 404`

### 1.2 请求头

- `Content-Type: application/json`（POST 且传 JSON 时）
- 跨域：服务端在非 production 下会返回 `Access-Control-Allow-Origin: *`，生产环境由 Nginx 等配置

### 1.3 Base URL

- 文档中接口路径为相对路径，前端需拼接实际域名（如 `https://your-api.com` 或同源省略）。

---

## 二、Agent 智能体接口

用于「智能投研助手」「迪士尼客服助手」「财富管理投顾」等 3D 流程可视化：先拿到图结构，再执行并按步骤驱动动画，最后展示最终状态（含 `response`）。

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

- `data` 的 key 为 `agent_id`，用于后续 schema、run 请求。
- `type`：`deliberative` | `reactive` | `hybrid`，可用于前端展示或筛选。

---

### 2.2 获取智能体图结构（3D 可视化用）

```http
GET /ai/agent/schema?agent_id=fund_qa_agent
```

| 参数        | 类型   | 说明                    |
|-------------|--------|-------------------------|
| `agent_id`  | string | 必填，如 `fund_qa_agent` |

**响应示例**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "nodes": [
      { "id": "input", "name": "用户输入", "type": "input", "icon": "📝", "description": "接收用户查询" },
      { "id": "agent", "name": "迪士尼客服助手", "type": "process", "icon": "🏰", "description": "..." },
      { "id": "output", "name": "输出", "type": "output", "icon": "📢", "description": "返回结果" }
    ],
    "edges": [
      { "source": "input", "target": "agent", "type": "normal" },
      { "source": "agent", "target": "output", "type": "normal" }
    ],
    "executionOrder": []
  }
}
```

- `executionOrder` 在 GET 时为空，真实执行顺序由 **POST run** 的 `data.executionOrder` / `data.steps` 提供。
- 前端可用 `nodes` + `edges` 绘制 3D 图，用 `executionOrder` 或 `steps[].nodeId` 高亮当前/已执行节点。

---

### 2.3 执行智能体（非流式）

```http
POST /ai/agent/run
Content-Type: application/json
```

**请求体**

| 字段         | 类型   | 必填 | 说明 |
|--------------|--------|------|------|
| `agent_id`   | string | 否   | 默认 `research_agent` |
| `input`      | object | 否   | 与具体 Agent 对应，见下表 |
| `stream`     | boolean| 否   | 默认 `false`；为 `true` 时走 SSE，见 2.4 |

**各 Agent 推荐 `input` 示例**

- **fund_qa_agent（迪士尼）**  
  `{ "messages": [ { "role": "user", "content": "上海迪士尼乐园的开放时间是多少？" } ] }`

- **research_agent**  
  `{ "research_topic": "新能源汽车行业投资机会", "industry_focus": "电动汽车制造", "time_horizon": "中期", ... }`

- **wealth_advisor_agent**  
  `{ "user_query": "根据当前市场情况，我应该如何调整投资组合？", ... }`

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
      "executionOrder": [ "agent", "tools", "agent" ]
    },
    "steps": [
      {
        "stepIndex": 0,
        "nodeId": "agent",
        "status": "end",
        "duration_ms": 120,
        "output": { ... },
        "label": "推理"
      },
      {
        "stepIndex": 1,
        "nodeId": "tools",
        "status": "end",
        "duration_ms": 800,
        "output": { ... },
        "label": "知识库检索"
      }
    ],
    "finalState": {
      "messages": [ ... ],
      "response": "上海迪士尼乐园的开放时间为..."
    },
    "executionOrder": [ "agent", "tools", "agent" ],
    "totalSteps": 3
  }
}
```

**前端使用建议**

- **状态**：用 `data.finalState` 展示最终状态；迪士尼等用 `data.finalState.response` 作为最终回答文案。
- **步骤**：用 `data.steps` 按 `stepIndex` 顺序驱动 3D 步骤动画；`data.steps[].label` 可作步骤说明（如「推理」「知识库检索」）。
- **进度**：`data.totalSteps` 为总步数，当前步为 `stepIndex + 1`，进度 = `(stepIndex + 1) / totalSteps * 100%`。

---

### 2.4 执行智能体（流式 SSE）

请求同上，但 `stream: true`。响应为 **SSE（text/event-stream）**，不再返回 JSON body。

**SSE 事件顺序**

1. **init**（一条）  
   - 携带初始 `graphData`、`agentMeta`，用于先画图再收步骤。

2. **step**（多条）  
   - 每执行完一个节点一条，携带当前步信息，与上面 `steps[]` 中单条结构一致。

3. **done**（一条）  
   - 携带 `finalState`、`steps`、`executionOrder`、`totalSteps`、`graphData`、`agentMeta`。

4. **error**（出错时）  
   - 携带 `error` 文案。

5. **结束**  
   - 最后一条为 `data: [DONE]\n\n`。

**每条 SSE 的 `data` 为 JSON 字符串，示例：**

```text
data: {"type":"init","graphData":{...},"agentMeta":{...}}

data: {"type":"step","step":{"stepIndex":0,"nodeId":"agent","status":"end","duration_ms":120,"output":{...},"label":"推理"}}

data: {"type":"step","step":{"stepIndex":1,"nodeId":"tools","status":"end","duration_ms":800,"output":{...},"label":"知识库检索"}}

data: {"type":"done","finalState":{...},"steps":[...],"executionOrder":[...],"totalSteps":2,"graphData":{...},"agentMeta":{...}}

data: [DONE]
```

**前端建议**

- 收到 `init` 后渲染/更新 3D 图（nodes/edges）。
- 每收到一条 `step` 更新当前步骤高亮与进度（stepIndex / totalSteps）。
- **仅在收到 `done` 后再展示 `finalState.response`**，避免「先出答案、后出步骤」的体验。
- 使用 `EventSource` 时需用 POST + body，标准 `EventSource` 不支持，可用 `fetch` + `ReadableStream` 或 axios 等按行解析 `data:`。

---

## 三、LangGraph 图执行接口

用于「循环/并行/路由」等演示图的 3D 可视化，数据形态与 Agent 对齐：图结构 + 步骤 + 最终状态。

### 3.1 获取图结构

```http
GET /ai/langgraph/graph?name=loop
```

| 参数 | 类型   | 说明 |
|------|--------|------|
| `name` | string | 图名称：`router` | `loop` | `parallel`，默认 `router` |

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

- **router**：`{ "query": "今天天气怎么样？", "intent": "", "response": "" }`
- **loop**：`{ "query": "示例问题", "messages": [], "next_step": "", "iteration": 0, "response": "" }`
- **parallel**：`{ "input_text": "示例文本", "analyses": [], "final_result": "", "response": "" }`  
  若只传 `query`，会映射到 `input_text`。

**响应示例（非流式）**

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "graphData": {
      "nodes": [ ... ],
      "edges": [ ... ],
      "executionOrder": [ "think", "decide", "think", "decide", "respond" ]
    },
    "steps": [
      {
        "stepIndex": 0,
        "nodeId": "think",
        "status": "end",
        "duration_ms": 200,
        "output": { "messages": [...], "iteration": 1 },
        "iteration": 1,
        "thought": "第一轮思考要点...",
        "label": "第1轮思考"
      },
      {
        "stepIndex": 1,
        "nodeId": "decide",
        "status": "end",
        "duration_ms": 5,
        "output": { "next_step": "think" },
        "nextStep": "think",
        "label": "继续思考"
      },
      {
        "stepIndex": 4,
        "nodeId": "respond",
        "status": "end",
        "duration_ms": 150,
        "output": { "response": "最终回答内容..." },
        "response": "最终回答内容...",
        "label": "最终回答"
      }
    ],
    "finalState": {
      "messages": [ ... ],
      "iteration": 3,
      "query": "...",
      "response": "最终回答内容..."
    },
    "executionOrder": [ "think", "decide", "think", "decide", "respond" ],
    "totalSteps": 5
  }
}
```

- **状态**：`data.finalState.response` 为最终回答。
- **步骤**：`data.steps` 含 `stepIndex`、`nodeId`、`label`，loop 图还有 `iteration`、`thought`、`nextStep` 等，便于时间线/步骤条展示。

---

### 3.3 执行图（流式 SSE）

与 Agent 流式类似：`stream: true` 时响应为 SSE。

**事件顺序**

1. **init**：`{ "type": "init", "graphData": { "nodes", "edges" } }`
2. **step**：`{ "type": "step", "step": { stepIndex, nodeId, status, duration_ms, output, label, ... } }`
3. **done**：`{ "type": "done", "finalState", "steps", "executionOrder", "totalSteps" }`
4. **error**：`{ "type": "error", "error": "..." }`
5. 结束：`data: [DONE]\n\n`

**前端建议**：与 Agent 一致——用 init 画图，用 step 更新步骤与进度，用 done 再展示 `finalState.response`。

---

## 四、数据字段速查

### 4.1 图结构 `graphData`

| 字段 | 类型 | 说明 |
|------|------|------|
| `nodes` | array | `{ id, name, type, icon, description }` |
| `edges` | array | `{ source, target, type: "normal" \| "conditional" }` |
| `executionOrder` | array | 本次执行的节点 id 顺序（run 后才有） |

### 4.2 单步 `step`

| 字段 | 类型 | 说明 |
|------|------|------|
| `stepIndex` | number | 从 0 开始的步序号 |
| `nodeId` | string | 节点 id，与 graphData.nodes[].id 对应 |
| `status` | string | 当前固定为 `"end"` |
| `duration_ms` | number | 本步耗时（毫秒） |
| `output` | object | 本步输出（原始状态片段） |
| `label` | string | 前端展示用（如「推理」「知识库检索」「第1轮思考」「最终回答」） |
| `iteration` | number | （loop 图）思考轮次 |
| `thought` | string | （loop 图）本轮思考内容 |
| `nextStep` | string | （loop 图）下一节点 |
| `response` | string | （respond 节点）最终回答片段 |

### 4.3 最终状态 `finalState`

- **Agent（迪士尼）**：至少含 `messages`、`response`（最终回答文案）。
- **LangGraph**：含图定义的 state 字段，如 `response`、`query`、`iteration`、`messages` 等；展示最终回答用 `finalState.response`。

### 4.4 进度与顺序

- **总步数**：`totalSteps`（与 `steps.length` 一致）。
- **当前步**：第 n 步对应 `stepIndex === n - 1`。
- **进度**：`(currentStepIndex + 1) / totalSteps`，或直接使用 `stepIndex` / `totalSteps`。

---

## 五、错误与边界

- **未知 agent_id / graph name**：GET schema 或 POST run 返回 400，`data.allowed` 会列出合法取值。
- **run 执行异常**：非流式时 `data.error` 有文案；流式时发 `type: "error"` 的 SSE 事件。
- **流式断开**：前端应处理连接中断，可提示「连接中断，请重试」并可选重发请求（非流式）保底。

---

## 六、小结

| 能力 | 接口 | 状态 | 步骤 | 流式 |
|------|------|------|------|------|
| Agent 列表 | GET /ai/agent/list | - | - | - |
| Agent 图结构 | GET /ai/agent/schema | - | - | - |
| Agent 执行 | POST /ai/agent/run | finalState（含 response） | steps（stepIndex/label） | stream: true → SSE |
| LangGraph 图结构 | GET /ai/langgraph/graph | - | - | - |
| LangGraph 执行 | POST /ai/langgraph/run | finalState（含 response） | steps（stepIndex/label/...） | stream: true → SSE |

前端可统一：**用 graphData 画图 → 用 steps 驱动步骤/进度 → 用 finalState.response 展示最终回答**；流式时先收 step 再在 done 后展示 response，保证「先步骤、后答案」的体验。
