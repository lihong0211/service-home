# 方案三：多 Agent 协作链 — 使用说明

## 约定

- **调用链**与**每步 Agent 输出**均会返回给前端，用于展示流水线与各步结果。
- 编排器返回结构：`chain`（调用链）、`artifacts`（每步产出）、`final_artifact`（最终结果）。

## 启动顺序

1. 启动三个能力 Agent（各占一个端口）：
   ```bash
   cd service/ai/a2a
   python -m agents.outline_agent    # 8001
   python -m agents.doc_agent       # 8002
   python -m agents.summary_agent   # 8003
   ```
   或分别开三个终端执行：
   ```bash
   python agents/outline_agent.py
   python agents/doc_agent.py
   python agents/summary_agent.py
   ```

2. 启动编排器 API（可选，单独跑时用 8010）：
   ```bash
   cd service/ai/a2a && python -m service.ai.a2a.run_orchestrator_api   # 8010
   ```
   主程序已通过 `service.ai.a2a` 包挂载 `POST /ai/a2a/chain`，一般无需单独起编排器。

## 前端调用示例

**主程序路由（推荐）**：Flask 主应用已注册 `POST /ai/a2a/chain`，与其它 `/ai/*` 接口同域：

```bash
curl -X POST http://localhost:3000/ai/a2a/chain \
  -H "Content-Type: application/json" \
  -d '{"topic":"A2A 协议简介"}'
```

返回格式：`{ "code": 0, "msg": "ok", "data": { "chain", "artifacts", "final_artifact" } }`。

**独立编排器**（仅跑 `run_orchestrator_api.py` 时）：

```bash
curl -X POST http://localhost:8010/ai/a2a/chain \
  -H "Content-Type: application/json" \
  -d '{"topic":"A2A 协议简介"}'
```

返回示例（节选）：

```json
{
  "chain": [
    {
      "step_index": 1,
      "agent_name": "OutlineAgent",
      "agent_version": "1.0",
      "status": "completed",
      "input_summary": "topic: A2A 协议简介",
      "output_summary": "大纲",
      "started_at": "...",
      "ended_at": "...",
      "error_message": null
    },
    ...
  ],
  "artifacts": [
    { "type": "outline", "content": { "topic": "...", "sections": [...] }, "meta": { ... } },
    { "type": "document", "content": { "title": "...", "paragraphs": [...] }, "meta": { ... } },
    { "type": "summary", "content": { "title": "...", "summary": "...", "key_points": [...] }, "meta": { ... } }
  ],
  "final_artifact": { "type": "summary", "content": { ... }, "meta": { ... } }
}
```

- **chain**：用于步骤条/时间线（步骤顺序、状态、时间）。
- **artifacts**：与 chain 一一对应，用于展示每步产出（大纲/正文/摘要）。
- **final_artifact**：最后一步产出，可用于主区域默认展示。

## 目录结构（对齐 mcp 包）

```
service/ai/a2a/
├── __init__.py            # 统一导出 get_result_for_frontend，供 routes 使用
├── schemas.py             # 统一 Artifact / ChainStep / OrchestrationResult
├── orchestrator.py        # 编排逻辑，聚合 chain + artifacts
├── run_orchestrator_api.py # 独立编排器 HTTP（8010，可选）
├── README-方案3.md        # 本说明
└── agents/
    ├── outline_agent.py   # 8001
    ├── doc_agent.py       # 8002
    └── summary_agent.py   # 8003
```
