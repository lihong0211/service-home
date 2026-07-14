# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Context7（最新文档）

涉及第三方库/API 时，先通过 **Context7 MCP** 查最新文档再写代码（Next.js、Supabase、Prisma、Tailwind、Cloudflare Workers 等）。用 `resolve-library-id` + `query-docs`；无需用户说 "use context7"。

## Commands

**Development (hot-reload):**
```bash
python main.py
```

**Production (multi-worker):**
```bash
UVICORN_WORKERS=4 PORT=3000 python -m uvicorn app.app:app --host 0.0.0.0 --port 3000 --workers 4
```

**Background restart:**
```bash
./restart_uvicorn.sh
```
The script uses `venv/bin/python` internally. Set `PORT` and `UVICORN_WORKERS` env vars to override defaults (3000 and 1).

**Install dependencies:**
```bash
pip install -r requirements.txt
```
Fine-tuning dependencies are in `requirements.finetuning.txt` and are optional for the main API.

## Environment

Create a `.env` file in the project root. Required variables:
- `DB_USER` / `DB_PASSWORD` — shared MySQL credentials (used as fallback for all DB configs)
- `AI_DB_USER` / `AI_DB_PASSWORD` / `AI_DB_HOST` / `AI_DB_DATABASE` — overrides for the AI database (defaults to `localhost:3306/ai`)
- `CORS_ORIGIN` — CORS origin (default `*`); set `FLASK_ENV=production` to disable built-in CORS middleware

## Architecture

This is a **FastAPI** service (migrated from Flask) providing AI capabilities.

### Request Flow

```
main.py → uvicorn → app/app.py → app/factory.py (create_app)
    → routes/__init__.py (api_router)
        → routes/ai.py (register_ai)
            → service/ai/*.py  (business logic)
```

All routes are registered twice: once at `/ai/*` and once at `/api/ai/*` for backwards compatibility with old frontend clients.

### Route Registration Pattern

`routes/ai.py` uses `_ai_route()` as a wrapper that:
1. Injects a SQLAlchemy `Session` via `SessionDep` (FastAPI dependency)
2. Writes the session into a `ContextVar` so that `db.session` works in service-layer code (Flask-SQLAlchemy compat shim in `app/database.py`)
3. Calls the business view function (sync or async) and normalizes its return value via `utils/api_result.py`

Business views return `dict`, `(dict, status_code)`, or a Starlette `Response` object. `normalize_api_result` converts all of these into a proper `JSONResponse` or passes through streaming responses.

### Key Layers

| Layer | Location | Purpose |
|---|---|---|
| App factory | `app/factory.py` | Middleware, exception handlers, router mounting, lifespan (A2A sub-agent startup) |
| Database | `app/database.py` | SQLAlchemy engine, `db` compat shim (Flask-SQLAlchemy drop-in), request-scoped session via ContextVar |
| Config | `config/db.py` | Reads DB credentials from `.env`; defines `DB_CONFIG`, `DB_AI_CONFIG`, `DB_PDD_CONFIG` |
| Routes | `routes/ai.py` | Maps URL paths to service functions; handles DB session injection |
| Services | `service/ai/` | All business logic; each file is a self-contained capability module |
| Models | `model/ai/` | SQLAlchemy ORM models for AI features (knowledge base, vector DB, etc.) |
| Utils | `utils/` | `api_result.py` (response normalization), `http_body.py` (request parsing), `response.py`, `db_pool.py` |

### Service Modules (`service/ai/`)

- **chat.py** — LLM chat (text + OCR)
- **knowledge.py** — Knowledge base CRUD, multi-format document ingestion (PDF/DOCX/PPTX/TXT/MD), chunking strategy
- **vector_db_qdrant.py** — Vector DB management backed by Qdrant; handles document embedding and similarity search
- **rag.py** — RAG pipeline: retrieval from vector DB, optional query rewriting and reranking, answer generation
- **rag_enhance.py** — Query rewrite (CASEA) and rerank (DashScope) helpers
- **stt.py** — Speech-to-text (faster-whisper); WebSocket streaming endpoint registered directly on the app in `factory.py`
- **tts.py** — Text-to-speech (edge-tts)
- **image_gen.py** / **image_gen_qwen.py** — Image generation
- **video_undstanding.py** / **video_gen_task.py** — Video understanding and async generation tasks
- **text2sql.py** — LangChain SQL agent (Text-to-SQL)
- **langchain.py** — LangGraph workflow execution
- **function_call.py** / **function_call_ppt.py** — Function calling demos
- **files.py** — Generic file upload/management; LibreOffice-based doc conversion
- **agent/** — Domain-specific agents: `agent_doctor.py`, `agent_fund_qa.py`, `agent_research.py`, `agent_wealth_advisor.py`
- **a2a/** — Agent-to-Agent (A2A) workflow: orchestrator + sub-agents (outline/doc/summary) that auto-start as separate processes on port 8001–8003 at app startup
- **mcp/** — MCP (Model Context Protocol) integrations: Gaode maps, PPT generation, weather, TTS, STT
- **finetuning/** — LoRA fine-tuned model inference (medical/legal); uses `peft` + `bitsandbytes` 4-bit quantization

### Database

Single MySQL database (`ai` schema by default). Models inherit from `Base` in `app/database.py`. The `db` object in `app/database.py` is a Flask-SQLAlchemy compatibility shim — `db.session` reads from the request-scoped ContextVar, which is set by the route layer before each request.

### Data Storage

- `data/vector_dbs/` — Qdrant vector index files on disk, organized by DB name
- `data/knowledge_base/` — Uploaded source documents, organized by `kb_id`
- `lora/` — LoRA adapter weights for fine-tuned models
- `workspace/` — Working directory for A2A document generation tasks

---

## MANDATORY: 全栈自动化任务工作流

**每个 task 同时覆盖后端（service-home）和前端（ai-dashboard），在一个 session 里全部完成后再提交两个仓库。**

两个仓库路径：
- 后端：`/Users/lihong/Desktop/personal/code/service-home`（当前目录）
- 前端：`/Users/lihong/Desktop/personal/code/ai-dashboard`

### Step 1: 初始化环境

```bash
# --- 后端 ---
cd /Users/lihong/Desktop/personal/code/service-home
if [ -d "./venv" ]; then source ./venv/bin/activate; fi
if ! curl -s http://localhost:3000/ping > /dev/null 2>&1; then
  nohup python main.py > /tmp/service-home.log 2>&1 &
  sleep 4
fi

# --- 前端 ---
cd /Users/lihong/Desktop/personal/code/ai-dashboard
if [ ! -d "node_modules" ]; then npm install; fi
if ! curl -s http://localhost:5173 > /dev/null 2>&1; then
  npm run dev &
  sleep 5
fi
```

### Step 2: 选择任务

读取 `/Users/lihong/Desktop/personal/code/service-home/task.json`，选择 id 最小的 `passes: false` 任务。

### Step 3a: 实现后端（service-home）

在 `/Users/lihong/Desktop/personal/code/service-home` 目录操作：

**路由注册规范（必须使用 `_ai_route`，不用装饰器）：**

```python
# routes/ai.py 顶部 import 区域添加：
from service.ai.my_feature import my_feature_api

# register_ai() 函数末尾添加：
_ai_route(router, "/ai/my-feature/action", my_feature_api, ["POST"])
# 带路径参数：
_ai_route(router, "/ai/my-feature/{item_id}", get_api, ["GET"], ["item_id"], {"item_id": int})
```

**Service 模块规范：**

```python
# service/ai/my_feature.py
from fastapi import Request

async def my_feature_api(request: Request):
    body = await request.json()
    return {"code": 0, "msg": "success", "data": result}
```

**后端测试：**

```bash
cd /Users/lihong/Desktop/personal/code/service-home
python -c "from routes.ai import register_ai; print('Import OK')"
curl -X POST http://localhost:3000/ai/my-feature/action \
  -H "Content-Type: application/json" -d '{"key": "value"}'
```

### Step 3b: 实现前端（ai-dashboard）

在 `/Users/lihong/Desktop/personal/code/ai-dashboard` 目录操作，**必须修改四处**：

1. **`src/service/{feature}.ts`** — API 调用层

```typescript
import request from './request'
import { streamRequest, type StreamChunk } from '../utils/streamChat'

export async function fetchData(params: { id: number }) {
  const res = await request.get('/ai/my-feature/list', { params })
  return res.data  // request.ts 已解包，res.data 即为后端 data 字段
}
export async function streamChat(q: string, opts: { onChunk: (c: StreamChunk) => void }) {
  await streamRequest('/ai/my-feature/chat', { question: q }, opts)
}
```

2. **`src/pages/{FeatureName}.tsx`** — 页面组件（函数式组件 + hooks）

3. **`src/config/routes.tsx`** — 在 `skillsRoutes` 数组末尾（coze 之前）添加路由项：

```tsx
import FeatureName from '../pages/FeatureName'
// skillsRoutes 中添加：
{ path: 'feature-path', component: FeatureName, label: 'Feature Label' },
```

4. **`src/layouts/MainLayout.tsx`** — 在 `skillsMenuItems[0].children` 中（Portal 之前）添加菜单项：

```tsx
// 顶部 import 添加图标
import { NewIcon } from '@ant-design/icons'
// children 中添加：
{ key: '/skills/feature-path', icon: <NewIcon />, label: 'Feature Label' },
```

**前端测试：**

```bash
cd /Users/lihong/Desktop/personal/code/ai-dashboard
npm run build  # TypeScript 必须无错误
# 用 Playwright MCP 打开 http://localhost:5173，导航到新页面，截图确认
```

### Step 4: 更新 progress.txt

在 `/Users/lihong/Desktop/personal/code/service-home/progress.txt` 末尾追加：

```
## [YYYY-MM-DD] - Task {id}: {任务名称}
### 后端实现：[service-home 新增文件 + 路由]
### 前端实现：[ai-dashboard 新增文件 + 路由/菜单]
### 测试结果：[后端 curl + 前端 build + 浏览器截图描述]
### 注意事项：[后续 Agent 需要了解的信息]
```

### Step 5: 提交两个仓库

**先将 task.json 中对应任务的 `passes` 改为 `true`，然后分别提交：**

```bash
# 提交并推送后端
cd /Users/lihong/Desktop/personal/code/service-home
git add .
git commit -m "feat: [Task {id}] {任务名称} - 后端实现"
git push

# 提交并推送前端
cd /Users/lihong/Desktop/personal/code/ai-dashboard
git add .
git commit -m "feat: [Task {id}] {任务名称} - 前端实现"
git push
```

**规则：task.json 的 passes 更新放在 service-home 的 commit 里。**

### ⚠️ 阻塞规则

遇到以下情况**两个仓库都不提交**，输出阻塞信息并停止：
- `.env` 缺少 API Key
- 数据库连接失败
- 关键依赖安装失败
- 前端 `npm run build` 有无法修复的 TypeScript 错误

```
🚫 任务阻塞 - 需要人工介入
**当前任务**: Task {id} - {名称}
**已完成**: [后端/前端哪部分完成了]
**阻塞原因**: [具体原因]
**需要人工帮助**: [步骤]
```
