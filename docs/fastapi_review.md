# FastAPI 改造后的架构评估：最佳实践与不合理之处

## 一、当前做得合理的地方

| 方面 | 现状 | 说明 |
|------|------|------|
| **依赖注入思路** | 有 `get_db()` 生成器 | 符合 FastAPI 推荐，但当前未在路由中普遍使用 |
| **多数据源** | `_RoutingSession` + `get_bind(mapper)` | 按 model 的 `__bind_key__` 选库，设计清晰 |
| **CORS** | 按环境变量控制 | 非 production 才加 CORS，生产可交给 Nginx |
| **Lifespan** | 建表、后台服务在 `lifespan` 中 | 启动/关闭逻辑集中，符合 FastAPI 生命周期 |
| **路由组织** | `APIRouter` + `register_ai` / `register_payment` | 按业务模块拆分，结构清楚 |

---

## 二、不合理或偏离最佳实践的地方

### 1. 同步视图 + 线程池：违背 FastAPI 异步模型

**现状**：所有业务视图仍是同步函数，通过 `run_in_executor` 在线程池里跑，再包一层 `run_flask_style_view`。

**问题**：
- 每个请求都占一个线程，高并发时线程池容易成为瓶颈。
- 无法利用 FastAPI/Starlette 的异步并发优势。
- 与 FastAPI 官方推荐的「async 路由 + async 业务」不一致。

**最佳实践**：  
逐步把 CPU/IO 密集逻辑改为 `async`，或至少对纯 IO 的 service 用 `async def` + `await`，只在确实需要时用 `run_in_executor` 包同步代码。

---

### 2. 全局请求上下文（compat.request + db.session）：隐式依赖

**现状**：
- `compat.request` 通过 ContextVar 在「当前请求」注入，业务里 `from compat import request` 直接用。
- `db.session` 由中间件注入到 ContextVar，BaseModel 和 service 里直接用 `db.session`。

**问题**：
- 依赖是隐式的，路由签名里看不到需要 `Request` 或 `Session`，不利于测试和阅读。
- 单元测试必须手动 `set_request_ctx` / `set_request_session`，否则报错。
- 与 FastAPI 的「显式依赖注入」理念相反。

**最佳实践**：  
- 路由层用 `Depends(get_db)` 拿到 `Session`，再传给 service；或 service 层通过参数接收 `session` / `request`。
- 逐步去掉对全局 `request` / `db.session` 的依赖，改为函数参数。

---

### 3. ~~中间件里创建 DB Session~~（已调整）

**现状（重构后）**：HTTP 路由通过 `SessionDep`（`Annotated[Session, Depends(get_db)]`）注入 session，在 handler 内 `set_request_session(db)` 供 legacy `db.session` / BaseModel 使用；**已移除** `DBSessionMiddleware`。

**仍存折中**：业务代码仍依赖 ContextVar 中的 `db.session`，非「纯函数式」传参；后续可逐步改为显式传入 `Session`。

---

### 4. 请求体重复解析（body/form 解析两次）

**现状**：  
在 `run_flask_style_view` 里先 `await request.json()` / `await request.form()` 得到 adapter，再在线程池里跑同步视图。  
若未来在路由层又用 `Body()` / `Form()` 等依赖，会再次读 body。

**问题**：  
Starlette/FastAPI 的 request body 默认只能消费一次，重复解析要么报错，要么需要把 body 缓存在某处（当前是缓存在 adapter 里，所以暂时没问题，但和「用 FastAPI 原生 Body/Form」不兼容）。

**最佳实践**：  
- 迁移期：保持「在 compat 里统一解析并缓存」即可，但要避免在同一个请求里再对 `request.body()` 做一次读取。
- 长期：路由层用 Pydantic/`Body`/`Form` 声明请求体，直接注入到路由函数，不再经过 compat 的 request。

---

### 5. 异常处理器注册方式（404/500）

**现状**：  
`@app.exception_handler(404)` 和 `@app.exception_handler(500)`。

**问题**：  
在 FastAPI/Starlette 里，404/500 通常是由框架抛出的 `HTTPException` 或内部异常触发的，类型是 `RequestValidationError`、`HTTPException` 等。直接注册 `exception_handler(404)` 可能接不到（因为 404 是状态码，不是异常类型）。

**最佳实践**：  
- 处理 `HTTPException`：  
  `@app.exception_handler(HTTPException)`，在 handler 里根据 `exc.status_code == 404` 等分支。
- 处理验证错误：  
  `@app.exception_handler(RequestValidationError)`。
- 通用兜底：  
  `@app.exception_handler(Exception)`（你已有），用于未捕获异常并返回 500。

---

### 6. WebSocket 与 HTTP 的 session 不一致

**现状**：  
HTTP 请求通过中间件有 `db.session`；WebSocket `/api/ai/stt/live` 在 `register_stt_ws_fastapi` 里处理，没有经过该中间件。

**问题**：  
若 WebSocket 逻辑里访问 `db.session` 或 `compat.request`，会拿不到或行为与 HTTP 不一致，容易踩坑。

**最佳实践**：  
- WebSocket 若需要 DB，在 handler 内显式 `SessionLocal()` 或通过依赖拿到 session，用完后关闭。
- 在文档中说明：WebSocket 当前无请求级 session，如需 DB 需自行创建/关闭 session。

---

### 7. 配置与安全

| 问题 | 现状 | 建议 |
|------|------|------|
| CORS `*` | 非 production 下 `allow_origins=["*"]` | 生产务必收窄为具体 frontend 域名列表 |
| 敏感配置 | DB 等配置从环境变量读，部分打印到日志 | 避免在日志中打印 password、token |
| 请求体大小 | 原 Flask 有 500MB 限制 | 若保留大文件上传，在 FastAPI/Starlette 侧用中间件或配置限制 body 大小，避免滥用 |

---

### 8. 兼容层长期保留的风险

**现状**：  
`compat` 下的 `request`、`jsonify`、`Response`、`send_file`、`stream_with_context` 等，是为兼容原 Flask 写法而设。

**问题**：
- 新同学会学两套风格（FastAPI 原生 vs compat）。
- 兼容层与 FastAPI 的 Pydantic、依赖注入、类型提示割裂，类型安全和可维护性不如原生写法。
- 长期保留会拖慢真正迁移到「全 FastAPI 风格」的节奏。

**最佳实践**：  
- 在 README 或本文件中明确：compat 仅作迁移期使用，新接口一律用 FastAPI 原生（路由参数、`Depends(get_db)`、Pydantic、`HTTPException`）。
- 制定按模块/按接口的迁移计划，逐步删除对 compat 的依赖。

---

### 9. 路由注册方式：闭包与 handler 命名

**现状**：  
`_flask_view(router, path, view, methods, ...)` 里用 `async def handler(request, **kwargs)` 动态创建 handler，并 `router.add_api_route(path, handler, methods=methods)`。

**问题**：
- 大量路由共用一个名字 `handler`，调试和 OpenAPI 文档里可读性差。
- 通过闭包捕获 `view`，若未正确绑定，可能出现「最后一个 view 覆盖前面」的 bug（当前用 `view` 参数传入，每次调用都新闭包，一般没问题，但不如显式命名清晰）。

**最佳实践**：  
- 给每个动态生成的 handler 设置唯一名字，例如 `handler.__name__ = f"wrap_{view.__name__}"`，便于文档和调试。
- 或改为显式写少量路由函数，每个里再调 `run_flask_style_view`，可读性更好。

---

### 10. BaseModel 与 db.session 的强耦合

**现状**：  
`BaseModel` 的 CRUD（insert/update/delete/query）都直接使用 `db.session`（来自 ContextVar）。

**问题**：
- 难以在测试中注入 mock session 或独立事务。
- 与「session 由路由/依赖注入」的最佳实践不一致。

**最佳实践**：  
- 长期可考虑：CRUD 方法接受可选 `session` 参数，若不传则用 `db.session`（兼容现有代码），测试时传入自己的 session。
- 或新模块直接采用「repository/service 显式接 session」的模式，与 FastAPI 依赖注入对齐。

---

## 三、改进优先级建议

| 优先级 | 项 | 理由 |
|--------|----|------|
| 高 | 修正 404/500 异常处理（改用 HTTPException 等） | 否则自定义错误响应可能不生效 |
| 高 | 文档说明 WebSocket 无请求级 session | 避免在 WS 里误用 db.session |
| 中 | 为动态 handler 设置唯一 `__name__` | 提升可调试性与 OpenAPI 质量 |
| 中 | 生产环境收窄 CORS、避免打印敏感配置 | 安全与合规 |
| 低 | 逐步用 Depends(get_db) 替代中间件注入 session | 与 FastAPI 推荐一致，便于测试 |
| 低 | 新接口用原生 FastAPI（Body/Form/依赖注入） | 降低对 compat 的依赖，提升可维护性 |
| 低 | 逐步将同步视图改为 async | 提升并发与资源利用 |

---

## 四、总结

当前改造在「能跑、兼容旧代码」方面是成功的，多数据源、路由拆分、lifespan 使用也合理。  
与最佳实践的主要差距在于：

1. **大量使用同步视图 + 线程池**，未发挥 FastAPI 异步优势。  
2. **全局 request/session 隐式注入**，与依赖注入、可测试性背道而驰。  
3. **异常处理与 FastAPI 约定不完全一致**（404/500 注册方式）。  
4. **兼容层长期存在**，不利于统一到 FastAPI 原生风格和类型安全。

建议把本文档作为「迁移期设计说明」和「后续迭代清单」，按优先级逐步收敛到更符合 FastAPI 最佳实践的实现。
