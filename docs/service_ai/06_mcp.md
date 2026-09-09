# MCP 工具集层（Model Context Protocol）

> 文件：`service/ai/mcp/`（`__init__.py`、`mcp_gaode.py`、`mcp_ppt.py`、`mcp_stt.py`、`mcp_tts.py`、`mcp_weather.py`）
> 路由：`routes/ai.py`（`/ai/mcp-gaode/*`、`/ai/mcp-ppt/*`、`/ai/mcp-weather/*`、`/ai/mcp-tts/*`、`/ai/mcp-stt/*`，共 18 条）

---

## 第一部分：背景演进

**问题背景**

LLM 本质上是一个无状态的文本生成器：它不知道当前时间、查不到实时天气、也没法直接操作外部系统去生成一份 PPT 或合成一段语音。要让 LLM "做事"而不只是"说话"，必须给它接上外部工具。但工具调用长期缺少统一标准——OpenAI 的 Function Calling 只在 OpenAI 兼容 API 内有效；LangChain Tools 把工具和框架强绑定；各家 Agent 平台各自定义了一套 JSON Schema 和调用约定，工具作者需要为每个平台重新适配一遍。Model Context Protocol（MCP，Anthropic 于 2024 年发布）要解决的正是这个"胶水层爆炸"问题：把"模型如何发现、调用、消费外部工具"这件事标准化成一个协议，工具只需实现一次 MCP Server，任何支持 MCP 的客户端（Claude Desktop、Cursor、本项目的 `qwen-agent`）都能直接接入。

**核心概念**

- **MCP Server**：暴露工具能力的服务端，通过标准化的 `mcpServers` 配置对外声明可用工具，支持 `sse`、`streamable-http` 等传输方式。本项目中，ChatPPT（YOO AI）、DashScope 天气/TTS/STT MCP、高德地图 MCP（`@amap/amap-maps-mcp-server`）都是外部 MCP Server，本项目不实现协议本身，只是协议的**消费方**。
- **MCP Client**：负责与 MCP Server 建立连接、发现工具列表、发起调用的一方。本项目统一使用 `qwen_agent.agents.Assistant` 作为 MCP Client——把 `mcpServers` 配置塞进 `function_list`，`Assistant` 内部完成协议细节（握手、工具发现、调用编排），业务代码只需要处理消息进出。
- **Tool Call 循环**：用户消息进入 `Assistant.run()` → LLM（qwen-max/qwen-turbo）推理是否需要调用工具 → 若需要，MCP Client 通过协议向 MCP Server 发起调用并取回结果 → 结果作为 `function` 角色消息回填给 LLM → LLM 继续推理，直到产出面向用户的最终文本。这个循环由 `bot.run(messages)` 以生成器形式逐步 yield，每个 yield 代表流程中的一步（LLM 输出、工具调用请求、工具返回）。

**演进脉络**

| 阶段 | 方案 | 特点 |
|---|---|---|
| 早期 | 手动解析 LLM 输出文本触发工具（正则/关键词匹配） | 脆弱，强依赖 Prompt 措辞 |
| Function Calling（OpenAI，2023） | 模型原生支持结构化 JSON 参数生成 | 可靠，但协议私有，仅 OpenAI 兼容 API 可用 |
| LangChain Tools | 统一的 `BaseTool` 抽象，生态插件丰富 | 与 LangChain 框架强绑定，跨框架复用需要二次封装 |
| **MCP（Anthropic，2024）** | 传输层协议 + 工具发现 + 结构化调用，与具体 LLM/框架解耦 | 工具实现一次，Claude / Qwen / GPT 等任意支持 MCP 的客户端均可接入 |

**本模块的定位**

`service/ai/mcp/` 是本项目所有 MCP Server 接入的集成层，共五个文件，覆盖四类外部能力：地图导航（高德）、PPT 生成（ChatPPT/YOO AI）、语音合成（DashScope TTS）、语音识别（DashScope STT）、天气查询（DashScope 天气市场）。五个文件结构高度同构——各自维护一个懒加载的 `Assistant` 单例、一套消息序列化工具函数、一组 `info/chat/chat-stream` 三段式 HTTP 视图——但在鉴权方式、错误处理粒度、业务后处理（如 `mcp_ppt.py` 的 PPT 生成记录落库）上各有差异，是研究"同一集成模式在不同外部服务上如何适配"的样本集合。

---

## 第二部分：架构剖析

**整体分层**

```
routes/ai.py（_ai_route 注册）
      │
      ▼
service/ai/mcp/mcp_*.py
      │
      ├── init_agent_service() ── 构建 mcpServers 配置 + qwen_agent.Assistant 实例（进程内单例 _bot）
      ├── _get_bot()            ── 懒加载，首次调用才真正建连
      ├── run_mcp_*_chat / run_mcp_*_chat_stream ── 驱动 bot.run(messages)，产出 NDJSON 流或聚合结果
      ├── _message_to_dict()    ── 统一 qwen-agent 消息对象/dict 为纯 dict，便于 JSON 序列化
      └── get_mcp_*_info()      ── 探测配置完整性 + 已发现的工具（plugins）列表
```

各文件职责划分：

| 文件 | 接入的 MCP Server | 传输方式 | 鉴权 |
|---|---|---|---|
| `mcp_gaode.py` | `@amap/amap-maps-mcp-server`（npx 本地子进程） | stdio（`command: npx`） | `AMAP_MAPS_API_KEY` 作为 env 传给子进程 |
| `mcp_ppt.py` | ChatPPT（`http://mcp.yoo-ai.com/mcp`） | streamable-http | `YOO_API_KEY` 拼在 URL query（`?key=`） |
| `mcp_weather.py` | DashScope 天气市场（`market-cmapi033617`） | streamable-http | `DASHSCOPE_API_KEY` 作为 `Authorization: Bearer` header |
| `mcp_tts.py` | DashScope `QwenTextToSpeech` | streamable-http | `MCP_TTS_API_KEY` 或 `DASHSCOPE_API_KEY` |
| `mcp_stt.py` | DashScope `SpeechToText` | streamable-http | `MCP_STT_API_KEY` 或 `DASHSCOPE_API_KEY` |

`__init__.py` 把五个文件的 HTTP 视图函数统一收拢导出，`routes/ai.py` 从这里批量 import 后用 `_ai_route()` 逐条注册到 `/ai/mcp-*` 路径（同时也会在 `/api/ai/*` 下注册一份，兼容旧前端）。

**核心数据流：一次 MCP 工具调用的完整路径**

以 `mcp_weather_chat_stream_api` 为例（其余四个文件走同一路径，仅工具和 System Prompt 不同）：

```
1. 前端 POST /ai/mcp-weather/chat-stream  body: {messages: [{role:"user", content:"北京天气"}]}
2. _ai_route 包装层注入 DB Session（本接口不使用 DB），调用 mcp_weather_chat_stream_api(request)
3. mcp_weather_chat_stream_api 读取 body.messages，校验非空后交给 run_mcp_weather_chat_stream(messages)
4. run_mcp_weather_chat_stream 内部：
   a. _get_bot() → 若 _bot 尚未创建，调 init_agent_service()：
        - _build_weather_mcp_config() 组装 {"mcpServers": {"weather": {url, headers}}}
        - 构造 qwen_agent.agents.Assistant(llm=qwen-max, function_list=[mcp_config])
        - Assistant 初始化时即与 MCP Server 握手，拉取工具列表（存入 bot.function_map）
   b. run_messages = [dict(m) for m in messages]（拷贝，避免污染调用方原始列表）
   c. for response in bot.run(run_messages): 逐步迭代
        - 第 1 步通常是 LLM 决定调用 weather 工具，产出 function_call 消息
        - qwen-agent 内部通过 MCP 协议向天气 MCP Server 发起调用，取回结果
        - 第 2 步是携带工具结果的 function 消息 + LLM 生成的最终自然语言回复
      每一步都经 _message_to_dict() 归一化后，包成 {"event": "step", "data": [...]}，
      json.dumps 后追加换行符，构成一行 NDJSON
5. mcp_weather_chat_stream_api 用 StreamingResponse(media_type="application/x-ndjson",
   headers={"X-Accel-Buffering": "no"}) 把生成器直接透传给客户端，实现边推理边下发
6. 任意环节抛异常，被 try/except 捕获，转成 {"event": "error", "data": {"message": ...}} 单独一行下发，
   不中断已发送的内容（区别于抛 500 中断整个响应）
```

`mcp_ppt.py` 在这条主线之外还叠加了一条旁路：`run_mcp_ppt_chat`（非流式）在拿到 `final_answer` 后调用 `_try_save_ppt_record()`，从 steps 中用 `_extract_ppt_id()` / `_extract_title_from_steps()` 提取 `ppt_id` 和标题，写入 `ppt_record` 表，供 `/ai/mcp-ppt/history` 分页查询历史生成记录；`/ai/mcp-ppt/status` 则完全绕开 LLM，直接调 YOO 官方 REST API（`GET /mcp/ppt/ppt-result`）查询进度，并顺带调用 `_sync_ppt_record()` 把最新状态（`status`/`page_count`/`preview_url`）同步回本地记录表。

**关键设计原则**

1. **懒加载单例（Lazy Singleton）**：五个文件都用模块级 `_bot: Optional[Any] = None` + `_get_bot()` 的模式，直到第一次真正被请求命中才创建 `Assistant` 实例并与 MCP Server 建连，避免进程启动时对所有外部服务做一次性握手（其中任一服务不可用会拖慢或阻塞启动）。代价是首次请求会有额外的建连延迟，且单例一旦创建失败会缓存失败状态吗——不会，`_bot` 仍为 `None`，下次请求会重新尝试，这是有意为之的自愈行为。
2. **SSL / 502 重试专项处理（`mcp_ppt.py`）**：`_is_ssl_error(e)` 只匹配错误信息中含 `ssl`、`eof occurred`、`unexpected_eof` 的异常，`_is_retryable_http_error(e)` 只匹配 `httpx.HTTPStatusError` 且状态码为 502 的异常。`_collect_bot_run()` 对这两类"网络抖动型"错误做指数退避重试（1.5s → 3s → 4.5s，共 3 次），对其余异常立即失败返回，避免把业务逻辑错误也无谓重试掩盖真实问题。
3. **配置探测优先于报错**：`mcp_tts.py`、`mcp_stt.py`、`mcp_weather.py` 的 `init_agent_service()` 在真正连接前先检查 `MCP_*_URL`/`API_KEY` 是否为空，为空则抛 `ValueError` 附带明确的配置提示，而不是让底层 SDK 抛出更难懂的连接异常；`get_mcp_*_info()` 统一捕获这类异常，转成 `config_required: true` + `config_hint` 的结构化响应，供前端渲染"请先配置 XXX"的提示，而不是裸 500。
4. **STT 的超时保护是独有的**：`mcp_stt.py::get_mcp_stt_info()` 用 `threading.Thread` + `queue.Queue` 把 `_get_bot()` 放到子线程里跑，主线程 `join(timeout=3)`，超时则抛 `TimeoutError`。这是因为该接口是"信息探测"类接口，语义上应该快速返回，不应该被一次卡住的 MCP 握手拖住；其余四个文件的 `info` 接口没有做这层保护，属于已知的不一致（见第五部分）。

**与行业标准方案对比**

| 维度 | MCP（本模块采用） | 传统 Function Calling | LangChain Tools |
|---|---|---|---|
| 工具复用性 | 高：工具实现一次，任意支持 MCP 的客户端（Claude/Qwen/Cursor）均可调用 | 低：JSON Schema 与调用协议是各厂商私有格式，迁移需重写 | 中：`BaseTool` 接口统一，但强绑定 LangChain 生态 |
| 协议标准化程度 | 是，Anthropic 定义的开放规范，独立于具体 LLM | 否，每家 API（OpenAI/文心/通义）参数格式略有差异 | 否，框架私有抽象，不构成跨框架协议 |
| 工具发现机制 | 是：MCP Server 主动暴露工具列表（本项目体现为 `bot.function_map`），客户端运行时自动获知 | 否：调用方需手工维护 tools 数组传给每次请求 | 否：需手工 `import` 并注册到 Agent 的 tools 列表 |
| 异步/长任务支持 | 较好：可对接有生命周期的远程任务（如 `mcp_ppt.py` 中 PPT 生成 + 轮询状态） | 弱：约定是同步调用同步返回，长任务需业务层自行加轮询层 | 有限：取决于具体 Tool 实现，框架本身不提供任务生命周期语义 |
| 部署形态 | 灵活：进程内 stdio（高德）、远程 HTTP/SSE（ChatPPT/DashScope）均可 | 仅限调用方进程内函数或其能访问的 HTTP 接口 | 同 Function Calling，工具本体运行在调用方进程内 |

**选型建议**：需要把工具能力对接多个不同 LLM 客户端、或直接复用第三方已发布的 MCP Server（如本项目的高德/ChatPPT/DashScope）时，MCP 是目前摩擦最小的方案。若只是给单一 OpenAI 兼容模型加几个内部函数，且不考虑跨客户端复用，传统 Function Calling 实现成本更低。若项目已经深度使用 LangChain 的 Agent/Chain 体系，用 LangChain Tools 能更好地和现有链路（Memory、Retriever）组合，但要接受生态锁定。

---

## 第三部分：代码实现深度解析

**核心函数/类清单**

| 函数/结构 | 所在文件 | 作用 |
|---|---|---|
| `init_agent_service()` | 五个文件均有 | 组装 `mcpServers` 配置，构造并返回 `qwen_agent.agents.Assistant` 实例；`mcp_tts.py`/`mcp_stt.py` 版本额外做前置配置校验和 `TaskGroup` 异常的错误信息重组 |
| `_get_bot()` | 五个文件均有 | 模块级单例懒加载入口，避免重复建连 |
| `run_mcp_*_chat_stream(messages)` | 五个文件均有（`mcp_gaode.py` 只有此接口） | 驱动 `bot.run()` 生成器，逐步产出 NDJSON 行，是所有流式接口的核心 |
| `run_mcp_*_chat(messages, model, system_message)` | `mcp_ppt.py`/`mcp_weather.py`/`mcp_tts.py`/`mcp_stt.py`（`mcp_gaode.py` 无） | 非流式版本，跑完整个 `bot.run()` 循环后一次性返回 `{reply_messages, steps, history, final_answer}` |
| `_message_to_dict(msg)` | 五个文件均有（逻辑完全一致） | 把 qwen-agent 返回的消息对象（可能是 dict 或自定义对象、`content` 可能是字符串或多段列表）统一压平成 `{role, content, name?, function_call?}` 的纯 dict，保证下游可直接 `json.dumps` |
| `_collect_bot_run(bot, messages)` | `mcp_ppt.py` | 带 SSL/502 重试的 `bot.run()` 同步收集包装器，仅非流式路径使用 |
| `_try_save_ppt_record(steps, final_answer, prompt)` / `_sync_ppt_record(ppt_id, api_data)` | `mcp_ppt.py` | 从对话结果或 YOO 官方状态接口中提取 `ppt_id`/标题/页数，落库到 `ppt_record` 表，是本模块唯一带持久化副作用的逻辑 |
| `query_ppt_status(ppt_id)` / `get_ppt_download_url(ppt_id)` / `get_ppt_editor_url(ppt_id)` | `mcp_ppt.py` | 不经过 LLM，直接用 `requests` 调 YOO 官方 REST API（`YOO_API_BASE = https://saas.api.yoo-ai.com`），分别对应查进度、拿下载链接、拿在线编辑器链接 |

**关键实现细节**

- **`_extract_ppt_id(obj)` 的多层兼容**（`mcp_ppt.py:349-380`）：MCP 工具返回的 JSON 结构不固定，可能是 `{"ppt_id": "xxx"}`、`{"id": "xxx"}`，也可能嵌套一层 `{"data": {"id": "xxx"}}`。函数先尝试把字符串 `json.loads`，再依次探测顶层和 `data` 子层的 `ppt_id`/`id` 字段，且要求值是长度 > 8 的字符串（过滤掉误命中的短字符串如状态码）。
- **`mcp_ppt_download_proxy_api` 的支付前置校验**（`mcp_ppt.py:576-622`）：下载接口不是直接把 YOO 的下载链接返回给前端，而是要求携带 `out_trade_no`，通过 `PayOrder.select_one_by({"out_trade_no", "biz_id": ppt_id})` 查订单，`order.status != 2`（2 表示已支付确认）时按不同状态码分别提示"请扫码付款/审核中/订单已关闭"。校验通过后才用 `requests.get(download_url, stream=True)` 把文件流原样透传给客户端（`Content-Disposition: attachment`），本质是一个"鉴权网关 + 反向代理"。
- **`_build_*_mcp_config()` 系列函数的差异化 URL 组装**：`mcp_gaode.py` 用 stdio 传输（`command: npx`，把 Key 放进子进程 `env`）；`mcp_ppt.py` 把 `YOO_API_KEY` 直接拼进 URL query string（`f"{DEFAULT_CHATPPT_MCP_URL}?key={YOO_API_KEY}"`）；`mcp_weather.py`/`mcp_tts.py`/`mcp_stt.py` 用 `streamable-http` + `headers.Authorization: Bearer`。三种鉴权方式的差异完全由各 MCP Server 自身的约定决定，本模块只是照各家协议适配。
- **`get_mcp_*_info()` 对 `plugins` 字段的产出方式**：并不是硬编码工具名单，而是 `list(bot.function_map.keys())`——`function_map` 是 `qwen_agent.Assistant` 在建连成功、完成工具发现后自动填充的字典，因此 `plugins` 字段真实反映了当前 MCP Server 实际暴露了哪些工具，具备一定的自省能力。

**设计决策与取舍**

1. **状态查询绕过 LLM，直接打 REST API**（`mcp_ppt.py` 的 `query_ppt_status`/`get_ppt_download_url`/`get_ppt_editor_url`）。PPT 生成是异步任务，前端需要高频轮询进度；如果每次轮询都走一遍 `bot.run()`（LLM 推理 + MCP 工具调用），每次轮询要多付出 1-3 秒的 LLM 推理延迟和一次 Token 消耗。既然 YOO 官方本就提供了直连 REST 接口，轮询类操作直接绕过 LLM，只有"生成"这个真正需要语言理解的动作才走 MCP+LLM 路径。取舍是：这部分代码不再享受 MCP 协议带来的可移植性，如果 YOO 官方 REST 接口变更，需要单独维护。
2. **SSL/502 只做定向重试，不做全异常重试**（`mcp_ppt.py::_collect_bot_run`）。ChatPPT MCP Server 部署在公网，实测存在偶发的 SSL 握手失败和 502。选择只对这两类"基础设施抖动"特征的异常做指数退避重试，而不是对 `bot.run()` 抛出的任何异常都重试——因为业务逻辑错误（如工具参数不合法、Prompt 被拒答）重试没有意义，反而会拖长用户等待、掩盖真实报错，让排障更困难。
3. **配置缺失时返回结构化提示而不是抛 500**（`mcp_tts.py`/`mcp_stt.py`/`mcp_weather.py` 的 `get_mcp_*_info`）。这几个 MCP Server 依赖环境变量配置 URL 和 Key，在开发/测试环境很容易漏配。选择在 `info` 接口层面吞掉 `ValueError`/`Exception`，返回 `{config_required: true, config_hint: "..."}` 而非让请求以 500 失败，是为了让前端能够区分"服务临时故障"和"从未配置"两种情况，展示不同的引导文案，而不是一律显示"系统错误"。代价是 `chat`/`chat-stream` 接口没有做同等的语义区分，配置缺失时依然是 500，属于对称性上的不一致（见第五部分）。
4. **PPT 生成记录写入失败不影响主流程**（`_try_save_ppt_record` 整体包在 `try/except Exception` 里，仅 `print` 不重新抛出）。历史记录落库是"锦上添花"的旁路能力，即使数据库连接失败或字段解析出错，也不应该让用户拿不到已经生成好的 PPT 结果。这是典型的"核心路径与旁路能力解耦"取舍，代价是记录丢失只能靠日志排查，没有告警机制。

---

## 第四部分：应用场景与实战

**核心使用场景**

- **地图与出行助手**（`mcp_gaode.py`）：路线规划、周边景点推荐、多日行程编排，System Prompt 中预置了三套详细的角色技能（路线规划/景点推荐/行程规划）和示例回复格式。
- **PPT 一键生成**（`mcp_ppt.py`）：输入主题或上传文档，调用 ChatPPT 生成演示文稿，支持异步轮询进度、在线预览编辑、付费下载、历史记录查询，是本模块中业务闭环最完整的一个。
- **文本转语音 / 语音转文字**（`mcp_tts.py`/`mcp_stt.py`）：接入 DashScope 官方 MCP 市场里的语音合成与识别服务，用自然语言驱动"把这段话念出来""帮我转写这段音频"。
- **实时天气查询**（`mcp_weather.py`）：接入 DashScope 天气市场 MCP，用于问答场景中"今天北京天气怎么样"这类需要实时外部数据的追问。

**快速上手**

环境依赖（`.env`）：

```bash
# 全局：qwen-agent 的 LLM 调用统一走 DashScope
DASHSCOPE_API_KEY=sk-xxxx

# mcp_gaode.py：高德地图 MCP（npx 拉起本地子进程，需要本机可执行 npx）
AMAP_MAPS_API_KEY=your-amap-key

# mcp_ppt.py：ChatPPT / YOO AI
YOO_API_KEY=your-yoo-key

# mcp_weather.py：可选，不设置则使用代码里的默认市场地址
MCP_WEATHER_URL=https://dashscope.aliyuncs.com/api/v1/mcps/market-cmapi033617/mcp

# mcp_tts.py / mcp_stt.py：可选，未设置则回退到 DASHSCOPE_API_KEY
MCP_TTS_URL=https://dashscope.aliyuncs.com/api/v1/mcps/QwenTextToSpeech/sse
MCP_TTS_API_KEY=sk-xxxx
MCP_STT_URL=https://dashscope.aliyuncs.com/api/v1/mcps/SpeechToText/sse
MCP_STT_API_KEY=sk-xxxx
```

依赖包：`pip install qwen-agent dashscope requests`；高德 MCP 还需要本机已安装 Node.js（`npx` 可用）。

代码示例 1：调用天气问答（一次性接口）

```python
import requests

resp = requests.post(
    "http://localhost:3000/ai/mcp-weather/chat",
    json={"messages": [{"role": "user", "content": "北京今天天气怎么样"}]},
)
data = resp.json()["data"]
print(data["final_answer"])   # 最终自然语言回复
```

代码示例 2：流式消费 PPT 生成过程（NDJSON）

```python
import requests, json

with requests.post(
    "http://localhost:3000/ai/mcp-ppt/chat-stream",
    json={"messages": [{"role": "user", "content": "帮我生成一份关于新能源汽车市场的PPT，10页"}]},
    stream=True,
) as r:
    for line in r.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if event["event"] == "step":
            print("推理步骤:", event["data"])
        elif event["event"] == "error":
            print("错误:", event["data"]["message"])
```

代码示例 3：轮询 PPT 生成状态并下载

```python
import requests, time

ppt_id = "xxxxxxxx"
while True:
    status = requests.get(
        "http://localhost:3000/ai/mcp-ppt/status", params={"ppt_id": ppt_id}
    ).json()["data"]
    state = status.get("data", {}).get("status") if isinstance(status.get("data"), dict) else status.get("status")
    if state == 2:      # 成功
        break
    if state == 3:       # 失败
        raise RuntimeError("PPT 生成失败")
    time.sleep(3)

dl = requests.get(
    "http://localhost:3000/ai/mcp-ppt/download-url", params={"ppt_id": ppt_id}
).json()
print(dl["data"])
```

**常见问题排查**

- **`config_required: true` / `TaskGroup` 相关报错**：`mcp_tts.py`/`mcp_stt.py` 会把底层 `anyio.TaskGroup` 的子异常重新格式化成人类可读的提示，通常意味着 `MCP_TTS_URL`/`MCP_STT_URL` 不可达或 `API_KEY` 无效，先用 `curl` 直连测一下对应 MCP URL。
- **`mcp_stt_info_api` 偶发 `MCP 服务连接超时（3秒）`**：`get_mcp_stt_info()` 对建连设了 3 秒超时保护，是刻意的快速失败设计，不代表服务完全不可用；`chat`/`chat-stream` 接口没有这个超时限制，仍可正常发起对话。
- **PPT 下载返回 402**：`/ai/mcp-ppt/download` 强制要求已支付订单（`out_trade_no` 对应的 `PayOrder.status == 2`），未支付、审核中或订单已关闭都会被拒绝，需先走支付流程。
- **PPT 历史记录里 `ppt_id` 缺失/记录没写入**：`_try_save_ppt_record` 依赖从 LLM 输出或工具返回中用正则/字段探测提取 `ppt_id`，若 ChatPPT 一侧返回格式变化，提取会静默失败（仅打印日志，不抛异常），需查看服务端日志中 `[PptRecord]` 前缀的输出定位。
- **高德地图无响应**：`mcp_gaode.py` 走 stdio 传输，依赖本机能执行 `npx -y @amap/amap-maps-mcp-server`，若容器/服务器环境缺少 Node.js 或无法访问 npm registry，该子进程会静默失败，需确认部署环境已具备 Node 运行时。

---

## 第五部分：评估与展望

**优势**

- 五个 MCP 接入模块高度同构（懒加载单例 + 统一消息序列化 + info/chat/chat-stream 三段式接口），新增一个 MCP Server 接入基本是复制模板改配置，学习成本低。
- `mcp_ppt.py` 针对公网 MCP Server 的不稳定性做了专项 SSL/502 重试，且轮询类操作绕开 LLM 直连官方 REST API，兼顾了鲁棒性和响应速度。
- `get_mcp_*_info()` 把"未配置"和"运行时故障"两类问题结构化区分开（`config_required`/`config_hint`/`config_status`），便于前端做差异化引导。
- PPT 生成结果落库（`ppt_record`）+ 支付网关代理下载，把一个纯 Agent 能力和项目已有的订单体系打通，是模块中最完整的端到端业务闭环。

**局限与技术债务**

- 五个文件里 `_message_to_dict()` 完全一致地复制了五份，`ROLE`/`CONTENT`/`NAME` 等常量定义也重复了五遍，没有抽取公共基类或工具函数，属于明显的重复代码。
- 错误处理粒度不一致：只有 `mcp_stt.py` 的 `info` 接口做了超时保护，只有 `mcp_ppt.py` 做了 SSL/502 重试，其余文件对网络抖动没有任何容错；配置缺失在 `info` 接口有结构化提示，在 `chat`/`chat-stream` 接口却只是裸 500。
- `mcp_ppt.py::_extract_ppt_id`/`_extract_title_from_steps` 依赖正则和字段名猜测从 LLM 输出中"捞"结构化数据，一旦 ChatPPT MCP Server 或 LLM 的输出格式变化，提取会静默失效且没有告警，是比较脆弱的一环。
- 每次进程重启后 `_bot` 单例失效，且没有连接池/连接复用，高并发场景下每个 worker 进程都要各自与 MCP Server 重新握手，缺少共享连接层。
- `mcp_gaode.py` 用 stdio 拉起 `npx` 子进程，相比其余四个文件的 HTTP 长连接方式，在多 worker 部署下会造成每个 worker 各自维护一个 Node 子进程，资源开销和可观测性都较差。

**演进建议**

- 短期：把 `_message_to_dict()`、`ROLE`/`CONTENT`/`NAME` 等公共常量和函数抽到 `service/ai/mcp/_common.py`，五个文件统一 import，消除重复代码，也便于未来统一升级消息处理逻辑。
- 中期：把 SSL/502 重试和 STT 的超时保护提炼成通用装饰器/包装函数，应用到全部五个文件，抹平当前不一致的容错水位；`chat`/`chat-stream` 接口补上与 `info` 接口对称的 `config_required` 结构化响应。
- 长期：评估把高德地图从 stdio/npx 子进程迁移为常驻的远程 MCP Server（或托管服务），消除多 worker 下重复拉子进程的开销；为 `mcp_ppt.py` 的 `ppt_id` 提取增加更结构化的兜底（如约束 ChatPPT 侧返回固定 JSON Schema），降低对正则匹配的依赖。

**行业前沿**

- **MCP 生态快速扩张**：Claude Desktop、Cursor、Zed、Windsurf 等主流 AI 工具均已原生支持 MCP，第三方 MCP Server 市场（如 mcp.run、Smithery）已聚合数百个可直接接入的工具，本项目当前接入的高德/ChatPPT/DashScope 只是这个生态的一小部分。
- **MCP 与 A2A 协议互补**：MCP 解决"Agent 如何调用工具"，A2A（本项目 `service/ai/a2a/` 已实现）解决"Agent 如何调用另一个 Agent"，两者分别对应 Agentic AI 系统里的纵向能力扩展和横向协作扩展，正在成为业界公认的两层通信基础设施。
- **本地优先的 MCP Server 趋势**：越来越多工具提供本机运行的 MCP Server（如浏览器自动化、文件系统访问类工具），数据不出本机、延迟更低，是相对于本项目当前清一色远程 HTTP MCP Server 的一个可参考方向。

---

## 变更记录

2026-08-18 全量重写
