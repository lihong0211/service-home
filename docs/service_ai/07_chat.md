# 对话能力层（Chat）

> 文件：`service/ai/chat.py`（已接入路由，生产实现）、`service/ai/ollama_chat.py`（未接入路由，孤立实现）
> 生成日期：2026-08-18

---

## 第一部分：背景演进

**问题背景**

"调用一次 LLM API" 和 "生产可用的对话接口" 之间有一段工程距离：请求体既可能是纯文本对话也可能带图片、既要支持一次性返回也要支持边生成边推的流式输出、多模态（尤其 OCR/视觉）模型的原始输出往往夹杂模板噪声和重复文本，必须做后处理才能直接展示给用户。本模块要解决的正是"把一个本地 Ollama 推理服务，包装成前端可以直接消费的、体验对齐云端产品的 HTTP 接口"。

**流式响应（SSE）的技术演进**

| 阶段 | 方案 | 特点 |
|------|------|------|
| 早期 | 同步等待完整回复再一次性返回 | 首字延迟等于总生成时间，长回答体验差 |
| 流式输出（SSE） | 边生成边以 `data: {...}\n\n` 逐块推送 | 首 token 延迟低，是目前对话类产品的事实标准 |
| 多模态输入 | 图片以 base64 塞进 `messages[].images` | 支持视觉理解 / OCR，无需额外文件上传接口 |
| 本地推理落地 | 基于 Ollama 的开源模型本地部署 | 免 API 费用、数据不出本机，但需要工程上补齐云端 API 天然具备的稳定性和后处理能力 |

**核心概念**

- **SSE（Server-Sent Events）**：HTTP 长连接上服务端单向持续推送文本帧的协议。本模块用 `StreamingResponse` + 生成器函数实现，每帧固定格式为 `data: {json}\n\n`，以 `data: [DONE]\n\n` 结束。
- **Ollama `/api/chat`**：本地推理服务暴露的 HTTP 接口，`stream=false` 返回一次性 JSON，`stream=true` 返回按行分隔的 JSON 流（每行一个增量 chunk，不是标准 SSE 格式，本模块负责把它转成 SSE）。
- **`keep_alive`**：Ollama 的模型常驻时长参数，`0` 表示请求结束立即从显存/内存卸载模型，`None`（不传）则沿用 Ollama 默认（约 5 分钟）保持热加载。
- **视觉输出去重**：本地小参数量视觉/OCR 模型（如 `qwen3-vl:2b`、`deepseek-ocr`）在长输出场景下容易陷入"复读"——整行重复或短语循环重复，是本模块后处理逻辑要专门解决的问题。

**本模块的定位**

`service/ai/` 目录下已有多个模块（`rag.py`、`langchain.py`、`chat_with_x/*` 等）对接云端 DashScope（通过 `_dashscope_common.py` 统一封装），但**本模块（`chat.py` / `ollama_chat.py`）不涉及云端 DashScope**，两个文件都是"本地 Ollama 对话"的实现，彼此是两套独立开发、未合并的方案：

- `chat.py`：**当前生产实现**。`routes/ai.py` 第 15 行 `from service.ai.chat import chat, ocr_chat`，并在 `register_ai()` 中注册为 `POST /ai/chat`、`POST /ai/orc`（第 237-238 行）。
- `ollama_chat.py`：**未被任何路由引用的孤立模块**（全仓库 `grep -rn "ollama_chat"` 除自身文件外无任何引用），设计比 `chat.py`更完善（统一 `image`/`images` 字段、支持 `vision` 参数在 OCR/VL 模型间切换），但从未接入 `routes/ai.py`，处于"写好了但没有接线"的状态。文档第三部分会对两者逐一说明，避免误当作同一套代码来读。

---

## 第二部分：架构剖析

**整体分层：云端对话 vs 本地 Ollama 对话的职责划分**

本仓库的对话能力实际上分成两条互不相交的路径：

```
云端路径（不在本文档范围）
  chat_with_x/*、langchain.py、rag.py 等
      → service/ai/_dashscope_common.py（统一 DashScope client/SSE 封装）
      → 阿里云 DashScope（OpenAI 兼容接口）

本地路径（本文档范围）
  routes/ai.py: POST /ai/chat, POST /ai/orc
      → service/ai/chat.py: chat() / ocr_chat()
      → requests.post(f"{OLLAMA_URL}/api/chat")
      → 本地 Ollama 进程（默认 http://localhost:11434）

  （孤立，未接线）
  service/ai/ollama_chat.py: chat() / ocr_chat()
      → 同样调用本地 Ollama /api/chat，但无路由入口
```

`chat.py` 承担"生产可用的本地对话/OCR 接口"职责：请求解析要快、要稳，视觉后处理要覆盖已知的复读问题；不承担多模型协商、复杂消息归一化的职责（这部分留在了未接线的 `ollama_chat.py` 里，尚未迁移进生产路径）。

**核心数据流：一次对话请求从输入到流式输出的完整路径**

```
客户端 POST /ai/chat  { messages, model?, stream: true, options? }
  │
  ▼
routes/ai.py: _ai_route() 生成的 handler（async）
  │  set_request_session(db) → _dispatch_ai_view(chat, request)
  ▼
_dispatch_ai_view(): inspect.iscoroutinefunction(chat) == False
  │  → await anyio.to_thread.run_sync(chat, request)   # 丢进线程池，不阻塞事件循环
  ▼
service/ai/chat.py: chat(request)                       # 运行在线程池的工作线程里
  │  data = anyio.from_thread.run(read_json_optional, request)
  │         ↑ 从工作线程"借道"回事件循环执行 await request.json()
  │  校验 messages / 取 model（默认 DEFAULT_MODEL）/ stream / options
  ▼
_stream_chat(model, messages, options, keep_alive=None, is_ocr=False, is_vision=False)
  │  requests.post(OLLAMA_URL/api/chat, stream=True, timeout=120)  # 阻塞式 HTTP，但已在工作线程，不影响事件循环
  │  for line in resp.iter_lines(): 解析每行 JSON → 组装 SSE 帧 → yield
  ▼
StreamingResponse(generate(), media_type="text/event-stream; charset=utf-8", ...)
  │  normalize_api_result() 识别出 StreamingResponse 直接透传
  ▼
客户端逐帧收到 data: {"message": {...}, "response": "增量文本", "thinking": "...", ...}\n\n
  直至 data: [DONE]\n\n
```

**关键设计原则：OCR 视觉去重**

本地小模型（尤其 2B 级别的视觉模型）在生成较长文本时容易出现两层重复：

1. **整行重复**：同一行内容被连续输出多次（如标题被念叨三遍）。
2. **行内短语循环**：单行内部是同一个短语的整数倍重复（如 `"ABABAB"`，本应只有 `"AB"`）。

`chat.py` 用 `_dedupe_vision_content()` 处理第一层，内部对每一行先调用 `_collapse_repeated_phrase()` 处理第二层，再做"连续相同行合并"。`ollama_chat.py` 只实现了第一层（没有 `_collapse_repeated_phrase`），去重能力弱于 `chat.py`——这是两个文件功能不对等的一个具体例证。

流式场景下去重更复杂：不能只对当前 chunk 做处理（重复往往跨 chunk 边界），必须对"累积到目前为止的全文"重新去重，再用字符串长度差截取出"相对上一帧新增的部分"下发，避免给前端推送已经展示过的内容。

**与行业标准方案对比**

| 维度 | 本模块（Ollama 本地部署） | 云端 API（如 DashScope / OpenAI） | 自建 vLLM/TGI 服务 |
|---|---|---|---|
| 部署与运维成本 | 单机 `ollama serve` 即可，几乎零运维 | 无需部署，按调用付费 | 需要 GPU 集群 + 服务化改造，运维成本高 |
| 吞吐与并发 | 单机单模型，并发能力弱，`chat.py` 未做请求排队/限流 | 云端弹性伸缩，高并发有保障 | 支持 PagedAttention/连续批处理，吞吐远高于 Ollama |
| 数据隐私 | 数据完全不出本机，适合敏感场景 | 数据经网络上传第三方 | 数据不出私有集群，同时具备高吞吐 |
| 多模态/OCR 效果 | 依赖开源小模型（2B 级），效果有限，需要额外后处理去重 | 大参数量商用模型，输出稳定、几乎无复读问题 | 取决于所选开源模型，通常优于 Ollama 默认量化模型 |
| 流式实现复杂度 | 需自行把 Ollama 逐行 JSON 转成 SSE，并处理去重 | SDK/网关通常已提供现成 SSE 封装 | 需自行实现，但生态工具（如 `vllm.entrypoints.openai`）已内置 OpenAI 兼容 SSE |

**选型建议**：本地开发调试、对数据隐私有硬性要求、或需要跑定制/微调模型时用本模块；生产环境有并发和效果要求、且能接受数据出域时优先云端 API（本仓库已有 `_dashscope_common.py` 路径）；如果既要本地部署又要生产级吞吐，应考虑用 vLLM 替换 Ollama（详见第五部分演进建议）。

---

## 第三部分：代码实现深度解析

**核心函数/类清单**

| 函数 | 所在文件 | 是否已接入路由 | 作用 |
|---|---|---|---|
| `chat(request)` | `chat.py`（`def`，同步） | 是，`POST /ai/chat` | 纯文本对话，不处理图片 |
| `ocr_chat(request)` | `chat.py`（`def`，同步） | 是，`POST /ai/orc` | 固定 `OCR_MODEL`，仅接收图片，走 OCR 后处理 |
| `_sync_chat(model, messages, options, keep_alive, is_ocr, is_vision)` | `chat.py` | 内部函数 | `stream=false` 时一次性调用 Ollama 并做后处理 |
| `_stream_chat(model, messages, options, keep_alive, is_ocr, is_vision)` | `chat.py` | 内部函数 | `stream=true` 时把 Ollama 的行式 JSON 流转成 SSE |
| `_dedupe_vision_content(text)` | `chat.py` / `ollama_chat.py`（两份独立实现，逻辑不同） | 内部函数 | 视觉输出行级去重 |
| `_collapse_repeated_phrase(s)` | 仅 `chat.py` | 内部函数 | 单行内重复短语折叠（`"ABABAB"→"AB"`） |
| `chat(request)` / `ocr_chat(request)` | `ollama_chat.py`（`async def`） | 否，未被任何路由引用 | 更完善的消息归一化 + `vision` 参数模型选择，但处于孤立状态 |

**关键实现细节 1：同步视图如何在异步框架里读请求体（流式 SSE 实现的前置条件）**

`chat.py` 里的 `chat()` / `ocr_chat()` 是普通 `def`（同步函数），这是仓库级别的架构约定：`routes/ai.py` 的 `_dispatch_ai_view()`（167-175 行）用 `inspect.iscoroutinefunction(view)` 判断视图类型，同步视图会被 `await anyio.to_thread.run_sync(_call)` 丢进线程池执行，避免阻塞事件循环里的其它协程。但视图内部要读请求体（`request.json()` / `request.body()`）是异步 API，只能在事件循环线程执行，因此函数体里用：

```python
data = anyio.from_thread.run(read_json_optional, request)
```

从工作线程"借道"回事件循环跑一次 `await read_json_optional(request)`，拿到结果后继续留在工作线程处理业务逻辑（包括后续对 Ollama 的阻塞式 `requests.post`，这类真阻塞调用天然适合放线程池，不需要再额外处理）。

**关键实现细节 2：流式 SSE 生成器（`_stream_chat`）**

```python
resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, stream=True, timeout=120)
resp.raise_for_status()
resp.encoding = "utf-8"
for line in resp.iter_lines(decode_unicode=True):
    if line:
        chunk = json.loads(line)                 # Ollama 每行是一个独立 JSON 对象，不是 SSE 格式
        msg = chunk.get("message", {})
        content = msg.get("content", "")
        ...
        out = dict(chunk)
        out.setdefault("response", content)       # 补一个 response 字段方便前端按老习惯读取
        out.setdefault("thinking", thinking)
        yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
yield "data: [DONE]\n\n"
```

`StreamingResponse` 的响应头显式设置了 `Cache-Control: no-cache`、`X-Accel-Buffering: no`（关闭 Nginx 反向代理的响应缓冲，否则流式效果会被代理层攒批打没）、`Connection: keep-alive`。生成器内部 `try/except` 包裹：任何异常（包括 Ollama 连接中断）都会被捕获并作为一帧 `data: {"error": "...", "done": true}\n\n` 下发，而不是让连接直接断掉、前端拿不到错误原因。

**关键实现细节 3：OCR 处理逻辑**

`ocr_chat()` 固定使用 `OCR_MODEL`（`chat.py` 中为 `"deepseek-ocr:latest"`），只接受图片输入（`images` 或单图 `image` 字段），内部拼装消息：

```python
messages = [{"role": "user", "content": "识别图中文字", "images": list(images)}]
```

`options` 默认带 `repeat_penalty: 1.35`（`chat.py`）——刻意调高重复惩罚系数以从模型采样层面减少复读，与后处理去重是两道互补的防线。OCR 输出经过两步清洗：

1. `OCR_STRIP_PREFIX` 正则去掉输出开头常见的 LaTeX/模板噪声（`<\begin`、`\begin{...}` 等，OCR 模型偶尔会把训练数据里的公式模板前缀带出来）；流式场景下只在"第一个非空 content/thinking 帧"做一次 `lstrip`，避免中间帧被误伤。
2. `_dedupe_vision_content()` 去除重复行/重复短语。

**设计决策与取舍**

1. **OCR 独立成接口而非通用视觉接口的参数分支**：`ocr_chat()` 固定 Prompt、固定模型、固定后处理策略，与通用 `chat()` 物理隔离。代价是两套接口有一定重复代码（`_sync_chat`/`_stream_chat` 倒是共用），但换来的是前端调用意图明确、OCR 特有逻辑不会污染通用对话路径。
2. **`keep_alive=0` 只用于 OCR，不用于普通对话**：OCR 请求偏低频、一次性，用完立刻卸载模型腾显存；普通对话希望保持模型热加载（`keep_alive=None` 沿用 Ollama 默认几分钟），减少连续对话时的重复加载延迟。这是牺牲"极限省显存"换"对话场景响应速度"的取舍。
3. **流式去重"整体重算 + 差分下发"而非"只处理当前 chunk"**：`_stream_chat` 里 `accumulated += content; deduped = _dedupe_vision_content(accumulated); content = deduped[len(prev_deduped):]`。每收到一帧都要对累积全文重新跑一遍去重，理论复杂度随文本长度增长而变差（`_collapse_repeated_phrase` 对单行做因子枚举，最坏是该行长度的平方级），但只有这样才能正确处理"重复内容跨越多个 chunk 边界"的情况；工程上判断在几千字的 OCR/对话场景内可接受，未做增量算法优化。
4. **两个功能重叠的模块并存而未合并**：`ollama_chat.py` 比 `chat.py` 多了 `image`/`images` 字段归一化、多图合并进最后一条 `user` 消息、`vision` 参数在 OCR/VL 间自动选模型等能力，理应是更优版本，但从未被接入路由——这是本模块当前最大的技术债务（见第五部分）。

---

## 第四部分：应用场景与实战

**核心使用场景**

- 本地私有化对话：使用 `DEFAULT_MODEL`（`chat.py` 默认 `"my-deepseek-r1-1.5"`，一个通过 `Modelfile` 自定义的 R1 风格推理模型）做纯文本对话，数据不出本机。
- 单据/文档 OCR：上传发票、截图等图片，调用 `POST /ai/orc` 走 `deepseek-ocr:latest` 识别文字，无需依赖云端 OCR 服务。
- 流式打字机效果：前端订阅 SSE，实现类 ChatGPT 的边生成边展示。

**快速上手：环境依赖**

```bash
# 安装并启动 Ollama（默认监听 11434 端口，对应代码里的 OLLAMA_URL）
brew install ollama
ollama serve

# 拉取 chat.py 实际用到的模型
ollama pull deepseek-ocr:latest
# DEFAULT_MODEL="my-deepseek-r1-1.5" 是自定义模型名，需要自备 Modelfile 用 `ollama create` 构建，
# 或直接改 service/ai/chat.py 里的 DEFAULT_MODEL 常量为已有模型（如 deepseek-r1:1.5b）
```

**代码示例 1：非流式纯文本对话**

```python
import requests

resp = requests.post("http://localhost:3000/ai/chat", json={
    "messages": [{"role": "user", "content": "用一句话介绍 SSE"}],
    "stream": False,
})
print(resp.json())
# {"code": 0, "message": {"role": "assistant", "content": "..."}}
```

**代码示例 2：流式对话（SSE）**

```python
import json, requests

resp = requests.post("http://localhost:3000/ai/chat", json={
    "messages": [{"role": "user", "content": "介绍一下深度学习"}],
    "stream": True,
}, stream=True)
for line in resp.iter_lines():
    if line and line.startswith(b"data: ") and line != b"data: [DONE]":
        chunk = json.loads(line[len(b"data: "):])
        print(chunk.get("response", ""), end="", flush=True)
```

**代码示例 3：OCR 识图（注意接口路径是 `/ai/orc` 不是 `/ai/ocr`）**

```python
import base64, requests

with open("invoice.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:3000/ai/orc", json={
    "images": [img_b64],
    "stream": False,
})
print(resp.json()["message"]["content"])
```

**常见问题排查**

- **`{"code": 503, "msg": "Ollama service not running"}`**：`chat()`/`ocr_chat()` 捕获了 `requests.exceptions.ConnectionError` 并转成该响应，说明本地 `ollama serve` 没启动，检查 11434 端口。
- **`{"code": 504, "msg": "Ollama request timeout"}`**：对 Ollama 的 HTTP 调用固定 `timeout=120` 秒，模型过大或机器算力不足会触发。
- **`POST /ai/ocr` 返回 404**：路由注册的路径是 `/ai/orc`（`routes/ai.py` 第 238 行，命名疑似笔误但已是线上路径，改动需同步前端），不要按直觉拼 `/ai/ocr`。
- **`DEFAULT_MODEL` 报模型不存在**：`"my-deepseek-r1-1.5"` 是自定义模型名，需先 `ollama create` 或改常量为已拉取的模型。
- **OCR 结果仍有重复行**：确认命中的是 `chat.py` 里的 `ocr_chat`（有 `_collapse_repeated_phrase` 双层去重），而不是误以为调用了 `ollama_chat.py`（该模块未接路由，根本不会被请求命中，也不具备行内短语去重能力）。

---

## 第五部分：评估与展望

**优势**

- 本地推理数据不出机器，适合隐私敏感场景；`keep_alive=0` 对 OCR 场景做了精细的显存回收控制。
- 针对本地小模型的复读问题设计了双层去重（行级 + 行内短语级）与流式增量下发，是这套方案里工程含量最高的部分。
- 同步 `_dispatch_ai_view` 线程池调度 + `anyio.from_thread.run` 借道读请求体的模式，让"同步业务代码"和"异步 FastAPI 框架"共存，避免了阻塞事件循环。

**局限与技术债务**

- **`ollama_chat.py` 是孤立代码**：功能比线上的 `chat.py`更完善（消息归一化、`vision` 参数、多图合并），却从未接入 `routes/ai.py`，长期存在会造成"两份对话实现该改哪个"的维护困惑，属于明确的技术债务。
- **`/ai/orc` 路径疑似笔误**：对外暴露的 OCR 接口路径不是语义清晰的 `/ai/ocr`，容易被新接入方拼错，且改路径会是破坏性变更。
- **无并发/限流控制**：`chat.py` 对 Ollama 的每次请求都是独立同步 HTTP 调用，没有队列或并发上限，多个请求同时打到单机 Ollama 时会互相抢占显存/算力，没有排队降级策略。
- **流式去重是全量重算**：`_stream_chat` 每帧都对累积全文重新跑 `_dedupe_vision_content`（其中 `_collapse_repeated_phrase` 对单行做因子枚举），文本越长单帧处理成本越高，未做增量化。
- **模型名硬编码在常量里**：`DEFAULT_MODEL`/`OCR_MODEL`/`VL_MODEL` 写死在代码里，换模型要改代码重启服务，无法运营侧动态切换。

**演进建议**

- 短期：要么把 `ollama_chat.py` 的能力合并进 `chat.py` 并接入路由，要么明确废弃删除，消除"两套实现"的认知负担；评估是否要把 `/ai/orc` 改成 `/ai/ocr`（需前端配合）。
- 中期：模型名配置化（读取环境变量/配置表而非常量），并为本地 Ollama 调用加简单的并发上限/排队，避免多请求打爆单机资源。
- 长期：生产环境如果需要更高并发和吞吐，参考第二部分对比表，评估用 vLLM/TGI 替换 Ollama，或者干脆把高质量对话场景切到已有的 DashScope 云端路径，本地路径收敛为"离线/隐私专用"分支。

**行业前沿**

- **本地推理引擎的持续优化**：Ollama、llama.cpp 生态在量化（AWQ/GPTQ/GGUF）和调度层持续优化，消费级硬件也能跑起过去需要多卡集群的模型。
- **视觉模型的复读问题正在被模型层解决**：新一代小参数量视觉模型通过更好的训练数据和采样策略（如更精细的 `repeat_penalty`/`no_repeat_ngram_size` 支持）在原生输出层面减少重复，未来后处理去重这类"打补丁"代码的必要性会逐步降低。
- **推测解码（Speculative Decoding）**：小模型起草 + 大模型校验的方式可显著提升本地推理速度，是 Ollama 等本地推理服务下一步值得关注的方向。

---

## 变更记录

2026-08-18 全量重写
