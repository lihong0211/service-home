# 工具能力层（Function Calling / TTS / STT / 图像生成）

> 覆盖文件：`service/ai/function_call.py`、`service/ai/function_call_ppt.py`、`service/ai/tts.py`、`service/ai/stt.py`（含 `app/factory.py` 中的 WebSocket 挂载）、`service/ai/image_gen.py`、`service/ai/image_gen_qwen.py`
> Text2SQL 见 `13_text2sql_hitl.md`，本文档不再涉及
> 最近更新：2026-08-18（全量重写）

---

## 第一部分：背景演进

### 1.1 问题背景

大语言模型本质上只是"文本续写器"：输入 token 序列，输出下一个 token 的概率分布。但真实业务需要模型"做事"——查天气、把文字读出来、把语音转成文字、把一句描述变成一张图。这些能力都不在模型权重里，必须依赖外部系统执行，模型只负责**决定调用什么、传什么参数**，执行本身发生在工具层。

本模块是这套"工具执行层"在 service-home 中的落地，按能力分为四类独立子系统：

| 子系统 | 文件 | 解决的问题 |
|---|---|---|
| Function Calling | `function_call.py` | LLM 自主决策 + 外部 API（高德天气）调用的多轮闭环 |
| Function Calling（复合场景） | `function_call_ppt.py` | 两个工具链式调用（查数据 → 生成 PPT），演示"工具的输出是下一个工具的输入" |
| TTS | `tts.py` | 文本 → 音频，语音交互的"出口" |
| STT | `stt.py` | 音频 → 文本，语音交互的"入口"，含流式/WebSocket |
| 图像生成 | `image_gen.py` / `image_gen_qwen.py` | 文本 → 图片，两条技术路线（本地 SDXL vs DiffSynth+LoRA 加速） |

### 1.2 核心概念

- **Function Calling（工具调用）**：模型在一次推理中不直接给出最终答案，而是输出一个结构化的 `tool_calls`（工具名 + JSON 参数）；框架代码执行该工具、把结果作为新的一条 `role=function` 消息追加进对话，再次请求模型，如此循环直到模型不再要求调用工具、给出自然语言最终答案。这是"ReAct 循环"的工程化、结构化实现。
- **多轮迭代上限（`max_iterations`）**：Function Calling 本质是一个 `while` 循环，必须有硬性迭代次数上限防止模型陷入"无限调用工具"的死循环——`function_call.py` 中为 5，`function_call_ppt.py` 中为 8（因为该场景天然需要至少两次工具调用：查数据 + 生成 PPT）。
- **工具链式组合**：`function_call_ppt.py` 展示了工具调用的进阶模式——`get_business_data` 的输出（数据库查询结果）会被模型读取、总结，再作为 `generate_pptx` 的输入参数（slides 内容），工具之间没有硬编码的调用顺序，完全由模型基于 system prompt 和当前对话状态决策。
- **合成与识别的对称性**：TTS（文字→音频）和 STT（音频→文字）是同一问题的两个方向。工程实现上两者都遵循"懒加载模型单例 + 临时文件落盘 + 用完即删"的模式，但底层技术栈完全不同（TTS 是云端流式合成协议，STT 是本地推理模型）。
- **设备与推理策略**：两个图像生成模块分别绑定不同硬件路径——`image_gen.py` 固定 Apple Silicon MPS 后端；`image_gen_qwen.py` 按环境变量自动探测 CUDA/CPU，体现"本地开发用 Mac，生产部署用 GPU 服务器"的双轨设计。

### 1.3 本模块定位

在 service-home 的分层里，工具能力层与 `chat.py`（纯对话）、`langchain.py`（LangGraph 工作流编排）、`text2sql.py`（数据库专用 Agent）平级，都属于 `service/ai/` 下的"能力模块"，通过 `routes/ai.py` 的 `_ai_route()` 统一暴露为 HTTP 接口。工具能力层的特点是**每个文件相对独立、无共享状态**（除 `function_call_ppt.py` 反向依赖 `text2sql.py` 和 `files.py` 之外），可以单独增删而不影响其他模块。

---

## 第二部分：架构剖析

### 2.1 整体分层

```
routes/ai.py（_ai_route 统一挂载，/ai/* 与 /api/ai/* 双前缀）
        │
        ├── function_call.py        run_function_calling_chat()      多轮 Function Calling（天气）
        ├── function_call_ppt.py    run_mcp_ppt_chat()                多轮 Function Calling（数据→PPT）
        │        └── 反向依赖 text2sql.py（查数据）、files.py（PPT 落盘 + manifest）
        ├── tts.py                  speech()                          文字 → MP3（edge-tts，异步云端调用）
        ├── stt.py                  transcribe() / transcribe_stream() / register_stt_ws*()
        │        （WebSocket 挂载在 app/factory.py::register_websocket_routes，不走 SessionDep）
        ├── image_gen.py            generate()                        文字 → PNG（SDXL，本地 MPS 推理）
        └── image_gen_qwen.py       generate_qwen()                   文字 → PNG（Qwen-Image + LoRA，仅保留代码未挂路由）
```

四个子系统各自职责单一：Function Calling 负责"决策 + 编排"，TTS/STT 负责"模态转换"，图像生成负责"内容生成"。彼此之间没有共享的运行时状态（懒加载的模型/pipeline 单例各自持有），唯一的耦合点是 `function_call_ppt.py` 显式 `import` 了 `text2sql.py` 和 `files.py` 的内部函数（`_get_business_data` 复用 `text2sql_run`；`_generate_pptx_save` 复用 `files.py` 的 `_ensure_upload_dirs`/`_load_manifest`/`_save_manifest`）。

### 2.2 关键数据流

**场景一：Function Calling 天气查询（`function_calling_chat_api` → `run_function_calling_chat`）**

```
POST /ai/function-calling/chat { messages: [...] }
  → dashscope.Generation.call(model, messages, tools=[WEATHER_TOOL], tool_choice="auto")
  → resp.output.choices[0].message 是否含 tool_calls？
       否 → content 即最终答案，跳出循环
       是 → 逐个 tool_call：解析 arguments JSON → _run_tool("get_current_weather", args)
              → get_weather_from_gaode(location, adcode) → requests.get 高德 REST API
              → 结果 json.dumps 成字符串，追加为 role=function 消息
            把 assistant(带 tool_calls) + 多条 function 消息追加进 messages
            回到循环顶部，最多 5 轮（max_iterations）
```

**场景二：PPT 链式工具调用（`run_mcp_ppt_chat`）**

```
用户："基于本月销售数据做个汇报PPT"
  → 第1轮：模型判断需要先查数据 → tool_calls=[get_business_data(question="本月销售数据")]
       → _get_business_data() 内部调用 text2sql.text2sql_run(question, model, max_rows=500)
       → 返回 {summary, sql, data} 作为 function 消息回填
  → 第2轮：模型基于查到的数据组织 slides 结构 → tool_calls=[generate_pptx(title, slides)]
       → _generate_pptx_save() 用 python-pptx 生成文件，写入 service/ai/files.py 的 manifest（file_id → {path, name, size, kb_id="ppt"}）
       → 返回 {file_id, preview_url, message}
  → 第3轮：模型无更多 tool_calls，输出最终文字（含下载链接提示）→ 跳出循环（最多 8 轮）
```

**场景三：语音合成（`speech`）**

```
POST /ai/tts { text, voice? }
  → 解析 content-type：JSON 走 request.json()，否则走 request.form()（两种输入方式兼容）
  → text 校验非空 → tempfile.mkstemp(suffix=".mp3") 生成临时路径
  → edge_tts.Communicate(text, voice).save(path)  —— 异步调用微软 Edge 只读 TTS 云端接口
  → 读入内存 BytesIO，unlink 临时文件 → StreamingResponse(media_type="audio/mpeg")
```

**场景四：语音识别（`transcribe`）**

```
POST /ai/stt/transcribe（multipart file 或 JSON audio_base64）
  → 保存为临时文件（.webm/.wav 按来源判断后缀）
  → _get_model() 懒加载单例 WhisperModel(model_size, device, compute_type)
  → model.transcribe(path, language, vad_filter=True, beam_size=5) 返回 segments 生成器 + info
  → 拼接 segments 文本、收集 {start, end, text} 列表、读取 info.language/language_probability
  → finally 块中无条件 os.unlink(audio_path)，避免临时文件堆积
```

**场景五：图像生成（`generate`，SDXL）**

```
POST /ai/image/generate { prompt, seed? }
  → _get_pipeline() 懒加载 StableDiffusionXLPipeline.from_pretrained(...).to("mps")，enable_attention_slicing() 降低显存峰值
  → pipe(prompt, height=96, width=96, num_inference_steps=25, guidance_scale=7.0, generator=seed 可选)
  → image.save(tmp) → 读入 BytesIO → StreamingResponse(media_type="image/png")
```

### 2.3 关键设计原则

1. **懒加载单例（Lazy Singleton）**：`_whisper_model`、`_image_pipe`、`_image_pipe_qwen` 均为模块级全局变量，首次请求时才初始化并常驻内存，避免服务启动时加载全部大模型拖慢冷启动，也避免每次请求重复加载模型的高延迟。代价是无并发保护——多个请求同时触发首次加载可能重复实例化（见第五部分局限）。
2. **临时文件 + finally 强制清理**：TTS/STT/图像生成都遵循"落盘临时文件 → 处理 → 读入内存 → 删除临时文件"的模式，而不是尝试纯内存流式处理，因为底层库（edge_tts、faster_whisper、diffusers）的 API 大多要求文件路径而非内存流。STT 用 `try/finally` 保证异常路径也能清理，避免临时文件泄漏。
3. **JSON / form 双输入兼容**：`tts.speech`、`stt.transcribe`、`image_gen.generate` 都通过判断 `Content-Type` 头，同时支持 `application/json` 和 `multipart/form-data`/`x-www-form-urlencoded` 两种请求体格式，兼容不同前端调用习惯（表单上传文件 vs JSON 传 base64）。
4. **工具调用循环的统一结构**：`function_call.py` 与 `function_call_ppt.py` 的核心循环高度同构（都是"请求模型 → 判断 tool_calls → 执行 → 回填 → 再请求"），但没有被抽成公共函数——这是有意为之的重复（详见第三部分设计取舍）。
5. **同步阻塞推理不做队列化**：STT 的 `model.transcribe()`、图像生成的 `pipe()` 调用都是同步阻塞操作，直接在 async 路由函数里调用，依赖 FastAPI/Starlette 的线程池承接，未引入 Celery/RQ 等任务队列（视频生成等真正耗时的任务走 `video_gen_task.py` 的异步任务模式，工具层这几个能力认为耗时可接受）。

### 2.4 与行业标准方案对比

**TTS 方案对比**

| 维度 | Edge-TTS（本项目采用） | 阿里云 DashScope TTS | Azure Cognitive Speech | Coqui/本地 TTS 模型 |
|---|---|---|---|---|
| 成本 | 免费，无需 API Key | 按字符计费 | 按字符计费，较贵 | 免费但需自建算力 |
| 音质 | 接近真人，多国语言/音色 | 优秀，中文场景优化好 | 优秀，企业级 SLA | 依赖模型，参差不齐 |
| 延迟 | 依赖微软服务端，网络抖动敏感 | 国内网络下更稳定 | 国际网络下稳定 | 本地推理，可控但需 GPU |
| 合规/稳定性 | 非官方 API（逆向 Edge 浏览器接口），无 SLA 保证，随时可能被限流/下线 | 官方商用 API，有 SLA | 官方商用 API，有 SLA | 完全自控，无外部依赖风险 |
| 集成复杂度 | 极低（`edge_tts.Communicate` 两行代码） | 中（需处理签名、鉴权） | 中（需 SDK + 密钥） | 高（需部署模型服务） |

**选型建议**：Demo/内部工具/低成本试错阶段用 Edge-TTS 性价比最高；一旦涉及生产环境对外承诺可用性，应迁移到 DashScope 或 Azure 等有 SLA 保障的商用 API（`service/ai/mcp/` 目录下已有 TTS/STT 的 MCP 封装可作为替换路径）。

**STT 方案对比**

| 维度 | faster-whisper（本项目采用） | OpenAI Whisper API | DashScope 语音识别 | 阿里云/腾讯云实时 ASR |
|---|---|---|---|---|
| 成本 | 免费，本地推理 | 按分钟计费 | 按调用计费 | 按分钟/并发计费 |
| 延迟 | 本地 CPU int8 下可接受，无网络往返 | 依赖网络 + 排队 | 依赖网络 | 专为实时优化，延迟最低 |
| 精度 | 依赖 model_size（base 精度一般，large-v3 接近商用水准） | 稳定高精度 | 中文场景优化较好 | 中文场景通常最优 |
| 实时流式 | 本项目为"分段转录"伪流式（SSE/WS 每次收完整段才转录），非真流式 | 不支持真流式 | 部分支持 | 原生支持真流式（边说边出字） |
| 隐私/数据出境 | 音频不出本机，适合敏感场景 | 音频上传境外服务器 | 音频上传云端 | 音频上传云端 |

**选型建议**：对隐私敏感或成本敏感的场景，faster-whisper 本地部署是合理选择，但如果产品需要"边说边出字"的真实时体验，当前 `stt.py` 的实现（等一段音频收完再整段转录）达不到，需要接入支持流式增量识别的商用 ASR（如阿里云实时语音识别）或切换到支持流式解码的开源方案。

---

## 第三部分：代码实现深度解析

### 3.1 核心函数/类清单

| 函数/常量 | 文件 | 作用 |
|---|---|---|
| `WEATHER_TOOL` | `function_call.py` | 天气查询工具的 JSON Schema 定义（DashScope Function Calling 格式） |
| `run_function_calling_chat(messages, model, system_message, max_iterations=5)` | `function_call.py` | 多轮 Function Calling 主循环，返回最终文本或 `{"error": ...}` |
| `GET_BUSINESS_DATA_TOOL` / `GENERATE_PPTX_TOOL` | `function_call_ppt.py` | 两个链式工具的 Schema 定义 |
| `run_mcp_ppt_chat(messages, model, system_message, max_iterations=8)` | `function_call_ppt.py` | 数据查询 → PPT 生成的多轮工具调用循环，返回 `{reply_messages, steps, history, final_answer}` |
| `_generate_pptx_save(title, slides)` | `function_call_ppt.py` | 用 `python-pptx` 落地生成 `.pptx` 文件，写入 `files.py` 的 manifest |
| `speech(request)` | `tts.py` | TTS 主接口，`edge_tts.Communicate(text, voice).save(path)` 后以 `StreamingResponse` 返回 MP3 |
| `transcribe(request)` / `transcribe_stream(request)` | `stt.py` | STT 主接口（一次性返回全部 segments）与 SSE 分段推送变体 |
| `register_stt_ws_fastapi(websocket)` | `stt.py` | FastAPI 原生 WebSocket 循环，接收 base64 音频分段并逐段回传文本 |
| `generate(request)` / `generate_qwen(request)` | `image_gen.py` / `image_gen_qwen.py` | 文生图主接口，分别对应 SDXL 与 Qwen-Image+LoRA 两条推理路径 |

### 3.2 关键实现细节

- **`_run_tool` 的参数容错**：`function_call.py` 与 `function_call_ppt.py` 中，模型返回的 `tool_calls[i].function.arguments` 是字符串形式的 JSON（DashScope 协议要求），解析时用 `try/except` 包裹 `json.loads`，失败时兜底为 `{}` 而不是抛异常中断整个对话——因为模型偶尔会返回格式不严格的参数（如尾逗号、非法转义），容错比硬失败更符合"多轮对话不应因单次解析失败而彻底中断"的产品预期。
- **`tool_calls` 的对象/字典双形态兼容**：DashScope SDK 返回的 `tool_calls` 里每个元素可能是 `dict` 也可能是有 `.function` 属性的对象（取决于 SDK 版本/序列化路径），两个文件里都用 `isinstance(tc, dict)` 分支处理，`getattr(fn, "name", "")` 与 `fn.get("name", "")` 并存，这是对上游 SDK 行为不完全稳定的防御性编码。
- **PPT 文件名的中文脱敏**：`_generate_pptx_save` 中磁盘文件名固定为 `{file_id}_report.pptx`（纯 uuid+英文），不直接用用户传入的 `title` 做文件名，避免中文/特殊字符在不同文件系统下的编码问题；对外展示的 `display_name` 才使用原始 `title`（截断到 80 字符 + `.pptx` 后缀），文件名与展示名分离。
- **STT 的三种输入形态统一到一个临时文件路径**：`_get_audio_path_from_request` 把"表单上传的文件对象"和"JSON 里的 base64（含或不含 `data:audio/webm;base64,` 前缀）"两种输入统一转换成一个磁盘临时文件路径，让下游的 `model.transcribe(path, ...)` 不用关心来源差异——这是一个典型的"边界适配层收口，核心逻辑保持单一路径"的设计。
- **WebSocket 与 Flask-sockets 双实现共存**：`stt.py` 里同时保留了 `register_stt_ws(sock)`（Flask-Sock 风格，同步 `ws.receive()`/`ws.send()`）和 `register_stt_ws_fastapi(websocket)`（FastAPI 原生 `await websocket.receive_text()`）两套逻辑完全一致的实现。`app/factory.py::register_websocket_routes` 里实际挂载的只有 FastAPI 版本（`/api/ai/stt/live`），`register_stt_ws` 是 Flask 时代迁移后留下的历史代码，当前未被任何路由调用。
- **`image_gen.py` 的分辨率固定为 96×96**：`generate()` 中 `height=96, width=96` 硬编码，而非从请求参数读取——这明显是本地 MPS 推理速度/显存的临时妥协值（生产可用的 SDXL 出图通常至少 768×768 起），实际使用前需要评估是否要放开为可配置参数。
- **`image_gen_qwen.py` 未挂载路由**：文件头注释"仅保留代码，不对外提供 HTTP 接口"，`routes/ai.py` 中确认没有引用 `generate_qwen`，这是刻意保留的候选实现（Qwen-Image + Turbo LoRA 2 步出图，速度远快于 SDXL 的 25 步），当图像生成需求升级时可直接注册路由启用。

### 3.3 设计决策与取舍

1. **两套 Function Calling 循环不做抽象复用** —— `run_function_calling_chat` 和 `run_mcp_ppt_chat` 的循环结构几乎相同，理论上可以抽出一个通用的 `run_tool_calling_loop(tools, tool_dispatcher, ...)`。但两者在"最终答案的组织形式"上有差异（前者只返回纯文本，后者要保留 `reply_messages`/`steps` 两套面向前端展示的结构），过早抽象会引入一个需要兼容两种输出形态的参数化接口，复杂度不降反升。当前选择是可接受的重复（Rule of Three 尚未触发）。
2. **懒加载单例不加锁** —— `_get_model()`/`_get_pipeline()` 在并发首次请求下可能重复初始化模型（竞态），但代价仅是短暂的重复内存占用而非数据错误，考虑到这几个接口本身QPS 很低（工具类演示/低频调用场景），团队选择不引入 `threading.Lock` 或 `asyncio.Lock` 换取代码简洁性，是刻意的复杂度取舍而非疏漏。
3. **STT 用整段转录模拟流式，而非真流式解码** —— faster-whisper 本身不支持增量流式输出（Whisper 架构是 attention-based，需要看到完整音频窗口才能解码），`transcribe_stream`/WebSocket 的"流式"实际是"客户端分段上传 → 服务端每段整体转录 → 逐段推回"，是工程上对"流式体验"的近似而非真正的流式 ASR。这个取舍在文档头部注释里已注明，但接口命名（`transcribe_stream`、`/stt/live`）容易让调用方误判为真流式，需要在对接文档中明确说明。
4. **PPT 生成复用 `files.py` 的 manifest 而非独立存储** —— `_generate_pptx_save` 没有为 PPT 建单独的数据表或索引，而是直接写入通用文件管理模块 `files.py` 的 JSON manifest（`kb_id="ppt"` 作为分类标记），减少了新增存储层的成本，代价是 PPT 文件与知识库文档共用同一套下载/预览接口（`/ai/files/{file_id}/preview`），语义上略显混用，但避免了重复造轮子。

---

## 第四部分：应用场景与实战

### 4.1 核心使用场景

- **场景 A：客服/助手类多轮对话中插入实时数据查询**（`function_call.py`）——用户问"北京今天天气怎么样"，模型无法凭训练数据回答实时天气，必须触发工具调用。
- **场景 B：自然语言驱动的报表自动化**（`function_call_ppt.py`）——业务人员一句话（"基于本月销售数据做个汇报PPT"）驱动"查数据库 → 组织内容 → 生成可下载文件"的端到端链路，是 Function Calling 从"回答问题"升级到"完成任务"的典型例子。
- **场景 C：无障碍/多模态交互的语音入口与出口**——STT 承接语音输入转文字（配合聊天或 Text2SQL 等下游能力），TTS 把模型回答朗读出来，组合起来构成语音助手闭环。
- **场景 D：营销/内容团队的快速配图**——`image_gen.py`/`image_gen_qwen.py` 用一句 prompt 生成示意图或素材图，规避商用图库授权成本。

### 4.2 快速上手

**环境依赖**

```bash
# TTS
pip install edge-tts
# STT
pip install faster-whisper
# 图像生成（SDXL，需 Apple Silicon）
pip install diffusers torch transformers accelerate safetensors
# Function Calling
pip install dashscope requests
# PPT 生成
pip install python-pptx
```

必须的环境变量：`DASHSCOPE_API_KEY`（Function Calling 依赖的模型调用）、`AMAP_MAPS_API_KEY`（高德天气查询）；可选：`TTS_VOICE`（默认 `zh-CN-XiaoxiaoNeural`）、`STT_MODEL`/`STT_DEVICE`/`STT_COMPUTE_TYPE`（默认 `base`/`cpu`/`int8`）。

**示例一：Function Calling 天气查询**

```bash
curl -X POST http://localhost:3000/ai/function-calling/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "北京今天天气怎么样"}]}'
```

**示例二：文字转语音**

```bash
curl -X POST http://localhost:3000/ai/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，欢迎使用语音合成", "voice": "zh-CN-XiaoxiaoNeural"}' \
  --output speech.mp3
```

**示例三：语音转文字（本地音频文件）**

```bash
curl -X POST http://localhost:3000/ai/stt/transcribe \
  -F "file=@/path/to/audio.wav" \
  -F "language=zh"
```

**示例四：文生图**

```bash
curl -X POST http://localhost:3000/ai/image/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "一只坐在窗边的橘猫，水彩风格", "seed": 42}' \
  --output cat.png
```

### 4.3 常见问题排查

| 现象 | 可能原因 | 排查方向 |
|---|---|---|
| Function Calling 一直返回空文本 | `max_iterations` 耗尽仍未拿到无 tool_calls 的回复 | 检查模型是否反复要求调用同一工具（可能是工具返回结果模型无法理解），或临时调大 `max_iterations` |
| `/ai/tts` 返回 500 | `edge_tts` 服务端限流/网络问题（非官方接口，不稳定） | 重试；长期方案考虑切换 `mcp_tts`（DashScope）等有 SLA 的替代路径 |
| STT 转录结果为空或乱码 | 音频格式/采样率不兼容，或 `language` 误指定导致强制解码错误语言 | 确认音频后缀与实际编码一致；`language` 留空走自动检测 |
| 图像生成接口卡住不返回 | 首次请求触发懒加载（下载/加载大模型权重耗时数十秒到几分钟） | 首次调用前预热（应用启动时主动调用一次 `_get_pipeline()`），或前端加长超时时间 |
| PPT 生成后 `preview_url` 404 | `files.py` manifest 未持久化成功，或 `kb_id="ppt"` 的目录未创建 | 检查 `data/knowledge_base/ppt/` 目录写权限，确认 `_save_manifest` 无异常 |
| WebSocket `/api/ai/stt/live` 连不上 | 客户端未使用 FastAPI 原生 WS 协议，或误连了未挂载的 Flask-Sock 路径 | 确认连接地址为 `/api/ai/stt/live`（`app/factory.py` 中挂载），`stt.py::register_stt_ws` 是未启用的历史代码 |

---

## 第五部分：评估与展望

### 5.1 优势

- **模块边界清晰**：四类能力各自独立成文件，无隐式全局状态耦合，新增/替换某个工具不影响其他模块。
- **快速可用、零外部依赖成本**：Edge-TTS 免费免 key，faster-whisper 本地推理免云端调用费，适合原型验证和内部工具场景，起步成本几乎为零。
- **Function Calling 链式组合已验证可行**：`function_call_ppt.py` 证明了"工具 A 输出 → 模型总结 → 工具 B 输入"的编排模式在当前技术栈下是可靠的，为后续接入更多工具（数据分析、文档生成等）提供了可复制的范式。
- **输入格式兼容性好**：TTS/STT/图像生成接口统一支持 JSON 与表单两种请求方式，降低了前端对接成本。

### 5.2 局限与技术债务

- **STT 的"流式"是伪流式**，真实时语音转写（边说边出字）尚未实现，命名上容易误导调用方。
- **懒加载模型无并发保护**，高并发首次请求下存在重复初始化的竞态风险（虽然后果轻微）。
- **`image_gen.py` 分辨率硬编码为 96×96**，不具备生产可用的出图质量，需要评估放开为可配置参数并做算力评估。
- **Edge-TTS 是非官方逆向接口**，没有 SLA，一旦微软调整策略随时可能失效，是当前语音合成链路里最大的单点风险。
- **`stt.py::register_stt_ws`（Flask-Sock 风格）已成死代码**，与实际生效的 `register_stt_ws_fastapi` 逻辑重复，增加维护心智负担，建议后续清理或明确标注废弃。
- **Function Calling 循环没有工具执行超时/重试机制**：`get_weather_from_gaode` 等外部 API 调用仅有 `timeout=10`，一次网络失败即在当轮对话中失败，没有指数退避重试。
- **图像生成两条路线（SDXL / Qwen-Image）并存但只挂载了一条**，`image_gen_qwen.py` 处于"代码就绪、路由未启用"的半成品状态，长期搁置会有代码腐化风险。

### 5.3 演进建议

1. 为 TTS/STT 引入可配置的"主备切换"：Edge-TTS/faster-whisper 作为默认免费档，DashScope/Azure 等商用 API 作为可选高可用档，通过环境变量切换。
2. 给懒加载单例加最小粒度的初始化锁（`asyncio.Lock`），消除并发竞态，成本极低。
3. 评估将 `image_gen_qwen.py`（2 步出图、速度远快于 SDXL 25 步）转正为主路径，替换或并行于 `image_gen.py`。
4. 为 Function Calling 的外部工具调用增加统一的超时/重试/降级封装，避免单次网络抖动打断整轮对话。
5. 清理 `stt.py` 中未使用的 `register_stt_ws`（Flask-Sock 版本），或明确注释其保留原因（如兼容旧版部署）。

### 5.4 行业前沿

- **原生流式语音模型**（如 GPT-4o Realtime、通义千问实时语音）已经把 STT+LLM+TTS 三段式管线合并为端到端的语音到语音模型，延迟和自然度远超"转文字→推理→合成"的三段式架构，是当前 `stt.py`+`chat.py`+`tts.py` 组合方式的长期替代方向。
- **MCP（Model Context Protocol）** 正在成为工具调用的跨平台标准协议，`service/ai/mcp/` 目录已有雏形，未来 Function Calling 层可以逐步迁移到标准 MCP Server 形式，替代当前每个工具手写 JSON Schema 的方式。
- **图像生成的少步蒸馏模型**（如本项目已引入的 Turbo LoRA、以及 SDXL-Turbo、LCM 等）是行业趋势，2-4 步出图逐渐替代 20-50 步的传统扩散采样，`image_gen_qwen.py` 的技术选型方向是正确的，值得优先推进落地。

---

## 变更记录

- **2026-08-18 全量重写**：本次为完全重写而非增量修改。Text2SQL 相关内容（原文档中的 `text2sql.py` 部分）已拆分至独立文档 `13_text2sql_hitl.md`，本文档不再包含 Text2SQL 内容。重新梳理了 Function Calling / TTS / STT / 图像生成四类能力的架构、数据流、行业对比与技术债务。
