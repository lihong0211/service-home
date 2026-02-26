# 对话能力层（Chat）

> 文件：`service/ai/chat.py`、`service/ai/ollama_chat.py`  
> 生成日期：2026-02-26

---

## 第一部分：技术背景与演进

**问题背景**

对话是最基础的 AI 能力，但"调用一次 LLM API"和"生产可用的对话服务"之间存在大量工程细节：流式输出（边生成边返回）、多模态输入（图片/音频）、OCR 后处理（去重复）、本地离线推理（不依赖外部 API）。本模块将这两类对话能力（云端 API 和本地 Ollama）封装为统一的 Flask 接口，处理所有工程细节。

**核心概念**

- **SSE（Server-Sent Events）**：服务器推送协议，让服务器向客户端持续推送文本数据，是流式 LLM 输出的标准实现方式。每条消息格式 `data: {...}\n\n`。
- **Ollama**：本地 LLM 推理服务，支持 Llama、Mistral、DeepSeek 等开源模型，提供与 OpenAI 兼容的 HTTP API。
- **Vision OCR**：让视觉模型识别图片中的文字，本项目专门针对 Ollama 视觉模型的流式输出做了去重处理。

**演进脉络**

| 阶段 | 方案 | 特点 |
|------|------|------|
| 早期 | 同步调用，等待完整回复 | 用户体验差，大响应需等待数十秒 |
| 流式输出（SSE） | 边生成边推送，首 token 延迟低 | 用户体验大幅改善，成为业界标准 |
| 多模态输入 | 图片/音频 base64 嵌入消息 | 支持 Vision、OCR、语音理解 |
| 本地推理（Ollama） | 开源模型本地部署 | 无 API 费用、数据不外传、支持定制模型 |

**本模块的定位**

`chat.py` 对接云端 DashScope（OpenAI 兼容接口），`ollama_chat.py` 对接本地 Ollama 服务。两者提供独立的 Flask API，前端可根据需求选择调用。Ollama 方案特别针对 OCR 场景做了后处理优化，是本地化 AI 能力的代表实现。

---

## 第二部分：架构剖析

**ollama_chat.py 核心流程**

```
POST /ai/ollama/chat（纯文本对话）
      │
      ├─ stream=false → _sync_chat() → requests.post Ollama API → JSON 响应
      └─ stream=true  → _stream_chat() → SSE 流式推送

POST /ai/ollama/ocr（OCR 识图）
      │  固定使用 OCR_MODEL，图片 base64 → Ollama
      │  stream=false → _sync_chat(is_ocr=True, is_vision=True)
      └─ stream=true  → _stream_chat(is_ocr=True, is_vision=True)
      
OCR 后处理流水线：
  原始输出 → OCR_STRIP_PREFIX（去LaTeX/模板前缀）→ _dedupe_vision_content（去重复行）
```

**流式视觉去重算法**

```python
# 视觉模型流式输出时容易出现行级重复，如：
# "这是标题\n这是标题\n这是标题\n第一段内容..."
# 去重策略：累积全文按行去重，只下发新增的非重复部分

accumulated = ""
prev_deduped = ""
for chunk in stream:
    content = chunk.get("content", "")
    accumulated += content
    deduped = _dedupe_vision_content(accumulated)   # 行级去重
    new_content = deduped[len(prev_deduped):]        # 只取新增部分
    prev_deduped = deduped
    yield new_content   # 推送给前端
```

`_collapse_repeated_phrase` 进一步处理行内重复短语（如 `"ABABAB"` → `"AB"`），处理视觉模型流式输出的两层重复。

**三个模型常量**

```python
DEFAULT_MODEL = "my-deepseek-r1-1.5"  # 自定义推理模型（R1 风格）
OCR_MODEL     = "deepseek-ocr:latest"  # 专用 OCR 模型
VL_MODEL      = "qwen3-vl:2b"         # 视觉语言模型
```

**与行业标准方案对比**

| 维度 | Ollama（本项目） | OpenAI API（云端） | vLLM（本地） |
|------|-----------------|-------------------|-------------|
| 部署方式 | 本地进程 | 云端 SaaS | 本地 GPU 服务 |
| API 兼容性 | OpenAI 兼容 | 原生 | OpenAI 兼容 |
| 模型选择 | Ollama Hub（100+ 模型） | GPT 系列 | HuggingFace 模型 |
| 吞吐性能 | 低（CPU/单 GPU） | 高（大规模集群） | 高（多 GPU 并行） |
| 数据隐私 | 完全本地 | 数据上传云端 | 完全本地 |
| **选型建议** | 本地开发、隐私敏感、定制模型 | 生产高并发、最新模型能力 | 生产本地、GPU 集群 |

---

## 第三部分：代码实现深度解析

**核心函数清单**

| 函数 | 文件 | 作用 |
|------|------|------|
| `chat()` | ollama_chat.py | 纯文本对话接口 |
| `ocr_chat()` | ollama_chat.py | 专用 OCR 识图接口 |
| `_sync_chat(model, messages, options, is_ocr, is_vision)` | ollama_chat.py | 同步调用 Ollama，含 OCR 后处理 |
| `_stream_chat(model, messages, options, is_ocr, is_vision)` | ollama_chat.py | 流式 SSE 调用，含视觉去重 |
| `_dedupe_vision_content(text)` | ollama_chat.py | 视觉输出行级去重 |
| `_collapse_repeated_phrase(s)` | ollama_chat.py | 行内重复短语去重 |

**设计决策与取舍**

**决策 1：OCR 专用接口而非通用视觉接口**  
原因：OCR 场景有特殊需求——固定输入格式（只有图片）、固定 Prompt（"识别图中文字"）、特殊后处理（去 LaTeX 前缀、去重复行）。将 OCR 独立为接口，避免通用接口被 OCR 的特殊逻辑污染。  
代价：两套接口略显冗余；可通过 `is_ocr` 参数统一，但目前接口层保持分离以便前端明确调用意图。

**决策 2：`keep_alive=0` for OCR**  
```python
return _sync_chat(OCR_MODEL, messages, options, keep_alive=0, is_ocr=True, is_vision=True)
```
OCR 请求完成后立即释放模型内存（`keep_alive=0`），因为 OCR 请求通常是低频的、一次性的，不需要模型常驻内存占用 GPU VRAM。普通对话不设置此参数，模型保持热加载状态。

**决策 3：流式去重算法的权衡**  
流式去重需要"回溯累积文本重新处理"，意味着每个 chunk 都要对累积全文做一次去重操作（时间复杂度随输出长度增加）。对于几千字的 OCR 输出，这个操作在毫秒级内完成，可接受；对于超长输出（万字以上）可能有性能问题。

---

## 第四部分：应用场景与实战

**使用场景**

- 本地 AI 对话：使用 DeepSeek-R1 等开源模型，数据不离开本机
- OCR 文字识别：上传图片，本地 OCR 模型识别文字，无需调用云端 OCR 服务
- 流式对话体验：前端实时展示 LLM 推理过程（类 ChatGPT 打字效果）

**环境依赖**

```bash
# 安装 Ollama
brew install ollama   # macOS

# 拉取模型
ollama pull deepseek-r1:1.5b    # 或其他自定义模型名称
ollama pull deepseek-ocr:latest
ollama pull qwen3-vl:2b

# 启动 Ollama 服务
ollama serve   # 默认端口 11434
```

**代码示例**

```python
import requests

# 流式对话（SSE）
resp = requests.post("http://localhost:5000/ai/ollama/chat", json={
    "messages": [{"role": "user", "content": "介绍一下深度学习"}],
    "stream": True
}, stream=True)
for line in resp.iter_lines():
    if line.startswith(b"data: ") and line != b"data: [DONE]":
        import json
        chunk = json.loads(line[6:])
        print(chunk.get("response", ""), end="", flush=True)

# OCR 识图
import base64
with open("invoice.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()
resp = requests.post("http://localhost:5000/ai/ollama/ocr", json={
    "images": [img_b64], "stream": False
})
print(resp.json()["message"]["content"])
```

**常见问题**

- **`ConnectionError: Ollama service not running`**：Ollama 未启动，执行 `ollama serve` 或 `brew services start ollama`。
- **模型名称 `my-deepseek-r1-1.5` 不存在**：这是通过 `Modelfile` 自定义的模型名，需要用 `ollama create` 创建；也可直接修改 `DEFAULT_MODEL` 为已有模型名。
- **OCR 结果有大量重复行**：`_dedupe_vision_content` 应该能处理，如仍出现重复，检查是否是同步接口（`stream=false`）——同步接口只做最终结果去重，而非流式去重。

---

## 第五部分：优缺点评估与未来展望

**优势**

- 本地 Ollama 方案数据完全不外传，适合隐私敏感场景
- OCR 专用后处理（去重/去前缀）针对视觉模型特点优化，输出质量显著改善
- `keep_alive=0` 精细控制模型内存占用，降低 OCR 场景资源浪费
- 同步/流式双模式满足不同前端需求

**已知局限**

- Ollama 单机 CPU 推理速度慢（相比 GPU 慢 10-50 倍），高并发场景吞吐不足
- 本地模型能力（1.5B-7B 参数）远弱于 GPT-4/Claude 3，复杂推理任务效果差
- `_collapse_repeated_phrase` 算法是 O(n²)，超长输出时性能有损

**演进建议**

- 短期：增加对 Ollama 连接失败的更友好错误提示（当前返回 503/504，可加 Ollama 安装指引）
- 中期：支持模型配置化（从数据库/配置文件读取模型名，而非代码常量），便于运营侧切换模型
- 长期：引入 vLLM 替换 Ollama，支持多 GPU 并行推理，提升吞吐量；或接入量化模型（GGUF/AWQ）降低显存需求

**行业前沿**

- **Ollama 多模态**：Ollama 不断扩展视觉模型支持（LLaVA、Moondream、Qwen-VL），本地多模态能力快速成熟
- **模型量化技术**：AWQ/GPTQ 4-bit 量化让 70B 参数模型在消费级 GPU 运行成为可能，本地部署成本大幅降低
- **推测解码（Speculative Decoding）**：用小模型草稿 + 大模型验证的方式提升推理速度 2-3 倍，Ollama 已部分支持
