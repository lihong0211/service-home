# 工具能力层（Function Call / Text2SQL / TTS / STT / 图像生成）

> 文件：`service/ai/function_call.py`、`function_call_ppt.py`、`text2sql.py`、`tts.py`、`stt.py`、`image_gen.py`、`image_gen_qwen.py`、`video_undstanding.py`  
> 生成日期：2026-02-26

---

## 第一部分：技术背景与演进

**问题背景**

LLM 本质上是文本生成器，但实际业务需要它能"做事"——查天气、生成图片、识别语音、合成语音、查数据库。这些能力都需要调用外部系统，而 LLM 需要知道什么时候调、调哪个、传什么参数。工具能力层封装了多种独立的 AI 工具，并将它们以 Flask 接口的形式提供给上层路由。

**核心概念**

- **Function Calling**：LLM 在推理中自主决定调用哪个工具及参数，而不是每次都生成最终文字。多轮交互直到 LLM 决定"不需要再调工具了，给出最终答案"。
- **Text2SQL**：将自然语言问题（"销售额最高的三个产品是什么"）翻译成 SQL 查询，并在数据库上执行，是 LLM + 数据库的直接集成。
- **TTS/STT**：语音合成（文字→音频）和语音识别（音频→文字），是语音交互的两个核心能力。

**演进脉络（以 Function Calling 为例）**

| 阶段 | 方案 | 特点 |
|------|------|------|
| 早期 | Prompt 工程（"如果需要天气，输出 [WEATHER:城市]"） | 脆弱，格式不稳定 |
| ReAct（2022） | 推理+行动交替，LLM 自主循环 | 可靠性提升，但结构化程度低 |
| **Function Calling（2023+）** | LLM 输出结构化 JSON 参数，框架执行工具 | 可靠、标准化，OpenAI/DashScope 均支持 |
| Tool Use（Claude）/ MCP | 更丰富的工具协议 | 支持异步工具、跨平台互操作 |

---

## 第二部分：架构剖析

**各工具模块概览**

| 模块 | 核心能力 | 关键依赖 |
|------|---------|---------|
| `function_call.py` | 天气查询（多轮 Function Calling） | DashScope + 高德 API |
| `function_call_ppt.py` | PPT 生成（Function Calling 触发） | DashScope + YOO API |
| `text2sql.py` | 自然语言 → SQL → 执行 | LangChain SQLAgent + DashScope |
| `tts.py` | 文字 → MP3 音频 | Edge-TTS（本地，免费） |
| `stt.py` | 音频 → 文字 + 时间戳 | faster-whisper（本地） |
| `image_gen.py` | 文字 → 图片（SDXL） | diffusers + Apple MPS |
| `image_gen_qwen.py` | 文字 → 图片（Qwen-Image） | DiffSynth + LoRA |
| `video_undstanding.py` | 视频理解分析 | DashScope 多模态 |

**Function Calling 多轮循环（function_call.py）**

```
用户问题："北京今天天气如何"
      │
      ▼ 第 1 轮：LLM 推理
        → finish_reason = "tool_calls"
        → tool_call: get_current_weather({location: "北京"})
      │
      ▼ 执行工具：get_weather_from_gaode("北京")
        → {"temperature": "15°C", "weather": "晴", ...}
      │
      ▼ 第 2 轮：LLM 推理（携带工具结果）
        → finish_reason = "stop"
        → "北京今天天气晴朗，气温 15°C，适合出行"
      │
输出：最终文本答案
```

**Text2SQL 安全机制**

```python
def _is_read_only_sql(sql: str) -> bool:
    s = re.sub(r"\s+", " ", sql.strip()).upper().lstrip("; ")
    # 白名单：只允许 SELECT 开头
    for keyword in ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "REPLACE"):
        if keyword in s and not s.startswith("SELECT"):
            return False
    return s.startswith("SELECT")
```

LangChain SQLAgent 生成 SQL 后，执行前强制过白名单，禁止任何写操作，防止 LLM "越权"修改数据。

**TTS 技术选型（tts.py）**

```python
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
# Edge-TTS：微软 Edge 浏览器 TTS 引擎，免 API Key，中文质量优秀
# 异步生成 MP3 → tempfile → Flask send_file
```

**STT 架构（stt.py）**

```python
STT_CONFIG = {
    "model_size": "base",      # tiny/base/small/medium/large-v3
    "device": "cpu",
    "compute_type": "int8",    # CPU 推理用 int8 量化
    "vad_filter": True,        # 语音活动检测，过滤静音段
    "beam_size": 5,
}
# 懒加载：_whisper_model = None，首次请求时加载模型
```

支持三种输入方式：multipart 文件上传、base64 JSON、WebSocket 实时流（`/ai/stt/live`）。

**与行业标准方案对比（TTS/STT）**

| 维度 | 本地方案（Edge-TTS + Whisper） | 云端方案（DashScope TTS/STT） | OpenAI Whisper API |
|------|-------------------------------|------------------------------|-------------------|
| 费用 | 免费 | 按量计费 | 按分钟计费 |
| 延迟 | TTS<1s，STT 取决于硬件 | TTS<500ms，STT<2s（云端） | STT<5s |
| 离线可用 | 是 | 否 | 否 |
| 语言支持 | TTS 400+语言，STT 99+语言 | 中英为主 | 99+语言 |
| 质量 | 好（Edge-TTS 接近自然人声） | 优秀 | 优秀 |
| **选型建议** | 成本敏感、数据隐私、离线需求 | 低延迟要求、企业级 SLA | 多语言、高精度 |

---

## 第三部分：代码实现深度解析

**核心设计决策**

**决策 1：TTS 使用 Edge-TTS 而非云端 API**  
Edge-TTS 是微软 Edge 浏览器内置的 TTS 引擎，通过逆向工程的非官方 Python 库调用，免费且中文质量优秀（`zh-CN-XiaoxiaoNeural`）。代价是非官方接口，微软可能随时更改协议；企业生产环境建议使用官方 DashScope TTS。

**决策 2：STT 懒加载 Whisper 模型**  
```python
_whisper_model = None
def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(...)
    return _whisper_model
```
`faster-whisper` 的 `base` 模型约 150MB，加载需 2-5 秒，不应在进程启动时就加载。懒加载保证启动速度，且只加载一次后全局复用。

**决策 3：Text2SQL 只允许 SELECT**  
LLM 生成 SQL 的能力强大，但也意味着如果不加限制，LLM 可能生成 `DROP TABLE`、`DELETE FROM` 等危险语句。`_is_read_only_sql` 白名单是一道必不可少的安全防线，且代码简单、易于审计。  
代价：部分合理的写操作（如统计汇总写入临时表）无法支持，但对"自然语言查询"场景这是合理限制。

**决策 4：图像生成固定使用 Apple MPS（image_gen.py）**  
```python
DEVICE = torch.device("mps")   # Apple Silicon GPU
```
代码写死了 MPS 设备，这意味着在 Linux/Windows（无 MPS）机器上会报错。这是开发机（Mac）的本地实验代码，不适合直接用于跨平台生产部署。

**决策 5：STT 支持 WebSocket 实时流**  
```python
@sock.route("/ai/stt/live")
def live(ws):
    while True:
        msg = ws.receive()   # 持续接收音频 chunk
        # 每收到一个 chunk 就转录并回传文字
        ws.send(json.dumps({"text": text}))
```
WebSocket 版允许客户端持续发送音频 chunk，服务端逐段识别并回传文字，实现"边说边出字"的实时体验，比 HTTP 接口更适合长语音输入场景。

---

## 第四部分：应用场景与实战

**使用场景**

- **Function Calling**：用户问天气/PPT 生成等需要外部数据的问题，LLM 自动调用工具后给出完整答案
- **Text2SQL**：业务人员用自然语言查询 AI 数据库，无需写 SQL
- **TTS**：将 AI 回答转为语音播报，适合语音助手、无障碍功能
- **STT**：用户语音输入，实时/离线转录为文字，适合语音交互界面
- **图像生成**：本地 Stable Diffusion 生成图片，无需调用 Midjourney 等云服务

**环境依赖**

```bash
# TTS
pip install edge-tts

# STT
pip install faster-whisper

# 图像生成（SDXL，Mac）
pip install torch diffusers transformers
# 需要 Apple Silicon Mac（MPS 设备）

# Qwen 图像生成
pip install diffsynth-engine

# Text2SQL
pip install langchain langchain-community langchain-openai sqlalchemy pymysql
export DASHSCOPE_API_KEY=sk-xxx
export AMAP_MAPS_API_KEY=xxx  # 高德天气
```

**代码示例**

```python
# TTS：调 HTTP 接口，返回 MP3
import requests
resp = requests.post("http://localhost:5000/ai/tts/speech", json={"text": "你好，欢迎使用AI助手"})
with open("output.mp3", "wb") as f:
    f.write(resp.content)

# STT：上传音频文件
resp = requests.post("http://localhost:5000/ai/stt/transcribe",
    files={"file": open("audio.wav", "rb")})
print(resp.json()["text"])     # 识别出的文字
print(resp.json()["segments"]) # [{start, end, text}, ...] 时间戳
```

**常见问题**

- **Edge-TTS 返回空或报错**：Edge-TTS 依赖微软服务器，网络不稳定时会失败；可重试或切换到 DashScope TTS。
- **Whisper 模型加载慢**：`base` 模型首次加载约 3-5 秒，属正常；后续请求复用模型实例，无此延迟。
- **`image_gen.py` 在 Linux 报错**：`DEVICE = torch.device("mps")` 写死 Apple MPS，需改为 `"cuda"` 或 `"cpu"`。

---

## 第五部分：优缺点评估与未来展望

**优势**

- TTS/STT 完全本地化，无 API 费用，数据不外传
- Text2SQL 写操作白名单安全可靠，适合对数据安全有要求的场景
- Function Calling 多轮循环处理复杂查询（如需要多次工具调用的问题）
- STT 支持 WebSocket 实时流，体验远超批量 HTTP 上传

**已知局限**

- `image_gen.py` 硬编码 MPS 设备，不可跨平台
- Edge-TTS 非官方库，稳定性无保证
- Text2SQL 只支持 SELECT，业务覆盖范围有限
- 图像生成质量（`num_inference_steps=25, 96x96`）明显偏低，属实验参数

**演进建议**

- 短期：`image_gen.py` 改为设备自动检测（`cuda` > `mps` > `cpu`），`image_gen_qwen.py` 同理；将图像尺寸和步数改为请求参数
- 中期：TTS 增加 DashScope 官方 TTS 作为备选，Edge-TTS 不可用时自动降级
- 长期：Text2SQL 扩展支持有限的写操作（INSERT、UPDATE with WHERE 条件限制），满足数据录入场景

**行业前沿**

- **实时 STT**：OpenAI Realtime API（2024）支持毫秒级音频流转录 + 实时 LLM 响应，是语音 Agent 的关键基础设施
- **端到端语音 LLM**：GPT-4o、Gemini 1.5 Pro 支持直接处理音频输入输出，不再需要 STT → LLM → TTS 三步管道，延迟从秒级降到毫秒级
- **视频理解**：多模态模型（Gemini 1.5 Pro、Qwen-VL-Max）支持对视频帧进行理解分析，`video_undstanding.py` 所用的 DashScope 方案是当前实用选择
