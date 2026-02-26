# MCP 工具集层（Model Context Protocol）

> 文件：`service/ai/mcp/`（mcp_ppt.py、mcp_tts.py、mcp_stt.py、mcp_weather.py、mcp_gaode.py）  
> 生成日期：2026-02-26

---

## 第一部分：技术背景与演进

**问题背景**

LLM 本身是无状态的文字生成器，无法主动获取外部数据（实时天气、地图信息）、也无法操控外部系统（生成 PPT、调用语音服务）。让 LLM 调用外部工具的需求由来已久，但各家的实现方式不统一——OpenAI 有 Function Calling，LangChain 有 Tools，各种 Agent 框架各自为政。MCP（Model Context Protocol，Anthropic 2024 年发布）的目标是标准化"LLM 与外部工具的连接方式"，让工具实现一次、所有支持 MCP 的模型都能调用。

**核心概念**

- **MCP Server**：提供工具能力的服务端，通过标准协议暴露工具列表和调用接口（本项目中，ChatPPT 是一个 MCP Server）。
- **MCP Client**：调用 MCP Server 工具的客户端，通常嵌入在 Agent 框架内（本项目使用 `qwen-agent` 的 `Assistant` 作为 MCP Client）。
- **Tool Call 循环**：LLM 生成工具调用参数 → 框架执行工具 → 将结果反馈给 LLM → LLM 继续推理，直到生成最终文本答案。

**演进脉络**

| 阶段 | 方案 | 特点 |
|------|------|------|
| 早期 | 手动解析 LLM 输出触发工具 | 脆弱，依赖 Prompt 格式 |
| Function Calling（OpenAI 2023） | 结构化 JSON 工具调用，模型原生支持 | 可靠，但每家 API 不通用 |
| LangChain Tools | 统一工具接口，生态丰富 | 框架锁定，不跨 LLM |
| **MCP（Anthropic 2024）** | 标准化协议，工具独立于 LLM | 工具复用，Claude/Qwen/GPT 均可接入 |

**本模块的定位**

`mcp/` 目录是 MCP 工具集的集成层，将多个外部能力（PPT 生成、TTS、STT、天气、地图）封装为可被 `qwen-agent` 调用的工具。其中 `mcp_ppt.py` 是最复杂的，通过 MCP 协议接入 ChatPPT（YOO AI）的 PPT 生成能力，是项目中最典型的 MCP 实战案例。

---

## 第二部分：架构剖析

**mcp_ppt.py 核心流程**

```
用户输入：PPT 主题/需求
      │
      ▼ qwen_agent.agents.Assistant（qwen-turbo + ChatPPT MCP Server）
      │   配置：mcp_servers=[{url: "http://mcp.yoo-ai.com/mcp", token: YOO_API_KEY}]
      │
      ├─► LLM 推理：判断需要调用哪个 MCP 工具
      │
      ├─► 调用 ChatPPT MCP 工具（通过 MCP 协议）
      │     ├─ create_ppt_task：提交 PPT 生成任务，返回 ppt_id
      │     └─ query_ppt_status：轮询生成进度
      │
      └─► 生成完成：LLM 输出包含 ppt_id 的最终文本
      
后续可调用：
  query_ppt_status(ppt_id)    → {status: 1/2/3, progress}
  get_ppt_download_url(ppt_id) → {download_url}
```

**SSL 重试机制**

```python
_SSL_RETRY_COUNT = 3
_SSL_RETRY_DELAY = 1.5  # 每次重试延迟翻倍（1.5s, 3s, 4.5s）

def _collect_bot_run(bot, messages):
    for attempt in range(_SSL_RETRY_COUNT):
        try:
            steps = list(bot.run(messages))
            return steps, None
        except Exception as e:
            if _is_ssl_error(e) and attempt < _SSL_RETRY_COUNT - 1:
                time.sleep(_SSL_RETRY_DELAY * (attempt + 1))
                continue
            return [], e
```

专门针对 `ssl` / `eof occurred` / `unexpected_eof` 的 SSL 握手失败，自动重试，对抗 ChatPPT MCP 服务的偶发网络抖动。

**PPT 状态码**

```
PPT_STATUS_PENDING    = 0   待处理
PPT_STATUS_PROCESSING = 1   生成中
PPT_STATUS_SUCCESS    = 2   成功（可获取下载链接）
PPT_STATUS_FAILED     = 3   失败
```

PPT 生成是异步任务，前端需要轮询 `query_ppt_status(ppt_id)` 直到状态为 2 或 3。

**各工具文件功能**

| 文件 | 工具服务 | 实现方式 |
|------|----------|---------|
| `mcp_ppt.py` | PPT 生成（YOO AI ChatPPT） | qwen-agent + MCP 协议 |
| `mcp_tts.py` | 文字转语音 | DashScope TTS MCP 工具 |
| `mcp_stt.py` | 语音转文字 | DashScope STT MCP 工具 |
| `mcp_weather.py` | 实时天气 | 天气 API MCP 工具 |
| `mcp_gaode.py` | 地图/位置/路线 | 高德地图 API MCP 工具 |

**与行业标准方案对比**

| 维度 | MCP（本项目） | OpenAI Function Calling | LangChain Tools |
|------|-------------|------------------------|----------------|
| 工具复用性 | 高（跨 LLM 可用） | 低（仅 OpenAI 兼容 API） | 中（框架内复用） |
| 协议标准化 | 是（Anthropic MCP Spec） | 非标准（私有） | 非标准（框架私有） |
| 异步工具支持 | 是（Task 生命周期） | 否（同步返回） | 有限 |
| 工具发现 | 是（MCP Server 暴露工具列表） | 否（手动定义） | 否（手动注册） |
| **选型建议** | 工具需跨模型复用、外部 MCP Server | 纯 OpenAI 生态、简单工具调用 | LangChain 生态内 |

---

## 第三部分：代码实现深度解析

**核心函数清单（mcp_ppt.py）**

| 函数 | 作用 |
|------|------|
| `query_ppt_status(ppt_id)` | 直接调 YOO API 查询生成进度（绕过 LLM） |
| `get_ppt_download_url(ppt_id)` | 获取 PPT 下载链接（status=2 时有效） |
| `generate_ppt_with_mcp(topic, model, extra_prompt)` | 完整 MCP 调用流程，返回 `{ppt_id, steps, final_text, ...}` |
| `_collect_bot_run(bot, messages)` | 带 SSL 重试的 `bot.run()` 包装器 |
| `_is_ssl_error(e)` | 判断是否为 SSL 握手失败 |

**`generate_ppt_with_mcp` 核心实现**

```python
def generate_ppt_with_mcp(topic, model="qwen-turbo", extra_prompt=""):
    bot = Assistant(
        llm={"model": model, "api_key": dashscope.api_key},
        function_list=[{
            "mcpServers": {
                "ChatPPT": {
                    "url": DEFAULT_CHATPPT_MCP_URL,
                    "headers": {"Authorization": f"Bearer {YOO_API_KEY}"}
                }
            }
        }],
    )
    messages = [{"role": "user", "content": f"请生成一份关于「{topic}」的PPT {extra_prompt}"}]
    steps, err = _collect_bot_run(bot, messages)
    # 从 steps 中提取 ppt_id（LLM 输出的文本中包含 ppt_id）
    ppt_id = _extract_ppt_id_from_steps(steps)
    return {"ppt_id": ppt_id, "steps": steps, "final_text": _get_final_text(steps)}
```

**设计决策与取舍**

**决策 1：`query_ppt_status` 绕过 LLM 直接调 API**  
原因：PPT 生成是异步任务，前端轮询状态时不需要经过 LLM 推理，直接调 REST API 更快（省去 LLM 调用的 1-3s 延迟）。MCP 工具用于触发生成，状态查询是纯数据操作。

**决策 2：SSL 错误专项重试**  
ChatPPT MCP Server 位于公网，偶发 SSL 握手超时（`EOF occurred in violation of protocol`）是常见瞬时错误。专项检测并重试，而不是对所有异常都重试（避免掩盖真实错误）。

**决策 3：`_collect_bot_run` 收集所有 steps**  
`bot.run()` 是生成器，每次 yield 一个推理/工具调用步骤。通过 `_collect_bot_run` 将所有步骤收集成列表，再从中提取 ppt_id 和最终文本，而不是只看最后一步，这让调试更方便（可以看到中间的工具调用过程）。

---

## 第四部分：应用场景与实战

**使用场景**

- 一键 PPT 生成：输入主题，AI 自动调用 ChatPPT 生成专业 PPT
- LLM + 工具的标准示范：`mcp_ppt.py` 是 MCP Protocol 在项目中的完整实现参考
- 多模态工具链：TTS + STT + 天气 + 地图，构建语音交互 + 位置感知的 AI 助手

**环境依赖**

```bash
pip install qwen-agent dashscope
export DASHSCOPE_API_KEY=sk-xxx
export YOO_API_KEY=xxx   # YOO AI PPT 生成 API Key
```

**代码示例**

```python
from service.ai.mcp.mcp_ppt import generate_ppt_with_mcp, query_ppt_status, get_ppt_download_url
import time

# 1. 触发 PPT 生成
result = generate_ppt_with_mcp("人工智能发展趋势", extra_prompt="包含数据图表，共10页")
ppt_id = result["ppt_id"]
print("PPT ID:", ppt_id)

# 2. 轮询状态
while True:
    status = query_ppt_status(ppt_id)
    state = status.get("data", {}).get("status")
    if state == 2:   # 成功
        break
    if state == 3:   # 失败
        raise Exception("PPT 生成失败")
    time.sleep(3)

# 3. 获取下载链接
dl = get_ppt_download_url(ppt_id)
print("下载地址:", dl["data"]["download_url"])
```

**常见问题**

- **SSL 错误频繁**：网络环境较差时重试 3 次后仍失败。可尝试配置代理或检查 `http://mcp.yoo-ai.com/mcp` 的可达性。
- **`ppt_id` 提取失败**：LLM 输出格式不固定，`_extract_ppt_id_from_steps` 的正则可能需要根据实际输出调整。
- **`YOO_API_KEY` 未设置**：`_yoo_headers()` 会返回 `{"Authorization": "Bearer None"}`，API 调用会返回 401。确保环境变量已配置。

---

## 第五部分：优缺点评估与未来展望

**优势**

- MCP 协议标准化，工具实现一次可复用于任何支持 MCP 的 LLM
- SSL 重试机制提升对不稳定网络的容错能力
- 直接调 API 查询 PPT 状态，绕过 LLM 推理，响应更快
- 模块化工具集（PPT/TTS/STT/天气/地图），互相独立，按需引入

**已知局限**

- `ppt_id` 从 LLM 输出文本中提取，依赖正则匹配，容易因 LLM 输出格式变化而失效
- PPT 生成完全依赖外部 YOO AI 服务，无降级方案
- MCP 连接每次重新建立（无连接复用），高并发场景性能有损

**演进建议**

- 短期：为 `_extract_ppt_id_from_steps` 增加多种提取策略（正则 + JSON 解析 + 关键词匹配）
- 中期：将 `ppt_id` 写入数据库，支持用户查询历史生成记录和状态
- 长期：实现 MCP Server 连接池，复用 WebSocket/SSE 连接

**行业前沿**

- **MCP 生态爆发**：Claude Desktop、Cursor、Zed 等工具已原生支持 MCP，工具市场（如 mcp.run）涌现数百个第三方 MCP Server
- **MCP + A2A 互补**：MCP 解决"Agent 调工具"，A2A 解决"Agent 调 Agent"，两个协议共同构成 Agentic AI 的通信基础设施
- **本地 MCP Server**：越来越多工具提供本地运行的 MCP Server（如 Playwright MCP 控制浏览器），无需外部 API，数据不出本机
