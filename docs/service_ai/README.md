# service/ai 技术文档总览

> 生成日期：2026-02-26  
> 对应代码：`service/ai/`（全模块）

---

## 模块架构图

```
                         ┌─────────────────────────────────────┐
                         │          HTTP 路由层（routes/ai.py）  │
                         └───────────────┬─────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
  ┌───────▼───────┐              ┌───────▼───────┐             ┌───────▼────────┐
  │  RAG 问答层    │              │  Agent 工作流   │             │   工具能力层    │
  │  (02_rag)     │              │  (04_agent)    │             │   (08_tools)   │
  └───────┬───────┘              └───────┬───────┘             └───────┬────────┘
          │                              │                              │
  ┌───────▼───────┐              ┌───────▼───────┐             ┌───────▼────────┐
  │  知识库管理层  │              │  A2A 协议层    │             │   MCP 工具集    │
  │ (03_knowledge)│              │   (05_a2a)    │             │   (06_mcp)     │
  └───────┬───────┘              └───────────────┘             └────────────────┘
          │
  ┌───────▼───────┐
  │   向量库层     │
  │ (01_vector_db)│
  └───────────────┘

  ┌───────────────┐  ┌────────────────┐  ┌──────────────────┐
  │   对话能力层   │  │   微调实验层    │  │ （独立：无 HTTP） │
  │   (07_chat)   │  │ (09_finetuning)│  │                  │
  └───────────────┘  └────────────────┘  └──────────────────┘
```

---

## 文档索引

| 序号 | 文档 | 对应代码 | 核心技术 |
|------|------|---------|---------|
| 01 | [向量库层](./01_vector_db.md) | `vector_db.py` | FAISS IndexIDMap、双存储同步、增量 Embedding |
| 02 | [RAG 检索增强生成层](./02_rag.md) | `rag.py`、`rag_enhance.py` | Query 改写（CASEA）、DashScope Rerank、流水线可观测性 |
| 03 | [知识库管理层](./03_knowledge_base.md) | `knowledge.py` | 多格式文档解析、分段策略、上传与向量化解耦 |
| 04 | [LangGraph 多 Agent 工作流层](./04_agent.md) | `agent/` | StateGraph、三种 Agent 范式、动态图可视化 |
| 05 | [A2A 协议层](./05_a2a.md) | `a2a/` | Google A2A Protocol、Task 生命周期、多 Agent 编排 |
| 06 | [MCP 工具集层](./06_mcp.md) | `mcp/` | Model Context Protocol、ChatPPT、SSL 重试 |
| 07 | [对话能力层](./07_chat.md) | `chat.py`、`ollama_chat.py` | 本地 Ollama、流式 SSE、OCR 视觉去重 |
| 08 | [工具能力层](./08_tools.md) | `function_call.py`、`text2sql.py`、`tts.py`、`stt.py`、`image_gen.py` | Function Calling、Text2SQL、Edge-TTS、Whisper、SDXL |
| 09 | [模型微调实验层](./09_finetuning.md) | `finetuning/` | LoRA/QLoRA、GRPO/R1、Unsloth、视觉微调 |

---

## 模块间依赖关系

```
知识库管理层 (03)
  └── 调用 → 向量库层 (01)

RAG 问答层 (02)
  ├── 调用 → 向量库层 (01)（检索）
  └── 调用 → 知识库管理层 (03)（寻址知识库 ID）

Agent 工作流层 (04)
  ├── 可调用 → RAG（知识库问答 Agent）
  └── 可调用 → 工具能力层 (08)（天气、SQL 等工具）

A2A 协议层 (05)
  └── 独立（通过 HTTP 调用外部 Agent 服务）

MCP 工具集层 (06)
  └── 独立（通过 MCP 协议调用外部工具服务）

对话能力层 (07)
  └── 独立（直接调 Ollama / DashScope API）

工具能力层 (08)
  └── 独立（各工具相互独立）

微调实验层 (09)
  └── 独立（离线训练脚本，不提供 HTTP 接口）
```

---

## 核心技术栈

| 类别 | 技术 |
|------|------|
| 向量检索 | FAISS（IndexFlatL2 + IndexIDMap）、NumPy |
| Embedding | DashScope `text-embedding-v4`（1024 维） |
| LLM 推理（云端） | DashScope（qwen-turbo、qwen3-rerank 等） |
| LLM 推理（本地） | Ollama（DeepSeek-R1、Qwen3-VL、自定义模型） |
| Agent 编排 | LangGraph `StateGraph`、LangChain |
| 跨 Agent 通信 | Google A2A Protocol、Pydantic 强类型 |
| 工具协议 | Anthropic MCP（qwen-agent Assistant 作为 Client） |
| 文档解析 | PyPDF2、python-docx、python-pptx、openpyxl、Tesseract OCR |
| 语音合成（TTS） | Edge-TTS（`zh-CN-XiaoxiaoNeural`） |
| 语音识别（STT） | faster-whisper（本地 base 模型） |
| 图像生成 | diffusers（SDXL）、DiffSynth（Qwen-Image）|
| 模型微调 | Unsloth + LoRA/QLoRA + TRL（SFTTrainer/GRPOTrainer） |
| Web 框架 | Flask + SSE（流式推送）+ WebSocket（实时语音） |
| 数据库 | MySQL（SQLAlchemy ORM） |

---

## 快速导航

**我想了解如何让 AI 回答业务知识库的问题** → [03 知识库管理层](./03_knowledge_base.md) + [02 RAG 层](./02_rag.md) + [01 向量库层](./01_vector_db.md)

**我想了解 LangGraph Agent 是怎么工作的** → [04 Agent 工作流层](./04_agent.md)

**我想了解多个 AI Agent 如何协作** → [05 A2A 协议层](./05_a2a.md)

**我想让 AI 调用外部工具（PPT/天气/地图）** → [06 MCP 工具集层](./06_mcp.md) + [08 工具能力层](./08_tools.md)

**我想了解如何在本地跑 LLM 对话** → [07 对话能力层](./07_chat.md)

**我想了解如何微调一个领域专用模型** → [09 微调实验层](./09_finetuning.md)
