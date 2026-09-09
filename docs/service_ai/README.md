# service/ai 技术文档总览

> 生成日期：2026-02-26 | 全量重写：2026-08-18
> 对应代码：`service/ai/`（全模块）

---

## 模块架构图

```
                         ┌─────────────────────────────────────┐
                         │          HTTP 路由层（routes/ai.py）  │
                         └───────────────┬─────────────────────┘
                                         │
     ┌──────────────┬──────────────────┼──────────────────┬──────────────────┐
     │              │                  │                  │                  │
┌────▼────┐  ┌──────▼──────┐   ┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼────────┐
│ RAG 问答 │  │ Agent 工作流 │   │  工具能力层    │  │  A2A 协议层    │  │   MCP 工具集    │
│ (02_rag)│  │ (04_agent)  │   │  (08_tools)   │  │  (05_a2a)     │  │   (06_mcp)     │
└────┬────┘  └──────┬──────┘   └───────┬───────┘  └───────────────┘  └────────────────┘
     │              │                  │
┌────▼────┐  ┌──────▼──────┐    ┌──────▼──────┐
│ 知识库管理│  │ Text2SQL    │    │ 对话能力层   │
│ (03_kb) │  │ +HITL (13)  │    │ (07_chat)   │
└────┬────┘  └─────────────┘    └─────────────┘
     │
┌────▼────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
│ 向量库层 │  │ BM25/ES     │  │ RAG 评测     │  │ 可观测性/Trace   │
│(01_vdb) │  │ (12_bm25)   │  │ (14_ragas)  │  │  (15_trace)     │
└─────────┘  └─────────────┘  └──────────────┘  └─────────────────┘

  ┌────────────────┐
  │   微调实验层     │
  │ (09_finetuning)│
  └────────────────┘
```

---

## 文档索引

| 序号 | 文档 | 对应代码 | 核心技术 |
|------|------|---------|---------|
| 01 | [向量库层](./01_vector_db.md) | `vector_db_qdrant.py` | Qdrant + MySQL 双写、Hybrid/MMR 检索、增量 Embedding |
| 02 | [RAG 检索增强生成层](./02_rag.md) | `rag.py`、`rag_enhance.py` | Query 改写（CASEA）、DashScope Rerank、Dense+BM25 融合 |
| 03 | [知识库管理层](./03_knowledge_base.md) | `knowledge.py`、`files.py`、`knowledge_mineru.py` | 多格式文档解析、分段策略、MinerU 精确解析 |
| 04 | [LangGraph 多 Agent 工作流层](./04_agent.md) | `agent/` | StateGraph、四种 Agent 范式、医生多轮问诊 |
| 05 | [A2A 协议层](./05_a2a.md) | `a2a/` | Google A2A Protocol、Task 生命周期、Host Agent 动态路由 |
| 06 | [MCP 工具集层](./06_mcp.md) | `mcp/` | Model Context Protocol、ChatPPT、SSL 重试 |
| 07 | [对话能力层](./07_chat.md) | `chat.py`、`ollama_chat.py` | 本地 Ollama、流式 SSE、视觉输出去重 |
| 08 | [工具能力层](./08_tools.md) | `function_call(_ppt).py`、`tts.py`、`stt.py`、`image_gen(_qwen).py` | Function Calling、Edge-TTS、Whisper、SDXL/Qwen-Image |
| 09 | [模型微调实验层](./09_finetuning.md) | `finetuning/` | LoRA/QLoRA、Unsloth、领域模型对比评测 |
| 12 | [BM25（Elasticsearch）检索](./12_bm25_elasticsearch.md) | `bm25_es.py` | MySQL→ES 同步、BM25 关键词检索、Dense+BM25 兜底 |
| 13 | [Text2SQL + 人工审核（HITL）层](./13_text2sql_hitl.md) | `text2sql.py` | LangChain SQL Agent、写操作 HITL 审核、枚举值自动纠正 |
| 14 | [RAG 效果评测层](./14_rag_eval_ragas.md) | `rag_eval.py` | RAGAS 指标评测、中文测试集自动生成、批量回归评测 |
| 15 | [可观测性 + 回流机制层](./15_observability_trace.md) | `model/ai/agent_trace.py` | 全链路 Trace 记录、用户反馈回流、Bad Case 列表 |

**补充文档（非白皮书格式，独立维护）：**

| 文档 | 说明 |
|------|------|
| [10_frontend_api.md](./10_frontend_api.md) | 前端对接说明（Agent / LangGraph 接口约定） |
| [11_retrieval_strategy_qdrant.md](./11_retrieval_strategy_qdrant.md) | Qdrant 向量化与召回策略参数细节 |

---

## 模块间依赖关系

```
知识库管理层 (03)
  └── 调用 → 向量库层 (01)

RAG 问答层 (02)
  ├── 调用 → 向量库层 (01)（Dense 检索）
  ├── 调用 → BM25/ES (12)（关键词兜底检索）
  ├── 调用 → 知识库管理层 (03)（寻址知识库 ID）
  └── 写入 → 可观测性层 (15)（trace 记录）

Agent 工作流层 (04)
  ├── 可调用 → RAG（知识库问答 Agent）
  ├── 可调用 → 工具能力层 (08)（天气、地图等工具）
  └── 写入 → 可观测性层 (15)（trace 记录）

Text2SQL + HITL 层 (13)
  └── 独立（LangChain SQL Agent + 人工审核状态机）

RAG 评测层 (14)
  ├── 读取 → 知识库管理层 (03)（生成测试集原文）
  └── 调用 → RAG 问答层 (02)（批量回归评测）

可观测性层 (15)
  ├── 被 RAG (02)、Agent (04) 写入
  └── 支撑 → RAG 评测层 (14)（Bad Case 数据来源）

A2A 协议层 (05)
  └── 独立（通过 HTTP 调用外部 Agent 子进程）

MCP 工具集层 (06)
  └── 独立（通过 MCP 协议调用外部工具服务）

对话能力层 (07)
  └── 独立（直接调本地 Ollama）

工具能力层 (08)
  └── 独立（各工具相互独立）

BM25/ES 层 (12)
  └── 被 RAG 问答层 (02) 调用（hybrid 检索兜底）

微调实验层 (09)
  └── 独立（离线训练脚本，不提供 HTTP 接口）
```

---

## 核心技术栈

| 类别 | 技术 |
|------|------|
| 向量检索 | Qdrant（默认）、MySQL 双写 |
| 关键词检索（BM25，可选） | Elasticsearch（MySQL→ES 同步，BM25 检索兜底） |
| Embedding | DashScope `text-embedding-v4`（1024 维） |
| LLM 推理（云端） | DashScope（qwen-turbo、qwen3-rerank 等） |
| LLM 推理（本地） | Ollama（DeepSeek-R1、Qwen3-VL、自定义模型） |
| Agent 编排 | LangGraph `StateGraph`、LangChain |
| 跨 Agent 通信 | Google A2A Protocol、Pydantic 强类型 |
| 工具协议 | Anthropic MCP（qwen-agent Assistant 作为 Client） |
| 文档解析 | PyPDF2、python-docx、python-pptx、openpyxl、Tesseract OCR、MinerU |
| RAG 效果评测 | RAGAS（faithfulness / answer relevancy / context precision 等指标） |
| 可观测性 | 自建 `agent_trace` 表 + 用户反馈回流机制 |
| Text2SQL | LangChain SQL Agent + Human-in-the-loop 审核 |
| 语音合成（TTS） | Edge-TTS（`zh-CN-XiaoxiaoNeural`） |
| 语音识别（STT） | faster-whisper（本地 base 模型） |
| 图像生成 | diffusers（SDXL）、DiffSynth（Qwen-Image） |
| 模型微调 | Unsloth + LoRA/QLoRA + TRL（SFTTrainer） |
| Web 框架 | FastAPI + SSE（流式推送）+ WebSocket（实时语音） |
| 数据库 | MySQL（SQLAlchemy ORM） |

---

## 快速导航

**我想了解如何让 AI 回答业务知识库的问题** → [03 知识库管理层](./03_knowledge_base.md) + [02 RAG 层](./02_rag.md) + [01 向量库层](./01_vector_db.md) + [12 BM25/ES](./12_bm25_elasticsearch.md)

**我想了解 LangGraph Agent 是怎么工作的** → [04 Agent 工作流层](./04_agent.md)

**我想了解多个 AI Agent 如何协作** → [05 A2A 协议层](./05_a2a.md)

**我想让 AI 调用外部工具（PPT/天气/地图）** → [06 MCP 工具集层](./06_mcp.md) + [08 工具能力层](./08_tools.md)

**我想了解如何在本地跑 LLM 对话** → [07 对话能力层](./07_chat.md)

**我想了解自然语言查数据库怎么做、写操作怎么审核** → [13 Text2SQL + HITL 层](./13_text2sql_hitl.md)

**我想了解 RAG 效果怎么评测、回归怎么跑** → [14 RAG 效果评测层](./14_rag_eval_ragas.md)

**我想了解线上问答是怎么被记录和分析的** → [15 可观测性 + 回流机制层](./15_observability_trace.md)

**我想了解如何微调一个领域专用模型** → [09 微调实验层](./09_finetuning.md)

---

## 变更记录

**2026-08-18 全量重写**：因近期 RAG、Text2SQL、知识库、A2A、可观测性等模块代码大幅变动，对 01-09、12 共 10 篇既有文档做全量重写（非增量 patch）；新增 13（Text2SQL+HITL）、14（RAG 评测/RAGAS）、15（可观测性/Trace）三篇文档，覆盖此前未文档化的新增能力。10、11 为非白皮书格式的补充文档，本轮未改动。
