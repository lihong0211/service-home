# Qdrant 向量化与召回策略（当前实现）

本文描述 `service/ai/vector_db_qdrant.py` 与 `service/ai/rag.py` 在本仓库中的 **向量化（indexing）** 与 **召回（retrieval）** 策略，包含默认行为、可调参数、以及 RAG 端的接入方式。

## 1. 总览

- **向量化模型**：DashScope OpenAI 兼容接口的 `text-embedding-v4`
- **向量维度**：`VECTOR_DB_DIMENSION`（默认 1024）
- **向量库**：Qdrant（每个 `db_name` 对应一个 collection）
- **距离度量**：COSINE（Qdrant 原生 `Distance.COSINE`）
- **召回策略（默认开启）**：
  - **Dense 向量召回**（Cosine 相似度）
  - **Hybrid 兜底**：在 dense 的基础上，额外尝试基于 `payload.text` 的关键词/full-text 匹配（可用则参与融合）
  - **MMR 多样性选择**：over-fetch 后用 MMR 选取最终 TopK，降低重复片段
  - **过滤**：支持 `category` 与 `metadata` 精确匹配过滤
  - **阈值**：可用 `score_threshold` 过滤低相关候选
  - **Rerank（可选）**：在 RAG 层对召回候选做语义精排（DashScope `qwen3-rerank`），提升最终上下文质量
  - **BM25（可选）**：对同一知识库的文本做关键词检索（Elasticsearch），用于“专有名词/编号/数字”类问题的兜底召回（见文末扩展）

> 说明：Hybrid/full-text 能力依赖 Qdrant 版本与 `qdrant-client` 对 “text payload index / MatchText” 的支持；不支持时会自动跳过，主流程仍为 dense + MMR。

## 2. 向量化（Indexing）策略

### 2.1 文档规范化

写入前会把输入 `documents` 规范化为：

- `id`：业务 doc_id（任意字符串）
- `text`：文本内容（必填）
- `category`：可选分类（常用于文件名/页码等）
- `metadata`：可选字典（用于来源、分段信息等）

### 2.2 Point ID 映射（稳定、可重建）

Qdrant 的 point id 只支持 UUID/uint64。实现中使用：

- `UUID5(namespace, doc_id)` 生成稳定 UUID
- payload 里保留真实 `doc_id`

这样可以做到：

- **同一 doc_id 多次 upsert 覆盖更新**
- **collection 重建后仍可保持 doc_id 的稳定映射**

### 2.3 Payload 结构

每个 point 的 payload 结构（核心字段）：

- `doc_id`: 业务 id（字符串）
- `text`: 原文
- `category`: 分类（可空）
- `metadata`: 元信息（可空）

### 2.4 Collection 与索引

collection 创建/校验：

- 如果已存在：校验向量维度与 `DIMENSION` 一致，否则根据 `QDRANT_RECREATE_COLLECTION_ON_DIM_MISMATCH` 决定是否重建
- 创建时使用 `Distance.COSINE`
- 默认会尝试创建 payload 索引：
  - `category`：keyword
  - `text`：text（full-text）

索引创建是 best-effort：失败会忽略，不影响写入与 dense 检索。

## 3. 召回（Retrieval）策略

召回入口：`service/ai/vector_db_qdrant.py::search_in_db(...)`

### 3.1 Dense 召回（基础）

1. 将 query 向量化为 embedding
2. 在 collection 上做 vector search
3. 返回 hits（默认包含 payload；MMR 时会额外请求向量）

**分数含义**：

- Qdrant COSINE：`score` 越大越相似
- 兼容旧字段：代码里同时输出 `distance = 1 - score`（仅用于 UI/兼容展示；不代表真实 L2 距离）

### 3.2 Over-fetch（候选集扩大）

当启用 MMR 或 Hybrid 时，会先拉取更多候选：

- `fetch_k = candidate_k`（未传则按 top_k 放大，且有上限保护）

目的：

- 让 MMR 有足够候选做多样性选择
- 让 Hybrid 候选有机会补齐 “必须字面匹配” 的内容

### 3.3 Hybrid 关键词兜底（默认开启）

在 dense 结果之外，额外尝试一次 “text 字段匹配” 的检索：

- 使用 `MatchText(text=query)`
- 仍然走 vector search + filter 的方式（用于与 dense 分数体系融合）

注意：

- 若服务端/客户端不支持 full-text/match-text，这一步会自动跳过（try/except 包裹）

## 3.8 术语速查（和本仓库实现对齐）

### Dense

- **Dense（稠密向量检索）**：把 query 与文档片段编码成向量，用余弦相似度检索语义相近内容。
- **本仓库对应实现**：`get_embedding()` + Qdrant `query_points/search`（`Distance.COSINE`）。
- **典型优势**：同义改写、语义匹配能力强。
- **典型风险**：对“数字/型号/专有名词/代码片段/条款号”这种 **字面精确** 的查询，可能出现“看起来相关、细节不准”。

### MMR

- **MMR（Maximal Marginal Relevance）**：在候选集里做“既相关又不重复”的选择，减少重复段落塞进上下文。
- **本仓库对应实现**：`search_in_db()` 中 over-fetch 后 `_mmr_select(...)`，用候选向量计算 cosine；拿不到向量会退化为原排序。

### mmr_lambda

- `**mmr_lambda`**：MMR 中“相关性 vs 多样性”的权重 \lambda\in[0,1]。
- **越大**：更偏“相关性优先”，可能更重复；**越小**：更偏“覆盖面”，可能引入噪声。

### 阈值过滤（score_threshold）

- **阈值过滤**：过滤掉低相关候选，避免把“硬凑”的段落塞进上下文。
- **本仓库对应实现**：将 `score_threshold` 传给 Qdrant，让服务端在返回前剔除。
- **注意**：在 COSINE 下 `score` 越大越好；阈值需要结合你的语料分布调参。

### text 字段匹配（Hybrid 兜底）

- **text 字段匹配**：对 payload 的 `text` 做全文匹配/包含匹配，用来兜底“必须字面命中”的场景。
- **本仓库对应实现**：`MatchText(text=query)` + filter（若 Qdrant/客户端不支持则跳过）。
- **它不是 BM25**：更多是“全文字段匹配兜底”，不具备 ES/Lucene 那种可调的倒排检索打分能力。

### Rerank（重排）

- **Rerank（重排）**：对召回候选做精排，把“真正最相关”的段落排到前面（尤其是 dense 相似度排序不稳定时）。
- **本仓库对应实现**：`service/ai/rag_enhance.py::rerank_documents()` 调用 DashScope `TextReRank`（默认模型 `qwen3-rerank`）。

### 3.4 融合与去重

融合后会做：

- **按 doc_id 去重**：同一 doc 可能在 dense 与 keyword 两路都命中
- **保留 score 更高的那条**
- 按 score 降序排序，形成候选集

### 3.5 MMR 多样性（默认开启）

当 `use_mmr=True` 且候选集大于 top_k 时：

- 使用 **MMR** 从候选中选择最终 top_k
- 目标：在保证相关性的前提下，最大化信息覆盖，减少重复片段

参数：

- `mmr_lambda`：\lambda \in [0,1]
  - 越大：更偏相关性
  - 越小：更偏多样性

实现细节：

- 会尽量使用 Qdrant 返回的候选向量计算 cosine
- 若拿不到候选向量，则退化为按 score 排序的前 top_k

### 3.6 过滤（category / metadata）

支持的过滤条件：

- `category`：
  - `None`：不限制
  - `""`：筛选 “空分类”
  - 其他字符串：精确匹配
- `metadata_filter`：`{"k":"v"}` → 匹配 payload 中 `metadata.k == v`

### 3.7 阈值过滤（score_threshold）

可传 `score_threshold`：

- 低于阈值的候选会被 Qdrant 端剔除
- 可用于减少“硬凑”的低相关段落进入上下文

## 4. RAG 接入与默认行为

RAG 入口：`service/ai/rag.py`

### 4.1 默认开启项

在 `rag_chat(...)` 与两条 API（`rag_ask_api` / `rag_search_api`）中，默认：

- `enable_hybrid = True`
- `enable_mmr = True`

如果要关闭，显式传：

- `enable_hybrid: false`
- `enable_mmr: false`

### 4.2 可用请求参数（POST body）

- `enable_query_rewrite`：是否启用 query 改写（默认 false）
- `enable_rerank`：是否启用 rerank（默认 false）
- `enable_hybrid`：hybrid 兜底（默认 true）
- `enable_mmr`：MMR 多样性（默认 true）
- `mmr_lambda`：0~1（默认 0.5）
- `score_threshold`：float（可选）
- `category`：string（可选）
- `metadata`：object（可选）

### 4.3 Rerank 与召回的配合

当 `enable_rerank=true`：

- RAG 会把 `retrieve_k` 放大（`top_k * 2`，上限 20），先召回更多，再由 rerank 选最终 top_k。
- 此时建议：
  - 保留 Hybrid + MMR（默认就会开启）
  - 适当调高 `retrieve_k` 或 `candidate_k`（若你要更激进的覆盖）

## 5. 建议的默认调参（经验值）

- **知识库 QA**：
  - `top_k`: 5~8
  - `mmr_lambda`: 0.5~0.7
  - `score_threshold`: 0.15~0.25（按语料而定，需观察空召回率）
- **代码/报错/编号类查询**：
  - 保持 `enable_hybrid=true`
  - `mmr_lambda` 可略升（0.65），避免多样性过强导致漏掉关键同类片段

## 6. 相关文件

- `service/ai/vector_db_qdrant.py`
  - collection / upsert / search / filters / hybrid / MMR
- `service/ai/rag.py`
  - RAG 调用 search，默认开启 hybrid/MMR，支持 API 参数透传
- `service/ai/bm25_es.py`
  - Elasticsearch 索引管理、MySQL→ES 同步、BM25 检索（可选组件）

