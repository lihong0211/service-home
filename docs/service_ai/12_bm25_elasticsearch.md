# BM25（Elasticsearch）检索：MySQL 同步与 RAG Hybrid 接入

本文档描述本仓库的 **BM25 关键词检索**能力：如何把 MySQL 中的 `vector_db_document` 同步到 Elasticsearch，并在 RAG 检索中作为 “Dense 语义召回之外的字面兜底” 使用。

> 对应代码：`service/ai/bm25_es.py`、`service/ai/rag.py`、`routes/ai.py`

## 1. 为什么要加 BM25

Dense 向量检索擅长语义匹配，但在以下场景容易出现“看起来相关、细节不准”：

- **专有名词/人名/公司名**（如产品型号、基金代码）
- **编号/条款号/错误码**（如 `E11000`、`RFC-XXXX`）
- **数字敏感问题**（如“9.6% 还是 9.8%？”）
- **代码片段/配置项名**（大小写、下划线、短 token）

BM25 基于倒排索引，更擅长上述“字面精确命中”。因此推荐用法是：

- **Dense（主召回） + BM25（兜底召回） + Rerank（可选精排）**

## 2. 组件与数据来源

- **数据来源**：MySQL `vector_db_document`（见 `model/ai/vector_db_document.py`）
- **索引引擎**：Elasticsearch（BM25 默认相似度）
- **索引粒度**：以 “向量库文档项” 为单位（每条记录一篇/一段文本）

ES 文档 ID 规则：

- `_id = "{vector_db_id}::{doc_id}"`
- 避免不同向量库的 `doc_id` 冲突

## 3. 环境变量配置

启用 BM25/ES：

- `ES_BM25_ENABLED=1`
- `ES_URL=http://127.0.0.1:9200`

认证（任选一种）：

- `ES_API_KEY=...`
- 或 `ES_USERNAME=...` + `ES_PASSWORD=...`

索引与分析器（可选）：

- `ES_BM25_INDEX=service_home_vector_db_docs`
- `ES_BM25_TEXT_ANALYZER=standard`
- `ES_BM25_TEXT_SEARCH_ANALYZER=standard`
- `ES_BM25_SHARDS=1`
- `ES_BM25_REPLICAS=0`

> 中文语料建议使用更适合中文的 analyzer（例如 IK），否则分词效果可能较弱；该部分属于 ES 运维/插件范畴，仓库默认用 `standard`。

## 4. MySQL → ES 同步

### 4.1 创建/确保索引存在

`service/ai/bm25_es.py::ensure_index()` 会在首次同步/检索时自动创建索引（如果不存在）。

映射（核心字段）：

- `vector_db_id`（integer）：用于按库过滤
- `doc_id`（keyword）：业务文档项 id
- `category`（keyword）：分类过滤
- `text`（text）：BM25 检索字段
- `metadata`（object）：可选元信息过滤（`metadata.k = v`）

### 4.2 同步接口

接口：`POST /ai/bm25/sync`

Body：

- `db_id` / `vector_db_id` 或 `db_name`
- `batch_size`（可选，默认 200）

示例：

```bash
curl -sS -X POST "http://127.0.0.1:8000/ai/bm25/sync" \
  -H "Content-Type: application/json" \
  -d '{"db_name":"kb_9","batch_size":200}'
```

返回：

- `synced`：同步条数
- `took_s`：耗时

## 5. 在 RAG 里启用 BM25（Hybrid）

`service/ai/rag.py` 新增参数：

- `enable_bm25`（默认 true）

当 `enable_bm25=true`：

- 先按既有逻辑从 Qdrant 做 dense/hybrid/MMR 召回
- 再从 ES 做 BM25 关键词兜底召回
- 对 `doc_id` 去重合并候选（当前为简单融合；可后续升级为 RRF）
- （可选）继续走 DashScope `qwen3-rerank` 精排

示例：

```bash
curl -sS -X POST "http://127.0.0.1:8000/ai/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_name":"kb_9",
    "question":"E11000 是什么错误",
    "top_k":5,
    "enable_bm25": true,
    "enable_rerank": true
  }'
```

## 6. 常见问题

- **Q：为什么我启用 enable_bm25 但效果没变化？**
  - A：先确认 ES 已同步（`/ai/bm25/sync`），并且 `ES_BM25_ENABLED=1` + `ES_URL` 配置正确。

- **Q：中文效果不好怎么办？**
  - A：需要 ES 侧中文分词（例如 IK）。仓库默认 analyzer 为 `standard`，更偏英文场景。

- **Q：同步会不会影响主链路？**
  - A：ES 作为可选组件，RAG 里调用 BM25 是 best-effort，失败会自动跳过，不会阻断 dense 检索。

