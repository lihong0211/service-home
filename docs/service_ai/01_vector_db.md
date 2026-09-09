# 向量库层（Vector DB）

> 文件：`service/ai/vector_db_qdrant.py`（1533 行）
> 关联表：`model/ai/vector_db.py`、`model/ai/vector_db_document.py`、`model/ai/vector_db_category.py`
> 生成日期：2026-08-18（全量重写，替代原 FAISS 版本文档）

---

## 第一部分：背景演进

### 问题背景

传统关系型数据库擅长精确匹配（`WHERE name = '张三'`），无法回答"和这段话意思相近的文档有哪些"这类语义查询。深度学习 Embedding 模型把任意文本压缩为固定维度的浮点向量后，语义相近的内容在向量空间中距离更近，检索问题转化为"在高维空间中找最近邻"。当候选集达到数万甚至数十亿条时，暴力计算全部余弦相似度不可行，需要专门的存储引擎和近似最近邻（ANN）索引结构来兼顾精度与延迟——这就是向量库存在的意义，也是当前 LLM RAG（检索增强生成）架构的基础设施。

### 核心概念

- **Embedding**：模型将文本映射到高维空间的浮点数组。本模块使用阿里 DashScope `text-embedding-v4`（`config/ai.py` 中 `DEFAULT_EMBEDDING_MODEL`），输出维度由 `VECTOR_DB_DIMENSION` 环境变量控制，默认 1024（`vector_db_qdrant.py:33`）。
- **Collection / Point**：Qdrant 的组织单元。一个业务向量库对应一个 collection，每条文档对应一个 point（向量 + payload）。本模块用 `_collection_name(db_name)` 做业务库名到 collection 名的映射（`vector_db_qdrant.py:59`）。
- **MMR（Maximal Marginal Relevance）与 Hybrid 检索**：在保证相关性的同时降低召回结果的重复度（MMR），并用关键词全文匹配兜底稠密向量漏检的场景（Hybrid）。两者是本模块检索链路的核心增强手段，详见第二、三部分。

### 演进脉络

| 时期 | 方案 | 特点 |
|------|------|------|
| 早期 | 暴力搜索（Brute Force） | O(n·d)，百万条以上延迟不可接受 |
| 2010s | LSH、IVF 等 ANN 算法 | 近似检索，速度大幅提升，精度有损 |
| 2017 | **FAISS**（Meta AI） | 库级方案，PQ 量化 + GPU 加速，无原生 CRUD/持久化 |
| 2019+ | **Qdrant / Milvus / Weaviate** 等系统化向量数据库 | 补齐 CRUD、payload 过滤、持久化、分布式、HNSW 索引 |
| 2021+ | **pgvector** | 向量能力作为 PostgreSQL 扩展，复用已有关系型数据库生态 |
| 2022+ | RAG 时代 | 向量库成为 LLM 的标配"外挂记忆"，Hybrid/Rerank/MMR 等检索工程成为标准动作 |
| **本项目 2026** | **由 FAISS + 本地文件迁移至 Qdrant** | 见下文"本模块定位" |

### 本模块在项目架构中的定位

模块文件头部注释明确写明：`Qdrant 向量库实现（替代 FAISS + 本地文件）`（`vector_db_qdrant.py:3`）。项目原先采用 FAISS `IndexFlatL2` + 本地 `.faiss`/`.npy`/`.json` 三件套的方案（见本文件旧版本，已被本次重写替换），现已迁移到独立部署的 Qdrant 服务，通过 `qdrant-client` 远程调用。模块对上层提供两条访问路径：

- **`routes/ai.py`** 通过 18 个 `_ai_route` 注册项（`routes/ai.py:295-311`）把 `list_api`／`create_api`／`search_api` 等 HTTP handler 暴露为 `/ai/vector-db/*` 系列接口，供前端管理界面直接调用。
- **`service/ai/knowledge.py`** 和 **`service/ai/rag.py`** 以 Python 函数调用方式复用 `create_vector_db`、`append_documents_batch`、`load_vector_db`、`search_in_db` 等核心函数，分别承担知识库文档入库和 RAG 问答检索职责——这是本模块最主要的内部消费方。

---

## 第二部分：架构剖析

### 整体分层

模块内部按职责可分为四层，全部在同一文件内，无独立子文件：

1. **Qdrant 客户端与 collection 管理层**（约 97-288 行）：`_get_qdrant_client()` 惰性单例、`_ensure_collection()` 建库/校验维度、`_ensure_payload_indexes()` 建 payload 索引、`_rename_qdrant_collection()` 改名迁移。
2. **向量读写原语层**（约 200-521 行）：`get_embedding()` 调用 Embedding API、`_upsert_points()`/`_delete_points_by_doc_ids()` 批量写入删除、`search_in_db()` 检索主入口（含 Hybrid + MMR 逻辑）。
3. **MySQL 管理数据层**（约 529-1125 行）：围绕 `VectorDb`/`VectorDbDocument`/`VectorDbCategory` 三张表的 CRUD，以及 `create_vector_db`/`sync_vector_db_from_disk`/`rebuild_vector_db_from_mysql` 等编排函数，负责把 Qdrant 写入与 MySQL 记录绑定在一起。
4. **HTTP 适配层**（约 1131-1533 行）：18 个 `xxx_api(request: Request)` 函数，做参数解析、类型转换、异常包装，调用第 2/3 层后返回 `{"code": 0, "msg": "ok", "data": ...}` 统一结构。

### 核心数据流

**写入路径（以 `create_api` 为例，`vector_db_qdrant.py:1136`）：**

1. 前端 POST `/ai/vector-db`，body 含 `name`/`documents`/`description`。
2. `create_api` 校验库名合法性（`DB_NAME_PATTERN`），拼出实际存储名 `vdb_{name}`，先在 MySQL `vector_db` 表插入一行拿到 `row_id`。
3. 调用 `create_vector_db(db_name, documents)`（`vector_db_qdrant.py:820`）：`_normalize_documents()` 清洗文档 → 若已存在同名 collection 先 `delete_collection` 再 `_ensure_collection()` 重建 → `_upsert_points()` 逐条调 `get_embedding()` 拿向量，按 64 条一批 `client.upsert()` 写入 Qdrant。
4. 写入成功后把规范化后的 `documents` 通过 `_save_documents_to_mysql()` 落回 `vector_db_document` 表，`_sync_categories_from_documents()` 同步分类到 `vector_db_category` 表。
5. Qdrant 写入失败时，`create_api` 捕获异常，回滚 MySQL 插入的行并清理 Qdrant collection（`vector_db_qdrant.py:1173-1176`），保证两侧不留半成品。

**检索路径（RAG 问答场景，`service/ai/rag.py` 调用 `search_in_db`）：**

1. `rag.py` 拿到用户问题后（可选先经 CASEA 查询改写），调用 `search_in_db(db_name, query, top_k=..., enable_hybrid=..., use_mmr=..., mmr_lambda=..., candidate_k=..., score_threshold=...)`。
2. `search_in_db()`（`vector_db_qdrant.py:290`）先 `get_embedding(query)` 拿查询向量，按 `category`/`metadata_filter` 构造 Qdrant `Filter`。
3. `_run_dense()` 执行稠密向量检索（`client.query_points` 或旧版 `client.search`），一次多召回 `fetch_k`（默认 `top_k` 的 5 倍，上限 200）条，为后续去重/MMR 留出候选池。
4. 若 `enable_hybrid=True`，额外用 `MatchText` 做一次全文关键词过滤检索，结果并入候选集，按 `doc_id` 去重取最高分。
5. 若 `use_mmr=True` 且候选数大于 `top_k`，调用 `_mmr_select()` 用 numpy 批量算相似度矩阵做多样性重排。
6. 组装 `[{doc, score, distance, rank}]` 返回；若设置了 `score_threshold` 导致结果为空，做一次不带阈值的单条兜底检索，避免 RAG 层拿到完全空的上下文。
7. `rag.py` 拿到 `results` 后拼 prompt 交给 LLM 生成回答（该部分逻辑不在本模块，见 `docs/service_ai/02_rag.md`）。

### 关键设计原则

- **业务 ID 与存储 ID 解耦**：Qdrant point id 只接受 uint64/UUID，业务 `doc_id` 可以是任意字符串。`_point_uuid()`（`vector_db_qdrant.py:48`）用 `uuid.uuid5(namespace, doc_id)` 做稳定单向映射，同一 `doc_id` 永远算出同一 UUID，真实 `doc_id` 另存入 payload 供检索结果回显，天然支持幂等 upsert。
- **双存储同步而非单一真相源**：Qdrant 是检索引擎（只认向量+payload），MySQL 三张表是管理视图（支持分页、分类、审计）。二者通过 `_save_documents_to_mysql()`/`_append_documents_to_mysql()`（写路径）和 `sync_vector_db_from_disk()`/`rebuild_vector_db_from_mysql()`（修复路径）保持一致，任一侧损坏都可以从另一侧重建。
- **惰性单例 + 惰性 import**：`_get_qdrant_client()`（`vector_db_qdrant.py:97`）用 `NameError` 探测的方式实现无需模块级声明的懒加载单例，`qdrant_client` 相关类型（`PointStruct`/`Filter`/`FieldCondition` 等）全部在函数体内 import，避免模块导入时就建立网络连接、也减少不使用 Qdrant 路径时的依赖负担。
- **过采样 + 后处理管线**：不直接向 Qdrant 要 `top_k` 条，而是先要 `fetch_k`（更大候选池）再做去重、MMR 重排、裁剪到 `top_k`，把"多样性/去重"这类 Qdrant 原生不支持的能力放在应用层实现。
- **失败静默降级**：payload 索引创建（`_ensure_payload_indexes`）、Hybrid 全文检索分支都包在 `try/except: pass` 中——不同版本 `qdrant-client`/Qdrant 服务端对 full-text index 的支持程度不一，缺失时自动退化为纯稠密检索，不阻断主流程。

### 与行业标准方案对比

| 维度 | 本项目实现（Qdrant + MySQL 双写） | Milvus | pgvector |
|------|------------------------------|--------|----------|
| 适用规模 | 千万级以内，单 Qdrant 实例 | 十亿至百亿级，原生分布式 | 百万级以内，受限于 PG 单机/主从架构 |
| 核心功能 | HNSW 近似检索 + payload 过滤 + 应用层 Hybrid/MMR | HNSW/IVF/DiskANN 多索引可选，原生标量+向量联合索引 | HNSW/IVFFlat，SQL 原生联表能力强 |
| 运维复杂度 | 中：需独立部署 Qdrant 服务，但无分布式协调开销 | 高：etcd/Pulsar/MinIO 等多组件集群 | 低：复用现有 PostgreSQL 运维体系 |
| 性能上限 | 单实例可达千万级向量、毫秒级检索；水平扩展需引入分片方案 | 数据面/计算面分离，理论无上限，适合超大规模 | 向量维度和数据量增大后性能明显下降于专用引擎 |
| **选型建议** | **当前项目定位**：中等规模多租户知识库，需要向量+结构化过滤混合查询，且已具备独立中间件运维能力 | 数据规模突破千万级或需要多租户强隔离、GPU 索引时迁移 | 已有 PostgreSQL 技术栈、规模小、希望减少中间件数量时优先选择 |

---

## 第三部分：代码实现深度解析

### 核心函数/类清单

**1. `get_embedding(text: str, max_retries: int = 3) -> list[float]`**（`vector_db_qdrant.py:78`）
调用 DashScope OpenAI 兼容接口 `embeddings.create(model=DEFAULT_EMBEDDING_MODEL, dimensions=DIMENSION, encoding_format="float")` 生成向量。失败时按 `2**attempt` 秒指数退避重试（1s/2s），耗尽重试后重新抛出最后一次异常。是全模块唯一的 Embedding 出口，写入和检索路径共用。

**2. `_ensure_collection(db_name: str) -> str`**（`vector_db_qdrant.py:130`）
若 collection 已存在，读取其向量维度并与当前 `DIMENSION` 比对；不一致时默认直接抛 `ValueError` 阻止误写，只有显式设置 `QDRANT_RECREATE_COLLECTION_ON_DIM_MISMATCH=1` 才会删库重建。不存在则用 `Distance.COSINE` 新建。无论哪个分支都会调用 `_ensure_payload_indexes()` 补建索引。

**3. `_upsert_points(db_name, docs, batch_size=64) -> int`**（`vector_db_qdrant.py:200`）
遍历文档逐条调 `get_embedding()`（若文档带 `embedding_text` 则优先用它而非 `text` 做向量化，用于父子切片场景），组装 `PointStruct(id=_point_uuid(doc_id), vector=vec, payload={doc_id, text, category, metadata?})`，每凑够 64 条调用一次 `client.upsert(wait=True)`，返回实际写入条数。

**4. `search_in_db(db_name, query, top_k=3, category=None, metadata_filter=None, enable_hybrid=None, use_mmr=None, mmr_lambda=0.5, candidate_k=None, score_threshold=None) -> list[dict]`**（`vector_db_qdrant.py:290`）
检索主入口，参数语义：
- `enable_hybrid`/`use_mmr` 缺省时分别取模块级开关 `DEFAULT_ENABLE_HYBRID`/`DEFAULT_USE_MMR`（环境变量 `VECTOR_DB_ENABLE_HYBRID_DEFAULT`/`VECTOR_DB_USE_MMR_DEFAULT`，默认均为 `True`）。
- `fetch_k` 计算：`candidate_k` 显式传入则用之，否则 `max(top_k, min(50, top_k*5))`，最终裁剪到 `[top_k, 200]` 区间，防止超大 `top_k` 拖垮 Qdrant 查询。
- 内部定义 `_run_dense`/`_payload_to_doc`/`_hit_score`/`_hit_vector`/`_mmr_select` 五个闭包函数分别负责稠密检索、payload 转文档结构、取分数、取候选向量（兼容多向量场景下的 dict 返回）、MMR 选择。
- 返回结构固定为 `[{"doc": {...}, "score": float, "distance": float, "rank": int}]`，`distance = max(0.0, 1.0 - score)`，供上层排序/展示。

**5. `_mmr_select(hits, limit) -> list`**（`vector_db_qdrant.py:390`，`search_in_db` 内部闭包）
用 numpy 一次性构造候选向量矩阵 `V_norm`、候选间相似度矩阵 `sim_matrix = V_norm @ V_norm.T`，以及候选与 query 的相关度向量 `rel`。逐步贪心选择：每轮取 `mmr_scores = λ·rel - (1-λ)·max_diversity_penalty`（`λ` 即 `mmr_lambda`）最大的候选加入结果集，直到选满 `limit` 条或候选耗尽。若某候选缺向量（`with_vectors=False` 时的关键词兜底命中），用其原始 `score` 作退化的相关度/多样性估计，不会因为部分候选缺向量而整体降级。

**6. `create_vector_db(db_name, documents=None) -> dict`**（`vector_db_qdrant.py:820`）
"创建或全量替换"语义：`documents` 非空时先删后建 collection 再批量 upsert（保证被删除文档的残留 point 不会遗留），为空时只确保 collection 存在。返回 `{count, path, collection, documents}`，`path` 固定格式为 `qdrant://{collection_name}`，仅作展示用途。

**7. `append_documents_batch(vector_db_id, documents) -> int`**（`vector_db_qdrant.py:896`）
增量追加：先 `load_vector_db()` 拿到该库已有的全部 `doc_id` 集合，跳过已存在的 `doc_id`，只对净新增文档调用 `_upsert_points()` 和 `_append_documents_to_mysql()`，避免重复 embedding 消耗 token。是 `knowledge.py` 知识库增量入库的核心依赖函数。

**8. `sync_vector_db_from_disk` / `rebuild_vector_db_from_mysql`**（`vector_db_qdrant.py:844` / `1097`）
两个互补的修复函数：前者读取历史遗留的磁盘 `metadata.json`（旧 FAISS 方案的产物，通过 `_storage_root()` 定位，兼容早期数据）写回 Qdrant + MySQL；后者以 MySQL `vector_db_document` 表为准全量重建 Qdrant collection。二者共同构成"任一侧数据受损都可从另一侧恢复"的容灾机制。

### 关键实现细节

- **数据结构选择**：`_mmr_select` 用 numpy 矩阵运算替代逐候选嵌套 Python 循环计算相似度（代码注释明确指出这一优化动机，`vector_db_qdrant.py:392`），把 O(候选数 × 已选数) 的 Python 级循环下沉为向量化的 `sim_matrix[np.ix_(...)]` 切片操作。
- **API 兼容性处理**：`qdrant-client >= 1.17` 移除了 `client.search`，统一使用 `client.query_points`；`_run_dense`/Hybrid 分支都用 `hasattr(client, "query_points")` 做运行时特性探测，同时兼容新旧客户端版本（`vector_db_qdrant.py:339`）。
- **边界处理**：`search_in_db` 对空查询直接返回 `[]`；对不存在的 collection（`_collection_physical_exists` 判假）直接返回 `[]` 而非抛异常，避免新建但尚未写入文档的知识库在检索时报错；`score_threshold` 把结果过滤为空时用不带阈值的单条兜底查询保底。
- **性能优化**：Hybrid 全文检索的候选量限制在 `min(fetch_k, 50)`，避免关键词匹配命中过多噪声候选拖慢去重/MMR 阶段；`_upsert_points` 用 64 条为一批写入，兼顾单次网络往返开销与内存占用。
- **安全边界**：`DB_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")`（`vector_db_qdrant.py:52`）在库名进入 collection 名拼接、磁盘路径拼接前统一校验，阻断路径穿越或非法 collection 名注入。

### 设计决策与取舍

**决策 1：从 FAISS + 本地文件迁移到 Qdrant 远程服务**
原因：FAISS 是纯计算库，没有原生 CRUD、payload 过滤、持久化和多进程共享能力；此前的实现依赖 `_db_cache` 进程内缓存和手工维护 `vectors.npy`，多 worker 部署下缓存一致性差，且无法做到"标量字段过滤走索引"。Qdrant 作为独立数据库服务原生解决这些问题。
代价：引入外部服务依赖（`QDRANT_URL` 不可用时全部向量能力失效），运维复杂度从"随应用部署"上升到"需要独立监控一个中间件"；网络往返延迟高于纯内存 FAISS 查询。

**决策 2：`doc_id` 与 Qdrant point id 解耦，用 UUID5 稳定映射**
原因：Qdrant 只接受 uint64/UUID 作为 point id，而业务侧 `doc_id` 需要支持任意字符串（如知识库分段 ID、外部系统主键）。
代价：多一层查表开销（每次写入/删除都要重算 UUID5），且 UUID5 是确定性哈希，理论上存在极低概率的碰撞风险（可忽略）；`doc_id` 真实值必须始终写入 payload，否则检索结果无法还原业务语义。

**决策 3：双写 MySQL 而非只用 Qdrant payload 做管理数据**
原因：管理界面需要分页列表、按分类统计、事务性的库改名/删除等能力，这些用 SQL 表达远比在 Qdrant 上模拟更自然；同时 MySQL 表可作为 Qdrant 数据损坏时的重建来源（`rebuild_vector_db_from_mysql`）。
代价：每次写操作都要保证两侧一致，存在中间态失败的风险（如 `_upsert_points` 成功但 `_save_documents_to_mysql` 失败）；目前没有分布式事务或补偿机制，仅在 `create_api` 的最外层做了"Qdrant 失败则回滚 MySQL"这一种方向的处理，反向失败（MySQL 写入失败但 Qdrant 已写入）未做显式回滚。

**决策 4：过采样 + 应用层 MMR/Hybrid 融合，而非依赖 Qdrant 原生能力**
原因：MMR 多样性重排和"稠密+关键词"融合排序不是 Qdrant 原生查询能力（尤其在项目使用的 Qdrant 版本下），且不同混合策略的调参（`mmr_lambda`、`fetch_k` 倍数）需要业务侧灵活控制。
代价：候选池过采样带来额外的网络传输和内存开销（`with_vectors=True` 时每个候选要传回完整 1024 维向量）；MMR 的 numpy 矩阵计算在候选数很大时（如 `fetch_k=200`）有 O(n²) 的相似度矩阵开销，虽然当前规模下可忽略。

**决策 5：Hybrid/payload 索引创建失败静默降级**
原因：`create_payload_index` 对 `text` 字段的 full-text 索引支持因 Qdrant 服务端版本、`qdrant-client` 版本组合而异，把这类环境差异当作硬性依赖会让检索功能在部分部署环境下完全不可用。
代价：功能"看似支持但实际未生效"时不会有任何报错或日志提示，只能靠检索效果异常间接发现（例如关键词精确匹配的文档没有被召回），排查成本较高。

---

## 第四部分：应用场景与实战

### 核心使用场景

- **知识库文档入库**：`service/ai/knowledge.py` 在文档上传/更新流程中调用 `create_vector_db()`（全量重建）和 `append_documents_batch()`（增量追加），完成 PDF/DOCX/PPTX 等多格式文档分块后的向量化落库。
- **RAG 问答检索**：`service/ai/rag.py` 在两处问答入口（`rag.py:177` 与 `rag.py:497`）调用 `search_in_db()`，配合可选的查询改写（CASEA）、Rerank、BM25 兜底，构成完整检索链路，详见 `docs/service_ai/11_retrieval_strategy_qdrant.md`。
- **管理端直连操作**：`routes/ai.py` 注册的 `/ai/vector-db/*` 系列接口供前端管理界面做库的增删改查、文档的单条编辑、分类管理，以及调试用的原始向量检索入口（`search_api` 固定 `use_mmr=False`，见下文"常见问题"）。

### 快速上手

**环境依赖：**

```bash
pip install qdrant-client openai numpy
export QDRANT_URL=http://localhost:6333        # 必填，默认已是此值
export QDRANT_API_KEY=xxx                       # 可选，托管 Qdrant 需要
export DASHSCOPE_API_KEY=sk-xxx                  # 必填，Embedding 依赖
export VECTOR_DB_DIMENSION=1024                  # 可选，需与所用 Embedding 模型输出维度一致
```

**代码示例（从本项目函数直接提炼）：**

```python
# 1. 创建向量库并写入文档（对应 create_vector_db，vector_db_qdrant.py:820）
from service.ai.vector_db_qdrant import create_vector_db, search_in_db, append_documents_batch

result = create_vector_db("vdb_my_kb", [
    {"id": "doc1", "text": "Qdrant 是一个开源向量检索引擎", "category": "AI工具"},
    {"id": "doc2", "text": "向量数据库用于语义相似度检索", "category": "概念"},
])
# result = {"count": 2, "path": "qdrant://vdb_vdb_my_kb", "collection": "...", "documents": [...]}

# 2. 检索（默认开启 Hybrid + MMR）
hits = search_in_db("vdb_my_kb", "有哪些向量检索工具", top_k=2)
# hits = [{"doc": {"id": "doc1", "text": "...", "category": "AI工具"}, "score": 0.83, "distance": 0.17, "rank": 1}, ...]

# 3. 增量追加（跳过已存在 doc_id，仅对新文档 embedding）
appended_count = append_documents_batch(vector_db_id=1, documents=[
    {"id": "doc3", "text": "Milvus 是分布式向量数据库", "category": "AI工具"},
])
```

**HTTP 调用示例（对应 `routes/ai.py:295-311` 注册的路由）：**

```bash
curl -X POST http://localhost:3000/ai/vector-db \
  -H "Content-Type: application/json" \
  -d '{"name": "my_kb", "documents": [{"id": "doc1", "text": "示例文本"}]}'

curl -X POST http://localhost:3000/ai/vector-db/search \
  -H "Content-Type: application/json" \
  -d '{"db_name": "vdb_my_kb", "query": "示例查询", "top_k": 3}'
```

### 常见问题排查

- **维度不匹配报错 `Qdrant collection 维度不匹配`**：切换了 Embedding 模型或修改了 `VECTOR_DB_DIMENSION` 后，已存在 collection 的向量维度与新配置不一致。默认拒绝写入以防止数据错乱；确认要重建时设置 `QDRANT_RECREATE_COLLECTION_ON_DIM_MISMATCH=1` 后重新创建库（`vector_db_qdrant.py:139-146`）。
- **检索结果为空但库里确实有数据**：检查 `_collection_physical_exists()` 是否为真——库名拼写错误、或 collection 尚未创建（新建库时 `documents` 传空）都会导致直接返回 `[]`。
- **`search_api` 调试接口的 `distance` 排序和界面显示对不上**：`search_api`（`vector_db_qdrant.py:1504`）显式传 `use_mmr=False`，用注释解释的原因是"避免 MMR 为了多样性把 distance 更大（更不相关）的结果排到前面"；若需要体验线上 RAG 的真实排序效果，应直接调用 `service/ai/rag.py` 的问答接口而非这个原始检索调试入口。
- **Hybrid 关键词兜底不生效**：取决于 Qdrant 服务端版本对 `text` 字段 full-text payload index 的支持，`_ensure_payload_indexes()` 和 `search_in_db` 里的 Hybrid 分支均为静默 `try/except`，不支持时不会报错，只是自动退化为纯稠密检索；确认版本支持后需对已有 collection 重新执行一次索引创建（新建 collection 会自动建）。
- **同名库改名后检索不到数据**：库改名走 `update_meta_api → _rename_qdrant_collection()`（`vector_db_qdrant.py:253`），逻辑是 scroll 旧 collection 全量数据后 upsert 到新 collection 再删旧库；大库改名耗时与文档量成正比，改名过程中应避免并发写入旧库名。

---

## 第五部分：评估与展望

### 优势

- **管理能力完整**：相比纯 FAISS 方案，原生支持 payload 过滤（`category`/`metadata`）、CRUD、多进程/多 worker 安全访问，无需自建缓存失效机制。
- **检索质量工程化**：Hybrid + MMR + 阈值兜底 + Rerank（RAG 层）的组合覆盖了"漏召""重复""跑题"等常见检索质量问题，且均可通过参数按查询场景动态开关。
- **双存储容灾**：Qdrant（检索）与 MySQL（管理）互为备份，`sync_vector_db_from_disk`/`rebuild_vector_db_from_mysql` 提供双向修复路径。
- **兼容性设计稳健**：对 `qdrant-client` 新旧 API（`search` vs `query_points`）、Hybrid 索引支持与否均做了运行时探测和优雅降级，升级客户端库或更换 Qdrant 部署版本时风险较低。

### 已知局限与技术债务

- **正向双写无补偿机制**：MySQL 写入失败但 Qdrant 已写入成功的场景没有自动回滚或重试，只能依赖 `rebuild_vector_db_from_mysql`/`sync_vector_db_from_disk` 人工触发修复。
- **Hybrid 关键词兜底静默降级不可观测**：功能失效无日志无告警，只能通过检索效果异常间接发现，缺乏可观测性埋点。
- **MMR 过采样的内存/带宽开销**：`with_vectors=True` 时每个候选传回完整向量，`fetch_k` 上限 200 时是可控的，但如果未来提高该上限或提高维度，网络开销会显著上升。
- **改名操作非原子**：`_rename_qdrant_collection` 通过 scroll+upsert 迁移数据，中途失败会导致新旧 collection 同时残留部分数据，缺少事务保护。
- **单 Qdrant 实例无分片方案**：当前实现假设一个 `QDRANT_URL` 对应单实例，未涉及分片/多副本路由，规模增长后需要额外引入 Qdrant 集群模式或代理层。

### 演进建议

- **短期**：为 Hybrid 索引创建失败、双写失败等静默路径补充结构化日志/告警，提升生产环境可观测性；补充"MySQL 写入失败回滚 Qdrant"的反向补偿逻辑。
- **中长期**：库规模持续增长后引入 Qdrant 集群模式（多分片/多副本）；评估将 Rerank/BM25 兜底等 RAG 层检索增强能力与本模块的检索接口做更清晰的分层（当前 `enable_hybrid`/`use_mmr`/BM25 分散在两个文件中，参数透传链路较长）。

### 行业前沿

- **原生 Hybrid Search**：新一代向量数据库（如 Qdrant 1.10+ 的 Query API、Weaviate）逐步把稠密+稀疏向量融合检索做成原生能力，未来可能替代本模块当前在应用层手工实现的 Hybrid 融合逻辑。
- **Matryoshka/二值量化 Embedding**：支持在检索时动态截断向量维度或使用二值量化降低存储和计算成本，在千万级以上规模时是重要的降本方向。
- **检索与生成一体化（Agentic RAG）**：检索不再是单次调用，而是由 LLM 自主决定是否检索、检索几次、如何组合多个数据源，向量库的角色从"被动检索接口"转变为"Agent 可调用的工具之一"。

---

## 变更记录

**2026-08-18 全量重写**

本次为通读 `service/ai/vector_db_qdrant.py`（1533 行）后的完全重写，与旧版本文档（描述的是 `service/ai/vector_db.py`，基于 FAISS + 本地文件 `.faiss`/`.npy`/`.json` 三件套的实现）相比关键变化：

- **存储引擎替换**：FAISS `IndexFlatL2` 本地索引 → Qdrant 远程 collection（HNSW + COSINE 距离），不再有 `vectors.npy` 增量更新和 `_db_cache` 进程内缓存机制，改为直连 Qdrant 服务。
- **ID 映射机制新增**：新增 `_point_uuid()` 的 UUID5 稳定映射解决 Qdrant point id 类型限制，旧版 FAISS 方案用 `IndexIDMap` 直接支持任意整数 ID，无此问题。
- **检索能力增强**：新增 Hybrid 关键词兜底检索、MMR 多样性重排、`score_threshold` 阈值过滤 + 空结果兜底，旧版仅有纯 L2 精确检索。
- **管理能力变化**：新增 `_rename_qdrant_collection()` 支持库改名；旧版磁盘方案改名需重命名整个目录。
- **对比方案调整**：对比表中的"本地实现"从"FAISS + 文件"更新为"Qdrant + MySQL 双写"，选型建议相应更新为"中等规模多租户知识库"场景。
- **关联文档**：检索策略的参数细节（Hybrid/MMR/Rerank/BM25 组合调参）已独立成文 `docs/service_ai/11_retrieval_strategy_qdrant.md`，本文第二、三部分做概要描述并指向该文档避免重复。
