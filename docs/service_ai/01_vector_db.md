# 向量库层（Vector DB）

> 文件：`service/ai/vector_db.py`（1353 行）  
> 生成日期：2026-02-26

---

## 第一部分：技术背景与演进

**问题背景**

传统数据库擅长精确匹配（`WHERE name = '张三'`），但无法回答"和这段话意思相近的文档有哪些"。随着深度学习的普及，Embedding 模型可以把任意文本、图片、音频压缩为一个固定维度的浮点向量，语义相似的内容在向量空间中距离更近。这就需要一种能高效存储和检索高维浮点向量的专用数据结构——向量库。

**核心概念**

- **Embedding**：深度模型将内容映射到高维空间的浮点数组。本项目使用阿里 DashScope `text-embedding-v4`，输出 1024 维归一化单位向量。
- **ANN（近似最近邻）**：在数百万向量中找出与查询向量最相近的 Top-K 个，允许损失少量精度换取速度。
- **IndexIDMap**：FAISS 的包装层，将向量的物理顺序编号映射到任意业务 ID，解耦检索引擎与业务语义。

**演进脉络**

| 时期 | 方案 | 特点 |
|------|------|------|
| 早期 | 暴力搜索（Brute Force） | O(n·d)，百万条以上延迟不可接受 |
| 2010s | LSH、IVF 等 ANN 算法 | 近似检索，速度大幅提升，精度有损 |
| 2017 | **FAISS**（Meta AI） | 引入 PQ 量化、GPU 加速，亿级向量工程化 |
| 2019+ | 系统化向量数据库（Milvus、Qdrant） | 补齐 CRUD、持久化、分布式等数据库能力 |
| 2022+ | RAG 时代 | 向量库成为 LLM 的标配"外挂记忆" |

**本模块的定位**

本项目选择 FAISS + 本地文件方案，处于演进史的"专用库"阶段。这是务实的选择：无需引入额外服务，单机即可运行，适合当前百万条以内的业务规模，且通过工程设计（`vectors.npy` 增量更新、`_db_cache` 进程缓存）弥补了部分"数据库"层面的能力缺失。

---

## 第二部分：架构剖析

**磁盘存储结构**（每个向量库对应一个目录）

```
data/vector_dbs/{db_name}/
├── index.faiss      # FAISS IndexIDMap 序列化（检索唯一来源）
├── metadata.json    # 文档列表 [{id, text, category, metadata?}, ...]
└── vectors.npy      # float32 原始向量矩阵，shape=(n, 1024)
```

**双存储同步策略**

```
磁盘（检索来源）          MySQL（管理来源）
  index.faiss    ◄──►   vector_db 表
  metadata.json  ◄──►   vector_db_document 表
  vectors.npy           vector_db_category 表
```

磁盘是运行时的真实来源；MySQL 是管理界面的数据来源。两者职责明确、互为备份，配套 `sync_vector_db_from_disk` 和 `rebuild_vector_db_from_mysql` 可从任一侧恢复故障。

**核心数据流**

```
写入：
  文本 → get_embedding() → DashScope API → float32[1024]
       → IndexIDMap.add_with_ids()
       → faiss.write_index() + json.dump() + np.save()
       → MySQL INSERT

检索：
  查询文本 → get_embedding() → float32[1024]
           → index.search(query_vec, top_k)
           → [(distance, vector_id), ...]
           → metadata[vector_id] 取出文档
           → [{doc, distance, rank}, ...]
```

**关键设计原则**

- **延迟 import**：`faiss`、`numpy` 不在模块顶层导入，仅在首次调用时按需加载，规避 FAISS C++ 析构顺序导致的 segfault。
- **进程内缓存**：`_db_cache` 缓存已加载的 Index + metadata，同进程重复检索零磁盘 I/O。
- **增量向量更新**：`vectors.npy` 保存原始向量矩阵，单条更新只修改对应行，无需重新 embed 整库。

**与行业标准方案对比**

| 维度 | 本地实现（FAISS + 文件） | Milvus / Qdrant | pgvector |
|------|------------------------|-----------------|---------|
| 适用规模 | < 100 万条，单机 | 百亿条，分布式 | < 100 万条 |
| 索引类型 | IndexFlatL2（精确） | HNSW/IVF/DiskANN 多选 | HNSW / IVFFlat |
| 混合检索 | 无，应用层过滤 | 原生向量+标量联合索引 | 支持，但性能有限 |
| 持久化 | 本地文件系统 | 对象存储 + WAL | PostgreSQL |
| 运维复杂度 | 极低（随应用部署） | 中（独立集群） | 低（复用 PG） |
| **选型建议** | 单机、快速验证、规模 < 百万 | 核心业务、高并发、规模 > 百万 | 已有 PG 栈、规模小 |

---

## 第三部分：代码实现深度解析

**核心函数清单**

| 函数 | 作用 |
|------|------|
| `get_embedding(text, max_retries=3)` | 调用 DashScope text-embedding-v4，指数退避重试（1s/2s/4s） |
| `create_vector_db(db_name, documents)` | 批量 embed + 建 FAISS 索引 + 落盘 + MySQL 写入 |
| `load_vector_db(db_name)` | 读取磁盘索引，优先命中 `_db_cache` |
| `search_in_db(db_name, query, top_k)` | 向量检索，返回 `[{doc, distance, rank}]` |
| `add_single_document(...)` | 追加单条，只对新文档调 embed |
| `update_single_document(...)` | 单条向量原地替换（修改 `vectors.npy` 对应行，重建 Index） |
| `delete_single_document(...)` | 删除一行，从矩阵剔除，重建 Index |
| `append_documents_batch(...)` | 批量追加，跳过已存在 doc_id |

**设计决策与取舍**

**决策 1：IndexFlatL2（精确搜索）而非 HNSW**  
原因：当前单库规模通常在数千到数万条，L2 精确搜索耗时 < 5ms，召回率 100%；知识库 RAG 场景不能容忍漏召。  
代价：O(n) 时间复杂度，超过 10 万条后延迟将线性增长。  
演进路径：超过 10 万条时，只需将 `faiss.IndexFlatL2` 替换为 `faiss.IndexHNSWFlat`，其余逻辑不变。

---

**决策 2：`vectors.npy` 增量更新**  
原因：FAISS Index 不支持原地修改/删除向量。没有 `vectors.npy` 时，更新一条文档需要对整库所有文档重新调 Embedding API（千条 = 数分钟 + 大量 token 消耗）。  
代价：额外的磁盘占用（n × 1024 × 4 字节，1 万条 ≈ 40MB）。  
实现细节：
```python
# 单条更新，只重新 embed 变更行
vectors_np = np.load(vectors_path)  # 读取整个矩阵
vectors_np[idx] = new_embedding     # 只改一行
np.save(vectors_path, vectors_np)   # 写回
# 用新矩阵重建 FAISS Index（纯内存操作，< 1ms）
new_index.add_with_ids(vectors_np, vector_ids_np)
```

---

**决策 3：进程内 `_db_cache`**  
原因：`faiss.read_index` 是 I/O 密集操作，RAG 场景每次问答都检索同一知识库，不能每次重新读盘。  
代价：多进程部署（如 Gunicorn 多 worker）时各 worker 缓存独立，一个 worker 写入后其他 worker 缓存不自动失效。  
当前状态：项目单进程运行，此问题暂不存在；若引入多进程，需增加基于文件 mtime 或 Redis 的缓存失效机制。

---

**决策 4：库名正则校验 `^[a-zA-Z0-9_-]+$`**  
原因：库名直接用于构造磁盘路径 `data/vector_dbs/{db_name}/`，不校验会有路径穿越风险（如 `db_name = "../../etc"`）。

---

**决策 5：旧库兼容 —— `vectors.npy` 懒生成**  
`vectors.npy` 是后期引入的。对于早期创建的无 `vectors.npy` 旧库，首次触发追加/更新时会全量 re-embed 并落盘，此后进入增量模式。这保证了向前兼容，无需迁移脚本。

---

## 第四部分：应用场景与实战

**使用场景**

- 知识库 RAG：`rag.py` 直接调用 `search_in_db` 检索相关文档片段
- 知识库管理：`knowledge.py` 调用 `create_vector_db` / `append_documents_batch` 完成文档向量化
- 直连向量库：路由层通过 `add_single_document` / `document_update_api` 等接口管理文档

**环境依赖**

```bash
pip install faiss-cpu numpy openai
export DASHSCOPE_API_KEY=sk-xxx
export VECTOR_DB_STORAGE=./data/vector_dbs   # 可选，默认值即此
```

**代码示例**

```python
from service.ai.vector_db import create_vector_db, search_in_db, add_single_document

# 1. 创建向量库
result = create_vector_db("my_kb", [
    {"id": "doc1", "text": "FAISS 是 Meta 开源的向量检索库", "category": "AI工具"},
    {"id": "doc2", "text": "向量数据库用于语义相似度检索", "category": "概念"},
])
# result = {"count": 2, "path": ".../data/vector_dbs/my_kb", "documents": [...]}

# 2. 检索
hits = search_in_db("my_kb", "向量检索工具有哪些", top_k=1)
# hits = [{"doc": {"id": "doc1", "text": "..."}, "distance": 0.12, "rank": 1}]

# 3. 追加单条
add_single_document(db_name="my_kb", text="Milvus 是分布式向量数据库", category="AI工具")
```

**常见问题**

- **进程退出时 segfault**：FAISS C++ 析构与 Python GC 顺序冲突。已通过延迟 import 缓解；仍出现时升级 `faiss-cpu` 版本。
- **Embedding 超时**：默认超时 30s。网络不稳定时调高 `OpenAI(timeout=60)`，或切换本地 Embedding 模型。
- **旧库缺少 `vectors.npy`**：首次更新时会触发全量 re-embed，耗时与库大小成正比，属预期行为。

---

## 第五部分：优缺点评估与未来展望

**优势**

- 零依赖部署：随应用启动，无需额外服务
- 增量 Embedding：`vectors.npy` 设计让单条更新成本接近 O(1) API 调用
- 数据韧性：双存储 + 两个修复函数，任意一侧故障均可恢复
- 精确召回：IndexFlatL2 在当前规模下保证 100% 召回，RAG 质量更稳定

**已知局限**

- 规模上限：超过 10 万条后 L2 精确搜索延迟线性增长
- 多进程缓存一致性：多 worker 部署时缓存无法自动同步
- 无混合检索：标量字段过滤只能在应用层手工实现，无法利用索引加速
- 单机瓶颈：数据存本地文件，无法水平扩展

**演进建议**

- 短期：超过 10 万条时切换 `IndexHNSWFlat`，接口不变
- 中期：多进程时引入文件 mtime 检查实现缓存失效；标量过滤在检索后做 post-filter
- 长期：规模到千万条级别时，整体迁移至 Milvus Lite（嵌入式，无独立服务）或 Qdrant，现有接口层不需改动

**行业前沿**

- **DiskANN / LSMDiskANN**：磁盘索引突破内存限制，借鉴 LSM-Tree 优化混合读写性能，适合 PB 级数据
- **CAGRA（NVIDIA）**：GPU 原生 HNSW，在 A100 上单卡 >1M QPS，适合超大规模批量检索
- **内置 Embedding 推理**：下一代向量数据库原生支持将原始文本直接写入，库内完成向量化，消除外部 API 依赖
