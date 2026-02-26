# RAG 检索增强生成层

> 文件：`service/ai/rag.py`（354 行）、`service/ai/rag_enhance.py`（179 行）  
> 生成日期：2026-02-26

---

## 第一部分：技术背景与演进

**问题背景**

大语言模型（LLM）有两个根本局限：训练知识有截止日期（不知道最新信息），以及不了解私有领域文档（公司内部知识库、产品手册）。直接让 LLM 回答相关问题，轻则答非所问，重则产生"幻觉"（编造可信但错误的内容）。RAG（Retrieval-Augmented Generation，检索增强生成）通过在生成前先检索相关文档并将其塞入上下文，从根本上解决了这两个问题。

**核心概念**

- **RAG**：先检索（Retrieval）后生成（Generation）——用问题在向量库中找最相关的文档片段，拼成上下文后交给 LLM 生成答案，答案有文档依据，幻觉大幅减少。
- **Query 改写（CASEA）**：用户提问往往含有歧义、指代、多意图，改写后的问题更适合向量检索，能召回更精准的文档。
- **Rerank（重排序）**：向量检索的排序依据是距离，但距离不等于相关性。Rerank 用语义理解模型对检索结果做精排，使最相关的文档排在最前面。

**演进脉络**

| 阶段 | 方案 | 局限 |
|------|------|------|
| 朴素 RAG | 直接向量检索 → LLM | 向量检索精度受限，多轮对话时 query 含指代 |
| 进阶 RAG | + Query 改写 | 解决模糊指代，但不能保证检索结果质量 |
| 高级 RAG | + Rerank 重排序 | 精排改善检索结果相关性，但增加延迟 |
| Modular RAG | 各模块可插拔，按场景组合 | 本项目即此阶段：改写和 Rerank 均为可选开关 |

**本模块的定位**

`rag.py` 是 RAG 流水线的主调度层，`rag_enhance.py` 提供两个可选增强组件（改写、重排）。两者向上为 HTTP 路由层提供 `rag_ask_api` / `rag_search_api`，向下调用 `vector_db.search_in_db` 完成检索。RAG 层是整个知识库问答能力的核心编排者。

---

## 第二部分：架构剖析

**完整 RAG 流水线**

```
POST /ai/rag/ask
      │
      ▼ rag_chat(kb_id/kb_name, question, ...)
      │
      ├─[enable_query_rewrite=True]──► query_rewrite(question)
      │   DashScope qwen-turbo + CASEA Prompt
      │   识别类型：上下文依赖/对比/模糊指代/多意图/反问/无需改写
      │   输出：{rewritten_query, query_type, confidence}
      │
      ├──► search_in_db(db_name, search_query, top_k * 2)
      │     FAISS L2 检索，Rerank 时多召一倍候选
      │
      ├─[enable_rerank=True]──────► rerank_documents(query, results, top_n)
      │   DashScope qwen3-rerank
      │   输出：{before: [原始顺序], after: [重排后顺序]}
      │
      ├──► 组装 Prompt（参考资料 + 问题）
      │
      └──► DashScope Chat Completion → answer
      
返回：{answer, sources, rewritten_query, before}
```

**知识库寻址逻辑**

`rag.py` 同时支持两种 ID 体系，内部自动解析：

```python
# 支持 知识库ID → 自动找对应向量库
# 支持 向量库ID / 向量库名称 → 直接使用
# 两套体系通过 knowledge_base.vector_db_id 外键打通
```

**关键设计原则**

- **可观测的增强链路**：Query 改写和 Rerank 均在返回值中携带 `before/after` 对比数据，前端可直接渲染改写前后对比、重排前后排序变化，无需额外调试接口。
- **Rerank 时多召候选**：`retrieve_k = min(20, top_k * 2)`——启用 Rerank 时先召回两倍候选，再精排取 top_k，确保精排有足够的候选可用。
- **检索与生成解耦**：`rag_search_api` 只做检索不生成，供前端展示向量搜索效果；`rag_ask_api` 是完整 RAG 流水线。

**与行业标准方案对比**

| 维度 | 本地实现 | LangChain RAG | LlamaIndex |
|------|---------|--------------|-----------|
| 灵活性 | 手工编排，逻辑透明 | 链式封装，调试困难 | 高抽象，概念多 |
| 增强组件 | 改写 + Rerank 可选开关 | 需自定义 Chain | 内置多种 Retriever |
| 可观测性 | 原生返回 before/after | 需额外回调 | 需额外追踪 |
| 上手成本 | 低，纯 Python 函数 | 中 | 高 |
| 生态集成 | 仅 DashScope | 支持数十种 LLM/向量库 | 支持数十种 |
| **选型建议** | 自有系统、需要可观测、DashScope 生态 | 快速原型、多模型切换 | 复杂文档索引场景 |

---

## 第三部分：代码实现深度解析

**核心函数清单**

| 函数 | 文件 | 作用 |
|------|------|------|
| `rag_chat(kb_id, kb_name, question, top_k, model, enable_query_rewrite, enable_rerank, ...)` | rag.py | 完整 RAG 流水线主函数 |
| `query_rewrite(query, conversation_history, model)` | rag_enhance.py | CASEA Query 改写，JSON 结构化输出 |
| `rerank_documents(query, documents, top_n, model)` | rag_enhance.py | DashScope qwen3-rerank 精排 |
| `rag_ask_api()` | rag.py | Flask 接口，POST `/ai/rag/ask` |
| `rag_search_api()` | rag.py | Flask 接口，POST `/ai/rag/search`，纯检索 |

**CASEA 改写的 6 种 Query 类型**

```
1. 上下文依赖型：含"还有"、"其他"→ 补全上文语境
2. 对比型：含"哪个更好"、"比较" → 展开为明确的对比问题  
3. 模糊指代型：含"它"、"这个" → 替换为具体指代
4. 多意图型：包含多个子问题 → 合并为主要意图的完整问句
5. 反问型：含"难道"、"不会" → 转为正向陈述检索
6. 无需改写：直接返回原问题，confidence=1.0
```

**Rerank 的 before/after 结构**

```python
# rerank_documents 返回
{
  "before": [{"doc": {...}, "rank": 1, "distance": 0.12}, ...],   # 向量检索原始顺序
  "after":  [{"doc": {...}, "rank": 1, "relevance_score": 0.95}, ...]  # 语义重排后顺序
}
```

两份列表同时返回给前端，让用户直观感受 Rerank 对排序的改善效果。

**设计决策与取舍**

**决策 1：Rerank 时多召两倍候选**  
原因：Rerank 的价值在于"从次优集合中找最优"，如果只召 top_k 条再精排，相当于在已经最优的集合里洗牌。多召两倍让精排有更多候选可选，召回率-精度 tradeoff 更优。  
代价：向量检索和 Rerank API 的调用量增加，延迟略升（约 +200ms）。

**决策 2：检索超时单独处理（`rag_search_api`）**  
```python
except Exception as e:
    if "timeout" in err_msg.lower():
        return jsonify({"code": 504, "msg": "检索超时"}), 504
    raise
```
向量检索超时（Embedding API 网络延迟）是可恢复的临时错误，单独捕获返回 504 而非 500，前端可据此提示用户重试。

**决策 3：Prompt 模板固化**  
```python
prompt = f"""基于以下参考资料回答问题。若资料中无相关内容，请说明无法从资料中得出答案。
参考资料：{context}
问题：{question}
请直接给出答案（可简要说明依据的段落或页码）："""
```
明确告知模型"资料中无相关内容时说明无法回答"，主动引导模型减少幻觉，而不是让模型"尽力回答"。

---

## 第四部分：应用场景与实战

**使用场景**

- 企业知识库问答：上传产品手册/规章制度，用户自然语言提问
- 多轮对话增强：`conversation_history` 参数传入历史对话，改写时补全上下文
- 纯检索效果评估：用 `rag_search_api` 单独测试向量检索 + Rerank 效果

**环境依赖**

```bash
pip install openai dashscope
export DASHSCOPE_API_KEY=sk-xxx
```

**代码示例**

```python
from service.ai.rag import rag_chat

# 基础调用
out = rag_chat(kb_name="product_manual", question="退款流程是什么？")
print(out["answer"])    # LLM 生成的答案
print(out["sources"])   # 检索到的文档片段

# 开启改写 + Rerank
out = rag_chat(
    kb_name="product_manual",
    question="它的退款周期多长？",
    enable_query_rewrite=True,    # 改写"它"为明确指代
    enable_rerank=True,           # 语义重排提升精度
    conversation_history="用户上一问：支付宝怎么付款？",
    top_k=5,
)
print(out["rewritten_query"])  # 改写后的 query
print(out["before"])           # Rerank 前的排序（对比用）
```

**常见问题**

- **Query 改写结果不稳定**：LLM 输出带 markdown 代码块时会触发 JSON 解析的容错逻辑（`for start in ("{", "```json")`），改写成功但 confidence 可能偏低。
- **Rerank 后答案质量下降**：排序更好不代表 LLM 理解更好，检查 `sources` 的 `relevance_score` 是否确实高于 `distance`；若发现重排结果奇怪，可能是 `qwen3-rerank` 对该领域效果有限。
- **知识库 ID 和向量库 ID 混用报错**：两套 ID 体系均支持，但若传入的 ID 既不是知识库记录也不是向量库记录，会抛出 `FileNotFoundError("知识库或向量库不存在")`，检查参数名是否正确（`kb_id` vs `db_id`）。

---

## 第五部分：优缺点评估与未来展望

**优势**

- 全链路可观测：改写前后、Rerank 前后均有对比数据，便于调优
- 按需开启增强：改写和 Rerank 都是布尔开关，不影响基础功能
- 知识库/向量库双 ID 透明寻址，调用方无需关心底层 ID 体系

**已知局限**

- RAG 质量上限由 Embedding 模型决定，当前 text-embedding-v4 对中文效果较好，但领域专业词汇可能召回偏差
- 固定 Prompt 模板，无法针对不同知识库类型（问答型、文档型）做差异化引导
- 没有 Chunk 级别的 Score 阈值过滤——距离很远的文档也会被塞入 context，可能引入噪声

**演进建议**

- 短期：增加 `min_score` 参数，过滤掉距离超过阈值的候选，减少噪声 context
- 中期：引入 HyDE（假设文档 Embedding）——先让 LLM 生成假设答案，用假设答案的向量去检索，提升召回质量
- 长期：实现对话级上下文记忆（将历史 QA 对也向量化存入），让 RAG 支持真正的多轮问答而非仅靠 `conversation_history` 字符串

**行业前沿**

- **Self-RAG**：LLM 自主决定是否需要检索、检索哪些内容，减少无意义的检索开销
- **RAPTOR**：对文档做递归摘要树，检索时可在不同粒度（段落/章节/全文）命中，解决长文档问答问题
- **GraphRAG**（微软）：将知识库构建为知识图谱，支持多跳推理型问答，弥补纯向量检索对关系型问题的不足
