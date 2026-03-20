# AI 应用工程师面试题精要

---

## 一、大模型基础

### 1. Transformer 核心架构

```
输入 → Embedding + 位置编码
     → N × (Multi-Head Attention + FFN + LayerNorm + 残差连接)
     → 输出层
```

- **Self-Attention：** Q、K、V 来自同一序列，捕捉序列内部依赖
- **Multi-Head：** 多组注意力头并行，捕捉不同维度语义
- **位置编码：** 弥补注意力机制无位置感知的缺陷（绝对/旋转 RoPE）

---

### 2. Attention 计算公式

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $\sqrt{d_k}$ 缩放防止点积过大导致 softmax 梯度消失
- 复杂度 O(n²)，是长文本的瓶颈（Flash Attention 优化 IO）

---

### 3. 主流大模型对比


| 模型             | 厂商        | 特点              |
| -------------- | --------- | --------------- |
| GPT-4o         | OpenAI    | 多模态，业界基准        |
| Claude 3.5     | Anthropic | 长上下文，代码强        |
| Gemini 1.5     | Google    | 超长上下文（1M token） |
| Qwen2.5        | 阿里        | 开源，中文表现好        |
| LLaMA 3        | Meta      | 开源基座，生态丰富       |
| DeepSeek-V3/R1 | 深度求索      | 开源，推理能力强，成本低    |


---

### 4. Token 是什么？

文本被分词器（Tokenizer）切分的最小单元，可以是词、子词或字符。

- 英文约 1 token ≈ 0.75 个单词
- 中文约 1 token ≈ 1~2 个汉字（模型不同）
- `max_tokens` 限制一次请求的输出长度
- `context window` 限制输入+输出的总 token 数

---

### 5. 常用推理参数


| 参数                  | 作用                  | 建议值                     |
| ------------------- | ------------------- | ----------------------- |
| `temperature`       | 随机性，越高越创意           | 0（确定性）~1（多样性）           |
| `top_p`             | 核采样，累计概率阈值          | 0.9（与 temperature 二选一调） |
| `top_k`             | 每步只考虑概率最高 k 个 token | 40~80                   |
| `max_tokens`        | 最大输出长度              | 按需设置                    |
| `frequency_penalty` | 降低已出现 token 的概率     | 防重复                     |
| `presence_penalty`  | 鼓励涉及新话题             | 增多样性                    |


> 生产环境 temperature=0 保证确定性；创意写作 0.7~1.0。

---

### 6. Prompt Engineering 核心技巧


| 技巧                   | 说明                          |
| -------------------- | --------------------------- |
| **角色设定**             | System prompt 赋予 AI 角色和行为规范 |
| **Few-shot**         | 在 prompt 中给几个示例，引导输出格式      |
| **Chain of Thought** | 让模型先推理再回答（"让我一步步思考"）        |
| **输出格式**             | 要求 JSON/XML/Markdown，便于解析   |
| **分步拆解**             | 复杂任务拆成多个 prompt，逐步完成        |
| **Self-consistency** | 多次采样取多数答案，提升准确率             |


---

---

## 二、向量数据库

### 1. 向量数据库是什么？

存储和检索**高维向量**（Embedding）的数据库，支持**相似度搜索**（ANN，近似最近邻）。

**核心能力：**

- 语义检索（而非关键词匹配）
- 毫秒级相似度查询
- 支持过滤（Metadata Filter）

---

### 2. Embedding（向量嵌入）

将文本/图像等转换为稠密向量的过程，语义相近的内容在向量空间中距离近。

```python
# OpenAI Embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="你好世界"
)
vector = response.data[0].embedding  # 1536 维向量
```

---

### 3. 相似度计算方法


| 方法    | 公式                                      | 适用场景      |
| ----- | --------------------------------------- | --------- |
| 余弦相似度 | $\cos\theta = \frac{A \cdot B}{|A||B|}$ | 文本语义（最常用） |
| 欧氏距离  | $\sqrt{\sum(a_i-b_i)^2}$                | 图像、推荐     |
| 点积    | $A \cdot B$                             | 归一化向量等价余弦 |


> 大多数 Embedding 模型输出已归一化，余弦相似度 = 点积。

---

### 4. ANN 索引算法


| 算法           | 原理          | 特点           |
| ------------ | ----------- | ------------ |
| **HNSW**     | 分层图结构导航     | 查询快，内存大，主流选择 |
| **IVF**      | 倒排文件，先聚类再搜索 | 内存小，速度中等     |
| **PQ（乘积量化）** | 向量压缩存储      | 节省内存，精度有损    |
| **IVF-PQ**   | IVF + PQ 组合 | 大规模场景        |


---

### 5. 主流向量数据库对比


|     | Chroma  | Milvus  | Qdrant     | Pinecone | Weaviate    |
| --- | ------- | ------- | ---------- | -------- | ----------- |
| 部署  | 本地/嵌入式  | 自托管     | 自托管        | 云托管      | 自托管/云       |
| 规模  | 小中      | 大规模     | 中大         | 大规模      | 中大          |
| 特色  | 轻量，开发友好 | 高性能，企业级 | Rust 编写，高效 | 全托管 SaaS | 多模态，GraphQL |
| 适合  | 快速原型    | 生产大规模   | 生产中规模      | 省运维      | 知识图谱        |


---

### 6. 向量检索 + 标量过滤

```python
# Qdrant 示例：语义搜索 + 元数据过滤
results = client.search(
    collection_name="docs",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
    ),
    limit=5
)
```

先过滤 metadata，再在过滤结果中做向量搜索（pre-filtering vs post-filtering）。

---

---

## 三、知识库

### 1. 知识库构建流程

```
原始文档（PDF/Word/网页）
    ↓ 文档解析（PyPDF2、LlamaParse）
    ↓ 文本清洗（去噪、格式统一）
    ↓ 文本切分（Chunking）
    ↓ 向量化（Embedding Model）
    ↓ 存入向量数据库
    ↓ （可选）建立关键词索引 BM25
```

---

### 2. Chunking 策略


| 策略   | 说明                       | 适用            |
| ---- | ------------------------ | ------------- |
| 固定大小 | 按 token 数切分，有重叠（overlap） | 通用            |
| 递归字符 | 按段落→句子→词逐级切分             | 结构化文档         |
| 语义切分 | 相邻句子 Embedding 相似度骤降处切   | 长文            |
| 文档结构 | 按标题/章节切分                 | Markdown、HTML |
| 父子切分 | 小块检索，返回大块上下文             | 精度+上下文兼顾      |


**关键参数：**

- `chunk_size`：每块 token 数（通常 256~1024）
- `chunk_overlap`：重叠 token 数，防止语义截断（通常 10%~20%）

---

### 3. 文档解析难点


| 文档类型 | 工具                           | 难点               |
| ---- | ---------------------------- | ---------------- |
| PDF  | PyPDF2、pdfplumber、LlamaParse | 扫描版（需 OCR）、表格、多列 |
| Word | python-docx                  | 嵌入式图表            |
| 网页   | BeautifulSoup、Firecrawl      | 动态渲染（需无头浏览器）     |
| 表格   | pandas、Camelot               | 合并单元格、跨页表格       |
| 图片   | OCR（Tesseract、PaddleOCR）     | 手写、低分辨率          |


---

### 4. 混合检索（Hybrid Search）

同时使用**向量检索（语义）**+ **BM25（关键词）**，结果融合后排序。

```
向量检索结果   →  RRF 融合算法  →  最终排名
BM25 检索结果  →
```

**RRF（Reciprocal Rank Fusion）：**  
$\text{score}(d) = \sum_r \frac{1}{k + r(d)}$，$k=60$，融合多路排名。

> 混合检索在专业词汇（代码、产品名、人名）场景比纯向量检索效果好。

---

---

## 四、RAG（检索增强生成）

### 1. RAG 基本流程

```
用户问题
    ↓ 向量化（Embedding）
    ↓ 向量数据库检索 Top-K 相关文档
    ↓ 构建 Prompt（问题 + 检索文档）
    ↓ LLM 生成答案
    ↓ 输出（可附来源引用）
```

**核心价值：** 让 LLM 回答私域/最新知识，减少幻觉，支持来源溯源。

---

### 2. Naive RAG vs Advanced RAG vs Modular RAG


|       | Naive RAG | Advanced RAG | Modular RAG  |
| ----- | --------- | ------------ | ------------ |
| 检索    | 单次向量检索    | 查询改写、混合检索、重排 | 灵活组合各模块      |
| 生成前处理 | 无         | 上下文压缩、去冗余    | 自定义 Pipeline |
| 生成后处理 | 无         | 引用验证、反事实检测   | 可扩展          |
| 适用    | 快速原型      | 生产场景         | 复杂场景         |


---

### 3. Query 改写技术


| 技术                      | 说明                     |
| ----------------------- | ---------------------- |
| **HyDE**（假设文档嵌入）        | 先让 LLM 生成假设答案，用答案向量检索  |
| **Query Expansion**     | 生成多个相关子问题同时检索，结果合并     |
| **Step-back Prompting** | 将具体问题泛化为更高层次的问题再检索     |
| **多查询检索**               | 生成 3~5 个不同表述的查询，合并去重结果 |


---

### 4. Reranker（重排序）

检索到 Top-K（如 20）个候选后，用**交叉编码器（Cross-Encoder）**精排，取 Top-N（如 5）送入 LLM。

```
向量检索 Top-20  →  Cross-Encoder Reranker  →  Top-5  →  LLM
```

- **Bi-Encoder（双编码器）：** 检索用，速度快，粗排
- **Cross-Encoder：** 重排用，精度高，速度慢（query+doc 联合编码）

常用模型：`bge-reranker`、`cross-encoder/ms-marco`

---

### 5. 上下文压缩

检索到的文档可能包含大量无关内容，压缩后送入 LLM：

- **LLMLingua：** 用小模型删除冗余 token，压缩率可达 20x
- **Map-Reduce：** 分块摘要后合并
- **选择性抽取：** 只保留与问题相关的句子

---

### 6. RAG 评估指标


| 指标                    | 说明                    | 工具    |
| --------------------- | --------------------- | ----- |
| **Context Recall**    | 检索文档覆盖了多少相关信息         | RAGAS |
| **Context Precision** | 检索文档中有多少是真正相关的        | RAGAS |
| **Faithfulness**      | 答案是否忠实于检索内容（无幻觉）      | RAGAS |
| **Answer Relevance**  | 答案是否回答了问题             | RAGAS |
| **End-to-End**        | 人工评估 / G-Eval（LLM 评估） | —     |


---

### 7. RAG vs Fine-tuning 选择


| 场景          | 推荐方案                     |
| ----------- | ------------------------ |
| 知识频繁更新      | RAG（无需重训）                |
| 私域文档问答      | RAG                      |
| 改变模型行为/风格   | Fine-tuning              |
| 专业领域能力提升    | Fine-tuning（或 RAG+FT 结合） |
| 实时数据        | RAG                      |
| 长尾知识 / 极多文档 | RAG                      |


---

### 8. GraphRAG

微软提出，将知识提取为**知识图谱**，支持全局主题摘要和关系推理。

```
文档 → 实体抽取 → 关系抽取 → 知识图谱
               ↓
     社区摘要（层次聚类）
               ↓
     全局/局部检索 → LLM
```

**优势：** 回答"这份报告的整体趋势是什么"类全局问题。  
**劣势：** 构建成本高，索引时间长。

---

---

## 五、Function Calling

### 1. 什么是 Function Calling？

让 LLM 决定何时调用哪个工具（函数），并生成符合格式的调用参数。

```
用户：北京今天天气怎么样？
  ↓
LLM：决定调用 get_weather(city="北京")
  ↓
代码执行函数，获取真实天气数据
  ↓
将结果返回给 LLM，生成最终回答
```

---

### 2. 完整调用流程

```python
# 1. 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取城市天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }
}]

# 2. 第一次调用：LLM 决定是否调用工具
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # auto/required/none
)

# 3. 解析工具调用
tool_call = response.choices[0].message.tool_calls[0]
func_name = tool_call.function.name
func_args = json.loads(tool_call.function.arguments)

# 4. 执行函数
result = get_weather(**func_args)

# 5. 第二次调用：将结果返回 LLM，生成最终答案
messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
final_response = client.chat.completions.create(model="gpt-4o", messages=messages)
```

---

### 3. Parallel Tool Calling

一次请求中 LLM 可同时调用多个工具：

```json
"tool_calls": [
  {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}},
  {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"city\":\"上海\"}"}}
]
```

并行执行后将两个结果都返回 LLM，减少 RTT。

---

### 4. 工具描述最佳实践

- **description 要清晰**：LLM 依赖描述决定是否调用，描述不准确会导致误用
- **参数 enum 限制取值范围**：防止 LLM 瞎猜
- **必填 vs 可选要明确**：`required` 数组只放必须参数
- **示例值**：在 description 中给出示例帮助 LLM 理解

---

---

## 六、MCP（Model Context Protocol）

### 1. 什么是 MCP？

Anthropic 提出的开放协议，定义 LLM 与外部工具/数据源的**标准化接口**，类似 AI 世界的 USB 接口。

```
LLM Host（Claude、Cursor）
    ↕ MCP 协议（JSON-RPC over stdio/SSE）
MCP Server（工具提供方）
    ├── Tools（可调用的函数）
    ├── Resources（可读取的数据）
    └── Prompts（预设提示模板）
```

---

### 2. MCP vs Function Calling


|     | Function Calling | MCP       |
| --- | ---------------- | --------- |
| 范围  | 单次对话内            | 跨应用、跨模型   |
| 标准化 | 各家格式不同           | 统一协议      |
| 复用性 | 需重复定义            | 一次开发，多处复用 |
| 生态  | 模型私有             | 开放社区      |


---

### 3. MCP Server 开发

```python
from mcp.server import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
def get_weather(city: str) -> str:
    """获取城市当前天气"""
    return f"{city}: 晴，25°C"

@mcp.resource("weather://forecast/{city}")
def get_forecast(city: str) -> str:
    """获取城市天气预报"""
    return f"{city} 未来7天: 晴转多云"

if __name__ == "__main__":
    mcp.run()
```

---

### 4. MCP 传输方式


| 方式                          | 场景                              |
| --------------------------- | ------------------------------- |
| **stdio**                   | 本地进程通信，Cursor/Claude Desktop 使用 |
| **SSE（Server-Sent Events）** | 远程服务，HTTP 长连接                   |
| **Streamable HTTP**         | 新标准，替代 SSE，支持双向流                |


---

---

## 七、Agent

### 1. Agent 是什么？

能够感知环境、自主规划、调用工具、完成目标的 LLM 应用模式。

```
感知（Perception）→ 思考/规划（Reasoning）→ 行动（Action）→ 观察（Observation）→ 循环
```

---

### 2. ReAct 框架

**Reason + Act**，交替推理和行动：

```
Thought: 用户想知道天气，我需要调用天气工具
Action: get_weather(city="北京")
Observation: 北京今天晴，25°C
Thought: 已获得天气信息，可以回答用户
Answer: 北京今天晴天，气温25°C，适合出行
```

---

### 3. Plan-and-Execute

先整体规划任务分解，再逐步执行：

```
用户目标 → Planner（生成步骤列表）
         → Executor（逐步执行每个步骤）
         → 必要时重新规划
```

**优于 ReAct 场景：** 长任务、步骤多、需要全局视野。

---

### 4. 主流 Agent 框架


| 框架                        | 特点                                       |
| ------------------------- | ---------------------------------------- |
| **LangChain Agents**      | 生态丰富，入门简单                                |
| **LangGraph**             | 状态机 + 图结构，可控性强，支持循环                      |
| **AutoGen**               | 多 Agent 对话框架，微软出品                        |
| **CrewAI**                | 角色化多 Agent，任务分工                          |
| **OpenAI Assistants API** | 托管 Agent，内置 code interpreter/file search |
| **Semantic Kernel**       | 微软，企业级，支持 C#/Python                      |


---

### 5. Agent 记忆类型


| 类型       | 说明        | 实现            |
| -------- | --------- | ------------- |
| **短期记忆** | 当前对话上下文   | Messages 列表   |
| **长期记忆** | 跨对话持久化记忆  | 向量数据库 / KV 存储 |
| **实体记忆** | 记录用户/实体信息 | 结构化存储         |
| **程序记忆** | 技能/SOP 知识 | Prompt 模板库    |


---

### 6. Tool Use 最佳实践

- 工具数量：单次调用不超过 20 个工具（模型能力限制）
- 工具描述：精准简洁，避免歧义
- 错误处理：工具失败要返回错误信息，让 Agent 重试或换策略
- 工具鉴权：敏感操作需要确认步骤（Human-in-the-loop）
- 防幻觉：不存在的工具调用要兜底处理

---

### 7. Human-in-the-Loop

在关键节点暂停等待人工审批，防止 Agent 执行不可逆操作：

```python
# LangGraph 示例
graph.add_node("review", human_review_node)
graph.add_edge("plan", "review")  # 执行前人工审核
graph.add_edge("review", "execute")
```

---

### 8. Agent 评估


| 维度      | 说明                  |
| ------- | ------------------- |
| 任务成功率   | 最终目标达成比例            |
| 工具调用准确率 | 选对工具、参数正确           |
| 步骤效率    | 是否有冗余步骤             |
| 幻觉率     | 凭空捏造结果比例            |
| 成本      | Token 消耗 / API 调用次数 |


---

---

## 八、A2A（Agent to Agent）

### 1. 什么是 A2A？

Google 提出的协议，让不同框架/厂商的 Agent 之间互相调用和协作，类似 AI 世界的微服务。

```
用户请求
    ↓
Orchestrator Agent（编排者）
    ├── 调用 Agent A（搜索专家）
    ├── 调用 Agent B（代码专家）
    └── 调用 Agent C（报告生成）
    ↓
汇总结果，返回用户
```

---

### 2. A2A 核心概念


| 概念                    | 说明                            |
| --------------------- | ----------------------------- |
| **Agent Card**        | Agent 的能力描述文件（JSON），类似 API 文档 |
| **Task**              | Agent 间传递的工作单元，有唯一 ID 和状态     |
| **Artifact**          | Task 产生的输出物（文件、文本等）           |
| **Push Notification** | Agent 完成后主动推送结果（异步）           |
| **Streaming**         | SSE 流式返回中间进度                  |


---

### 3. A2A vs MCP


|      | A2A           | MCP                   |
| ---- | ------------- | --------------------- |
| 通信对象 | Agent ↔ Agent | LLM ↔ Tools/Resources |
| 协议层  | HTTP + JSON   | JSON-RPC（stdio/SSE）   |
| 能力描述 | Agent Card    | Tool Schema           |
| 场景   | 多 Agent 协作编排  | 单 Agent 工具扩展          |


> 两者互补：MCP 扩展 Agent 的工具能力，A2A 编排多个 Agent 协作。

---

### 4. 多 Agent 设计模式


| 模式                      | 说明                        |
| ----------------------- | ------------------------- |
| **Orchestrator-Worker** | 一个主 Agent 分配任务给多个专家 Agent |
| **Peer-to-Peer**        | Agent 间平等协商（适合辩论/评审）      |
| **Pipeline**            | 流水线，输出传给下一个 Agent         |
| **Supervisor**          | 监督者审查 Worker 结果，不满意则重试    |
| **Blackboard**          | 共享状态板，Agent 读写协作          |


---

---

## 九、Fine-tuning（微调）

### 1. 为什么微调？


| 目的           | 说明                  |
| ------------ | ------------------- |
| 领域知识注入       | 让模型掌握专业领域语言/知识      |
| 输出格式固定       | 强制输出特定 JSON/模板      |
| 行为/风格定制      | 模拟特定角色或语气           |
| 减少 Prompt 长度 | 把 few-shot 示例"烧"进参数 |
| 提升小模型能力      | 小模型微调后超过大模型基础能力     |


---

### 2. 微调方法分类

**全量微调（Full Fine-tuning）**

- 更新所有参数，效果最好
- 需大量 GPU 显存，成本高

**参数高效微调（PEFT）**


| 方法                | 原理                  | 特点                |
| ----------------- | ------------------- | ----------------- |
| **LoRA**          | 插入低秩矩阵 A、B，只训练 A、B  | 主流，显存小，效果好        |
| **QLoRA**         | LoRA + 4-bit 量化基础模型 | 消费级 GPU 可用        |
| **Prompt Tuning** | 只训练输入的软提示向量         | 极少参数，效果较弱         |
| **Prefix Tuning** | 在每层添加可训练前缀          | 比 Prompt Tuning 强 |
| **IA³**           | 缩放激活值，参数极少          | 快速，效果中等           |


---

### 3. LoRA 原理

将权重更新矩阵 $\Delta W$ 分解为两个低秩矩阵之积：

$$W' = W + \Delta W = W + BA$$

- $W \in \mathbb{R}^{d \times d}$（冻结），$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$（可训练）
- $r \ll d$，大幅减少参数量
- 推理时可合并：$W' = W + BA$，无额外推理开销

---

### 4. 数据准备

**格式（Instruction Tuning）：**

```json
{
  "messages": [
    {"role": "system", "content": "你是专业的法律顾问"},
    {"role": "user", "content": "劳动合同试用期最长多久？"},
    {"role": "assistant", "content": "根据《劳动合同法》第19条..."}
  ]
}
```

**数据质量 > 数量：**

- 宁要 1000 条高质量数据，不要 10000 条噪声数据
- 多样性：覆盖目标场景的各种情况
- 一致性：标注规范统一

---

### 5. SFT vs RLHF vs DPO


|     | SFT        | RLHF             | DPO         |
| --- | ---------- | ---------------- | ----------- |
| 全称  | 监督微调       | 人类反馈强化学习         | 直接偏好优化      |
| 数据  | (输入, 输出) 对 | 人类排序偏好           | (优选, 劣选) 对  |
| 流程  | 简单，直接训练    | 复杂（SFT→奖励模型→PPO） | 较简单（无需奖励模型） |
| 效果  | 基础对齐       | 人类价值对齐（GPT-4）    | 接近 RLHF，成本低 |


---

### 6. 灾难性遗忘

微调时模型可能忘记原有通用能力。

**缓解方案：**

- **数据混合（Replay）：** 微调数据中混入原始通用数据（5%~10%）
- **小学习率：** 降低对原有参数的破坏
- **LoRA：** 不修改原始权重，天然缓解遗忘
- **EWC：** 弹性权重巩固（学术方法）

---

### 7. 超参数设置参考


| 参数              | 参考值               | 说明           |
| --------------- | ----------------- | ------------ |
| `learning_rate` | 1e-4 ~ 3e-4（LoRA） | 全量微调更小（1e-5） |
| `batch_size`    | 4~32              | 视显存          |
| `num_epochs`    | 2~5               | 数据量小多训几轮     |
| `lora_r`        | 8~64              | 秩，越大参数越多效果越好 |
| `lora_alpha`    | `2 * lora_r`      | 缩放系数         |
| `warmup_ratio`  | 0.03~0.1          | 预热步骤比例       |


---

### 8. 微调评估

- **困惑度（Perplexity）：** 语言模型基础指标，越低越好
- **任务指标：** BLEU/ROUGE（文本生成）、Accuracy（分类）、F1（NER）
- **人工评估：** 准确性、流畅性、有用性打分
- **LLM-as-Judge：** 用 GPT-4 评估输出质量

---

---

## 十、LLM 应用架构 & 工程实践

### 1. LLMOps 核心组件

```
数据管理 → Prompt 管理 → 模型服务
    ↓           ↓            ↓
版本控制    A/B 测试    推理优化（量化/并发）
    ↓           ↓            ↓
评估体系 → 监控告警 → 迭代优化
```

---

### 2. 幻觉（Hallucination）

LLM 生成的内容与事实不符，但看起来很可信。

**类型：**

- **事实性幻觉：** 捏造不存在的数据/引用
- **忠实性幻觉：** 答案与提供的上下文矛盾

**缓解方案：**

- RAG 提供事实依据
- 温度设为 0（减少随机性）
- 要求模型引用来源
- 输出验证（LLM 二次审查）
- Fine-tuning 纠正领域错误

---

### 3. 上下文窗口管理


| 策略       | 说明               |
| -------- | ---------------- |
| 滑动窗口     | 保留最近 N 轮对话       |
| 摘要压缩     | 定期将历史对话压缩为摘要     |
| Token 计数 | 超出阈值前主动压缩        |
| 选择性记忆    | 用向量检索相关历史，而非全部载入 |


---

### 4. 流式输出（Streaming）

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True
)
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content: print(content, end="", flush=True)
```

**意义：** 首 token 延迟低，用户体验好（打字机效果）。

---

### 5. 推理优化


| 技术                             | 说明                         |
| ------------------------------ | -------------------------- |
| **量化（Quantization）**           | FP16→INT8/INT4，减少显存，轻微精度损失 |
| **KV Cache**                   | 缓存 Attention 的 K/V，避免重复计算  |
| **Flash Attention**            | 优化 I/O，Attention 计算更快      |
| **连续批处理**                      | 动态组合请求，提升 GPU 利用率          |
| **投机采样（Speculative Decoding）** | 小模型起草，大模型验证，提升速度           |
| **vLLM**                       | 开源高性能推理框架，PagedAttention   |


---

### 6. Prompt 注入攻击

攻击者通过精心构造输入，覆盖系统提示或欺骗模型执行恶意操作。

**防御：**

- 输入过滤（检测 prompt injection 模式）
- 权限最小化（工具调用只给必要权限）
- 输出验证（不直接执行 LLM 输出）
- 使用 Guardrails（NeMo Guardrails、LlamaGuard）

---

### 7. 常用开源工具栈


| 类别        | 工具                                         |
| --------- | ------------------------------------------ |
| LLM 框架    | LangChain、LlamaIndex、Haystack              |
| Agent 框架  | LangGraph、AutoGen、CrewAI                   |
| 向量数据库     | Chroma、Milvus、Qdrant、Weaviate              |
| Embedding | BGE、text-embedding-3、M3E                   |
| 推理框架      | vLLM、Ollama、TGI（Text Generation Inference） |
| 评估        | RAGAS、DeepEval、PromptFlow                  |
| 监控        | LangSmith、Langfuse、Phoenix                 |
| 微调        | LLaMA-Factory、Axolotl、Unsloth              |
| 部署        | FastAPI + vLLM、BentoML、Ray Serve           |


---

### 8. RAG vs Agent 选型


| 场景         | 推荐                        |
| ---------- | ------------------------- |
| 文档问答、知识检索  | RAG                       |
| 多步骤任务、需要工具 | Agent                     |
| 需要实时数据     | Agent（带搜索工具）              |
| 低延迟场景      | RAG（Agent 多轮开销大）          |
| 高风险操作      | Agent + Human-in-the-loop |
| 复杂工作流      | LangGraph / 自定义编排         |


---

*文档共计 80+ 题，持续更新。*