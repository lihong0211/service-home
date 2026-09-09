# RAG 效果评测层（RAGAS 测试集生成 + 批量回归评测）

> 文件：`service/ai/rag_eval.py`（577 行）、`model/ai/rag_testset.py`
> 关联改动：`service/ai/rag.py`（`full_contexts` 字段、trace 截断上限）、`service/ai/rag_enhance.py`（`build_rag_answer_prompt` 完整性优化）
> 生成日期：2026-08-18

---

## 第一部分：背景演进

**问题背景**

RAG（检索增强生成）系统上线之后最大的痛点不是"能不能跑"，而是"效果好不好、有没有退步"。传统做法要么靠人工抽查几条问答"看着还行"，要么完全没有量化手段——改一次检索参数、换一次 embedding 模型、调一次 prompt，都无法确认是变好还是变差。这种"凭感觉"的评测方式在团队协作和持续迭代中会迅速失控：没有基准就没法做回归测试，没有回归测试就不敢随便优化。

RAG 效果的量化难点在于它是一个多环节的复合系统——检索质量（有没有召回相关片段）和生成质量（有没有忠实于片段、有没有正面回答问题）需要分开衡量，而且理想答案（ground truth）往往需要人工撰写，成本很高。

**RAGAS 框架简介**

[RAGAS](https://github.com/explodinggpt/ragas)（RAG Assessment）是目前 RAG 评测领域的主流开源框架，核心思路是"用 LLM 当裁判"（LLM-as-judge），把评测指标拆成两类：

- **不需要标准答案（reference-free）**：
  - `faithfulness`（忠实性）：答案中的论断是否都能在检索到的上下文里找到依据，衡量模型有没有"编造"。
  - `answer_relevancy`（答案相关性）：答案是否正面回应了问题本身，而不是答非所问。
- **需要标准答案（reference-based）**：
  - `context_precision`（上下文精确率）：检索到的片段里真正跟问题相关的占比，衡量检索有没有引入噪音。
  - `context_recall`（上下文召回率）：回答问题所需的信息是否都被检索到了，衡量检索有没有漏召。
  - `answer_correctness`（答案正确性）：生成的答案跟标准答案的语义匹配程度。

RAGAS 同时提供 `TestsetGenerator`，能基于知识库原文自动合成"问题-上下文-标准答案"三元组测试集，省去人工编写 ground truth 的成本。

**演进脉络**（按 commit 顺序）

| 阶段 | commit | 内容 |
|------|--------|------|
| 单次评测 | `7debf32` feat: RAG 效果评测接入 RAGAS | 新增 `/ai/rag/evaluate`，跑一次真实 RAG 或直接传现成问答对打分 |
| 测试集自动化 | `25c4459` feat: RAGAS 测试集自动生成 + 批量回归评测 | 新增 `rag_testset_item` 表 + 测试集生成/批量评测接口 |
| 中文化与可维护性 | `763f832` feat: RAG 测试集中文生成/CRUD + 回答完整性优化 | 测试集生成改为中文输出，补齐单条增/改/删接口，RAG 回答 prompt 增加完整性要求 |

**本模块的定位**

`rag_eval.py` 是 RAG 系统的"质检层"：向上通过 HTTP 接口暴露单次评测、测试集管理、批量回归评测三类能力；向下依赖 `service/ai/rag.py` 的 `rag_chat()` 拿到真实的检索片段与生成答案，并复用项目已有的 DashScope（阿里云百炼，OpenAI 兼容协议）配置作为 evaluator LLM/embeddings，不需要额外申请 OpenAI Key。它不参与线上问答链路，只在开发迭代和回归验证时被调用。

---

## 第二部分：架构剖析

**整体分层**

```
┌─────────────────────────────────────────────────────────┐
│ 测试集生成（合成数据）                                      │
│   _fetch_kb_chunks → _generate_testset → generate_testset_api │
│   落库：RagTestsetItem（rag_testset_item 表）              │
│   + 单条 CRUD：create/update/delete/list_testset_item_api  │
├─────────────────────────────────────────────────────────┤
│ 单次评测 / 批量回归评测执行                                  │
│   evaluate_sample（核心打分函数，RAGAS 5 项指标）            │
│   evaluate_rag_api（单次，POST /ai/rag/evaluate）           │
│   _batch_evaluate_async + batch_evaluate_api（批量回归）     │
├─────────────────────────────────────────────────────────┤
│ 结果存储                                                    │
│   测试集：落库持久化（rag_testset_item）                     │
│   评测结果：不落库，实时计算 + 接口返回（含 avg/min 汇总）      │
└─────────────────────────────────────────────────────────┘
```

评测结果本身是"无状态"的——每次调用 `/ai/rag/testset/evaluate` 都是实时跑一遍 RAG + RAGAS 打分，返回给调用方，不落库存历史。这是刻意的简化：当前阶段回归评测是"改动后手动跑一次看数字"，还没有到需要跟踪历史趋势曲线的阶段，避免过早引入一张历史评测结果表。

**核心数据流：从知识库文档到回归评测报告**

```
① 知识库已有分段（knowledge_base_document / knowledge_base_segment，见 03_knowledge_base.md）
      │
      ▼ _fetch_kb_chunks(kb_id)：取全部分段原文，超过 MAX_TESTSET_CHUNKS(10) 随机采样
      │
      ▼ _generate_testset(kb_id, size)：
      │    1. 分段文本 → KnowledgeGraph 节点（NodeType.CHUNK）
      │    2. apply_transforms(default_transforms_for_prechunked) 补全实体关系，构建知识图谱
      │    3. default_query_distribution(llm, kg) 按知识图谱实际情况决定启用哪些 synthesizer
      │    4. 用 _get_zh_prompts_by_synthesizer 缓存的中文 prompt 替换默认英文模板
      │    5. TestsetGenerator.generate() 产出 DataFrame（user_input/reference/reference_contexts/synthesizer_name）
      │
      ▼ generate_testset_api：落库为 RagTestsetItem（question/ground_truth/source_context/synthesizer）
      │
      (人工可选：create/update/delete_testset_item_api 修正或补充数据)
      │
      ▼ batch_evaluate_api(kb_id, testsetIds?)：
      │    对每条测试集问题依次：rag_chat() 拿真实答案+检索片段 → evaluate_sample() 打 5 项分
      │
      ▼ 返回逐条结果 + summary（各指标 avg）+ summaryMin（各指标 min）
           用于对比"这次改动前后，指标有没有整体退步 / 有没有某条被拖到很低"
```

**关键设计原则**

1. **evaluator 与业务 LLM 复用同一套凭据**：`_get_evaluator()` 用项目已有的 DashScope 配置（`DEFAULT_CHAT_MODEL=qwen-turbo`、`DEFAULT_EMBEDDING_MODEL=text-embedding-v4`）初始化 RAGAS 的 `llm_factory`/`embedding_factory`，不引入新的第三方依赖或密钥管理。
2. **失败隔离到最小粒度**：单个指标失败不拖垮整批打分（`evaluate_sample` 内部逐指标 try/except），单条测试集失败不拖垮整批回归评测（`_batch_evaluate_async` 内部逐条 try/except）。
3. **合成数据只作起点，不当金标准**：模块注释和接口文档反复强调 `_generate_testset` 产出的 `ground_truth` 是模型合成的，正式当团队基准前建议人工过一遍——这也是为什么要单独提供 `create/update/delete_testset_item_api` 这套人工介入的 CRUD。
4. **中文优先**：知识库源文档是中文场景，因此测试集生成、prompt 都做了针对性中文适配，而不是直接使用 RAGAS 默认的英文模板。

**与行业标准方案对比**

| 维度 | RAGAS（本项目采用） | TruLens | DeepEval |
|------|---------------------|---------|----------|
| 评测方法论 | LLM-as-judge，reference-free + reference-based 指标分离 | LLM-as-judge + Feedback Functions，侧重可观测性回路 | LLM-as-judge，pytest 风格断言式评测 |
| 测试集生成 | 内置 `TestsetGenerator`，基于知识图谱自动合成多跳/单跳问题 | 无内置生成器，依赖外部数据集 | 内置 Synthesizer，但生态更偏 OpenAI |
| 生态集成 | LangChain/LlamaIndex 原生集成 | 深度绑定可观测性 Dashboard（TruLens app） | 与 CI/CD（pytest）集成更自然 |
| LLM 供应商依赖 | 通过 OpenAI 兼容接口即可接入任意供应商（本项目接入 DashScope） | 官方示例偏 OpenAI，但可自定义 provider | 官方示例偏 OpenAI，可自定义 |
| 部署形态 | 纯 Python 库，无需额外服务 | 需要额外起 Dashboard 服务做可视化 | 纯 Python 库 + CLI，可接 Confident AI 云端 |
| 中文支持 | 无原生中文 prompt，需手动 `adapt_prompts` 翻译（本项目已做） | 同样需要自行适配 | 同样需要自行适配 |

**选型建议**：三者内核都是 LLM-as-judge，差异主要在生态集成方式和测试集生成能力。本项目选择 RAGAS 主要因为：① 它的 `TestsetGenerator` 能直接基于已有知识库分段自动合成中文测试集，省去人工编写 ground truth；② 纯 Python 库形态，不需要额外部署可视化服务，符合本项目"轻量、自建路由层"的架构风格；③ 通过 OpenAI 兼容接口即可无缝接入已有的 DashScope 配置。如果后续需要团队级的评测历史看板/趋势追踪，可以考虑接入 TruLens 或让当前接口的评测结果落库并配合已有的可观测性模块（见 `service/ai/rag.py` 的 trace 采集）。

---

## 第三部分：代码实现深度解析

**核心函数/类清单**

| 函数/类 | 作用 |
|---------|------|
| `_get_evaluator()` | 惰性初始化 evaluator LLM/embeddings，模块级单例，跨请求复用 |
| `_get_zh_prompts_by_synthesizer(llm)` | 把默认 query synthesizer 的英文 prompt 模板翻译成中文并按类名缓存 |
| `evaluate_sample(query, answer, contexts, ground_truth=None)` | 核心打分函数，对 5 项 RAGAS 指标并发打分，逐指标容错 |
| `evaluate_rag_api(request)` | `POST /ai/rag/evaluate`，单次评测入口 |
| `_fetch_kb_chunks(kb_id, limit=MAX_TESTSET_CHUNKS)` | 取知识库全部分段原文，超限随机采样 |
| `_generate_testset(kb_id, size)` | 调 RAGAS 生成测试集的核心逻辑：建知识图谱、跑 transforms、中文化 synthesizer |
| `generate_testset_api` / `list_testset_api` / `create_testset_item_api` / `update_testset_item_api` / `delete_testset_item_api` | 测试集 CRUD 接口 |
| `_batch_evaluate_async` / `batch_evaluate_api` | `POST /ai/rag/testset/evaluate`，批量回归评测 |

**关键实现细节**

1. **测试集自动生成的 prompt 策略**（`_get_zh_prompts_by_synthesizer` + `_generate_testset`）

   RAGAS 默认的 query synthesizer（`single_hop_specific`/`multi_hop_abstract`/`multi_hop_specific`）自带英文 few-shot 示例，即便源文档是中文，仅靠一句中文提示压不住这个偏置。解决方式是用 `synthesizer.adapt_prompts("chinese", llm=llm)` 把模板连示例一起整体翻译成中文，按 synthesizer 类名（`type(synthesizer).__name__`）缓存到模块级字典 `_zh_prompts_by_synthesizer`，避免每次请求都重新翻译一遍（翻译本身也要调 LLM，有成本）。

   `_generate_testset` 没有直接调用更省事的 `TestsetGenerator.generate_with_chunks`（它内部用的是未翻译的默认 distribution），而是手动重复它内部构建知识图谱的步骤：把分段文本包成 `Node(type=NodeType.CHUNK, ...)`，用 `apply_transforms(kg, default_transforms_for_prechunked(...))` 补全实体关系，再调 `default_query_distribution(llm, kg)`。这样做的原因是：`default_query_distribution` 是否启用 `multi_hop` 系列 synthesizer，取决于知识图谱里有没有构建出可用的实体关系簇（cluster）——chunk 数少或语义稀疏时经常没有簇。只有拿到真正基于当前知识库构建出的 `kg` 去调这个函数，才能跟 `generate_with_chunks` 内部行为保持一致；如果图省事直接复用一份写死的 distribution，会导致 `multi_hop` synthesizer 在没有簇的图谱上直接报错（`"No relationships match the provided condition. Cannot form clusters."`，代码注释里标注为"已实测触发"）。

   另一个关键坑点记录在函数 docstring 里：`_generate_testset` 被刻意写成**同步函数、直接调用**，不包一层 `async`/`anyio.from_thread.run`。原因是 `generate`/`apply_transforms` 底层会自己起一个 event loop 跑内部的异步 LLM 调用；如果从已经在跑的 event loop（路由层 `anyio.from_thread.run` 桥接出来的那个）里再调用它，会触发"loop 套 loop"死锁——实测表现是直接卡死、不报错，非常隐蔽。`generate_testset_api` 内部用 `_generate_testset(kb_id, size)` 直接同步调用，与 `rag_chat` 的调用方式保持一致。

2. **CRUD 管理**（`RagTestsetItem` + `create/update/delete_testset_item_api`）

   测试集数据落在 `rag_testset_item` 表（`model/ai/rag_testset.py`），字段包括 `kb_id`/`kb_name`（冗余存储，方便列表展示不用 join）、`question`/`ground_truth`/`source_context`（生成依据的原文片段，供人工核对）、`synthesizer`（生成方式标记）。

   自动生成的数据 `synthesizer` 字段是 `single_hop_specific`/`multi_hop_abstract` 等 RAGAS 内部类名；`create_testset_item_api` 手动补录数据时固定写死 `synthesizer="manual"`，用来跟自动生成的数据区分开，批量评测和展示逻辑不需要为此分叉。`update_testset_item_api` 只更新请求体里显式传了的字段（用 `"question" in data` 判断存在性，而非 `data.get("question")` 判断真值），没传的字段维持原样，用于人工修正 RAGAS 生成的问答对。`delete_testset_item_api` 是硬删——测试集数据没有下游外键引用，不需要软删。

3. **批量回归评测怎么对比历史结果**（`_batch_evaluate_async` + `batch_evaluate_api`）

   本模块**没有**做"跟历史评测结果自动 diff"这种存储+比对逻辑，"回归"依赖的是使用方自己手动对比两次返回值的 `summary`。具体机制：`batch_evaluate_api` 接收 `kb_id` + 可选的 `testsetIds`（不传则跑该知识库全部测试集），对每条测试集问题依次调用 `rag_chat()` 得到当前代码/配置下的真实答案，再用 `evaluate_sample()` 打分。汇总时同时计算 `summary`（每项指标算失败的记录会被 `None` 值跳过，不计入平均，避免拉低整体分数）和 `summaryMin`（每项指标的最小值）——之所以两个都要，是因为"平均分再高，只要有一条烂得离谱，也说明某类问题没处理好，光看平均容易被掩盖"。使用方在改动代码前后各跑一次这个接口，人工比对两次的 `summary`/`summaryMin` 数值即为回归结果。

   `_batch_evaluate_async` 刻意**串行**而非并发跑每条测试集（`for it in items` 循环 + `await`，不是 `asyncio.gather`）：每条本身就要好几次 LLM 调用（RAG 检索生成 + 5 项 RAGAS 指标各一次 LLM 调用），并发太高容易触发 DashScope 限流，这里用总耗时换稳定性——接口文档里标注单条约 15-20 秒，调用方需要给足够长的超时。

4. **"回答完整性优化"具体做了什么**（`service/ai/rag_enhance.py` 的 `build_rag_answer_prompt`，非 `rag_eval.py` 本体但由本轮回归评测驱动发现）

   commit `763f832` 在 RAG 回答生成 prompt 中新增一句：

   > "回答时请尽量完整覆盖参考资料中与问题相关的信息点，不要只回应字面问题而漏掉资料里提到的限制条件、注意事项或补充建议。"

   这是引入 RAGAS 回归评测后，通过观察 `context_recall`/`answer_correctness` 等指标发现的真实问题——模型倾向于只字面回答问题本身，容易漏掉上下文里提到的限制条件、注意事项等衍生信息，导致召回类指标偏低。这条 prompt 调整本身没有改动 `rag_eval.py`，但体现了本模块"评测驱动优化"的实际价值：先有量化指标，才能定位到具体的 prompt 缺陷并验证修复效果。

**设计决策与取舍**

1. **测试集分段数上限 `MAX_TESTSET_CHUNKS=10`，超限随机采样而非取前 N 个**：生成阶段要对每个分段跑 Summary/Themes/NER 等好几轮 LLM 抽取，分段数一多耗时会指数级上升——实测 30 个分段生成 3 条题目跑了 3 分钟以上还没完成，10 个分段生成同样数量能在 1~2 分钟内跑完。用随机采样而不是截取文档开头的前 N 段，是为了避免生成的测试集只覆盖文档开头，跑几次能大致覆盖到全文档的不同角落。这是"交互式可用性"与"测试集覆盖完整度"之间的取舍，代价是同一个知识库多次生成的测试集不完全一致。

2. **评测结果不落库，只做实时计算 + 接口返回**：当前定位是"改动前后手动跑一遍看数字"，还没有到需要长期趋势追踪、多次评测历史对比的阶段。取舍在于：省去了一张评测历史表和相应的查询/清理逻辑，代价是无法在系统内直接查看"上一次评测是什么时候、结果如何"，需要使用方自己保存接口返回值做外部比对。

3. **单次评测支持"跳过重新跑 RAG"（直接传 `answer`+`contexts`）**：`evaluate_rag_api` 允许调用方要么传 `kb_id`/`kb_name` + `question` 触发一次真实 RAG 调用，要么直接传现成的 `answer`+`contexts` 跳过检索和生成步骤直接打分。这个分支是为了支持"对已经产出的问答记录做事后评测"场景（比如从可观测性 trace 里挑一条历史记录出来复核），不强制每次评测都要重新跑一遍完整 RAG 流程。

4. **`_score` 包一层零参 async 可调用而非提前建好 coroutine**：`evaluate_sample` 里 `_score(name, run)` 的 `run` 参数设计成 `lambda: XXXMetric(...).ascore(...)` 这种零参可调用，而不是直接传入 `XXXMetric(...).ascore(...)` 构造好的 coroutine 对象。原因是要让"构造期报错"（比如指标类初始化参数传错）也能被 `try/except` 兜住——如果直接传 coroutine 对象，构造过程中的异常会在 `asyncio.gather` 收集任务列表之前就直接抛出，导致同一批里其他本来能正常算出结果的指标也一起失败。

**代码质量说明**：`_generate_testset` 中 `from langchain_core.documents import Document` 这行导入在当前实现里未被实际使用（生成逻辑已改为手动构建 `KnowledgeGraph`/`Node`，不再走 `Document` 包装），属于 `763f832` 重构后遗留的死代码，不影响功能。

---

## 第四部分：应用场景与实战

**核心使用场景**

1. **知识库新建/文档更新后，快速验证 RAG 效果**：先用 `POST /ai/rag/testset/generate` 从知识库分段自动合成一批中文测试集，再用 `POST /ai/rag/testset/evaluate` 批量跑一遍，看 5 项指标的整体水平。
2. **改动检索参数、embedding 模型、生成 prompt 前后做回归对比**：对同一批已落库的测试集（`testsetIds` 不传即跑全部）在改动前后各跑一次批量评测，比对 `summary`/`summaryMin` 有没有下降。
3. **单条问答的即席评测**：不依赖测试集，直接对某个 `question`（可选 `ground_truth`）跑一次 `/ai/rag/evaluate`，用于调试单个 case。
4. **人工修正合成测试集**：RAGAS 自动生成的 `ground_truth` 是模型合成的，可能存在偏差；用 `PUT /ai/rag/testset/{item_id}` 人工修正后再纳入长期回归基准。

**快速上手**

环境依赖：

```
ragas==0.4.3
langchain>=0.3.0（TestsetGenerator 依赖 langchain_core.documents）
DashScope API Key（复用 config/ai.py 的 dashscope_api_key()，无需单独申请 OpenAI Key）
```

示例一：单次评测（直接传现成问答对，跳过重新跑 RAG）

```bash
curl -X POST http://localhost:3000/ai/rag/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "退款需要多久到账？",
    "answer": "退款一般1-3个工作日到账，节假日可能延迟。",
    "contexts": ["退款处理时效：正常情况下1-3个工作日内原路退回，遇法定节假日顺延。"],
    "ground_truth": "退款1-3个工作日到账，节假日顺延。"
  }'
# 返回 scores 中含 faithfulness、answer_relevancy、context_precision、context_recall、answer_correctness
```

示例二：从知识库自动生成中文测试集

```bash
curl -X POST http://localhost:3000/ai/rag/testset/generate \
  -H "Content-Type: application/json" \
  -d '{"kb_id": 21, "size": 5}'
# 慢接口，10 个分段生成几条题目约 1~2 分钟；前端需给足超时（建议 >= 5 分钟）
```

示例三：批量回归评测

```bash
curl -X POST http://localhost:3000/ai/rag/testset/evaluate \
  -H "Content-Type: application/json" \
  -d '{"kb_id": 21}'
# 慢接口：题目数 × 单条耗时（约 15-20s），返回 summary（各指标平均分）+ summaryMin（各指标最低分）
```

**常见问题排查**

| 现象 | 原因 | 排查方向 |
|------|------|----------|
| `POST /ai/rag/testset/generate` 卡死不报错 | 从已在运行的 event loop 里间接调用了内部同步生成逻辑，触发 loop 套 loop 死锁 | 确认调用路径没有再包一层 `anyio.from_thread.run`；`generate_testset_api` 内部已直接同步调用 `_generate_testset` |
| 生成的测试集全是英文 | 使用了未经中文化的 synthesizer prompt（旧版本行为） | 确认走的是当前版本代码，`_get_zh_prompts_by_synthesizer` 已对 `single_hop_specific`/`multi_hop_abstract`/`multi_hop_specific` 做了中文翻译并生效 |
| 生成报错 `No relationships match the provided condition. Cannot form clusters.` | 知识库分段数太少或语义过于稀疏，无法构建出 multi_hop 所需的实体关系簇 | 增加知识库分段数量，或接受该知识库只能生成 single_hop 类型的题目 |
| `scores` 里某项指标是 `null` 且带 `_errors` | 该指标在 RAGAS 内部结构化输出解析失败（依赖模型的 function calling 能力） | 查看 `_errors` 字段里的具体报错信息；确认 evaluator 模型（`DEFAULT_CHAT_MODEL`）支持 function calling |
| `context_precision`/`context_recall`/`answer_correctness` 一直是 `null` | 请求没有传 `ground_truth` | 这三项指标依赖标准答案，未传 `ground_truth` 时按设计直接跳过，不是 bug |
| 本机常驻后台服务跑测试集生成特别慢（观测到单次 LLM 调用 240s+） | macOS 对 launchd 常驻后台进程的资源节流（App Nap 一类机制） | 这类长耗时批量 LLM 任务改为前台直接跑（`python main.py`），实测同样操作前台约 10s 完成 |
| 没有检索到任何上下文报错 | 知识库为空或该问题在当前知识库检索不到相关内容 | 先确认知识库已完成向量化（见 `03_knowledge_base.md`），或换一个更贴近知识库内容的问题 |

---

## 第五部分：评估与展望

**优势**

- 复用项目已有的 DashScope 配置，接入成本低，不需要额外申请 OpenAI Key 或搭建评测服务。
- 测试集生成与批量回归评测打通，形成"自动合成基准 → 反复回归验证"的闭环，而不是一次性看一眼就扔的评测。
- 失败隔离粒度做到单指标、单条目级别，批量评测不会因为个别 LLM 调用失败而整体中断。
- 针对中文场景做了专门适配（synthesizer prompt 翻译），弥补了 RAGAS 默认英文模板在中文知识库上的短板。

**局限与技术债务**

- 评测结果不落库，无法在系统内直接查看历史趋势，依赖使用方自己保存和比对接口返回值。
- 批量评测串行执行，题目数一多耗时线性增长（约 15-20s/条），没有并发/限流平衡的中间方案。
- 合成测试集的 `ground_truth` 未经人工审核就可以直接用于回归评测，存在"用不准的基准衡量效果"的风险，虽然有 CRUD 接口支持人工修正，但没有强制的审核流程。
- `_generate_testset` 中存在未使用的 `Document` 导入（见第三部分"代码质量说明"），属于重构遗留，不影响功能但可以清理。
- 本机常驻后台服务对这类长耗时批量 LLM 任务有明显的系统级节流问题（App Nap），目前只能靠"改前台跑"规避，没有从架构上解决（比如换成不受节流影响的常驻方式）。

**演进建议**

- 若团队协作规模扩大，可考虑给评测结果增加落库能力（复用 `AgentTrace` 类似的观测表结构），支持趋势看板和多次评测历史对比。
- 批量评测可探索"小并发 + 限流退避"替代当前的完全串行，在稳定性和速度之间找更优的平衡点。
- 测试集生成/审核流程可以增加"待审核"状态字段，区分"RAGAS 生成未审"和"人工已核对"两种可信度级别，批量评测汇总时可以分别展示。

**行业前沿**

RAGAS 生态仍在快速迭代（本项目锁定 `ragas==0.4.3`），近期趋势包括：更细粒度的多跳（multi-hop）测试集合成能力、对非 OpenAI LLM 供应商的 function calling 兼容性持续改善、以及与 LangSmith/Langfuse 等可观测性平台的评测结果回流集成。本项目目前的可观测性模块（trace 采集、隐式反馈）与 RAGAS 评测尚未打通，是未来可以探索的方向——把批量回归评测的结果和线上真实流量的 trace 数据关联起来，用真实反馈校准合成测试集的可信度。

---

## 变更记录

- 2026-08-18：新建文档，因 `rag_eval.py` 功能增长（RAGAS 测试集生成、批量回归评测）独立成篇。
