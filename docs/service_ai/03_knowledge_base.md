# 知识库管理层（Knowledge Base）

> 文件：`service/ai/knowledge.py`（1850 行）、`service/ai/files.py`  
> 生成日期：2026-02-26，2026-07-31 更新（分段策略从固定长度扩展为 4 种）

---

## 第一部分：技术背景与演进

**问题背景**

向量检索只处理浮点向量，而业务侧需要的是上传一份 PDF 然后就能被问答——这中间有大量"脏活"：解析文档格式、分段切块、写数据库、生成向量、处理上传重复等。知识库管理层将这些工作封装成一个面向业务的操作界面，让使用者不必了解向量库的底层细节。

**核心概念**

- **知识库（Knowledge Base）**：业务层概念，包含一组文档及其对应的向量库，有名称、描述、解析策略等元信息。
- **分段（Segment/Chunk）**：文档被切成的最小检索单元，大小由 `chunk_size`（默认 1000 字符）和 `chunk_overlap`（默认 200 字符重叠）控制。
- **向量化（Vectorize）**：将分段文本批量调 Embedding API 生成向量并写入 FAISS 索引的过程，是最耗时的步骤。

**演进脉络**

| 阶段 | 方案 | 特点 |
|------|------|------|
| 早期 | 手动准备文本 → 直接调向量库 API | 用户门槛高，只支持纯文本 |
| 文档解析库兴起 | PDFMiner、python-docx 等专用解析库 | 支持多格式，但格式适配碎片化 |
| 知识库平台化 | LangChain/LlamaIndex Document Loaders | 统一接口，生态丰富，抽象层厚 |
| **本项目** | 自研多格式解析 + 三表结构管理 | 轻量、可控、与业务数据库深度集成 |

支持格式的演进：`TXT → PDF → DOCX → PPTX → Excel → 图片（OCR）`，并通过 LibreOffice 支持旧版 `.doc`/`.ppt`。

**本模块的定位**

知识库层是"文档到向量"管道的统一入口，向上为 HTTP 路由层提供完整的知识库 CRUD 接口，向下委托 `vector_db.py` 完成向量操作。它维护三张核心 MySQL 表：`knowledge_base`（库元信息）、`knowledge_base_document`（文档记录）、`knowledge_base_segment`（分段文本），形成完整的文档生命周期管理。

---

## 第二部分：架构剖析

**三表结构**

```
knowledge_base
  ├── id, name, description
  ├── vector_db_id (FK → vector_db.id)
  ├── parsing_strategy, chunking_strategy
  └── create_at, update_at

knowledge_base_document
  ├── knowledge_base_id (FK)
  ├── file_name, path (磁盘路径)
  ├── file_id, status
  └── create_at

knowledge_base_segment
  ├── document_id (FK)
  ├── text (分段文本)
  ├── index (段序号)
  └── segment_metadata (JSON，含 source/page 等)
```

**文档上传到可检索的完整流程**

```
POST /ai/knowledge-base/upload (multipart)
      │
      ▼ 1. 校验文件类型、处理重名（同名文件先删旧再上传）
      │
      ▼ 2. 保存文件到 data/knowledge_base/kb_{知识库name}/{timestamp}_{filename}
      │
      ▼ 3. parse_file_to_documents(file_path, filename, chunk_size, chunk_overlap,
      │                            chunking_strategy, hierarchy_level, retain_hierarchy)
      │     按扩展名路由到对应解析函数，解析工具 + 分段策略见下表
      │
      ▼ 4. _add_document_and_segments_to_kb()
      │     写 knowledge_base_document + knowledge_base_segment（父子切片时先插父块拿真实 id 再插子块写 parent_id）
      │     此时还未生成向量
      │
      (到此为止，文档已落库但不可检索)

POST /ai/knowledge-base/vectorize
      │
      ▼ vectorize_knowledge_base(kb_id)
            ├─ 已有 vector_db_id？
            │    ├─ 无变化 → 跳过
            │    ├─ 只有新增 → append_documents_batch（增量 embed）
            │    └─ 有删除 → _rebuild_vector_db_index（全量重建）
            └─ 首次：create_vector_db + 写 MySQL + 回填 kb.vector_db_id
      （父子切片时：父块本身不建索引，只作为子块的上下文来源；子块的向量算自己的文本，
        但返回给 LLM 的 text 换成父块正文——用 embedding_text 字段实现，见 vector_db_qdrant.py）
```

### 文件类型 → 解析工具 / 分段策略速查表

`chunking_strategy` 取值：`fixed`（默认，固定长度）/ `structure`（标题层级）/ `hierarchy`（父子切片）/ `semantic`（语义切片）。下表是默认 `parsing_strategy=fast`（现有轻量解析）下的支持情况；`parsing_strategy=precise` 时 PDF/DOCX/PPTX/XLSX/XLS/图片改走 MinerU（见本节最后一小节），structure/hierarchy 对 PDF 的 ⚠️ 回退会变成实际支持——但没有真机验证过效果。

| 文件类型 | 解析工具/库 | fixed | structure | hierarchy | semantic |
|---|---|---|---|---|---|
| PDF `.pdf` | `PyPDF2`（`PdfReader`，按页提取文字层） | ✅ 每页内定长切分 | ⚠️ 回退 fixed（PDF 无可靠标题结构） | ⚠️ 回退 fixed | ✅ 每页内按句子相似度切分 |
| DOCX `.docx` | `python-docx`（段落 + 表格提取） | ✅ 段落/表格边界累加到接近 chunk_size 再切（保留自然边界，不是纯定长） | ✅ 按 `Heading 1..N` / `Title` 样式识别层级，每段带 `heading_path` breadcrumb | ✅ 按 `hierarchy_level` 生成父块，父块内再切子块，写 `parent_id` | ✅ 忽略标题结构，对全文按语义相似度切 |
| DOC `.doc` | LibreOffice 转 `.docx` 后按上一行处理 | 同 DOCX | 同 DOCX | 同 DOCX | 同 DOCX |
| PPTX `.pptx` | `python-pptx`（按幻灯片提取，单页 < 40 字时并入下一页，避免空标题页检索得分虚高） | ✅ 唯一支持的策略 | ⚠️ 忽略 `chunking_strategy`，恒定走 fixed 那一套逻辑 | ⚠️ 同上 | ⚠️ 同上（`parse_file_to_documents` 对 `.ppt/.pptx` 根本不传 chunking_strategy 参数） |
| PPT `.ppt` | LibreOffice 转 `.pptx` 后按上一行处理 | 同 PPTX | 同 PPTX | 同 PPTX | 同 PPTX |
| TXT `.txt` | 内置 `open()` 读取全文 | ✅ | ⚠️ 回退 fixed（纯文本无标题概念） | ⚠️ 回退 fixed | ✅ |
| MD `.md` | 内置 `open()` 读取全文 | ✅ | ✅ 按 `#`~`######` 层级识别标题（空行分块，块首行是标题时视为 heading） | ✅ 同 DOCX 的父子逻辑，标题来源换成 `#` 层级 | ✅ |
| Excel `.xlsx`/`.xls` | `openpyxl` / `xlrd`（按工作表逐行转文本，`\t` 拼接） | ✅ | ⚠️ 回退 fixed（表格无标题层级概念） | ⚠️ 回退 fixed | ✅ |
| 图片 `.jpg/.png/.gif/.webp/.bmp` | `pytesseract`（Tesseract OCR；先限制最长边 2000px 缩放，`--psm 6` 单列文本块模式提速） | ✅（OCR 文本按定长切） | ⚠️ 回退 fixed | ⚠️ 回退 fixed | ✅（OCR 文本按语义切） |

**语义切片（semantic）的通用兜底**：按 `_split_sentences` 用中英文句末标点/换行切句，逐句调 `get_embedding` 算相邻句相似度，低于阈值（0.6）处断开分组；文本超过 3 万字符时（逐句 embedding 成本过高）或整段只有 0~1 句时，自动回退固定长度切片，不会因为文本过长而卡死或狂调 embedding 接口。

**父子切片（hierarchy）只对 DOCX/MD 生效**：`hierarchy_level`（默认 3）控制哪一级标题算父块边界；父块的 `segment_metadata.is_parent = true`，子块的 `parent_id` 指向父块的真实 `knowledge_base_segment.id`。`retain_hierarchy=false` 时子块不写 `heading_path` metadata（父块本身不受影响）。

**关键设计原则**

- **上传与向量化解耦**：上传只解析落库，向量化是独立步骤。这允许用户先上传多个文件、调整分段参数、预览分段效果，最后一次性向量化，避免反复调 Embedding API。
- **增量智能向量化**：`vectorize_knowledge_base` 会比对当前分段 ID 集合与已向量化 ID 集合，只有新增才增量 embed，有删除才全量重建。
- **同名文件覆盖**：批量上传时，同名文件（忽略大小写）会删除旧文档的分段和磁盘文件，重新入库，避免重复段落污染向量库。
- **文件保留原则**：解析失败时文件仍保存到磁盘，创建文档记录（分段数为 0），用户后续可重新分段，不因解析错误丢失原文件。

**与行业标准方案对比**

| 维度 | 本地实现 | LangChain Document Loaders | LlamaIndex |
|------|---------|---------------------------|-----------|
| 格式支持 | PDF/DOCX/PPTX/TXT/MD/Excel/图片 | 100+ 格式 Loader | 50+ 格式 Loader |
| 分段策略 | fixed/structure/hierarchy/semantic 四选一（自定义 chunk_size/overlap） | 多种策略可选 | 多种策略 + 语义分段 |
| 数据库集成 | 原生集成 MySQL 三表管理 | 无，需自行实现 | 无，需自行实现 |
| 上传与向量化解耦 | 是（分两步） | 否（通常一步完成） | 否 |
| 分段预览 | 支持（落库前可预览，落库后也可查看） | 不支持 | 不支持 |
| **选型建议** | 需要文档管理 UI、与业务库集成 | 快速原型、多格式批量导入 | 复杂文档结构、语义分段 |

---

## 第三部分：代码实现深度解析

**核心函数清单**

| 函数 | 作用 |
|------|------|
| `parse_file_to_documents(file_path, filename, chunk_size, chunk_overlap, chunking_strategy, hierarchy_level, retain_hierarchy)` | 按扩展名 + chunking_strategy 路由到对应解析函数，返回 `[{id, text, category, metadata?}]` |
| `_chunk_text(text, chunk_size, chunk_overlap)` | 固定长度分片核心算法 |
| `_merge_units_to_chunks(units, chunk_size, chunk_overlap)` | 段落/正文块累加到接近 chunk_size 再切，DOCX 固定分片、结构感知、父子切片共用 |
| `_semantic_chunk_text(text, chunk_size)` / `_split_sentences` / `_cosine_similarity` | 语义切片：逐句 embedding + 相邻相似度断句 |
| `_docx_units_with_headings` / `_md_units_with_headings` | 提取带标题层级的 units（DOCX 按 `Heading N` 样式，MD 按 `#` 数量） |
| `_build_heading_sections` / `_group_sections_for_hierarchy` / `_documents_from_heading_units` | 结构感知/父子切片的公共装配逻辑，供 DOCX 和 MD 复用 |
| `vectorize_knowledge_base(kb_id)` | 智能向量化：增量/全量重建自动判断；父子切片时子块的 embedding_text 用自身文本、text 换成父块正文 |
| `upload_knowledge_base_api()` | 多文件上传处理，含同名覆盖、格式校验、OCR 跳过选项、chunking_strategy 透传 |
| `execute_segments_api()` | 重新分段（支持调整 chunk_size/chunking_strategy 后重新切割已有文档） |
| `_add_document_and_segments_to_kb` / `_resegment_one_document` | 落库时按 `_local_parent_ref` 把父子切片的临时引用解析成真实 `parent_id`（父块必须排在其子块之前） |
| `_ocr_image_to_text(image_path)` | Tesseract OCR，含图片缩放优化（限制 2000px）和 PSM 6 快速模式 |

**分块算法细节**

```python
def _chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - chunk_overlap   # 下一块起点回退 overlap 个字符
    return chunks
```

`chunk_overlap` 保证相邻分段之间有 200 字符的重叠，防止关键信息被切断在两段之间无法被检索到。

**设计决策与取舍**

**决策 1：DOCX 合并全文后分片（不保留段落边界）**  
原因：保留段落边界会让短段落（如标题、单行表格行）单独成一个分段，检索时这些噪声段落会消耗 top-k 名额。合并全文后按字符数分片，段落密度更均匀。  
代价：跨段落的语义边界可能被切断，但 `chunk_overlap=200` 缓解了大部分情况。

**决策 2：图片 OCR 失败时保留文件（分段数为 0）**  
```python
except Exception as parse_err:
    out = _add_document_and_segments_to_kb(kb_id, fn, [], path=saved_path)
    documents_result.append({..., "segment_count": 0, "error": str(parse_err)})
```
原因：Tesseract 未安装时批量上传不应整体失败。文件保留后用户可安装 Tesseract 再调用"重新分段"接口。  
代价：用户可能不注意 segment_count=0 就直接向量化，导致该文件内容不可检索——接口响应中的 `error` 字段需要前端提醒用户。

**决策 3：`skip_ocr` 参数**  
批量上传图片时可选 `skip_ocr=true`，跳过 OCR 直接保存文件（分段数为 0），避免大批量图片上传时因 OCR 导致接口超时。后续可单独调用分段接口处理。

**决策 4：知识库与向量库解耦**  
知识库管理文档的原始文本和分段结构；向量库只管浮点向量和相似度检索。两者通过 `vector_db_id` 外键关联。好处是一个向量库可以被多个知识库或多个场景共用（如同一向量库同时服务 RAG 问答和相似度搜索 API）。

---

## 第四部分：应用场景与实战

**使用场景**

- 企业知识库建设：上传产品文档、FAQ、规章制度，自动分段向量化后支持 RAG 问答
- 文档分段效果调优：上传后不立即向量化，先预览分段，调整 `chunk_size` 参数，重新分段直到满意，最后向量化

**环境依赖**

```bash
pip install PyPDF2 python-docx python-pptx openpyxl xlrd pytesseract Pillow
# macOS OCR（可选）
brew install tesseract tesseract-lang
# LibreOffice（.doc/.ppt 转换，可选）
brew install --cask libreoffice
export DASHSCOPE_API_KEY=sk-xxx
```

**代码示例**

```python
from service.ai.knowledge import parse_file_to_documents, vectorize_knowledge_base

# 1. 解析文件为分段（不入库，仅预览）
docs = parse_file_to_documents("report.pdf", "report.pdf", chunk_size=800, chunk_overlap=150)
print(len(docs), "个分段")
print(docs[0])  # {"id": "p1", "text": "...", "category": "第1页"}

# 2. 向量化已有知识库（kb_id=1）
result = vectorize_knowledge_base(1)
print(result)
# {"vector_db_id": 5, "count": 128, "created": True, "incremental": False}
```

**常见问题**

- **`.doc` 文件解析失败**：需要安装 LibreOffice（`brew install --cask libreoffice`），否则抛出包含安装提示的 `ValueError`。
- **PDF 解析为空**：扫描版 PDF 没有文字层，`PyPDF2` 无法提取文本。解决方案：用 OCR 工具（如 Adobe、云服务）将扫描 PDF 转为可搜索 PDF 后重新上传。
- **向量化后检索不到内容**：检查 `knowledge_base.vector_db_id` 是否已填写（可调用 `/ai/knowledge-base/detail` 确认）；若为 null 说明向量化未完成。

---

## 第五部分：优缺点评估与未来展望

**优势**

- 上传与向量化解耦，支持分段预览调优，减少 Embedding API 浪费
- 增量向量化：只对新增分段调 API，大知识库追加文档成本低
- 同名文件自动覆盖，批量上传友好
- OCR 失败不影响整体上传，文件保留后续可重试

**已知局限**

- 结构感知/父子切片只对 DOCX/MD 生效；PDF/PPTX/Excel/图片没有可靠的标题结构，这两种策略对它们会静默回退成 fixed
- PPTX/PPT 完全不接受 `chunking_strategy` 参数（`parse_file_to_documents` 里这两个分支没有传），不管前端选了什么策略都固定用"按幻灯片 + 短页并入下一页"这一套逻辑
- PDF 仅用 `PyPDF2` 提取文字层，扫描版/图表型 PDF 无法处理
- DOCX 表格提取为纯文本（`" | ".join(cell.text)`），复杂表格格式丢失
- 语义切片是逐句调用 embedding API，文本很长时耗时较高（已加 3 万字符自动回退兜底，但仍比固定分片慢很多）
- 向量化为同步操作，大型知识库（千条以上分段）会导致 HTTP 请求超时

**演进建议**

- 短期：向量化改为异步任务（后台线程 + 进度轮询接口），解决大知识库超时问题
- 短期：PPTX/PPT 接入 `chunking_strategy`（至少支持 semantic，做法与 PDF/TXT 一致，把 `_chunk_text` 换成 `_chunk_text_by_strategy` 即可）
- 长期：MinerU 已接入（见下），后续可评估 MinerU 输出的 Markdown 标题层级质量、`hierarchy_level` 语义是否需要针对 MinerU 场景微调

### `parsing_strategy`：MinerU 可选解析方式（2026-07-31 已实现）

**背景**：MinerU（OpenDataLab）原生支持 PDF/DOCX/PPTX/XLSX/图片，统一转成带标题层级的 Markdown，能解决当前 PDF 无标题结构（structure/hierarchy 只能回退 fixed）和 PPTX 不支持 chunking_strategy 这两个短板。但它是重量级依赖（版面检测 + OCR + 表格/公式识别模型），不适合替换现有轻量解析（PyPDF2/python-docx/python-pptx/openpyxl），做成了**用户可选**的解析方式，不选就完全走原来那一套。

**实现方式**：

1. `parsing_strategy`（`knowledge_base.parsing_strategy`，取值 `fast`/`precise`，字段早就在 schema 里，之前从未被实际读取）：`fast`（默认）= 现有解析全套，一行没改；`precise` = 走 MinerU。
2. 新增独立文件 **`service/ai/knowledge_mineru.py`**，`knowledge.py` 里任何现有的 `_documents_from_pdf/_documents_from_docx/_documents_from_pptx/...` 一行没动。该文件通过 subprocess 调 `mineru` CLI（`mineru -p <文件> -o <临时目录> -b pipeline`，pipeline backend 纯 CPU）把文件转成 Markdown，再复用 `knowledge.py` 里的 `_md_units_with_headings`/`_documents_from_heading_units`/`_merge_units_to_chunks`/`_semantic_chunk_text` 解析标题层级和分段，返回跟现有解析函数一样的 `[{id, text, category, metadata?}]` 形状。
3. 接入点在 `parse_file_to_documents()` 顶部：新增 `parsing_strategy: str = "fast"` 参数，`precise` 且文件类型在 MinerU 覆盖范围内（`is_mineru_supported()`）时委托给 `knowledge_mineru.parse_file_to_documents_mineru(...)`，否则原有 if/elif 扩展名分派逻辑完全不变。`upload_knowledge_base_api`/`execute_segments_api`/`_resegment_one_document` 透传该参数，没加新接口。
4. **MinerU 未安装时**：`shutil.which("mineru")` 检测不到会直接 `raise ValueError`（提示 `pip install "mineru[pipeline]"`），跟现有 LibreOffice/Tesseract 缺失时的报错模式一致，不会让整个上传/分段流程崩掉。
5. **TXT/MD 不接入 MinerU**（本来就是纯文本/已有标题结构），`.doc/.ppt` 目前也不接入（MinerU 官方支持 `.doc/.ppt` 是走 precision 模式，跟本项目用的 pipeline backend 组合尚未验证，先只覆盖 `.pdf/.docx/.pptx/.xlsx/.xls`+图片）。
6. 前端 `KnowledgeBaseNew.tsx` Step2「创建设置」加了"解析方式"选择器（快速解析/精细解析），选 precise 时有提示条说明覆盖范围和耗时。

**已知限制**：本地开发环境未安装 MinerU，`precise` 路径只验证了"未安装时优雅报错"，MinerU 真正解析出的 Markdown 质量、`chunking_strategy=hierarchy` 配合 MinerU 标题的实际效果还没有真机验证过。

**行业前沿**

- **Docling**（IBM）：开源多格式文档解析库，支持 PDF 版面分析、表格结构识别、图文分离，质量远超 `PyPDF2`——本项目 PDF 结构感知/父子切片的短板可以靠它解决
- **语义分段（Semantic Chunking）**：本项目已实现逐句 embedding + 相似度断句的基础版本；更进一步的做法是用专门的分段模型（而非通用 embedding 模型）打分，或者滑动窗口对比而非只看相邻句
- **文档理解模型**：多模态 LLM（GPT-4V、Qwen-VL）直接处理 PDF 页面截图，无需解析文字层，彻底解决扫描版问题
