# 知识库管理层（Knowledge Base）

> 覆盖代码：`service/ai/knowledge.py`（1877 行）、`service/ai/files.py`（508 行）、`service/ai/knowledge_mineru.py`（108 行，新增）
> 路由：`routes/ai.py` 中以 `/ai/knowledge-base/*` 和 `/ai/files/*` 为前缀注册

---

## 第一部分：背景演进

### 问题背景

向量检索本身只关心「一段文本 → 一个向量 → 相似度排序」，但业务侧要解决的问题是「用户上传一份 PDF/DOCX/PPTX，之后就能被问答」。这中间横亘着一整条脏活链路：识别文件格式、把二进制内容还原成结构化文本、按检索粒度切块、落库管理文档与分段的生命周期、调用 Embedding 生成向量、处理重名/失败/增量更新等边界情况。知识库管理层（`service/ai/knowledge.py` + `service/ai/files.py`）就是把这条链路封装成一组面向业务的 HTTP 接口，使调用方不需要感知向量库的底层细节。

### 传统方案的局限：简单文本抽取 vs 结构化解析

文档解析大致经历了两代方法：

- **简单文本抽取**：用 `PyPDF2.extract_text()`、`python-docx` 读段落这类 API，把文档当作一条纯文本流线性拼接。速度快、依赖轻，但会丢失版面结构——标题层级、表格边界、多栏排版、图文关系全部被拍平成一维文本，检索出的分段经常是"半句话"或者"表格里孤立的一个单元格"。
- **结构化解析**：先做版面分析（layout analysis），识别标题、正文、表格、公式、图片区域，再按结构重建文档（通常输出为带标题层级的 Markdown）。检索出的分段能保留"这段话属于哪个章节"这类上下文，答案质量更高，但依赖版面检测 + OCR + 表格识别模型，计算成本和部署复杂度显著上升。

本项目的核心解析函数（`_documents_from_pdf`、`_documents_from_docx` 等）长期以来走的是第一条路线：`PyPDF2` 按页抽取文字层、`python-docx` 读段落与表格。这条路线对 DOCX/MD 这类本身带机器可读标题标签的格式效果尚可（`Heading 1..N` 样式、`#` 语法），但对 PDF、PPTX、Excel、图片完全没有可用的结构信息——本模块的 `structure`/`hierarchy` 分段策略对这四类格式只能静默回退成定长切分（见第二部分对照表）。

### 核心概念

- **知识库（Knowledge Base）**：业务层聚合概念，对应 `knowledge_base` 表一行，包含名称、描述、解析策略（`parsing_strategy`）、分段策略（`chunking_strategy`）、关联向量库（`vector_db_id`）等元信息。
- **文档（Document）**：一次上传对应 `knowledge_base_document` 表一行，记录原始文件名、磁盘路径、状态（`pending/parsing/segmented/failed`）。
- **分段（Segment/Chunk）**：文档被切成的最小检索单元，对应 `knowledge_base_segment` 表一行，大小由 `chunk_size`（默认 1000 字符）和 `chunk_overlap`（默认 200 字符重叠）控制；父子切片模式下还有 `parent_id` 关联。
- **向量化（Vectorize）**：把分段文本批量调用 Embedding API 转成向量、写入向量库索引的过程，是整条链路里最耗时的一步，被刻意与上传解耦成独立步骤。

### 演进脉络

| 阶段 | 方案 | 特点 |
|------|------|------|
| 早期 | 手动准备纯文本 → 直接调向量库 API | 用户门槛高，只支持纯文本 |
| 文档解析库兴起 | `PyPDF2`、`python-docx`、`python-pptx` 等专用库 | 支持多格式，但每种格式适配逻辑碎片化，无统一抽象 |
| 知识库平台化 | LangChain / LlamaIndex Document Loaders | 统一 Loader 接口，生态丰富，但抽象层厚、与自有业务库集成需要额外胶水代码 |
| 结构化解析工具兴起 | Unstructured.io、LlamaParse、**MinerU**（OpenDataLab）等专用文档解析服务/工具 | 版面检测 + OCR + 表格/公式识别一体化，输出结构化 Markdown/JSON，解决"简单抽取丢结构"的问题，但通常是重量级依赖或云服务 |
| **本项目现状** | 自研轻量解析（`fast`）+ MinerU 可选精细解析（`precise`）+ 三表结构管理 | 默认路径轻量可控，同时为需要更高解析质量的场景提供可选升级路径 |

支持格式的演进路径：`TXT → PDF → DOCX → PPTX → Excel → 图片（OCR）`，并通过 LibreOffice 无缝支持旧版 `.doc`/`.ppt`。2026-07-31 起新增 `parsing_strategy=precise`，可选接入 MinerU 作为解析质量的升级路径。

### 本模块的定位

知识库层是「文档到向量」管道的统一入口：向上为 HTTP 路由层提供完整的知识库 CRUD 与文档管理接口，向下委托 `service/ai/vector_db_qdrant.py` 完成实际的向量写入与索引维护。它维护三张 MySQL 表（`knowledge_base`、`knowledge_base_document`、`knowledge_base_segment`），构成完整的文档生命周期管理；文件的物理存储与格式转换（LibreOffice）则下沉到 `service/ai/files.py`；`service/ai/knowledge_mineru.py` 是这条链路里新增的一条**可选解析支路**，不改变默认行为。

---

## 第二部分：架构剖析

### 三层职责划分

```
知识库管理层 (knowledge.py)          文件处理层 (files.py)              文档解析器层
────────────────────────           ─────────────────────            ─────────────────────
· 知识库/文档/分段的 CRUD            · 通用文件上传与 manifest 管理      · _documents_from_pdf / _docx / ...
· 上传编排：调解析器 → 落库           · LibreOffice 格式转换              (knowledge.py 内，parsing_strategy=fast)
· 分段策略路由                       · 文档预览（转 PDF / HTML 预览）    · parse_file_to_documents_mineru
· 向量化编排（委托 vector_db_qdrant） · 独立于知识库的文件存取接口          (knowledge_mineru.py，parsing_strategy=precise)
```

`knowledge_mineru.py` 在解析链路中的位置是**新增的可选路径，不是替代**：`parse_file_to_documents()`（knowledge.py:731）顶部新增了 `parsing_strategy: str = "fast"` 参数，默认值 `fast` 时走的是文件里原有的一整套 `_documents_from_pdf/_documents_from_docx/_documents_from_pptx/...` 分派逻辑，一行未改；仅当调用方显式传入 `precise` 且文件类型在 MinerU 覆盖范围内（`is_mineru_supported()`）时，才会委托给 `knowledge_mineru.parse_file_to_documents_mineru()`。两条路径的输出形状完全一致（`[{"id", "text", "category", "metadata"?}, ...]`），下游的 `_add_document_and_segments_to_kb` 落库逻辑无需区分来源。

### 核心数据流：一次文档上传到可检索状态

```
POST /ai/knowledge-base/upload (multipart, 可多文件)
      │
      ▼ 1. 校验文件扩展名；同名文件（忽略大小写）先删除旧文档的分段与磁盘文件再继续
      │
      ▼ 2. _save_upload_to_kb_folder()：保存到 data/knowledge_base/kb_{知识库name}/{时间戳}_{原文件名}
      │
      ▼ 3. parse_file_to_documents(file_path, filename, chunk_size, chunk_overlap,
      │                            chunking_strategy, hierarchy_level, retain_hierarchy, parsing_strategy)
      │     parsing_strategy=fast（默认）：按扩展名路由到 _documents_from_* 系列函数
      │     parsing_strategy=precise：委托 knowledge_mineru.parse_file_to_documents_mineru()
      │
      ▼ 4. _add_document_and_segments_to_kb()
      │     写 knowledge_base_document + knowledge_base_segment
      │     （hierarchy 父子切片时：父块先插入拿到真实 id，携带同一 _local_parent_ref 的子块再写 parent_id）
      │     此时分段已落库，但向量库尚未更新——文档还不可被检索
      │
      (中间态：可调 GET /ai/knowledge-base/segments/preview 预览已落库分段，
       或调 POST /ai/knowledge-base/segments/execute 重新分段调整效果)

POST /ai/knowledge-base/vectorize { knowledge_base_id }
      │
      ▼ vectorize_knowledge_base(kb_id)
            ├─ kb.vector_db_id 已存在？
            │    ├─ 分段 id 集合与向量库现有 id 集合一致 → 跳过，count=0
            │    ├─ 只有新增 id → append_documents_batch()（增量 embed，只对新分段调 Embedding API）
            │    └─ 有 id 被删除 → _rebuild_vector_db_index()（全量重建，保证索引一致）
            └─ 首次向量化：create_vector_db() 建库 → 写 vector_db 表 → 回填 kb.vector_db_id
      （父子切片时：父块本身不建索引，只作为子块的上下文来源；子块的 embedding 用自身文本算，
        但返回给 LLM 的 text 字段换成父块正文——通过 embedding_text 字段实现分离，
        向量化逻辑见 vectorize_knowledge_base 内 parent_text_by_id 的处理）
```

### 文件类型 → 解析工具 / 分段策略速查表

`chunking_strategy` 取值：`fixed`（默认，固定长度，兼容旧值 `custom`）/ `structure`（标题层级）/ `hierarchy`（父子切片）/ `semantic`（语义切片）。下表基于默认 `parsing_strategy=fast`；`parsing_strategy=precise` 时 PDF/DOCX/PPTX/XLSX/XLS/图片改走 MinerU（见第三部分「设计决策 3」），structure/hierarchy 对 PDF/PPTX/Excel/图片的 ⚠️ 回退会变成实际支持，但目前尚无真机验证效果数据。

| 文件类型 | 解析工具/库 | fixed | structure | hierarchy | semantic |
|---|---|---|---|---|---|
| PDF `.pdf` | `PyPDF2.PdfReader`，按页提取文字层 | ✅ 每页内定长切分 | ⚠️ 回退 fixed（PDF 无可靠标题结构） | ⚠️ 回退 fixed | ✅ 每页内按句子相似度切分 |
| DOCX `.docx` | `python-docx`，段落 + 表格提取 | ✅ 段落/表格边界累加到接近 chunk_size 再切（保留自然边界） | ✅ 按 `Heading 1..N`/`Title` 样式识别层级，每段带 `heading_path` breadcrumb | ✅ 按 `hierarchy_level` 生成父块，父块内再切子块，写 `parent_id` | ✅ 忽略标题结构，对全文按语义相似度切 |
| DOC `.doc` | LibreOffice 转 `.docx` 后按上一行处理 | 同 DOCX | 同 DOCX | 同 DOCX | 同 DOCX |
| PPTX `.pptx` | `python-pptx`，按幻灯片提取，单页文本 < 40 字时并入下一页 | ✅ 唯一支持的策略 | ⚠️ 忽略 `chunking_strategy`，恒定走 fixed 那一套逻辑 | ⚠️ 同上 | ⚠️ 同上（`parse_file_to_documents` 对 `.ppt/.pptx` 分支根本不传 chunking_strategy 参数） |
| PPT `.ppt` | LibreOffice 转 `.pptx` 后按上一行处理 | 同 PPTX | 同 PPTX | 同 PPTX | 同 PPTX |
| TXT `.txt` | 内置 `open()` 读取全文 | ✅ | ⚠️ 回退 fixed（纯文本无标题概念） | ⚠️ 回退 fixed | ✅ |
| MD `.md` | 内置 `open()` 读取全文 | ✅ | ✅ 按 `#`~`######` 层级识别标题（空行分块，块首行是标题行时视为 heading） | ✅ 同 DOCX 的父子逻辑，标题来源换成 `#` 层级 | ✅ |
| Excel `.xlsx`/`.xls` | `openpyxl`/`xlrd`，按工作表逐行转文本，`\t` 拼接 | ✅ | ⚠️ 回退 fixed（表格无标题层级概念） | ⚠️ 回退 fixed | ✅ |
| 图片 `.jpg/.png/.gif/.webp/.bmp` | `pytesseract`（Tesseract OCR，限制最长边 2000px 缩放，`--psm 6` 单列文本块模式提速） | ✅（OCR 文本按定长切） | ⚠️ 回退 fixed | ⚠️ 回退 fixed | ✅（OCR 文本按语义切） |

**语义切片（semantic）的通用兜底**：`_split_sentences` 按中英文句末标点/换行切句，逐句调 `get_embedding` 算相邻句相似度，低于阈值 `_SEMANTIC_SIMILARITY_THRESHOLD=0.6` 处断开分组；文本超过 `_SEMANTIC_MAX_CHARS=30000` 字符（逐句 embedding 成本过高）或整段只有 0～1 句时，自动回退固定长度切片（重叠比例改为 `chunk_size*0.2`），不会因文本过长而卡死或狂调 embedding 接口。

**父子切片（hierarchy）只对 DOCX/MD 生效**：`hierarchy_level`（默认 3，范围 1-6）控制哪一级标题算父块边界（`_group_sections_for_hierarchy`）；父块的 `segment_metadata.is_parent=true`，子块的 `parent_id` 指向父块在 `knowledge_base_segment` 表里的真实 id。`retain_hierarchy=false` 时子块不写 `heading_path`/`heading` metadata（父块本身不受影响）。

### 关键设计原则

- **上传与向量化解耦**：上传只解析落库，向量化是独立的第二步接口。允许用户先批量上传、调整分段参数反复预览，最后一次性向量化，避免反复调用 Embedding API 浪费成本。
- **增量智能向量化**：`vectorize_knowledge_base` 比对当前分段 id 集合与向量库已有 id 集合，只有新增才走增量 embed，有删除才触发全量重建，兼顾成本与一致性。
- **解析方式与分段策略正交**：`parsing_strategy`（解析引擎：fast/precise）与 `chunking_strategy`（切块算法：fixed/structure/hierarchy/semantic）是两个独立维度，MinerU 输出的 Markdown 复用的是 knowledge.py 里同一套 `_md_units_with_headings`/`_documents_from_heading_units`/`_merge_units_to_chunks`/`_semantic_chunk_text` 分段工具，没有另起一套分段实现。

### 与行业标准方案对比

| 维度 | 本地实现（本项目） | Unstructured.io | LlamaParse | LangChain Document Loaders |
|------|---------|---------|---------|---------------------------|
| 格式支持 | PDF/DOCX/PPTX/TXT/MD/Excel/图片；MinerU 可选路径覆盖 PDF/DOCX/PPTX/XLSX/XLS/图片 | 20+ 格式，含扫描件版面分析 | PDF 为主，特别擅长表格/公式 | 100+ 格式 Loader，生态最广 |
| 结构化能力 | fast 路径仅 DOCX/MD 有标题结构；precise（MinerU）覆盖 PDF/PPTX/Excel/图片的版面检测 | 内置版面检测模型，结构化程度高 | 云端 LLM 驱动解析，结构化质量高（尤其复杂表格） | 依赖具体 Loader，参差不齐 |
| 部署形态 | 本地 Python 库（fast）/ 本地 CLI 子进程（precise，MinerU 开源自托管） | 本地库或云 API 两种模式 | 仅云 API（LlamaCloud），需联网调用 | 本地库，部分 Loader 依赖外部服务 |
| 数据库集成 | 原生集成 MySQL 三表管理（知识库/文档/分段生命周期） | 无，需自行实现 | 无，需自行实现 | 无，需自行实现 |
| 上传与向量化解耦 | 是（分两步，支持分段预览调优） | 否，通常一体化 pipeline | 否 | 否，通常一步完成 |
| 分段预览 | 支持（落库前可预览返回值，落库后可查库） | 不支持 | 不支持 | 不支持 |
| **选型建议** | 需要文档管理 UI、与业务库深度集成、成本敏感 | 需要开箱即用的高质量结构化解析、可自托管 | 追求最高解析质量、可接受云端调用与按量计费 | 快速原型、需要覆盖长尾格式 |

---

## 第三部分：代码实现深度解析

### 核心函数/类清单

| 函数 | 位置 | 作用 |
|------|------|------|
| `parse_file_to_documents(file_path, filename, chunk_size=1000, chunk_overlap=200, chunking_strategy="fixed", hierarchy_level=3, retain_hierarchy=True, parsing_strategy="fast")` | knowledge.py:731 | 解析总入口：按扩展名 + `chunking_strategy` + `parsing_strategy` 路由，返回 `[{id, text, category, metadata?}]` |
| `vectorize_knowledge_base(knowledge_base_id) -> dict` | knowledge.py:75 | 智能向量化：增量新增 / 全量重建 / 首次创建三分支自动判断；处理父子切片的 `embedding_text` 分离 |
| `_merge_units_to_chunks(units, chunk_size, chunk_overlap) -> list[str]` | knowledge.py:211 | 段落/正文单元累加到接近 `chunk_size` 再切，超长单元内部再定长切分；被固定分片、结构感知、父子切片、MinerU 分段共用 |
| `_semantic_chunk_text(text, chunk_size=1000) -> list[str]` | knowledge.py:256 | 语义切片：逐句 `get_embedding` + 相邻句余弦相似度 < 0.6 处断句；超长文本/单句文本自动回退定长 |
| `_documents_from_heading_units(units, source, chunk_size, chunk_overlap, chunking_strategy, hierarchy_level=3, retain_hierarchy=True) -> list[dict]` | knowledge.py:410 | structure/hierarchy 的统一装配入口，供 DOCX/MD/MinerU 输出的 Markdown 共用 |
| `_add_document_and_segments_to_kb(knowledge_base_id, file_name, documents, file_id=None, path=None) -> dict` | knowledge.py:858 | 落库：写 `knowledge_base_document` + 逐条写 `knowledge_base_segment`；按 `_local_parent_ref` 把父子切片的临时引用解析成真实 `parent_id`（要求父块排在其子块之前） |
| `upload_knowledge_base_api(request) -> dict` | knowledge.py:1291 | 多文件上传总编排：同名覆盖、格式校验、`skip_ocr` 跳过、解析失败兜底保留文件 |
| `execute_segments_api(request) -> dict` | knowledge.py:1202 | 重新分段接口：支持对已有文档批量 `document_ids` 重新分段，或对已上传文件 `kb_id+file_id+file_name` 首次分段入库 |
| `is_mineru_supported(filename) -> bool` / `parse_file_to_documents_mineru(file_path, filename, ...) -> list[dict]` | knowledge_mineru.py:27 / 67 | MinerU 解析路径：判断文件类型是否在覆盖范围内；调 CLI 转 Markdown 后复用 knowledge.py 的标题分段工具 |

### 关键实现细节

**1. 固定长度分片算法（`_chunk_text`，knowledge.py:192）**

```python
def _chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - chunk_overlap   # 下一块起点回退 overlap 个字符，形成重叠窗口
        if start >= len(text):
            break
    return chunks
```

`chunk_overlap` 保证相邻分段之间有固定字符数的重叠窗口，防止关键信息恰好落在切分边界上而无法被任一分段完整检索到。这是除 `semantic` 外所有格式的兜底切分逻辑。

**2. MinerU 解析路径的接入方式（`parse_file_to_documents_mineru`，knowledge_mineru.py:67）**

```python
def _run_mineru_to_markdown(file_path, backend="pipeline"):
    if not shutil.which("mineru"):
        raise ValueError('MinerU 未安装：请先执行 pip install "mineru[pipeline]" 后重试...')
    out_dir = tempfile.mkdtemp(prefix="mineru_out_")
    try:
        result = subprocess.run(
            ["mineru", "-p", file_path, "-o", out_dir, "-b", backend],
            capture_output=True, text=True, timeout=600,
        )
        ...  # 在 out_dir 中查找生成的 .md 文件并读取
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
```

MinerU 不是作为 Python 包导入的，而是通过 `subprocess` 调用其命令行工具（`mineru` CLI），`backend="pipeline"` 是纯 CPU 推理模式，适合无 GPU 的服务器环境。拿到 Markdown 文本后，`parse_file_to_documents_mineru` 直接复用 knowledge.py 里已有的 `_md_units_with_headings`（提取标题层级）→ `_documents_from_heading_units`（structure/hierarchy 装配）或 `_merge_units_to_chunks`/`_semantic_chunk_text`（fixed/semantic），**没有为 MinerU 输出重新实现一套分段逻辑**——MinerU 只负责把非结构化的原始文件变成结构化的 Markdown，分段算法与 fast 路径下的 MD 文件走的是完全同一份代码。

**3. 边界处理：解析失败时文件仍然保留（`upload_knowledge_base_api`，knowledge.py:1418-1448）**

```python
try:
    documents = parse_file_to_documents(tmp_path, fn, chunk_size=chunk_size, ...)
except Exception as parse_err:
    out = _add_document_and_segments_to_kb(kb_id, fn, [], path=saved_path)
    documents_result.append({..., "segment_count": 0, "error": f"文件解析失败: {str(parse_err)}"})
    continue
```

无论是解析异常（如 Tesseract/LibreOffice 未安装）还是解析返回空列表（如扫描版 PDF 无文字层），文件都已经在此之前通过 `_save_upload_to_kb_folder` 落盘，文档记录也会被创建（`segment_count=0`），不会因为某一步解析失败就丢弃已上传的原始文件——用户可以在环境就绪后调用 `POST /ai/knowledge-base/segments/execute` 对该文档重新分段。

### 设计决策与取舍

**决策 1：DOCX 固定分片合并全文，不保留原始段落边界**
`_documents_from_docx` 的 `fixed` 分支把所有段落/表格文本拼接成 `units` 列表后交给 `_merge_units_to_chunks` 统一按字符数累加切块，而不是每个段落单独成一个分段。原因：保留段落边界会让标题、单行表格行这类短段落各自独立成一个分段，检索时这些低信息密度的段落会挤占 top-k 名额。代价：跨段落的语义边界可能被切断，用 `chunk_overlap=200` 缓解。

**决策 2：图片 OCR 失败/未安装时保留文件而非拒绝上传**
`_ocr_image_to_text` 捕获所有异常统一返回 `None`，上层 `_documents_from_image` 返回空列表时 `parse_file_to_documents` 抛 `ValueError`，但 `upload_knowledge_base_api` 会捕获该异常、仍创建 `segment_count=0` 的文档记录。原因：Tesseract 未安装不应导致批量上传的其他文件也失败。代价：调用方若不检查响应里的 `error` 字段，可能直接向量化而遗漏该文件内容——这是接口契约层面需要前端配合提醒的地方。

**决策 3：为什么引入 MinerU 作为可选解析路径，而不是替换现有解析**
现有 `fast` 路径（`PyPDF2`/`python-docx`/`python-pptx`/`openpyxl`）对 PDF/PPTX/Excel/图片没有可靠的标题结构信息，导致这四类格式的 `structure`/`hierarchy` 分段策略只能静默回退成 `fixed`（见第二部分对照表）。MinerU 原生支持这些格式的版面检测并统一输出带标题层级的 Markdown，能补上这个短板。但 MinerU 是重量级依赖（版面检测 + OCR + 表格/公式识别模型，`pip install "mineru[pipeline]"` 体积和推理耗时都显著高于现有轻量库），不适合作为默认路径强制替换。因此做成 `knowledge_base.parsing_strategy` 字段驱动的用户可选项：`fast`（默认，现有解析全套代码零改动）/ `precise`（委托 `knowledge_mineru.py`，仅在文件类型受 MinerU 支持时生效）。`parsing_strategy` 字段本身在 schema 里早已存在，此前从未被实际读取，这次实现是首次让它生效。

**决策 4：知识库与向量库解耦**
知识库表管理文档的原始文本和分段结构；向量库（`vector_db_qdrant.py`）只管浮点向量和相似度检索，两者通过 `vector_db_id` 外键关联。好处是同一个向量库理论上可以被多个知识库或多种检索场景共用。

**决策 5：分段落库与向量化拆成两个 HTTP 接口**
`execute_segments_api`/`upload_knowledge_base_api` 只做「解析 + 落库」，检索能力要等 `vectorize_knowledge_base_api` 显式调用后才生效。好处是用户可以反复调整 `chunk_size`/`chunking_strategy` 预览效果（`GET /ai/knowledge-base/segments/preview`），确认满意后再一次性触发 Embedding 调用，避免每次调参都重新计算全部向量的浪费。

---

## 第四部分：应用场景与实战

### 核心使用场景

- **企业知识库建设**：批量上传产品文档、FAQ、规章制度（PDF/DOCX/PPTX/Excel），解析分段后向量化，供 RAG 问答模块（`service/ai/rag.py`）检索。
- **文档分段效果调优**：上传后不立即向量化，先用 `segments/preview` 查看分段结果，调整 `chunk_size`/`chunking_strategy` 后调用 `segments/execute` 重新分段，满意后再向量化，避免反复消耗 Embedding API 额度。
- **扫描件/复杂版面文档解析**：默认 `fast` 路径对 PDF 只能提取文字层、PPTX 不支持标题结构切分；这类文档可选择 `parsing_strategy=precise` 走 MinerU，获得带标题层级的结构化分段。

### 快速上手

**环境依赖**

```bash
# 基础解析依赖（fast 路径，默认必需）
pip install PyPDF2 python-docx python-pptx openpyxl xlrd pytesseract Pillow

# macOS 下 OCR 支持（图片解析，可选）
brew install tesseract tesseract-lang

# LibreOffice（.doc/.ppt 转换，可选）
brew install --cask libreoffice

# MinerU（precise 路径，仅需要精细解析时安装）
pip install "mineru[pipeline]"   # CPU 环境
# 或 pip install "mineru[core]"  # 有 GPU/VLM 服务时

export DASHSCOPE_API_KEY=sk-xxx   # 语义切片、向量化需要 Embedding API
```

**示例 1：解析文件为分段（不落库，仅预览效果）**

```python
from service.ai.knowledge import parse_file_to_documents

docs = parse_file_to_documents("report.pdf", "report.pdf", chunk_size=800, chunk_overlap=150)
print(len(docs), "个分段")
print(docs[0])  # {"id": "p1", "text": "...", "category": "第1页"}
```

**示例 2：向量化已有知识库**

```python
from service.ai.knowledge import vectorize_knowledge_base

result = vectorize_knowledge_base(1)
print(result)
# 首次: {"vector_db_id": 5, "count": 128, "created": True}
# 增量: {"vector_db_id": 5, "count": 3, "created": False, "incremental": True}
```

**示例 3：走 MinerU 精细解析（需已安装 MinerU CLI）**

```python
from service.ai.knowledge import parse_file_to_documents

docs = parse_file_to_documents(
    "report.pdf", "report.pdf",
    chunking_strategy="hierarchy", hierarchy_level=2,
    parsing_strategy="precise",   # 走 MinerU，PDF 也能拿到标题层级
)
print(len(docs), "个分段（含父块）")
```

**curl 示例：上传并追加到已有知识库**

```bash
curl -X POST http://localhost:3000/ai/knowledge-base/upload \
  -F "file=@report.pdf" -F "kb_id=1" -F "chunking_strategy=structure"
```

### 常见问题排查

- **`.doc`/`.ppt` 文件解析失败**：需要安装 LibreOffice（`brew install --cask libreoffice`），否则 `convert_doc_to_docx_with_libreoffice`/`convert_ppt_to_pptx_with_libreoffice` 返回 `None`，最终抛出包含安装提示的 `ValueError`。
- **PDF 解析为空**：扫描版 PDF 没有文字层，`PyPDF2` 无法提取文本。可选方案：用 OCR 工具将扫描 PDF 转为可搜索 PDF 后重新上传，或改用 `parsing_strategy=precise`（MinerU 内置 OCR，能处理部分扫描件，但效果未经真机验证）。
- **`parsing_strategy=precise` 报错 "MinerU 未安装"**：`shutil.which("mineru")` 检测不到可执行文件时直接抛错，需先 `pip install "mineru[pipeline]"`。
- **向量化后检索不到内容**：确认 `knowledge_base.vector_db_id` 是否已填写（调 `GET /ai/knowledge-base/detail` 确认），为 `null` 说明尚未调用过 `POST /ai/knowledge-base/vectorize`。
- **上传响应里某个文件 `segment_count=0`**：查看该条目的 `error` 字段——可能是 OCR/LibreOffice 未安装、扫描版 PDF、或用户主动传了 `skip_ocr=true` 跳过图片 OCR，文件本身已保存，可后续调用 `segments/execute` 重新分段。

---

## 第五部分：评估与展望

### 优势

- 上传与向量化解耦，支持分段预览调优，减少 Embedding API 浪费。
- 增量向量化：只对新增分段调用 Embedding API，大知识库追加文档成本低。
- 同名文件自动覆盖、批量上传对失败文件容错（保留原文件，不影响批次内其他文件）。
- 新增 MinerU 可选路径，在不改动默认行为、不引入强依赖的前提下，为 PDF/PPTX/Excel/图片提供了结构化解析的升级选项。

### 局限与技术债务

- 结构感知/父子切片在 `fast` 路径下只对 DOCX/MD 生效；PDF/PPTX/Excel/图片没有可靠标题结构，这两种策略对它们静默回退成 fixed，容易让不了解实现细节的调用方误以为策略生效了。
- PPTX/PPT 完全不接受 `chunking_strategy` 参数（`parse_file_to_documents` 里这两个分支未透传该参数），不管前端选了什么策略都固定走"按幻灯片+短页并入下一页"这一套逻辑。
- `fast` 路径下 PDF 仅用 `PyPDF2` 提取文字层，扫描版/图表型 PDF 无法处理；DOCX 表格提取为纯文本（`" | ".join(cell.text)`），复杂表格格式（合并单元格、嵌套表格）会丢失。
- 语义切片逐句调用 Embedding API，长文本处理耗时高（已有 3 万字符自动回退兜底，但仍明显慢于固定分片）。
- 向量化为同步 HTTP 请求，大型知识库（千条以上分段）首次向量化可能导致请求超时。
- MinerU 路径目前只验证了"未安装时优雅报错"这一分支，其真实解析出的 Markdown 质量、`hierarchy_level` 与 MinerU 标题层级的配合效果尚无真机验证数据；`.doc`/`.ppt` 也暂未接入 MinerU（其精确模式与本项目使用的 pipeline backend 组合尚未验证）。

### 演进建议

- 短期：向量化改为异步任务（后台线程/任务队列 + 进度轮询接口），解决大知识库首次向量化的超时问题。
- 短期：PPTX/PPT 接入 `chunking_strategy`（至少支持 semantic），做法可参照 PDF/TXT，把内部的 `_chunk_text` 调用换成 `_chunk_text_by_strategy` 即可。
- 中期：对已接入的 MinerU 路径补充真机验证——评估其输出 Markdown 的标题层级准确度，针对性调整 `hierarchy_level` 语义或分段合并策略。
- 长期：评估将 DOCX/Excel 复杂表格结构（合并单元格等）保留为结构化 metadata 而非纯文本拼接，提升检索时表格类内容的可用性。

### 行业前沿

- **Docling**（IBM 开源）：多格式文档解析库，支持 PDF 版面分析、表格结构识别、图文分离，质量优于 `PyPDF2`，是本项目 `fast` 路径短板的另一种可能补齐方案。
- **语义分段的进阶做法**：本项目已实现逐句 embedding + 相邻相似度断句的基础版本；更进一步可用专门的分段判别模型（而非通用 embedding 模型）打分，或用滑动窗口对比替代只看相邻句。
- **多模态文档理解**：GPT-4V、Qwen-VL 等多模态 LLM 可直接处理 PDF 页面截图，无需依赖文字层提取，是彻底解决扫描版 PDF 问题的方向，MinerU 这类工具的 VLM backend 也是在往这个方向演进。

---

## 变更记录

**2026-08-18 全量重写**：本次重写基于对 `service/ai/knowledge.py`、`service/ai/files.py`、`service/ai/knowledge_mineru.py` 的完整代码走查，核心变化是新增了 **MinerU 精确解析能力**（`service/ai/knowledge_mineru.py`）：通过 `knowledge_base.parsing_strategy`（`fast`/`precise`）字段驱动，在 `parse_file_to_documents()` 中作为可选解析支路接入，不改动默认 `fast` 路径下任何现有解析函数；MinerU 输出的 Markdown 复用 `knowledge.py` 中既有的标题层级分段工具（`_md_units_with_headings`/`_documents_from_heading_units`/`_merge_units_to_chunks`/`_semantic_chunk_text`），解决了 PDF/PPTX/Excel/图片在 `fast` 路径下无法使用 `structure`/`hierarchy` 分段策略的短板。该能力目前仅验证了未安装时的报错路径，真实解析效果待后续真机验证。
