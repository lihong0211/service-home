# 知识库管理层（Knowledge Base）

> 文件：`service/ai/knowledge.py`（1406 行）、`service/ai/files.py`  
> 生成日期：2026-02-26

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
      ▼ 2. 保存文件到 data/knowledge_base/{kb_id}/{timestamp}_{filename}
      │
      ▼ 3. parse_file_to_documents(file_path, filename, chunk_size, chunk_overlap)
      │     PDF  → 按页提取 → 固定长度分片（category = "第N页"）
      │     DOCX → 合并段落+表格全文 → 固定长度分片
      │     PPTX → 按幻灯片提取，每页合并为一条
      │     TXT/MD → 直接分片
      │     Excel → 按工作表行提取 → 分片
      │     图片  → Tesseract OCR → 分片（失败时保存文件但分段为0）
      │     .doc/.ppt → LibreOffice 转换后处理
      │
      ▼ 4. _add_document_and_segments_to_kb()
      │     写 knowledge_base_document + knowledge_base_segment
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
```

**关键设计原则**

- **上传与向量化解耦**：上传只解析落库，向量化是独立步骤。这允许用户先上传多个文件、调整分段参数、预览分段效果，最后一次性向量化，避免反复调 Embedding API。
- **增量智能向量化**：`vectorize_knowledge_base` 会比对当前分段 ID 集合与已向量化 ID 集合，只有新增才增量 embed，有删除才全量重建。
- **同名文件覆盖**：批量上传时，同名文件（忽略大小写）会删除旧文档的分段和磁盘文件，重新入库，避免重复段落污染向量库。
- **文件保留原则**：解析失败时文件仍保存到磁盘，创建文档记录（分段数为 0），用户后续可重新分段，不因解析错误丢失原文件。

**与行业标准方案对比**

| 维度 | 本地实现 | LangChain Document Loaders | LlamaIndex |
|------|---------|---------------------------|-----------|
| 格式支持 | PDF/DOCX/PPTX/TXT/MD/Excel/图片 | 100+ 格式 Loader | 50+ 格式 Loader |
| 分段策略 | 固定长度（自定义 chunk_size/overlap） | 多种策略可选 | 多种策略 + 语义分段 |
| 数据库集成 | 原生集成 MySQL 三表管理 | 无，需自行实现 | 无，需自行实现 |
| 上传与向量化解耦 | 是（分两步） | 否（通常一步完成） | 否 |
| 分段预览 | 支持（落库前可预览，落库后也可查看） | 不支持 | 不支持 |
| **选型建议** | 需要文档管理 UI、与业务库集成 | 快速原型、多格式批量导入 | 复杂文档结构、语义分段 |

---

## 第三部分：代码实现深度解析

**核心函数清单**

| 函数 | 作用 |
|------|------|
| `parse_file_to_documents(file_path, filename, chunk_size, chunk_overlap)` | 按扩展名路由到对应解析函数，返回 `[{id, text, category}]` |
| `_chunk_text(text, chunk_size, chunk_overlap)` | 固定长度分片核心算法 |
| `vectorize_knowledge_base(kb_id)` | 智能向量化：增量/全量重建自动判断 |
| `upload_knowledge_base_api()` | 多文件上传处理，含同名覆盖、格式校验、OCR 跳过选项 |
| `execute_segments_api()` | 重新分段（支持调整 chunk_size 后重新切割已有文档） |
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

- 分段策略仅支持固定长度，不支持语义分段（按句子、按标题、按段落边界）
- PDF 仅用 `PyPDF2` 提取文字层，扫描版/图表型 PDF 无法处理
- DOCX 表格提取为纯文本（`" | ".join(cell.text)`），复杂表格格式丢失
- 向量化为同步操作，大型知识库（千条以上分段）会导致 HTTP 请求超时

**演进建议**

- 短期：向量化改为异步任务（后台线程 + 进度轮询接口），解决大知识库超时问题
- 中期：引入语义分段策略（按自然段落边界、按标题层级），提升分段质量
- 长期：PDF 解析引入 OCR 后备（`pdfplumber` + 云 OCR），支持扫描版 PDF

**行业前沿**

- **Docling**（IBM）：开源多格式文档解析库，支持 PDF 版面分析、表格结构识别、图文分离，质量远超 `PyPDF2`
- **语义分段（Semantic Chunking）**：通过计算相邻句子的 Embedding 相似度，在语义断裂处切分，比固定长度分片检索质量更高
- **文档理解模型**：多模态 LLM（GPT-4V、Qwen-VL）直接处理 PDF 页面截图，无需解析文字层，彻底解决扫描版问题
