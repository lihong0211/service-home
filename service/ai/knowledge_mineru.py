#!/usr/bin/env python3
"""
MinerU 解析：knowledge_base.parsing_strategy = "precise" 时的可选解析路径。

- 完全独立于 knowledge.py 里现有的 _documents_from_pdf/_documents_from_docx/_documents_from_pptx/...
  （parsing_strategy 默认 "fast"，走那一套，一行不改）。
- 只在显式选择 precise 时才会被 import，未安装 MinerU 不影响默认流程。
- 通过 MinerU CLI（不是 Python 包导入）把文件转成 Markdown，再复用 knowledge.py 里已有的
  结构感知/父子切片/固定/语义分段工具，输出形状与 parse_file_to_documents 完全一致。
- MinerU 目前原生支持 PDF/DOCX/PPTX/XLSX/XLS/图片；TXT/MD 本身已是纯文本/已有标题结构，不需要
  也不支持 MinerU（调用方应对这两种扩展名保持 parsing_strategy=fast）。

安装（可选，任选其一，仅在需要 precise 解析时才需要）：
    pip install "mineru[pipeline]"   # CPU 环境
    pip install "mineru[core]"       # 有 GPU / VLM 服务时
参考：https://github.com/opendatalab/mineru
"""
import os
import shutil
import subprocess
import tempfile

_MINERU_SUPPORTED_EXT = (".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
_MINERU_TIMEOUT_SECONDS = 600


def is_mineru_supported(filename: str) -> bool:
    """该扩展名是否在 MinerU 覆盖范围内；不在范围内时调用方应回退 fast 解析。"""
    return (filename or "").lower().endswith(_MINERU_SUPPORTED_EXT)


def _run_mineru_to_markdown(file_path: str, backend: str = "pipeline") -> str:
    """
    调 MinerU CLI 把文件解析为 Markdown，返回 markdown 文本。
    backend 默认 pipeline：纯 CPU 推理，不依赖 GPU 或额外的 VLM 服务，适合服务器环境；
    有 GPU 时可传 backend="hybrid-engine"（MinerU 默认值）效果更好。
    """
    if not shutil.which("mineru"):
        raise ValueError(
            "MinerU 未安装：请先执行 pip install \"mineru[pipeline]\" 后重试，"
            "或参考 https://github.com/opendatalab/mineru"
        )
    out_dir = tempfile.mkdtemp(prefix="mineru_out_")
    try:
        result = subprocess.run(
            ["mineru", "-p", file_path, "-o", out_dir, "-b", backend],
            capture_output=True, text=True, timeout=_MINERU_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise ValueError(f"MinerU 解析失败: {(result.stderr or result.stdout or '').strip()[:500]}")
        md_path = None
        for root, _dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith(".md"):
                    md_path = os.path.join(root, f)
                    break
            if md_path:
                break
        if not md_path:
            raise ValueError("MinerU 未生成 Markdown 输出")
        with open(md_path, "r", encoding="utf-8", errors="ignore") as fp:
            return fp.read()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def parse_file_to_documents_mineru(
    file_path: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunking_strategy: str = "fixed",
    hierarchy_level: int = 3,
    retain_hierarchy: bool = True,
) -> list[dict]:
    """
    parsing_strategy="precise" 时的解析入口：把文件转成 Markdown 后，复用 knowledge.py 里
    已有的标题层级/分段工具（跟 MD 走的是同一套代码），按 chunking_strategy 生成分段。
    返回形状与 parse_file_to_documents 一致：[{"id", "text", "category", "metadata"?}, ...]。
    调用前应先用 is_mineru_supported() 检查文件类型。
    """
    from service.ai.knowledge import (
        _documents_from_heading_units,
        _md_units_with_headings,
        _merge_units_to_chunks,
        _semantic_chunk_text,
    )

    if not is_mineru_supported(filename):
        raise ValueError(f"MinerU 解析暂不支持该文件类型: {filename}，仅支持 PDF/DOCX/PPTX/XLSX/XLS/图片")
    source = filename or "mineru"
    text = _run_mineru_to_markdown(file_path)
    if not text.strip():
        return []

    if chunking_strategy in ("structure", "hierarchy"):
        units = _md_units_with_headings(text)
        return _documents_from_heading_units(
            units, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy, hierarchy_level=hierarchy_level, retain_hierarchy=retain_hierarchy,
        )
    if chunking_strategy == "semantic":
        chunks = _semantic_chunk_text(text, chunk_size=chunk_size)
    else:
        # fixed：仍按 MinerU 给出的标题/段落边界累加合并，而不是对整篇 Markdown 无脑截断
        units = [u["text"] for u in _md_units_with_headings(text)]
        chunks = _merge_units_to_chunks(units, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [{"id": f"{source}_c{i}", "text": c, "category": source} for i, c in enumerate(chunks)]
