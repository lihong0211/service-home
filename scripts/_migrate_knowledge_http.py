#!/usr/bin/env python3
"""一次性脚本：knowledge.py 去 compat，改为 async + Request。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "service" / "ai" / "knowledge.py"
text = path.read_text(encoding="utf-8")

# 1) 顶部导入
if "from fastapi import Request" not in text:
    text = text.replace(
        "from datetime import datetime\n",
        "from datetime import datetime\n\n"
        "from fastapi import Request\n"
        "from utils.http_body import (\n"
        "    collect_upload_files_from_form,\n"
        "    query_dict,\n"
        "    read_json_optional,\n"
        "    read_json_or_form_fields,\n"
        "    write_upload_to_disk,\n"
        ")\n",
        1,
    )

# 2) 去掉 compat 行
text = re.sub(r"(?m)^\s*from compat import .*\n", "", text)

# 3) API 函数签名
text = re.sub(
    r"^def list_knowledge_bases_api\(\):",
    "async def list_knowledge_bases_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def create_knowledge_base_api\(\):",
    "async def create_knowledge_base_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def create_knowledge_base_from_pdf_api\(\):",
    "async def create_knowledge_base_from_pdf_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def preview_segments_from_db_api\(\):",
    "async def preview_segments_from_db_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def execute_segments_api\(\):",
    "async def execute_segments_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def upload_knowledge_base_api\(\):",
    "async def upload_knowledge_base_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def list_knowledge_base_documents_api\(\):",
    "async def list_knowledge_base_documents_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def get_document_segments_api\(document_id: int\):",
    "async def get_document_segments_api(request: Request, document_id: int):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def preview_knowledge_document_api\(document_id: int\):",
    "async def preview_knowledge_document_api(request: Request, document_id: int):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def sync_knowledge_base_from_disk_api\(\):",
    "async def sync_knowledge_base_from_disk_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def rebuild_knowledge_base_api\(\):",
    "async def rebuild_knowledge_base_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def vectorize_knowledge_base_api\(\):",
    "async def vectorize_knowledge_base_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def vectorize_with_file_api\(\):",
    "async def vectorize_with_file_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def update_knowledge_base_api\(\):",
    "async def update_knowledge_base_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def delete_knowledge_base_api\(\):",
    "async def delete_knowledge_base_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def delete_knowledge_base_document_api\(\):",
    "async def delete_knowledge_base_document_api(request: Request):",
    text,
    flags=re.M,
)
text = re.sub(
    r"^def get_knowledge_base_detail_api\(\):",
    "async def get_knowledge_base_detail_api(request: Request):",
    text,
    flags=re.M,
)

# 4) request 访问模式
text = text.replace(
    "data = request.get_json(silent=True) or {}",
    "data = await read_json_optional(request) or {}",
)
text = text.replace("data = request.get_json() or {}", "data = await read_json_optional(request) or {}")
text = text.replace(
    "data = request.get_json(silent=True) or request.form or {}",
    "data = (await read_json_or_form_fields(request)) or {}",
)
text = text.replace(
    "data = request.get_json(silent=True) or request.form",
    "data = await read_json_or_form_fields(request)",
)
text = text.replace("request.args.get", "query_dict(request).get")
text = text.replace("return jsonify(", "return (")

# 5) create_knowledge_base_from_pdf：在 docstring 后插入 form = await request.form()
marker = "async def create_knowledge_base_from_pdf_api(request: Request):\n"
idx = text.find(marker)
if idx != -1 and "form = await request.form()" not in text[idx : idx + 800]:
    insert_at = idx + len(marker)
    # 跳过 docstring
    rest = text[insert_at:]
    if rest.lstrip().startswith('"""'):
        end = rest.find('"""', 3)
        if end != -1:
            insert_at = insert_at + end + 3
    text = text[:insert_at] + "\n    form = await request.form()\n" + text[insert_at:]

# 6) create_from_pdf 内 request.form / files / f.save
text = text.replace(
    "    name = (request.form.get(\"name\") or request.form.get(\"db\") or \"\").strip()",
    "    name = (form.get(\"name\") or form.get(\"db\") or \"\").strip()",
)
text = text.replace("    f = request.files.get(\"file\") or request.files.get(\"pdf\")", "    f = form.get(\"file\") or form.get(\"pdf\")")
text = text.replace("    description = (request.form.get(\"description\") or \"\").strip() or None", "    description = (form.get(\"description\") or \"\").strip() or None")
text = text.replace("        chunk_size = int(request.form.get(\"chunk_size\") or \"1000\")", "        chunk_size = int(form.get(\"chunk_size\") or \"1000\")")
text = text.replace("        chunk_overlap = int(request.form.get(\"chunk_overlap\") or \"200\")", "        chunk_overlap = int(form.get(\"chunk_overlap\") or \"200\")")
text = text.replace("            f.save(tmp_path)", "            await write_upload_to_disk(f, tmp_path)")

# 7) upload_knowledge_base_api：在 import 块之后插 form
um = "async def upload_knowledge_base_api(request: Request):\n"
u_idx = text.find(um)
if u_idx != -1 and "form = await request.form()" not in text[u_idx : u_idx + 1200]:
    # 在最后一个 from model.ai import KnowledgeBase 之后（紧跟 file_list）
    sub = text[u_idx : u_idx + 2000]
    anchor = "    from model.ai import KnowledgeBase\n"
    pos = sub.find(anchor)
    if pos != -1:
        ins = u_idx + pos + len(anchor)
        text = text[:ins] + "    form = await request.form()\n" + text[ins:]

text = text.replace(
    "    for key in request.files:\n        if key in (\"file\", \"file[]\", \"files\") or (key.startswith(\"file[\") and key.endswith(\"]\")):\n            file_list.extend(request.files.getlist(key))",
    "    for key in form.keys():\n        if key in (\"file\", \"file[]\", \"files\") or (key.startswith(\"file[\") and key.endswith(\"]\")):\n            file_list.extend(form.getlist(key))",
)
text = text.replace(
    "        single = request.files.get(\"file\")",
    "        single = form.get(\"file\")",
)
text = text.replace("    name = (request.form.get(\"name\")", "    name = (form.get(\"name\")")
text = text.replace("    kb_id = request.form.get(\"kb_id\")", "    kb_id = form.get(\"kb_id\")")
text = text.replace("    kb_name = (request.form.get(\"kb_name\")", "    kb_name = (form.get(\"kb_name\")")
text = text.replace("    description = (request.form.get(\"description\")", "    description = (form.get(\"description\")")
text = text.replace("    skip_ocr = request.form.get(\"skip_ocr\", \"\").lower()", "    skip_ocr = (form.get(\"skip_ocr\") or \"\").lower()")
text = text.replace("        chunk_size = int(request.form.get(\"chunk_size\")", "        chunk_size = int(form.get(\"chunk_size\")")
text = text.replace("        chunk_overlap = int(request.form.get(\"chunk_overlap\")", "        chunk_overlap = int(form.get(\"chunk_overlap\")")
text = text.replace("                    f.save(tmp_path)", "                    await write_upload_to_disk(f, tmp_path)")

# 8) 在 upload 开头用 collect_upload_files_from_form 初始化 file_list（替换原 file_list 构建前几行）
old_block = """    # 支持多文件：收集所有名为 file / file[] / file[0],file[1] / files 的表单文件
    file_list = []
    for key in form.keys():
        if key in ("file", "file[]", "files") or (key.startswith("file[") and key.endswith("]")):
            file_list.extend(form.getlist(key))
    if not file_list:
        single = form.get("file")
        if single and single.filename:
            file_list = [single]
"""
new_block = """    # 支持多文件：收集所有名为 file / file[] / file[0] / files 的表单文件
    file_list = collect_upload_files_from_form(form)
"""
if old_block in text:
    text = text.replace(old_block, new_block)
else:
    # 可能未完全匹配，尝试仅替换循环
    pass

# 9) jsonify 变 tuple：return ({...}) 已是 dict；原 return jsonify 已变成 return ( — 需把结尾 `)` 改成 `)` 对多行 return
# 脚本把 `return jsonify({` 变成了 `return (({` — 检查
text = text.replace("return (({", "return ({")
text = text.replace("return ((", "return (")

path.write_text(text, encoding="utf-8")
print("patched", path)
