#!/usr/bin/env python3
"""一次性脚本：vector_db.py HTTP 接口去 compat。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "service" / "ai" / "vector_db.py"
text = path.read_text(encoding="utf-8")

text = text.replace("from compat import request, jsonify\n", "", 1)
if "from fastapi import Request" not in text:
    text = text.replace(
        "from openai import OpenAI\n",
        "from openai import OpenAI\n\n"
        "from fastapi import Request\n"
        "from utils.http_body import query_dict, read_json_optional\n",
        1,
    )

for name in (
    "list_api",
    "create_api",
    "detail_api",
    "update_api",
    "update_meta_api",
    "delete_api",
    "sync_from_disk_api",
    "rebuild_api",
    "documents_api",
    "document_add_api",
    "document_update_api",
    "document_delete_api",
    "categories_api",
    "category_add_api",
    "category_update_api",
    "category_delete_api",
    "search_api",
):
    text = re.sub(
        rf"^def {name}\(\):",
        f"async def {name}(request: Request):",
        text,
        flags=re.M,
    )

text = text.replace(
    "data = request.get_json(silent=True) or {}",
    "data = await read_json_optional(request) or {}",
)
text = text.replace("data = request.get_json() or {}", "data = await read_json_optional(request) or {}")
text = text.replace("request.args.get", "query_dict(request).get")
text = text.replace("return jsonify(", "return (")

path.write_text(text, encoding="utf-8")
print("patched", path)
