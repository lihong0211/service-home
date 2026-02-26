"""service/ai 技术文档接口：按编号返回对应 Markdown 文件。"""
import os

from flask import send_file, jsonify

_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "docs",
    "service_ai",
)

_DOC_MAP = {
    0: "README.md",
    1: "01_vector_db.md",
    2: "02_rag.md",
    3: "03_knowledge_base.md",
    4: "04_agent.md",
    5: "05_a2a.md",
    6: "06_mcp.md",
    7: "07_chat.md",
    8: "08_tools.md",
    9: "09_finetuning.md",
}


def service_ai_doc_api(doc_id: int):
    """
    GET /ai/docs/<doc_id>

    返回 service/ai 对应编号的技术文档（Markdown 文件）。

    doc_id 对应关系：
      0  → README.md（目录总览）
      1  → 01_vector_db.md
      2  → 02_rag.md
      3  → 03_knowledge_base.md
      4  → 04_agent.md
      5  → 05_a2a.md
      6  → 06_mcp.md
      7  → 07_chat.md
      8  → 08_tools.md
      9  → 09_finetuning.md
    """
    filename = _DOC_MAP.get(doc_id)
    if not filename:
        return (
            jsonify({"code": 404, "msg": f"文档编号 {doc_id} 不存在，有效范围：0-9"}),
            404,
        )
    filepath = os.path.join(_DOCS_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"code": 500, "msg": f"文档文件未找到：{filename}"}), 500
    return send_file(filepath, mimetype="text/markdown; charset=utf-8")
