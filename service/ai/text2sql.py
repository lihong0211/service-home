#!/usr/bin/env python3
"""
Text2SQL：基于 ai 库表结构，将自然语言转为 SQL 并执行，返回改写后的 SQL 与执行结果。
- 仅执行 SELECT，禁止写操作
- 使用 DashScope 兼容接口生成 SQL
"""

import os
import re
from urllib.parse import quote_plus

from flask import request, jsonify
from sqlalchemy import create_engine, text, inspect
from openai import OpenAI

from config.db import DB_AI_CONFIG


# ai 库连接 URL（与 app 中 ai bind 一致）
def _ai_engine():
    encoded = quote_plus(DB_AI_CONFIG["password"])
    url = (
        f"mysql+pymysql://{DB_AI_CONFIG['user']}:{encoded}"
        f"@{DB_AI_CONFIG['host']}:{DB_AI_CONFIG['port']}/{DB_AI_CONFIG['database']}"
        f"?charset={DB_AI_CONFIG.get('charset', 'utf8mb4')}"
    )
    return create_engine(url, pool_pre_ping=True)


# 表名只允许字母、数字、下划线（防注入，且与 data.sql 中表名一致）
TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


def _get_allowed_tables(engine) -> list:
    """获取 ai 库中所有表名（白名单）。"""
    return inspect(engine).get_table_names()


def _get_schema_description(engine) -> str:
    """获取 ai 库表及列信息，供 LLM 生成 SQL。"""
    insp = inspect(engine)
    tables = insp.get_table_names()
    parts = []
    for t in sorted(tables):
        cols = insp.get_columns(t)
        col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
        parts.append(f"- {t}: {col_desc}")
    return "\n".join(parts) if parts else "（无表）"


def _is_read_only_sql(sql: str) -> bool:
    """只允许 SELECT，禁止 INSERT/UPDATE/DELETE 等写操作。"""
    s = re.sub(r"\s+", " ", sql.strip()).upper()
    s = s.lstrip("; ")
    if not s:
        return False
    # 允许 SELECT；禁止写操作与 DDL
    for keyword in ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "REPLACE"):
        if keyword in s and not s.startswith("SELECT"):
            return False
    return s.startswith("SELECT")


def _extract_sql_from_llm_response(content: str) -> str:
    """从 LLM 返回中提取单条 SQL（去除 markdown 代码块等）。"""
    content = (content or "").strip()
    # 去掉 ```sql ... ``` 或 ``` ... ```
    for marker in ("```sql", "```"):
        if marker in content:
            start = content.find(marker) + len(marker)
            end = content.find("```", start)
            if end == -1:
                end = len(content)
            content = content[start:end]
    return content.strip().rstrip(";").strip()


def text2sql_run(question: str, model: str = "qwen-turbo", max_rows: int = 500) -> dict:
    """
    根据自然语言问题生成 SQL 并执行，仅允许 SELECT。
    :return: {"sql": str, "data": list[dict], "error": str | None}
    """
    if not (question or "").strip():
        return {"sql": "", "data": [], "error": "请提供 question"}
    engine = _ai_engine()
    schema_desc = _get_schema_description(engine)
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60.0,
    )
    prompt = f"""你是一个 MySQL 专家。根据以下数据库表结构，生成一条且仅一条 SELECT 查询语句来回答用户问题。不要解释，只输出 SQL。

数据库表结构（ai 库）：
{schema_desc}

用户问题：{question.strip()}

要求：只输出一条 SELECT 语句，不要用 markdown 包裹，不要分号结尾。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        raw_sql = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"sql": "", "data": [], "error": f"生成 SQL 失败: {e}"}
    sql = _extract_sql_from_llm_response(raw_sql)
    if not sql:
        return {"sql": raw_sql or "", "data": [], "error": "未能解析出有效 SQL"}
    if not _is_read_only_sql(sql):
        return {"sql": sql, "data": [], "error": "仅支持 SELECT 查询，禁止写操作"}
    try:
        with engine.connect() as conn:
            # 可选：限制返回行数，防止大结果集
            if max_rows and " LIMIT " not in sql.upper():
                sql = f"{sql.rstrip()} LIMIT {max_rows}"
            result = conn.execute(text(sql))
            rows = result.mappings().fetchall()
            data = [dict(r) for r in rows]
        return {"sql": sql, "data": data, "error": None}
    except Exception as e:
        return {"sql": sql, "data": [], "error": f"执行 SQL 失败: {e}"}


def text2sql_api():
    """
    POST /ai/text2sql
    Body: { "question": "自然语言问题", "model": "qwen-turbo", "max_rows": 500 }
    返回：{ "sql": "改写后的 SQL", "data": 执行结果列表, "error": 错误信息或 null }
    """
    data = request.get_json() or {}
    question = (data.get("question") or data.get("query") or "").strip()
    model = (data.get("model") or "qwen-turbo").strip() or "qwen-turbo"
    max_rows = data.get("max_rows", 500)
    try:
        max_rows = max(1, min(2000, int(max_rows)))
    except (TypeError, ValueError):
        max_rows = 500
    out = text2sql_run(question=question, model=model, max_rows=max_rows)
    if out.get("error"):
        user_error = not any(x in (out.get("error") or "") for x in ("生成 SQL 失败", "执行 SQL 失败"))
        code = 400 if user_error else 500
        return jsonify({
            "code": code,
            "msg": out["error"],
            "data": {"sql": out.get("sql", ""), "data": out.get("data", [])},
        }), code
    return jsonify({
        "code": 0,
        "msg": "ok",
        "data": {"sql": out["sql"], "data": out["data"]},
    })


def table_data_run(table_name: str, page: int = 1, page_size: int = 20) -> dict:
    """
    按表名 + 分页查询 ai 库原始数据，返回该表所有字段。
    :return: {"table": str, "columns": list[str], "data": list[dict], "total": int, "page": int, "page_size": int, "error": str | None}
    """
    if not (table_name or "").strip():
        return {"table": "", "columns": [], "data": [], "total": 0, "page": 1, "page_size": 20, "error": "请提供表名"}
    table_name = table_name.strip()
    if not TABLE_NAME_PATTERN.match(table_name):
        return {"table": table_name, "columns": [], "data": [], "total": 0, "page": page, "page_size": page_size, "error": "表名仅允许字母、数字、下划线"}
    page = max(1, int(page)) if page else 1
    page_size = max(1, min(500, int(page_size))) if page_size else 20
    engine = _ai_engine()
    allowed = _get_allowed_tables(engine)
    if table_name not in allowed:
        return {"table": table_name, "columns": [], "data": [], "total": 0, "page": page, "page_size": page_size, "error": f"表不存在或不可访问，可选表：{', '.join(sorted(allowed))}"}
    insp = inspect(engine)
    columns = [c["name"] for c in insp.get_columns(table_name)]
    offset = (page - 1) * page_size
    try:
        with engine.connect() as conn:
            # MySQL 表名用反引号包裹，避免保留字
            count_sql = text(f"SELECT COUNT(*) AS cnt FROM `{table_name}`")
            total = conn.execute(count_sql).scalar() or 0
            select_sql = text(f"SELECT * FROM `{table_name}` LIMIT {page_size} OFFSET {offset}")
            result = conn.execute(select_sql)
            rows = result.mappings().fetchall()
            data = [dict(r) for r in rows]
        return {"table": table_name, "columns": columns, "data": data, "total": total, "page": page, "page_size": page_size, "error": None}
    except Exception as e:
        return {"table": table_name, "columns": columns, "data": [], "total": 0, "page": page, "page_size": page_size, "error": f"查询失败: {e}"}


def table_data_api():
    """
    GET 或 POST /ai/table-data
    参数：table（表名，必填）、page（默认 1）、page_size（默认 20，最大 500）
    返回：该表所有字段及分页数据。{ "table", "columns", "data", "total", "page", "page_size" }
    """
    if request.method == "GET":
        params = request.args
    else:
        params = request.get_json() or {}
    table_name = (params.get("table") or params.get("table_name") or "").strip()
    page = params.get("page", 1)
    page_size = params.get("page_size", 20)
    try:
        page = max(1, int(page))
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = max(1, min(500, int(page_size)))
    except (TypeError, ValueError):
        page_size = 20
    out = table_data_run(table_name=table_name, page=page, page_size=page_size)
    if out.get("error"):
        return jsonify({"code": 400, "msg": out["error"], "data": out}), 400
    return jsonify({"code": 0, "msg": "ok", "data": {k: v for k, v in out.items() if k != "error"}})
