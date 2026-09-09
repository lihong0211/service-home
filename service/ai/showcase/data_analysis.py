#!/usr/bin/env python3
"""AI数据分析：上传 CSV/Excel，自然语言转 DuckDB SQL 查询。"""

import io
import os
import uuid
import json
import time

import duckdb
import pandas as pd
import dashscope
from fastapi import Request

from utils.http_body import read_json_optional
from config.ai import DEFAULT_CHAT_MODEL

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 内存会话存储：{session_id: {"conn": duckdb.Connection, "table": str, "ts": float}}
_sessions: dict = {}
_SESSION_TTL = 3600  # 1 hour


def _cleanup_sessions():
    now = time.time()
    expired = [k for k, v in _sessions.items() if now - v["ts"] > _SESSION_TTL]
    for k in expired:
        try:
            _sessions[k]["conn"].close()
        except Exception:
            pass
        del _sessions[k]


def _nl_to_sql(question: str, table_name: str, columns: list[dict]) -> str:
    col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
    prompt = (
        f"You are a DuckDB SQL expert. Generate a single valid DuckDB SQL SELECT statement.\n"
        f"Table name: {table_name}\n"
        f"Columns: {col_desc}\n"
        f"Question: {question}\n"
        f"Rules:\n"
        f"- Return ONLY the SQL statement, no explanation, no markdown, no backticks\n"
        f"- Use only SELECT statements\n"
        f"- Column names with spaces must be quoted with double quotes\n"
    )
    resp = dashscope.Generation.call(
        model=DEFAULT_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
        temperature=0,
    )
    if not resp or not resp.output or not resp.output.choices:
        raise ValueError("DashScope returned empty response")
    sql = (resp.output.choices[0].message.content or "").strip()
    # Remove markdown code fences if present
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.startswith("```")).strip()
    return sql


async def upload_data_file_api(request: Request):
    _cleanup_sessions()
    form = await request.form()
    file = form.get("file")
    if not file:
        return {"code": 400, "msg": "Missing file field"}

    filename = getattr(file, "filename", "") or ""
    content = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            return {"code": 400, "msg": "Unsupported file type. Use .csv, .xlsx, or .xls"}
    except Exception as e:
        return {"code": 400, "msg": f"Failed to parse file: {e}"}

    # Replace spaces in column names with underscores for easier SQL
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    session_id = str(uuid.uuid4())
    table_name = "data_table"
    conn = duckdb.connect(database=":memory:")
    conn.register(table_name, df)

    _sessions[session_id] = {"conn": conn, "table": table_name, "df": df, "ts": time.time()}

    columns = [{"name": c, "type": str(df[c].dtype)} for c in df.columns]
    preview = df.head(10).fillna("").values.tolist()

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "session_id": session_id,
            "columns": columns,
            "row_count": len(df),
            "preview": preview,
        },
    }


async def query_data_api(request: Request):
    body = await read_json_optional(request) or {}
    session_id = body.get("session_id", "")
    question = body.get("question", "").strip()

    if not session_id or not question:
        return {"code": 400, "msg": "Missing session_id or question"}

    session = _sessions.get(session_id)
    if not session:
        return {"code": 404, "msg": "Session not found or expired. Please re-upload the file."}

    session["ts"] = time.time()
    conn: duckdb.DuckDBPyConnection = session["conn"]
    table_name: str = session["table"]
    df: pd.DataFrame = session["df"]
    columns = [{"name": c, "type": str(df[c].dtype)} for c in df.columns]

    try:
        sql = _nl_to_sql(question, table_name, columns)
    except Exception as e:
        return {"code": 500, "msg": f"Failed to generate SQL: {e}"}

    try:
        result_df = conn.execute(sql).df()
    except Exception as e:
        return {"code": 400, "msg": f"SQL execution error: {e}", "data": {"sql": sql}}

    result_columns = list(result_df.columns)
    result_rows = result_df.fillna("").values.tolist()

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "sql": sql,
            "columns": result_columns,
            "rows": result_rows,
        },
    }
