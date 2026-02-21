#!/usr/bin/env python3
"""Text2SQL：自然语言转 SQL 并执行，仅允许 SELECT。"""

import os
import re
from urllib.parse import quote_plus

from flask import request

from utils.response import api_response
from sqlalchemy import create_engine, text, inspect
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

from config.db import DB_AI_CONFIG


def _ai_db_uri() -> str:
    encoded = quote_plus(DB_AI_CONFIG["password"])
    return (
        f"mysql+pymysql://{DB_AI_CONFIG['user']}:{encoded}"
        f"@{DB_AI_CONFIG['host']}:{DB_AI_CONFIG['port']}/{DB_AI_CONFIG['database']}"
        f"?charset={DB_AI_CONFIG.get('charset', 'utf8mb4')}"
    )


def _ai_engine():
    return create_engine(_ai_db_uri(), pool_pre_ping=True)


TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


def _is_read_only_sql(sql: str) -> bool:
    s = re.sub(r"\s+", " ", sql.strip()).upper()
    s = s.lstrip("; ")
    if not s:
        return False
    for keyword in (
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "REPLACE",
    ):
        if keyword in s and not s.startswith("SELECT"):
            return False
    return s.startswith("SELECT")


def _extract_sql_from_agent_steps(intermediate_steps: list) -> str:
    sql = ""
    for step in intermediate_steps or []:
        action, _ = step
        if getattr(action, "tool", "") == "sql_db_query":
            inp = getattr(action, "tool_input", None)
            if isinstance(inp, dict):
                sql = (inp.get("query") or inp.get("sql") or "").strip()
            elif isinstance(inp, str):
                sql = inp.strip()
    return sql


def text2sql_run(question: str, model: str = "qwen-turbo", max_rows: int = 500) -> dict:
    if not (question or "").strip():
        return {"sql": "", "data": [], "error": "请提供 question"}
    engine = _ai_engine()
    db = SQLDatabase.from_uri(_ai_db_uri())
    llm = ChatOpenAI(
        model=model,
        temperature=0.01,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        request_timeout=120,
    )
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=False,
        top_k=5,
        max_iterations=4,
        max_execution_time=180,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    try:
        response = agent_executor.invoke(
            {"input": question.strip()},
        )
    except Exception as e:
        return {"sql": "", "data": [], "error": f"生成 SQL 失败: {e}"}
    steps = response.get("intermediate_steps", [])
    sql = _extract_sql_from_agent_steps(steps)
    if not sql:
        output = response.get("output", "")
        return {
            "sql": "",
            "data": [{"answer": output}] if output else [],
            "error": None if output else "未能解析出有效 SQL",
        }
    if not _is_read_only_sql(sql):
        return {"sql": sql, "data": [], "error": "仅支持 SELECT 查询，禁止写操作"}
    try:
        with engine.connect() as conn:
            if max_rows and " LIMIT " not in sql.upper():
                sql = f"{sql.rstrip().rstrip(';')} LIMIT {max_rows}"
            result = conn.execute(text(sql))
            rows = result.mappings().fetchall()
            data = [dict(r) for r in rows]
        return {"sql": sql, "data": data, "error": None}
    except Exception as e:
        return {"sql": sql, "data": [], "error": f"执行 SQL 失败: {e}"}


def text2sql_api():
    data = request.get_json() or {}
    question = (data.get("question") or data.get("query") or "").strip()
    model = (data.get("model") or "qwen-turbo").strip() or "qwen-turbo"
    max_rows = data.get("max_rows", 500)
    try:
        max_rows = max(1, min(2000, int(max_rows)))
    except (TypeError, ValueError):
        max_rows = 500
    out = text2sql_run(question=question, model=model, max_rows=max_rows)
    return api_response(
        out,
        error_code_fn=lambda o: (
            400
            if not any(
                x in (o.get("error") or "") for x in ("生成 SQL 失败", "执行 SQL 失败")
            )
            else 500
        ),
        error_data={"sql": out.get("sql", ""), "data": out.get("data", [])},
        success_data={"sql": out["sql"], "data": out["data"]},
    )


def table_data_run(table_name: str, page: int = 1, page_size: int = 20) -> dict:
    if not (table_name or "").strip():
        return {
            "table": "",
            "columns": [],
            "data": [],
            "total": 0,
            "page": 1,
            "page_size": 20,
            "error": "请提供表名",
        }
    table_name = table_name.strip()
    if not TABLE_NAME_PATTERN.match(table_name):
        return {
            "table": table_name,
            "columns": [],
            "data": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "error": "表名仅允许字母、数字、下划线",
        }
    page = max(1, int(page)) if page else 1
    page_size = max(1, min(500, int(page_size))) if page_size else 20
    engine = _ai_engine()
    allowed = inspect(engine).get_table_names()
    if table_name not in allowed:
        return {
            "table": table_name,
            "columns": [],
            "data": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "error": f"表不存在或不可访问，可选表：{', '.join(sorted(allowed))}",
        }
    insp = inspect(engine)
    columns = [c["name"] for c in insp.get_columns(table_name)]
    offset = (page - 1) * page_size
    try:
        with engine.connect() as conn:
            count_sql = text(f"SELECT COUNT(*) AS cnt FROM `{table_name}`")
            total = conn.execute(count_sql).scalar() or 0
            select_sql = text(
                f"SELECT * FROM `{table_name}` LIMIT {page_size} OFFSET {offset}"
            )
            result = conn.execute(select_sql)
            rows = result.mappings().fetchall()
            data = [dict(r) for r in rows]
        return {
            "table": table_name,
            "columns": columns,
            "data": data,
            "total": total,
            "page": page,
            "page_size": page_size,
            "error": None,
        }
    except Exception as e:
        return {
            "table": table_name,
            "columns": columns,
            "data": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "error": f"查询失败: {e}",
        }


def table_data_api():
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
    return api_response(out)
