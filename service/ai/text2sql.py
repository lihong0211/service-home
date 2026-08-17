#!/usr/bin/env python3
"""
Text2SQL：自然语言转 SQL 并执行。
只读 SELECT 直接执行；写操作（INSERT/UPDATE/DELETE 等）不再直接拒绝，而是接入
【人工处理】的 interrupt() 机制暂停执行，等人工审核批准（可编辑 SQL 后批准）才会
真正执行——跟 service/ai/langchain.py 的 HITL 演示图是同一套技术方案，这里是它在
"有真实副作用的场景"里的落地：批准了是真的会写 ai 库，不是摆设。
"""

import os
import re
import sqlite3
import uuid
from typing import Literal, TypedDict
from urllib.parse import quote_plus

import anyio.from_thread
from fastapi import Request
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from utils.http_body import query_dict, read_json_optional
from utils.response import api_response
from sqlalchemy import create_engine, text, inspect
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

from config.db import DB_AI_CONFIG
from config.ai import DEFAULT_CHAT_MODEL


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


_SQL_FENCE_PATTERN = re.compile(r"```(?:sql)?\s*(.+?)```", re.IGNORECASE | re.DOTALL)


def _extract_sql_from_answer_text(answer: str) -> str:
    """
    兜底：写操作场景下，Agent 有时会推理出正确的 SQL 却只把它写进最终文字回答（```sql ...```
    代码块）里，不调用 sql_db_query 工具真正执行——大概率是模型自己对"执行 DML"仍有保留，
    即便系统提示词已经放开了限制。这种情况下 intermediate_steps 里提取不到 SQL，
    只能退而求其次从回答文本里正则抠出代码块。取最后一个匹配（Agent 常先给分析再给结论）。
    """
    matches = _SQL_FENCE_PATTERN.findall(answer or "")
    if not matches:
        return ""
    return matches[-1].strip().rstrip(";").strip()


#  create_sql_agent 默认系统提示词（SQL_PREFIX）里硬编码了一行
#  "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)"——不是我们代码层面
#  拦截的，是 LLM 看到这行提示后直接拒绝生成，连 SQL 都不产出，实测确认过（问它"插入一条
#  数据"，Agent 直接用自然语言回答"不支持"，压根没调用 SQL 工具）。写操作现在改成走人工
#  审核而不是硬拒绝，所以生成阶段要把这条限制换掉，否则 review 节点永远看不到写操作 SQL。
_SQL_PREFIX_ALLOW_WRITE = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

You MAY generate DML statements (INSERT, UPDATE, DELETE) when the user explicitly asks to change data. \
A human will review and approve the statement before it is actually executed against the database — you \
are only responsible for producing a syntactically correct statement, not for deciding whether it is safe.

If the question does not seem related to the database, just return "I don't know" as the answer."""


class _ReadOnlySQLDatabase(SQLDatabase):
    """generate 阶段专用：LangChain 的 sql_db_query 工具会真的拿 db.run() 去执行 SQL——
    Agent 为了"验证自己写的对不对"经常会主动调用这个工具，哪怕问题是"插入一条数据"，
    实测确认过（同一条 INSERT 在 generate 阶段先真跑一次，approve 之后 _t2s_execute 又
    真跑一次，agent_booking 里插出两行重复数据）。这就绕过了人工审核——review 节点根本
    没机会拦截，数据已经改了。所以 generate 阶段把 db.run() 拦下来，写操作一律拒绝真正
    执行，只允许 Agent 把 SQL 文本写进最终答案，真正的执行留给 approve 之后的 _t2s_execute。
    """

    def run(self, command, *args, **kwargs):
        if not _is_read_only_sql(command if isinstance(command, str) else str(command)):
            return "该工具在生成阶段不会真正执行写操作 SQL，需要人工审核通过后才会执行。请直接在最终答案中给出这条 SQL。"
        return super().run(command, *args, **kwargs)


def _generate_sql(question: str, model: str = DEFAULT_CHAT_MODEL, allow_write: bool = False) -> dict:
    """
    只负责"自然语言 -> SQL"，不执行。返回 {sql, answer, error}：
    - 生成失败：error 有值
    - Agent 直接给了文字答案、没有走 SQL 查询（比如纯聊天式提问）：answer 有值，sql 为空
    - 正常生成出 SQL：sql 有值
    allow_write=True 时替换默认系统提示词，允许 Agent 生成 DML（配合 review 节点的人工
    审核使用；allow_write=False 保持 LangChain 默认行为，Agent 自己就会拒绝生成 DML，
    旧版 text2sql_run 用这个默认值，行为不变）。allow_write=True 时还要用
    _ReadOnlySQLDatabase，防止 Agent 自己在生成阶段就把写操作真执行了（见其类注释）。
    """
    if not (question or "").strip():
        return {"sql": "", "answer": "", "error": "请提供 question"}
    db_cls = _ReadOnlySQLDatabase if allow_write else SQLDatabase
    db = db_cls.from_uri(_ai_db_uri())
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
        prefix=_SQL_PREFIX_ALLOW_WRITE if allow_write else None,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    try:
        response = agent_executor.invoke({"input": question.strip()})
    except Exception as e:
        return {"sql": "", "answer": "", "error": f"生成 SQL 失败: {e}"}
    steps = response.get("intermediate_steps", [])
    sql = _extract_sql_from_agent_steps(steps)
    output = response.get("output", "")
    if not sql and allow_write:
        sql = _extract_sql_from_answer_text(output)
    if not sql:
        return {"sql": "", "answer": output, "error": None if output else "未能解析出有效 SQL"}
    return {"sql": sql, "answer": "", "error": None}


def _execute_select(sql: str, max_rows: int = 500) -> dict:
    """执行只读 SELECT，自动补 LIMIT。"""
    engine = _ai_engine()
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


def _execute_write(sql: str) -> dict:
    """
    执行写操作 SQL（INSERT/UPDATE/DELETE 等），只有经过人工审核批准才会走到这里。
    用 engine.begin()（不是 connect()）——SQLAlchemy 2.x 下 begin() 才会在成功退出时自动
    提交事务，connect() 默认不提交，写操作用 connect() 会看起来"执行成功"但实际没落库。
    """
    engine = _ai_engine()
    try:
        with engine.begin() as conn:
            result = conn.execute(text(sql))
            affected = result.rowcount if result.rowcount is not None else 0
        return {"sql": sql, "data": [{"affected_rows": affected}], "error": None}
    except Exception as e:
        return {"sql": sql, "data": [], "error": f"执行 SQL 失败: {e}"}


def text2sql_run(question: str, model: str = DEFAULT_CHAT_MODEL, max_rows: int = 500) -> dict:
    """
    旧版一体化入口：生成 + 执行，写操作直接拒绝（不接人工审核）。
    保留给还在用 /ai/text2sql 这个老接口的调用方；新调用方应该用
    run_text2sql_graph（对应 /ai/text2sql/run + /ai/text2sql/resume），写操作会走人工审核
    而不是被硬拒绝。
    """
    gen = _generate_sql(question, model)
    if gen.get("error"):
        return {"sql": "", "data": [], "error": gen["error"]}
    if not gen.get("sql"):
        answer = gen.get("answer", "")
        return {"sql": "", "data": [{"answer": answer}] if answer else [], "error": None if answer else "未能解析出有效 SQL"}
    sql = gen["sql"]
    if not _is_read_only_sql(sql):
        return {"sql": sql, "data": [], "error": "仅支持 SELECT 查询，禁止写操作（如需写操作请用 /ai/text2sql/run，会走人工审核）"}
    return _execute_select(sql, max_rows)


# ---------------------------------------------------------------------------
# 【模块：人工处理 + 权限控制】写操作 SQL 的审批流程。
# generate（生成 SQL，不执行）→ review（SELECT 直接放行；写操作 interrupt() 暂停，
# 等人工审核批准/拒绝/编辑）→ execute（真正执行）。跟 service/ai/langchain.py 的 HITL
# 演示图是同一套 interrupt()/Command/SqliteSaver 技术方案，区别是这里 execute 节点
# 是真实的数据库写操作，不是打印一行字符串。
# ---------------------------------------------------------------------------


class Text2SqlState(TypedDict):
    question: str
    model: str
    max_rows: int
    sql: str
    answer: str
    decision: str
    data: list
    error: str | None
    executed: bool  # 幂等标记，见 _t2s_execute 的注释


def _t2s_generate(state: Text2SqlState) -> dict:
    gen = _generate_sql(state["question"], state.get("model") or DEFAULT_CHAT_MODEL, allow_write=True)
    return {"sql": gen.get("sql", ""), "answer": gen.get("answer", ""), "error": gen.get("error")}


def _t2s_review(state: Text2SqlState) -> Command[Literal["execute", "__end__"]]:
    """SELECT 直接放行执行；写操作暂停等人工审核——权限控制 confirm 档的落地点。"""
    if state.get("error") or not state.get("sql"):
        return Command(goto=END)
    sql = state["sql"]
    if _is_read_only_sql(sql):
        return Command(goto="execute", update={"decision": "approved"})
    decision = interrupt({
        "question": "检测到写操作 SQL，是否批准执行？可编辑 SQL 后批准，或拒绝。",
        "sql": sql,
    })
    if isinstance(decision, str) and decision.strip():
        return Command(goto="execute", update={"decision": "approved", "sql": decision.strip()})
    if decision:
        return Command(goto="execute", update={"decision": "approved"})
    return Command(goto=END, update={"decision": "rejected", "error": "已拒绝执行该 SQL"})


def _t2s_execute(state: Text2SqlState) -> dict:
    """
    实测发现：resume 之后这个节点会被调用两次，导致写操作 SQL 真的执行了两遍（同一条
    INSERT 插进两行）——LangGraph "中断节点 resume 时从头重新执行" 是文档明确写的行为
    （见 docs/langgraph/langgraph-interrupts.md），但按官方示例 review→process 这种
    Command 动态路由不应该让下游节点也跑两次；具体是 checkpointer/多进程时序的哪个环节
    导致的没有继续深挖，改成对有真实副作用的这个节点做幂等防御——这是无论根因是什么都该
    做的事，跟【副作用回滚】那节 Saga 补偿事务同一个"生产系统要能扛住重复执行"的思路。
    """
    if state.get("executed"):
        return {}
    sql = state["sql"]
    out = _execute_select(sql, state.get("max_rows") or 500) if _is_read_only_sql(sql) else _execute_write(sql)
    return {"sql": out["sql"], "data": out["data"], "error": out["error"], "executed": True}


def _get_text2sql_checkpointer() -> SqliteSaver:
    """跟 HITL 的 _get_hitl_checkpointer 同一个模式：多 worker 部署必须用共享文件，不能用
    进程内存的 MemorySaver，否则 interrupt 后的 resume 请求打到另一个 worker 就找不到暂停状态。"""
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "checkpoints")
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(db_dir, "text2sql.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_text2sql_graph(checkpointer=None):
    builder = StateGraph(Text2SqlState)
    builder.add_node("generate", _t2s_generate)
    builder.add_node("review", _t2s_review, destinations=("execute", END))
    builder.add_node("execute", _t2s_execute)
    builder.set_entry_point("generate")
    builder.add_edge("generate", "review")
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=checkpointer or _get_text2sql_checkpointer())


# 模块级单例：跟 HITL 的 _HITL_GRAPH 一样，interrupt/resume 依赖同一个编译后的图对象 +
# 同一个 checkpointer 实例，不能像 router/loop/parallel 那样每次请求重新 build。
_TEXT2SQL_GRAPH = build_text2sql_graph()


def run_text2sql_graph(input_state: dict | None, thread_id: str, resume=None) -> dict:
    """
    执行/恢复 Text2SQL 审批图。首次调用不传 resume：SELECT 直接返回结果；写操作命中
    interrupt 后返回 waitingForInput=True + interrupt payload（含建议 SQL）。
    第二次调用带上同一个 thread_id + resume（人工决定），从暂停点继续执行到 END。
    """
    graph = _TEXT2SQL_GRAPH
    config = {"configurable": {"thread_id": thread_id}}
    if resume is not None:
        run_input = Command(resume=resume)
    else:
        default = {
            "question": "", "model": DEFAULT_CHAT_MODEL, "max_rows": 500,
            "sql": "", "answer": "", "decision": "", "data": [], "error": None, "executed": False,
        }
        run_input = {**default, **(input_state or {})}
    result = graph.invoke(run_input, config=config)
    interrupts = result.get("__interrupt__")
    if interrupts:
        first = interrupts[0]
        payload = getattr(first, "value", first)
        return {"threadId": thread_id, "waitingForInput": True, "interrupt": payload, "sql": result.get("sql", "")}
    return {
        "threadId": thread_id,
        "waitingForInput": False,
        "sql": result.get("sql", ""),
        "answer": result.get("answer", ""),
        "data": result.get("data", []),
        "error": result.get("error"),
        "decision": result.get("decision", ""),
    }


def text2sql_api(request: Request):
    data = anyio.from_thread.run(read_json_optional, request) or {}
    question = (data.get("question") or data.get("query") or "").strip()
    model = (data.get("model") or DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
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


def text2sql_hitl_api(request: Request):
    """
    POST /ai/text2sql/run  body: {question, threadId?, model?, max_rows?}
    只读 SELECT 直接返回执行结果（waitingForInput=false）；写操作 SQL 暂停，返回
    waitingForInput=true + interrupt（含建议 SQL），需要调 /ai/text2sql/resume 批准/
    编辑/拒绝后才会真正执行。不传 threadId 时生成一个新的（每次都是新会话）。
    """
    data = anyio.from_thread.run(read_json_optional, request) or {}
    question = (data.get("question") or data.get("query") or "").strip()
    if not question:
        return ({"code": 400, "msg": "请提供 question", "data": None}, 400)
    thread_id = data.get("threadId") or data.get("thread_id") or str(uuid.uuid4())
    model = (data.get("model") or DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
    max_rows = data.get("max_rows", 500)
    try:
        max_rows = max(1, min(2000, int(max_rows)))
    except (TypeError, ValueError):
        max_rows = 500
    try:
        out = run_text2sql_graph({"question": question, "model": model, "max_rows": max_rows}, thread_id)
    except Exception as e:
        return ({"code": 400, "msg": str(e), "data": None}, 400)
    return {"code": 0, "msg": "ok", "data": out}


def text2sql_resume_api(request: Request):
    """POST /ai/text2sql/resume  body: {threadId, resume: true(批准)/false(拒绝)/"编辑后的SQL"(批准并采用)}"""
    data = anyio.from_thread.run(read_json_optional, request) or {}
    thread_id = data.get("threadId") or data.get("thread_id")
    if not thread_id:
        return ({"code": 400, "msg": "请提供 threadId", "data": None}, 400)
    resume_value = data.get("resume")
    try:
        out = run_text2sql_graph(None, thread_id, resume=resume_value)
    except Exception as e:
        return ({"code": 400, "msg": str(e), "data": None}, 400)
    return {"code": 0, "msg": "ok", "data": out}


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


def table_data_api(request: Request):
    if request.method == "GET":
        params = query_dict(request)
    else:
        params = anyio.from_thread.run(read_json_optional, request) or {}
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
