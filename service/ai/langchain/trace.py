"""
【模块 4／8：可观测性】+【模块 5／8：回流机制】共用同一张表 model/ai/agent_trace.py：
_persist_trace 是可观测性的采集入口（每次图运行落一条，记录耗时/步骤/成败）；
submit_trace_feedback/list_bad_cases 是回流机制的采集+挖掘入口（人工/用户标注 好评/差评，
差评连同天然的 status=error 一起构成 bad case 池，供后续回流进知识库/prompt/训练集——
这三处回流目的地本身是人工/离线流程，本项目只负责把"哪些是 bad case"这一步做实）。
完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 对应两节。
"""

import json
import logging

logger = logging.getLogger(__name__)

def _persist_trace(
    graph_name: str,
    state: dict,
    run_result: dict | None,
    duration_ms: int,
    error: str | None = None,
) -> int | None:
    """
    【可观测性】把一次图执行（invoke 或 stream 一轮）落一条 trace 记录，供排查"哪一步慢/哪一步错"
    和后续 bad case 回流分析用。用独立 session（SessionLocal），不读请求级 db.session：SSE 分支
    落库发生在 StreamingResponse 对象返回之后，此时 _ai_route 已经 clear_request_session，
    db.session 会抛 RuntimeError。落库失败只记日志，不抛出，不影响主流程。
    返回新插入行的 id（落库失败时返回 None）——【回流机制】前端要靠这个 id 调
    /ai/langgraph/trace/feedback 给这次回答打反馈，所以调用方要把它透传进响应体。
    """
    from app.database import SessionLocal
    from model.ai.agent_trace import AgentTrace

    steps = (run_result or {}).get("steps", [])
    final_state = (run_result or {}).get("finalState", {}) or {}
    session = SessionLocal()
    try:
        row = AgentTrace(
            graph_name=graph_name,
            thread_id=state.get("threadId") or state.get("thread_id"),
            user_id=state.get("user_id"),
            # 输入/输出是回流分析的底子，截断上限放宽到 15000（TEXT 列 65535 字节，utf8mb4 最坏
            # 4 字节/字符时安全上限约 16383 字符），避免长回答被 2000 字符截断丢信息
            input_summary=(state.get("query") or state.get("input_text") or "")[:15000],
            output_summary=(final_state.get("response") or "")[:15000],
            status="error" if error else "success",
            error_message=error,
            total_steps=len(steps),
            duration_ms=duration_ms,
            steps_detail=json.dumps(
                [{"nodeId": s.get("nodeId"), "duration_ms": s.get("duration_ms")} for s in steps],
                ensure_ascii=False,
            ),
        )
        session.add(row)
        session.commit()
        return row.id
    except Exception:
        logger.exception("agent trace 落库失败，不影响主流程")
        session.rollback()
        return None
    finally:
        session.close()


def submit_trace_feedback(trace_id: int, rating: str, note: str = "") -> bool:
    """
    【回流机制】采集入口：给一条已存在的 trace 打显式反馈（good/bad）。这是飞轮的第一步——
    没有反馈信号，后面"哪些是 bad case"就无从谈起。rating 只接受 good/bad，其他值直接拒绝
    （不做静默纠正，调用方传错参数应该显式失败，而不是被悄悄改成别的值）。
    返回 True/False 表示是否成功命中并更新了一条记录。
    """
    if rating not in ("good", "bad"):
        raise ValueError(f"rating 只能是 good/bad，收到: {rating!r}")
    from app.database import SessionLocal
    from model.ai.agent_trace import AgentTrace

    session = SessionLocal()
    try:
        row = session.query(AgentTrace).filter(AgentTrace.id == trace_id).first()
        if row is None:
            return False
        row.feedback = rating
        row.feedback_note = note[:2000] if note else None
        session.commit()
        return True
    except Exception:
        logger.exception("trace 反馈写入失败")
        session.rollback()
        return False
    finally:
        session.close()


def mark_trace_copied(trace_id: int) -> bool:
    """
    【回流机制】隐式反馈采集：用户复制了这次回答，说明答案至少"有用到可以复制走"，
    跟点赞点踩一样是回流信号，但不需要用户主动评价——复制动作本身已经是信号。
    返回 True/False 表示是否成功命中并更新了一条记录。
    """
    from app.database import SessionLocal
    from model.ai.agent_trace import AgentTrace

    session = SessionLocal()
    try:
        row = session.query(AgentTrace).filter(AgentTrace.id == trace_id).first()
        if row is None:
            return False
        row.copied = True
        session.commit()
        return True
    except Exception:
        logger.exception("trace 复制信号写入失败")
        session.rollback()
        return False
    finally:
        session.close()


def list_traces(graph_name: str | None = None, status: str | None = None, limit: int = 50) -> list[dict]:
    """
    【可观测性】全量查询入口：不像 list_bad_cases 那样只挑 error/bad，这里是"这段时间到底
    跑了哪些请求、每个耗时多少、成功没成功"的完整视图，支撑排查慢请求/看整体健康度，
    而不只是回流场景的 bad case 挖掘。steps_detail 原样透传（前端展开显示每个节点耗时）。
    """
    from app.database import SessionLocal
    from model.ai.agent_trace import AgentTrace

    session = SessionLocal()
    try:
        query = session.query(AgentTrace)
        if graph_name:
            query = query.filter(AgentTrace.graph_name == graph_name)
        if status:
            query = query.filter(AgentTrace.status == status)
        rows = query.order_by(AgentTrace.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "graph_name": r.graph_name,
                "thread_id": r.thread_id,
                "user_id": r.user_id,
                "input_summary": r.input_summary,
                "output_summary": r.output_summary,
                "status": r.status,
                "error_message": r.error_message,
                "total_steps": r.total_steps,
                "duration_ms": r.duration_ms,
                "steps_detail": r.steps_detail,
                "feedback": r.feedback,
                "feedback_note": r.feedback_note,
                "copied": bool(r.copied),
                "created_at": r.create_at.isoformat() if r.create_at else None,
            }
            for r in rows
        ]
    finally:
        session.close()


def list_bad_cases(graph_name: str | None = None, limit: int = 50) -> list[dict]:
    """
    【回流机制】挖掘入口：拉取 bad case 候选池——status='error'（系统自己判定的失败）或
    feedback='bad'（人工/用户标注的不满意）任一命中即算。这是回流的第二步：从原始信号里
    筛出"值得回流"的样本。真正"回流去哪"（知识库补全 / prompt 新增约束 / 攒训练数据）是
    后续人工或离线批处理的事，不在这个函数职责内——这里只负责把候选池准确地找出来。
    """
    from app.database import SessionLocal
    from model.ai.agent_trace import AgentTrace

    session = SessionLocal()
    try:
        query = session.query(AgentTrace).filter(
            (AgentTrace.status == "error") | (AgentTrace.feedback == "bad")
        )
        if graph_name:
            query = query.filter(AgentTrace.graph_name == graph_name)
        rows = query.order_by(AgentTrace.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "graph_name": r.graph_name,
                "thread_id": r.thread_id,
                "user_id": r.user_id,
                "input_summary": r.input_summary,
                "output_summary": r.output_summary,
                "status": r.status,
                "error_message": r.error_message,
                "feedback": r.feedback,
                "feedback_note": r.feedback_note,
                "copied": bool(r.copied),
                "created_at": r.create_at.isoformat() if r.create_at else None,
            }
            for r in rows
        ]
    finally:
        session.close()
