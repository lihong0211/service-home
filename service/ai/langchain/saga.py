"""
【模块 8／8：副作用回滚】补偿事务（Saga Pattern）——权限控制挡的是"执行前要不要批准"，
这里管的是"万一已经执行了、后面某一步又失败了，怎么把已经产生的副作用撤销掉"，两者互补。
完整设计说明见 service/ai/AGENT_ARCHITECTURE.md 「副作用回滚」一节。

核心思路：每个可能有副作用的正向动作（execute）都配一个补偿动作（undo），Saga 编排器按
顺序执行；任意一步失败，就把已经成功的步骤按**逆序**依次 undo，undo 拿的是对应 execute
当初返回的结果（比如新插入行的主键），而不是重新猜状态。

下面 AgentBooking（真实 MySQL 表）+ create_booking/charge_payment 两步是唯一一个真的会
产生外部（数据库）副作用的演示场景，用来验证"回滚真的发生了"，不是纸面上的接口设计——
项目里其余节点全部只读，没有真实副作用可回滚，硬造 undo 接口给它们只是摆设，没有意义。
"""

import logging
import uuid

import anyio.from_thread
from fastapi import Request

from utils.http_body import read_json_optional

logger = logging.getLogger(__name__)

class SagaStep:
    """一步 Saga：execute 是正向动作，undo 是对应的补偿动作，undo 接收 execute 的返回值。"""

    def __init__(self, name: str, execute, undo):
        self.name = name
        self.execute = execute
        self.undo = undo


def run_saga(steps: list) -> dict:
    """
    Saga 编排器：按顺序跑 steps；任意一步抛异常，立刻停止往后执行，把已完成的步骤按逆序
    依次调用 undo 补偿。某一步的 undo 本身也失败时不会中断其余补偿（尽量撤销更多），但会
    记进日志——这种情况在真实系统里代表"需要人工介入核对账目"，补偿机制不是万能的。
    返回 {"status": "committed"|"rolled_back", "completed": [...], "compensated": [...], "error": str|None}
    """
    completed = []  # [(step, execute返回值)]
    for step in steps:
        try:
            result = step.execute()
            completed.append((step, result))
            print(f"✅ Saga step 执行成功: {step.name}")
        except Exception as e:
            print(f"❌ Saga step 失败: {step.name}，错误: {e}，开始逆序回滚已完成步骤")
            compensated = []
            for done_step, done_result in reversed(completed):
                try:
                    done_step.undo(done_result)
                    compensated.append(done_step.name)
                    print(f"↩️ 已回滚: {done_step.name}")
                except Exception:
                    logger.exception(f"补偿失败: {done_step.name}，需要人工介入核对")
            return {
                "status": "rolled_back",
                "completed": [s.name for s, _ in completed],
                "compensated": compensated,
                "error": str(e),
            }
    return {"status": "committed", "completed": [s.name for s, _ in completed], "compensated": [], "error": None}


def _create_booking_step(thread_id: str, item: str, amount: int) -> SagaStep:
    """execute 真实插入一行 AgentBooking，返回主键；undo 按主键删除同一行——一一对应。"""

    def execute():
        from app.database import SessionLocal
        from model.ai.agent_booking import AgentBooking

        session = SessionLocal()
        try:
            booking = AgentBooking(thread_id=thread_id, item=item, amount=amount, status="pending")
            session.add(booking)
            session.commit()
            session.refresh(booking)
            return booking.id
        finally:
            session.close()

    def undo(booking_id):
        from app.database import SessionLocal
        from model.ai.agent_booking import AgentBooking

        session = SessionLocal()
        try:
            row = session.query(AgentBooking).filter(AgentBooking.id == booking_id).first()
            if row:
                session.delete(row)
                session.commit()
        finally:
            session.close()

    return SagaStep("create_booking", execute, undo)


def _charge_payment_step(should_fail: bool) -> SagaStep:
    """
    演示用扣款步骤：should_fail=True 时故意抛异常，模拟支付渠道失败，触发 run_saga 逆序回滚。
    undo 是 no-op——因为这一步从未真正提交成功过（execute 直接抛异常），没有可撤销的状态；
    真实接入支付渠道时，这里的 undo 才需要真的调 stripe.refunds.create 之类的退款接口。
    """

    def execute():
        if should_fail:
            raise RuntimeError("支付渠道返回失败（演示用：故意触发，用来验证补偿回滚）")
        return "charged"

    def undo(_result):
        pass

    return SagaStep("charge_payment", execute, undo)


def saga_demo_api(request: Request):
    """
    POST /ai/langgraph/saga-demo  body: {item?, amount?, failPayment?(默认true), threadId?}
    【副作用回滚】演示端点：两步 Saga——create_booking（真实写 MySQL）→ charge_payment。
    failPayment=true（默认）：第二步故意失败，触发回滚，返回后查 agent_booking 表应该
    查不到这次生成的行；failPayment=false：两步都成功，booking 行保留。
    """
    body = anyio.from_thread.run(read_json_optional, request) or {}
    item = (body.get("item") or "测试预订").strip()
    amount = int(body.get("amount") or 100)
    fail_payment = bool(body.get("failPayment", True))
    thread_id = body.get("threadId") or str(uuid.uuid4())
    steps = [
        _create_booking_step(thread_id, item, amount),
        _charge_payment_step(fail_payment),
    ]
    result = run_saga(steps)
    return {"code": 0, "msg": "ok", "data": {**result, "threadId": thread_id}}
