"""
【模块 8／8：副作用回滚】Saga 补偿事务演示用的真实副作用表，代表"一次预订"。
create_booking 这一步会真实写入一行；如果后续 charge_payment 步骤失败，
Saga 编排器会调用对应的 undo 把这一行删掉——这是本项目里唯一一个真的有
外部（数据库）副作用、可以拿来验证"回滚真的发生了"的表，其余节点都是只读演示。
"""
from sqlalchemy import Column, String, Integer
from app.app import db
from model.common.base_model import BaseModel


class AgentBooking(db.Model, BaseModel):
    __tablename__ = "agent_booking"
    __bind_key__ = "ai"

    thread_id = Column(String(100), nullable=True, comment="所属会话/请求标识")
    item = Column(String(200), nullable=False, comment="预订项目")
    amount = Column(Integer, nullable=False, comment="金额，单位分")
    status = Column(String(20), nullable=False, default="pending", comment="pending/paid，行存在即代表未回滚")
