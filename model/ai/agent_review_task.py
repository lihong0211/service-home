"""
【人工审核】集中审核列表：把 text2sql 写操作、HITL 演示等所有 interrupt() 命中的待审核项
统一落到这张表，供 /ai/review/* 系列接口和前端"人工审核列表"页面查询/审批，取代原来
分散在各自页面里各写一套的审核卡片。source 区分是哪个业务的审核项（text2sql/hitl），
approve/reject 时按 source 分发去调用对应的 run_text2sql_graph/run_hitl_graph 完成 resume。
"""
from sqlalchemy import Column, String, Text, DateTime
from app.app import db
from model.common.base_model import BaseModel


class AgentReviewTask(db.Model, BaseModel):
    __tablename__ = "agent_review_task"
    __bind_key__ = "ai"

    source = Column(String(20), nullable=False, comment="审核来源: text2sql/hitl")
    thread_id = Column(String(100), nullable=False, comment="对应图执行的 thread_id，resume 时要用")
    question = Column(Text, nullable=True, comment="审核提示文案，来自 interrupt() payload 的 question 字段")
    content = Column(Text, nullable=True, comment="待审核的具体内容：text2sql 是 SQL 文本，hitl 是 AI 建议文本")
    warnings = Column(Text, nullable=True, comment="JSON 数组字符串，仅 text2sql 有——枚举值核对警告")
    status = Column(String(20), nullable=False, default="pending", comment="pending/approved/rejected")
    result = Column(Text, nullable=True, comment="审批后 resume 的结果摘要，JSON 字符串")
    resolved_at = Column(DateTime(), nullable=True, comment="审批完成时间")
