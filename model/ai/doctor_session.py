"""
全科医生智能体的问诊会话索引，供"历史会话列表"展示用。
完整对话状态（messages/patient_info）仍在 LangGraph SqliteSaver checkpointer 里，
这张表只存列表页需要的摘要字段，每轮对话后 upsert 一次。
"""
from sqlalchemy import Column, String, Integer, Text, Index
from app.app import db
from model.common.base_model import BaseModel


class DoctorSession(db.Model, BaseModel):
    __bind_key__ = "ai"
    __tablename__ = "doctor_session"

    session_id = Column(String(64), nullable=False, comment="会话ID，同 LangGraph thread_id")
    chief_complaint = Column(String(200), nullable=True, comment="主诉预览，供列表展示")
    phase = Column(String(20), nullable=False, default="collecting", comment="collecting/completed")
    turn_count = Column(Integer, nullable=False, default=0, comment="已问诊轮次")
    completion_pct = Column(Integer, nullable=False, default=0, comment="信息完整度 0-100")
    audit_summary = Column(Text, nullable=True, comment="AI 审核总结：提问质量评价 + prompt 优化建议，按需生成")

    __table_args__ = (
        Index("ix_doctor_session_session_id", "session_id", unique=True),
    )
