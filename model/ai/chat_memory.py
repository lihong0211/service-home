from sqlalchemy import Column, String, Text, Index
from app.app import db
from model.common.base_model import BaseModel


class ChatMemory(db.Model, BaseModel):
    __bind_key__ = "ai"
    __tablename__ = "chat_memory"

    user_id = Column(String(64), nullable=False, index=True, comment="用户ID")
    content = Column(Text, nullable=False, comment="记忆内容")
    memory_type = Column(String(20), nullable=False, default="fact", comment="记忆类型: fact/preference/context")

    __table_args__ = (
        Index("ix_chat_memory_user_id", "user_id"),
    )
