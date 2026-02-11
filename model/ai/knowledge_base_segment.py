"""
知识库分段表 - 文档解析+分段后的文本块，支持层级，与 vector_db 同库 (ai)。
"""
from sqlalchemy import Column, Text
from sqlalchemy.dialects.mysql import INTEGER, JSON
from app.app import db
from model.common.base_model import BaseModel


class KnowledgeBaseSegment(db.Model, BaseModel):
    __bind_key__ = "ai"
    __tablename__ = "knowledge_base_segment"

    document_id = Column(INTEGER(11), nullable=False, index=True, comment="文档 id")
    text = Column(Text, nullable=False, comment="分段文本")
    index = Column(INTEGER(11), nullable=False, default=0, comment="文档内顺序")
    parent_id = Column(INTEGER(11), nullable=True, index=True, comment="按层级分段时的父分段 id")
    segment_metadata = Column("metadata", JSON, nullable=True, comment="层级路径、标题等扩展信息")
