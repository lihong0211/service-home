"""
知识库文档表 - 知识库下的原始文件记录，与 vector_db 同库 (ai)。
"""
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.mysql import INTEGER
from app.app import db
from model.common.base_model import BaseModel


class KnowledgeBaseDocument(db.Model, BaseModel):
    __bind_key__ = "ai"
    __tablename__ = "knowledge_base_document"

    knowledge_base_id = Column(INTEGER(11), nullable=False, index=True, comment="知识库 id")
    file_name = Column(String(255), nullable=False, comment="原始文件名")
    path = Column(String(512), nullable=True, comment="本机存储路径")
    file_id = Column(String(64), nullable=True, comment="上传文件 id")
    file_size = Column(INTEGER(11), nullable=True, comment="文件大小字节")
    status = Column(String(32), nullable=False, default="pending", comment="pending/parsing/segmented/failed")
    error_message = Column(Text, nullable=True, comment="失败时的错误信息")
