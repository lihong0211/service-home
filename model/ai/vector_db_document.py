"""
向量库文档项表 - 主库 MySQL，存每个向量库下的文档 item，支持按库查询。
"""
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.mysql import INTEGER, JSON
from app.app import db
from model.common.base_model import BaseModel


class VectorDbDocument(db.Model, BaseModel):
    __bind_key__ = "ai"
    __tablename__ = "vector_db_document"

    vector_db_id = Column(INTEGER(11), nullable=False, index=True, comment="向量库 id")
    doc_id = Column(String(128), nullable=False, comment="文档项业务 id")
    text = Column(Text, nullable=False, comment="文档文本")
    category = Column(String(128), nullable=True, comment="分类")
    document_metadata = Column("metadata", JSON, nullable=True, comment="扩展信息：segment_id,document_id,层级,来源等")
