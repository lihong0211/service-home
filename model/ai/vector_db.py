"""
向量库索引表 - 主库 MySQL，仅存名称（索引）等元信息，向量与文档存磁盘。
"""
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.mysql import INTEGER
from app.app import db
from model.common.base_model import BaseModel


class VectorDb(db.Model, BaseModel):
    __bind_key__ = "ai"
    __tablename__ = "vector_db"

    name = Column(String(128), nullable=False, unique=True, comment="向量库名称，用作索引标识")
    description = Column(Text, nullable=True, comment="描述")
