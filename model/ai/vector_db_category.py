"""
向量库分类表 - 每个向量库可维护自己的分类，新增/编辑文档时可选择分类。
"""
from sqlalchemy import Column, String
from sqlalchemy.dialects.mysql import INTEGER
from app.app import db
from model.common.base_model import BaseModel


class VectorDbCategory(db.Model, BaseModel):
    __bind_key__ = "ai"
    __tablename__ = "vector_db_category"

    vector_db_id = Column(INTEGER(11), nullable=False, index=True, comment="向量库 id")
    name = Column(String(128), nullable=False, comment="分类名称")
    sort_order = Column(INTEGER(11), nullable=True, default=0, comment="排序，越小越靠前")
