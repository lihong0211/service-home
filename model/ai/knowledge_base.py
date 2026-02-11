"""
知识库表 - 元信息 + 解析策略 + 分段策略，与 vector_db 同库 (ai)。
"""
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.mysql import INTEGER, JSON, TINYINT
from app.app import db
from model.common.base_model import BaseModel


class KnowledgeBase(db.Model, BaseModel):
    __bind_key__ = "ai"
    __tablename__ = "knowledge_base"

    name = Column(String(128), nullable=False, comment="知识库名称")
    description = Column(Text, nullable=True, comment="描述")
    parsing_strategy = Column(String(32), nullable=True, default="fast", comment="解析方式: fast / precise")
    precise_options = Column(JSON, nullable=True, comment="精准解析选项")
    content_filter = Column(Text, nullable=True, comment="内容过滤规则")
    chunking_strategy = Column(String(32), nullable=True, default="custom", comment="分段方式: auto / custom / hierarchy")
    chunking_config = Column(JSON, nullable=True, comment="自定义分段配置")
    hierarchy_level = Column(INTEGER(11), nullable=True, default=3, comment="按层级分段时的层级深度")
    retain_hierarchy = Column(TINYINT(1), nullable=True, default=1, comment="检索切片是否保留层级信息 0/1")
    vector_db_id = Column(INTEGER(11), nullable=True, index=True, comment="关联的向量库 id")
