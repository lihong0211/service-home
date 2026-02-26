"""
PPT 生成记录表 ppt_record
每次 chat 成功提取到 ppt_id 后自动写入，方便后续查询历史
"""
from sqlalchemy import Column, String, SmallInteger, Text, Integer
from app.app import db
from model.common.base_model import BaseModel


class PptRecord(db.Model, BaseModel):
    __tablename__ = "ppt_record"
    __bind_key__ = "ai"

    ppt_id    = Column(String(128), nullable=False, unique=True, comment="YOO-AI ppt_id")
    title     = Column(String(255), nullable=True,  comment="PPT 标题（从生成结果提取）")
    prompt    = Column(Text,        nullable=True,  comment="用户输入的主题/提示词")
    page_count= Column(Integer,     nullable=True,  comment="页数")
    status    = Column(SmallInteger,nullable=False, default=1, comment="1=生成中 2=成功 3=失败")
    preview_url= Column(Text,       nullable=True,  comment="免费预览链接")
    process_url= Column(Text,       nullable=True,  comment="生成进度链接")
