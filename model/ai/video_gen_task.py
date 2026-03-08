"""
火山方舟文生视频任务表 video_gen_task
用于持久化 task_id，便于 Dify 提交后在其他地方查询状态与结果（仅支持最近 7 天）
"""
from sqlalchemy import Column, String, Text, Integer
from app.app import db
from model.common.base_model import BaseModel


class VideoGenTask(db.Model, BaseModel):
    __tablename__ = "video_gen_task"
    __bind_key__ = "ai"

    task_id = Column(String(128), nullable=False, unique=True, comment="火山方舟任务 ID，如 cgt-20260302195613-k42bm")
    status = Column(String(32), nullable=False, default="submitted", comment="submitted|queued|running|succeeded|failed|cancelled|expired")
    prompt = Column(Text, nullable=True, comment="文生视频提示词")
    model = Column(String(128), nullable=True, comment="模型名，如 doubao-seedance-1-5-pro-251215")
    video_url = Column(Text, nullable=True, comment="生成完成后的视频地址")
    resolution = Column(String(32), nullable=True, comment="如 720p")
    ratio = Column(String(32), nullable=True, comment="如 16:9")
    duration = Column(Integer, nullable=True, comment="时长秒")
    source = Column(String(64), nullable=True, comment="来源，如 dify")
