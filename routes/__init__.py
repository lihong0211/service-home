# routes/__init__.py
"""
路由模块：统一注册到 api 蓝图。
- routes/english.py：/english/* 单词、词根、词缀、对话、生活用语
- routes/peach.py：/peach/* 阿里报告、检查结果、插件统计
- routes/ai.py：/ai/* 对话、STT、TTS、图像、视频、知识库、向量库、RAG、文件
"""
from flask import Blueprint

from routes.english import register_english
from routes.peach import register_peach
from routes.ai import register_ai

api_bp = Blueprint("api", __name__)

register_english(api_bp)
register_peach(api_bp)
register_ai(api_bp)
