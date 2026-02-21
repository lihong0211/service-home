# routes/__init__.py
"""
路由模块：统一注册到 api 蓝图。
- routes/ai.py：/ai/* 对话、STT、TTS、图像、视频、知识库、向量库、RAG、文件
"""
from flask import Blueprint

from routes.ai import register_ai

api_bp = Blueprint("api", __name__)

register_ai(api_bp)
