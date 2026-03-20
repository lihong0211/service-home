# routes/__init__.py
"""
路由模块：统一注册到 FastAPI APIRouter。
- routes/ai.py：/ai/* 对话、STT、TTS、图像、视频、知识库、向量库、RAG、文件
- routes/payment.py：/payment/* 支付订单
"""
from fastapi import APIRouter

from routes.ai import register_ai

api_router = APIRouter()

register_ai(api_router)
