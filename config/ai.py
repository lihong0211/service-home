# config/ai.py
# DashScope / LLM 相关的共享配置常量，避免模型名、base_url 在各 service/ai/*.py 里硬编码散落。
import os

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEFAULT_CHAT_MODEL = "qwen-turbo"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_RERANK_MODEL = "qwen3-rerank"


def dashscope_api_key() -> str:
    return os.getenv("DASHSCOPE_API_KEY")
