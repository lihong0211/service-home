# config/db.py
# MySQL 数据库连接配置
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 加载 .env 文件（确保环境变量已加载）
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

_BASE = {
    "host": "localhost",
    "user": "",  # 从环境变量 DB_USER 读取
    "password": "",  # 从环境变量 DB_PASSWORD 读取
    "database": "english",
    "port": "3306",
    "charset": "utf8mb4",
}


def _get_config(
    prefix: str,
    defaults: Dict[str, Any],
    label: str,
    fallback_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """统一拉取 DB 配置。prefix 如 DB_/PDD_DB_/AI_DB_；fallback_prefix 时该 key 先读 prefix 再读 fallback（如 AI 回退到 DB_）。"""
    config = {}
    for k, v in defaults.items():
        env_key = prefix + k.upper()
        if fallback_prefix:
            v = os.environ.get(env_key, os.environ.get(fallback_prefix + k.upper(), v))
        else:
            v = os.environ.get(env_key, v)
        # 确保字符串值不为 None（环境变量未设置时可能返回 None）
        if v is None:
            v = defaults.get(k, "")
        config[k] = int(v) if k == "port" else v
    config["autocommit"] = True
    return config


# 初始化（模块加载时校验）
DB_CONFIG = _get_config("DB_", {**_BASE}, "数据库")
DB_PDD_CONFIG = _get_config("PDD_DB_", {**_BASE, "database": "pdd_report"}, "PDD数据库")
DB_AI_CONFIG = _get_config(
    "AI_DB_", {**_BASE, "database": "ai"}, "AI数据库", fallback_prefix="DB_"
)

# 调试：打印配置状态（启动时）
if not DB_CONFIG.get("user") or not DB_CONFIG.get("password"):
    import sys
    print(f"[DB Config Warning] user={DB_CONFIG.get('user') or 'NOT SET'}, password={'SET' if DB_CONFIG.get('password') else 'NOT SET'}", file=sys.stderr)
    print(f"[DB Config Warning] DB_USER env={os.environ.get('DB_USER', 'NOT SET')}, DB_PASSWORD env={'SET' if os.environ.get('DB_PASSWORD') else 'NOT SET'}", file=sys.stderr)
