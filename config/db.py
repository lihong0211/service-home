# config/db.py
# MySQL 数据库连接配置
import os
from typing import Dict, Any, Optional

_BASE = {
    "host": "localhost",
    "user": "root",
    "password": "Yy123456@",
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
        config[k] = int(v) if k == "port" else v
    config["autocommit"] = True
    return config


# 初始化（模块加载时校验）
DB_CONFIG = _get_config("DB_", {**_BASE}, "数据库")
DB_PDD_CONFIG = _get_config("PDD_DB_", {**_BASE, "database": "pdd_report"}, "PDD数据库")
DB_AI_CONFIG = _get_config(
    "AI_DB_", {**_BASE, "database": "ai"}, "AI数据库", fallback_prefix="DB_"
)
