# app/app.py
"""
应用包入口：导出 FastAPI `app` 与 `db`（供 model / service 使用 `from app.app import db`）。
加载顺序：先导入 database（提供 db），再 `create_app()`，避免 routes 反向 import 时循环依赖。
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app.database import Base, SessionLocal, db, engine, engines, get_db
from app.factory import create_app

app = create_app()

__all__ = ["app", "db", "get_db", "Base", "engine", "engines", "SessionLocal"]
