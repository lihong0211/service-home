# app/__init__.py
"""
应用包（FastAPI）。

- ASGI 应用：`from app.app import app`（uvicorn 根模块 `app.py` 亦导出同名 `app`）
- 工厂函数：`from app.factory import create_app`
- 依赖注入：`from app.deps import SessionDep`
"""
__all__: list[str] = []
