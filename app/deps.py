# app/deps.py
"""
FastAPI 依赖注入（最佳实践：显式 Depends，避免隐式全局 session）。
"""
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import get_db

# 路由中写法：async def foo(db: SessionDep): ...
SessionDep = Annotated[Session, Depends(get_db)]
