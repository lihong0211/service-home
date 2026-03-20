# app/database.py
"""
FastAPI + SQLAlchemy 2.0 数据库层
- 仅连接 AI 库（与 model 中 __bind_key__=\"ai\" 一致）
- 请求级 session 通过 contextvar 注入，兼容原有 db.session 用法
"""
from contextvars import ContextVar
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.engine import Engine

from config.db import DB_AI_CONFIG

# 供模型继承，替代 Flask-SQLAlchemy 的 db.Model
Base = declarative_base()


def _make_url(config: dict) -> str:
    pwd = quote_plus(config["password"])
    return (
        f"mysql+pymysql://{config['user']}:{pwd}"
        f"@{config['host']}:{config['port']}/{config['database']}"
        f"?charset={config.get('charset', 'utf8mb4')}"
    )


# 单一 AI 库 engine（AI_DB_*，未设时回退到 DB_*，见 config.db）
engine: Engine = create_engine(
    _make_url(DB_AI_CONFIG),
    pool_size=50,
    max_overflow=200,
    pool_recycle=3600,
    pool_timeout=30,
    pool_pre_ping=True,
    echo=False,
)

# 兼容旧代码 engines.get("ai") / 多进程 fork 后遍历 dispose
engines: dict[str, Engine] = {"ai": engine}

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=Session,
)

# 请求级 session 存于 contextvar
_session_ctx: ContextVar[Session | None] = ContextVar("sqlalchemy_session", default=None)


def set_request_session(session: Session) -> None:
    """路由/依赖：为当前请求设置 session。"""
    _session_ctx.set(session)


def clear_request_session() -> None:
    """路由/依赖：清除当前请求的 session。"""
    _session_ctx.set(None)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求一个 session，请求结束关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _query_property():
    """兼容 Flask-SQLAlchemy 的 Model.query：返回当前 request session 的 query。"""

    class _QueryDescriptor:
        def __get__(self, obj, type_):
            if type_ is None:
                return self
            return _session_ctx.get().query(type_) if _session_ctx.get() else None

    return _QueryDescriptor()


class _DbCompat:
    """兼容原 Flask-SQLAlchemy 的 db 对象：db.Model、db.session、db.engine、db.engines。"""

    @property
    def Model(self):
        return Base

    query_property = staticmethod(_query_property)

    @property
    def session(self) -> Session:
        s = _session_ctx.get()
        if s is None:
            raise RuntimeError(
                "No request-scoped session. Use get_db() in FastAPI or set session in middleware."
            )
        return s

    @property
    def engine(self) -> Engine:
        return engine

    @property
    def engines(self) -> dict:
        return engines


db = _DbCompat()
