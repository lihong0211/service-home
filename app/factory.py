# app/factory.py
"""
应用工厂：集中注册中间件、路由、WebSocket、异常处理（FastAPI 推荐结构）。
"""
import os
import sys
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import db, Base
from config.db import DB_AI_CONFIG


def _validate_db_config() -> None:
    """仅校验 AI 库凭据（AI_DB_*，未设时回退 DB_*）。"""
    if not DB_AI_CONFIG.get("user"):
        raise ValueError("请配置 AI 库用户：AI_DB_USER 或 DB_USER（.env）")
    if not DB_AI_CONFIG.get("password"):
        raise ValueError("请配置 AI 库密码：AI_DB_PASSWORD 或 DB_PASSWORD（.env）")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if hasattr(db.engine, "dispose"):
            db.engine.dispose(close=True)
        Base.metadata.create_all(bind=db.engine)
    except Exception as e:
        print(f"[DB Init Warning] 数据库初始化失败: {e}", flush=True)
        import traceback

        traceback.print_exc()
    root = Path(__file__).resolve().parent.parent
    for rel_path, label in [
        ("service/ai/a2a/agents/outline_agent.py", "OutlineAgent :8001"),
        ("service/ai/a2a/agents/doc_agent.py", "DocAgent    :8002"),
        ("service/ai/a2a/agents/summary_agent.py", "SummaryAgent:8003"),
    ]:
        script = root / rel_path
        if script.exists():
            subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[BG] {label} started", flush=True)
    yield


def _http_exception_payload(exc: HTTPException) -> dict:
    detail = exc.detail
    if isinstance(detail, str):
        return {"code": exc.status_code, "msg": detail}
    if isinstance(detail, list):
        return {"code": exc.status_code, "msg": "Request error", "data": detail}
    if isinstance(detail, dict):
        return {
            "code": exc.status_code,
            "msg": detail.get("msg", "Error"),
            "data": detail,
        }
    return {"code": exc.status_code, "msg": "Error"}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            content=_http_exception_payload(exc),
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "code": 422,
                "msg": "参数校验失败",
                "data": exc.errors(),
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def handle_file_not_found(request: Request, exc: FileNotFoundError):
        return JSONResponse(content={"code": 404, "msg": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError):
        return JSONResponse(content={"code": 400, "msg": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def handle_exception(request: Request, exc: Exception):
        try:
            import logging

            logging.getLogger("uvicorn.error").exception("Unhandled exception")
        except Exception:
            pass
        return JSONResponse(content={"code": 500, "msg": str(exc)}, status_code=500)


def register_websocket_routes(app: FastAPI) -> None:
    """WebSocket 不走 HTTP 的 Depends(get_db)；当前 STT 不访问 DB。"""
    from fastapi import WebSocket

    @app.websocket("/api/ai/stt/live")
    async def stt_live_ws(websocket: WebSocket):
        from service.ai.stt import register_stt_ws_fastapi

        await register_stt_ws_fastapi(websocket)


def create_app() -> FastAPI:
    _validate_db_config()
    print(
        f"[DB Config] AI 库 user={DB_AI_CONFIG['user']}, host={DB_AI_CONFIG['host']}, "
        f"database={DB_AI_CONFIG['database']}, password_set={bool(DB_AI_CONFIG['password'])}"
    )

    app = FastAPI(
        title="service-home API",
        description="FastAPI（由 Flask 迁移）",
        lifespan=lifespan,
    )

    cors_origin = os.environ.get("CORS_ORIGIN", "*")
    cors_in_app = os.environ.get("FLASK_ENV") != "production"
    if cors_in_app:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[cors_origin] if cors_origin != "*" else ["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
            allow_headers=["Content-Type", "Authorization", "X-Admin-Token"],
        )

    register_exception_handlers(app)

    from routes import api_router

    app.include_router(api_router, prefix="/api")

    register_websocket_routes(app)

    return app
