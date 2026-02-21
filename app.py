# app.py
"""
主应用文件
"""
import os
import subprocess
import sys
import threading
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（必须在导入其他模块之前）
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from flask import jsonify, request, Response
from flask_sock import Sock
from werkzeug.exceptions import RequestEntityTooLarge
from app.app import app, db
from routes import api_bp

# 跨域：测试方案——只在 Flask 加 CORS，Nginx（home 段）不加；家里服务器不要设 FLASK_ENV=production
CORS_ORIGIN = "*"
CORS_HEADERS = {
    "Access-Control-Allow-Origin": CORS_ORIGIN,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
}
CORS_IN_APP = os.environ.get("FLASK_ENV") != "production"


@app.before_request
def cors_preflight():
    if request.method == "OPTIONS":
        return Response("", 204, headers=CORS_HEADERS if CORS_IN_APP else {})


@app.after_request
def cors_headers(response):
    if CORS_IN_APP:
        for k, v in CORS_HEADERS.items():
            response.headers[k] = v
    return response


# 注册蓝图
app.register_blueprint(api_bp)

# WebSocket（实时 STT）
sock = Sock(app)
from service.ai.stt import register_stt_ws

register_stt_ws(sock)

# 初始化数据库表
# 注意：如果之前有连接池缓存，需要先清理
try:
    with app.app_context():
        # 清理旧的连接池（如果有）
        if hasattr(db.engine, 'dispose'):
            db.engine.dispose(close=True)
        db.create_all()
except Exception as e:
    # 如果数据库连接失败，打印错误但不阻止应用启动
    print(f"[DB Init Warning] 数据库初始化失败: {e}", flush=True)
    import traceback
    traceback.print_exc()


# 错误处理：HTTP 状态码
@app.errorhandler(404)
def not_found(error):
    return (
        jsonify(
            {
                "code": 404,
                "msg": "Not Found",
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    return (
        jsonify(
            {
                "code": 500,
                "msg": "Internal Server Error",
            }
        ),
        500,
    )


# 统一处理接口中抛出的异常，避免每个接口重复 try/except
@app.errorhandler(FileNotFoundError)
def handle_file_not_found(e):
    return jsonify({"code": 404, "msg": str(e)}), 404


@app.errorhandler(ValueError)
def handle_value_error(e):
    return jsonify({"code": 400, "msg": str(e)}), 400


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(e):
    """处理文件上传大小超限"""
    return (
        jsonify(
            {
                "code": 413,
                "msg": "文件大小超过限制（最大 500MB），请压缩文件或分批上传",
            }
        ),
        413,
    )


@app.errorhandler(Exception)
def handle_exception(e):
    # 打印完整堆栈，便于定位 500（否则只返回 JSON，控制台看不到 traceback）
    try:
        app.logger.exception("Unhandled exception")
    except Exception:
        pass
    return jsonify({"code": 500, "msg": str(e)}), 500


_bg_services_started = False
_bg_services_lock = threading.Lock()


def _start_background_services():
    """随主进程启动后台服务（仅启动一次）"""
    global _bg_services_started
    with _bg_services_lock:
        if _bg_services_started:
            return
        _bg_services_started = True

    root = Path(__file__).resolve().parent

    # A2A agents — 各自独立子进程，携带独立 uvicorn 事件循环
    a2a_agents = [
        ("service/ai/a2a/agents/outline_agent.py", "OutlineAgent :8001"),
        ("service/ai/a2a/agents/doc_agent.py",     "DocAgent    :8002"),
        ("service/ai/a2a/agents/summary_agent.py", "SummaryAgent:8003"),
    ]
    for rel_path, label in a2a_agents:
        script = root / rel_path
        if script.exists():
            subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[BG] {label} started", flush=True)
        else:
            print(f"[BG] {label} script not found: {script}", flush=True)

    # DNS updater — 守护线程（内置文件锁，天然防多实例）
    try:
        from service.dns_updater import run_loop
        t = threading.Thread(target=run_loop, daemon=True, name="dns-updater")
        t.start()
        print("[BG] dns-updater started", flush=True)
    except Exception as exc:
        print(f"[BG] dns-updater failed to start: {exc}", flush=True)


_start_background_services()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
