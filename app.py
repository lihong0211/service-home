# app.py
"""
主应用文件
"""
from flask import jsonify
from flask_sock import Sock
from app.app import app, db
from routes import api_bp

# 注册蓝图
app.register_blueprint(api_bp)

# WebSocket（实时 STT）
sock = Sock(app)
from service.ai.stt import register_stt_ws
register_stt_ws(sock)

# 初始化数据库表
with app.app_context():
    db.create_all()


# 错误处理
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


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)
