# app/app.py
"""
应用初始化
"""
from urllib.parse import quote_plus
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config.db import DB_CONFIG, DB_PDD_CONFIG

app = Flask(__name__)

# 配置主数据库
encoded_password = quote_plus(DB_CONFIG["password"])
mysql_url = (
    f"mysql+pymysql://{DB_CONFIG['user']}:{encoded_password}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    f"?charset={DB_CONFIG.get('charset', 'utf8mb4')}"
)

app.config["SQLALCHEMY_DATABASE_URI"] = mysql_url

# 配置 PDD 数据库
encoded_pdd_password = quote_plus(DB_PDD_CONFIG["password"])
pdd_mysql_url = (
    f"mysql+pymysql://{DB_PDD_CONFIG['user']}:{encoded_pdd_password}"
    f"@{DB_PDD_CONFIG['host']}:{DB_PDD_CONFIG['port']}/{DB_PDD_CONFIG['database']}"
    f"?charset={DB_PDD_CONFIG.get('charset', 'utf8mb4')}"
)

app.config["SQLALCHEMY_BINDS"] = {"pdd": pdd_mysql_url}
app.config["SQLALCHEMY_POOL_SIZE"] = 50
app.config["SQLALCHEMY_POOL_TIMEOUT"] = 30
app.config["SQLALCHEMY_MAX_OVERFLOW"] = 200
app.config["SQLALCHEMY_POOL_RECYCLE"] = 3600
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_size": 50,
    "max_overflow": 200,
    "pool_recycle": 3600,
    "pool_timeout": 30,
    "echo": False,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ECHO"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json;charset=utf-8"
app.config["JSON_AS_ASCII"] = False
# 识图接口会传 base64 大 body，放宽限制（默认约 1MB）
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# 初始化数据库
db = SQLAlchemy(app)
