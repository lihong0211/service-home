# app/app.py
"""
应用初始化
"""
import os
from urllib.parse import quote_plus
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config.db import DB_CONFIG, DB_PDD_CONFIG

app = Flask(__name__)

# 验证数据库配置（使用副本避免被修改）
_db_config = dict(DB_CONFIG)
_db_pdd_config = dict(DB_PDD_CONFIG)


# 配置主数据库
# 再次验证配置值（防止在构建字符串时被修改）
host = str(_db_config.get("host", "")).strip()
user = str(_db_config.get("user", "")).strip()
password = str(_db_config.get("password", "")).strip()
database = str(_db_config.get("database", "")).strip()
port = int(_db_config.get("port", 3306))
charset = str(_db_config.get("charset", "utf8mb4")).strip()


# 对密码进行 URL 编码（处理特殊字符如 @、# 等）
encoded_password = quote_plus(password)
mysql_url = (
    f"mysql+pymysql://{user}:{encoded_password}"
    f"@{host}:{port}/{database}"
    f"?charset={charset}"
)

app.config["SQLALCHEMY_DATABASE_URI"] = mysql_url


# 验证 PDD 数据库配置
pdd_host = str(_db_pdd_config.get("host", "")).strip()
pdd_user = str(_db_pdd_config.get("user", "")).strip()
pdd_password = str(_db_pdd_config.get("password", "")).strip()
pdd_database = str(_db_pdd_config.get("database", "")).strip()
pdd_port = int(_db_pdd_config.get("port", 3306))
pdd_charset = str(_db_pdd_config.get("charset", "utf8mb4")).strip()


# 对密码进行 URL 编码
encoded_pdd_password = quote_plus(pdd_password)
pdd_mysql_url = (
    f"mysql+pymysql://{pdd_user}:{encoded_pdd_password}"
    f"@{pdd_host}:{pdd_port}/{pdd_database}"
    f"?charset={pdd_charset}"
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

# 初始化数据库
db = SQLAlchemy(app)
