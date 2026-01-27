#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WSGI 入口文件 - 用于生产环境部署
"""
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.db import DB_CONFIG, DB_PDD_CONFIG

_db_config_check = dict(DB_CONFIG)
_db_pdd_config_check = dict(DB_PDD_CONFIG)

from app.app import app
from routes import api_bp

# 注册蓝图
app.register_blueprint(api_bp)


# 错误处理
@app.errorhandler(404)
def not_found(error):
    from flask import jsonify

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
    from flask import jsonify

    return (
        jsonify(
            {
                "code": 500,
                "msg": "Internal Server Error",
            }
        ),
        500,
    )
