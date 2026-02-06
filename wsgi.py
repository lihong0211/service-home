#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WSGI 入口 - 给 gunicorn 等用：gunicorn -c gunicorn.conf.py wsgi:app
根目录的 app.py 与包 app/ 同名，import app 会拿到包而非根 app.py，故按路径加载根 app.py 取可调用 app。
"""
import importlib.util
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 加载项目根目录的 app.py（与 app 包同名，必须按文件加载）
_app_py = os.path.join(project_root, "app.py")
_spec = importlib.util.spec_from_file_location("_app_main", _app_py)
_app_main = importlib.util.module_from_spec(_spec)
sys.modules["_app_main"] = _app_main
_spec.loader.exec_module(_app_main)
app = _app_main.app
