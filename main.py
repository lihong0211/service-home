#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
开发入口：Uvicorn 跑 FastAPI，带热重载
"""
import os
import sys

# 必须在 import uvicorn 之前检查：项目根目录若存在 math.py，会遮蔽标准库 math，
# 导致 random/email 等链式导入失败：ImportError: cannot import name 'log' from 'math'
_project_root = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile(os.path.join(_project_root, "math.py")):
    print(
        "错误：项目根目录存在 math.py，会遮蔽 Python 标准库 math。\n"
        "请重命名该文件（例如 math_helpers.py）并修改引用后再启动。",
        file=sys.stderr,
    )
    sys.exit(1)

import uvicorn

project_root = _project_root
sys.path.insert(0, project_root)

if __name__ == "__main__":
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
    )
