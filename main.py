#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
开发入口：Uvicorn 跑 FastAPI，带热重载
"""
import os
import sys
import uvicorn

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
    )
