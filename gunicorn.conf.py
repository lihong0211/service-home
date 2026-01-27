# -*- coding: utf-8 -*-
"""
Gunicorn 配置文件
用于生产环境部署
"""
import multiprocessing
import os

# 服务器配置
bind = f"0.0.0.0:{os.environ.get('PORT', 3000)}"
workers = multiprocessing.cpu_count() * 2 + 1  # 推荐的工作进程数
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# 日志配置
accesslog = "-"  # 输出到 stdout
errorlog = "-"  # 输出到 stderr
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

# 进程命名
proc_name = "service"

# 性能优化
max_requests = 1000  # 处理请求数后重启 worker，防止内存泄漏
max_requests_jitter = 50  # 随机化重启，避免所有 worker 同时重启

# 安全
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# 预加载应用（节省内存，但可能导致 worker 间数据不一致）
preload_app = False

# 优雅重启
graceful_timeout = 30
