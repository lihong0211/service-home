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
# 上传+解析+向量化可能较久，与 Nginx proxy_read_timeout 对齐
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 300))
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

# 预加载应用：在 master 进程 import 一次，background services 只启动一次，
# 各 worker fork 后继承 _bg_services_started=True，不会重复启动后台服务
preload_app = True

# 优雅重启（worker 处理完当前请求后再退出）
graceful_timeout = 60
