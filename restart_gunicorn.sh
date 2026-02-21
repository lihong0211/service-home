#!/bin/bash
# Gunicorn 重启脚本

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 查找 gunicorn 主进程
PID=$(ps aux | grep "[g]unicorn.*wsgi:app" | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "未找到运行中的 gunicorn 进程，直接启动..."
    nohup "$DIR/venv/bin/python" -m gunicorn -c "$DIR/gunicorn.conf.py" wsgi:app >> "$DIR/logs/gunicorn.log" 2>&1 &
    echo "Gunicorn 已启动 (PID: $!)"
else
    echo "找到 gunicorn 主进程 PID: $PID"
    echo "发送 HUP 信号进行优雅重启..."
    kill -HUP $PID
    echo "重启信号已发送，等待进程重启..."
    sleep 2
    echo "当前 gunicorn 进程:"
    ps aux | grep "[g]unicorn.*wsgi:app" | head -3
fi
