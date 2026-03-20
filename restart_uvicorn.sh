#!/bin/bash
# Uvicorn 重启（生产可多 worker；开发请用 python main.py）

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PORT="${PORT:-3000}"
WORKERS="${UVICORN_WORKERS:-1}"

PID=$(pgrep -f "uvicorn.*app\.app:app" | head -1)

mkdir -p "$DIR/logs"

if [ -z "$PID" ]; then
    echo "未找到运行中的 uvicorn，启动..."
    nohup "$DIR/venv/bin/python" -m uvicorn app.app:app \
        --host 0.0.0.0 --port "$PORT" --workers "$WORKERS" \
        >> "$DIR/logs/uvicorn.log" 2>&1 &
    echo "uvicorn 已启动 (PID: $!)"
else
    echo "找到 uvicorn 主进程 PID: $PID，终止后重启..."
    kill "$PID"
    sleep 2
    nohup "$DIR/venv/bin/python" -m uvicorn app.app:app \
        --host 0.0.0.0 --port "$PORT" --workers "$WORKERS" \
        >> "$DIR/logs/uvicorn.log" 2>&1 &
    echo "uvicorn 已重启 (PID: $!)"
fi
