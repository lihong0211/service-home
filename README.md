# EN API - Python版本

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

开发（热重载）：

```bash
python main.py
```

生产（多进程可调 `UVICORN_WORKERS`，端口可调 `PORT`）：

```bash
UVICORN_WORKERS=4 PORT=3000 python -m uvicorn app.app:app --host 0.0.0.0 --port 3000 --workers 4
```

或使用脚本重启/后台启动：`./restart_uvicorn.sh`（依赖项目内 `venv/bin/python`）。
