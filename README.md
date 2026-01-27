# EN API - Python版本

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用 Gunicorn 启动服务

```bash
gunicorn -c gunicorn.conf.py wsgi:app
```
