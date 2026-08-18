# service-home：FastAPI AI 服务。requirements.txt 里 torch/transformers 等较重，
# 首次构建下载量大，正常现象。GPU 相关代码路径都做了 torch.cuda.is_available() 判断，
# 没 GPU（本地 Docker Desktop）会自动退化到 CPU，不会启动失败。
FROM python:3.12-slim

# 系统依赖：
# - tesseract-ocr(-chi-sim)：knowledge.py 的图片 OCR，代码里已有 TesseractNotFoundError
#   兜底（不装也不会崩，只是没有 OCR），装上保留完整功能
# - ffmpeg：faster-whisper（STT）/ 部分音频处理需要
# - build-essential：个别包（如 faiss-cpu 的间接依赖）没有 slim 镜像对应架构的预编译轮子时
#   需要本地编译
# 不装 LibreOffice：files.py 里文档转 PDF 已有"未安装则跳过预览、改为下载"的兜底逻辑，
# 而 LibreOffice 体积大（apt 装下来 500MB+），按需再加
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-chi-sim \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/（向量库、知识库源文件、checkpoint sqlite）、lora/、workspace/ 都通过
# docker-compose 的 volume 挂载出来持久化，镜像本身不含这些运行时数据。

EXPOSE 3000

# A2A 子智能体（8001-8004）由 app 启动时自己拉起子进程，只在容器内部
# localhost 互相通信，不需要额外 EXPOSE。
CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "3000"]
