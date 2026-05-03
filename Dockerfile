# ── DocMind Dockerfile ──
# 多阶段构建：预下载模型 + 运行时

# ---------- Stage 1: 下载 embedding 模型 ----------
FROM python:3.12-slim AS model-downloader

RUN pip install --no-cache-dir sentence-transformers

# 预下载 text2vec-base-chinese 到 /models
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('shibing624/text2vec-base-chinese'); \
m.save('/models/text2vec-base-chinese'); \
print('Model saved')"

# ---------- Stage 2: 运行时 ----------
FROM python:3.12-slim

LABEL maintainer="DocMind"
LABEL description="AI 文档智能助手 - MiMo V2.5 + RAG"

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先拷贝依赖文件，利用 Docker 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 从 Stage 1 拷贝预下载的模型
COPY --from=model-downloader /models /app/data/models/text2vec-base-chinese

# 拷贝源码
COPY docmind/ docmind/
COPY .env.example .env

# 创建数据目录
RUN mkdir -p /app/data/uploads /app/data/index /app/data/history

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV EMBEDDING_MODEL=/app/data/models/text2vec-base-chinese
ENV HOST=0.0.0.0
ENV PORT=8501
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# 暴露端口 (Streamlit + FastAPI)
EXPOSE 8501 8000

# 数据卷
VOLUME ["/app/data/uploads", "/app/data/index", "/app/data/history"]

# 默认启动 API 服务（可通过 docker-compose 覆盖）
CMD ["python", "-m", "docmind.api", "--host", "0.0.0.0", "--port", "8000"]
