"""DocMind REST API 启动脚本

用法:
    python -m docmind.api          # 默认 0.0.0.0:8000
    python -m docmind.api --port 9000
    python -m docmind.api --host 127.0.0.1 --port 8080 --reload
"""
from __future__ import annotations

import argparse
import uvicorn

from docmind.config import Config


def main():
    parser = argparse.ArgumentParser(description="DocMind REST API Server")
    parser.add_argument("--host", default=Config.API_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=Config.API_PORT, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    uvicorn.run(
        "docmind.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
