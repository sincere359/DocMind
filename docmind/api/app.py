"""FastAPI 应用创建与配置"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docmind.api.routes_documents import router as doc_router
from docmind.api.routes_qa import router as qa_router
from docmind.api.routes_analysis import router as analysis_router
from docmind.api.routes_auth import router as auth_router


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="DocMind API",
        version="2.0.0",
        description="AI 文档智能助手 REST API — 基于 MiMo V2.5 + RAG",
    )

    # CORS（允许前端跨域调用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由注册
    app.include_router(auth_router, prefix="/api/auth", tags=["用户认证"])
    app.include_router(doc_router, prefix="/api/documents", tags=["文档管理"])
    app.include_router(qa_router, prefix="/api/qa", tags=["智能问答"])
    app.include_router(analysis_router, prefix="/api/analysis", tags=["文档分析"])

    @app.get("/health", tags=["系统"])
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()
