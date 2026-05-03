"""API 依赖注入：获取共享的引擎实例 + JWT 认证"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from docmind.vector_store import VectorStore
from docmind.rag_engine import RAGEngine
from docmind.summarizer import Summarizer
from docmind.extractor import Extractor
from docmind.auth import verify_jwt, get_user_index_dir

security = HTTPBearer(auto_error=False)

# 全局共享实例（无认证时使用）
_store: VectorStore | None = None


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """从 JWT 中提取当前用户名，无 token 则返回 None"""
    if credentials is None:
        return None
    payload = verify_jwt(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="无效或过期的 token")
    return payload.get("sub")


def get_store(username: str | None = Depends(get_current_user)) -> VectorStore:
    """获取 VectorStore 实例（per-user 隔离）"""
    global _store
    if username:
        # Per-user 索引隔离
        index_dir = get_user_index_dir(username)
        store = VectorStore()
        store.load(dir_path=index_dir)
        return store

    # 无认证：共享实例
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store


def get_engine(store: VectorStore = Depends(get_store)) -> RAGEngine:
    """获取 RAG 引擎实例"""
    return RAGEngine(store)


def get_summarizer() -> Summarizer:
    """获取摘要生成器实例"""
    return Summarizer()


def get_extractor() -> Extractor:
    """获取信息提取器实例"""
    return Extractor()
