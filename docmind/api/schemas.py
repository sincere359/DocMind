"""API 请求 / 响应 Pydantic 模型"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── 通用 ──

class MessageResponse(BaseModel):
    """通用消息响应"""
    ok: bool = True
    message: str = ""


class ErrorResponse(BaseModel):
    """错误响应"""
    ok: bool = False
    error: str


# ── 文档管理 ──

class DocumentMeta(BaseModel):
    """单文档元信息"""
    source: str
    chunk_count: int
    total_chars: int


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: list[DocumentMeta]
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    """文档删除响应"""
    ok: bool = True
    message: str = ""
    deleted_source: str


# ── 问答 ──

class QARequest(BaseModel):
    """问答请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int | None = Field(None, ge=1, le=20, description="检索数量")
    chat_history: list[dict[str, str]] | None = Field(None, description="多轮对话历史")


class QAResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    query: str
    model: str = ""
    error: str = ""


# ── 分析 ──

class SummaryRequest(BaseModel):
    """摘要请求"""
    sources: list[str] | None = Field(None, description="指定文档名列表，空则全部")
    style: str = Field("detailed", pattern="^(brief|detailed|academic)$")


class SummaryResponse(BaseModel):
    """摘要响应"""
    title: str = ""
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    doc_type: str = ""
    word_count: int = 0


class ExtractRequest(BaseModel):
    """信息提取请求"""
    sources: list[str] | None = Field(None, description="指定文档名列表，空则全部")


class CompareRequest(BaseModel):
    """文档对比请求"""
    source_a: str = Field(..., description="文档 A 名称")
    source_b: str = Field(..., description="文档 B 名称")


class CompareResponse(BaseModel):
    """文档对比响应"""
    common_points: list[str] = Field(default_factory=list)
    differences: list[dict[str, str]] = Field(default_factory=list)
    conclusion: str = ""


# ── 健康检查 ──

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "2.0.0"
    indexed_documents: int = 0
    total_chunks: int = 0
