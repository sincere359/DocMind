"""文档分析路由：摘要 / 关键信息提取 / 文档对比"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from docmind.api.deps import get_store, get_summarizer, get_extractor
from docmind.api.schemas import (
    SummaryRequest,
    SummaryResponse,
    ExtractRequest,
    CompareRequest,
    CompareResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_chunks(sources: list[str] | None):
    """获取指定文档的 chunks，如果 sources 为空则返回全部"""
    store = get_store()
    if store.is_empty:
        raise HTTPException(status_code=400, detail="请先上传并索引文档")

    if sources:
        chunks = [c for c in store.chunks if c.source in sources]
        if not chunks:
            raise HTTPException(status_code=404, detail="未找到指定文档的索引")
    else:
        chunks = store.chunks

    return chunks


@router.post("/summary", response_model=SummaryResponse)
async def summarize(req: SummaryRequest):
    """生成文档摘要"""
    chunks = _get_chunks(req.sources)
    summarizer = get_summarizer()
    result = summarizer.summarize(chunks, style=req.style)
    return SummaryResponse(**result)


@router.post("/extract")
async def extract_info(req: ExtractRequest):
    """提取文档关键信息"""
    chunks = _get_chunks(req.sources)
    extractor = get_extractor()
    result = extractor.extract(chunks)
    return result


@router.post("/compare", response_model=CompareResponse)
async def compare_documents(req: CompareRequest):
    """对比两个文档"""
    store = get_store()
    if store.is_empty:
        raise HTTPException(status_code=400, detail="请先上传并索引文档")

    chunks_a = [c for c in store.chunks if c.source == req.source_a]
    chunks_b = [c for c in store.chunks if c.source == req.source_b]

    if not chunks_a:
        raise HTTPException(status_code=404, detail=f"文档不存在: {req.source_a}")
    if not chunks_b:
        raise HTTPException(status_code=404, detail=f"文档不存在: {req.source_b}")

    extractor = get_extractor()
    result = extractor.compare_documents(chunks_a, chunks_b, req.source_a, req.source_b)
    return CompareResponse(**result)
