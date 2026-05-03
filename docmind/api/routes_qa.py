"""智能问答路由：同步问答 / 流式问答"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from docmind.api.deps import get_engine
from docmind.api.schemas import QARequest, QAResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=QAResponse)
async def ask_question(req: QARequest):
    """同步问答"""
    engine = get_engine()
    result = engine.ask(
        query=req.query,
        top_k=req.top_k,
        chat_history=req.chat_history,
    )
    return QAResponse(
        answer=result.answer,
        sources=result.sources,
        query=result.query,
        model=result.model,
        error=result.error,
    )


@router.post("/ask/stream")
async def ask_question_stream(req: QARequest):
    """流式问答 (SSE)"""
    engine = get_engine()

    def _error_stream(msg: str):
        yield f"data: {json.dumps({'error': msg}, ensure_ascii=False)}\n\n"

    if engine.store.is_empty:
        return StreamingResponse(
            _error_stream("请先上传文档"),
            media_type="text/event-stream",
        )

    # 预检索，拿到 sources
    results = engine.store.search(req.query, top_k=req.top_k)
    if not results:
        return StreamingResponse(
            _error_stream("未找到相关文档内容"),
            media_type="text/event-stream",
        )

    # 发送 sources 作为首条事件
    _, sources = engine._build_context(results)
    context, _ = engine._build_context(results)
    messages = engine._build_messages(context, req.query, req.chat_history)

    async def _stream() -> AsyncGenerator[str, None]:
        # 先推送 sources
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

        # 流式推送回答 token
        try:
            for token in engine.client.chat_stream(messages=messages, temperature=0.2):
                yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
