"""文档管理路由：上传 / 列表 / 删除"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from docmind.config import Config
from docmind.api.deps import get_store, get_engine
from docmind.api.schemas import (
    DocumentListResponse,
    DocumentMeta,
    DocumentDeleteResponse,
    MessageResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=MessageResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    """上传文档并自动索引"""
    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")

    saved_paths: list[str] = []
    for f in files:
        # 检查扩展名
        suffix = Path(f.filename).suffix.lower()
        if suffix not in {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls", ".pptx"}:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {suffix}")

        # 检查大小
        content = await f.read()
        if len(content) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"文件过大: {f.filename}")

        # 保存到 uploads
        dest = Config.UPLOAD_DIR / f.filename
        dest.write_bytes(content)
        saved_paths.append(str(dest))

    # 索引
    try:
        engine = get_engine()
        added = engine.add_documents(saved_paths)
    except Exception as e:
        logger.error("索引失败: %s", e)
        raise HTTPException(status_code=500, detail=f"索引失败: {e}")

    return MessageResponse(message=f"上传并索引完成，新增 {added} 个文本块")


@router.get("/list", response_model=DocumentListResponse)
async def list_documents():
    """列出所有已索引文档"""
    store = get_store()
    stats = store.get_document_stats()
    docs = [
        DocumentMeta(source=s.source, chunk_count=s.chunk_count, total_chars=s.total_chars)
        for s in stats
    ]
    return DocumentListResponse(documents=docs, total_chunks=store.total_chunks)


@router.delete("/{source:path}", response_model=DocumentDeleteResponse)
async def delete_document(source: str):
    """删除指定文档及其索引"""
    store = get_store()
    if not store.has_source(source):
        raise HTTPException(status_code=404, detail=f"文档不存在: {source}")

    ok = store.remove_source(source)
    if not ok:
        raise HTTPException(status_code=500, detail="删除失败")

    # 同时删除上传文件
    file_path = Config.UPLOAD_DIR / source
    if file_path.exists():
        file_path.unlink()

    return DocumentDeleteResponse(message="删除成功", deleted_source=source)


@router.post("/clear", response_model=MessageResponse)
async def clear_all_documents():
    """清空所有文档索引"""
    store = get_store()
    store.clear()
    return MessageResponse(message="已清空所有文档索引")
