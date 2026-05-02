"""文本向量化：基于 sentence-transformers 本地嵌入"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np

from docmind.config import Config

logger = logging.getLogger(__name__)

# 全局模型缓存
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("加载 Embedding 模型: %s", Config.EMBEDDING_MODEL)
        _model = SentenceTransformer(Config.EMBEDDING_MODEL)
        logger.info("模型加载完成，维度: %d", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """将文本列表转为向量矩阵 (n, dim)"""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """将单条查询转为向量 (dim,)"""
    model = _get_model()
    embedding = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
    return np.array(embedding[0], dtype=np.float32)


def get_embedding_dim() -> int:
    """获取嵌入维度"""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
