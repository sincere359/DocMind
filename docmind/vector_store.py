"""FAISS 向量存储：文档索引 + 相似度检索"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from docmind.config import Config
from docmind.document_parser import Chunk
from docmind.embeddings import embed_texts, embed_query, get_embedding_dim

logger = logging.getLogger(__name__)


class VectorStore:
    """基于 FAISS 的向量存储与检索"""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []
        self._dim: int | None = None

    # ── 索引构建 ──

    def build(self, chunks: list[Chunk]) -> None:
        """从文档分块构建索引"""
        if not chunks:
            logger.warning("没有分块可索引")
            return

        self.chunks = chunks
        texts = [c.content for c in chunks]

        logger.info("正在向量化 %d 个文本块...", len(texts))
        vectors = embed_texts(texts)
        self._dim = vectors.shape[1]

        # 使用内积相似度（向量已归一化，等价于余弦相似度）
        self.index = faiss.IndexFlatIP(self._dim)
        self.index.add(vectors)

        logger.info("索引构建完成: %d 条, 维度 %d", self.index.ntotal, self._dim)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """增量添加文档分块"""
        if not chunks:
            return

        texts = [c.content for c in chunks]
        vectors = embed_texts(texts)

        if self.index is None:
            self._dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self._dim)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info("增量添加 %d 条, 总计 %d 条", len(chunks), self.index.ntotal)

    # ── 检索 ──

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """检索与 query 最相关的 top_k 个分块，返回 (chunk, score) 列表"""
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = top_k or Config.TOP_K
        query_vec = embed_query(query).reshape(1, -1)

        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    # ── 持久化 ──

    def save(self, dir_path: str | Path | None = None) -> None:
        """保存索引和元数据到磁盘"""
        dir_path = Path(dir_path or Config.INDEX_DIR)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(dir_path / "index.faiss"))

        with open(dir_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("索引已保存到 %s", dir_path)

    def load(self, dir_path: str | Path | None = None) -> bool:
        """从磁盘加载索引，返回是否成功"""
        dir_path = Path(dir_path or Config.INDEX_DIR)

        index_file = dir_path / "index.faiss"
        chunks_file = dir_path / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            self._dim = self.index.d
            logger.info("索引已加载: %d 条", self.index.ntotal)
            return True
        except Exception as e:
            logger.error("加载索引失败: %s", e)
            return False

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0
