"""FAISS 向量存储：文档索引 + 相似度检索 + 文档管理"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from docmind.config import Config
from docmind.document_parser import Chunk
from docmind.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)


@dataclass
class DocumentMeta:
    """单文档统计信息"""
    source: str
    chunk_count: int
    total_chars: int
    file_size: int = 0  # bytes
    indexed_at: str = ""


class VectorStore:
    """基于 FAISS 的向量存储与检索"""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []
        self._dim: int | None = None
        self.source_index: dict[str, list[int]] = {}  # source -> [chunk indices]

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

        self.index = faiss.IndexFlatIP(self._dim)
        self.index.add(vectors)

        self._rebuild_source_index()
        logger.info("索引构建完成: %d 条, 维度 %d", self.index.ntotal, self._dim)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """增量添加文档分块（自动跳过已存在的文档）"""
        if not chunks:
            return

        # 去重：跳过已索引的文档
        new_chunks = [c for c in chunks if not self.has_source(c.source)]
        if not new_chunks:
            logger.info("所有文档已索引，跳过")
            return

        texts = [c.content for c in new_chunks]
        vectors = embed_texts(texts)

        if self.index is None:
            self._dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self._dim)

        start_idx = len(self.chunks)
        self.index.add(vectors)
        self.chunks.extend(new_chunks)

        # 更新 source_index
        for i, chunk in enumerate(new_chunks):
            idx = start_idx + i
            self.source_index.setdefault(chunk.source, []).append(idx)

        logger.info("增量添加 %d 条, 总计 %d 条", len(new_chunks), self.index.ntotal)

    # ── 文档管理 ──

    def has_source(self, source: str) -> bool:
        """检查指定文档是否已索引"""
        return source in self.source_index

    def remove_source(self, source: str) -> bool:
        """删除指定文档的所有 chunks 并重建索引"""
        if source not in self.source_index:
            return False

        # 保留非该文档的 chunks
        remaining_chunks = [c for c in self.chunks if c.source != source]

        if remaining_chunks:
            # 重建索引
            texts = [c.content for c in remaining_chunks]
            vectors = embed_texts(texts)
            self.index = faiss.IndexFlatIP(self._dim or vectors.shape[1])
            self.index.add(vectors)
        else:
            self.index = None

        self.chunks = remaining_chunks
        self._rebuild_source_index()
        self.save()
        logger.info("已删除文档: %s", source)
        return True

    def clear(self) -> None:
        """清空整个索引"""
        self.index = None
        self.chunks = []
        self.source_index = {}
        self.save()
        logger.info("索引已清空")

    def get_document_stats(self) -> list[DocumentMeta]:
        """获取每个已索引文档的统计信息"""
        stats: dict[str, dict] = {}
        for chunk in self.chunks:
            src = chunk.source
            if src not in stats:
                stats[src] = {"chunk_count": 0, "total_chars": 0}
            stats[src]["chunk_count"] += 1
            stats[src]["total_chars"] += len(chunk.content)

        return [
            DocumentMeta(source=s, chunk_count=d["chunk_count"], total_chars=d["total_chars"])
            for s, d in stats.items()
        ]

    def get_sources(self) -> list[str]:
        """获取所有已索引文档名称"""
        return list(self.source_index.keys())

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

        with open(dir_path / "source_index.pkl", "wb") as f:
            pickle.dump(self.source_index, f)

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

            # 加载 source_index（兼容旧版本没有此文件的情况）
            source_index_file = dir_path / "source_index.pkl"
            if source_index_file.exists():
                with open(source_index_file, "rb") as f:
                    self.source_index = pickle.load(f)
            else:
                self._rebuild_source_index()

            logger.info("索引已加载: %d 条", self.index.ntotal)
            return True
        except Exception as e:
            logger.error("加载索引失败: %s", e)
            return False

    # ── 内部方法 ──

    def _rebuild_source_index(self) -> None:
        """从 chunks 重建 source_index"""
        self.source_index = {}
        for i, chunk in enumerate(self.chunks):
            self.source_index.setdefault(chunk.source, []).append(i)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0
