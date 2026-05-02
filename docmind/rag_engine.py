"""RAG 引擎：检索增强生成核心"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from docmind.config import Config
from docmind.document_parser import Chunk
from docmind.vector_store import VectorStore
from docmind.mimo_client import get_mimo_client

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG 回复"""
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    model: str = ""


class RAGEngine:
    """检索增强生成引擎"""

    def __init__(self, vector_store: VectorStore | None = None):
        self.store = vector_store or VectorStore()
        self.client = get_mimo_client()

    # ── 核心问答 ──

    def ask(self, query: str, top_k: int | None = None, stream: bool = False) -> RAGResponse:
        """基于文档的智能问答"""
        if self.store.is_empty:
            return RAGResponse(
                answer="请先上传文档，我才能回答问题。",
                query=query,
            )

        # 1. 检索相关文档片段
        results = self.store.search(query, top_k=top_k or Config.TOP_K)
        if not results:
            return RAGResponse(
                answer="未找到与问题相关的文档内容，请尝试换个表述。",
                query=query,
            )

        # 2. 构建上下文
        context_parts = []
        sources = []
        for i, (chunk, score) in enumerate(results):
            context_parts.append(f"【文档片段 {i+1}】(来源: {chunk.source}, 相关度: {score:.3f})\n{chunk.content}")
            sources.append({
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 4),
                "preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
            })

        context = "\n\n".join(context_parts)

        # 3. 生成回答
        system_prompt = self._build_system_prompt()

        user_prompt = f"""## 参考文档

{context}

## 用户问题

{query}

请基于以上参考文档内容回答用户问题。要求：
1. 回答必须基于文档内容，不要编造信息
2. 如果文档中没有相关信息，请明确告知
3. 引用文档内容时标注来源片段编号
4. 回答要完整、准确、有条理"""

        if stream:
            # 流式返回
            answer_stream = self.client.reasoning_chat_stream(
                system=system_prompt,
                user=user_prompt,
                temperature=0.2,
            )
            return RAGResponse(
                answer="",  # 流式模式下 answer 为空，由调用方消费 stream
                sources=sources,
                query=query,
                model=Config.MIMO_MODEL_PRO,
                _stream=answer_stream,  # type: ignore
            )

        answer = self.client.reasoning_chat(
            system=system_prompt,
            user=user_prompt,
            temperature=0.2,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            model=Config.MIMO_MODEL_PRO,
        )

    def ask_stream(self, query: str, top_k: int | None = None):
        """流式问答，逐 token 返回"""
        if self.store.is_empty:
            yield "请先上传文档，我才能回答问题。"
            return

        results = self.store.search(query, top_k=top_k or Config.TOP_K)
        if not results:
            yield "未找到与问题相关的文档内容，请尝试换个表述。"
            return

        context_parts = []
        for i, (chunk, score) in enumerate(results):
            context_parts.append(f"【文档片段 {i+1}】(来源: {chunk.source})\n{chunk.content}")

        context = "\n\n".join(context_parts)

        system_prompt = self._build_system_prompt()
        user_prompt = f"""## 参考文档

{context}

## 用户问题

{query}

请基于以上参考文档内容回答用户问题。回答必须基于文档内容，不要编造信息。引用时标注片段编号。"""

        for token in self.client.reasoning_chat_stream(
            system=system_prompt,
            user=user_prompt,
            temperature=0.2,
        ):
            yield token

    # ── 文档索引 ──

    def index_documents(self, file_paths: list[str]) -> int:
        """解析并索引文档，返回分块总数"""
        from docmind.document_parser import parse_documents
        chunks = parse_documents(file_paths)
        self.store.build(chunks)
        self.store.save()
        return len(chunks)

    # ── Prompt 工程 ──

    def _build_system_prompt(self) -> str:
        return """你是 DocMind 文档智能助手，一个专业的文档分析AI。

你的核心能力：
1. 基于用户提供的文档内容，准确回答问题
2. 提取关键信息，生成结构化摘要
3. 对比多个文档，发现异同
4. 用清晰、专业的语言组织回答

回答原则：
- 严格基于文档内容，不编造、不推测
- 标注信息来源（片段编号）
- 如果文档中没有相关信息，明确告知
- 回答条理清晰，使用列表、加粗等格式
- 适当使用表格对比信息"""
