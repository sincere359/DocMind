"""RAG 引擎：检索增强生成核心"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, Tuple

from docmind.config import Config
from docmind.document_parser import Chunk
from docmind.vector_store import VectorStore
from docmind.mimo_client import get_mimo_client, MimoAPIError

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG 回复"""
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    model: str = ""
    error: str = ""


class RAGEngine:
    """检索增强生成引擎"""

    def __init__(self, vector_store: VectorStore | None = None):
        self.store = vector_store or VectorStore()
        self.client = get_mimo_client()

    # ── 核心问答 ──

    def ask(self, query: str, top_k: int | None = None, chat_history: list[dict] | None = None) -> RAGResponse:
        """基于文档的智能问答（同步）"""
        if self.store.is_empty:
            return RAGResponse(answer="请先上传文档，我才能回答问题。", query=query)

        # 1. 检索相关文档片段
        results = self.store.search(query, top_k=top_k or Config.TOP_K)
        if not results:
            return RAGResponse(answer="未找到与问题相关的文档内容，请尝试换个表述。", query=query)

        # 2. 构建上下文和来源
        context, sources = self._build_context(results)

        # 3. 构建消息列表（含多轮历史）
        messages = self._build_messages(context, query, chat_history)

        # 4. 生成回答
        try:
            answer = self.client.chat(messages=messages, temperature=0.2)
        except MimoAPIError as e:
            return RAGResponse(answer="", query=query, error=str(e))

        return RAGResponse(answer=answer, sources=sources, query=query, model=Config.MIMO_MODEL_PRO)

    def ask_stream(self, query: str, top_k: int | None = None, chat_history: list[dict] | None = None):
        """流式问答，逐 token 返回"""
        if self.store.is_empty:
            yield "请先上传文档，我才能回答问题。"
            return

        results = self.store.search(query, top_k=top_k or Config.TOP_K)
        if not results:
            yield "未找到与问题相关的文档内容，请尝试换个表述。"
            return

        context, _ = self._build_context(results)
        messages = self._build_messages(context, query, chat_history)

        try:
            for token in self.client.chat_stream(messages=messages, temperature=0.2):
                yield token
        except MimoAPIError as e:
            yield f"\n\n⚠️ {e}"

    def ask_with_sources(
        self, query: str, top_k: int | None = None, chat_history: list[dict] | None = None
    ) -> Tuple[Generator[str, None, None], list[dict]]:
        """流式问答 + 返回来源信息，供 UI 分离渲染"""
        if self.store.is_empty:
            return (t for t in ["请先上传文档，我才能回答问题。"]), []

        results = self.store.search(query, top_k=top_k or Config.TOP_K)
        if not results:
            return (t for t in ["未找到与问题相关的文档内容，请尝试换个表述。"]), []

        context, sources = self._build_context(results)
        messages = self._build_messages(context, query, chat_history)

        try:
            stream = self.client.chat_stream(messages=messages, temperature=0.2)
        except MimoAPIError as e:
            return (t for t in [f"⚠️ {e}"]), sources

        return stream, sources

    # ── 文档索引 ──

    def index_documents(self, file_paths: list[str]) -> int:
        """解析并索引文档，返回分块总数"""
        from docmind.document_parser import parse_documents
        chunks = parse_documents(file_paths)
        self.store.build(chunks)
        self.store.save()
        return len(chunks)

    def add_documents(self, file_paths: list[str]) -> int:
        """增量添加文档，返回新增分块数"""
        from docmind.document_parser import parse_documents
        chunks = parse_documents(file_paths)
        self.store.add_chunks(chunks)
        self.store.save()
        return len(chunks)

    # ── 内部方法 ──

    def _build_context(self, results: list[tuple[Chunk, float]]) -> tuple[str, list[dict]]:
        """构建检索上下文和来源信息"""
        context_parts = []
        sources = []
        for i, (chunk, score) in enumerate(results):
            context_parts.append(
                f"【文档片段 {i+1}】(来源: {chunk.source}, 相关度: {score:.3f})\n{chunk.content}"
            )
            sources.append({
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 4),
                "content": chunk.content,
                "preview": chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content,
            })
        return "\n\n".join(context_parts), sources

    def _build_messages(
        self, context: str, query: str, chat_history: list[dict] | None = None
    ) -> list[dict]:
        """构建完整消息列表，含系统提示、历史对话和当前问题"""
        system_prompt = self._build_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]

        # 注入多轮对话历史（最近 N 轮）
        if chat_history:
            max_turns = Config.MAX_HISTORY_TURNS
            recent = chat_history[-(max_turns * 2):]  # 每轮 = user + assistant
            for msg in recent:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # 当前问题 + 参考文档
        user_prompt = f"""## 参考文档

{context}

## 用户问题

{query}

请基于以上参考文档内容回答用户问题。要求：
1. 回答必须基于文档内容，不要编造信息
2. 如果文档中没有相关信息，请明确告知
3. 引用文档内容时标注来源片段编号
4. 回答要完整、准确、有条理"""

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _build_system_prompt(self) -> str:
        return """你是 DocMind 文档智能助手，一个专业的文档分析AI。

你正在与用户进行多轮对话，请综合考虑之前的对话上下文和当前问题。

核心能力：
1. 基于用户提供的文档内容，准确回答问题
2. 如果用户追问，基于之前的回答深入展开
3. 提取关键信息，生成结构化摘要
4. 对比多个文档，发现异同
5. 用清晰、专业的语言组织回答

回答原则：
- 严格基于文档内容，不编造、不推测
- 标注信息来源（片段编号）
- 如果文档中没有相关信息，明确告知
- 回答条理清晰，使用列表、加粗等格式
- 适当使用表格对比信息"""
