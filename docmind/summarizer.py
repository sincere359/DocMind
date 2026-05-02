"""文档摘要生成器"""
from __future__ import annotations

import json
import logging

from docmind.document_parser import Chunk
from docmind.mimo_client import get_mimo_client
from docmind.config import Config

logger = logging.getLogger(__name__)


class Summarizer:
    """文档智能摘要"""

    def __init__(self):
        self.client = get_mimo_client()

    def summarize(self, chunks: list[Chunk], style: str = "detailed") -> dict:
        """
        生成文档摘要

        Args:
            chunks: 文档分块列表
            style: 摘要风格 - brief(简短) / detailed(详细) / academic(学术)
        """
        full_text = "\n\n".join([c.content for c in chunks])

        # 文档过长时先分块摘要再合并
        if len(full_text) > 8000:
            return self._hierarchical_summarize(chunks, style)

        system_prompt = self._get_system_prompt(style)

        user_prompt = f"""请对以下文档内容生成{style}摘要：

---
{full_text}
---

请严格按 JSON 格式返回：
{{
    "title": "文档核心主题",
    "summary": "摘要正文",
    "key_points": ["要点1", "要点2", "要点3"],
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "doc_type": "文档类型（论文/报告/合同/技术文档/其他）",
    "word_count": 原文大致字数
}}"""

        response = self.client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.client.model_pro,
            temperature=0.2,
            json_mode=True,
        )

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "title": "摘要",
                "summary": response,
                "key_points": [],
                "keywords": [],
                "doc_type": "未知",
                "word_count": len(full_text),
            }

        return result

    def _hierarchical_summarize(self, chunks: list[Chunk], style: str) -> dict:
        """分层摘要：先分块摘要，再合并"""
        # 每 10 个 chunk 为一组
        group_size = 10
        chunk_groups = [chunks[i:i+group_size] for i in range(0, len(chunks), group_size)]

        sub_summaries = []
        for i, group in enumerate(chunk_groups):
            text = "\n\n".join([c.content for c in group])
            prompt = f"请用2-3句话概括以下文档片段的核心内容：\n\n{text}"
            summary = self.client.fast_chat(
                system="你是一个文档摘要专家，擅长精炼概括。",
                user=prompt,
            )
            sub_summaries.append(summary)

        # 合并子摘要
        combined = "\n\n".join([f"片段{i+1}: {s}" for i, s in enumerate(sub_summaries)])

        system_prompt = self._get_system_prompt(style)
        user_prompt = f"""以下是一篇长文档各部分的摘要，请合并为一份完整的{style}摘要：

{combined}

请严格按 JSON 格式返回：
{{
    "title": "文档核心主题",
    "summary": "摘要正文",
    "key_points": ["要点1", "要点2", "要点3"],
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "doc_type": "文档类型",
    "word_count": 原文大致字数
}}"""

        response = self.client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.client.model_pro,
            temperature=0.2,
            json_mode=True,
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "title": "摘要",
                "summary": response,
                "key_points": sub_summaries,
                "keywords": [],
                "doc_type": "未知",
                "word_count": sum(len(c.content) for c in chunks),
            }

    def _get_system_prompt(self, style: str) -> str:
        prompts = {
            "brief": "你是文档摘要专家。请用3-5句话概括文档核心内容，简洁精炼。",
            "detailed": "你是文档摘要专家。请生成详细摘要，包含：核心主题、关键论点、数据支撑、结论建议。使用列表结构，条理清晰。",
            "academic": "你是学术论文摘要专家。请按学术摘要格式生成：研究背景、方法、主要发现、结论。语言严谨客观。",
        }
        return prompts.get(style, prompts["detailed"])
