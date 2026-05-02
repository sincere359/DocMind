"""MiMo API 客户端，基于 OpenAI 兼容 SDK"""
from __future__ import annotations

import json
import logging
from typing import Generator

from openai import OpenAI

from docmind.config import Config

logger = logging.getLogger(__name__)


class MimoClient:
    """MiMo Token Plan API 封装"""

    def __init__(self):
        self.client = OpenAI(
            api_key=Config.MIMO_API_KEY,
            base_url=Config.MIMO_BASE_URL,
        )
        self.model_pro = Config.MIMO_MODEL_PRO
        self.model_fast = Config.MIMO_MODEL_FAST

    # ── 核心调用 ──

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """同步聊天，返回 assistant 文本"""
        kwargs: dict = dict(
            model=model or self.model_pro,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """流式聊天，逐 token 返回"""
        stream = self.client.chat.completions.create(
            model=model or self.model_pro,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # ── 高级封装 ──

    def reasoning_chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        """深度推理模式 (Pro 模型)"""
        return self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=self.model_pro,
            temperature=temperature,
            max_tokens=4096,
        )

    def fast_chat(self, system: str, user: str, temperature: float = 0.5) -> str:
        """快速模式 (Fast 模型)"""
        return self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=self.model_fast,
            temperature=temperature,
            max_tokens=2048,
        )

    def reasoning_chat_stream(self, system: str, user: str, temperature: float = 0.2):
        """深度推理流式"""
        return self.chat_stream(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=self.model_pro,
            temperature=temperature,
            max_tokens=4096,
        )

    def count_tokens(self, text: str) -> int:
        """粗略估算 token 数（中文约 1.5 字/token）"""
        return int(len(text) * 0.7)


# 全局单例
_client: MimoClient | None = None


def get_mimo_client() -> MimoClient:
    global _client
    if _client is None:
        _client = MimoClient()
    return _client
