"""MiMo API 客户端，基于 OpenAI 兼容 SDK"""
from __future__ import annotations

import json
import logging
import time
from typing import Generator

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from docmind.config import Config

logger = logging.getLogger(__name__)


class MimoAPIError(Exception):
    """MiMo API 调用异常"""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class MimoClient:
    """MiMo Token Plan API 封装"""

    MAX_RETRIES = 2
    RETRY_DELAY = 1.0  # 秒

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
        """同步聊天，返回 assistant 文本，含重试逻辑"""
        kwargs: dict = dict(
            model=model or self.model_pro,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return content or ""
            except RateLimitError as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning("Rate limit, %.1fs 后重试 (%d/%d)", delay, attempt + 1, self.MAX_RETRIES)
                    time.sleep(delay)
                else:
                    raise MimoAPIError("API 请求频率超限，请稍后再试", e)
            except APIConnectionError as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning("连接失败, %.1fs 后重试 (%d/%d)", delay, attempt + 1, self.MAX_RETRIES)
                    time.sleep(delay)
                else:
                    raise MimoAPIError("无法连接 MiMo API，请检查网络", e)
            except APIError as e:
                raise MimoAPIError(f"API 错误: {e.message}", e)
            except Exception as e:
                raise MimoAPIError(f"未知错误: {e}", e)

        raise MimoAPIError("请求失败", last_error)

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """流式聊天，逐 token 返回"""
        try:
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
        except RateLimitError as e:
            raise MimoAPIError("API 请求频率超限，请稍后再试", e)
        except APIConnectionError as e:
            raise MimoAPIError("无法连接 MiMo API，请检查网络", e)
        except APIError as e:
            raise MimoAPIError(f"API 错误: {e.message}", e)
        except Exception as e:
            raise MimoAPIError(f"未知错误: {e}", e)

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


# 全局单例
_client: MimoClient | None = None


def get_mimo_client() -> MimoClient:
    global _client
    if _client is None:
        _client = MimoClient()
    return _client
