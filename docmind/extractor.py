"""关键信息提取器"""
from __future__ import annotations

import json
import logging

from docmind.document_parser import Chunk
from docmind.mimo_client import get_mimo_client

logger = logging.getLogger(__name__)


class Extractor:
    """文档关键信息提取"""

    def __init__(self):
        self.client = get_mimo_client()

    def extract(self, chunks: list[Chunk]) -> dict:
        """提取文档关键信息"""
        full_text = "\n\n".join([c.content for c in chunks])

        # 长文档分段提取
        if len(full_text) > 6000:
            return self._extract_long(chunks)

        system_prompt = """你是文档信息提取专家。请从文档中提取以下关键信息，严格按 JSON 格式返回：

{
    "entities": {
        "people": ["人名列表"],
        "organizations": ["组织/机构列表"],
        "locations": ["地点列表"],
        "dates": ["日期/时间列表"]
    },
    "numbers": [
        {"value": "数值", "context": "数值含义", "unit": "单位"}
    ],
    "key_terms": [
        {"term": "术语", "definition": "定义/解释"}
    ],
    "actions": ["需要执行的行动/待办事项"],
    "conclusions": ["核心结论/观点"],
    "risks": ["风险/问题/挑战"],
    "recommendations": ["建议/推荐"]
}

如果某类信息不存在，返回空列表。不要编造信息。"""

        user_prompt = f"请从以下文档中提取关键信息：\n\n{full_text}"

        response = self.client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.client.model_pro,
            temperature=0.1,
            json_mode=True,
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "提取结果解析失败", "raw": response}

    def _extract_long(self, chunks: list[Chunk]) -> dict:
        """长文档分段提取再合并"""
        group_size = 8
        all_entities = {"people": [], "organizations": [], "locations": [], "dates": []}
        all_numbers = []
        all_conclusions = []
        all_risks = []
        all_recommendations = []

        for i in range(0, len(chunks), group_size):
            group = chunks[i:i+group_size]
            text = "\n\n".join([c.content for c in group])

            result = self._extract_single(text)
            # 合并去重
            for key in all_entities:
                all_entities[key].extend(result.get("entities", {}).get(key, []))
            all_numbers.extend(result.get("numbers", []))
            all_conclusions.extend(result.get("conclusions", []))
            all_risks.extend(result.get("risks", []))
            all_recommendations.extend(result.get("recommendations", []))

        # 去重
        for key in all_entities:
            all_entities[key] = list(dict.fromkeys(all_entities[key]))
        all_conclusions = list(dict.fromkeys(all_conclusions))
        all_risks = list(dict.fromkeys(all_risks))
        all_recommendations = list(dict.fromkeys(all_recommendations))

        return {
            "entities": all_entities,
            "numbers": all_numbers[:20],
            "key_terms": [],
            "actions": [],
            "conclusions": all_conclusions[:10],
            "risks": all_risks[:10],
            "recommendations": all_recommendations[:10],
        }

    def _extract_single(self, text: str) -> dict:
        """提取单段文本的关键信息"""
        system_prompt = """从文本中提取关键信息，JSON 格式返回：
{
    "entities": {"people": [], "organizations": [], "locations": [], "dates": []},
    "numbers": [{"value": "", "context": "", "unit": ""}],
    "conclusions": [],
    "risks": [],
    "recommendations": []
}"""

        response = self.client.fast_chat(
            system=system_prompt,
            user=f"提取关键信息：\n{text[:3000]}",
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"entities": {}, "numbers": [], "conclusions": [], "risks": [], "recommendations": []}

    def compare_documents(self, chunks_a: list[Chunk], chunks_b: list[Chunk], name_a: str = "文档A", name_b: str = "文档B") -> dict:
        """对比两个文档的异同"""
        text_a = "\n".join([c.content for c in chunks_a[:10]])[:5000]
        text_b = "\n".join([c.content for c in chunks_b[:10]])[:5000]

        system_prompt = f"""你是文档对比分析专家。请对比以下两个文档，找出异同点。

严格按 JSON 格式返回：
{{
    "common_points": ["共同点1", "共同点2"],
    "differences": [
        {{"aspect": "对比维度", "{name_a}": "文档A的内容", "{name_b}": "文档B的内容"}}
    ],
    "conclusion": "对比结论"
}}"""

        user_prompt = f"""## {name_a}
{text_a}

## {name_b}
{text_b}

请对比分析这两个文档的异同。"""

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
            return {"error": "对比结果解析失败", "raw": response}
