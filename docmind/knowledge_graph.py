"""知识图谱构建器：从文档中提取实体与关系，生成图数据"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from docmind.document_parser import Chunk
from docmind.mimo_client import get_mimo_client

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """图节点"""
    id: str
    label: str
    category: str = "default"  # person / org / location / concept / event
    size: int = 25


@dataclass
class GraphEdge:
    """图边"""
    source: str
    target: str
    label: str = ""
    weight: float = 1.0


@dataclass
class KnowledgeGraph:
    """知识图谱数据"""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def to_agraph_data(self) -> dict:
        """转换为 streamlit-agraph 需要的格式"""
        nodes = [
            {
                "id": n.id,
                "label": n.label,
                "size": n.size,
                "color": _category_color(n.category),
            }
            for n in self.nodes
        ]
        edges = [
            {
                "source": e.source,
                "target": e.target,
                "label": e.label,
            }
            for e in self.edges
        ]
        return {"nodes": nodes, "edges": edges}


# 类别 → 颜色
_CATEGORY_COLORS = {
    "person": "#e74c3c",
    "org": "#3498db",
    "organization": "#3498db",
    "location": "#2ecc71",
    "concept": "#9b59b6",
    "event": "#f39c12",
    "date": "#1abc9c",
    "default": "#95a5a6",
}


def _category_color(cat: str) -> str:
    return _CATEGORY_COLORS.get(cat.lower(), _CATEGORY_COLORS["default"])


class KnowledgeGraphBuilder:
    """基于 LLM 的知识图谱构建"""

    def __init__(self):
        self.client = get_mimo_client()

    def build(self, chunks: list[Chunk], max_chunks: int = 15) -> KnowledgeGraph:
        """
        从文档 chunks 中构建知识图谱

        策略：
        1. 每 3-5 个 chunk 一组，LLM 提取实体+关系
        2. 合并去重，生成最终图
        """
        # 限制 chunk 数量避免 token 消耗过大
        selected = chunks[:max_chunks]
        group_size = 5

        all_nodes: dict[str, GraphNode] = {}
        all_edges: list[GraphEdge] = []

        for i in range(0, len(selected), group_size):
            group = selected[i:i + group_size]
            text = "\n\n".join([c.content for c in group])

            try:
                result = self._extract_graph(text)
            except Exception as e:
                logger.warning("图谱提取失败 (chunk %d-%d): %s", i, i + len(group), e)
                continue

            # 合并节点
            for node_data in result.get("nodes", []):
                nid = node_data.get("id", node_data.get("label", ""))
                if nid and nid not in all_nodes:
                    all_nodes[nid] = GraphNode(
                        id=nid,
                        label=node_data.get("label", nid),
                        category=node_data.get("category", "default"),
                        size=node_data.get("size", 25),
                    )

            # 合并边（去重）
            existing_pairs = {(e.source, e.target) for e in all_edges}
            for edge_data in result.get("edges", []):
                src = edge_data.get("source", "")
                tgt = edge_data.get("target", "")
                if src and tgt and (src, tgt) not in existing_pairs:
                    all_edges.append(GraphEdge(
                        source=src,
                        target=tgt,
                        label=edge_data.get("label", ""),
                        weight=edge_data.get("weight", 1.0),
                    ))
                    existing_pairs.add((src, tgt))

        # 根据连接数调整节点大小
        connection_count: dict[str, int] = {}
        for e in all_edges:
            connection_count[e.source] = connection_count.get(e.source, 0) + 1
            connection_count[e.target] = connection_count.get(e.target, 0) + 1

        for nid, node in all_nodes.items():
            node.size = 25 + connection_count.get(nid, 0) * 5

        return KnowledgeGraph(nodes=list(all_nodes.values()), edges=all_edges)

    def _extract_graph(self, text: str) -> dict:
        """调用 LLM 从文本中提取实体和关系"""
        system_prompt = """你是知识图谱构建专家。从文本中提取实体和关系，严格按 JSON 格式返回：

{
    "nodes": [
        {"id": "唯一标识", "label": "显示名称", "category": "person|org|location|concept|event|date"}
    ],
    "edges": [
        {"source": "节点id", "target": "节点id", "label": "关系描述"}
    ]
}

规则：
1. 实体 id 用简洁英文或拼音，label 用原文
2. category 尽量准确分类
3. 只提取文本中明确提到的关系
4. 关系标签简洁（如"属于"、"位于"、"创建"、"包含"）
5. 如果文本太短或无实质内容，返回空数组"""

        response = self.client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请从以下文本中提取知识图谱：\n\n{text[:3000]}"},
            ],
            model=self.client.model_fast,
            temperature=0.1,
            json_mode=True,
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("图谱 JSON 解析失败: %s", response[:200])
            return {"nodes": [], "edges": []}
