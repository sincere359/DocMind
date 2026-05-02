"""导出功能：将摘要/提取/聊天记录导出为 JSON/TXT/Markdown"""
from __future__ import annotations

import json
from datetime import datetime


class Exporter:
    """导出结果为可下载格式"""

    @staticmethod
    def to_json(data: dict, filename: str | None = None) -> tuple[str, str]:
        """导出为 JSON，返回 (content, filename)"""
        content = json.dumps(data, ensure_ascii=False, indent=2)
        if not filename:
            filename = f"docmind_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return content, filename

    @staticmethod
    def to_text(data: dict, title: str = "DocMind Export") -> tuple[str, str]:
        """导出为可读文本格式"""
        lines = [
            "=" * 50,
            f"  {title}",
            f"  导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
        ]

        def _flatten(obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)) and v:
                        lines.append(f"{prefix}{k}:")
                        _flatten(v, indent + 1)
                    else:
                        lines.append(f"{prefix}{k}: {v}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        _flatten(item, indent)
                        lines.append("")
                    else:
                        lines.append(f"{prefix}- {item}")

        _flatten(data)
        content = "\n".join(lines)
        filename = f"docmind_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return content, filename

    @staticmethod
    def summary_to_markdown(summary: dict, source: str) -> tuple[str, str]:
        """导出摘要为 Markdown"""
        lines = [
            f"# {summary.get('title', '文档摘要')}",
            "",
            f"**来源**: {source}  ",
            f"**类型**: {summary.get('doc_type', '未知')}  ",
            f"**关键词**: {', '.join(summary.get('keywords', []))}  ",
            "",
            "## 摘要",
            "",
            summary.get("summary", ""),
            "",
        ]
        if summary.get("key_points"):
            lines.append("## 关键要点")
            lines.append("")
            for pt in summary["key_points"]:
                lines.append(f"- {pt}")
            lines.append("")

        content = "\n".join(lines)
        filename = f"summary_{source}_{datetime.now().strftime('%Y%m%d')}.md"
        return content, filename

    @staticmethod
    def chat_to_markdown(messages: list[dict]) -> tuple[str, str]:
        """导出聊天记录为 Markdown"""
        lines = [
            "# DocMind 对话记录",
            "",
            f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]
        for msg in messages:
            role = "🧑 用户" if msg["role"] == "user" else "🤖 DocMind"
            lines.append(f"### {role}")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")
            lines.append("---")
            lines.append("")

        content = "\n".join(lines)
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        return content, filename
