"""对话历史持久化管理"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from docmind.config import Config


class ChatHistoryManager:
    """管理聊天记录的保存/加载/删除"""

    def __init__(self, history_dir: Path | None = None):
        self.history_dir = history_dir or Config.HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_conversation(self, messages: list[dict], conversation_id: str | None = None) -> str:
        """保存对话，返回对话 ID"""
        if not conversation_id:
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        filepath = self.history_dir / f"{conversation_id}.json"
        data = {
            "id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages,
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return conversation_id

    def load_conversation(self, conversation_id: str) -> list[dict] | None:
        """根据 ID 加载对话"""
        filepath = self.history_dir / f"{conversation_id}.json"
        if not filepath.exists():
            return None
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return data.get("messages", [])
        except (json.JSONDecodeError, KeyError):
            return None

    def list_conversations(self) -> list[dict]:
        """列出所有保存的对话（仅元数据）"""
        conversations = []
        for fp in sorted(self.history_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                conversations.append({
                    "id": data.get("id", fp.stem),
                    "created_at": data.get("created_at", ""),
                    "message_count": data.get("message_count", 0),
                    "preview": (data.get("messages", [{}])[0].get("content", ""))[:50] if data.get("messages") else "",
                })
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        return conversations

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除指定对话"""
        filepath = self.history_dir / f"{conversation_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def load_latest(self) -> list[dict]:
        """加载最近的对话，若无则返回空列表"""
        convs = self.list_conversations()
        if not convs:
            return []
        return self.load_conversation(convs[0]["id"]) or []
