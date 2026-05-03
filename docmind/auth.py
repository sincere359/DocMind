"""用户认证：注册 / 登录 / JWT 验证 / per-user FAISS 索引"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from docmind.config import Config

logger = logging.getLogger(__name__)

# ── 用户数据模型 ──


@dataclass
class User:
    id: int
    username: str
    password_hash: str
    created_at: float


# ── SQLite UserStore ──


class UserStore:
    """用户存储（SQLite）"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or Config.DATA_DIR / "users.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ── CRUD ──

    def create_user(self, username: str, password: str) -> User | None:
        """创建用户，成功返回 User，用户名已存在返回 None"""
        pw_hash = self._hash_password(password)
        try:
            with self._conn() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, pw_hash, time.time()),
                )
                uid = cursor.lastrowid
                return User(id=uid, username=username, password_hash=pw_hash, created_at=time.time())
        except sqlite3.IntegrityError:
            return None

    def get_user(self, username: str) -> User | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if row:
                return User(*row)
        return None

    def verify_user(self, username: str, password: str) -> User | None:
        """验证用户名+密码，成功返回 User，失败返回 None"""
        user = self.get_user(username)
        if user and user.password_hash == self._hash_password(password):
            return user
        return None

    def list_users(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT username FROM users ORDER BY id").fetchall()
            return [r[0] for r in rows]

    def delete_user(self, username: str) -> bool:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM users WHERE username = ?", (username,))
            return cursor.rowcount > 0

    # ── 密码哈希 ──

    @staticmethod
    def _hash_password(password: str) -> str:
        """SHA-256 + salt 哈希"""
        salt = "docmind_salt_2024"
        return hashlib.sha256(f"{salt}{password}{salt}".encode()).hexdigest()


# ── JWT 简易实现（不依赖 PyJWT）──


def _base64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _base64url_decode(s: str) -> bytes:
    import base64
    padding = 4 - len(s) % 4
    s += "=" * padding
    return base64.urlsafe_b64decode(s)


def create_jwt(payload: dict, secret: str | None = None) -> str:
    """创建 JWT token"""
    secret = secret or Config.JWT_SECRET
    header = {"alg": "HS256", "typ": "JWT"}

    header_b64 = _base64url_encode(json.dumps(header).encode())
    payload_b64 = _base64url_encode(json.dumps(payload).encode())

    signing_input = f"{header_b64}.{payload_b64}"
    signature = hashlib.sha256(f"{signing_input}.{secret}".encode()).hexdigest()
    sig_b64 = _base64url_encode(signature.encode())

    return f"{signing_input}.{sig_b64}"


def verify_jwt(token: str, secret: str | None = None) -> dict | None:
    """验证 JWT token，有效返回 payload，无效返回 None"""
    secret = secret or Config.JWT_SECRET
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hashlib.sha256(f"{signing_input}.{secret}".encode()).hexdigest()
        actual_sig = _base64url_decode(sig_b64).decode()

        if actual_sig != expected_sig:
            return None

        payload = json.loads(_base64url_decode(payload_b64))

        # 检查过期
        if "exp" in payload and payload["exp"] < time.time():
            return None

        return payload
    except Exception:
        return None


def create_access_token(username: str) -> str:
    """为用户创建 access token"""
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + Config.JWT_EXPIRE_SECONDS,
    }
    return create_jwt(payload)


# ── Per-user FAISS 索引隔离 ──


def get_user_index_dir(username: str) -> Path:
    """获取用户专属的索引目录"""
    user_hash = hashlib.md5(username.encode()).hexdigest()[:12]
    index_dir = Config.INDEX_DIR / f"user_{user_hash}"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir
