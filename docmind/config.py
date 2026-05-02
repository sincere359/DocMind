"""全局配置，从 .env 加载"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


class Config:
    # ── MiMo API ──
    MIMO_API_KEY: str = os.getenv("MIMO_API_KEY", "")
    MIMO_BASE_URL: str = os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    MIMO_MODEL_PRO: str = os.getenv("MIMO_MODEL_PRO", "mimo-v2.5-pro")
    MIMO_MODEL_FAST: str = os.getenv("MIMO_MODEL_FAST", "mimo-v2.5")

    # ── Embedding ──
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # ── Paths ──
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    INDEX_DIR: Path = DATA_DIR / "index"
    HISTORY_DIR: Path = DATA_DIR / "history"

    # ── Limits ──
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "5"))

    # ── Cleanup ──
    TEMP_CLEANUP: bool = os.getenv("TEMP_CLEANUP", "true").lower() == "true"

    # ── Server ──
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8501"))

    @classmethod
    def ensure_dirs(cls):
        """确保必要目录存在"""
        for d in [cls.UPLOAD_DIR, cls.INDEX_DIR, cls.HISTORY_DIR]:
            d.mkdir(parents=True, exist_ok=True)


Config.ensure_dirs()
