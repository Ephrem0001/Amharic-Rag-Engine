from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # Paths
    INDEX_DIR: Path = Path("./indexes")
    UPLOAD_DIR: Path = Path("./data/uploads")

    # Models
    EMBEDDING_MODEL_BASE: str = "rasyosef/roberta-amharic-text-embedding-base"
    EMBEDDING_MODEL_MEDIUM: str = "rasyosef/roberta-amharic-text-embedding-medium"
    EMBEDDING_MODEL_FINETUNED: Path = Path("models/embeddings_finetuned")

    GENERATOR_MODEL: str = "EthioNLP/Amharic-LLAMA-all-data"

    # Hardware
    DEVICE: str = "cpu"
    TORCH_DTYPE: str = "float32"

    # Chunking
    CHUNK_TARGET_CHARS: int = 1000
    CHUNK_OVERLAP_RATIO: float = 0.15

    # Retrieval
    TOP_K_DEFAULT: int = 5

    # Text generation defaults
    TEXTGEN_MAX_NEW_TOKENS: int = 256
    TEXTGEN_TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
