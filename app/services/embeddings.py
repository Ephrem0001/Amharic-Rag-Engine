from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from app.core.config import settings


class EmbeddingModelType(str, Enum):
    BASE = "base"
    MEDIUM = "medium"
    FINETUNED = "finetuned"


# Cap input length to avoid OOM; SentenceTransformer handles truncation internally
SAFE_MAX_CHARS = 2000


def _get_model_name(model_type: EmbeddingModelType) -> str:
    if model_type == EmbeddingModelType.BASE:
        return settings.EMBEDDING_MODEL_BASE
    if model_type == EmbeddingModelType.MEDIUM:
        return settings.EMBEDDING_MODEL_MEDIUM
    if model_type == EmbeddingModelType.FINETUNED:
        return str(settings.EMBEDDING_MODEL_FINETUNED)
    return settings.EMBEDDING_MODEL_BASE


def _safe_text(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    if len(t) > SAFE_MAX_CHARS:
        t = t[:SAFE_MAX_CHARS]
    return t


def _load_encoder_with_slow_tokenizer(model_id_or_path: str) -> SentenceTransformer:
    """Build SentenceTransformer with use_fast=False to avoid PyPreTokenizerTypeWrapper on Windows."""
    logger.info(f"Loading embedding model with slow tokenizer: {model_id_or_path}")
    transformer = Transformer(
        model_id_or_path,
        tokenizer_args={"use_fast": False},
    )
    pooling = Pooling(transformer.get_word_embedding_dimension(), "mean")
    return SentenceTransformer(modules=[transformer, pooling])


@lru_cache(maxsize=8)
def _load_encoder(model_id_or_path: str) -> SentenceTransformer:
    """Load embedding model; prefer slow tokenizer to avoid fast-tokenizer bugs on Windows."""
    logger.info(f"Loading embedding model: {model_id_or_path}")
    try:
        return _load_encoder_with_slow_tokenizer(model_id_or_path)
    except Exception as e:
        logger.warning(f"Slow tokenizer failed ({e}), trying default SentenceTransformer load")
        return SentenceTransformer(model_id_or_path)


def resolve_embedding_model(model_type: EmbeddingModelType) -> Tuple[str, SentenceTransformer]:
    name = _get_model_name(model_type)
    return name, _load_encoder(name)


def embed_texts(
    texts: List[str],
    model_type: EmbeddingModelType = EmbeddingModelType.BASE,
    batch_size: int = 32,
) -> Tuple[str, np.ndarray]:
    model_name, model = resolve_embedding_model(model_type)

    texts = [_safe_text(x) for x in texts]
    if not texts:
        return model_name, np.zeros((0, 0), dtype=np.float32)

    # encode() returns normalized embeddings for retrieval models; numpy by default
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    if isinstance(vectors, np.ndarray):
        pass
    else:
        vectors = np.array(vectors, dtype=np.float32)

    return model_name, vectors
