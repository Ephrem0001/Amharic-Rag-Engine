from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings


EmbeddingModelType = Literal["base", "medium", "finetuned"]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_id_or_path: str) -> SentenceTransformer:
    logger.info(f"Loading embedding model: {model_id_or_path}")
    return SentenceTransformer(model_id_or_path, device=settings.DEVICE)


def resolve_embedding_model(model_type: EmbeddingModelType) -> Tuple[str, SentenceTransformer]:
    if model_type == "base":
        model_name = settings.EMBEDDING_MODEL_BASE
        return model_name, _load_sentence_transformer(model_name)
    if model_type == "medium":
        model_name = settings.EMBEDDING_MODEL_MEDIUM
        return model_name, _load_sentence_transformer(model_name)

    # finetuned
    finetuned_path = settings.EMBEDDING_MODEL_FINETUNED
    if isinstance(finetuned_path, Path):
        model_name = str(finetuned_path)
    else:
        model_name = str(finetuned_path)
    return model_name, _load_sentence_transformer(model_name)


def embed_texts(texts: Sequence[str], model_type: EmbeddingModelType = "base", batch_size: int = 32) -> Tuple[str, np.ndarray]:
    model_name, model = resolve_embedding_model(model_type)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    vectors = vectors.astype(np.float32)
    vectors = _normalize(vectors)
    return model_name, vectors
