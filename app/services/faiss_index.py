from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from app.core.config import settings

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # type: ignore
    _faiss_import_error = e
else:
    _faiss_import_error = None


@dataclass
class IndexHandle:
    index: "faiss.Index"
    vector_id_to_chunk_id: List[str]
    model_type: str


_lock = RLock()
_cache: Dict[tuple[str, str], IndexHandle] = {}


def _paths(document_id: str, model_type: str) -> Tuple[Path, Path]:
    idx_dir = settings.INDEX_DIR
    idx_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = idx_dir / f"{document_id}__{model_type}.faiss"
    map_path = idx_dir / f"{document_id}__{model_type}.mapping.json"
    return faiss_path, map_path


def ensure_faiss_available() -> None:
    if faiss is None:  # pragma: no cover
        raise RuntimeError(
            "FAISS is not available. Install faiss-cpu first. "
            f"Import error: {_faiss_import_error}"
        )


def build_and_save_index(
    document_id: str,
    model_type: str,
    vectors: np.ndarray,
    chunk_ids: List[str],
) -> IndexHandle:
    ensure_faiss_available()

    if len(vectors) != len(chunk_ids):
        raise ValueError("vectors and chunk_ids length mismatch")

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    handle = IndexHandle(index=index, vector_id_to_chunk_id=chunk_ids, model_type=model_type)

    faiss_path, map_path = _paths(document_id, model_type)
    logger.info(f"Saving FAISS index: {faiss_path}")
    faiss.write_index(index, str(faiss_path))

    logger.info(f"Saving mapping: {map_path}")
    map_path.write_text(
        json.dumps({"vector_id_to_chunk_id": chunk_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with _lock:
        _cache[(document_id, model_type)] = handle

    return handle


def load_index(document_id: str, model_type: str) -> IndexHandle:
    ensure_faiss_available()
    with _lock:
        if (document_id, model_type) in _cache:
            return _cache[(document_id, model_type)]

    faiss_path, map_path = _paths(document_id, model_type)
    if not faiss_path.exists() or not map_path.exists():
        raise FileNotFoundError(f"Index files not found for {document_id} ({model_type})")

    index = faiss.read_index(str(faiss_path))
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    vector_id_to_chunk_id = mapping.get("vector_id_to_chunk_id", [])
    handle = IndexHandle(index=index, vector_id_to_chunk_id=vector_id_to_chunk_id, model_type=model_type)

    with _lock:
        _cache[(document_id, model_type)] = handle

    return handle


def search(
    document_id: str,
    model_type: str,
    query_vector: np.ndarray,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    handle = load_index(document_id, model_type)

    if query_vector.ndim == 1:
        q = query_vector.reshape(1, -1).astype(np.float32)
    else:
        q = query_vector.astype(np.float32)

    scores, ids = handle.index.search(q, top_k)
    # faiss returns arrays of shape (1, k)
    vector_ids = [int(x) for x in ids[0].tolist() if int(x) != -1]
    vector_scores = [float(s) for s in scores[0].tolist()[: len(vector_ids)]]
    return vector_ids, vector_scores
