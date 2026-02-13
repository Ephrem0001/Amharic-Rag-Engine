from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Literal, Optional

from loguru import logger
from sqlalchemy.orm import Session

from app.db import models
from app.services.embeddings import EmbeddingModelType, embed_texts
from app.services.faiss_index import load_index, search


@dataclass
class RetrievedChunk:
    chunk_id: str
    page_number: int
    score: float
    chunk_text: str


def retrieve_chunks(
    db: Session,
    document_id: str,
    question: str,
    top_k: int,
    model_type: EmbeddingModelType = "base",
) -> List[RetrievedChunk]:
    model_name, qvecs = embed_texts([question], model_type=model_type, batch_size=1)
    qvec = qvecs[0]

    try:
        handle = load_index(document_id, model_type)
    except FileNotFoundError:
        # Helpful error message
        raise FileNotFoundError(
            f"FAISS index not found for document_id={document_id} model_type={model_type}. "
            "Upload the document first or run /admin/reindex."
        )

    vector_ids, scores = search(document_id, model_type, qvec, top_k=top_k)

    chunk_ids: List[str] = []
    for vid in vector_ids:
        if 0 <= vid < len(handle.vector_id_to_chunk_id):
            chunk_ids.append(handle.vector_id_to_chunk_id[vid])

    if not chunk_ids:
        return []

    # Fetch chunks in one query
    rows = (
        db.query(models.Chunk)
        .filter(models.Chunk.id.in_(chunk_ids))
        .all()
    )
    by_id = {str(r.id): r for r in rows}

    results: List[RetrievedChunk] = []
    for vid, score in zip(vector_ids, scores):
        if 0 <= vid < len(handle.vector_id_to_chunk_id):
            cid = handle.vector_id_to_chunk_id[vid]
            row = by_id.get(str(cid))
            if row is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=str(row.id),
                    page_number=int(row.page_number),
                    score=float(score),
                    chunk_text=row.chunk_text,
                )
            )

    # Sort by score desc (faiss already does, but keep safe)
    results.sort(key=lambda r: r.score, reverse=True)
    logger.info(
        f"Retrieved {len(results)} chunks for doc={document_id} model={model_type} top_k={top_k}"
    )
    return results
