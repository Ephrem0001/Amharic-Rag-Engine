from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db import models
from app.services.embeddings import EmbeddingModelType
from app.services.retrieval import retrieve_chunks


def _hit_rate(rels: List[int]) -> float:
    return 1.0 if any(rels) else 0.0


def _recall_at_k(rels: List[int], total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return float(sum(rels)) / float(total_relevant)


def _mrr(rels: List[int]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / float(i)
    return 0.0


def _ndcg(rels: List[int]) -> float:
    # binary gain
    dcg = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            dcg += 1.0 / math.log2(i + 1.0)
    # ideal is all ones at the top
    ideal = sorted(rels, reverse=True)
    idcg = 0.0
    for i, r in enumerate(ideal, start=1):
        if r:
            idcg += 1.0 / math.log2(i + 1.0)
    return (dcg / idcg) if idcg > 0 else 0.0


@dataclass
class EvalResult:
    hit_rate_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    mean_latency_ms: float
    n_questions: int


def evaluate(
    db: Session,
    model_type: EmbeddingModelType,
    top_k: int,
    document_id: Optional[str] = None,
) -> EvalResult:
    q = db.query(models.EvalQuestion)
    if document_id:
        q = q.filter(models.EvalQuestion.document_id == document_id)
    questions = q.all()

    if not questions:
        return EvalResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    hits, recalls, mrrs, ndcgs = [], [], [], []
    latencies = []

    for item in questions:
        doc_id = str(item.document_id) if item.document_id else None
        if not doc_id:
            # Skip rows that don't specify document_id (cannot evaluate retrieval)
            continue

        expected_pages = set(item.expected_pages or [])
        expected_chunk_ids = set(item.expected_chunk_ids or [])

        t0 = time.perf_counter()
        retrieved = retrieve_chunks(db, doc_id, item.question, top_k=top_k, model_type=model_type)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        rels: List[int] = []
        for r in retrieved:
            is_rel = False
            if expected_chunk_ids:
                is_rel = r.chunk_id in expected_chunk_ids
            elif expected_pages:
                is_rel = r.page_number in expected_pages
            rels.append(1 if is_rel else 0)

        total_relevant = len(expected_chunk_ids) if expected_chunk_ids else len(expected_pages)

        hits.append(_hit_rate(rels))
        recalls.append(_recall_at_k(rels, total_relevant))
        mrrs.append(_mrr(rels))
        ndcgs.append(_ndcg(rels))

    if not hits:
        return EvalResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    return EvalResult(
        hit_rate_at_k=sum(hits) / len(hits),
        recall_at_k=sum(recalls) / len(recalls),
        mrr_at_k=sum(mrrs) / len(mrrs),
        ndcg_at_k=sum(ndcgs) / len(ndcgs),
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        n_questions=len(hits),
    )
