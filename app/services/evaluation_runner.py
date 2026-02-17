"""
Shared evaluation logic: load questions, run retrieval per model, compute metrics, persist to eval_runs.

Used by scripts/evaluate.py and POST /admin/evaluate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db import models
from app.services.embeddings import EmbeddingModelType
from app.services.evaluation_metrics import compute_all_metrics
from app.services.retrieval import retrieve_chunks


def load_eval_questions(path: Path) -> list[dict]:
    """Load eval questions from JSON or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")
    with path.open(encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def resolve_relevant_chunk_ids(
    db: Session,
    document_id: str,
    expected_pages: list[int] | None,
    expected_chunk_ids: list[str] | None,
) -> set[str]:
    """Return set of relevant chunk IDs from expected_chunk_ids or by resolving expected_pages."""
    if expected_chunk_ids:
        return {str(cid).strip() for cid in expected_chunk_ids}
    if expected_pages:
        rows = (
            db.query(models.Chunk.id)
            .filter(
                models.Chunk.document_id == document_id,
                models.Chunk.page_number.in_(expected_pages),
            )
            .all()
        )
        return {str(r.id) for r in rows}
    return set()


def run_retrieval_for_model(
    db: Session,
    eval_questions: list[dict],
    model_type: EmbeddingModelType,
    top_k: int,
) -> tuple[list[list[str]], list[set[str]], list[float]]:
    """
    Run retrieval for each question. Returns (list_retrieved_ids, list_relevant_ids, latencies_ms).
    Skips questions where document_id or relevant set is missing.
    """
    list_retrieved: list[list[str]] = []
    list_relevant: list[set[str]] = []
    latencies_ms: list[float] = []

    for eq in eval_questions:
        document_id = eq.get("document_id")
        question = eq.get("question")
        if not document_id or not question:
            continue
        doc_id_str = str(document_id).strip()
        relevant = resolve_relevant_chunk_ids(
            db,
            doc_id_str,
            eq.get("expected_pages"),
            eq.get("expected_chunk_ids"),
        )
        if not relevant:
            continue
        try:
            t0 = time.perf_counter()
            chunks = retrieve_chunks(db, doc_id_str, question, top_k=top_k, model_type=model_type)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except FileNotFoundError:
            raise
        retrieved_ids = [c.chunk_id for c in chunks]
        list_retrieved.append(retrieved_ids)
        list_relevant.append(relevant)
        latencies_ms.append(elapsed_ms)

    return list_retrieved, list_relevant, latencies_ms


def run_evaluation(
    db: Session,
    eval_questions: list[dict],
    top_k: int = 10,
    skip_finetuned: bool = False,
) -> dict[str, Any]:
    """
    Run retrieval evaluation for base (and optionally finetuned) model.
    Returns dict with eval_file, top_k, num_questions, models: { base: {...}, finetuned: {...} }.
    """
    k_values = [5, 10]
    valid = []
    for eq in eval_questions:
        doc_id = eq.get("document_id") and str(eq["document_id"]).strip()
        q = eq.get("question")
        if not doc_id or not q:
            continue
        rel = resolve_relevant_chunk_ids(db, doc_id, eq.get("expected_pages"), eq.get("expected_chunk_ids"))
        if rel:
            valid.append(eq)
    if not valid:
        return {"eval_file": "", "top_k": top_k, "num_questions": 0, "models": {}}

    results: dict = {"eval_file": "", "top_k": top_k, "num_questions": len(valid), "models": {}}
    for model_type in [EmbeddingModelType.BASE, EmbeddingModelType.FINETUNED]:
        if model_type == EmbeddingModelType.FINETUNED and skip_finetuned:
            results["models"]["finetuned"] = {"skipped": True, "reason": "skip-finetuned"}
            continue
        name = model_type.value
        try:
            list_retrieved, list_relevant, latencies_ms = run_retrieval_for_model(db, valid, model_type, top_k)
        except FileNotFoundError:
            results["models"][name] = {"skipped": True, "reason": "index_not_found"}
            continue
        metrics = compute_all_metrics(list_retrieved, list_relevant, k_values)
        mean_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        results["models"][name] = {
            **metrics,
            "mean_latency_ms": round(mean_latency_ms, 2),
            "num_queries": len(list_retrieved),
        }
        # Attach base-model retrieval data for failure analysis (spec: at least 5 failure cases in report)
        if model_type == EmbeddingModelType.BASE:
            results["_failure_data"] = {
                "list_retrieved": list_retrieved,
                "list_relevant": list_relevant,
                "valid_questions": valid,
            }
    return results


def persist_eval_runs(
    db: Session,
    run_name: str,
    results: dict,
    eval_file: str = "",
    top_k: int = 10,
    k_for_stored: int = 10,
) -> None:
    """
    Insert one row per model into eval_runs (run_name, embedding_model_type, top_k, recall_at_k, ...).
    Uses metrics at k=k_for_stored for the stored columns.
    """
    for model_name in ("base", "finetuned"):
        m = results.get("models", {}).get(model_name)
        if not m or m.get("skipped"):
            continue
        recall = m.get(f"recall_at_{k_for_stored}")
        mrr = m.get(f"mrr_at_{k_for_stored}")
        ndcg = m.get(f"ndcg_at_{k_for_stored}")
        hit = m.get(f"hit_rate_at_{k_for_stored}")
        latency = m.get("mean_latency_ms")
        run = models.EvalRun(
            run_name=run_name,
            embedding_model_type=model_name,
            embedding_model_name=None,
            embedding_model_path=None,
            top_k=top_k,
            recall_at_k=recall,
            mrr_at_k=mrr,
            ndcg_at_k=ndcg,
            hit_rate_at_k=hit,
            mean_latency_ms=latency,
        )
        db.add(run)
    db.commit()
