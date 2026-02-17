from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import models
from app.services.embeddings import embed_texts, EmbeddingModelType
from app.services.faiss_index import build_and_save_index
from app.services.evaluation_runner import load_eval_questions, run_evaluation, persist_eval_runs

router = APIRouter()

# Project root for resolving relative eval file paths
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EvaluateRequest(BaseModel):
    """Optional body for POST /admin/evaluate."""

    eval_file: str | None = None  # path to JSON/JSONL (relative to project root or absolute)
    top_k: int = 10
    skip_finetuned: bool = False
    store_results: bool = True


@router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(models.Document).order_by(models.Document.created_at.desc()).all()
    return [{"document_id": str(d.id), "title": d.title, "created_at": d.created_at} for d in docs]


@router.post("/reindex/{document_id}")
def reindex_document(
    document_id: str,
    embedding_model_type: EmbeddingModelType = Query("base", description="base | medium | finetuned"),
    db: Session = Depends(get_db),
):
    """Rebuild FAISS index for a document (e.g. after changing embedding model)."""
    chunks = (
        db.query(models.Chunk)
        .filter(models.Chunk.document_id == document_id)
        .order_by(models.Chunk.page_number.asc(), models.Chunk.chunk_index.asc())
        .all()
    )
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for this document_id.")

    chunk_texts = [c.chunk_text for c in chunks]
    chunk_ids = [str(c.id) for c in chunks]

    model_name, vectors = embed_texts(chunk_texts, model_type=embedding_model_type, batch_size=32)

    for i, c in enumerate(chunks):
        c.vector_id = i
        c.embedding_model = model_name

    build_and_save_index(document_id, embedding_model_type, vectors, chunk_ids)
    db.commit()

    return {
        "document_id": document_id,
        "embedding_model_type": embedding_model_type,
        "embedding_model_name": model_name,
        "chunks": len(chunks),
        "status": "reindexed",
    }


@router.post("/evaluate")
def evaluate_retrieval(
    body: EvaluateRequest | None = None,
    db: Session = Depends(get_db),
):
    """
    Run retrieval evaluation (base and optionally finetuned), return metrics.
    Optionally store one row per model in eval_runs (run_name, embedding_model_type, top_k, recall_at_k, etc.).
    """
    req = body or EvaluateRequest()
    eval_file = req.eval_file or "data/eval_questions.json"
    path = Path(eval_file)
    if not path.is_absolute():
        path = _PROJECT_ROOT / eval_file
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Eval file not found: {path}")

    try:
        eval_questions = load_eval_questions(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load eval file: {e}") from e

    if not eval_questions:
        raise HTTPException(status_code=400, detail="No eval questions in file.")

    top_k = max(10, req.top_k)
    results = run_evaluation(db, eval_questions, top_k=top_k, skip_finetuned=req.skip_finetuned)
    results["eval_file"] = str(path)
    results["num_questions"] = len(eval_questions)

    if results.get("num_questions", 0) == 0:
        raise HTTPException(status_code=400, detail="No valid questions with ground truth.")

    run_name = None
    if req.store_results:
        run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        persist_eval_runs(db, run_name, results, eval_file=str(path), top_k=top_k)

    results.pop("_failure_data", None)  # internal use only; do not return
    return {"run_name": run_name, **results}
