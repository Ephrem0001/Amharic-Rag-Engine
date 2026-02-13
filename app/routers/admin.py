from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import models
from app.services.embeddings import embed_texts, EmbeddingModelType
from app.services.faiss_index import build_and_save_index
from app.services.evaluation import evaluate

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(models.Document).order_by(models.Document.created_at.desc()).all()
    return [{"document_id": str(d.id), "title": d.title, "created_at": d.created_at} for d in docs]


@router.post("/reindex/{document_id}")
def reindex_document(document_id: str, db: Session = Depends(get_db)):
    # Rebuild FAISS index using finetuned embedding model
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

    model_name, vectors = embed_texts(chunk_texts, model_type="finetuned", batch_size=32)

    for i, c in enumerate(chunks):
        c.vector_id = i
        c.embedding_model = model_name

    build_and_save_index(document_id, "finetuned", vectors, chunk_ids)
    db.commit()

    return {
        "document_id": document_id,
        "embedding_model_type": "finetuned",
        "embedding_model_name": model_name,
        "chunks": len(chunks),
        "status": "reindexed",
    }


class EvaluateRequest(BaseModel):
    run_name: str = Field(..., description="Any label for this evaluation run")
    embedding_model_type: EmbeddingModelType = "base"
    top_k: int = Field(5, ge=1, le=50)
    document_id: Optional[str] = None


@router.post("/evaluate")
def evaluate_run(req: EvaluateRequest, db: Session = Depends(get_db)):
    result = evaluate(db=db, model_type=req.embedding_model_type, top_k=req.top_k, document_id=req.document_id)

    run = models.EvalRun(
        run_name=req.run_name,
        embedding_model_name=req.embedding_model_type,
        embedding_model_path=str(req.embedding_model_type),
        top_k=req.top_k,
        recall_at_k=result.recall_at_k,
        mrr_at_k=result.mrr_at_k,
        ndcg_at_k=result.ndcg_at_k,
        hit_rate_at_k=result.hit_rate_at_k,
        mean_latency_ms=result.mean_latency_ms,
    )
    db.add(run)
    db.commit()

    return {
        "run_id": str(run.id),
        "run_name": req.run_name,
        "embedding_model_type": req.embedding_model_type,
        "top_k": req.top_k,
        "n_questions": result.n_questions,
        "metrics": {
            "hit_rate_at_k": result.hit_rate_at_k,
            "recall_at_k": result.recall_at_k,
            "mrr_at_k": result.mrr_at_k,
            "ndcg_at_k": result.ndcg_at_k,
            "mean_latency_ms": result.mean_latency_ms,
        },
    }
