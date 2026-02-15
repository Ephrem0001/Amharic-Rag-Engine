from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import models
from app.services.embeddings import embed_texts, EmbeddingModelType
from app.services.faiss_index import build_and_save_index

router = APIRouter()


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
