from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db import models
from app.services.embeddings import EmbeddingModelType
from app.services.retrieval import retrieve_chunks, RetrievedChunk
from app.services.generator import generate_answer

router = APIRouter()


class RetrieveRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = Field(default_factory=lambda: settings.TOP_K_DEFAULT, ge=0, le=50)
    embedding_model_type: EmbeddingModelType = "base"


class RetrievedItem(BaseModel):
    chunk_id: str
    page_number: int
    score: float
    chunk_text: str


class RetrieveResponse(BaseModel):
    document_id: str
    embedding_model_type: EmbeddingModelType
    top_k: int
    results: List[RetrievedItem]


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, db: Session = Depends(get_db)):
    top_k = req.top_k if req.top_k > 0 else settings.TOP_K_DEFAULT
    try:
        chunks = retrieve_chunks(
            db=db,
            document_id=req.document_id,
            question=req.question,
            top_k=top_k,
            model_type=req.embedding_model_type,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RetrieveResponse(
        document_id=req.document_id,
        embedding_model_type=req.embedding_model_type,
        top_k=top_k,
        results=[RetrievedItem(**c.__dict__) for c in chunks],
    )


class AskRequest(BaseModel):
    question: str
    document_id: Optional[str] = Field(None, description="Optional. If omitted, the most recently uploaded document is used.")
    top_k: int = Field(default_factory=lambda: settings.TOP_K_DEFAULT, ge=0, le=50)
    embedding_model_type: EmbeddingModelType = "base"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "በሕጉ የሚጠበቀው የአስተዲዯር ውሳኔ የሚጀምረው በአዋጁ አንቀጽ 20፣ 21፣ እና 22 መሠረት ሲሆን ይኸውም",
                }
            ]
        }
    }


class Citation(BaseModel):
    page_number: int
    chunk_id: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    document_id: str


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    document_id = req.document_id
    if not document_id:
        latest = db.query(models.Document).order_by(models.Document.created_at.desc()).first()
        if not latest:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No documents in the database. "
                    "Upload a PDF with POST /documents/upload (see /docs). "
                    "If upload returned 500, fix the error (e.g. install sentencepiece, restart server). "
                    "List existing documents: GET /admin/documents."
                ),
            )
        document_id = str(latest.id)

    top_k = req.top_k if req.top_k > 0 else settings.TOP_K_DEFAULT
    try:
        retrieved = retrieve_chunks(
            db=db,
            document_id=document_id,
            question=req.question,
            top_k=top_k,
            model_type=req.embedding_model_type,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    contexts = [(c.page_number, c.chunk_text) for c in retrieved]
    try:
        answer = generate_answer(req.question, contexts)
    except Exception as e:
        logger.exception("RAG ask: generator failed")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {e!s}")

    citations: List[Citation] = []
    for c in retrieved:
        snippet = c.chunk_text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."
        citations.append(Citation(page_number=c.page_number, chunk_id=c.chunk_id, snippet=snippet))

    return AskResponse(answer=answer, citations=citations, document_id=document_id)
