from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.services.embeddings import EmbeddingModelType
from app.services.retrieval import retrieve_chunks, RetrievedChunk
from app.services.generator import generate_answer

router = APIRouter(prefix="/rag", tags=["rag"])


class RetrieveRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = Field(default_factory=lambda: settings.TOP_K_DEFAULT, ge=1, le=50)
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
    try:
        chunks = retrieve_chunks(
            db=db,
            document_id=req.document_id,
            question=req.question,
            top_k=req.top_k,
            model_type=req.embedding_model_type,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RetrieveResponse(
        document_id=req.document_id,
        embedding_model_type=req.embedding_model_type,
        top_k=req.top_k,
        results=[RetrievedItem(**c.__dict__) for c in chunks],
    )


class AskRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = Field(default_factory=lambda: settings.TOP_K_DEFAULT, ge=1, le=50)
    embedding_model_type: EmbeddingModelType = "base"


class Citation(BaseModel):
    page_number: int
    chunk_id: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    try:
        retrieved = retrieve_chunks(
            db=db,
            document_id=req.document_id,
            question=req.question,
            top_k=req.top_k,
            model_type=req.embedding_model_type,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    contexts = [(c.page_number, c.chunk_text) for c in retrieved]
    answer = generate_answer(req.question, contexts)

    citations: List[Citation] = []
    for c in retrieved:
        snippet = c.chunk_text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."
        citations.append(Citation(page_number=c.page_number, chunk_id=c.chunk_id, snippet=snippet))

    return AskResponse(answer=answer, citations=citations)
