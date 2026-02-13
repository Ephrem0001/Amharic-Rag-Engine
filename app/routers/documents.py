from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db import models
from app.services.pdf_extract import extract_pdf_pages
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingModelType, embed_texts
from app.services.faiss_index import build_and_save_index

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    embedding_model_type: EmbeddingModelType = Query("base", description="base | medium | finetuned"),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4()
    safe_name = Path(file.filename).name
    save_path = settings.UPLOAD_DIR / f"{doc_id}__{safe_name}"

    # Save to disk
    with save_path.open("wb") as f:
        f.write(file.file.read())

    # Create Document row
    doc = models.Document(id=doc_id, title=safe_name, file_path=str(save_path))
    db.add(doc)
    db.flush()

    pages = extract_pdf_pages(save_path)
    if not pages or all(not p.text for p in pages):
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF. If it's scanned images, OCR is required (not included).",
        )

    # Insert pages
    page_rows = [
        models.Page(document_id=doc_id, page_number=p.page_number, text=p.text)
        for p in pages
        if p.text.strip()
    ]
    db.add_all(page_rows)
    db.flush()

    # Chunk pages and insert chunks
    chunk_rows: list[models.Chunk] = []
    chunk_texts: list[str] = []
    for p in pages:
        if not p.text.strip():
            continue
        chunks = chunk_text(p.text, target_chars=settings.CHUNK_TARGET_CHARS, overlap_ratio=settings.CHUNK_OVERLAP_RATIO)
        for c in chunks:
            row = models.Chunk(
                document_id=doc_id,
                page_number=p.page_number,
                chunk_index=c.chunk_index,
                chunk_text=c.chunk_text,
            )
            chunk_rows.append(row)
            chunk_texts.append(c.chunk_text)

    if not chunk_rows:
        raise HTTPException(status_code=400, detail="No chunks created from PDF text.")

    db.add_all(chunk_rows)
    db.flush()  # assigns IDs

    chunk_ids = [str(r.id) for r in chunk_rows]

    # Embed and build index
    model_name, vectors = embed_texts(chunk_texts, model_type=embedding_model_type, batch_size=32)

    # Update vector_id + embedding_model for bookkeeping
    for i, r in enumerate(chunk_rows):
        r.vector_id = i
        r.embedding_model = model_name

    build_and_save_index(str(doc_id), embedding_model_type, vectors, chunk_ids)

    db.commit()

    logger.info(f"Uploaded document {doc_id} pages={len(pages)} chunks={len(chunk_rows)} model={embedding_model_type}")

    return {
        "document_id": str(doc_id),
        "title": safe_name,
        "pages": len(pages),
        "chunks": len(chunk_rows),
        "embedding_model_type": embedding_model_type,
        "embedding_model_name": model_name,
    }
