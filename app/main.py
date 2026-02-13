from __future__ import annotations

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging
from app.routers import documents, rag, admin


setup_logging()

app = FastAPI(title="Amharic RAG (No Docker)")


@app.on_event("startup")
def _startup():
    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(documents.router)
app.include_router(rag.router)
app.include_router(admin.router)
