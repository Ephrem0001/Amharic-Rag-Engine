from fastapi import FastAPI
from loguru import logger

from app.routers import documents, rag, admin

app = FastAPI(title="Amharic RAG")


@app.on_event("startup")
def startup_checks():
    """Warn if embedding backend is missing."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        logger.info("Embedding backend OK (sentence-transformers)")
    except ImportError:
        logger.error(
            "sentence-transformers is NOT installed. Upload and /rag/ask will fail. "
            "Run: pip install sentence-transformers   then restart the server."
        )


@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(rag.router, prefix="/rag", tags=["rag"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
