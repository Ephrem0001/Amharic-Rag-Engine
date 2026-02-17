"""
Reindex all documents (or a subset) with the finetuned embedding model.

Use after fine-tuning (train_embeddings.py) so that FAISS indices exist for
embedding_model_type=finetuned. Then retrieval and /rag/ask can use the
finetuned model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.core.config import settings
from app.db.session import SessionLocal
from app.db import models
from app.services.embeddings import embed_texts, EmbeddingModelType
from app.services.faiss_index import build_and_save_index


def main() -> None:
    p = argparse.ArgumentParser(description="Reindex documents with finetuned embedding model")
    p.add_argument("--document-id", type=str, default=None, help="Reindex only this document (UUID); default all")
    p.add_argument("--dry-run", action="store_true", help="Only list documents that would be reindexed")
    args = p.parse_args()

    finetuned_path = Path(settings.EMBEDDING_MODEL_FINETUNED)
    if not finetuned_path.exists():
        print(f"Finetuned model not found at {finetuned_path}. Run scripts/train_embeddings.py first.")
        sys.exit(1)

    db = SessionLocal()
    try:
        q = db.query(models.Document)
        if args.document_id:
            q = q.filter(models.Document.id == args.document_id)
        docs = q.all()

        if not docs:
            print("No documents found.")
            sys.exit(0)

        if args.dry_run:
            for d in docs:
                print(str(d.id))
            print(f"Would reindex {len(docs)} document(s). Run without --dry-run to reindex.")
            return

        for d in docs:
            doc_id = str(d.id)
            chunks = (
                db.query(models.Chunk)
                .filter(models.Chunk.document_id == doc_id)
                .order_by(models.Chunk.page_number.asc(), models.Chunk.chunk_index.asc())
                .all()
            )
            if not chunks:
                print(f"  {doc_id}: no chunks, skip")
                continue
            chunk_texts = [c.chunk_text for c in chunks]
            chunk_ids = [str(c.id) for c in chunks]
            model_name, vectors = embed_texts(chunk_texts, model_type=EmbeddingModelType.FINETUNED, batch_size=32)
            for i, c in enumerate(chunks):
                c.vector_id = i
                c.embedding_model = model_name
            build_and_save_index(doc_id, EmbeddingModelType.FINETUNED.value, vectors, chunk_ids)
            db.commit()
            print(f"  {doc_id}: reindexed {len(chunks)} chunks")

        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
