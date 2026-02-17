"""
Build data/eval_questions.json from the latest document in the DB (for running evaluation).
Use when you have uploaded at least one PDF and want to run scripts/evaluate.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from app.db.session import SessionLocal
from app.db import models


def main() -> None:
    out_path = _project_root / "data" / "eval_questions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        doc = db.query(models.Document).order_by(models.Document.created_at.desc()).first()
        if not doc:
            print("No documents in DB. Upload a PDF first (POST /documents/upload), then run this again.")
            sys.exit(1)

        chunks = (
            db.query(models.Chunk)
            .filter(models.Chunk.document_id == doc.id)
            .order_by(models.Chunk.page_number.asc(), models.Chunk.chunk_index.asc())
            .limit(10)
            .all()
        )
        if not chunks:
            print("No chunks for latest document. Re-upload or run reindex.")
            sys.exit(1)

        doc_id = str(doc.id)
        chunk_ids = [str(c.id) for c in chunks]
        pages = list({c.page_number for c in chunks})

        eval_data = [
            {
                "document_id": doc_id,
                "question": "ይህ ሰነድ ስለ ምን ነው?",
                "expected_chunk_ids": chunk_ids[:5],
                "expected_pages": pages[:5],
            },
            {
                "document_id": doc_id,
                "question": "በዚህ ሰነድ ውስጥ ዋና ዋና ርዕሶች ምንድን ናቸው?",
                "expected_pages": pages[:5],
            },
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        print(f"Created {out_path} (document_id={doc_id}, {len(chunks)} chunks). Run: python scripts/evaluate.py")
    finally:
        db.close()


if __name__ == "__main__":
    main()
