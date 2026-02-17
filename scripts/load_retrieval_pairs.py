"""
Load retrieval pairs from a JSONL file into the database for fine-tuning.

Each line must be a JSON object with:
  - document_id (UUID string)
  - query (string)
  - positive_chunk_id (UUID string)
  - hard_negative_chunk_ids (list of UUID strings, optional)

Use this when you have data/retrieval_pairs.jsonl from another source, or after
copying/generating pairs externally. Then run scripts/train_embeddings.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.db.session import SessionLocal
from app.db import models


def main() -> None:
    p = argparse.ArgumentParser(description="Load retrieval pairs from JSONL into DB for fine-tuning")
    p.add_argument("--input", "-i", type=str, default="data/retrieval_pairs.jsonl", help="Input JSONL path")
    p.add_argument("--limit", type=int, default=0, help="Max lines to load (0 = all)")
    p.add_argument("--skip-invalid", action="store_true", help="Skip rows with missing/invalid chunk IDs")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    db = SessionLocal()
    loaded = 0
    skipped = 0

    try:
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.limit and loaded >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Line {i + 1}: invalid JSON: {e}")
                    skipped += 1
                    continue

                doc_id_s = row.get("document_id")
                query = row.get("query")
                pos_id_s = row.get("positive_chunk_id")
                neg_ids = row.get("hard_negative_chunk_ids")
                if not doc_id_s or not query or not pos_id_s:
                    if args.skip_invalid:
                        skipped += 1
                        continue
                    print(f"Line {i + 1}: missing document_id, query, or positive_chunk_id")
                    skipped += 1
                    continue

                try:
                    doc_id = UUID(doc_id_s)
                    pos_id = UUID(pos_id_s)
                except (ValueError, TypeError):
                    if args.skip_invalid:
                        skipped += 1
                        continue
                    print(f"Line {i + 1}: invalid UUID(s)")
                    skipped += 1
                    continue

                neg_uuids = []
                if isinstance(neg_ids, list):
                    for n in neg_ids:
                        try:
                            neg_uuids.append(str(UUID(n)))
                        except (ValueError, TypeError):
                            pass

                # Optionally verify chunks exist
                if args.skip_invalid:
                    pos_exists = db.query(models.Chunk).filter(models.Chunk.id == pos_id).first()
                    if not pos_exists:
                        skipped += 1
                        continue

                pair = models.RetrievalPair(
                    document_id=doc_id,
                    query=query,
                    positive_chunk_id=pos_id,
                    hard_negative_chunk_ids=neg_uuids if neg_uuids else None,
                )
                db.add(pair)
                loaded += 1

        db.commit()
        print(f"Loaded {loaded} retrieval pairs from {path}" + (f" (skipped {skipped})" if skipped else ""))
    finally:
        db.close()


if __name__ == "__main__":
    main()
