from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.db.session import SessionLocal  # noqa
from app.db import models  # noqa
from app.services.embeddings import embed_texts  # noqa
from app.services.faiss_index import load_index, search  # noqa


def make_query_from_chunk(text: str) -> str:
    # Very simple heuristic query generator (fast, no LLM needed).
    snippet = " ".join((text or "").strip().split())
    if len(snippet) > 120:
        snippet = snippet[:120].rstrip() + "..."
    return f"ይህ ክፍል ምን ይገልጻል? እባክህ ከዚህ ጽሑፍ መሠረት አስረዳ: {snippet}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--document-id", default=None, help="Only build pairs for one document_id")
    p.add_argument("--hard-negatives", type=int, default=3)
    p.add_argument("--max-pairs", type=int, default=500)
    p.add_argument("--out", default="data/retrieval_pairs.jsonl")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        q = db.query(models.Chunk)
        if args.document_id:
            q = q.filter(models.Chunk.document_id == args.document_id)
        chunks = q.order_by(models.Chunk.document_id.asc(), models.Chunk.page_number.asc(), models.Chunk.chunk_index.asc()).all()

        if not chunks:
            print("No chunks found. Upload a document first.")
            return

        # Group by document for negative mining
        by_doc = {}
        for c in chunks:
            by_doc.setdefault(str(c.document_id), []).append(c)

        written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for doc_id, doc_chunks in by_doc.items():
                # ensure index exists
                try:
                    handle = load_index(doc_id, "base")
                except Exception as e:
                    print(f"Skipping doc_id={doc_id} because base index is missing. Upload with embedding_model_type=base first.")
                    continue

                all_ids = [str(c.id) for c in doc_chunks]

                for c in doc_chunks:
                    if written >= args.max_pairs:
                        break

                    query = make_query_from_chunk(c.chunk_text)
                    _, qvecs = embed_texts([query], model_type="base", batch_size=1)
                    qvec = qvecs[0]

                    # Search for hard negatives
                    vec_ids, _scores = search(doc_id, "base", qvec, top_k=20)
                    hard = []
                    for vid in vec_ids:
                        if 0 <= vid < len(handle.vector_id_to_chunk_id):
                            cand = handle.vector_id_to_chunk_id[vid]
                            if cand != str(c.id) and cand not in hard:
                                hard.append(cand)
                        if len(hard) >= args.hard_negatives:
                            break

                    # Fallback: random negatives if needed
                    if len(hard) < args.hard_negatives:
                        pool = [x for x in all_ids if x != str(c.id) and x not in hard]
                        random.shuffle(pool)
                        hard.extend(pool[: (args.hard_negatives - len(hard))])

                    pair = models.RetrievalPair(
                        document_id=c.document_id,
                        query=query,
                        positive_chunk_id=c.id,
                        hard_negative_chunk_ids=hard,
                    )
                    db.add(pair)

                    f.write(json.dumps(
                        {
                            "document_id": str(c.document_id),
                            "query": query,
                            "positive_chunk_id": str(c.id),
                            "hard_negative_chunk_ids": hard,
                        },
                        ensure_ascii=False
                    ) + "\n")

                    written += 1

                db.commit()

        print(f"Wrote {written} retrieval pairs to {out_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
