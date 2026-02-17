"""
Fine-tune the base Amharic embedding model on (query, positive, negative) triplets.

Data source: either retrieval_pairs in the DB (from build_dataset.py or
load_retrieval_pairs.py) or a JSONL file (--data) with document_id, query,
positive_chunk_id, hard_negative_chunk_ids. Chunk text is always resolved from the DB.

Output: saved model directory (default from EMBEDDING_MODEL_FINETUNED). After
training, run reindex for each document with embedding_model_type=finetuned, or
use scripts/reindex_finetuned.py to reindex all.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sentence_transformers import SentenceTransformer, InputExample, losses  # noqa

from app.core.config import settings  # noqa
from app.db.session import SessionLocal  # noqa
from app.db import models  # noqa

# Reproducibility: fixed seeds for training (spec)
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pairs_from_jsonl(path: Path, limit: int) -> List[dict]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit and len(pairs) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def main() -> None:
    set_seeds()

    p = argparse.ArgumentParser(description="Fine-tune embedding model on retrieval triplets")
    p.add_argument("--epochs", type=int, default=1, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--output", "-o", type=str, default=None, help="Output model dir (default: EMBEDDING_MODEL_FINETUNED)")
    p.add_argument("--limit", type=int, default=0, help="Max triplets (0 = all)")
    p.add_argument("--data", type=str, default=None, help="JSONL path instead of DB (chunk text still from DB)")
    args = p.parse_args()

    out_dir = Path(args.output or str(settings.EMBEDDING_MODEL_FINETUNED))
    out_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        if args.data:
            path = Path(args.data)
            if not path.exists():
                print(f"File not found: {path}")
                sys.exit(1)
            raw = load_pairs_from_jsonl(path, args.limit)
            # Normalize to same shape as DB rows: query, positive_chunk_id, hard_negative_chunk_ids
            pair_like = [
                {
                    "query": r.get("query", ""),
                    "positive_chunk_id": r.get("positive_chunk_id"),
                    "hard_negative_chunk_ids": r.get("hard_negative_chunk_ids") or [],
                }
                for r in raw
                if r.get("query") and r.get("positive_chunk_id")
            ]
            if args.limit:
                pair_like = pair_like[: args.limit]
        else:
            q = db.query(models.RetrievalPair)
            if args.limit:
                q = q.limit(args.limit)
            pairs_db = q.all()
            pair_like = [
                {
                    "query": p_.query,
                    "positive_chunk_id": str(p_.positive_chunk_id),
                    "hard_negative_chunk_ids": [str(x) for x in (p_.hard_negative_chunk_ids or [])],
                }
                for p_ in pairs_db
            ]

        if not pair_like:
            print("No retrieval pairs found. Run scripts/build_dataset.py or load_retrieval_pairs.py, or use --data <jsonl>.")
            sys.exit(1)

        chunk_ids = set()
        for p_ in pair_like:
            chunk_ids.add(p_["positive_chunk_id"])
            chunk_ids.update(p_["hard_negative_chunk_ids"])

        chunks = db.query(models.Chunk).filter(models.Chunk.id.in_(list(chunk_ids))).all()
        by_id = {str(c.id): c.chunk_text for c in chunks}

        examples: List[InputExample] = []
        for p_ in pair_like:
            pos = by_id.get(p_["positive_chunk_id"])
            if not pos:
                continue
            neg_ids = p_["hard_negative_chunk_ids"]
            if not neg_ids:
                continue
            neg = by_id.get(neg_ids[0])
            if not neg:
                continue
            examples.append(InputExample(texts=[p_["query"], pos, neg]))

        if not examples:
            print("Not enough valid triplets (missing chunk text in DB?). Ensure chunks exist for all pair chunk IDs.")
            sys.exit(1)

        print(f"Training on {len(examples)} triplets (base model: {settings.EMBEDDING_MODEL_BASE})")

        model = SentenceTransformer(settings.EMBEDDING_MODEL_BASE, device=settings.DEVICE)

        train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.TripletLoss(model=model)

        warmup_steps = max(0, int(len(train_dataloader) * args.epochs * 0.1))

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.lr},
            show_progress_bar=True,
        )

        model.save(str(out_dir))
        print(f"Saved finetuned model to: {out_dir}")
        print("Next: reindex documents with finetuned model (e.g. scripts/reindex_finetuned.py or POST /admin/reindex/{id}?embedding_model_type=finetuned)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
