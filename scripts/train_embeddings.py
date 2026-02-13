from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sentence_transformers import SentenceTransformer, InputExample, losses  # noqa

from app.core.config import settings  # noqa
from app.db.session import SessionLocal  # noqa
from app.db import models  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output", type=str, default="models/embeddings_finetuned")
    p.add_argument("--limit", type=int, default=2000)
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        pairs = db.query(models.RetrievalPair).limit(args.limit).all()
        if not pairs:
            print("No retrieval_pairs found. Run scripts/build_dataset.py first.")
            return

        # Collect chunk texts needed
        chunk_ids = set()
        for p_ in pairs:
            chunk_ids.add(str(p_.positive_chunk_id))
            for n in (p_.hard_negative_chunk_ids or []):
                chunk_ids.add(str(n))

        chunks = db.query(models.Chunk).filter(models.Chunk.id.in_(list(chunk_ids))).all()
        by_id = {str(c.id): c.chunk_text for c in chunks}

        examples: List[InputExample] = []
        for p_ in pairs:
            pos = by_id.get(str(p_.positive_chunk_id))
            if not pos:
                continue
            neg_ids = p_.hard_negative_chunk_ids or []
            if not neg_ids:
                continue
            neg = by_id.get(str(neg_ids[0]))
            if not neg:
                continue
            examples.append(InputExample(texts=[p_.query, pos, neg]))

        if not examples:
            print("Not enough valid triplets found.")
            return

        print(f"Training triplets: {len(examples)}")

        model = SentenceTransformer(settings.EMBEDDING_MODEL_BASE, device=settings.DEVICE)

        train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.TripletLoss(model=model)

        warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.lr},
            show_progress_bar=True,
        )

        model.save(str(out_dir))
        print(f"Saved finetuned model to: {out_dir}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
