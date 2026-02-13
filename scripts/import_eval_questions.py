from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.db.session import SessionLocal  # noqa
from app.db import models  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Path to eval_questions.jsonl")
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    db = SessionLocal()
    try:
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = models.EvalQuestion(
                    document_id=obj.get("document_id"),
                    question=obj["question"],
                    expected_pages=obj.get("expected_pages"),
                    expected_chunk_ids=obj.get("expected_chunk_ids"),
                )
                db.add(q)
                n += 1
        db.commit()
        print(f"Imported {n} eval questions from {path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
