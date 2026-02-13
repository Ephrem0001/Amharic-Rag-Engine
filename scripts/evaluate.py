from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.db.session import SessionLocal  # noqa
from app.services.evaluation import evaluate  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/eval.md")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.utcnow().strftime("eval_%Y%m%d_%H%M%S")

    db = SessionLocal()
    try:
        rows = []
        for model_type in ("base", "finetuned"):
            for k in (5, 10):
                res = evaluate(db=db, model_type=model_type, top_k=k)
                rows.append((model_type, k, res.n_questions, res.hit_rate_at_k, res.recall_at_k, res.mrr_at_k, res.ndcg_at_k, res.mean_latency_ms))

        md = []
        md.append(f"# Evaluation Report: {run_name}\n")
        md.append("| model | K | n | HitRate@K | Recall@K | MRR@K | nDCG@K | mean latency (ms) |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.1f} |")
        md.append("")
        out_path.write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote report to {out_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
