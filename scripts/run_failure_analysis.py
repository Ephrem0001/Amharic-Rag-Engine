"""
Run failure analysis: for each eval question, run retrieval, compare retrieved vs expected,
and write a failure analysis report (reports/failure_analysis_<timestamp>.md).

Usage:
  python scripts/run_failure_analysis.py --eval-file data/eval_questions.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from loguru import logger

from app.db.session import SessionLocal
from app.services.retrieval import retrieve_chunks
from app.services.embeddings import EmbeddingModelType
from app.services.evaluation_runner import load_eval_questions, resolve_relevant_chunk_ids


def short_id(chunk_id: str) -> str:
    """Display-friendly chunk id (first 8 chars)."""
    return chunk_id.replace("-", "")[:8] if chunk_id else "?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval failure analysis")
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval_questions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    eval_path = _project_root / args.eval_file if not args.eval_file.is_absolute() else args.eval_file
    eval_questions = load_eval_questions(eval_path)
    if not eval_questions:
        logger.error("No eval questions loaded.")
        sys.exit(1)

    out_dir = _project_root / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = max(args.top_k, 5)

    db = SessionLocal()
    failures: list[dict] = []  # list of {question, expected, retrieved, reason, fix}

    try:
        for eq in eval_questions:
            doc_id = eq.get("document_id") and str(eq["document_id"]).strip()
            question = eq.get("question")
            if not doc_id or not question:
                continue
            relevant = resolve_relevant_chunk_ids(db, doc_id, eq.get("expected_pages"), eq.get("expected_chunk_ids"))
            if not relevant:
                continue

            try:
                chunks = retrieve_chunks(db, doc_id, question, top_k=top_k, model_type=EmbeddingModelType.BASE)
            except FileNotFoundError:
                logger.warning(f"Index not found for doc={doc_id}, skipping.")
                continue

            retrieved_ids = [c.chunk_id for c in chunks]
            retrieved_set = set(retrieved_ids)
            missed = relevant - retrieved_set

            if missed:
                # At least one expected chunk was not in top-k → failure
                expected_display = ", ".join(short_id(c) for c in list(missed)[:3])
                retrieved_display = ", ".join(short_id(c) for c in retrieved_ids[:5])
                rank_of_first_hit = None
                for i, cid in enumerate(retrieved_ids, 1):
                    if cid in relevant:
                        rank_of_first_hit = i
                        break
                if rank_of_first_hit and rank_of_first_hit > 1:
                    reason = f"Expected chunk(s) not in top-{top_k}, or ranked at position {rank_of_first_hit} (semantic similarity)."
                else:
                    reason = f"Expected chunk(s) not in top-{top_k} (semantic similarity / vocabulary gap)."
                failures.append({
                    "question": question,
                    "expected": expected_display,
                    "retrieved": retrieved_display,
                    "reason": reason,
                    "fix": "Add fine-tuning data for this query–chunk pair; or increase top_k / chunk overlap.",
                })
    finally:
        db.close()

    # Write report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"failure_analysis_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Failure Analysis Report — Amharic RAG Retrieval\n\n")
        f.write("Expected vs retrieved chunks for questions where at least one relevant chunk was **not** in top-k.\n\n")
        f.write(f"- **Eval file:** {eval_path}\n")
        f.write(f"- **Top-K:** {top_k}\n")
        f.write(f"- **Failures:** {len(failures)}\n\n---\n\n")
        for i, fail in enumerate(failures, 1):
            f.write(f"## Failure {i}\n\n")
            f.write(f"**Question:**  \n{fail['question']}\n\n")
            f.write(f"**Expected (missed):**  \n{fail['expected']}\n\n")
            f.write(f"**Retrieved (top-5):**  \n{fail['retrieved']}\n\n")
            f.write(f"**Reason:**  \n{fail['reason']}\n\n")
            f.write(f"**Fix:**  \n{fail['fix']}\n\n---\n\n")
        if not failures:
            f.write("No failures: all expected chunks appeared in the retrieved top-k.\n")
    logger.info(f"Failure analysis saved to {report_path} ({len(failures)} failures).")


if __name__ == "__main__":
    main()
