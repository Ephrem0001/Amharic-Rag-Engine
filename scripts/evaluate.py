"""
Evaluation script: load eval questions, run retrieval for base and finetuned models,
compute Recall@5/10, MRR@5/10, nDCG@5/10, Hit Rate, latency; save results and generate report.

Usage:
  python scripts/evaluate.py --eval-file data/eval_questions.json
  python scripts/evaluate.py --eval-file data/eval_questions.json --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Project root on path and .env loaded
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

# Load .env before importing app (config reads DATABASE_URL)
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from sqlalchemy.orm import Session
from loguru import logger

from app.core.config import settings
from app.db.session import SessionLocal
from app.db import models
from app.services.retrieval import retrieve_chunks
from app.services.embeddings import EmbeddingModelType
from app.services.evaluation_metrics import compute_all_metrics


def load_eval_questions(path: Path) -> list[dict]:
    """Load eval questions from JSON or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    with open(path, encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    if not isinstance(data, list):
        data = [data]
    return data


def resolve_relevant_chunk_ids(db: Session, document_id: str, expected_pages: list[int] | None, expected_chunk_ids: list[str] | None) -> set[str]:
    """Return set of relevant chunk IDs from expected_chunk_ids or by resolving expected_pages."""
    if expected_chunk_ids:
        return {str(cid).strip() for cid in expected_chunk_ids}
    if expected_pages:
        rows = (
            db.query(models.Chunk.id)
            .filter(
                models.Chunk.document_id == document_id,
                models.Chunk.page_number.in_(expected_pages),
            )
            .all()
        )
        return {str(r.id) for r in rows}
    return set()


def run_retrieval_for_model(
    db: Session,
    eval_questions: list[dict],
    model_type: EmbeddingModelType,
    top_k: int,
) -> tuple[list[list[str]], list[set[str]], list[float]]:
    """
    Run retrieval for each question. Returns (list_retrieved_ids, list_relevant_ids, latencies_ms).
    Skips questions where document_id or relevant set is missing; logs and continues.
    """
    list_retrieved: list[list[str]] = []
    list_relevant: list[set[str]] = []
    latencies_ms: list[float] = []

    for eq in eval_questions:
        document_id = eq.get("document_id")
        question = eq.get("question")
        if not document_id or not question:
            logger.warning("Skipping eval question: missing document_id or question")
            continue

        doc_id_str = str(document_id).strip()
        relevant = resolve_relevant_chunk_ids(
            db,
            doc_id_str,
            eq.get("expected_pages"),
            eq.get("expected_chunk_ids"),
        )
        if not relevant:
            logger.warning(f"Skipping question (no ground truth): {question[:50]}...")
            continue

        try:
            t0 = datetime.now().timestamp()
            chunks = retrieve_chunks(db, doc_id_str, question, top_k=top_k, model_type=model_type)
            elapsed_ms = (datetime.now().timestamp() - t0) * 1000
        except FileNotFoundError as e:
            logger.warning(f"Index missing for {model_type} doc={doc_id_str}: {e}")
            raise

        retrieved_ids = [c.chunk_id for c in chunks]
        list_retrieved.append(retrieved_ids)
        list_relevant.append(relevant)
        latencies_ms.append(elapsed_ms)

    return list_retrieved, list_relevant, latencies_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval: base vs finetuned")
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval_questions.json"), help="JSON or JSONL with eval questions")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Directory for results and report")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top_k (used for @5 and @10 metrics)")
    parser.add_argument("--skip-finetuned", action="store_true", help="Only evaluate base model (e.g. when finetuned index missing)")
    args = parser.parse_args()

    eval_path = args.eval_file
    if not eval_path.is_absolute():
        eval_path = _project_root / eval_path

    eval_questions = load_eval_questions(eval_path)
    if not eval_questions:
        logger.error("No eval questions loaded. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded {len(eval_questions)} eval questions from {eval_path}")

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = _project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = max(args.top_k, 10)
    k_values = [5, 10]
    results: dict = {"eval_file": str(eval_path), "top_k": top_k, "num_questions": len(eval_questions), "models": {}}

    db: Session = SessionLocal()
    try:
        # Resolve relevant sets once (same for both models)
        valid_questions = []
        for eq in eval_questions:
            doc_id = eq.get("document_id") and str(eq["document_id"]).strip()
            q = eq.get("question")
            if not doc_id or not q:
                continue
            rel = resolve_relevant_chunk_ids(db, doc_id, eq.get("expected_pages"), eq.get("expected_chunk_ids"))
            if rel:
                valid_questions.append((eq, rel))
        if not valid_questions:
            logger.error("No valid questions with ground truth. Exiting.")
            sys.exit(1)
        logger.info(f"Valid questions with ground truth: {len(valid_questions)}")

        eval_dicts = [eq for eq, _ in valid_questions]
        for model_type in [EmbeddingModelType.BASE, EmbeddingModelType.FINETUNED]:
            if model_type == EmbeddingModelType.FINETUNED and args.skip_finetuned:
                results["models"]["finetuned"] = {"skipped": True, "reason": "skip-finetuned"}
                continue
            name = model_type.value
            try:
                list_retrieved, list_relevant, latencies_ms = run_retrieval_for_model(
                    db, eval_dicts, model_type, top_k
                )
            except FileNotFoundError:
                results["models"][name] = {"skipped": True, "reason": "index_not_found"}
                continue

            metrics = compute_all_metrics(list_retrieved, list_relevant, k_values)
            mean_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
            results["models"][name] = {
                **metrics,
                "mean_latency_ms": round(mean_latency_ms, 2),
                "num_queries": len(list_retrieved),
            }
            logger.info(f"{name}: Recall@5={metrics['recall_at_5']:.4f} MRR@10={metrics['mrr_at_10']:.4f} nDCG@10={metrics['ndcg_at_10']:.4f} latency_ms={mean_latency_ms:.2f}")
    finally:
        db.close()

    # Save results JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"eval_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_path}")

    # Report
    report_path = out_dir / f"eval_report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation Report\n\n")
        f.write(f"- **Eval file:** {results['eval_file']}\n")
        f.write(f"- **Top-K:** {results['top_k']}\n")
        f.write(f"- **Num questions:** {results['num_questions']}\n\n")
        f.write("## Metrics\n\n")
        f.write("| Model      | Recall@5 | Recall@10 | MRR@5 | MRR@10 | nDCG@5 | nDCG@10 | Hit@5 | Hit@10 | Latency (ms) |\n")
        f.write("|------------|----------|-----------|-------|--------|--------|---------|-------|--------|---------------|\n")
        for name in ("base", "finetuned"):
            m = results["models"].get(name)
            if not m or m.get("skipped"):
                reason = m.get("reason", "not run") if m else "not run"
                f.write(f"| {name}        | —        | —         | —     | —      | —      | —       | —     | —      | {reason} |\n")
                continue
            f.write(
                f"| {name} | {m.get('recall_at_5', 0):.4f} | {m.get('recall_at_10', 0):.4f} | "
                f"{m.get('mrr_at_5', 0):.4f} | {m.get('mrr_at_10', 0):.4f} | "
                f"{m.get('ndcg_at_5', 0):.4f} | {m.get('ndcg_at_10', 0):.4f} | "
                f"{m.get('hit_rate_at_5', 0):.4f} | {m.get('hit_rate_at_10', 0):.4f} | "
                f"{m.get('mean_latency_ms', 0):.2f} |\n"
            )
        f.write("\n## Comparison (Base vs Fine-tuned)\n\n")
        base_m = results["models"].get("base")
        ft_m = results["models"].get("finetuned")
        if base_m and ft_m and not base_m.get("skipped") and not ft_m.get("skipped"):
            for key in ["recall_at_5", "recall_at_10", "mrr_at_5", "mrr_at_10", "ndcg_at_5", "ndcg_at_10", "mean_latency_ms"]:
                b = base_m.get(key, 0)
                f = ft_m.get(key, 0)
                diff = f - b if key != "mean_latency_ms" else b - f  # lower latency is better
                better = "✓" if (diff > 0 and key != "mean_latency_ms") or (key == "mean_latency_ms" and diff > 0) else ""
                f.write(f"- **{key}:** base={b:.4f} finetuned={f:.4f} (diff: {diff:+.4f}) {better}\n")
        else:
            f.write("Run both models without --skip-finetuned to see comparison.\n")
        f.write(f"\n*Generated at {datetime.now().isoformat()}*\n")
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
