"""
Evaluation script: load eval questions, run retrieval for base and finetuned models,
compute Recall@5/10, MRR@5/10, nDCG@5/10, Hit Rate, latency; save results, report, and store in eval_runs.

Usage:
  python scripts/evaluate.py --eval-file data/eval_questions.json
  python scripts/evaluate.py --eval-file data/eval_questions.json --no-store
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

from loguru import logger

from app.db.session import SessionLocal
from app.services.evaluation_runner import (
    load_eval_questions,
    run_evaluation,
    persist_eval_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval: base vs finetuned")
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval_questions.json"), help="JSON or JSONL with eval questions")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Directory for results and report")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top_k (used for @5 and @10 metrics)")
    parser.add_argument("--skip-finetuned", action="store_true", help="Only evaluate base model (e.g. when finetuned index missing)")
    parser.add_argument("--no-store", action="store_true", help="Do not write results to eval_runs table")
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
    db = SessionLocal()
    try:
        results = run_evaluation(db, eval_questions, top_k=top_k, skip_finetuned=args.skip_finetuned)
    finally:
        db.close()

    results["eval_file"] = str(eval_path)
    results["num_questions"] = len(eval_questions)

    if results.get("num_questions", 0) == 0:
        logger.error("No valid questions with ground truth. Exiting.")
        sys.exit(1)

    for name in ("base", "finetuned"):
        m = results.get("models", {}).get(name)
        if m and not m.get("skipped"):
            logger.info(
                f"{name}: Recall@5={m.get('recall_at_5', 0):.4f} MRR@10={m.get('mrr_at_10', 0):.4f} "
                f"nDCG@10={m.get('ndcg_at_10', 0):.4f} latency_ms={m.get('mean_latency_ms', 0):.2f}"
            )

    if not args.no_store:
        run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db2 = SessionLocal()
        try:
            persist_eval_runs(db2, run_name, results, eval_file=str(eval_path), top_k=top_k)
            logger.info(f"Stored eval runs as run_name={run_name}")
        finally:
            db2.close()

    # Extract failure data for report (do not persist to JSON)
    fd = results.pop("_failure_data", None)

    # Save results JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"eval_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_path}")

    # Failure analysis (spec: at least 5 cases in main report; use base-model retrieval)
    top_k_report = top_k
    failures: list[dict] = []
    if fd:
        list_retrieved = fd["list_retrieved"]
        list_relevant = fd["list_relevant"]
        valid_q = fd["valid_questions"]

        def _short_id(cid: str) -> str:
            return (cid or "?").replace("-", "")[:8]

        for i, (retrieved_ids, relevant) in enumerate(zip(list_retrieved, list_relevant)):
            if i >= len(valid_q):
                break
            eq = valid_q[i]
            question = eq.get("question", "")
            retrieved_set = set(retrieved_ids)
            missed = relevant - retrieved_set
            if not missed:
                continue
            expected_display = ", ".join(_short_id(c) for c in list(missed)[:3])
            retrieved_display = ", ".join(_short_id(c) for c in retrieved_ids[:5])
            rank_of_first = None
            for r, cid in enumerate(retrieved_ids, 1):
                if cid in relevant:
                    rank_of_first = r
                    break
            if rank_of_first and rank_of_first > 1:
                reason = f"Expected chunk(s) not in top-{top_k_report}, or ranked at position {rank_of_first} (semantic similarity)."
            else:
                reason = f"Expected chunk(s) not in top-{top_k_report} (semantic similarity / vocabulary gap)."
            failures.append({
                "question": question,
                "expected": expected_display,
                "retrieved": retrieved_display,
                "reason": reason,
                "fix": "Add fine-tuning data for this query–chunk pair; or increase top_k / chunk overlap.",
            })
    # Spec: at least 5 failure cases; take first 5 if more, or all if fewer
    failure_cases = failures[:5] if len(failures) >= 5 else failures

    # Build report content (metrics + comparison + failure analysis)
    report_lines = [
        "# Retrieval Evaluation Report",
        "",
        f"- **Eval file:** {results['eval_file']}",
        f"- **Top-K:** {results['top_k']}",
        f"- **Num questions:** {results['num_questions']}",
        "",
        "## Metrics",
        "",
        "| Model      | Recall@5 | Recall@10 | MRR@5 | MRR@10 | nDCG@5 | nDCG@10 | Hit@5 | Hit@10 | Latency (ms) |",
        "|------------|----------|-----------|-------|--------|--------|---------|-------|--------|---------------|",
    ]
    for name in ("base", "finetuned"):
        m = results["models"].get(name)
        if not m or m.get("skipped"):
            reason = m.get("reason", "not run") if m else "not run"
            report_lines.append(f"| {name}        | —        | —         | —     | —      | —      | —       | —     | —      | {reason} |")
            continue
        report_lines.append(
            f"| {name} | {m.get('recall_at_5', 0):.4f} | {m.get('recall_at_10', 0):.4f} | "
            f"{m.get('mrr_at_5', 0):.4f} | {m.get('mrr_at_10', 0):.4f} | "
            f"{m.get('ndcg_at_5', 0):.4f} | {m.get('ndcg_at_10', 0):.4f} | "
            f"{m.get('hit_rate_at_5', 0):.4f} | {m.get('hit_rate_at_10', 0):.4f} | "
            f"{m.get('mean_latency_ms', 0):.2f} |"
        )
    report_lines.extend([
        "",
        "## Comparison (Base vs Fine-tuned)",
        "",
    ])
    base_m = results["models"].get("base")
    ft_m = results["models"].get("finetuned")
    if base_m and ft_m and not base_m.get("skipped") and not ft_m.get("skipped"):
        for key in ["recall_at_5", "recall_at_10", "mrr_at_5", "mrr_at_10", "ndcg_at_5", "ndcg_at_10", "mean_latency_ms"]:
            b = base_m.get(key, 0)
            f = ft_m.get(key, 0)
            diff = f - b if key != "mean_latency_ms" else b - f
            better = "✓" if (diff > 0 and key != "mean_latency_ms") or (key == "mean_latency_ms" and diff > 0) else ""
            report_lines.append(f"- **{key}:** base={b:.4f} finetuned={f:.4f} (diff: {diff:+.4f}) {better}")
    else:
        report_lines.append("Run both models without --skip-finetuned to see comparison.")
    report_lines.extend([
        "",
        "## Failure analysis (at least 5 cases)",
        "",
        f"Expected vs retrieved chunks for questions where at least one relevant chunk was **not** in top-{top_k_report}. "
        f"Total failures (base model): {len(failures)}. Below: first 5 or all if fewer.",
        "",
    ])
    if failure_cases:
        for idx, fail in enumerate(failure_cases, 1):
            report_lines.extend([
                f"### Failure {idx}",
                "",
                f"**Question:**  {fail['question']}",
                "",
                f"**Expected (missed):**  {fail['expected']}",
                "",
                f"**Retrieved (top-5):**  {fail['retrieved']}",
                "",
                f"**Reason:**  {fail['reason']}",
                "",
                f"**Fix:**  {fail['fix']}",
                "",
                "---",
                "",
            ])
    else:
        report_lines.append("No failures: all expected chunks appeared in the retrieved top-k.")
    report_lines.extend(["", f"*Generated at {datetime.now().isoformat()}*", ""])
    report_content = "\n".join(report_lines)

    report_path = out_dir / f"eval_report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Report saved to {report_path}")

    # Spec deliverable: reports/eval.md (exact filename)
    eval_md = out_dir / "eval.md"
    with open(eval_md, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Deliverable report saved to {eval_md}")


if __name__ == "__main__":
    main()
