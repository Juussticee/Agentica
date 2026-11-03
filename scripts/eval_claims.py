"""Run the retrieval-ranking and claim-verification evaluation (offline).

Usage:
    env\\Scripts\\python.exe scripts\\eval_claims.py
    env\\Scripts\\python.exe scripts\\eval_claims.py --print
    env\\Scripts\\python.exe scripts\\eval_claims.py --fail-on-regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import evidence_claims_eval  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval ranking + claim verification (offline).")
    parser.add_argument("--dataset",
                        default=str(evidence_claims_eval.DEFAULT_DATASET))
    parser.add_argument("--out-dir",
                        default=str(evidence_claims_eval.DEFAULT_OUT_DIR))
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args(argv)

    dataset = evidence_claims_eval.load_dataset(Path(args.dataset))
    report = evidence_claims_eval.run_evaluation(dataset)
    json_path, md_path = evidence_claims_eval.write_report(
        report, Path(args.out_dir), print_markdown=args.print)

    r = report["retrieval"]["summary"]
    c = report["claims"]["summary"]
    print(f"dataset: {report['dataset']} ({report['mode']})")
    print(f"retrieval recall@K: {r['baseline_recall_at_k']} -> "
          f"{r['reranked_recall_at_k']} | MRR: {r['baseline_mrr']} -> "
          f"{r['reranked_mrr']} | dupes dropped: {r['duplicates_dropped']}")
    print(f"claim verdicts: {c['exact_match']}/{c['cases']} exact | "
          f"support precision {c['claim_support_precision']} | "
          f"contradiction detection {c['contradiction_detection']} | "
          f"false verification {c['false_verification_rate']}")
    print(f"retrieval regressed: {', '.join(r['regressed']) or 'none'}")
    print(f"claim failures: {', '.join(c['failed']) or 'none'}")
    print(f"report: {json_path}")
    print(f"summary: {md_path}")

    if args.fail_on_regression and (r["regressed"] or c["failed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
