"""Run the evidence-grounded general answering evaluation (offline).

The harness lives in `app/evidence_eval.py`; this script is the CLI. Every
case runs twice — retrieval disabled (baseline) and enabled (grounded) — with a
scripted provider double, so the comparison is deterministic and costs no
model calls.

Usage:
    env\\Scripts\\python.exe scripts\\eval_evidence.py
    env\\Scripts\\python.exe scripts\\eval_evidence.py --print
    env\\Scripts\\python.exe scripts\\eval_evidence.py --fail-on-regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import evidence_eval  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Baseline vs evidence-grounded general answering (offline).")
    parser.add_argument("--dataset", default=str(evidence_eval.DEFAULT_DATASET))
    parser.add_argument("--out-dir", default=str(evidence_eval.DEFAULT_OUT_DIR))
    parser.add_argument("--print", action="store_true",
                        help="print the markdown summary")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit 1 when a case that passed before now fails")
    args = parser.parse_args(argv)

    dataset = evidence_eval.load_dataset(Path(args.dataset))
    report = evidence_eval.run_evaluation(dataset)
    json_path, md_path = evidence_eval.write_report(
        report, Path(args.out_dir), print_markdown=args.print)

    base = report["conditions"]["baseline"]["summary"]
    ground = report["conditions"]["grounded"]["summary"]
    print(f"dataset: {report['dataset']} ({report['mode']})")
    print(f"cases passed: baseline {base['passed']}/{base['cases']} -> "
          f"grounded {ground['passed']}/{ground['cases']}")
    print(f"factual support: {base['factual_support_rate']} -> "
          f"{ground['factual_support_rate']}")
    print(f"unsupported claims: {base['unsupported_claim_rate']} -> "
          f"{ground['unsupported_claim_rate']}")
    print(f"citation accuracy: {base['citation_accuracy']} -> "
          f"{ground['citation_accuracy']}")
    print(f"improved: {', '.join(report['improved_cases']) or 'none'}")
    print(f"regressions: {', '.join(report['regressions']) or 'none'}")
    print(f"report: {json_path}")
    print(f"summary: {md_path}")

    if args.fail_on_regression and report["regressions"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
