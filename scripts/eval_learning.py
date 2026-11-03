"""Run the learning A/B benchmark through the real GraphRunner (2026-10-30).

The harness lives in `app/learning/benchmark.py`; this script is the offline
CLI. Each case runs twice (baseline without a lesson set, learning-enabled
with the applicable active lessons), inside `writes_disabled()` so a measured
run cannot teach the system. The provider is a scripted double: mode is
labelled `offline_deterministic`, no live model, no network.

Usage:
    env\\Scripts\\python.exe scripts\\eval_learning.py
    env\\Scripts\\python.exe scripts\\eval_learning.py --out-dir storage\\learning_eval
    env\\Scripts\\python.exe scripts\\eval_learning.py --print --fail-on-regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.learning import benchmark  # noqa: E402
from app.learning import evaluation  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic A/B benchmark for Agentica's learned "
                    "lessons (offline).")
    parser.add_argument("--dataset", default=str(benchmark.DEFAULT_DATASET))
    parser.add_argument("--out-dir", default=str(benchmark.DEFAULT_OUT_DIR))
    parser.add_argument("--store-dir",
                        help="directory for the throwaway benchmark stores "
                             "(default: temporary directories)")
    parser.add_argument("--print", action="store_true",
                        help="print the markdown summary")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit 1 when any task regressed")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    dataset = evaluation.load_dataset(dataset_path)
    report = benchmark.run_benchmark(
        dataset,
        store_dir=Path(args.store_dir) if args.store_dir else None,
        dataset_path=dataset_path)
    json_path, md_path = benchmark.write_report(
        report, Path(args.out_dir), print_markdown=args.print)

    summary = report["summary"]
    print(f"benchmark: {report['benchmark_version']} "
          f"({report['task_count']} tasks x 2 conditions)")
    print(f"completeness: baseline "
          f"{summary['baseline']['completeness'] * 100:.1f}% -> learning "
          f"{summary['learning']['completeness'] * 100:.1f}% "
          f"({summary['delta']['completeness_pp']:+.1f} pp)")
    print(f"qa findings: {summary['baseline']['qa_findings']} -> "
          f"{summary['learning']['qa_findings']}")
    print(f"improved {summary['delta']['improved_tasks']}, regressed "
          f"{summary['delta']['regressed_tasks']}, unchanged "
          f"{summary['delta']['unchanged_tasks']}")
    for regression in summary["regressions"]:
        print(f"  regression: {regression['task_id']} "
              f"({regression['kind']}) "
              f"{regression.get('requirement_id') or ''}")
    print(f"report: {json_path}")
    print(f"summary: {md_path}")
    if args.fail_on_regression and summary["delta"]["regressions"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
