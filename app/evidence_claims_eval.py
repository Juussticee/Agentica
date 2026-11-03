"""Adversarial evaluation: retrieval ranking and claim verification.

Two deterministic suites over authoritative fixtures
(`benchmarks/evidence_claims_v1.json`):

  retrieval  app.evidence.select vs the raw similarity order the index returns
             (precision@K, recall@K, hit@1, MRR, duplicate handling, latency);
  claims     app.claim_check.verify vs expected verdicts
             (support precision/recall, contradiction/unsupported detection,
             false-verification rate, citation accuracy).

No model, no network, no vector store. The fixture passages and verdicts are
the ground truth; a model's output is never used as an expected answer.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from app import claim_check, evidence
from app.config import PROJECT_ROOT, STORAGE_DIR

DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "evidence_claims_v1.json"
DEFAULT_OUT_DIR = STORAGE_DIR / "evidence_eval"


def load_dataset(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("retrieval_cases") and not data.get("claim_cases"):
        raise ValueError("evidence claims dataset has no cases")
    return data


def _nodes(passages: List[Dict[str, Any]]) -> list:
    return [
        SimpleNamespace(text=p["text"], score=p.get("score", 0.8),
                        metadata={"file_name": p["file_name"],
                                  "page_label": p.get("page_label")},
                        node_id=f"fixture-{index}")
        for index, p in enumerate(passages)
    ]


def _ids(nodes: list, case_nodes: list) -> List[int]:
    """Map selected nodes back to their fixture index (identity by text)."""
    out = []
    for node in nodes:
        for index, candidate in enumerate(case_nodes):
            if candidate.text == node.text and index not in out:
                out.append(index)
                break
    return out


def _ranking_metrics(order: List[int], relevant: set, k: int) -> Dict[str, Any]:
    top = order[:k]
    hits = [index for index in top if index in relevant]
    reciprocal = 0.0
    for rank, index in enumerate(order, start=1):
        if index in relevant:
            reciprocal = 1.0 / rank
            break
    return {
        "precision_at_k": round(len(hits) / k, 3) if k else 0.0,
        "recall_at_k": round(len(hits) / len(relevant), 3) if relevant else 0.0,
        "hit_at_1": bool(order and order[0] in relevant),
        "mrr": round(reciprocal, 3),
    }


def _retrieval_case(case: Dict[str, Any]) -> Dict[str, Any]:
    nodes = _nodes(case["passages"])
    relevant = set(case.get("relevant") or [])
    k = min(evidence.EVIDENCE_TOP_K, len(nodes))
    started = time.perf_counter()
    picked = evidence.select(case["question"], nodes)
    elapsed = (time.perf_counter() - started) * 1000

    similarity_order = sorted(
        range(len(nodes)),
        key=lambda i: -(getattr(nodes[i], "score", 0.0) or 0.0))
    reranked_order = _ids(picked["nodes"], nodes)
    return {
        "id": case["id"], "note": case.get("note", ""),
        "question": case["question"],
        "relevant": sorted(relevant),
        "similarity_order": similarity_order,
        "reranked_order": reranked_order,
        "dropped_duplicates": picked["dropped_duplicates"],
        "baseline": _ranking_metrics(similarity_order, relevant, k),
        "reranked": _ranking_metrics(reranked_order, relevant, k),
        "latency_ms": round(elapsed, 2),
    }


def _claim_case(case: Dict[str, Any]) -> Dict[str, Any]:
    nodes = _nodes(case["passages"])
    started = time.perf_counter()
    result = claim_check.verify(case["answer"], nodes)
    elapsed = (time.perf_counter() - started) * 1000
    actual = [c["status"] for c in result["claims"]]
    expected = list(case.get("expect") or [])
    return {
        "id": case["id"], "note": case.get("note", ""),
        "answer": case["answer"], "expected": expected, "actual": actual,
        "matches": actual == expected,
        "claims": result["claims"],
        "latency_ms": round(elapsed, 2),
    }


def _summarize_retrieval(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(key: str, condition: str) -> float:
        return round(sum(row[condition][key] for row in rows) /
                     max(1, len(rows)), 3)

    return {
        "cases": len(rows),
        "baseline_precision_at_k": mean("precision_at_k", "baseline"),
        "reranked_precision_at_k": mean("precision_at_k", "reranked"),
        "baseline_recall_at_k": mean("recall_at_k", "baseline"),
        "reranked_recall_at_k": mean("recall_at_k", "reranked"),
        "baseline_hit_at_1": mean("hit_at_1", "baseline"),
        "reranked_hit_at_1": mean("hit_at_1", "reranked"),
        "baseline_mrr": mean("mrr", "baseline"),
        "reranked_mrr": mean("mrr", "reranked"),
        "duplicates_dropped": sum(row["dropped_duplicates"] for row in rows),
        "mean_latency_ms": round(
            sum(row["latency_ms"] for row in rows) / max(1, len(rows)), 2),
        "improved": [row["id"] for row in rows
                     if (row["reranked"]["recall_at_k"]
                         > row["baseline"]["recall_at_k"])
                     or (row["reranked"]["mrr"] > row["baseline"]["mrr"])],
        "regressed": [row["id"] for row in rows
                      if (row["reranked"]["recall_at_k"]
                          < row["baseline"]["recall_at_k"])
                      or (row["reranked"]["mrr"] < row["baseline"]["mrr"])],
    }


def _summarize_claims(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected_positive = {"supported", "partial"}
    classified = [(row["expected"][0], row["actual"][0])
                  for row in rows if row["expected"] and row["actual"]]
    true_positive = sum(1 for exp, act in classified
                        if exp == act)
    supported_expected = sum(1 for exp, _ in classified
                             if exp in expected_positive)
    supported_actual = sum(1 for _, act in classified
                           if act in expected_positive)
    supported_hits = sum(1 for exp, act in classified
                         if exp in expected_positive and act in expected_positive)
    contradictions = [c for c in classified if c[0] == "contradicted"]
    unsupported = [c for c in classified if c[0] == "unsupported"]
    return {
        "cases": len(rows),
        "exact_match": sum(1 for row in rows if row["matches"]),
        "claim_support_precision": (
            round(supported_hits / supported_actual, 3)
            if supported_actual else None),
        "claim_support_recall": (
            round(supported_hits / supported_expected, 3)
            if supported_expected else None),
        "contradiction_detection": (
            round(sum(1 for exp, act in contradictions
                      if act == "contradicted") / len(contradictions), 3)
            if contradictions else None),
        "unsupported_detection": (
            round(sum(1 for exp, act in unsupported
                      if act == "unsupported") / len(unsupported), 3)
            if unsupported else None),
        "false_verification_rate": (
            round(sum(1 for exp, act in classified
                      if exp != "supported" and act == "supported")
                  / len(classified), 3)
            if classified else None),
        "mean_latency_ms": round(
            sum(row["latency_ms"] for row in rows) / max(1, len(rows)), 2),
        "failed": [row["id"] for row in rows if not row["matches"]],
        "accuracy": round(true_positive / len(classified), 3)
                    if classified else None,
    }


def run_evaluation(dataset: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = [_retrieval_case(c) for c in dataset.get("retrieval_cases") or []]
    claims = [_claim_case(c) for c in dataset.get("claim_cases") or []]
    return {
        "dataset": dataset.get("version"),
        "harness_version": dataset.get("harness_version"),
        "mode": dataset.get("mode", "offline_deterministic"),
        "mode_note": dataset.get("mode_note", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrieval": {"summary": _summarize_retrieval(retrieval),
                      "cases": retrieval},
        "claims": {"summary": _summarize_claims(claims), "cases": claims},
    }


def markdown(report: Dict[str, Any]) -> str:
    r = report["retrieval"]["summary"]
    c = report["claims"]["summary"]
    lines = [
        "# Retrieval & claim verification — evaluation",
        "",
        f"- dataset: `{report['dataset']}` (harness {report['harness_version']})",
        f"- mode: **{report['mode']}** — {report['mode_note']}",
        f"- generated: {report['generated_at']}",
        "",
        "## Retrieval ranking (similarity order vs reranked)",
        "",
        "| metric | similarity order | reranked |",
        "| --- | --- | --- |",
        f"| precision@K | {r['baseline_precision_at_k']} | "
        f"{r['reranked_precision_at_k']} |",
        f"| recall@K | {r['baseline_recall_at_k']} | "
        f"{r['reranked_recall_at_k']} |",
        f"| hit@1 | {r['baseline_hit_at_1']} | {r['reranked_hit_at_1']} |",
        f"| MRR | {r['baseline_mrr']} | {r['reranked_mrr']} |",
        "",
        f"- duplicate passages dropped: {r['duplicates_dropped']}",
        f"- mean selection latency: {r['mean_latency_ms']} ms",
        f"- improved: {', '.join(r['improved']) or 'none'}",
        f"- regressed: {', '.join(r['regressed']) or 'none'}",
        "",
        "## Claim verification",
        "",
        f"- cases: {c['cases']}, exact verdict match: {c['exact_match']}",
        f"- claim support precision: {c['claim_support_precision']}",
        f"- claim support recall: {c['claim_support_recall']}",
        f"- contradiction detection: {c['contradiction_detection']}",
        f"- unsupported detection: {c['unsupported_detection']}",
        f"- false verification rate: {c['false_verification_rate']}",
        f"- mean latency: {c['mean_latency_ms']} ms",
        f"- failed cases: {', '.join(c['failed']) or 'none'}",
        "",
        "## Cases",
        "",
        "| case | expected | actual | note |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["retrieval"]["cases"]:
        lines.append(
            f"| {row['id']} (retrieval) | — | reranked "
            f"{row['reranked_order']} (MRR {row['reranked']['mrr']}) | "
            f"{row['note'][:60]} |")
    for row in report["claims"]["cases"]:
        lines.append(
            f"| {row['id']} | {', '.join(row['expected']) or '—'} | "
            f"{', '.join(row['actual']) or '—'} | {row['note'][:60]} |")
    lines.append("")
    return "\n".join(lines)


def write_report(report: Dict[str, Any], out_dir: Path,
                 print_markdown: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evidence_claims_report.json"
    md_path = out_dir / "evidence_claims_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = markdown(report)
    md_path.write_text(text, encoding="utf-8")
    if print_markdown:
        print(text)
    return json_path, md_path
