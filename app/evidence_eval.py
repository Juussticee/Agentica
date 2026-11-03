"""Evaluation harness: model-knowledge answering vs evidence-grounded answering.

WHAT THIS MEASURES
The same general route, the same QA gate and the same scripted provider, under
two conditions:

  baseline   retrieval is disabled (the pre-2026-11-01 behaviour: the model
             answers from its own knowledge, nothing is checked against the
             local corpus);
  grounded   the real retrieval path runs against the case's AUTHORITATIVE
             fixture passages, and the answer is composed from them through
             the existing grounding gate.

WHAT IT DOES NOT MEASURE
The model. The provider is a scripted double that repeats a prepared reply, so
every difference reported here is produced by the PIPELINE (which claims
survive, what gets cited, what QA reports). The expected facts, licence values
and figures live in the dataset as fixtures — never in model output — so a case
can fail even when the model's prose looks plausible.

Deterministic, offline, no network, no vector store, no Ollama.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

from app import evidence
from app.config import PROJECT_ROOT, STORAGE_DIR

DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "evidence_general_v1.json"
DEFAULT_OUT_DIR = STORAGE_DIR / "evidence_eval"

# QA findings this harness classifies. Kept explicit so "no claim flags" is a
# real assertion rather than "no flags at all".
CLAIM_FLAGS = (
    "cannot support",
    "licence",
    "conflicts with the consulted passages",
    "did not retrieve",
    "cites none",
    "refers to local documents",
)


class FixtureModel:
    """Scripted provider double: the reply depends only on the prompt."""

    def __init__(self, from_evidence: str, from_memory: str):
        self.from_evidence = from_evidence
        self.from_memory = from_memory
        self.prompts: List[str] = []

    def invoke(self, prompt):
        text = str(prompt)
        self.prompts.append(text)
        if evidence.EVIDENCE_MARKER in text:
            return SimpleNamespace(content=self.from_evidence)
        return SimpleNamespace(content=self.from_memory)


def load_dataset(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("evidence evaluation dataset has no cases")
    return data


def _nodes(case: Dict[str, Any]) -> list:
    return [
        SimpleNamespace(text=p["text"], score=p.get("score", 0.8),
                        metadata={"file_name": p["file_name"],
                                  "page_label": p.get("page_label")})
        for p in case.get("passages") or []
    ]


def _claim_flags(flags: List[str]) -> List[str]:
    return [f for f in flags if any(c in f for c in CLAIM_FLAGS)]


def _answer_case(case: Dict[str, Any], grounded: bool) -> Dict[str, Any]:
    """Run one answer case under one condition and collect its outcome."""
    nodes = _nodes(case)
    model = FixtureModel(case.get("grounded_answer", ""),
                         case.get("memory_answer", ""))
    state: Dict[str, Any] = {
        "question": case["question"], "route": "general", "sources": [],
        "qa_flags": [], "agent_answers": {}, "lessons": [],
    }
    error = None
    retriever = (lambda *_a, **_k: nodes) if grounded else (lambda *_a, **_k: [])
    try:
        with patch.object(evidence, "retrieve", side_effect=retriever), \
                patch("app.llm_client.get_llm", return_value=model):
            from app.graph import general_node

            update = general_node(state)
        state.update(update)
        from app.qa_node import run_qa

        flags = run_qa(state)
    except Exception as exc:  # noqa: BLE001 — a crash IS a measured outcome
        error = f"{type(exc).__name__}: {exc}"
        flags = []
    answer = str(state.get("answer") or "")
    ev = state.get("evidence") if isinstance(state.get("evidence"), dict) else {}
    return {
        "answer": answer,
        "sources": list(state.get("sources") or []),
        "flags": flags,
        "claim_flags": _claim_flags(flags),
        "from": ev.get("answered_from"),
        "status": ev.get("status"),
        "coverage": ev.get("coverage"),
        "conflict": ev.get("conflict"),
        "missing": ev.get("missing") or [],
        "error": error,
    }


def _judge(case: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Per-case checks. Every check is deterministic and fixture-backed."""
    expect = case.get("expect") or {}
    answer = outcome["answer"]
    checks: Dict[str, bool] = {}
    checks["executed"] = outcome["error"] is None and bool(answer.strip())
    checks["from"] = outcome["from"] == expect.get("from")
    checks["must_include"] = all(
        text.lower() in answer.lower()
        for text in expect.get("must_include") or []
    )
    checks["must_not_include"] = not any(
        text.lower() in answer.lower()
        for text in expect.get("must_not_include") or []
    )
    if expect.get("must_include_sources"):
        checks["sources"] = any(s.startswith(evidence.SOURCE_PREFIX)
                                for s in outcome["sources"])
    else:
        checks["sources"] = not outcome["sources"]
    wanted = expect.get("flag")
    if wanted == "figure":
        checks["flag"] = any("cannot support" in f
                             for f in outcome["claim_flags"])
    elif wanted == "licence":
        checks["flag"] = any("licence" in f.lower()
                             for f in outcome["claim_flags"])
    elif wanted == "contradicted":
        checks["flag"] = any("conflicts with the consulted passages" in f
                             for f in outcome["claim_flags"])
    else:
        checks["flag"] = not outcome["claim_flags"]
    if expect.get("partial"):
        checks["partial"] = ("Not stated by the passages" in answer
                             and bool(outcome["missing"]))
    if case.get("expect_source_prefix"):
        checks["sources"] = any(
            s.startswith(case["expect_source_prefix"])
            for s in outcome["sources"])
    checks["passed"] = all(checks.values())
    return checks


def _route_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from app.supervisor import supervisor

    route = str(supervisor({"question": case["question"]}).get("route") or "")
    return {"route": route, "answer": route, "sources": [], "flags": [],
            "claim_flags": [], "from": None, "status": None, "coverage": None,
            "conflict": None, "missing": [], "error": None,
            "checks": {"route": route == case.get("expect_route"),
                       "passed": route == case.get("expect_route")}}


def _followup_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from app.followup import followup_node

    state = {"question": case["question"], "route": "followup",
             "sources": [], "qa_flags": [], "agent_answers": {},
             "prior_execution": case["prior"]}
    model = FixtureModel("", case.get("followup_answer", ""))
    error = None
    try:
        with patch("app.llm_client.get_llm", return_value=model):
            update = followup_node(state)
        state.update(update)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    sources = list(state.get("sources") or [])
    prefix = case.get("expect_source_prefix") or ""
    checks = {"executed": error is None,
              "sources": any(s.startswith(prefix) for s in sources)}
    checks["passed"] = all(checks.values())
    return {"answer": str(state.get("answer") or ""), "sources": sources,
            "flags": [], "claim_flags": [], "from": None, "status": None,
            "coverage": None, "conflict": None, "missing": [],
            "error": error, "checks": checks}


def _run_case(case: Dict[str, Any], grounded: bool) -> Dict[str, Any]:
    kind = case.get("kind")
    if kind == "route":
        return _route_case(case)
    if kind == "followup":
        return _followup_case(case)
    outcome = _answer_case(case, grounded=grounded)
    outcome["checks"] = _judge(case, outcome)
    return outcome


def _summarize(cases: List[Dict[str, Any]],
               results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    answer_cases = [c for c in cases if c.get("kind") == "answer"]
    outcomes = [results[c["id"]] for c in answer_cases]
    total = len(outcomes) or 1

    def rate(pred) -> float:
        return round(sum(1 for o in outcomes if pred(o)) / total, 3)

    from app.qa_node import _figures

    evidence_cases = [o for o in outcomes
                      if (o["from"] == "evidence")]
    memory_cases = [o for o in outcomes if o["from"] == "model"]
    numeric = [o for o in outcomes if _figures(o["answer"])]
    return {
        "cases": len(outcomes),
        "passed": sum(1 for o in outcomes if o["checks"].get("passed")),
        "factual_support_rate": rate(lambda o: o["checks"].get("must_include")),
        "unsupported_claim_rate": rate(
            lambda o: any("cannot support" in f for f in o["claim_flags"])),
        "contradicted_claim_rate": rate(
            lambda o: any("conflicts with the consulted passages" in f
                          for f in o["claim_flags"])),
        "citation_accuracy": (
            round(sum(1 for o in evidence_cases
                      if o["checks"].get("sources")) / len(evidence_cases), 3)
            if evidence_cases else None),
        "evidence_coverage": (
            round(sum(float(o["coverage"] or 0) for o in outcomes
                      if o["coverage"] is not None) /
                  max(1, sum(1 for o in outcomes
                             if o["coverage"] is not None)), 3)),
        "partial_answer_quality": (
            round(sum(1 for o in outcomes if o["checks"].get("partial")) /
                  max(1, sum(1 for c in answer_cases
                             if (c.get("expect") or {}).get("partial"))), 3)),
        "correct_abstention": (
            round(sum(1 for o in memory_cases
                      if o["checks"].get("must_include")
                      and not o["sources"]) / len(memory_cases), 3)
            if memory_cases else None),
        "unnecessary_abstention": rate(
            lambda o: o["from"] == "model"
            and (o["checks"].get("from") is False)),
        "numerical_grounding_accuracy": (
            round(sum(1 for o in numeric if o["checks"].get("flag")) /
                  len(numeric), 3) if numeric else None),
        "execution_reliability": rate(lambda o: o["checks"].get("executed")),
    }


def run_evaluation(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Run every case under both conditions; return the full report."""
    cases = dataset["cases"]
    report: Dict[str, Any] = {
        "dataset": dataset.get("version"),
        "harness_version": dataset.get("harness_version"),
        "mode": dataset.get("mode", "offline_deterministic"),
        "mode_note": dataset.get("mode_note", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "conditions": {},
        "cases": [],
    }
    for condition, grounded in (("baseline", False), ("grounded", True)):
        results = {c["id"]: _run_case(c, grounded) for c in cases}
        report["conditions"][condition] = {
            "summary": _summarize(cases, results),
            "results": results,
        }

    baseline = report["conditions"]["baseline"]["results"]
    grounded = report["conditions"]["grounded"]["results"]
    regressions = []
    improvements = []
    for case in cases:
        before = baseline[case["id"]]["checks"].get("passed")
        after = grounded[case["id"]]["checks"].get("passed")
        if before and not after:
            regressions.append(case["id"])
        if after and not before:
            improvements.append(case["id"])

    base_sum = report["conditions"]["baseline"]["summary"]
    ground_sum = report["conditions"]["grounded"]["summary"]
    report["delta"] = {
        key: (None if base_sum[key] is None or ground_sum[key] is None
              else round(ground_sum[key] - base_sum[key], 3))
        for key in ("factual_support_rate", "unsupported_claim_rate",
                    "contradicted_claim_rate", "citation_accuracy",
                    "evidence_coverage", "correct_abstention",
                    "execution_reliability")
    }
    report["improved_cases"] = improvements
    report["regressions"] = regressions

    for case in cases:
        report["cases"].append({
            "id": case["id"],
            "kind": case.get("kind", "answer"),
            "question": case.get("question"),
            "note": case.get("note", ""),
            "baseline": baseline[case["id"]],
            "grounded": grounded[case["id"]],
        })
    return report


def markdown(report: Dict[str, Any]) -> str:
    base = report["conditions"]["baseline"]["summary"]
    ground = report["conditions"]["grounded"]["summary"]
    lines = [
        "# Evidence-grounded general answering — evaluation",
        "",
        f"- dataset: `{report['dataset']}` (harness {report['harness_version']})",
        f"- mode: **{report['mode']}** — {report['mode_note']}",
        f"- generated: {report['generated_at']}",
        "",
        "## Metrics (same pipeline, retrieval off vs on)",
        "",
        "| metric | baseline | grounded | delta |",
        "| --- | --- | --- | --- |",
    ]
    for key in ("factual_support_rate", "unsupported_claim_rate",
                "contradicted_claim_rate", "citation_accuracy",
                "evidence_coverage", "partial_answer_quality",
                "correct_abstention", "unnecessary_abstention",
                "execution_reliability"):
        before, after = base.get(key), ground.get(key)
        delta = (None if before is None or after is None
                 else round(after - before, 3))
        lines.append(
            f"| `{key}` | {before} | {after} | {delta} |")
    lines += [
        "",
        f"- cases passed: baseline {base['passed']}/{base['cases']} → "
        f"grounded {ground['passed']}/{ground['cases']}",
        f"- improved cases: {', '.join(report['improved_cases']) or 'none'}",
        f"- regressions: {', '.join(report['regressions']) or 'none'}",
        "",
        "## Per case",
        "",
        "| case | question | baseline | grounded | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        def cell(result: Dict[str, Any]) -> str:
            if result["checks"].get("passed"):
                return "pass"
            failed = [k for k, v in result["checks"].items()
                      if v is False]
            return "FAIL (" + ", ".join(failed) + ")"

        lines.append(
            f"| {row['id']} | {str(row['question'])[:58]} | "
            f"{cell(row['baseline'])} | {cell(row['grounded'])} | "
            f"{row['note'][:70]} |")
    lines.append("")
    return "\n".join(lines)


def write_report(report: Dict[str, Any], out_dir: Path,
                 print_markdown: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evidence_general_report.json"
    md_path = out_dir / "evidence_general_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = markdown(report)
    md_path.write_text(text, encoding="utf-8")
    if print_markdown:
        print(text)
    return json_path, md_path
