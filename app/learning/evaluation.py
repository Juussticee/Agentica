"""Learning evaluation: scoring, comparison and reporting (2026-10-30).

The harness that RUNS a benchmark lives in `scripts/eval_learning.py`; this
module is the part that can be reasoned about on its own:

  * `validate_dataset`   a benchmark file must declare its own success criteria
  * `score_answer`       requirement completeness, measured on the final answer
  * `evaluate_condition` one side (baseline or learning-enabled) of one task
  * `compare_task`       the diff, with regressions detected explicitly
  * `summarize`          totals, per capability, per lesson
  * `build_report` / `render_markdown`   machine- and human-readable output

Rules that keep the numbers honest:

  * requirements are checked on the ANSWER stored by the execution, never on
    the prompt, so a lesson that merely reaches a prompt cannot look useful;
  * "retrieved" and "applied" are different fields; only `applied` comes from
    the execution's own `learning.lessons_applied` event;
  * a lesson is credited with an improvement only through the outcome diff of
    a task where it was applied, and is blamed for a regression the same way;
  * missing/blank answers count as unsatisfied, not as unknown;
  * nothing here writes to the store, activates a lesson or changes a skill.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.learning import calibration as C

HARNESS_VERSION = 1

CHECK_KINDS = ("contains_all", "contains_any", "not_contains_all", "regex",
               "min_words", "max_words")

# Behavioural metrics a requirement may be tagged with (2026-10-30, learning
# calibration). They are reported SEPARATELY, never folded into one score: a
# benchmark that "improved" while becoming more restrictive has failed.
METRICS = ("overgeneralization_rate", "unnecessary_restriction_rate",
           "clarification_overuse", "initiative_retention",
           "cross_capability_regression", "constraint_accuracy",
           "preference_scope_accuracy")

OUTCOMES = ("improved", "regression", "mixed", "unchanged")


class DatasetError(ValueError):
    """The benchmark file cannot be evaluated as written."""


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def validate_dataset(dataset: Any) -> Dict[str, Any]:
    """Check the shape a benchmark must have before anything is executed."""
    if not isinstance(dataset, dict):
        raise DatasetError("dataset must be a JSON object")
    for key in ("version", "lessons", "cases"):
        if key not in dataset:
            raise DatasetError(f"dataset is missing {key!r}")
    lessons = dataset["lessons"]
    if not isinstance(lessons, list):
        raise DatasetError("lessons must be a list")
    lesson_ids = set()
    for lesson in lessons:
        if not isinstance(lesson, dict) or not lesson.get("lesson_id"):
            raise DatasetError("every lesson needs a lesson_id")
        if lesson["lesson_id"] in lesson_ids:
            raise DatasetError(f"duplicate lesson id {lesson['lesson_id']!r}")
        lesson_ids.add(lesson["lesson_id"])
        if not str(lesson.get("corrective_behavior") or "").strip():
            raise DatasetError(
                f"lesson {lesson['lesson_id']!r} has no corrective_behavior")
    cases = dataset["cases"]
    if not isinstance(cases, list) or not cases:
        raise DatasetError("cases must be a non-empty list")
    task_ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise DatasetError("every case must be an object")
        task_id = case.get("task_id")
        if not task_id:
            raise DatasetError("every case needs a task_id")
        if task_id in task_ids:
            raise DatasetError(f"duplicate task id {task_id!r}")
        task_ids.add(task_id)
        for field in ("capability", "request", "baseline_answer"):
            if not str(case.get(field) or "").strip():
                raise DatasetError(f"{task_id}: {field} is required")
        requirements = case.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            raise DatasetError(f"{task_id}: at least one requirement is required")
        seen = set()
        for requirement in requirements:
            if not isinstance(requirement, dict) or not requirement.get("id"):
                raise DatasetError(f"{task_id}: every requirement needs an id")
            if requirement["id"] in seen:
                raise DatasetError(
                    f"{task_id}: duplicate requirement {requirement['id']!r}")
            seen.add(requirement["id"])
            check = requirement.get("check") or {}
            kind = check.get("kind")
            if kind not in CHECK_KINDS:
                raise DatasetError(
                    f"{task_id}/{requirement['id']}: unsupported check {kind!r}")
            if kind == "regex" and not check.get("pattern"):
                raise DatasetError(
                    f"{task_id}/{requirement['id']}: regex needs a pattern")
            if kind in ("contains_all", "contains_any", "not_contains_all") and not (
                    isinstance(check.get("terms"), list) and check["terms"]):
                raise DatasetError(
                    f"{task_id}/{requirement['id']}: needs a non-empty terms list")
            if kind in ("min_words", "max_words") and not isinstance(
                    check.get("words"), int):
                raise DatasetError(
                    f"{task_id}/{requirement['id']}: {kind} needs an integer")
            metric = requirement.get("metric")
            if metric is not None and metric not in METRICS:
                raise DatasetError(
                    f"{task_id}/{requirement['id']}: unknown metric {metric!r}")
        for lesson_id in case.get("lesson_ids") or []:
            if lesson_id not in lesson_ids:
                raise DatasetError(
                    f"{task_id}: unknown lesson fixture {lesson_id!r}")
    for fixture in dataset.get("feedback") or []:
        if not isinstance(fixture, dict) or not str(fixture.get("text") or ""):
            raise DatasetError("every feedback fixture needs text")
        if fixture.get("expected_feedback_type") not in (
                None, *C.FEEDBACK_TYPES):
            raise DatasetError(
                f"feedback {fixture.get('id')!r}: unknown feedback type")
        if fixture.get("expected_scope") not in (None, "task", "workspace",
                                                "capability", "any"):
            raise DatasetError(
                f"feedback {fixture.get('id')!r}: unknown scope")
    return dataset


def load_dataset(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return validate_dataset(json.load(handle))


# --------------------------------------------------------------------------- #
# Requirement scoring
# --------------------------------------------------------------------------- #

def check_requirement(answer: str, requirement: Dict[str, Any]) -> bool:
    """Whether one declared requirement is satisfied by `answer`."""
    text = str(answer or "")
    lowered = text.lower()
    check = requirement.get("check") or {}
    kind = check.get("kind")
    if kind == "contains_all":
        return all(str(term).lower() in lowered for term in check["terms"])
    if kind == "contains_any":
        return any(str(term).lower() in lowered for term in check["terms"])
    if kind == "not_contains_all":
        return not any(str(term).lower() in lowered for term in check["terms"])
    if kind == "regex":
        try:
            return re.search(str(check["pattern"]), text) is not None
        except re.error:
            return False
    if kind == "min_words":
        return len(re.findall(r"\S+", text)) >= int(check["words"])
    if kind == "max_words":
        return len(re.findall(r"\S+", text)) <= int(check["words"])
    return False


def score_answer(answer: str,
                 requirements: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Completeness of one answer against the case's declared requirements."""
    total = len(requirements)
    satisfied: List[str] = []
    missing: List[str] = []
    for requirement in requirements:
        (satisfied if check_requirement(answer, requirement) else missing).append(
            str(requirement.get("id")))
    return {
        "total": total,
        "satisfied": len(satisfied),
        "satisfied_ids": satisfied,
        "missing_ids": missing,
        "completeness": (len(satisfied) / total) if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# Calibration audit (deterministic, no model, no execution)
# --------------------------------------------------------------------------- #

# Expected-audit field -> the actual field it is compared with. The widening
# check is reported under its own name in `actual`, so the pair is explicit
# rather than assumed to match.
_AUDIT_FIELDS = (
    ("feedback_type", "feedback_type"),
    ("impact", "impact"),
    ("scope", "scope"),
    ("learnable", "learnable"),
    ("temporary", "temporary"),
    ("overgeneralization", "blocked_for_widening"),
)


def audit_calibration(fixtures: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify feedback fixtures and compare with their expected reading.

    This is the gate's own before/after: it does not run a task, it asks what
    the learning system would do with a message. Every assertion is declared
    in the dataset, so a change in classification is visible instead of silent.
    """
    rows: List[Dict[str, Any]] = []
    for fixture in fixtures or []:
        text = str(fixture.get("text") or "")
        assessment = C.assess(
            text,
            kind=str(fixture.get("kind") or ""),
            missing=fixture.get("missing") or [],
            behavior=str(fixture.get("behavior") or ""),
        )
        widened = bool(C.blocking(assessment))
        expected_widening = fixture.get("expect_overgeneralization")
        rows.append({
            "id": fixture.get("id"),
            "text": text,
            "actual": {
                "feedback_type": assessment["feedback_type"],
                "impact": assessment["impact"],
                "scope": assessment["scope"],
                "learnable": assessment["learnable"],
                "temporary": assessment["temporary"],
                "restricts": assessment["restricts"],
                "blocked_for_widening": widened,
                "blocking_findings": [f["kind"] for f in C.blocking(assessment)],
            },
            "expected": {
                "feedback_type": fixture.get("expected_feedback_type"),
                "impact": fixture.get("expected_impact"),
                "scope": fixture.get("expected_scope"),
                "learnable": fixture.get("expected_learnable"),
                "temporary": fixture.get("expected_temporary"),
                "overgeneralization": expected_widening,
            },
        })

    def field_accuracy(name: str) -> Optional[float]:
        checked = [r for r in rows
                   if r["expected"][name] is not None]
        if not checked:
            return None
        ok = sum(1 for r in checked if r["actual"][name] == r["expected"][name])
        return round(ok / len(checked), 4)

    overgeneralization = [r for r in rows
                          if r["expected"]["overgeneralization"] is not None]
    flagged = sum(1 for r in overgeneralization
                  if r["actual"]["blocked_for_widening"]
                  == r["expected"]["overgeneralization"])
    return {
        "fixtures": rows,
        "count": len(rows),
        "accuracy": {
            "feedback_type": field_accuracy("feedback_type"),
            "impact": field_accuracy("impact"),
            "scope": field_accuracy("scope"),
            "learnable": field_accuracy("learnable"),
            "temporary": field_accuracy("temporary"),
            "overgeneralization": (round(flagged / len(overgeneralization), 4)
                                   if overgeneralization else None),
        },
        "mismatches": [
            {"id": r["id"], "field": field,
             "expected": r["expected"][field],
             "actual": r["actual"][actual_field]}
            for r in rows for field, actual_field in _AUDIT_FIELDS
            if r["expected"][field] is not None
            and r["actual"][actual_field] != r["expected"][field]
        ],
    }


# --------------------------------------------------------------------------- #
# One condition
# --------------------------------------------------------------------------- #

def evaluate_condition(case: Dict[str, Any], row: Dict[str, Any],
                       *, applied: Iterable[str] = (),
                       retrieved: Iterable[str] = (),
                       restricted: Iterable[str] = ()) -> Dict[str, Any]:
    """Describe one side of one task from its stored execution row."""
    answer = str(row.get("answer") or "")
    requirements = score_answer(answer, case.get("requirements") or [])
    qa_flags = [str(flag) for flag in (row.get("qa_flags") or [])]
    error = row.get("error") if isinstance(row.get("error"), dict) else {}
    status = str(row.get("status") or "")
    return {
        "task_id": case.get("task_id"),
        "status": status,
        "succeeded": status == "completed",
        "answer_present": bool(answer.strip()),
        "answer_chars": len(answer.strip()),
        "requirements": requirements,
        "qa": {
            "count": len(qa_flags),
            # The store records flag text, not a severity. Reporting a made-up
            # severity would be fabrication, so it is explicitly unavailable.
            "severity": "not_recorded",
            "flags": qa_flags,
            "categories": sorted({_flag_category(flag) for flag in qa_flags}),
        },
        "error_category": str(error.get("category") or "") or None,
        "failure_kind": _failure_kind(status, error, answer),
        "retrieved": sorted({str(x) for x in retrieved}),
        "applied": sorted({str(x) for x in applied}),
        "restricted": sorted({str(x) for x in (restricted or ())}),
    }


def _flag_category(flag: str) -> str:
    text = flag.strip()
    if text.lower().startswith("qa:"):
        text = text[3:].strip()
    return text.split("—")[0].split(":")[0].strip().lower() or "other"


def _failure_kind(status: str, error: Dict[str, Any], answer: str) -> Optional[str]:
    category = str(error.get("category") or "")
    if category == "timeout":
        return "timeout"
    if status == "failed":
        return category or "failed"
    if not str(answer or "").strip():
        return "empty_answer"
    if status not in ("completed", "waiting_for_approval"):
        return f"status:{status}"
    return None


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def compare_task(baseline: Dict[str, Any],
                 learning: Dict[str, Any]) -> Dict[str, Any]:
    """Diff the two conditions of one task and name every regression."""
    base_reqs = set(baseline["requirements"]["satisfied_ids"])
    learned_reqs = set(learning["requirements"]["satisfied_ids"])
    gained = sorted(learned_reqs - base_reqs)
    lost = sorted(base_reqs - learned_reqs)
    base_qa = int(baseline["qa"]["count"])
    learned_qa = int(learning["qa"]["count"])
    new_flags = [flag for flag in learning["qa"]["flags"]
                 if flag not in baseline["qa"]["flags"]]

    regressions: List[Dict[str, Any]] = []
    for requirement_id in lost:
        regressions.append({"kind": "requirement_lost",
                            "requirement_id": requirement_id,
                            "detail": "satisfied in baseline, not with learning"})
    if learned_qa > base_qa:
        regressions.append({"kind": "qa_regression",
                            "detail": f"{learned_qa - base_qa} new QA finding(s)",
                            "new_flags": new_flags})
    if baseline["succeeded"] and not learning["succeeded"]:
        regressions.append({"kind": "execution_regression",
                            "detail": f"baseline {baseline['status']} -> "
                                      f"learning {learning['status']}"})
    if baseline["answer_present"] and not learning["answer_present"]:
        regressions.append({"kind": "output_lost",
                            "detail": "the learning-enabled run produced no answer"})

    if regressions:
        outcome = "regression" if not (gained or learned_qa < base_qa) else "mixed"
    elif gained or learned_qa < base_qa:
        outcome = "improved"
    else:
        outcome = "unchanged"

    base_pct = baseline["requirements"]["completeness"]
    learned_pct = learning["requirements"]["completeness"]
    return {
        "task_id": baseline.get("task_id"),
        "outcome": outcome,
        "gained": gained,
        "lost": lost,
        "qa_delta": learned_qa - base_qa,
        "new_qa_flags": new_flags,
        "completeness_baseline": round(base_pct, 4),
        "completeness_learning": round(learned_pct, 4),
        "completeness_delta_pp": round((learned_pct - base_pct) * 100, 2),
        "applied_lessons": learning["applied"],
        # Applied lessons that impose a restriction: named separately so a
        # "better" answer that only became more cautious is visible.
        "restricted_lessons": learning.get("restricted") or [],
        # Selected by the same relevance function for the task's capability but
        # never inserted into a prompt: measurable, and NOT credited as used.
        "retrieved_not_applied": sorted(
            set(learning["retrieved"]) - set(learning["applied"])),
        "regressions": regressions,
    }


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #

def _sum_requirements(conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(c["requirements"]["total"] for c in conditions)
    satisfied = sum(c["requirements"]["satisfied"] for c in conditions)
    return {
        "requirements": total,
        "satisfied": satisfied,
        "completeness": round((satisfied / total) if total else 0.0, 4),
        "qa_findings": sum(c["qa"]["count"] for c in conditions),
        "succeeded": sum(1 for c in conditions if c["succeeded"]),
        "failed": sum(1 for c in conditions if not c["succeeded"]),
        "empty_answers": sum(1 for c in conditions
                             if not c["answer_present"]),
        "timeouts": sum(1 for c in conditions
                        if c["failure_kind"] == "timeout"),
        "applied_lessons": sorted({i for c in conditions
                                   for i in c["applied"]}),
        "retrieved_lessons": sorted({i for c in conditions
                                     for i in c["retrieved"]}),
    }


def metric_rollup(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-metric satisfaction, baseline vs learning (kept separate).

    A metric's `rate` is its UNMET share, so lower is better for every entry
    (overgeneralization, unnecessary restriction, clarification overuse, cross-
    capability regression). `initiative_retention` and `constraint_accuracy`
    are stated as requirements too, so they are read the same way.
    """
    by_metric: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        requirements = task.get("requirements_by_id") or {}
        for requirement_id, metric in requirements.items():
            entry = by_metric.setdefault(metric, {
                "metric": metric, "total": 0,
                "baseline_satisfied": 0, "learning_satisfied": 0,
                "tasks": [],
            })
            entry["total"] += 1
            entry["tasks"].append(task["task_id"])
            if requirement_id in task["baseline"]["requirements"]["satisfied_ids"]:
                entry["baseline_satisfied"] += 1
            if requirement_id in task["learning"]["requirements"]["satisfied_ids"]:
                entry["learning_satisfied"] += 1
    out = []
    for entry in by_metric.values():
        total = entry["total"] or 1
        entry["baseline_rate"] = round(
            1 - entry["baseline_satisfied"] / total, 4)
        entry["learning_rate"] = round(
            1 - entry["learning_satisfied"] / total, 4)
        out.append(entry)
    return sorted(out, key=lambda e: str(e["metric"]))


def summarize(tasks: List[Dict[str, Any]],
              calibration: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
    """Totals, per capability and per lesson, from task comparisons only.

    `calibration`, when given, adds the deterministic gate's own accuracy to
    the same report; it is never mixed into the completeness number.
    """
    base = _sum_requirements([t["baseline"] for t in tasks])
    learn = _sum_requirements([t["learning"] for t in tasks])
    diffs = [t["comparison"] for t in tasks]

    regressions = []
    for task in tasks:
        for regression in task["comparison"]["regressions"]:
            regressions.append({"task_id": task["task_id"], **regression})

    by_capability: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        entry = by_capability.setdefault(task["capability"], {
            "capability": task["capability"], "tasks": 0,
            "completeness_baseline": 0.0, "completeness_learning": 0.0,
            "improved": 0, "regressions": 0, "unchanged": 0,
        })
        entry["tasks"] += 1
        entry["completeness_baseline"] += (
            task["baseline"]["requirements"]["completeness"])
        entry["completeness_learning"] += (
            task["learning"]["requirements"]["completeness"])
        outcome = task["comparison"]["outcome"]
        if outcome in ("improved", "mixed"):
            entry["improved"] += 1
        if outcome in ("regression", "mixed"):
            entry["regressions"] += 1
        if outcome == "unchanged":
            entry["unchanged"] += 1
    for entry in by_capability.values():
        tasks_n = entry["tasks"] or 1
        entry["completeness_baseline"] = round(
            entry["completeness_baseline"] / tasks_n, 4)
        entry["completeness_learning"] = round(
            entry["completeness_learning"] / tasks_n, 4)

    lesson_stats: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        comparison = task["comparison"]
        for lesson_id in comparison["applied_lessons"]:
            entry = lesson_stats.setdefault(lesson_id, {
                "lesson_id": lesson_id, "applied_in": [],
                "improved": 0, "regressed": 0, "unchanged": 0,
            })
            entry["applied_in"].append(task["task_id"])
            if comparison["outcome"] in ("improved", "mixed"):
                entry["improved"] += 1
            if comparison["outcome"] in ("regression", "mixed"):
                entry["regressed"] += 1
            if comparison["outcome"] == "unchanged":
                entry["unchanged"] += 1
        for lesson_id in comparison["retrieved_not_applied"]:
            entry = lesson_stats.setdefault(lesson_id, {
                "lesson_id": lesson_id, "applied_in": [],
                "improved": 0, "regressed": 0, "unchanged": 0,
            })
            entry.setdefault("retrieved_not_applied_in", []).append(
                task["task_id"])

    metrics = metric_rollup(tasks)
    gate = (calibration or {}).get("accuracy") or {}
    for metric in metrics:
        name = metric["metric"]
        if name == "overgeneralization_rate" and gate.get(
                "overgeneralization") is not None:
            # The gate is measured once, not per task: report its accuracy
            # beside the behavioural rate rather than inventing a task split.
            metric["gate_accuracy"] = gate["overgeneralization"]
        if name == "constraint_accuracy" and gate.get("feedback_type") \
                is not None:
            metric["gate_accuracy"] = gate["feedback_type"]
        if name == "preference_scope_accuracy" and gate.get("scope") is not None:
            metric["gate_accuracy"] = gate["scope"]

    return {
        "tasks": len(tasks),
        "baseline": base,
        "learning": learn,
        "metrics": metrics,
        "calibration_accuracy": gate,
        "delta": {
            "completeness_pp": round(
                (learn["completeness"] - base["completeness"]) * 100, 2),
            "requirements_satisfied": learn["satisfied"] - base["satisfied"],
            "qa_findings": learn["qa_findings"] - base["qa_findings"],
            "succeeded": learn["succeeded"] - base["succeeded"],
            "empty_answers": learn["empty_answers"] - base["empty_answers"],
            "regressions": len(regressions),
            "improved_tasks": sum(1 for d in diffs
                                  if d["outcome"] in ("improved", "mixed")),
            "regressed_tasks": sum(1 for d in diffs
                                   if d["outcome"] in ("regression", "mixed")),
            "unchanged_tasks": sum(1 for d in diffs
                                   if d["outcome"] == "unchanged"),
        },
        "regressions": regressions,
        "by_capability": sorted(by_capability.values(),
                                key=lambda e: str(e["capability"])),
        "lessons": sorted(lesson_stats.values(),
                          key=lambda e: str(e["lesson_id"])),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(dataset: Dict[str, Any], tasks: List[Dict[str, Any]],
                 *, mode: str, provider: Dict[str, Any],
                 started_at: str, finished_at: str,
                 dataset_path: Optional[str] = None,
                 notes: Optional[Iterable[str]] = None,
                 calibration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the full machine-readable report from real task results."""
    metric_by_requirement = {
        str(case.get("task_id")): {
            str(req.get("id")): req.get("metric")
            for req in case.get("requirements") or [] if req.get("metric")
        }
        for case in dataset.get("cases") or []
    }
    enriched = [
        {**task, "requirements_by_id": metric_by_requirement.get(
            str(task.get("task_id")), {})}
        for task in tasks
    ]
    summary = summarize(enriched, calibration)
    return {
        "harness_version": HARNESS_VERSION,
        "benchmark_version": dataset.get("version"),
        "benchmark_path": dataset_path,
        "mode": mode,
        "live_model": False,
        "provider": provider,
        "started_at": started_at,
        "finished_at": finished_at,
        "task_count": len(tasks),
        "summary": summary,
        "tasks": enriched,
        "calibration": calibration,
        "notes": list(notes or []),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: Dict[str, Any]) -> str:
    """Human-readable summary generated from the same data as the JSON."""
    summary = report["summary"]
    base = summary["baseline"]
    learn = summary["learning"]
    delta = summary["delta"]
    lines = [
        f"# Learning evaluation — {report['benchmark_version']}",
        "",
        f"- Mode: **{report['mode']}** (live model: "
        f"{'yes' if report['live_model'] else 'no'})",
        f"- Provider: {report['provider'].get('description', 'unknown')}",
        f"- Run: {report['started_at']} → {report['finished_at']}",
        f"- Tasks: {report['task_count']} (each executed twice)",
        "",
        "| Metric | Baseline | Learning enabled | Difference |",
        "|---|---:|---:|---:|",
        f"| Requirement completeness | {_pct(base['completeness'])} | "
        f"{_pct(learn['completeness'])} | {delta['completeness_pp']:+.1f} pp |",
        f"| Requirements satisfied | {base['satisfied']}/{base['requirements']} | "
        f"{learn['satisfied']}/{learn['requirements']} | "
        f"{delta['requirements_satisfied']:+d} |",
        f"| QA findings | {base['qa_findings']} | {learn['qa_findings']} | "
        f"{delta['qa_findings']:+d} |",
        f"| Successful executions | {base['succeeded']} | {learn['succeeded']} | "
        f"{delta['succeeded']:+d} |",
        f"| Empty answers | {base['empty_answers']} | {learn['empty_answers']} | "
        f"{delta['empty_answers']:+d} |",
        f"| Regressions | — | {delta['regressions']} | +{delta['regressions']} |",
        "",
        f"Improved tasks: {delta['improved_tasks']} · regressed: "
        f"{delta['regressed_tasks']} · unchanged: {delta['unchanged_tasks']}",
        "",
        "## Per task",
        "",
        "| Task | Capability | Baseline | Learning | Difference | Outcome |",
        "|---|---|---:|---:|---:|---|",
    ]
    for task in report["tasks"]:
        comparison = task["comparison"]
        lines.append(
            f"| {task['task_id']} | {task['capability']} | "
            f"{_pct(comparison['completeness_baseline'])} | "
            f"{_pct(comparison['completeness_learning'])} | "
            f"{comparison['completeness_delta_pp']:+.1f} pp | "
            f"{comparison['outcome']} |")

    if summary.get("metrics"):
        lines += ["", "## Behaviour metrics", "",
                  "| Metric | Baseline unmet | Learning unmet | Gate accuracy | "
                  "Tasks |",
                  "|---|---:|---:|---:|---|"]
        for metric in summary["metrics"]:
            gate = metric.get("gate_accuracy")
            lines.append(
                f"| {metric['metric']} | {_pct(metric['baseline_rate'])} | "
                f"{_pct(metric['learning_rate'])} | "
                f"{_pct(gate) if gate is not None else '—'} | "
                f"{len(metric['tasks'])} |")
        lines += ["", "A rate is the share of that metric's requirements that "
                    "were NOT met, so lower is better. Gate accuracy is the "
                    "deterministic classifier's agreement with the dataset's "
                    "declared expectation; it is not a model score."]

    if summary["regressions"]:
        lines += ["", "## Regressions", ""]
        for regression in summary["regressions"]:
            detail = regression.get("requirement_id") or regression.get("detail")
            lines.append(f"- **{regression['task_id']}** "
                         f"({regression['kind']}): {detail}")

    lines += ["", "## Lessons", ""]
    if not summary["lessons"]:
        lines.append("No lesson was applied to any task.")
    else:
        lines += ["| Lesson | Applied in | Improved | Regressed |",
                  "|---|---|---:|---:|"]
        for lesson in summary["lessons"]:
            used = ", ".join(lesson["applied_in"]) or "—"
            lines.append(f"| {lesson['lesson_id']} | {used} | "
                         f"{lesson['improved']} | {lesson['regressed']} |")

    lines += ["", "## Caveats", ""]
    lines += [f"- {note}" for note in report.get("notes") or []]
    lines.append("")
    return "\n".join(lines)
