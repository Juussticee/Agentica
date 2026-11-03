"""A/B benchmark harness for the learning loop (2026-10-30).

Each benchmark case is executed TWICE through the REAL GraphRunner on its own
throwaway store, with identical input and provider configuration:

  baseline   GraphRunner(lessons_enabled=False) — no lesson set is supplied.
             Nothing stored is disabled or deleted.
  learning   GraphRunner() — the normal path: active seeded lessons are
             selected by the real relevance function and inserted by the real
             nodes.

Both conditions run inside `writes_disabled()`, so a measured run cannot
create a lesson, a skill version or a feedback record; the lesson id set is
also asserted unchanged afterwards.

The provider is a SCRIPTED DOUBLE and the mode is labelled
`offline_deterministic`. It answers with the case's `baseline_answer`,
appends the learned requirement text it was given, and obeys an explicit
one-sentence directive. No live model, no network.
"""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.execution.runner import GraphRunner
from app.execution.store import ExecutionStore
from app.learning import evaluation
from app.learning import retrieval
from app.learning import writes_disabled

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# v2 is the current benchmark: it keeps v1's six tasks and adds the behavioural
# drift cases plus the classification fixtures. v1 stays in the repository so
# earlier reports keep their meaning (they name their own version).
DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "learning_eval_v2.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "storage" / "learning_eval"

LEARNED_MARKER = "Learned requirements for this kind of task"
BREVITY_RE = re.compile(r"\b(single short sentence|one sentence|be brief)\b",
                        re.IGNORECASE)



class BenchmarkModel:
    """Scripted provider double: deterministic, offline, no network.

    It reacts ONLY to the prompt it is given, exactly like a model would:
      * no learned block  -> the case's baseline answer;
      * learned block     -> baseline answer + the learned requirement text;
      * a one-sentence directive -> a single short sentence (obeyed).
    The question is matched against the dataset so a shared double can serve
    every case without hidden per-case state.
    """

    def __init__(self, cases):
        self.cases = list(cases)
        self.prompts = []

    def _case_for(self, prompt: str):
        for case in self.cases:
            if case["request"] in prompt:
                return case
        return None

    @staticmethod
    def _behaviours(prompt: str):
        if LEARNED_MARKER not in prompt:
            return []
        block = prompt.split(LEARNED_MARKER, 1)[1]
        found = []
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("- [") or "]" not in line:
                continue
            behaviour = line.split("]", 1)[1].strip()
            behaviour = re.sub(r"\s*\(check:[^)]*\)\s*$", "", behaviour)
            if behaviour:
                found.append(behaviour)
        return found

    def invoke(self, prompt):
        text = str(prompt)
        self.prompts.append(text)
        case = self._case_for(text)
        if case is None:
            # Any non-benchmark prompt (planner, feedback analysis) gets a
            # neutral deterministic reply; the benchmark never depends on it.
            return SimpleNamespace(content="not applicable")
        behaviours = self._behaviours(text)
        if not behaviours:
            return SimpleNamespace(content=case["baseline_answer"])
        if any(BREVITY_RE.search(behaviour) for behaviour in behaviours):
            return SimpleNamespace(content="Short answer.")
        answer = case["baseline_answer"] + "\n\n" + "\n".join(behaviours)
        return SimpleNamespace(content=answer)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_lessons(store, dataset, lesson_ids=None) -> list:
    """Insert the dataset's lesson fixtures as ACTIVE lessons (temp store).

    `lesson_ids` limits the seeds to the fixtures a case declares, so a task's
    two conditions are measured in that task's own lesson environment. Cases
    therefore cannot leak a conflicting preference into each other.
    """
    wanted = set(lesson_ids) if lesson_ids is not None else None
    ids = []
    for lesson in dataset["lessons"]:
        if wanted is not None and lesson["lesson_id"] not in wanted:
            continue
        lesson_id = lesson["lesson_id"]
        # Calibration fields are part of the fixture: a drift case needs a
        # lesson that is honestly labelled a restriction (or a one-off), not a
        # restriction hidden inside a neutral row.
        scope = lesson.get("scope") or "capability"
        temporary = bool(lesson.get("temporary"))
        # A fixture may declare a non-active status: a lesson that the real
        # loop would hold for review (a temporary instruction, a restriction
        # that contradicts a standing preference) must be seeded as stored so
        # the benchmark can prove it is NOT applied.
        status = str(lesson.get("status") or "active")
        calibration = {
            "feedback_type": lesson.get("feedback_type")
                             or "general_preference",
            "impact": lesson.get("impact") or "preference_adjustment",
            "restricts": bool(lesson.get("restricts")),
            "learnable": True,
            "temporary": temporary,
            "scope": scope,
            "scope_detail": {
                "kind": ("single_task" if temporary else
                         "project" if scope == "workspace" else "standing"),
                "capability": lesson.get("capability") or "any",
                "conditions": list(lesson.get("conditions") or []),
                "narrow_object": bool(lesson.get("narrow_object")),
            },
            "overgeneralization": list(
                lesson.get("overgeneralization") or []),
            "review_reasons": list(lesson.get("review_reasons") or []),
        }
        store.create_lesson({
            "lesson_id": lesson_id,
            "status": status,
            "category": lesson.get("category") or "preference",
            "capability": lesson.get("capability") or "any",
            "scope": lesson.get("scope") or "capability",
            "problem": lesson.get("problem") or "benchmark fixture",
            "root_cause": lesson.get("root_cause"),
            "corrective_behavior": lesson["corrective_behavior"],
            "validation": lesson.get("validation"),
            "confidence": float(lesson.get("confidence") or 0.8),
            "evidence": [{"kind": "benchmark_fixture",
                          "at": _now(),
                          "keywords": lesson.get("keywords") or []}],
            "source_execution_id": None,
            "source_text": None,
            "origin": "user_feedback",
            "feedback_type": calibration["feedback_type"],
            "impact": calibration["impact"],
            "temporary": temporary,
            "calibration": calibration,
        })
        ids.append(lesson_id)
    return ids


def _applied_ids(store, execution_id: str) -> list:
    for event in store.get_events(execution_id):
        if event.get("type") != "learning.lessons_applied":
            continue
        payload = event.get("payload") or {}
        return [str(x) for x in payload.get("lesson_ids") or []]
    return []


def run_case(dataset, case, model, store_dir) -> dict:
    """Run ONE task in both conditions on its own throwaway store.

    A store per task keeps the two conditions fair (identical input, provider
    and lesson environment) and stops one task's lessons from leaking into
    another's measurement.
    """
    store = ExecutionStore(Path(store_dir) / "learning_eval.sqlite3")
    try:
        seed_lessons(store, dataset, case.get("lesson_ids") or [])
        before = {l["lesson_id"] for l in store.list_lessons()}
        baseline_runner = GraphRunner(store, lessons_enabled=False)
        learning_runner = GraphRunner(store)
        active = store.list_lessons(status="active")

        with mock.patch("app.llm_client.get_llm", return_value=model):
            with writes_disabled("learning benchmark"):
                baseline_row = baseline_runner.run(
                    baseline_runner.create(case["request"])["id"])
                learning_row = learning_runner.run(
                    learning_runner.create(case["request"])["id"])

        after = {l["lesson_id"] for l in store.list_lessons()}
        contaminated = sorted(after - before)
        if contaminated:
            raise RuntimeError(
                f"{case['task_id']}: benchmark run created lessons (learning "
                "state contaminated): " + ", ".join(contaminated))

        # Retrieved: the SAME relevance function the nodes use, for the task's
        # capability. Applied: the execution's own event, never inferred.
        capability = str(case["capability"])
        retrieved = [
            lesson.get("lesson_id")
            for lesson in retrieval.select(active, capability, case["request"])
        ]
        applied = _applied_ids(store, learning_row["id"])
        restricting = {
            lesson["lesson_id"] for lesson in active
            if (lesson.get("calibration") or {}).get("impact")
            in ("constraint", "safety_requirement")
        }

        baseline = evaluation.evaluate_condition(case, baseline_row)
        learning = evaluation.evaluate_condition(
            case, learning_row, applied=applied, retrieved=retrieved,
            restricted=[i for i in applied if i in restricting])
        return {
            "task_id": case["task_id"],
            "capability": capability,
            "request": case["request"],
            "lesson_fixtures": list(case.get("lesson_ids") or []),
            "baseline_execution_id": baseline_row["id"],
            "learning_execution_id": learning_row["id"],
            "baseline": baseline,
            "learning": learning,
            "comparison": evaluation.compare_task(baseline, learning),
            "expected_outcome": case.get("expected_outcome"),
        }
    finally:
        store.close()


def run_benchmark(dataset, *, store_dir=None, dataset_path=None,
                  model=None) -> dict:
    """Execute every case in both conditions and build the report."""
    started = _now()
    model = model or BenchmarkModel(dataset["cases"])
    tasks = []
    with tempfile.TemporaryDirectory(prefix="agentica_eval_") as temp:
        root = Path(store_dir) if store_dir else Path(temp)
        root.mkdir(parents=True, exist_ok=True)
        for index, case in enumerate(dataset["cases"]):
            case_dir = root / f"{index:02d}_{case['task_id']}"
            case_dir.mkdir(parents=True, exist_ok=True)
            tasks.append(run_case(dataset, case, model, case_dir))

    calibration = evaluation.audit_calibration(dataset.get("feedback") or [])
    summary = evaluation.summarize(tasks)
    fixture_ids = ", ".join(
        sorted({l["lesson_id"] for l in dataset["lessons"]}))
    notes = [
        "Mode is offline_deterministic: the provider is a scripted double, so "
        "these numbers measure the harness and the learning loop, not the "
        "quality of a live model. A live-model run is a separate labelled mode.",
        "Requirements are checked on stored execution answers, never on prompts.",
        f"Lesson fixtures available in the dataset: {fixture_ids}. Each task "
        "runs on its own store seeded only with the fixtures it declares, so "
        "one task's lessons cannot change another's measurement.",
        "Benchmark runs execute with learning writes disabled; the lesson set "
        "is verified unchanged after the run.",
        "Behaviour metrics are reported separately (never summed into one "
        "score): a task that 'improved' while restricting unrelated work is a "
        "failure, not a win.",
    ]
    if summary["delta"]["regressions"]:
        notes.append(
            "Regressions are reported, not hidden: a lesson that was applied "
            "and lost a requirement is evidence AGAINST that lesson's current "
            "wording. Nothing is changed automatically.")
    report = evaluation.build_report(
        dataset, tasks, mode="offline_deterministic", calibration=calibration,
        provider={"description": "scripted benchmark double (offline, no "
                                 "network, no live model)",
                  "live": False},
        started_at=started, finished_at=_now(),
        dataset_path=str(dataset_path) if dataset_path else None,
        notes=notes,
    )
    return report


def write_report(report, out_dir: Path, *, print_markdown=False) -> tuple:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{report['benchmark_version']}_{stamp}_{uuid.uuid4().hex[:6]}"
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    markdown = evaluation.render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    if print_markdown:
        print(markdown)
    return json_path, md_path


