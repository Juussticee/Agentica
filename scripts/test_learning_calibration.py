"""Learning calibration — safety, scope and overcorrection (2026-10-30).

What this suite proves, in the order the guard runs:

  calibration   what KIND of feedback a message is (specific correction,
                contextual/general preference, task constraint, temporary
                instruction, frustration, quality feedback, safety request),
                what impact it has, how wide it may apply, and whether the
                proposed behaviour reads WIDER than the user's own words
  policy        a temporary instruction never becomes active, a restriction is
                held for review, frustration is not learned at all, repeated
                criticism escalates investigation without widening behaviour
  conflicts     a contextual preference does not silently replace a standing
                one; a repeat never demotes an active lesson
  retrieval     a one-off lesson is never selected for a later task
  evaluation    the deterministic gate's accuracy and the behavioural drift
                metrics are computed from the dataset, separately from
                completeness

No model is called: `llm_call` is injected, and the analysis text is fixed.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.execution.store import ExecutionStore  # noqa: E402
from app.learning import calibration as C  # noqa: E402
from app.learning import evaluation  # noqa: E402
from app.learning import lessons as L  # noqa: E402
from app.learning import retrieval, signals  # noqa: E402

V2_DATASET = PROJECT_ROOT / "benchmarks" / "learning_eval_v2.json"

NARROW = "Don't modify this file, I need it exactly as it is."


def analysis(behavior, problem="the earlier work was not right",
             validation="the requirement is visibly met"):
    return (f"PROBLEM: {problem}\nCAUSE: unknown\nBEHAVIOUR: {behavior}\n"
            f"VALIDATION: {validation}\nKEYWORDS: file, change, task")


class StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ExecutionStore(Path(self._tmp.name) / "calibration.sqlite3")
        self.addCleanup(self.store.close)

    def record(self, text, behavior, *, capability="", kind="", llm=None,
               problem=None):
        signal = signals.detect(text, capability=capability)
        self.assertIsNotNone(signal, f"not detected as feedback: {text!r}")
        if kind:
            signal = signal.__class__(**{**signal.__dict__, "kind": kind})
        payload = (analysis(behavior) if problem is None
                   else analysis(behavior, problem=problem))
        return L.record(self.store, signal, source_execution_id="ex-1",
                        llm_call=llm or (lambda _p: payload))


class TestClassification(unittest.TestCase):
    def test_narrow_file_restriction_is_a_task_constraint(self):
        a = C.assess(NARROW, kind="constraint",
                     behavior="Leave this file unchanged; edit a copy.")
        self.assertEqual(a["feedback_type"], C.TASK_CONSTRAINT)
        self.assertEqual(a["impact"], C.CONSTRAINT)
        self.assertEqual(a["scope"], C.SCOPE_WORKSPACE)
        self.assertTrue(a["restricts"])
        self.assertEqual(C.blocking(a), [])

    def test_the_same_message_with_a_blanket_behaviour_is_blocked(self):
        a = C.assess(NARROW, kind="constraint",
                     behavior="Never modify any file in the repository.")
        kinds = [f["kind"] for f in C.blocking(a)]
        self.assertIn("restriction_broadened", kinds)

    def test_temporary_instruction_is_scoped_to_the_task(self):
        a = C.assess("Just this once, answer in bullet points.", kind="format",
                     behavior="Answer in bullet points.")
        self.assertEqual(a["feedback_type"], C.TEMPORARY_INSTRUCTION)
        self.assertTrue(a["temporary"])
        self.assertEqual(a["scope"], C.SCOPE_TASK)
        self.assertEqual(a["scope_detail"]["kind"], "single_task")
        self.assertIn("temporary_made_permanent",
                      [f["kind"] for f in C.blocking(a)])

    def test_frustration_alone_is_not_learnable(self):
        a = C.assess("This is useless, you're not helping at all.",
                     kind="frustration")
        self.assertEqual(a["feedback_type"], C.FRUSTRATION)
        self.assertFalse(a["learnable"])

    def test_frustration_with_a_named_requirement_is_learnable(self):
        a = C.assess(
            "This is useless — I needed the original dates preserved.",
            kind="insufficient", missing=["the original dates preserved"])
        self.assertTrue(a["learnable"])
        self.assertNotEqual(a["feedback_type"], C.FRUSTRATION)

    def test_repeated_criticism_escalates_instead_of_widening(self):
        a = C.assess("You made the same mistake again with the comparison.",
                     kind="repeated")
        self.assertEqual(a["escalate"], "investigate_recurring_failure")
        self.assertFalse(a["restricts"])
        self.assertEqual(a["impact"], C.QUALITY)

    def test_a_safety_request_is_never_a_plain_preference(self):
        a = C.assess("Be careful with production changes from now on.",
                     kind="future_directive",
                     behavior="Ask before touching production.")
        self.assertEqual(a["feedback_type"], C.SAFETY_CONSTRAINT)
        self.assertEqual(a["impact"], C.SAFETY)
        self.assertTrue(a["restricts"])

    def test_a_capability_scope_never_widens_to_every_task(self):
        a = C.assess("The research report skipped licensing.", kind="insufficient",
                     missing=["licensing"],
                     behavior="Cite licensing for every researched option.",
                     capability="any")
        self.assertIn("capability_widened",
                      [f["kind"] for f in C.blocking(a)])

    def test_restriction_without_a_user_request_is_flagged(self):
        a = C.assess("The answer was too long.", kind="format",
                     behavior="Never answer in more than one sentence.")
        self.assertIn("unrequested_restriction",
                      [f["kind"] for f in C.blocking(a)])

    def test_a_quality_requirement_is_not_a_restriction(self):
        a = C.assess("These results aren't enough, I need benchmarks.",
                     kind="insufficient", missing=["benchmarks"],
                     behavior="Always include a benchmark number.")
        self.assertFalse(a["restricts"])
        self.assertEqual(C.blocking(a), [])


class TestActivationPolicy(StoreCase):
    def test_a_narrow_restriction_is_stored_but_not_applied(self):
        outcome = self.record(NARROW, "Leave this file unchanged; edit a copy.",
                              capability="coding")
        self.assertEqual(outcome["lesson"]["status"], L.CANDIDATE)
        self.assertFalse(outcome["decision"]["auto_applied"])
        self.assertIn("restriction", outcome["decision"]["decision_reason"])
        self.assertEqual(L.active_lessons(self.store, "coding"), [])

    def test_a_temporary_instruction_is_never_active(self):
        outcome = self.record(
            "Just this once, answer in bullet points.",
            "For this task only, answer in bullet points.", capability="general")
        lesson = outcome["lesson"]
        self.assertEqual(lesson["status"], L.CANDIDATE)
        self.assertTrue(lesson["temporary"])
        self.assertIn("temporary", outcome["decision"]["decision_reason"])
        self.assertEqual(retrieval.select([lesson], "general", "anything"), [])

    def test_frustration_records_nothing_at_all(self):
        outcome = self.record("This is useless, you're not helping at all.",
                              "", capability="general")
        self.assertEqual(outcome["action"], "not_learned")
        self.assertEqual(self.store.list_lessons(), [])

    def test_repeating_a_held_lesson_does_not_activate_it(self):
        first = self.record(NARROW, "Never modify any file in the repository.",
                            capability="coding")
        second = self.record(NARROW, "Never modify any file in the repository.",
                             capability="coding")
        self.assertEqual(second["action"], "merged")
        stored = self.store.get_lesson(first["lesson"]["lesson_id"])
        self.assertEqual(stored["status"], L.CANDIDATE)
        self.assertEqual(len(stored["evidence"]), 2)

    def test_a_repeat_does_not_demote_an_active_lesson(self):
        outcome = self.record(
            "These results aren't enough, I need benchmarks.",
            "Include a benchmark number.", capability="research")
        self.assertEqual(outcome["lesson"]["status"], L.ACTIVE)
        # The same correction again, now analysed into a restriction: the
        # active lesson must keep working.
        again = L.record(
            self.store,
            signals.detect("These results aren't enough, I need benchmarks.",
                           capability="research"),
            source_execution_id="ex-2",
            llm_call=lambda _p: analysis("Never answer without a benchmark."))
        self.assertEqual(again["action"], "merged")
        self.assertEqual(again["lesson"]["status"], L.ACTIVE)

    def test_calibration_is_stored_with_the_lesson(self):
        outcome = self.record(
            "From now on always cite a source for every claim.",
            "Cite a source for every factual claim.", capability="research")
        lesson = self.store.get_lesson(outcome["lesson"]["lesson_id"])
        self.assertEqual(lesson["feedback_type"], C.GENERAL_PREFERENCE)
        self.assertEqual(lesson["impact"], C.PREFERENCE)
        self.assertFalse(lesson["temporary"])
        self.assertTrue(lesson["calibration"])
        self.assertEqual(lesson["calibration"]["scope"], "capability")
        self.assertEqual(lesson["calibration"]["scope_detail"]["kind"],
                         "standing")


class TestConflicts(StoreCase):
    def test_a_contextual_preference_does_not_replace_a_standing_one(self):
        standing = self.record(
            "From now on always cite a source for every claim.",
            "Cite a source for every factual claim.", capability="research")
        self.assertEqual(standing["lesson"]["status"], L.ACTIVE)
        contextual = self.record(
            "In this project, use bullets instead of citations in summaries.",
            "Skip citations in this project.", capability="research")
        self.assertEqual(contextual["lesson"]["status"], L.CANDIDATE)
        self.assertEqual(
            self.store.get_lesson(standing["lesson"]["lesson_id"])["status"],
            L.ACTIVE)
        self.assertEqual(
            contextual["lesson"]["calibration"].get("conflict_with"),
            [standing["lesson"]["lesson_id"]])

    def test_a_standing_preference_still_supersedes_a_contradicting_one(self):
        self.record("Next time always use a table for comparisons.",
                    "Use a markdown table.", capability="research")
        newer = self.record(
            "Actually, next time use bullets for comparisons instead.",
            "Use bullet points.", capability="research")
        self.assertEqual(newer["action"], "superseded")
        statuses = [l["status"] for l in self.store.list_lessons()]
        self.assertEqual(statuses.count(L.ACTIVE), 1)
        self.assertEqual(statuses.count(L.SUPERSEDED), 1)

    def test_an_unrelated_preference_is_left_alone(self):
        kept = self.record("Next time always include benchmarks.",
                           "Include a benchmark number.", capability="research",
                           problem="the answer had no benchmark figures")
        self.record("Next time always use a table for comparisons.",
                    "Use a markdown table.", capability="research",
                    problem="the comparison was unreadable prose")
        self.assertEqual(
            self.record("Actually, next time use bullets for comparisons.",
                        "Use bullet points.", capability="research",
                        problem="the comparison was unreadable prose")["action"],
            "superseded")
        self.assertEqual(
            self.store.get_lesson(kept["lesson"]["lesson_id"])["status"],
            L.ACTIVE)


class TestRetrievalScope(StoreCase):
    def test_a_one_off_lesson_is_never_selected_later(self):
        lesson = self.record(
            "Just this once, answer in bullet points, this is a one-off.",
            "Answer in bullet points for this task only.",
            capability="general")["lesson"]
        self.assertEqual(lesson["status"], L.CANDIDATE)
        self.assertEqual(retrieval.select([lesson], "general", "anything"), [])
        self.assertEqual(retrieval.notes({"lessons": [lesson]}, "general",
                                        "anything"), "")

    def test_a_scoped_lesson_is_selected_for_its_capability_only(self):
        lesson = self.record(
            "This research report is not enough — I needed licensing.",
            "Cite licensing terms for every researched option.",
            capability="research")["lesson"]
        self.assertEqual(lesson["status"], L.ACTIVE)
        self.assertIn("licensing",
                      retrieval.notes({"lessons": [lesson]}, "research",
                                      "compare options").lower())
        self.assertEqual(
            retrieval.notes({"lessons": [lesson]}, "coding", "patch a file"),
            "")


class TestSchemaRoundTrip(StoreCase):
    def test_calibration_columns_persist_and_decode(self):
        lesson = self.record("This is not enough, I need benchmarks.",
                             "Include a benchmark number.",
                             capability="research")["lesson"]
        rows = self.store.list_lessons()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["impact"], C.QUALITY)
        self.assertEqual(rows[0]["temporary"], 0)
        self.assertIsInstance(rows[0]["calibration"], dict)
        updated = self.store.update_lesson(
            lesson["lesson_id"],
            calibration={"feedback_type": "quality_feedback", "restricts": True})
        self.assertTrue(updated["calibration"]["restricts"])

    def test_a_corrupt_calibration_blob_does_not_break_the_row(self):
        lesson = self.record("This is not enough, I need benchmarks.",
                             "Include a benchmark number.",
                             capability="research")["lesson"]
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE lessons SET calibration = 'not json' WHERE lesson_id = ?",
                (lesson["lesson_id"],))
            self.store._conn.commit()
        row = self.store.get_lesson(lesson["lesson_id"])
        self.assertEqual(row["calibration"], {})


class TestGateAudit(unittest.TestCase):
    def test_a_broad_behaviour_for_a_narrow_message_is_caught(self):
        audit = evaluation.audit_calibration([
            {"id": "narrow", "text": NARROW, "kind": "constraint",
             "behavior": "Leave this file unchanged; edit a copy.",
             "expected_feedback_type": "task_constraint",
             "expected_impact": "constraint", "expected_scope": "workspace",
             "expected_learnable": True, "expect_overgeneralization": False},
            {"id": "broad", "text": NARROW, "kind": "constraint",
             "behavior": "Never modify any file in the repository.",
             "expected_feedback_type": "task_constraint",
             "expected_impact": "constraint", "expected_scope": "workspace",
             "expected_learnable": True, "expect_overgeneralization": True},
        ])
        self.assertEqual(audit["mismatches"], [])
        self.assertEqual(audit["accuracy"]["overgeneralization"], 1.0)

    def test_a_wrong_expectation_is_reported_not_hidden(self):
        audit = evaluation.audit_calibration([
            {"id": "bad", "text": "This is useless, you're not helping.",
             "kind": "frustration",
             "expected_feedback_type": "quality_feedback",
             "expected_learnable": False},
        ])
        self.assertEqual(audit["accuracy"]["feedback_type"], 0.0)
        self.assertEqual(audit["mismatches"][0]["field"], "feedback_type")

    def test_the_shipped_v2_dataset_audits_cleanly(self):
        dataset = evaluation.load_dataset(V2_DATASET)
        audit = evaluation.audit_calibration(dataset["feedback"])
        self.assertEqual(audit["mismatches"], [], audit["mismatches"])
        for field, value in audit["accuracy"].items():
            with self.subTest(field=field):
                self.assertEqual(value, 1.0)


class TestDriftMetrics(unittest.TestCase):
    def _task(self, task_id, *, baseline_ok, learning_ok, metric):
        def condition(ok):
            return {"requirements": {
                "satisfied_ids": ["r"] if ok else [],
                "missing_ids": [] if ok else ["r"]}}

        return {"task_id": task_id, "capability": "general",
                "baseline": condition(baseline_ok),
                "learning": condition(learning_ok),
                "requirements_by_id": {"r": metric}}

    def test_metrics_are_separate_and_read_as_unmet_shares(self):
        rollup = evaluation.metric_rollup([
            self._task("a", baseline_ok=True, learning_ok=True,
                       metric="initiative_retention"),
            self._task("b", baseline_ok=True, learning_ok=False,
                       metric="initiative_retention"),
            self._task("c", baseline_ok=True, learning_ok=False,
                       metric="unnecessary_restriction_rate"),
        ])
        by_name = {m["metric"]: m for m in rollup}
        self.assertEqual(by_name["initiative_retention"]["baseline_rate"], 0.0)
        self.assertEqual(by_name["initiative_retention"]["learning_rate"], 0.5)
        self.assertEqual(
            by_name["unnecessary_restriction_rate"]["learning_rate"], 1.0)
        self.assertNotIn("completeness", by_name)

    def test_the_v2_dataset_declares_nearly_every_documented_metric(self):
        dataset = evaluation.load_dataset(V2_DATASET)
        declared = {req.get("metric")
                    for case in dataset["cases"]
                    for req in case["requirements"] if req.get("metric")}
        # overgeneralization_rate is a property of the GATE, measured by
        # `audit_calibration`, not of a task answer.
        self.assertEqual(declared, set(evaluation.METRICS)
                         - {"overgeneralization_rate"})
        audit = evaluation.audit_calibration(dataset["feedback"])
        self.assertIsNotNone(audit["accuracy"]["overgeneralization"])

    def test_every_documented_metric_has_a_task(self):
        dataset = evaluation.load_dataset(V2_DATASET)
        tasks = [c for c in dataset["cases"]
                 if any(r.get("metric") for r in c["requirements"])]
        self.assertGreaterEqual(len(tasks), 6)

    def test_the_dataset_validator_rejects_an_unknown_metric(self):
        dataset = json.loads(V2_DATASET.read_text(encoding="utf-8"))
        dataset["cases"][0]["requirements"][0]["metric"] = "vibes"
        with self.assertRaises(evaluation.DatasetError):
            evaluation.validate_dataset(dataset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
