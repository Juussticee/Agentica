"""Learning evaluation harness — offline, deterministic (2026-10-30).

Two layers are tested separately:

  * pure scoring/comparison (`app.learning.evaluation`) with synthetic rows, so
    every rule (completeness, QA delta, regression kinds, per-lesson stats) is
    pinned independent of the graph;
  * the real A/B run (`app.learning.benchmark`) through the actual
    GraphRunner with a scripted provider double — no network, no live model.

The end-to-end tests assert the causal chain the benchmark exists to measure:
the same request, run without a lesson set, scores lower than the same request
run with the applicable lesson, and a conflicting lesson is reported as a
REGRESSION rather than an improvement.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.execution.runner import GraphRunner  # noqa: E402
from app.execution.store import ExecutionStore  # noqa: E402
from app.learning import benchmark  # noqa: E402
from app.learning import evaluation  # noqa: E402
from app.learning import writes_disabled  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "benchmarks" / "learning_eval_v1.json"


def requirement(kind, **kwargs):
    return {"id": kwargs.pop("id", "req"), "check": {"kind": kind, **kwargs}}


def condition(answer="", *, status="completed", qa=(), error=None):
    return {
        "task_id": "t1",
        "status": status,
        "succeeded": status == "completed",
        "answer_present": bool(str(answer).strip()),
        "answer_chars": len(str(answer).strip()),
        "requirements": evaluation.score_answer(
            answer, [requirement("contains_all", id="a", terms=["alpha"]),
                     requirement("contains_all", id="b", terms=["beta"])]),
        "qa": {"count": len(qa), "severity": "not_recorded",
               "flags": list(qa), "categories": []},
        "error_category": (error or {}).get("category"),
        "failure_kind": None,
        "retrieved": [],
        "applied": [],
    }


class TestDataset(unittest.TestCase):
    def base(self):
        return {
            "version": "v-test",
            "lessons": [{"lesson_id": "l1", "corrective_behavior": "Do X."}],
            "cases": [{
                "task_id": "c1", "capability": "general", "request": "q",
                "baseline_answer": "a", "lesson_ids": ["l1"],
                "requirements": [requirement("contains_all", terms=["alpha"])],
            }],
        }

    def test_the_shipped_dataset_is_valid(self):
        dataset = evaluation.load_dataset(DATASET_PATH)
        self.assertEqual(dataset["version"], "learning_eval_v1")
        self.assertGreaterEqual(len(dataset["cases"]), 5)
        self.assertTrue(all(case["requirements"] for case in dataset["cases"]))

    def test_unknown_lesson_fixture_is_rejected(self):
        dataset = self.base()
        dataset["cases"][0]["lesson_ids"] = ["nope"]
        with self.assertRaises(evaluation.DatasetError):
            evaluation.validate_dataset(dataset)

    def test_unsupported_check_is_rejected(self):
        dataset = self.base()
        dataset["cases"][0]["requirements"][0]["check"] = {"kind": "vibes"}
        with self.assertRaises(evaluation.DatasetError):
            evaluation.validate_dataset(dataset)

    def test_duplicate_task_ids_are_rejected(self):
        dataset = self.base()
        dataset["cases"].append(dict(dataset["cases"][0]))
        with self.assertRaises(evaluation.DatasetError):
            evaluation.validate_dataset(dataset)

    def test_missing_baseline_answer_is_rejected(self):
        dataset = self.base()
        dataset["cases"][0]["baseline_answer"] = ""
        with self.assertRaises(evaluation.DatasetError):
            evaluation.validate_dataset(dataset)


class TestScoring(unittest.TestCase):
    def test_requirement_kinds(self):
        self.assertTrue(evaluation.check_requirement(
            "Alpha and BETA", requirement("contains_all", terms=["alpha", "beta"])))
        self.assertFalse(evaluation.check_requirement(
            "alpha", requirement("contains_all", terms=["alpha", "beta"])))
        self.assertTrue(evaluation.check_requirement(
            "only alpha", requirement("contains_any", terms=["alpha", "gamma"])))
        self.assertFalse(evaluation.check_requirement(
            "alpha present", requirement("not_contains_all", terms=["alpha"])))
        self.assertTrue(evaluation.check_requirement(
            "Recommendation: pick A", requirement(
                "regex", pattern="(?im)^recommendation:")))
        self.assertTrue(evaluation.check_requirement(
            "one two three", requirement("min_words", words=3)))
        self.assertFalse(evaluation.check_requirement(
            "one two", requirement("min_words", words=3)))

    def test_completeness_is_satisfied_over_total(self):
        scored = evaluation.score_answer(
            "alpha only",
            [requirement("contains_all", id="a", terms=["alpha"]),
             requirement("contains_all", id="b", terms=["beta"]),
             requirement("contains_all", id="c", terms=["gamma"])])
        self.assertEqual(scored["total"], 3)
        self.assertEqual(scored["satisfied"], 1)
        self.assertEqual(scored["missing_ids"], ["b", "c"])
        self.assertAlmostEqual(scored["completeness"], 1 / 3)

    def test_blank_answer_scores_zero_rather_than_unknown(self):
        scored = evaluation.score_answer(
            "", [requirement("contains_all", terms=["alpha"])])
        self.assertEqual(scored["completeness"], 0.0)

    def test_a_broken_regex_is_unsatisfied_not_a_crash(self):
        self.assertFalse(evaluation.check_requirement(
            "anything", {"id": "r", "check": {"kind": "regex",
                                              "pattern": "(["}}))


class TestCondition(unittest.TestCase):
    case = {"task_id": "c1", "capability": "general", "request": "q",
            "baseline_answer": "a",
            "requirements": [requirement("contains_all", terms=["alpha"])]}

    def test_answer_is_measured_not_the_prompt(self):
        row = {"answer": "alpha", "status": "completed", "qa_flags": []}
        result = evaluation.evaluate_condition(self.case, row,
                                               applied=["l1"],
                                               retrieved=["l1", "l2"])
        self.assertEqual(result["requirements"]["satisfied"], 1)
        self.assertEqual(result["applied"], ["l1"])
        self.assertEqual(result["retrieved"], ["l1", "l2"])

    def test_missing_answer_is_recorded_as_empty(self):
        result = evaluation.evaluate_condition(
            self.case, {"answer": "", "status": "completed"})
        self.assertFalse(result["answer_present"])
        self.assertEqual(result["failure_kind"], "empty_answer")
        self.assertEqual(result["requirements"]["satisfied"], 0)

    def test_timeout_and_failure_categories(self):
        timeout = evaluation.evaluate_condition(
            self.case, {"answer": "", "status": "failed",
                        "error": {"category": "timeout"}})
        self.assertEqual(timeout["failure_kind"], "timeout")
        failed = evaluation.evaluate_condition(
            self.case, {"answer": "", "status": "failed",
                        "error": {"category": "provider_unavailable"}})
        self.assertEqual(failed["failure_kind"], "provider_unavailable")

    def test_qa_severity_is_not_invented(self):
        result = evaluation.evaluate_condition(
            self.case, {"answer": "alpha", "status": "completed",
                        "qa_flags": ["QA: plan incomplete — task(s) x"]})
        self.assertEqual(result["qa"]["severity"], "not_recorded")
        self.assertEqual(result["qa"]["categories"], ["plan incomplete"])


class TestComparison(unittest.TestCase):
    def test_requirement_gained_is_an_improvement(self):
        base = condition("alpha")
        learn = condition("alpha beta")
        diff = evaluation.compare_task(base, learn)
        self.assertEqual(diff["outcome"], "improved")
        self.assertEqual(diff["gained"], ["b"])
        self.assertEqual(diff["regressions"], [])
        self.assertEqual(diff["completeness_delta_pp"], 50.0)

    def test_requirement_lost_is_a_regression(self):
        diff = evaluation.compare_task(condition("alpha beta"), condition("alpha"))
        self.assertEqual(diff["outcome"], "regression")
        self.assertEqual(diff["lost"], ["b"])
        self.assertEqual(diff["regressions"][0]["kind"], "requirement_lost")

    def test_new_qa_finding_is_a_regression(self):
        diff = evaluation.compare_task(
            condition("alpha beta"),
            condition("alpha beta", qa=["QA: something new"]))
        self.assertEqual(diff["outcome"], "regression")
        kinds = [r["kind"] for r in diff["regressions"]]
        self.assertIn("qa_regression", kinds)
        self.assertEqual(diff["qa_delta"], 1)

    def test_fewer_qa_findings_counts_as_improvement(self):
        diff = evaluation.compare_task(
            condition("alpha", qa=["QA: a"]),
            condition("alpha"))
        self.assertEqual(diff["outcome"], "improved")
        self.assertEqual(diff["qa_delta"], -1)

    def test_execution_and_output_regressions(self):
        diff = evaluation.compare_task(
            condition("alpha beta"),
            condition("", status="failed"))
        kinds = {r["kind"] for r in diff["regressions"]}
        self.assertIn("execution_regression", kinds)
        self.assertIn("output_lost", kinds)

    def test_gain_and_regression_together_is_mixed(self):
        # Gained requirement `a`, lost requirement `b`, plus a new QA flag.
        diff = evaluation.compare_task(
            condition("beta"),
            condition("alpha", qa=["QA: new"]))
        self.assertEqual(diff["outcome"], "mixed")

    def test_identical_conditions_are_unchanged(self):
        diff = evaluation.compare_task(condition("alpha"), condition("alpha"))
        self.assertEqual(diff["outcome"], "unchanged")
        self.assertEqual(diff["completeness_delta_pp"], 0.0)


class TestSummarize(unittest.TestCase):
    def _task(self, task_id, base, learn, *, applied=(), retrieved=(),
              capability="general"):
        b = condition(base)
        l = condition(learn)
        l["applied"] = list(applied)
        l["retrieved"] = list(retrieved)
        return {"task_id": task_id, "capability": capability,
                "baseline": b, "learning": l,
                "comparison": evaluation.compare_task(b, l)}

    def test_totals_and_per_lesson_credit(self):
        tasks = [
            self._task("t1", "alpha", "alpha beta", applied=["l1"],
                       retrieved=["l1"]),
            self._task("t2", "alpha beta", "alpha", applied=["l2"],
                       retrieved=["l2"]),
        ]
        summary = evaluation.summarize(tasks)
        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(summary["delta"]["improved_tasks"], 1)
        self.assertEqual(summary["delta"]["regressed_tasks"], 1)
        lessons = {l["lesson_id"]: l for l in summary["lessons"]}
        self.assertEqual(lessons["l1"]["improved"], 1)
        self.assertEqual(lessons["l2"]["regressed"], 1)

    def test_retrieved_but_not_applied_is_not_credited(self):
        tasks = [self._task("t1", "alpha", "alpha", applied=[],
                            retrieved=["l9"])]
        summary = evaluation.summarize(tasks)
        lesson = summary["lessons"][0]
        self.assertEqual(lesson["applied_in"], [])
        self.assertEqual(lesson["retrieved_not_applied_in"], ["t1"])

    def test_regressions_are_listed_with_their_task(self):
        tasks = [self._task("t2", "alpha beta", "alpha", applied=["l2"])]
        summary = evaluation.summarize(tasks)
        self.assertEqual(summary["regressions"][0]["task_id"], "t2")


class TestReporting(unittest.TestCase):
    def test_markdown_is_generated_from_the_report_numbers(self):
        dataset = {"version": "v-test"}
        base = condition("alpha")
        learn = condition("alpha beta")
        learn["applied"] = ["l1"]
        learn["retrieved"] = ["l1"]
        task = {"task_id": "t1", "capability": "general",
                "request": "q", "lesson_fixtures": ["l1"],
                "baseline_execution_id": "b", "learning_execution_id": "l",
                "baseline": base, "learning": learn,
                "comparison": evaluation.compare_task(base, learn)}
        report = evaluation.build_report(
            dataset, [task], mode="offline_deterministic",
            provider={"description": "scripted double", "live": False},
            started_at="start", finished_at="end", notes=["a caveat"])
        json.dumps(report)  # must be serialisable as-is
        markdown = evaluation.render_markdown(report)
        self.assertIn("| Requirement completeness | 50.0% | 100.0% | +50.0 pp |",
                      markdown)
        self.assertIn("Improved tasks: 1", markdown)
        self.assertIn("a caveat", markdown)
        self.assertIn("l1", markdown)


class TestBenchmarkEndToEnd(unittest.TestCase):
    """The real GraphRunner, both conditions, the shipped dataset."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = evaluation.load_dataset(DATASET_PATH)
        cls.report = benchmark.run_benchmark(cls.dataset)

    def test_both_conditions_used_the_identical_request(self):
        for task in self.report["tasks"]:
            self.assertEqual(task["baseline"]["status"], "completed")
            self.assertEqual(task["learning"]["status"], "completed")

    def test_baseline_received_no_lessons(self):
        for task in self.report["tasks"]:
            self.assertEqual(task["baseline"]["applied"], [])
            self.assertEqual(task["baseline"]["retrieved"], [])

    def test_learning_applied_only_the_declared_fixtures(self):
        for task in self.report["tasks"]:
            with self.subTest(task=task["task_id"]):
                # Applied must be a subset of what the task declared: a lesson
                # for another capability can be seeded and still never apply.
                self.assertTrue(
                    set(task["learning"]["applied"])
                    <= set(task["lesson_fixtures"]))
                self.assertEqual(
                    sorted(task["learning"]["applied"]),
                    sorted(task["comparison"]["applied_lessons"]))

    def test_outcomes_match_each_cases_expectation(self):
        for task in self.report["tasks"]:
            with self.subTest(task=task["task_id"]):
                self.assertEqual(task["comparison"]["outcome"],
                                 task["expected_outcome"])

    def test_measured_improvement_and_regression(self):
        summary = self.report["summary"]
        self.assertGreater(summary["learning"]["completeness"],
                           summary["baseline"]["completeness"])
        self.assertEqual(summary["delta"]["improved_tasks"], 3)
        self.assertEqual(summary["delta"]["regressed_tasks"], 1)
        self.assertEqual(summary["delta"]["regressions"], 3)
        regressed = [r for r in summary["regressions"]
                     if r["task_id"] == "gen-brevity-regression"]
        self.assertTrue(regressed)

    def test_scoping_case_received_nothing_for_another_capability(self):
        task = next(t for t in self.report["tasks"]
                    if t["task_id"] == "gen-unrelated-scope")
        self.assertEqual(task["learning"]["applied"], [])
        self.assertEqual(task["comparison"]["outcome"], "unchanged")

    def test_control_case_is_unchanged_in_both_conditions(self):
        task = next(t for t in self.report["tasks"]
                    if t["task_id"] == "gen-plain")
        self.assertEqual(
            task["baseline"]["requirements"]["satisfied"],
            task["learning"]["requirements"]["satisfied"])

    def test_report_declares_offline_mode_and_carries_provenance(self):
        self.assertEqual(self.report["mode"], "offline_deterministic")
        self.assertFalse(self.report["live_model"])
        for task in self.report["tasks"]:
            self.assertTrue(task["baseline_execution_id"])
            self.assertTrue(task["learning_execution_id"])
            self.assertNotEqual(task["baseline_execution_id"],
                                task["learning_execution_id"])

    def test_repeated_runs_are_reproducible(self):
        again = benchmark.run_benchmark(self.dataset)
        self.assertEqual(again["summary"], self.report["summary"])


class TestNoContamination(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ExecutionStore(Path(self._tmp.name) / "eval.sqlite3")
        self.addCleanup(self.store.close)

    def _seed(self):
        benchmark.seed_lessons(self.store, evaluation.load_dataset(DATASET_PATH))

    def test_disabled_writes_do_not_record_a_correction(self):
        self._seed()
        before = {l["lesson_id"] for l in self.store.list_lessons()}
        from app.learning import lessons as L
        from app.learning import signals

        signal = signals.detect(
            "These results aren't enough. I need benchmarks and licensing.",
            capability="general")
        with writes_disabled("test"):
            outcome = L.record(
                self.store, signal, source_execution_id=None,
                llm_call=lambda _p: "BEHAVIOUR: include benchmarking.")
        self.assertEqual(outcome["action"], "skipped")
        self.assertEqual({l["lesson_id"] for l in self.store.list_lessons()},
                         before)

    def test_disabled_writes_do_not_record_a_failure(self):
        self._seed()
        from app.learning import failures

        before = {l["lesson_id"] for l in self.store.list_lessons()}
        outcome = failures.observe(
            self.store, execution_id="e1", question="q", route="research",
            error={"category": "provider_unavailable"})
        self.assertFalse(outcome["recorded"])
        self.assertEqual({l["lesson_id"] for l in self.store.list_lessons()},
                         before)

    def test_a_disabled_lesson_is_never_applied(self):
        self._seed()
        lesson = self.store.list_lessons()[0]
        self.store.update_lesson(lesson["lesson_id"], status="disabled")
        model = benchmark.BenchmarkModel([
            {"task_id": "c", "request": "What is a vector database?",
             "baseline_answer": "A vector database stores embeddings."}])
        runner = GraphRunner(self.store)
        with benchmark.mock.patch("app.llm_client.get_llm",
                                  return_value=model):
            row = runner.run(runner.create("What is a vector database?")["id"])
        applied = benchmark._applied_ids(self.store, row["id"])
        self.assertNotIn(lesson["lesson_id"], applied)
        # The disabled lesson's own text never reaches a prompt (other active
        # fixtures may still apply — this test disables only one).
        self.assertFalse(any(lesson["corrective_behavior"] in prompt
                             for prompt in model.prompts))

    def test_benchmark_run_leaves_the_lesson_set_untouched(self):
        self._seed()
        before = {l["lesson_id"] for l in self.store.list_lessons()}
        dataset = evaluation.load_dataset(DATASET_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            benchmark.run_benchmark(
                dataset, store_dir=Path(tmp),
                model=benchmark.BenchmarkModel(dataset["cases"]))
        self.assertEqual({l["lesson_id"] for l in self.store.list_lessons()},
                         before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
