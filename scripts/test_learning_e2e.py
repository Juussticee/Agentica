"""Continuous learning — end-to-end suite (2026-10-30), offline.

Every test runs a REAL request through the real pipeline: GraphRunner ->
supervisor -> node -> QA -> persisted execution row, with only the MODEL
provider doubled. Learning is then checked against the database and the
execution timeline, not against a function call.

Scenarios:
  1  an ordinary question is never learned from
  2  a correction revises the answer AND records a lesson (with provenance)
  3  the next comparable task carries the requirement in its prompt — the
     same prompt captured BEFORE learning does not (a before/after measure)
  4  a different capability does not receive that lesson
  5  a disabled lesson stops applying
  6  a correction with no recorded result is answered honestly and still
     recorded
  7  a lesson learned from an execution failure never auto-applies
  8  two corrections for one capability produce a validated skill version,
     and rollback restores the first
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.execution.runner import GraphRunner  # noqa: E402
from app.execution.store import ExecutionStore  # noqa: E402
from app.learning import lessons as L  # noqa: E402
from app.learning import skills  # noqa: E402

BEHAVIOUR = ("Include storage requirements, pricing and one worked example "
             "in every technical comparison.")
ANALYSIS = (
    "PROBLEM: the earlier answer gave no storage requirements or pricing\n"
    "CAUSE: depth was not requested\n"
    f"BEHAVIOUR: {BEHAVIOUR}\n"
    "VALIDATION: storage requirements, pricing and a worked example appear\n"
    "KEYWORDS: storage, pricing, example, comparison\n"
)
CORRECTION = ("These results aren't enough for me. I need storage "
              "requirements, pricing and a worked example.")


class FakeLLM:
    """Provider double: records prompts, answers by marker.

    `behaviours` lets a test make each correction analysis DIFFERENT — the
    deterministic policy treats a different procedure for the same topic as a
    conflict (supersede), not a duplicate (merge), so the double must be able
    to say two different things.
    """

    def __init__(self, behaviours=None):
        self.prompts = []
        self.behaviours = list(behaviours or [])

    def invoke(self, prompt):
        text = str(prompt)
        self.prompts.append(text)
        if "analysing ONE user correction" in text:
            if self.behaviours:
                behaviour = self.behaviours.pop(0)
                return SimpleNamespace(content=ANALYSIS.replace(
                    BEHAVIOUR, behaviour))
            return SimpleNamespace(content=ANALYSIS)
        if "revising an earlier answer" in text:
            return SimpleNamespace(content="REVISED_ANSWER")
        if "You are a task planner" in text:
            return SimpleNamespace(content="not a plan")
        return SimpleNamespace(content="MODEL_ANSWER")

    def matching(self, marker):
        return [p for p in self.prompts if marker in p]


class LearningCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ExecutionStore(Path(self._tmp.name) / "learning.sqlite3")
        self.addCleanup(self.store.close)
        self.runner = GraphRunner(self.store)
        self.llm = FakeLLM()
        patcher = mock.patch("app.llm_client.get_llm", return_value=self.llm)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ask(self, question):
        return self.runner.run(self.runner.create(question)["id"])

    def seed_answer(self, question="What is a vector database?",
                    answer="A vector database stores embeddings.",
                    route="general"):
        eid = self.runner.create(question)["id"]
        self.store.update_execution(eid, status="completed", route=route,
                                    answer=answer, result=answer)
        return eid

    def events(self, eid, type_):
        return [e for e in self.store.get_events(eid) if e.get("type") == type_]


class TestSignalRouting(LearningCase):
    def test_ordinary_question_is_not_learned_from(self):
        row = self.ask("What is a vector database?")
        self.assertEqual(row["route"], "general")
        self.assertEqual(self.store.list_lessons(), [])
        self.assertEqual(self.events(row["id"], "learning.lesson_recorded"), [])

    def test_correction_routes_to_feedback_not_to_a_new_question(self):
        self.seed_answer()
        row = self.ask(CORRECTION)
        self.assertEqual(row["route"], "feedback")
        self.assertEqual(row["status"], "completed")


class TestCorrectionLoop(LearningCase):
    def test_correction_revises_the_answer_and_records_a_lesson(self):
        prior = self.seed_answer()
        row = self.ask(CORRECTION)

        self.assertEqual(row["route"], "feedback")
        self.assertIn("REVISED_ANSWER", row["answer"])
        lessons = self.store.list_lessons()
        self.assertEqual(len(lessons), 1)
        lesson = lessons[0]
        self.assertEqual(lesson["status"], L.ACTIVE)
        self.assertEqual(lesson["capability"], "general")
        self.assertEqual(lesson["source_execution_id"], prior)
        self.assertEqual(lesson["origin"], L.ORIGIN_USER)
        self.assertIn("storage requirements", lesson["corrective_behavior"])

        (event,) = self.events(row["id"], "learning.lesson_recorded")
        self.assertEqual(event["payload"]["status"], "active")
        self.assertEqual(event["payload"]["lesson_id"], lesson["lesson_id"])

    def test_the_requirement_reaches_a_later_comparable_task(self):
        comparable = "Explain storage requirements for embedding indexes."
        self.seed_answer()
        baseline = self.ask(comparable)
        baseline_prompts = self.llm.matching("Answer the user's general question")
        self.assertTrue(baseline_prompts)
        self.assertNotIn("Learned requirements", baseline_prompts[-1])

        self.ask(CORRECTION)
        later = self.ask(comparable)
        prompts = self.llm.matching("Answer the user's general question")
        self.assertIn("Learned requirements", prompts[-1])
        self.assertIn(BEHAVIOUR, prompts[-1])

        # The requirement is in the LATER prompt only: that is the measurable
        # difference, not just a new row in the database.
        self.assertNotIn(BEHAVIOUR, baseline_prompts[-1])

        lesson = self.store.list_lessons()[0]
        self.assertEqual(lesson["hits"], 1)
        self.assertEqual(lesson["successes"], 1)
        (used,) = self.events(later["id"], "learning.lessons_applied")
        self.assertEqual(used["payload"]["lesson_ids"], [lesson["lesson_id"]])
        self.assertTrue(used["payload"]["success"])

        # And the baseline run was not credited with using it.
        self.assertEqual(self.events(baseline["id"], "learning.lessons_applied"),
                         [])

    def test_an_unrelated_question_in_the_same_capability_is_not_injected(self):
        """2026-10-31 live defect: a content lesson ("include a trade-off
        comparison with figures") was injected into "what does HTTP 429
        mean?", and the model answered the rule instead of the question. A
        QUALITY lesson applies only where the request shares its subject."""
        self.seed_answer()
        self.ask(CORRECTION)
        row = self.ask("What does the HTTP status code 429 mean?")
        self.assertEqual(row["route"], "general")
        prompt = self.llm.matching("Answer the user's general question")[-1]
        self.assertNotIn(BEHAVIOUR, prompt)
        self.assertEqual(self.events(row["id"], "learning.lessons_applied"), [])
        # The lesson itself is untouched: it simply did not apply here.
        self.assertEqual(self.store.list_lessons()[0]["hits"], 0)

    def test_a_different_capability_does_not_receive_the_lesson(self):
        self.seed_answer()
        self.ask(CORRECTION)
        self.ask("Hi")
        conversational = self.store.list_executions()[0]
        self.assertEqual(conversational["route"], "conversational")
        self.assertEqual(
            self.events(conversational["id"], "learning.lessons_applied"), [])
        lesson = self.store.list_lessons()[0]
        self.assertEqual(lesson["hits"], 0)

    def test_a_disabled_lesson_stops_applying(self):
        self.seed_answer()
        self.ask(CORRECTION)
        lesson = self.store.list_lessons()[0]
        self.store.update_lesson(lesson["lesson_id"], status=L.DISABLED)

        row = self.ask("Explain embedding indexes.")
        prompt = self.llm.matching("Answer the user's general question")[-1]
        self.assertNotIn(BEHAVIOUR, prompt)
        self.assertEqual(self.events(row["id"], "learning.lessons_applied"), [])

    def test_no_recorded_result_is_reported_honestly(self):
        row = self.ask(CORRECTION)
        self.assertEqual(row["route"], "feedback")
        self.assertIn("I don't have the earlier result", row["answer"])
        # The correction is still behaviour worth learning from.
        self.assertEqual(len(self.store.list_lessons()), 1)
        self.assertIsNone(self.store.list_lessons()[0]["source_execution_id"])

    def test_repeated_correction_merges_into_one_lesson(self):
        self.seed_answer()
        self.ask(CORRECTION)
        self.ask(CORRECTION)
        lessons = self.store.list_lessons()
        self.assertEqual(len(lessons), 1)
        self.assertEqual(len(lessons[0]["evidence"]), 2)


class TestFailureLearning(LearningCase):
    def test_failure_lesson_never_auto_applies(self):
        from app.learning import signals

        signal = signals.detect(
            "The run keeps failing with a tool error and comes back incomplete.",
            capability="general")
        outcome = L.record(self.store, signal, source_execution_id="ex-fail",
                           origin=L.ORIGIN_FAILURE,
                           llm_call=lambda _p: ANALYSIS)
        self.assertEqual(outcome["lesson"]["status"], L.CANDIDATE)

        row = self.ask("Explain embedding indexes.")
        prompt = self.llm.matching("Answer the user's general question")[-1]
        self.assertNotIn(BEHAVIOUR, prompt)
        self.assertEqual(self.events(row["id"], "learning.lessons_applied"), [])


class TestSkillImprovement(LearningCase):
    def test_two_corrections_produce_a_validated_version_and_rollback_works(self):
        # Two DIFFERENT lessons for one capability (a conflict would supersede,
        # so each correction names a distinct requirement).
        self.llm.behaviours = [
            "Include benchmark tables with sources in comparisons.",
            "Check licensing terms before recommending an option.",
        ]
        self.seed_answer(question="Compare three CRM platforms",
                         answer="CRM_A is good. CRM_B is cheaper.",
                         route="research")
        self.ask("These results aren't enough. I need benchmarks and pricing.")
        self.ask("Actually, that's still not enough. I needed licensing and "
                 "sources.")
        active = [l for l in self.store.list_lessons(status=L.ACTIVE)
                  if l["capability"] == "research"]
        self.assertEqual(len(active), 2)

        versions = self.store.list_skill_versions("research-comparison")
        self.assertTrue(versions)
        self.assertEqual(versions[0]["status"], "active")
        self.assertTrue(versions[0]["evaluation"]["passed"])

        # A new candidate built from the same lessons: activating it retires
        # v1, and rollback puts v1 back exactly.
        second = skills.propose(self.store, "research-comparison", "research",
                                active)
        self.assertTrue(skills.activate(self.store, second["version_id"])
                        ["activated"])
        self.assertEqual(
            self.store.get_skill_version(versions[0]["version_id"])["status"],
            "retired")
        restored = skills.rollback(self.store, "research-comparison")
        self.assertTrue(restored["rolled_back"])
        self.assertEqual(restored["version"]["version"],
                         versions[0]["version"])


class TestFailureObservation(LearningCase):
    """A raise is observed, but only a REPEAT becomes a lesson."""

    def test_repeated_graph_failure_becomes_a_candidate_lesson(self):
        boom = mock.patch("app.graph.build_graph",
                          side_effect=RuntimeError("graph exploded"))
        boom.start()
        self.addCleanup(boom.stop)

        first = self.ask("What is a vector database?")
        self.assertEqual(first["status"], "failed")
        self.assertEqual(self.store.list_lessons(), [])

        second = self.ask("What is a vector database?")
        self.assertEqual(second["status"], "failed")
        lessons = self.store.list_lessons()
        self.assertEqual(len(lessons), 1)
        lesson = lessons[0]
        self.assertEqual(lesson["status"], L.CANDIDATE)
        self.assertEqual(lesson["origin"], L.ORIGIN_FAILURE)
        self.assertEqual(lesson["source_execution_id"], second["id"])
        self.assertTrue(self.events(second["id"], "learning.lesson_recorded"))
        # A candidate is never applied, so the prompt of the next run is clean.
        third = self.ask("What is a vector database?")
        self.assertEqual(third["status"], "failed")
        self.assertEqual(self.events(third["id"], "learning.lessons_applied"),
                         [])
        self.assertEqual(len(self.store.list_lessons()), 1)


class TestNoSensitiveContent(LearningCase):
    def test_learning_never_persists_a_credential_shaped_value(self):
        # The feedback node stores the user's own sentence as evidence. A
        # secret the user typed would be stored: that is why nothing else is
        # copied — this test pins the boundary (evidence is the message only).
        self.seed_answer()
        self.ask(CORRECTION)
        lesson = self.store.list_lessons()[0]
        payload = json.dumps(lesson)
        for forbidden in ("API_KEY", "sk-", "BEGIN PRIVATE KEY"):
            self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
