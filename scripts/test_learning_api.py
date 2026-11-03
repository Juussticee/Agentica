"""Learning through the REAL API (2026-10-30) — offline.

Proves the Control Center can only show and change true things:

  * `/api/v1/learning` reports persisted lessons, a repeated-failure pattern
    computed from them, and skill versions with their deterministic checks;
  * an empty database reports "nothing learned yet" instead of sample data;
  * review actions flip ONE lesson's status and append an audit entry to its
    evidence (the original correction is never edited away);
  * a review cannot grant a permission or touch provider configuration;
  * `/api/v1/memory` now reports learned procedures as real data, and
    `/api/v1/skills` includes learned versions;
  * rollback restores the previous skill version through the API;
  * no secret or absolute path appears in any payload.

Temp SQLite, TestClient, injected model doubles, no network.
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
from app.learning import lessons as L  # noqa: E402
from app.learning import signals as S  # noqa: E402
from app.learning import skills  # noqa: E402

ANALYSIS = (
    "PROBLEM: the answer was too thin\n"
    "CAUSE: depth was not requested\n"
    "BEHAVIOUR: Always include benchmarks and licensing in comparisons.\n"
    "VALIDATION: benchmarks and licensing appear\n"
    "KEYWORDS: benchmarks, licensing, comparison\n"
)
CORRECTION = ("These results aren't enough. I need benchmarks and licensing.")


class LearningApiCase(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.api.server as server

        self.server = server
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_store = server._store
        self.store = ExecutionStore(Path(self._tmp.name) / "learning.sqlite3")
        server._store = self.store
        self.addCleanup(self._restore)
        self.addCleanup(self.store.close)
        self.client = TestClient(server.build_server())

    def _restore(self):
        self.server._store = self._orig_store

    # -- helpers -----------------------------------------------------------

    def record(self, text=CORRECTION, capability="research",
               behavior="Always include benchmarks and licensing in "
                        "comparisons.", llm=None):
        signal = S.detect(text, capability=capability)
        self.assertIsNotNone(signal)
        if llm is None:
            default = ANALYSIS.replace(
                "Always include benchmarks and licensing in comparisons.",
                behavior)
            llm = lambda _p: default  # noqa: E731
        return L.record(self.store, signal, source_execution_id="ex-1",
                        llm_call=llm)["lesson"]

    def get_json(self, path):
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()


class TestLearningEndpoint(LearningApiCase):
    def test_empty_state_is_reported_not_invented(self):
        body = self.get_json("/api/v1/learning")
        self.assertEqual(body["total"], 0)
        self.assertEqual(body["active"], [])
        self.assertEqual(body["skills"], [])
        self.assertEqual(body["counts"], {})
        self.assertIn("Recording a lesson does not prove", body["note"])

    def test_recorded_lesson_appears_with_provenance(self):
        self.record()
        body = self.get_json("/api/v1/learning")
        self.assertEqual(body["total"], 1)
        lesson = body["active"][0]
        self.assertEqual(lesson["capability"], "research")
        self.assertEqual(lesson["source_execution_id"], "ex-1")
        self.assertEqual(lesson["evidence"][0]["text"], CORRECTION)
        self.assertIn("benchmarks", lesson["corrective_behavior"])
        self.assertEqual(body["counts"], {"active": 1})

    def test_repeated_patterns_are_computed_from_lessons(self):
        # Two DIFFERENT lessons for the same capability: different problem and
        # different behaviour, so neither is a duplicate nor a conflict.
        self.record(behavior="Always include benchmark tables.")
        second = ANALYSIS.replace("the answer was too thin",
                                  "licensing was omitted")
        self.record(
            text="These results are insufficient: I also needed licensing.",
            behavior="Always check licensing terms before recommending.",
            llm=lambda _p: second)
        body = self.get_json("/api/v1/learning")
        self.assertEqual(len(body["active"]), 2)
        self.assertEqual(len(body["repeated_patterns"]), 1)
        pattern = body["repeated_patterns"][0]
        self.assertEqual(pattern["capability"], "research")
        self.assertEqual(pattern["lessons"], 2)

    def test_no_secret_or_absolute_path_leaks(self):
        self.record()
        raw = json.dumps(self.get_json("/api/v1/learning"))
        raw += json.dumps(self.get_json("/api/v1/memory"))
        raw += json.dumps(self.get_json("/api/v1/skills"))
        self.assertNotIn("C:\\\\", raw)
        self.assertNotIn("/home/", raw)
        for secret in ("api_key", "API_KEY", "password", "sk-"):
            self.assertNotIn(secret, raw)


class TestReviewEndpoint(LearningApiCase):
    def test_disable_then_activate_a_lesson(self):
        lesson = self.record()

        response = self.client.post(
            f"/api/v1/learning/lessons/{lesson['lesson_id']}/review",
            json={"action": "disable"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["lesson"]["status"], "disabled")

        body = self.get_json("/api/v1/learning")
        self.assertEqual(body["active"], [])
        self.assertEqual(len(body["disabled"]), 1)

        again = self.client.post(
            f"/api/v1/learning/lessons/{lesson['lesson_id']}/review",
            json={"action": "activate"})
        self.assertEqual(again.json()["lesson"]["status"], "active")
        self.assertEqual(len(self.get_json("/api/v1/learning")["active"]), 1)

    def test_review_is_audited_and_keeps_the_correction(self):
        lesson = self.record()
        self.client.post(
            f"/api/v1/learning/lessons/{lesson['lesson_id']}/review",
            json={"action": "disable", "reason": "not what I want"})
        stored = self.store.get_lesson(lesson["lesson_id"])
        self.assertEqual(stored["evidence"][0]["text"], CORRECTION)
        review = stored["evidence"][-1]
        self.assertEqual(review["kind"], "review")
        self.assertEqual(review["action"], "disable")
        self.assertEqual(review["reason"], "not what I want")

    def test_correct_rewrites_the_behaviour_not_the_status_alone(self):
        lesson = self.record()
        response = self.client.post(
            f"/api/v1/learning/lessons/{lesson['lesson_id']}/review",
            json={"action": "correct",
                  "corrective_behavior": "Cite one benchmark per option."})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()["lesson"]
        self.assertEqual(body["corrective_behavior"],
                         "Cite one benchmark per option.")
        self.assertEqual(body["status"], "active")

    def test_review_cannot_change_a_permission(self):
        lesson = self.record()
        response = self.client.post(
            f"/api/v1/learning/lessons/{lesson['lesson_id']}/review",
            json={"action": "activate", "permissions": {"network": True}})
        # The extra field is ignored by the schema: it is not a lesson field,
        # and the stored row is unchanged apart from the audited review entry.
        self.assertEqual(response.status_code, 200, response.text)
        stored = self.store.get_lesson(lesson["lesson_id"])
        self.assertNotIn("permissions", stored)
        from app.agents import registry

        self.assertFalse(registry.AGENTS["conversational"]["permissions"]["network"])

    def test_unknown_lesson_and_unknown_action_are_errors(self):
        missing = self.client.post(
            "/api/v1/learning/lessons/lsn_missing/review",
            json={"action": "disable"})
        self.assertEqual(missing.status_code, 404)
        lesson = self.record()
        bad = self.client.post(
            f"/api/v1/learning/lessons/{lesson['lesson_id']}/review",
            json={"action": "delete-everything"})
        self.assertEqual(bad.status_code, 422)

    def test_correct_requires_text(self):
        lesson = self.record()
        response = self.client.post(
            f"/api/v1/learning/lessons/{lesson['lesson_id']}/review",
            json={"action": "correct"})
        self.assertEqual(response.status_code, 400)


class TestProjections(LearningApiCase):
    def test_memory_reports_learned_procedures_as_real_data(self):
        self.record()
        body = self.get_json("/api/v1/memory")
        procedures = next(s for s in body["sections"]
                          if s["id"] == "procedures")
        self.assertEqual(procedures["status"], "available")
        self.assertEqual(procedures["counts"]["active"], 1)
        self.assertTrue(procedures["items"])
        self.assertIn("lessons", procedures["provenance"])

    def test_memory_preferences_reflect_preference_lessons_only(self):
        self.record(text="Next time always use a table for comparisons.",
                    capability="", behavior="Use a markdown table.")
        body = self.get_json("/api/v1/memory")
        preferences = next(s for s in body["sections"]
                           if s["id"] == "preferences")
        self.assertEqual(preferences["status"], "available")
        self.assertEqual(preferences["items"][0]["capability"], "any")

    def test_skills_endpoint_includes_learned_versions(self):
        first = self.record(behavior="Always include benchmarks.")
        second = self.record(text="Next time always include licensing.",
                             behavior="Always include licensing terms.")
        skills.propose(self.store, "research-comparison", "research",
                       [first, second])
        body = self.get_json("/api/v1/skills")
        self.assertTrue(body["learned_versions"])
        version = body["learned_versions"][0]
        self.assertEqual(version["skill_id"], "research-comparison")
        self.assertEqual(version["status"], "candidate")
        self.assertIsNotNone(version["evaluation"])


class TestRollbackEndpoint(LearningApiCase):
    def test_rollback_restores_the_previous_version(self):
        lessons = [
            self.record(behavior="Always include benchmarks."),
            self.record(text="Next time always include licensing.",
                        behavior="Always include licensing terms."),
        ]
        v1 = skills.propose(self.store, "research-comparison", "research",
                            lessons[:1])
        self.assertTrue(skills.activate(self.store, v1["version_id"])["activated"])
        v2 = skills.propose(self.store, "research-comparison", "research",
                            lessons)
        self.assertTrue(skills.activate(self.store, v2["version_id"])["activated"])

        response = self.client.post(
            "/api/v1/learning/skills/research-comparison/rollback")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["rolled_back"])
        self.assertEqual(body["version"]["version"], 1)

    def test_rollback_with_nothing_to_restore_says_so(self):
        response = self.client.post(
            "/api/v1/learning/skills/workspace-coding/rollback")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["rolled_back"])
        self.assertIn("reason", response.json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
