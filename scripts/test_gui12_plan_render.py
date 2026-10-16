"""GUI-12 tests: planner visualization data contract (deterministic).

Covers the smallest backend/API change that exposes the REAL executed plan:
persistence (store `plan` column + schema migration), the runner's
`execution.planned` event, the /plan endpoint, the detail-response plan
projection, and graceful handling of missing/legacy/corrupt plan data.
No network, no LLM quota.
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.execution.events import EXECUTION_COMPLETED, EXECUTION_PLANNED  # noqa: E402
from app.execution.runner import GraphRunner  # noqa: E402
from app.execution.store import ExecutionStore, serialize_plan  # noqa: E402


def make_store():
    td = tempfile.mkdtemp()
    return ExecutionStore(Path(td) / "test.sqlite3")


class TestSerializePlan(unittest.TestCase):
    def test_none_and_garbage_degrade_to_none(self):
        self.assertIsNone(serialize_plan(None))
        self.assertIsNone(serialize_plan({"no_tasks": True}))
        self.assertIsNone(serialize_plan({"tasks": "not-a-list"}))

    def test_task_fields_projected_bookkeeping_pruned(self):
        plan = {
            "goal": "g",
            "source": "llm",
            "dispatcher_visits": 7,  # internal bookkeeping must NOT leak
            "tasks": [{
                "task_id": "t1", "capability": "research",
                "objective": "Do research", "depends_on": [],
                "required_inputs": ["question"],
                "expected_output": "Summary",
                "done_when": "Summary available",
                "status": "completed", "result": "Found things",
                "dispatcher_note": "internal",
            }],
        }
        out = serialize_plan(plan)
        self.assertEqual(out["goal"], "g")
        t = out["tasks"][0]
        for key in ("task_id", "capability", "capability_label", "objective",
                    "depends_on", "required_inputs", "expected_output",
                    "done_when", "status", "result", "result_truncated",
                    "error", "block_reason"):
            self.assertIn(key, t)
        self.assertNotIn("dispatcher_note", t)
        self.assertEqual(out["tasks"][0]["status"], "completed")
        # Whole serialized dict never contains internal counters.
        self.assertNotIn("dispatcher_visits", json.dumps(out))

    def test_long_result_trimmed_and_flagged(self):
        plan = {"goal": "g", "tasks": [{
            "task_id": "t1", "capability": "research", "status": "completed",
            "result": "x" * 1000,
        }]}
        out = serialize_plan(plan)
        self.assertTrue(out["tasks"][0]["result_truncated"])
        self.assertLessEqual(len(out["tasks"][0]["result"]), 400)

    def test_unknown_capability_kept_not_fabricated(self):
        plan = {"goal": "g", "tasks": [{
            "task_id": "t1", "capability": "mystery-cap", "status": "pending",
        }]}
        out = serialize_plan(plan)
        self.assertEqual(out["tasks"][0]["capability_label"], "mystery-cap")


class TestStorePlanColumn(unittest.TestCase):
    def test_plan_roundtrip(self):
        s = make_store()
        s.create_execution("e1", "q")
        s.update_execution("e1", plan={"goal": "g", "tasks": []})
        row = s.get_execution("e1")
        self.assertEqual(row["plan"], {"goal": "g", "tasks": []})
        s.close()

    def test_legacy_row_has_plan_none(self):
        s = make_store()
        s.create_execution("e1", "q")  # never got a plan
        self.assertIsNone(s.get_execution("e1")["plan"])
        s.close()

    def test_corrupt_plan_blob_degrades_to_none(self):
        s = make_store()
        s.create_execution("e1", "q")
        s._conn.execute("UPDATE executions SET plan = ? WHERE id = ?",
                        ("{not json", "e1"))
        s._conn.commit()
        self.assertIsNone(s.get_execution("e1")["plan"])
        s.close()

    def test_list_rows_omit_plan(self):
        s = make_store()
        s.create_execution("e1", "q")
        s.update_execution("e1", plan={"goal": "g", "tasks": []})
        rows = s.list_executions()
        self.assertNotIn("plan", rows[0])
        s.close()

    def test_migration_adds_column_to_legacy_db(self):
        td = Path(tempfile.mkdtemp())
        db = td / "legacy.sqlite3"
        # Create the OLD (GUI-2) schema without `plan` and insert a row.
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE executions (id TEXT PRIMARY KEY, question TEXT NOT NULL,"
            " status TEXT NOT NULL, route TEXT, answer TEXT, qa_flags TEXT,"
            " error TEXT, created_at TEXT NOT NULL, started_at TEXT,"
            " completed_at TEXT, result TEXT)")
        conn.execute("INSERT INTO executions (id, question, status, created_at)"
                     " VALUES ('old1', 'q', 'completed', '2026-01-01T00:00:00')")
        conn.commit()
        conn.close()

        s = ExecutionStore(db)  # opening must migrate transparently
        row = s.get_execution("old1")
        self.assertEqual(row["status"], "completed")
        self.assertIsNone(row["plan"])
        # Migration is real: the column now exists for new writes.
        s.update_execution("old1", plan={"goal": "g", "tasks": []})
        self.assertEqual(s.get_execution("old1")["plan"], {"goal": "g", "tasks": []})
        s.close()


class _FakeGraphMixin:
    """Install faked graph agents (established suite convention)."""

    def setUp(self):
        self.store = make_store()
        self.runner = GraphRunner(self.store)
        import app.graph as gm
        self.gm = gm
        self._snapshot = {n: getattr(gm, n) for n in
                          ("books_agent", "research_agent", "feasibility_agent",
                           "erp_agent")}

        def fake_books(state):
            return {"answer": "BOK", "sources": ["books"],
                    "agent_answers": {**(state.get("agent_answers") or {}),
                                      "books": "BOK"}}

        def fake_feasibility(state):
            return {"answer": "STUDY", "sources": ["feasibility"],
                    "agent_answers": {**(state.get("agent_answers") or {}),
                                      "feasibility": "STUDY"}}

        self.gm.books_agent = fake_books
        self.gm.feasibility_agent = fake_feasibility

    def tearDown(self):
        for n, fn in self._snapshot.items():
            setattr(self.gm, n, fn)
        self.store.close()


class TestRunnerPersistsPlan(_FakeGraphMixin, unittest.TestCase):
    def _run_planned(self, question):
        # "always" forces the planner route; get_llm failure → deterministic
        # fallback plan (source='fallback'), mirroring an offline planner.
        with mock.patch("app.planner.PLANNER_MODE", "always"), \
                mock.patch("app.llm_client.get_llm",
                           side_effect=RuntimeError("provider down")):
            e = self.runner.create(question)
            return self.runner.run(e["id"]), e["id"]

    def test_planned_run_persists_plan_with_statuses(self):
        done, eid = self._run_planned(
            "What does chapter 3 say about propolis?")
        self.assertEqual(done["status"], "completed")
        plan = done["plan"]
        self.assertIsInstance(plan, dict)
        tids = [t["task_id"] for t in plan["tasks"]]
        self.assertEqual(tids, ["t1", "t2"])  # books + synthesis default plan
        statuses = {t["task_id"]: t["status"] for t in plan["tasks"]}
        self.assertEqual(statuses, {"t1": "completed", "t2": "completed"})
        # Result summaries are the REAL outputs of the executed tasks.
        self.assertEqual(plan["tasks"][0]["result"], "BOK")

    def test_planned_event_precedes_completion(self):
        _, eid = self._run_planned(
            "What does chapter 3 say about propolis?")
        types = [ev["type"] for ev in self.store.get_events(eid)]
        self.assertIn(EXECUTION_PLANNED, types)
        self.assertLess(types.index(EXECUTION_PLANNED),
                        types.index(EXECUTION_COMPLETED))

    def test_planned_event_payload_carries_renderable_tasks(self):
        _, eid = self._run_planned(
            "What does chapter 3 say about propolis?")
        planned = [ev for ev in self.store.get_events(eid)
                   if ev["type"] == EXECUTION_PLANNED][0]
        payload = planned["payload"]
        self.assertEqual(payload["plan"]["source"], "fallback")
        self.assertEqual(len(payload["plan"]["tasks"]), 2)
        self.assertIn("status", payload["plan"]["tasks"][0])

    def test_simple_run_has_no_plan(self):
        # "off": byte-identical pre-Phase-8 path; no plan may ever appear.
        with mock.patch("app.planner.PLANNER_MODE", "off"):
            e = self.runner.create("What does chapter 3 say about propolis?")
            done = self.runner.run(e["id"])
        self.assertEqual(done["status"], "completed")
        self.assertIsNone(done["plan"])
        self.assertNotIn(EXECUTION_PLANNED,
                         [ev["type"] for ev in self.store.get_events(e["id"])])

    def test_failed_run_has_no_plan_but_still_loads(self):
        def bad(state):
            raise RuntimeError("ConnectionError: provider down")
        self.gm.research_agent = bad
        with mock.patch("app.planner.PLANNER_MODE", "off"):
            e = self.runner.create("Latest AI news this week")
            done = self.runner.run(e["id"])
        self.assertEqual(done["status"], "failed")
        self.assertIsNone(done["plan"])
        self.assertIn("human_message", done["error"])


class TestPlanEndpoint(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._td = Path(tempfile.mkdtemp())
        import app.api.server as server

        self.server = server
        self.store = ExecutionStore(self._td / "api.sqlite3")
        self.client = __import__("fastapi.testclient",
                                 fromlist=["TestClient"]).TestClient(
                                     server.build_server())
        self.server._store = self.store
        import app.graph as gm
        self.gm = gm
        self._snapshot = {n: getattr(gm, n) for n in
                          ("books_agent", "research_agent", "feasibility_agent",
                           "erp_agent")}

        def fake_books(state):
            return {"answer": "BOK", "sources": ["books"],
                    "agent_answers": {**(state.get("agent_answers") or {}),
                                      "books": "BOK"}}

        self.gm.books_agent = fake_books

    def tearDown(self):
        for n, fn in self._snapshot.items():
            setattr(self.gm, n, fn)
        self.store.close()
        self.server._store = None

    def _create(self, question, planner_mode):
        with mock.patch("app.planner.PLANNER_MODE", planner_mode), \
                mock.patch("app.llm_client.get_llm",
                           side_effect=RuntimeError("provider down")):
            return self.client.post(
                "/api/v1/executions", json={"question": question}).json()

    def test_plan_endpoint_serves_real_plan(self):
        created = self._create("What does chapter 3 say about propolis?",
                               "always")
        r = self.client.get(f"/api/v1/executions/{created['id']}/plan")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["execution_id"], created["id"])
        plan = body["plan"]
        self.assertIsNotNone(plan)
        self.assertEqual([t["task_id"] for t in plan["tasks"]],
                         ["t1", "t2"])
        self.assertEqual({t["status"] for t in plan["tasks"]}, {"completed"})
        self.assertEqual(plan["source"], "fallback")

    def test_plan_endpoint_null_for_simple_route(self):
        created = self._create("What does chapter 3 say about propolis?",
                               "off")
        r = self.client.get(f"/api/v1/executions/{created['id']}/plan")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["plan"])

    def test_plan_endpoint_null_for_legacy_row(self):
        # Simulate a pre-GUI-12 execution: no plan ever persisted.
        self.store.create_execution("legacy1", "old question")
        self.store.update_execution("legacy1", status="completed",
                                    answer="old answer")
        r = self.client.get("/api/v1/executions/legacy1/plan")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["plan"])

    def test_plan_endpoint_unknown_id_404_human_error(self):
        r = self.client.get("/api/v1/executions/nope/plan")
        self.assertEqual(r.status_code, 404)
        self.assertIn("human_message", r.json()["detail"])

    def test_detail_response_includes_plan_projection(self):
        created = self._create("What does chapter 3 say about propolis?",
                               "always")
        detail = self.client.get(
            f"/api/v1/executions/{created['id']}").json()
        self.assertIsNotNone(detail["plan"])
        self.assertIn("capability_label", detail["plan"]["tasks"][0])

    def test_detail_response_backward_compatible_without_plan(self):
        created = self._create("What does chapter 3 say about propolis?",
                               "off")
        detail = self.client.get(
            f"/api/v1/executions/{created['id']}").json()
        self.assertIsNone(detail["plan"])
        # Existing detail fields are untouched.
        for key in ("id", "question", "status", "answer", "qa_flags",
                    "created_at"):
            self.assertIn(key, detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
