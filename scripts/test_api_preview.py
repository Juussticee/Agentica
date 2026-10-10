"""GUI-2 tests: API preview / application boundary (14 tests, deterministic).

Covers health, CRUD execution endpoints, event cursor, validation, 404s and
storage failures. The graph is faked for deterministic runs; no LLM.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.execution.store import ExecutionStore  # noqa: E402


def make_client(tmpdir: Path):
    """Fresh app + store per test (import-time global reset)."""
    import app.api.server as server

    store = ExecutionStore(tmpdir / "api.sqlite3")
    client = TestClient(server.build_server())
    return server, store, client


class TestHealth(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._td = Path(tempfile.mkdtemp())
        self.server, self.store, self.client = make_client(self._td)
        self.server._store = self.store

    def tearDown(self):
        self.store.close()

    def test_health_ok_shape(self):
        r = self.client.get("/api/v1/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("status", "backend", "ai", "version"):
            self.assertIn(key, body)

    def test_health_ai_ready_not_configured_when_no_key(self):
        with mock.patch("app.config.LLM_PROVIDER", "gemini"), \
             mock.patch("app.config.GOOGLE_API_KEY", None):
            r = self.client.get("/api/v1/health")
        self.assertEqual(r.json()["ai"]["state"], "not_configured")

    def test_health_reports_ollama_provider_name(self):
        with mock.patch("app.config.LLM_PROVIDER", "ollama"), \
             mock.patch("app.config.OLLAMA_BASE_URL", "http://localhost:11434"), \
             mock.patch("app.config.MAIN_MODEL", "test-model"):
            r = self.client.get("/api/v1/health")
        self.assertEqual(r.json()["ai"]["provider"], "ollama")


class TestExecutionsCRUD(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._td = Path(tempfile.mkdtemp())
        self.server, self.store, self.client = make_client(self._td)
        self.server._store = self.store
        import app.graph as gm

        self.gm = gm
        self._snapshot = {n: getattr(gm, n) for n in
                          ("books_agent", "research_agent", "feasibility_agent", "erp_agent")}

        def fake_books(state):
            return {"answer": "BOK", "sources": ["books"],
                    "agent_answers": {**(state.get("agent_answers") or {}), "books": "BOK"}}

        gm.books_agent = fake_books

    def tearDown(self):
        for n, fn in self._snapshot.items():
            setattr(self.gm, n, fn)
        self.store.close()

    def test_create_and_run_returns_completed(self):
        r = self.client.post("/api/v1/executions",
                             json={"question": "What does chapter 3 say about propolis?"})
        self.assertEqual(r.status_code, 201)
        body = r.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["answer"], "BOK")

    def test_create_validates_question_length(self):
        r = self.client.post("/api/v1/executions", json={"question": "hi"})
        self.assertEqual(r.status_code, 422)

    def test_create_rejects_missing_question(self):
        r = self.client.post("/api/v1/executions", json={})
        self.assertEqual(r.status_code, 422)

    def test_list_returns_created_execution(self):
        self.client.post("/api/v1/executions",
                         json={"question": "What does chapter 3 say about propolis?"})
        r = self.client.get("/api/v1/executions")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 1)

    def test_get_by_id_returns_detail(self):
        created = self.client.post(
            "/api/v1/executions",
            json={"question": "What does chapter 3 say about propolis?"},
        ).json()
        r = self.client.get(f"/api/v1/executions/{created['id']}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], created["id"])

    def test_get_unknown_id_is_404_with_human_error(self):
        r = self.client.get("/api/v1/executions/nonexistent")
        self.assertEqual(r.status_code, 404)
        self.assertIn("human_message", r.json()["detail"])

    def test_events_after_cursor(self):
        created = self.client.post(
            "/api/v1/executions",
            json={"question": "What does chapter 3 say about propolis?"},
        ).json()
        r = self.client.get(f"/api/v1/executions/{created['id']}/events")
        self.assertEqual(r.status_code, 200)
        all_events = r.json()["events"]
        self.assertGreaterEqual(len(all_events), 2)
        r2 = self.client.get(
            f"/api/v1/executions/{created['id']}/events",
            params={"after": all_events[0]["event_id"]},
        )
        remaining = r2.json()["events"]
        self.assertEqual([e["event_id"] for e in remaining],
                         [e["event_id"] for e in all_events][1:])

    def test_events_unknown_id_is_404(self):
        r = self.client.get("/api/v1/executions/nonexistent/events")
        self.assertEqual(r.status_code, 404)

    def test_failed_execution_reports_failure_honestly(self):
        def bad(state):
            raise RuntimeError("ConnectionError: provider down")

        self.gm.research_agent = bad
        r = self.client.post("/api/v1/executions",
                             json={"question": "Latest AI news this week"})
        self.assertEqual(r.status_code, 201)  # request accepted; run failed honestly
        body = r.json()
        self.assertEqual(body["status"], "failed")
        self.assertIn("human_message", body["error"])

    def test_event_order_is_monotonic_per_execution(self):
        created = self.client.post(
            "/api/v1/executions",
            json={"question": "What does chapter 3 say about propolis?"},
        ).json()
        events = self.client.get(
            f"/api/v1/executions/{created['id']}/events").json()["events"]
        ids = [e["event_id"] for e in events]
        self.assertEqual(ids, sorted(ids))


class TestApiIsolation(unittest.TestCase):
    def test_two_clients_do_not_share_state(self):
        import tempfile

        td1 = Path(tempfile.mkdtemp())
        td2 = Path(tempfile.mkdtemp())
        s1, store1, c1 = make_client(td1)
        s2, store2, c2 = make_client(td2)
        try:
            # Same app object is served by both clients; each test point gets
            # its own store by swapping the module-level handle.
            s1._store = store1
            c1.post("/api/v1/executions",
                    json={"question": "What does chapter 3 say about propolis?"})
            s1._store = None
            s2._store = store2
            self.assertEqual(c2.get("/api/v1/executions").json()["count"], 0)
        finally:
            store1.close()
            store2.close()
            import app.api.server as server_mod
            server_mod._store = None


if __name__ == "__main__":
    unittest.main(verbosity=2)
