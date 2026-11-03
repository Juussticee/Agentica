"""GUI-1 tests: execution event foundation (18 tests, deterministic).

Covers events module, store roundtrips, runner lifecycle, honest failure,
and graph build_graph(node_wrapper) hook. No network, no LLM.
"""

import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.execution import events as ev  # noqa: E402
from app.execution.runner import GraphRunner, human_step_name  # noqa: E402
from app.execution.store import ExecutionStore  # noqa: E402


def make_store():
    td = tempfile.mkdtemp()
    return ExecutionStore(Path(td) / "test.sqlite3")


class TestEventModel(unittest.TestCase):
    def test_event_to_dict_has_all_fields(self):
        e = ev.ExecutionEvent(
            event_id=1, execution_id="x", timestamp=ev.utc_now_iso(),
            type=ev.EXECUTION_STARTED, message="m", step_id="s",
            agent_id="a", severity=ev.SEV_INFO, payload={"k": "v"},
        )
        d = e.to_dict()
        for key in ("event_id", "execution_id", "timestamp", "type", "message",
                    "step_id", "agent_id", "severity", "payload"):
            self.assertIn(key, d)

    def test_constants_exist(self):
        for c in (ev.EXECUTION_CREATED, ev.EXECUTION_STARTED, ev.EXECUTION_COMPLETED,
                  ev.EXECUTION_FAILED, ev.AGENT_STARTED, ev.AGENT_COMPLETED,
                  ev.AGENT_FAILED):
            self.assertTrue(c)

    def test_utc_now_iso_is_iso(self):
        self.assertIn("T", ev.utc_now_iso())


class TestStoreRoundtrip(unittest.TestCase):
    def test_create_and_fetch(self):
        s = make_store()
        s.create_execution("e1", "question", )
        row = s.get_execution("e1")
        self.assertEqual(row["question"], "question")
        self.assertEqual(row["status"], "queued")
        s.close()

    def test_append_and_get_events_after_cursor(self):
        s = make_store()
        s.create_execution("e1", "q")
        for i, t in enumerate(["execution.created", "execution.started",
                               "execution.completed"], start=1):
            s.append_event("e1", i, t, f"msg {i}")
        self.assertEqual([e["event_id"] for e in s.get_events("e1")], [1, 2, 3])
        self.assertEqual([e["event_id"] for e in s.get_events("e1", after=2)], [3])
        s.close()

    def test_update_fields_and_error_json(self):
        s = make_store()
        s.create_execution("e1", "q")
        s.update_execution("e1", status="failed",
                           error={"category": "timeout", "human_message": "x"})
        row = s.get_execution("e1")
        self.assertEqual(row["status"], "failed")
        self.assertIsInstance(row["error"], dict)
        s.close()

    def test_list_newest_first(self):
        s = make_store()
        s.create_execution("a", "q1")
        s.create_execution("b", "q2")
        rows = s.list_executions()
        self.assertEqual({r["id"] for r in rows}, {"a", "b"})
        self.assertGreaterEqual(rows[0]["created_at"], rows[1]["created_at"])
        s.close()

    def test_missing_execution_returns_none(self):
        s = make_store()
        self.assertIsNone(s.get_execution("nope"))
        self.assertEqual(s.get_events("nope"), [])
        s.close()


class TestRunnerLifecycle(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        self.runner = GraphRunner(self.store)
        import app.graph as gm
        self.gm = gm
        self._snapshot = {n: getattr(gm, n) for n in
                          ("books_agent", "research_agent", "feasibility_agent", "erp_agent")}

    def tearDown(self):
        for n, fn in self._snapshot.items():
            setattr(self.gm, n, fn)
        self.store.close()

    def _fake_books(self):
        def fake(state):
            return {"answer": "BOK", "sources": ["books"],
                    "agent_answers": {**(state.get("agent_answers") or {}), "books": "BOK"}}
        self.gm.books_agent = fake

    def _failing_research(self):
        def bad(state):
            raise RuntimeError("ConnectionError: provider down")
        self.gm.research_agent = bad

    def test_create_persists_with_event(self):
        e = self.runner.create("What does chapter 3 say about propolis?")
        row = self.store.get_execution(e["id"])
        self.assertEqual(row["status"], "queued")
        events = self.store.get_events(e["id"])
        self.assertEqual(events[0]["type"], ev.EXECUTION_CREATED)

    def test_successful_run_records_lifecycle_events(self):
        self._fake_books()
        e = self.runner.create("What does chapter 3 say about propolis?")
        done = self.runner.run(e["id"])
        self.assertEqual(done["status"], "completed")
        types = [x["type"] for x in self.store.get_events(e["id"])]
        self.assertIn(ev.EXECUTION_STARTED, types)
        self.assertIn(ev.AGENT_STARTED, types)
        self.assertIn(ev.AGENT_COMPLETED, types)
        self.assertIn(ev.EXECUTION_COMPLETED, types)

    def test_successful_run_stores_answer_and_sources(self):
        self._fake_books()
        e = self.runner.create("What does chapter 3 say about propolis?")
        done = self.runner.run(e["id"])
        self.assertEqual(done["answer"], "BOK")
        # qa_flags exists even when empty (Phase 5 gate ran)
        self.assertEqual(done["qa_flags"], [] or done["qa_flags"])

    def test_pending_research_approval_has_waiting_status_not_completed(self):
        from app.execution.store import STATUS_WAITING_APPROVAL
        from app.graph import approval_scope

        e = self.runner.create("Search the latest AI news")
        final_state = {
            "question": "Search the latest AI news",
            "route": "research",
            "answer": "I have NOT searched the web for this yet. Approval request ID: abcdef123456",
            "sources": [],
            "qa_flags": [],
        }
        graph = type("Graph", (), {"invoke": lambda _self, _state: final_state})()
        with mock.patch("app.graph.build_graph", return_value=graph):
            # The same scoped store is used by production GraphRunner.run.
            with approval_scope(self.store, e["id"]):
                self.store.create_or_reuse_approval(
                    operation="research", description="Web search for latest AI news",
                    payload={"question": final_state["question"], "route": "research"},
                    execution_id=e["id"],
                )
                done = self.runner.run(e["id"])

        self.assertEqual(done["status"], STATUS_WAITING_APPROVAL)
        self.assertIsNone(done["error"])
        self.assertEqual(self.store.get_events(e["id"])[-1]["type"],
                         ev.EXECUTION_WAITING_APPROVAL)

    def test_pending_approval_wins_over_incomplete_plan_status(self):
        """A plan paused by a pending approval is WAITING, not failed: the
        unfinished tasks are a consequence of the pause, not a defect."""
        from app.execution.store import STATUS_WAITING_APPROVAL
        from app.graph import approval_scope

        e = self.runner.create("Research then summarize the latest AI news")
        final_state = {
            "question": "Research then summarize the latest AI news",
            "route": "research",
            "answer": "I have NOT searched the web for this yet. Approval request ID: feedbeef0001",
            "sources": [],
            "qa_flags": ["QA: plan incomplete — task(s) not completed: t2"],
            "plan": {"source": "test", "tasks": [
                {"task_id": "t1", "capability": "research", "status": "failed",
                 "error": "external web research was not authorized — an approval request is pending"},
                {"task_id": "t2", "capability": "synthesis", "status": "blocked",
                 "block_reason": "a dependency did not complete successfully"},
            ]},
        }
        graph = type("Graph", (), {"invoke": lambda _self, _state: final_state})()
        with mock.patch("app.graph.build_graph", return_value=graph):
            with approval_scope(self.store, e["id"]):
                self.store.create_or_reuse_approval(
                    operation="research",
                    description="Web search for latest AI news",
                    payload={"question": final_state["question"],
                             "route": "research"},
                    execution_id=e["id"],
                )
                done = self.runner.run(e["id"])

        self.assertEqual(done["status"], STATUS_WAITING_APPROVAL)
        self.assertIsNone(done["error"], "a paused run must not claim failure")
        self.assertIsNotNone(done["plan"], "the plan stays visible for the UI")
        last = self.store.get_events(e["id"])[-1]
        self.assertEqual(last["type"], ev.EXECUTION_WAITING_APPROVAL)
        self.assertNotEqual(last["payload"].get("category"), "incomplete_plan")

    def test_incomplete_plan_is_failed_not_completed_and_preserves_answer(self):
        e = self.runner.create("Create a multi-step report")
        final_state = {
            "question": "Create a multi-step report",
            "route": "books",
            "answer": "Partial result from completed work.",
            "sources": ["books"],
            "agent_answers": {"books": "Partial result from completed work."},
            "qa_flags": ["QA: plan incomplete — task(s) not completed: t2"],
            "plan": {"source": "test", "tasks": [
                {"task_id": "t1", "capability": "books", "status": "completed"},
                {"task_id": "t2", "capability": "synthesis", "status": "blocked",
                 "block_reason": "dependency failed"},
            ]},
        }
        from types import SimpleNamespace
        from unittest.mock import patch

        graph = SimpleNamespace(invoke=lambda _state: final_state)
        with patch("app.graph.build_graph", return_value=graph):
            done = self.runner.run(e["id"])

        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["answer"], final_state["answer"])
        self.assertEqual(done["qa_flags"], final_state["qa_flags"])
        self.assertEqual(done["error"]["category"], "incomplete_plan")
        self.assertEqual(done["error"]["details"]["incomplete_tasks"][0]["status"],
                         "blocked")
        events = self.store.get_events(e["id"])
        self.assertEqual(events[-1]["type"], ev.EXECUTION_FAILED)
        self.assertEqual(events[-1]["payload"]["category"], "incomplete_plan")

    def test_ai_failure_is_failed_not_completed(self):
        self._failing_research()
        e = self.runner.create("Latest AI news this week")
        done = self.runner.run(e["id"])
        self.assertEqual(done["status"], "failed")
        self.assertIn("AI", json.dumps(done["error"]))

    def test_failure_event_is_error_severity(self):
        self._failing_research()
        e = self.runner.create("Latest AI news this week")
        self.runner.run(e["id"])
        events = self.store.get_events(e["id"])
        failed = [x for x in events if x["severity"] == "error"]
        self.assertTrue(failed)
        self.assertIn(ev.EXECUTION_FAILED, [x["type"] for x in failed])

    def test_execution_failure_has_human_message_no_raw_trace(self):
        self._failing_research()
        e = self.runner.create("Latest AI news this week")
        done = self.runner.run(e["id"])
        self.assertNotIn("Traceback", json.dumps(done["error"]))
        self.assertNotIn("stack", json.dumps(done["error"]).lower())

    def test_events_survive_store_reopen(self):
        self._fake_books()
        e = self.runner.create("What does chapter 3 say about propolis?")
        self.runner.run(e["id"])
        before = len(self.store.get_events(e["id"]))
        self.store.close()
        reopened = ExecutionStore(self.store.db_path)
        after = len(reopened.get_events(e["id"]))
        self.assertEqual(before, after)
        reopened.close()

    def test_run_unknown_execution_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.runner.run("does-not-exist")


class TestGraphWrapperHook(unittest.TestCase):
    def test_node_wrapper_observes_every_node(self):
        import app.graph as gm
        snapshot = {n: getattr(gm, n) for n in
                    ("books_agent", "research_agent", "feasibility_agent", "erp_agent")}
        try:
            def fake_books(state):
                return {"answer": "BOK", "sources": ["books"],
                        "agent_answers": {"books": "BOK"}}

            gm.books_agent = fake_books
            seen = []

            def wrapper(name, fn):
                seen.append(name)
                return fn

            g = gm.build_graph(node_wrapper=wrapper)
            g.invoke({"question": "What does chapter 3 say about propolis?"})
            # wrapper only records reading, not wrapping; 'qa' is never wrapped
            # wrapper wraps every answer node (2026-10-28: +general
            # +unsupported; 2026-10-30: +coding, +data, +followup, +feedback
            # (the feedback node records a learning signal, so it must appear
            # in the timeline); qa is deliberately NOT wrapped
            self.assertEqual(sorted(seen), ["books", "coding", "data", "erp",
                                            "feasibility", "feedback",
                                            "followup", "general", "research",
                                            "supervisor", "unsupported"])

            # default path unchanged
            seen.clear()
            g2 = gm.build_graph()
            g2.invoke({"question": "What does chapter 3 say about propolis?"})
            self.assertEqual(seen, [])
        finally:
            for n, fn in snapshot.items():
                setattr(gm, n, fn)

    def test_human_step_names_are_jargon_free(self):
        for node in ("supervisor", "books", "research", "erp", "feasibility"):
            name = human_step_name(node)
            self.assertNotIn("_", name)
            self.assertTrue(name[0].isupper())


if __name__ == "__main__":
    unittest.main(verbosity=2)
