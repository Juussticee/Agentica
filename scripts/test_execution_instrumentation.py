"""W2 — structured execution events + persisted provenance (deterministic).

Covers, with no network (requests/_crawl faked), no LLM and no Chroma:

  * event-model constants and the version stamp consumers read;
  * the in-process sink: call ordering, no-op outside a run, and a failing
    sink / failing event write NEVER breaking the execution it describes;
  * provenance persistence (`executions.sources`) and the additive,
    backward-compatible schema migration for legacy rows;
  * operation instrumentation as it happens: books retrieval, bounded
    recovery outcome, the books->research fallback decision, and the
    search/crawl tool outcomes;
  * privacy: payloads carry identifiers, counts, durations and coarse error
    categories only — never passage text, answer text, queries, secrets or
    raw exception messages.
"""

import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.books_agent import books_agent  # noqa: E402
from app.config import NOT_FOUND_PHRASE  # noqa: E402
from app.execution import events as ev  # noqa: E402
from app.execution import instrumentation as instr  # noqa: E402
from app.execution.runner import GraphRunner  # noqa: E402
from app.execution.store import ExecutionStore  # noqa: E402
from langgraph.graph import END  # noqa: E402

ORBITAL = "orbital_kinetics_principles.docx"
ORBITAL_PASSAGE = (
    "Orbital Kinetics Principles. A Hohmann transfer between circular orbits "
    "requires two engine burns and depends on the delta-v budget of the "
    "spacecraft."
)
DELTA_QUESTION = (
    "According to orbital_kinetics_principles.docx, what does delta-v mean, "
    "and how is it used in orbital mechanics?"
)
RECOVERED = (
    "A Hohmann transfer between circular orbits requires two engine burns "
    "and depends on the delta-v budget of the spacecraft."
)
# A marker that must NEVER appear in any event payload: it is passage text.
PASSAGE_MARKER = "Hohmann transfer between circular orbits requires two engine"


def _store():
    return ExecutionStore(Path(tempfile.mkdtemp()) / "test.sqlite3")


def _node(score, text=ORBITAL_PASSAGE, file_name=ORBITAL, page="3"):
    return SimpleNamespace(score=score,
                           metadata={"file_name": file_name,
                                     "page_label": page},
                           text=text)


class _FakeResponse:
    def __init__(self, text, nodes):
        self._text = text
        self.source_nodes = nodes

    def __str__(self):
        return self._text


class _ChatMsg:
    def __init__(self, text):
        self.content = text


class _ScriptedLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return _ChatMsg(self.reply)


class TestEventModel(unittest.TestCase):
    def test_new_event_types_exist(self):
        for c in (ev.RETRIEVAL_COMPLETED, ev.TOOL_COMPLETED, ev.TOOL_FAILED,
                  ev.RECOVERY_OUTCOME, ev.FALLBACK_DECIDED):
            self.assertTrue(c)

    def test_event_schema_version_is_declared(self):
        self.assertEqual(ev.EVENT_SCHEMA_VERSION, 1)

    def test_emit_is_a_noop_outside_a_run(self):
        self.assertFalse(instr.is_recording())
        instr.emit(ev.RETRIEVAL_COMPLETED, "should not be recorded",
                   payload={"nodes": 1})      # must not raise
        self.assertFalse(instr.is_recording())

    def test_sink_receives_versioned_payload_in_call_order(self):
        seen = []
        with instr.install(lambda t, msg, **kw: seen.append((t, msg, kw))):
            instr.emit(ev.TOOL_COMPLETED, "first", payload={"a": 1})
            instr.emit(ev.TOOL_FAILED, "second")
        self.assertEqual([s[0] for s in seen],
                         [ev.TOOL_COMPLETED, ev.TOOL_FAILED])
        self.assertEqual(seen[0][2]["payload"]["a"], 1)
        for _, _, kw in seen:
            self.assertEqual(kw["payload"]["schema_version"],
                             ev.EVENT_SCHEMA_VERSION)

    def test_install_is_restored_even_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            with instr.install(lambda *a, **k: None):
                raise RuntimeError("boom")
        self.assertFalse(instr.is_recording())


class TestFailSoftBoundaries(unittest.TestCase):
    """Requirement: event bookkeeping never corrupts the main result."""

    def _run_books(self, response, question):
        import app.agents.books_agent as ba

        orig_engine, orig_llm = ba._get_query_engine, ba.get_llm
        ba._get_query_engine = lambda: SimpleNamespace(query=lambda q: response)
        ba.get_llm = lambda *a, **k: _ScriptedLLM(NOT_FOUND_PHRASE)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                return books_agent({"question": question})
        finally:
            ba._get_query_engine, ba.get_llm = orig_engine, orig_llm

    def test_a_raising_sink_never_breaks_an_agent(self):
        def boom(*args, **kwargs):
            raise RuntimeError("sink down")

        with instr.install(boom):
            out = self._run_books(
                _FakeResponse(RECOVERED, [_node(0.78)]), DELTA_QUESTION)
        self.assertIn("two engine burns", out["answer"])

    def test_a_failing_event_write_never_fails_the_run(self):
        store = _store()

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        store.append_event = boom          # every event write fails
        runner = GraphRunner(store)
        import app.graph as gm

        snapshot = {n: getattr(gm, n) for n in
                    ("books_agent", "research_agent", "feasibility_agent",
                     "erp_agent")}
        try:
            gm.books_agent = lambda state: {
                "answer": "BOK", "sources": ["books"],
                "agent_answers": {"books": "BOK"}}
            e = runner.create("What does chapter 3 say about propolis?")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                done = runner.run(e["id"])
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["answer"], "BOK")
            self.assertIn("event persistence failed", buf.getvalue())
        finally:
            for n, fn in snapshot.items():
                setattr(gm, n, fn)
            store.close()


class TestRunnerProvenanceAndOrdering(unittest.TestCase):
    def setUp(self):
        self.store = _store()
        self.runner = GraphRunner(self.store)
        import app.graph as gm

        self.gm = gm
        self._snapshot = {n: getattr(gm, n) for n in
                          ("books_agent", "research_agent", "feasibility_agent",
                           "erp_agent")}

    def tearDown(self):
        for n, fn in self._snapshot.items():
            setattr(self.gm, n, fn)
        self.store.close()

    def test_sources_are_persisted_and_returned(self):
        self.gm.books_agent = lambda state: {
            "answer": "BOK", "sources": ["books"],
            "agent_answers": {"books": "BOK"}}
        e = self.runner.create("What does chapter 3 say about propolis?")
        done = self.runner.run(e["id"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["sources"], ["books"])
        listed = {row["id"]: row for row in self.store.list_executions()}
        self.assertEqual(listed[e["id"]]["sources"], ["books"])
        completed = [x for x in self.store.get_events(e["id"])
                     if x["type"] == ev.EXECUTION_COMPLETED]
        self.assertTrue(completed)
        self.assertEqual(completed[0]["payload"]["sources"], ["books"])

    def test_research_provenance_urls_are_persisted(self):
        self.gm.research_agent = lambda state: {
            "answer": "ANS", "sources": ["research", "https://example.org/a"],
            "agent_answers": {"research": "ANS"}}
        e = self.runner.create("Latest AI news this week")
        done = self.runner.run(e["id"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["sources"],
                         ["research", "https://example.org/a"])

    def test_operation_events_interleave_with_node_lifecycle(self):
        def fake_books(state):
            instr.emit(ev.RETRIEVAL_COMPLETED,
                       "Retrieved 1 supporting passage(s)",
                       agent_id="books", step_id="books",
                       payload={"nodes": 1, "score": 0.78})
            return {"answer": "BOK", "sources": ["books"],
                    "agent_answers": {"books": "BOK"}}

        self.gm.books_agent = fake_books
        e = self.runner.create("What does chapter 3 say about propolis?")
        self.runner.run(e["id"])
        events = self.store.get_events(e["id"])

        ids = [x["event_id"] for x in events]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)), "event ids must not collide")

        idx = [i for i, x in enumerate(events)
               if x["type"] == ev.RETRIEVAL_COMPLETED]
        self.assertEqual(len(idx), 1)
        i = idx[0]
        self.assertEqual(events[i - 1]["type"], ev.AGENT_STARTED)
        self.assertEqual(events[i - 1]["agent_id"], "books")
        self.assertEqual(events[i + 1]["type"], ev.AGENT_COMPLETED)
        self.assertEqual(events[i + 1]["agent_id"], "books")

    def test_completed_event_precedes_nothing_after_it(self):
        self.gm.books_agent = lambda state: {
            "answer": "BOK", "sources": ["books"],
            "agent_answers": {"books": "BOK"}}
        e = self.runner.create("What does chapter 3 say about propolis?")
        self.runner.run(e["id"])
        types = [x["type"] for x in self.store.get_events(e["id"])]
        self.assertEqual(types[-1], ev.EXECUTION_COMPLETED)


class TestLegacySchemaCompatibility(unittest.TestCase):
    def test_legacy_database_is_migrated_additively(self):
        path = Path(tempfile.mkdtemp()) / "legacy.sqlite3"
        conn = sqlite3.connect(str(path))
        conn.executescript(
            """CREATE TABLE executions (
                id TEXT PRIMARY KEY, question TEXT NOT NULL,
                status TEXT NOT NULL, route TEXT, answer TEXT,
                qa_flags TEXT, error TEXT, created_at TEXT NOT NULL,
                started_at TEXT, completed_at TEXT, result TEXT, plan TEXT);
               CREATE TABLE events (
                execution_id TEXT NOT NULL, event_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL, type TEXT NOT NULL,
                message TEXT NOT NULL, step_id TEXT, agent_id TEXT,
                severity TEXT NOT NULL DEFAULT 'info', payload TEXT,
                PRIMARY KEY (execution_id, event_id));
               INSERT INTO executions (id, question, status, created_at,
                    answer, result)
                VALUES ('legacy1', 'old question', 'completed',
                        '2026-10-01T00:00:00+00:00', 'old answer',
                        'old answer');
               INSERT INTO events VALUES ('legacy1', 1,
                    '2026-10-01T00:00:00+00:00', 'execution.created',
                    'Execution created', NULL, NULL, 'info', '{}');"""
        )
        conn.commit()
        conn.close()

        store = ExecutionStore(path)
        try:
            row = store.get_execution("legacy1")     # must not raise
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["answer"], "old answer")
            # pre-existing provenance is NULL, reported as None — never a
            # fabricated empty value pretending provenance was recorded.
            self.assertIsNone(row["sources"])
            self.assertEqual([x["event_id"] for x in store.get_events("legacy1")],
                             [1])
            # new writes work on the migrated schema
            store.update_execution("legacy1", sources=["books", "research"])
            self.assertEqual(store.get_execution("legacy1")["sources"],
                             ["books", "research"])
        finally:
            store.close()

    def test_corrupt_sources_column_degrades_to_none(self):
        store = _store()
        store.create_execution("e1", "q")
        store._conn.execute("UPDATE executions SET sources = 'not-json' "
                            "WHERE id = 'e1'")
        store._conn.commit()
        try:
            self.assertIsNone(store.get_execution("e1")["sources"])
        finally:
            store.close()


class BooksInstrumentationBase(unittest.TestCase):
    def _run(self, response, question, reply=NOT_FOUND_PHRASE, sink=None):
        """Run the REAL books_agent with a scripted engine/LLM; returns
        (out, events) where events were captured during the run."""
        import app.agents.books_agent as ba

        orig_engine, orig_llm = ba._get_query_engine, ba.get_llm
        captured = []

        def capture(t, msg, **kw):
            captured.append((t, msg, kw))

        ba._get_query_engine = lambda: SimpleNamespace(query=lambda q: response)
        ba.get_llm = lambda *a, **k: _ScriptedLLM(reply)
        try:
            with instr.install(capture):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    out = books_agent({"question": question})
        finally:
            ba._get_query_engine, ba.get_llm = orig_engine, orig_llm
        return out, captured, buf.getvalue()


class TestBooksInstrumentation(BooksInstrumentationBase):
    def test_retrieval_event_records_identifiers_only(self):
        out, captured, _ = self._run(
            _FakeResponse(RECOVERED, [_node(0.78)]), DELTA_QUESTION)
        retrieval = [c for c in captured if c[0] == ev.RETRIEVAL_COMPLETED]
        self.assertEqual(len(retrieval), 1)
        payload = retrieval[0][2]["payload"]
        self.assertEqual(payload["nodes"], 1)
        self.assertEqual(payload["score"], 0.78)
        self.assertIn("duration_ms", payload)
        self.assertEqual(payload["chunks"],
                         [{"file_name": ORBITAL, "page_label": "3",
                           "score": 0.78}])
        # identifiers and scores only — never the passage itself
        self.assertNotIn(PASSAGE_MARKER, json.dumps(payload))

    def test_refusal_records_recovery_outcome_when_retry_is_denied(self):
        # Off-topic question: coverage denies the retry, and the denial is
        # now a durable event instead of only an absent log line.
        out, captured, _ = self._run(
            _FakeResponse(NOT_FOUND_PHRASE, [_node(0.78)]),
            "What is quantum tunneling in particle physics?")
        outcomes = [c for c in captured if c[0] == ev.RECOVERY_OUTCOME]
        self.assertEqual(len(outcomes), 1)
        payload = outcomes[0][2]["payload"]
        self.assertEqual(payload["outcome"], "skipped_not_eligible")
        self.assertIsNone(payload["reason"])
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE))

    def test_refusal_records_successful_recovery(self):
        out, captured, _ = self._run(
            _FakeResponse(NOT_FOUND_PHRASE, [_node(0.78)]),
            DELTA_QUESTION, reply=RECOVERED)
        outcomes = [c for c in captured if c[0] == ev.RECOVERY_OUTCOME]
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0][2]["payload"]["outcome"], "succeeded")
        self.assertIn("two engine burns", out["answer"])

    def test_refusal_records_rejected_recovery_with_reason(self):
        out, captured, _ = self._run(
            _FakeResponse(NOT_FOUND_PHRASE, [_node(0.78)]),
            DELTA_QUESTION, reply=NOT_FOUND_PHRASE)
        outcomes = [c for c in captured if c[0] == ev.RECOVERY_OUTCOME]
        self.assertEqual(len(outcomes), 1)
        payload = outcomes[0][2]["payload"]
        self.assertEqual(payload["outcome"], "rejected")
        self.assertEqual(payload["reason"], "second_refusal")
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE))

    def test_answered_run_without_refusal_records_no_recovery_event(self):
        out, captured, _ = self._run(
            _FakeResponse(RECOVERED, [_node(0.78)]), DELTA_QUESTION)
        self.assertFalse([c for c in captured
                          if c[0] == ev.RECOVERY_OUTCOME])
        self.assertIn("two engine burns", out["answer"])

    def test_no_payload_contains_answer_text_or_secrets(self):
        out, captured, _ = self._run(
            _FakeResponse(RECOVERED, [_node(0.78)]), DELTA_QUESTION)
        blob = json.dumps([c[2] for c in captured]).lower()
        self.assertNotIn(PASSAGE_MARKER.lower(), blob)
        self.assertNotIn("two engine burns", blob)
        self.assertNotIn("google_api_key", blob)
        self.assertNotIn("api_key", blob)


class TestFallbackDecisionEvent(unittest.TestCase):
    def setUp(self):
        import app.graph as gm

        self.gm = gm
        self._after_books = gm._after_books

    def _state(self, question, reason="self_refusal_after_recovery"):
        return {"question": question, "route": "books",
                "books_refusal": {"reason": reason, "score": 0.75,
                                  "nodes": 3, "recovery": "rejected"},
                "agent_answers": {"books": NOT_FOUND_PHRASE},
                "sources": ["books"]}

    def _capture(self, state):
        captured = []

        def sink(t, msg, **kw):
            captured.append((t, msg, kw))

        with instr.install(sink):
            nxt = self._after_books(state)
        return nxt, captured

    def test_allowed_fallback_is_recorded_with_its_code(self):
        nxt, captured = self._capture(
            self._state("What does the NIST AI RMF say about X?"))
        self.assertEqual(nxt, "research")
        events = [c for c in captured if c[0] == ev.FALLBACK_DECIDED]
        self.assertEqual(len(events), 1)
        payload = events[0][2]["payload"]
        self.assertTrue(payload["allowed"])
        self.assertTrue(payload["code"].startswith("books_insufficient:"))
        self.assertEqual(payload["schema_version"], ev.EVENT_SCHEMA_VERSION)

    def test_denied_local_only_fallback_is_recorded_too(self):
        nxt, captured = self._capture(
            self._state("Using only this document, what does it say about X?"))
        self.assertEqual(nxt, END)
        events = [c for c in captured if c[0] == ev.FALLBACK_DECIDED]
        self.assertEqual(len(events), 1)
        payload = events[0][2]["payload"]
        self.assertFalse(payload["allowed"])
        self.assertIn("local_only_request", payload["code"])

    def test_answered_books_run_records_no_fallback_decision(self):
        state = {"question": "What does chapter 3 say about propolis?",
                 "route": "books", "sources": ["books"],
                 "agent_answers": {"books": "answer"}}
        nxt, captured = self._capture(state)
        self.assertEqual(nxt, END)
        self.assertFalse([c for c in captured
                          if c[0] == ev.FALLBACK_DECIDED])


class TestToolEvents(unittest.TestCase):
    def test_search_success_records_count_and_duration_only(self):
        from app.tools import searxng_client as sc

        response = mock.Mock()
        response.raise_for_status = lambda: None
        response.json = lambda: {"results": [
            {"title": "T1", "url": "https://example.org/a",
             "content": "snippet text"}]}
        captured = []

        def sink(t, msg, **kw):
            captured.append((t, msg, kw))

        with mock.patch.object(sc.requests, "get", return_value=response):
            with instr.install(sink):
                results = sc.search("the original user question", num_results=3)

        self.assertEqual(len(results), 1)
        done = [c for c in captured if c[0] == ev.TOOL_COMPLETED]
        self.assertEqual(len(done), 1)
        payload = done[0][2]["payload"]
        self.assertEqual(payload["tool"], "searxng.search")
        self.assertEqual(payload["result_count"], 1)
        self.assertIn("duration_ms", payload)
        blob = json.dumps(payload)
        # the query is the user's question — already stored on the run; it
        # is never duplicated into an event payload.
        self.assertNotIn("the original user question", blob)
        self.assertNotIn("snippet text", blob)

    def test_search_failure_records_category_not_message(self):
        from app.tools import searxng_client as sc

        captured = []

        def sink(t, msg, **kw):
            captured.append((t, msg, kw))

        with mock.patch.object(sc.requests, "get",
                               side_effect=RuntimeError(
                                   "Read timed out for "
                                   "http://localhost:8888/search?q=SECRETQ")):
            with instr.install(sink):
                results = sc.search("SECRETQ")

        self.assertEqual(results, [])
        failed = [c for c in captured if c[0] == ev.TOOL_FAILED]
        self.assertEqual(len(failed), 1)
        payload = failed[0][2]["payload"]
        self.assertEqual(payload["tool"], "searxng.search")
        self.assertEqual(payload["error_type"], "RuntimeError")
        blob = json.dumps(payload)
        self.assertNotIn("SECRETQ", blob)          # no query echo
        self.assertNotIn("Read timed out", blob)   # no raw message

    def test_crawl_success_records_host_and_size_not_content(self):
        from app.tools import crawl_extractor as ce

        page_text = "Full page body that must not be stored. " * 20
        captured = []

        def sink(t, msg, **kw):
            captured.append((t, msg, kw))

        with mock.patch.object(ce, "_crawl", return_value=page_text):
            with instr.install(sink):
                text = ce.extract("https://example.org/article?id=42")

        self.assertTrue(text)
        done = [c for c in captured if c[0] == ev.TOOL_COMPLETED]
        self.assertEqual(len(done), 1)
        payload = done[0][2]["payload"]
        self.assertEqual(payload["tool"], "crawl.extract")
        self.assertEqual(payload["host"], "example.org")
        self.assertEqual(payload["chars"], len(text))   # stripped output
        blob = json.dumps(payload)
        self.assertNotIn("Full page body", blob)
        self.assertNotIn("id=42", blob)

    def test_crawl_failure_records_category_only(self):
        from app.tools import crawl_extractor as ce

        captured = []

        def sink(t, msg, **kw):
            captured.append((t, msg, kw))

        with mock.patch.object(ce, "_crawl",
                               side_effect=RuntimeError("boom with SECRETQ")):
            with instr.install(sink):
                text = ce.extract("https://example.org/article?q=SECRETQ")

        self.assertEqual(text, "")
        failed = [c for c in captured if c[0] == ev.TOOL_FAILED]
        self.assertEqual(len(failed), 1)
        payload = failed[0][2]["payload"]
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertNotIn("SECRETQ", json.dumps(payload))
        self.assertNotIn("boom", json.dumps(payload))

    def test_tools_are_a_noop_without_a_run(self):
        from app.tools import searxng_client as sc

        response = mock.Mock()
        response.raise_for_status = lambda: None
        response.json = lambda: {"results": []}
        with mock.patch.object(sc.requests, "get", return_value=response):
            self.assertEqual(sc.search("q"), [])   # no sink installed


class TestEventWriteObservability(unittest.TestCase):
    """W2.1 item 4: event-storage degradation is visible through the
    EXISTING storage health row, without failing a successful run."""

    def test_a_failing_event_write_is_counted_while_the_run_stays_green(self):
        store = _store()

        class _EventsBrokenConn:
            """Delegates everything EXCEPT INSERT INTO events, so the REAL
            append_event fails inside its own try/except (no monkeypatch of
            the method being tested)."""

            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args, **kwargs):
                if sql.lstrip().upper().startswith("INSERT INTO EVENTS"):
                    raise sqlite3.OperationalError("disk I/O error")
                return self._conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        runner = GraphRunner(store)
        import app.graph as gm

        snapshot = {n: getattr(gm, n) for n in
                    ("books_agent", "research_agent", "feasibility_agent",
                     "erp_agent")}
        real_conn = store._conn
        try:
            store._conn = _EventsBrokenConn(real_conn)
            gm.books_agent = lambda state: {
                "answer": "BOK", "sources": ["books"],
                "agent_answers": {"books": "BOK"}}
            e = runner.create("What does chapter 3 say about propolis?")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                done = runner.run(e["id"])
            # the answer lifecycle is untouched ...
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["answer"], "BOK")
            self.assertIn("event persistence failed", buf.getvalue())
            # ... and the degradation is now counted, not just printed
            self.assertGreater(store.event_write_failures, 0)
        finally:
            store._conn = real_conn
            for n, fn in snapshot.items():
                setattr(gm, n, fn)
            store.close()

    def test_storage_health_row_reports_the_write_failures(self):
        from app import services_health

        store = _store()
        try:
            healthy = services_health.check_storage(store)
            self.assertEqual(healthy["config"]["event_write_failures"], 0)
            store.event_write_failures = 3      # simulated observed failures
            degraded = services_health.check_storage(store)
            self.assertEqual(degraded["config"]["event_write_failures"], 3)
            if healthy["state"] == "running":
                # only when the underlying database itself is healthy does
                # the write counter get to express itself as `degraded`
                self.assertEqual(degraded["state"], "degraded")
                self.assertIn("3 execution event write(s) failed",
                              degraded["reason"])
                self.assertIn("timeline", degraded["next_action"] +
                              degraded["reason"])
                self.assertNotIn("error", degraded)  # no failure object
        finally:
            store.close()

    def test_services_endpoint_hands_the_live_store_to_the_probe(self):
        from app import services_health as sh_module
        import app.api.server as server
        from fastapi.testclient import TestClient

        store = _store()
        captured = {}

        def fake_snapshot(store=None):
            captured["store"] = store
            return {"checked_at": None, "services": []}

        original = server._store
        server._store = store
        try:
            with mock.patch.object(sh_module, "services_snapshot",
                                   fake_snapshot):
                client = TestClient(server.build_server())
                resp = client.get("/api/v1/services")
            self.assertEqual(resp.status_code, 200)
            self.assertIs(captured["store"], store)
        finally:
            server._store = original
            store.close()


class TestSinkIsolation(unittest.TestCase):
    """W2.1: one execution's events must never reach another execution.

    Synchronization is by `threading.Barrier` only — no sleeps, no timing
    assumptions: every assertion holds whatever the OS scheduler does,
    because the threads only ever pass a barrier after the state under test
    has been established.
    """

    def test_overlapping_sinks_each_receive_only_their_own_events(self):
        both_installed = threading.Barrier(2, timeout=10)
        both_emitted = threading.Barrier(2, timeout=10)
        seen = {"A": [], "B": []}
        errors = []

        def worker(tag):
            def sink(t, msg, **kw):
                seen[tag].append((t, msg, kw))

            try:
                with instr.install(sink):
                    both_installed.wait()      # A and B are BOTH recording
                    instr.emit(ev.TOOL_COMPLETED, f"{tag}-in",
                               payload={"run": tag})
                    both_emitted.wait()        # ... and both have emitted
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(tag,),
                                    name=f"run-{tag}")
                   for tag in ("A", "B")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
            self.assertFalse(t.is_alive(), "worker deadlocked on a barrier")
        self.assertEqual(errors, [])

        # only its own event, never the other run's
        self.assertEqual([m for _, m, _ in seen["A"]], ["A-in"])
        self.assertEqual([m for _, m, _ in seen["B"]], ["B-in"])

        # nothing is recording any more and no stale sink survives: an emit
        # outside a run must not be written into a finished execution.
        self.assertFalse(instr.is_recording())
        instr.emit(ev.TOOL_COMPLETED, "late", payload={"run": "main"})
        self.assertEqual([m for _, m, _ in seen["A"]], ["A-in"])
        self.assertEqual([m for _, m, _ in seen["B"]], ["B-in"])

    def test_nested_install_restores_the_outer_sink(self):
        outer, inner = [], []

        def record(target):
            return lambda t, msg, **kw: target.append(msg)

        with instr.install(record(outer)):
            instr.emit(ev.TOOL_COMPLETED, "first")
            with instr.install(record(inner)):
                instr.emit(ev.TOOL_COMPLETED, "nested")
            instr.emit(ev.TOOL_COMPLETED, "outer-again")
        self.assertEqual(outer, ["first", "outer-again"])
        self.assertEqual(inner, ["nested"])
        self.assertFalse(instr.is_recording())

    def test_an_exception_inside_a_nested_install_still_restores(self):
        outer = []
        with instr.install(lambda t, msg, **kw: outer.append(msg)):
            with self.assertRaises(RuntimeError):
                with instr.install(lambda *a, **k: None):
                    instr.emit(ev.TOOL_COMPLETED, "while-nested")
                    raise RuntimeError("boom")
            instr.emit(ev.TOOL_COMPLETED, "after-nested")
        self.assertEqual(outer, ["after-nested"])
        self.assertFalse(instr.is_recording())


class TestConcurrentRunnerIsolation(unittest.TestCase):
    """End-to-end: two overlapping `runner.run()` calls on ONE shared
    runner+store keep strictly separate event streams (W2.1)."""

    def setUp(self):
        self.store = _store()
        self.runner = GraphRunner(self.store)   # shared: worst case

    def tearDown(self):
        self.store.close()

    def test_two_overlapping_runs_each_get_only_their_own_events(self):
        import app.graph as gm

        both_inside = threading.Barrier(2, timeout=10)
        both_passed = threading.Barrier(2, timeout=10)

        def fake_build(node_wrapper=None):
            class _Graph:
                def invoke(self, state):
                    q = state["question"]
                    instr.emit(ev.RETRIEVAL_COMPLETED, "mid-run",
                               agent_id="books", step_id="books",
                               payload={"question": q})
                    both_inside.wait()   # both runs are mid-node together
                    both_passed.wait()
                    instr.emit(ev.TOOL_COMPLETED, "tail-run",
                               payload={"question": q})
                    return {"answer": f"ANSWER:{q}", "route": "books",
                            "qa_flags": [], "sources": ["books"]}
            return _Graph()

        questions = {"A": "question about the alpha topic",
                     "B": "question about the beta topic"}
        results, errors = {}, {}

        def run(tag):
            try:
                created = self.runner.create(questions[tag])
                results[tag] = self.runner.run(created["id"])
            except Exception as exc:  # noqa: BLE001
                errors[tag] = exc

        with mock.patch.object(gm, "build_graph", fake_build):
            threads = [threading.Thread(target=run, args=(tag,),
                                        name=f"exec-{tag}")
                       for tag in ("A", "B")]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=20)
                self.assertFalse(t.is_alive(), "run deadlocked")

        self.assertEqual(errors, {})
        self.assertFalse(instr.is_recording(), "stale sink left behind")

        op_types = (ev.RETRIEVAL_COMPLETED, ev.TOOL_COMPLETED)
        for tag in ("A", "B"):
            with self.subTest(run=tag):
                eid = results[tag]["id"]
                self.assertEqual(results[tag]["status"], "completed")
                ops = [x for x in self.store.get_events(eid)
                       if x["type"] in op_types]
                # both of its own events, none lost, none borrowed
                self.assertEqual(len(ops), 2, [x["type"] for x in ops])
                for event in ops:
                    self.assertEqual(event["payload"]["question"],
                                     questions[tag], event)
                    self.assertEqual(event["payload"]["schema_version"],
                                     ev.EVENT_SCHEMA_VERSION)

    def test_emit_after_all_runs_is_recorded_nowhere(self):
        import app.graph as gm

        def fake_build(node_wrapper=None):
            class _Graph:
                def invoke(self, state):
                    return {"answer": "done", "route": "books",
                            "qa_flags": [], "sources": []}
            return _Graph()

        with mock.patch.object(gm, "build_graph", fake_build):
            created = self.runner.create("a question worth running")
            done = self.runner.run(created["id"])

        before = len(self.store.get_events(done["id"]))
        self.assertFalse(instr.is_recording())
        instr.emit(ev.TOOL_COMPLETED, "should vanish",
                   payload={"question": "not an execution"})
        self.assertEqual(len(self.store.get_events(done["id"])), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
