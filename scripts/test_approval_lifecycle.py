"""GUI-10 approval lifecycle (2026-10-28) — persisted, decision-bound, offline.

Proves the full backend contract that connects the fail-closed research gate
to a persisted ApprovalRequest (CONTROL_CENTER_PLAN §12), covering every
required case:

  store    : create/retrieve pending, approve, reject, unknown ids, duplicate
             and competing decisions, expiry, reopen persistence, constrained
             operation types;
  gate     : books fallback creates (and reuses) a pending request with its
             id in the answer; pending/rejected/expired/mismatched/missing
             approvals never execute; a valid approval runs ONLY its bound
             operation, exactly once; store failure => denial;
  runner   : resume executes an approved operation, honors rejection with an
             honest local-only close, refuses pending/expired/unknown;
  API      : GET/POST contracts and 404/409/410/422 error semantics;
  unchanged: local RAG needs no approval, routing pins still hold.

Every external seam (SearXNG, crawl) is replaced module-wide for this suite,
so ANY research execution attempt outside a faked node fails the run.
"""

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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

import app.agents.books_agent as ba  # noqa: E402
import app.graph as gm  # noqa: E402
import app.supervisor as sup  # noqa: E402
from app.config import NOT_FOUND_PHRASE  # noqa: E402
from app.execution.runner import GraphRunner  # noqa: E402
from app.execution.store import (  # noqa: E402
    ApprovalConflict,
    ApprovalDecisionRequired,
    ApprovalExpired,
    ApprovalNotFound,
    ExecutionStore,
)
from app.execution.store import APPROVAL_PENDING, APPROVAL_REJECTED  # noqa: E402
from app.state import with_agent_output  # noqa: E402
from app.tools import crawl_extractor, searxng_client  # noqa: E402

QUESTION = "What year was the AI Risk Management Framework released?"
REFUSAL = {"reason": "low_confidence", "score": 0.30, "nodes": 1,
           "recovery": "skipped_not_eligible"}
LOCAL_ANSWER = "I could not find that in your indexed documents."
RESEARCH_URL = "https://example.org/rmf"


def _blocked(*_args, **_kwargs):
    raise AssertionError(
        "network blocked: approval lifecycle tests must make zero external "
        "requests — a real search here means the gate failed")


_ORIG_SEAMS: dict = {}


def setUpModule():
    _ORIG_SEAMS["search"] = searxng_client.search
    _ORIG_SEAMS["extract"] = crawl_extractor.extract
    searxng_client.search = _blocked
    crawl_extractor.extract = _blocked


def tearDownModule():
    searxng_client.search = _ORIG_SEAMS["search"]
    crawl_extractor.extract = _ORIG_SEAMS["extract"]


def _tmp_store(ttl: int | None = None) -> ExecutionStore:
    store = ExecutionStore(Path(tempfile.mkdtemp()) / "approvals.sqlite3")
    if ttl is not None:
        store.approval_ttl_seconds = ttl
    return store


def _payload(question: str = QUESTION, route: str = "books",
             refusal: dict | None = REFUSAL,
             local: str | None = LOCAL_ANSWER) -> dict:
    payload = {"question": question, "route": route, "books_refusal": refusal}
    if refusal and local:
        payload["local_answer"] = local
    return payload


def _task(task_id: str, capability: str, depends_on, status: str,
          **extra) -> dict:
    task = {"task_id": task_id, "capability": capability,
            "objective": task_id, "depends_on": list(depends_on),
            "status": status}
    task.update(extra)
    return task


APPROVAL_BLOCKED_ERROR = (
    f"{gm.APPROVAL_BLOCKED_TASK_PREFIX} "
    "(no_research_authorization_in_this_request) — an approval request "
    "is pending"
)


class _ResearchRecorder:
    """Stands in for the research node; counts executions."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, state):
        self.calls.append(dict(state))
        return with_agent_output(state, "research", "EXTERNAL_RESEARCH_ANSWER",
                                 ["research", RESEARCH_URL])


class _ArgumentRecorder:
    """Stands in for a plan-dispatched node; counts executions and can fail."""

    def __init__(self, name: str, answer: str | None = None,
                 error: Exception | None = None, calls: list | None = None):
        self.name = name
        self.answer = answer or f"{name.upper()}_ANSWER"
        self.error = error
        self.calls = calls if calls is not None else []

    def __call__(self, state):
        self.calls.append(dict(state))
        if self.error is not None:
            raise self.error
        return with_agent_output(state, self.name, self.answer, [self.name])


class _GraphPatchCase(unittest.TestCase):
    def setUp(self):
        self.recorder = _ResearchRecorder()
        self._orig_research = gm.research_agent
        gm.research_agent = self.recorder
        self.addCleanup(self._restore)

    def _restore(self):
        gm.research_agent = self._orig_research


# ---------------------------------------------------------------------------
# 1. Store lifecycle
# ---------------------------------------------------------------------------


class TestStoreLifecycle(unittest.TestCase):
    def setUp(self):
        self.store = _tmp_store()
        self.addCleanup(self.store.close)

    def _create(self, execution_id="ex1", **kw):
        # approvals bind to a real execution row (FK enforced by design).
        # create_execution is INSERT OR REPLACE + ON DELETE CASCADE, so it
        # must only run once per id — re-running would delete approvals.
        if execution_id and self.store.get_execution(execution_id) is None:
            self.store.create_execution(execution_id, QUESTION)
        kwargs = dict(operation="research",
                      description="Web search (up to 8 results) for: " + QUESTION,
                      payload=_payload(),
                      execution_id=execution_id)
        kwargs.update(kw)
        return self.store.create_or_reuse_approval(**kwargs)

    def test_pending_request_created_and_retrievable(self):
        created = self._create()
        self.assertEqual(created["status"], APPROVAL_PENDING)
        self.assertEqual(created["operation"], "research")
        self.assertEqual(created["execution_id"], "ex1")
        self.assertEqual(len(created["approval_id"]), 12)
        int(created["approval_id"], 16)          # unguessable hex token
        self.assertFalse(created["reused"])
        self.assertTrue(created["created_at"] and created["expires_at"])
        self.assertIsNone(created["decided_at"])
        self.assertIsNone(created["executed_at"])
        self.assertIn(QUESTION, created["description"])
        self.assertEqual(created["payload"]["question"], QUESTION)

        fetched = self.store.get_approval(created["approval_id"])
        self.assertEqual(fetched["approval_id"], created["approval_id"])
        self.assertEqual(fetched["payload"], _payload())
        listed = self.store.list_approvals(APPROVAL_PENDING)
        self.assertEqual([a["approval_id"] for a in listed],
                         [created["approval_id"]])

    def test_duplicate_delivery_reuses_the_same_request(self):
        first = self._create()
        second = self._create()
        self.assertEqual(first["approval_id"], second["approval_id"])
        self.assertTrue(second["reused"])
        self.assertEqual(len(self.store.list_approvals()), 1)
        # a DIFFERENT execution or bound payload is a different operation
        other_exec = self._create(execution_id="ex2")
        self.assertNotEqual(other_exec["approval_id"], first["approval_id"])
        other_payload = self._create(payload=_payload(question="Other question"))
        self.assertNotEqual(other_payload["approval_id"], first["approval_id"])
        self.assertEqual(len(self.store.list_approvals()), 3)

    def test_only_research_operations_are_supported(self):
        with self.assertRaises(ValueError):
            self._create(operation="delete_files")
        with self.assertRaises(ValueError):     # payload must bind a question
            self._create(payload={"route": "books"})
        with self.assertRaises(ValueError):     # description is mandatory
            self._create(description="   ")

    def test_approving_the_exact_pending_request(self):
        created = self._create()
        decided = self.store.decide_approval(created["approval_id"], "approved")
        self.assertEqual(decided["status"], "approved")
        self.assertEqual(decided["decision"], "approved")
        self.assertTrue(decided["decided_at"])
        self.assertFalse(decided["already_decided"])
        self.assertEqual(self.store.get_approval(
            created["approval_id"])["status"], "approved")

    def test_rejecting_a_pending_request(self):
        created = self._create()
        decided = self.store.decide_approval(created["approval_id"], "rejected")
        self.assertEqual(decided["status"], APPROVAL_REJECTED)
        self.assertEqual(decided["decision"], "rejected")

    def test_unknown_identifiers_fail_with_not_found(self):
        self.assertIsNone(self.store.get_approval("ffffffffffff"))
        with self.assertRaises(ApprovalNotFound):
            self.store.decide_approval("ffffffffffff", "approved")
        with self.assertRaises(ValueError):
            self.store.decide_approval(self._create()["approval_id"],
                                       "maybe")

    def test_duplicate_decisions_are_idempotent_and_never_overwritten(self):
        created = self._create()
        aid = created["approval_id"]
        first = self.store.decide_approval(aid, "rejected")
        repeat = self.store.decide_approval(aid, "rejected")
        self.assertTrue(repeat["already_decided"])
        self.assertEqual(repeat["decided_at"], first["decided_at"])
        self.assertEqual(repeat["status"], APPROVAL_REJECTED)
        # the OPPOSITE decision must never replace the recorded one
        with self.assertRaises(ApprovalConflict):
            self.store.decide_approval(aid, "approved")
        self.assertEqual(self.store.get_approval(aid)["status"],
                         APPROVAL_REJECTED)

    def test_expired_request_cannot_be_approved(self):
        self.store.approval_ttl_seconds = -1
        created = self._create()
        with self.assertRaises(ApprovalExpired):
            self.store.decide_approval(created["approval_id"], "approved")
        self.assertEqual(self.store.get_approval(
            created["approval_id"])["status"], "expired")
        # terminal: a later decision cannot resurrect it
        with self.assertRaises(ApprovalConflict):
            self.store.decide_approval(created["approval_id"], "rejected")

    def test_competing_decisions_have_exactly_one_winner(self):
        aid = self._create()["approval_id"]
        results: dict = {}
        barrier = threading.Barrier(2)

        def decide(outcome):
            barrier.wait()
            try:
                results[outcome] = ("ok",
                                    self.store.decide_approval(aid, outcome))
            except ApprovalConflict:
                results[outcome] = ("conflict", None)

        threads = [threading.Thread(target=decide, args=(o,))
                   for o in ("approved", "rejected")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        statuses = sorted(kind for kind, _ in results.values())
        self.assertEqual(statuses, ["conflict", "ok"],
                         f"exactly one decision must win: {results}")
        final = self.store.get_approval(aid)
        self.assertIn(final["status"], ("approved", "rejected"))
        self.assertEqual(final["decision"], final["status"])
        self.assertTrue(final["decided_at"])

    def test_persistence_survives_store_reopen(self):
        created = self._create()
        decided = self.store.decide_approval(created["approval_id"], "approved")
        path = self.store.db_path
        self.store.close()
        reopened = ExecutionStore(path)
        self.addCleanup(reopened.close)
        row = reopened.get_approval(created["approval_id"])
        self.assertEqual(row["status"], "approved")
        self.assertEqual(row["decided_at"], decided["decided_at"])
        self.assertEqual(row["payload"]["question"], QUESTION)
        self.assertEqual(reopened.list_approvals("approved")[0]["approval_id"],
                         created["approval_id"])

    def test_claims_are_single_execution(self):
        aid = self._create()["approval_id"]
        self.assertFalse(self.store.claim_approval_execution(aid),
                         "cannot claim a pending approval")
        self.store.decide_approval(aid, "approved")
        self.assertTrue(self.store.claim_approval_execution(aid))
        self.assertFalse(self.store.claim_approval_execution(aid),
                         "one approval may execute exactly once")
        self.store.release_approval_execution(aid)
        self.assertTrue(self.store.claim_approval_execution(aid))

    # -- durable resume lease (2026-10-29) --------------------------------

    def _shared(self):
        path = Path(tempfile.mkdtemp()) / "shared_claim.sqlite3"
        first = ExecutionStore(path)
        second = ExecutionStore(path)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        first.create_execution("ex1", QUESTION)
        return first, second

    def _expire_lease(self, store, execution_id="ex1"):
        """Leave the lease in the past, as a crashed holder would."""
        with store._lock:
            store._conn.execute(
                "UPDATE executions SET resume_lease_expires_at = ? "
                "WHERE id = ?", ("2000-01-01T00:00:00+00:00", execution_id))
            store._conn.commit()

    def test_resume_claim_excludes_a_second_store_connection(self):
        first, second = self._shared()

        with first.resume_claim("ex1") as claimed:
            self.assertTrue(claimed)
            with second.resume_claim("ex1") as other:
                self.assertFalse(
                    other, "a second connection/process must not resolve the "
                           "same execution concurrently")

        # Released on exit, so a later retry from ANY connection can run.
        with second.resume_claim("ex1") as acquired:
            self.assertTrue(acquired)
            with first.resume_claim("ex1") as other:
                self.assertFalse(other)

    def test_expired_lease_is_taken_over_and_stale_release_is_a_noop(self):
        first, second = self._shared()

        outer = first.resume_claim("ex1")
        self.assertTrue(outer.__enter__())
        self.addCleanup(lambda: outer.__exit__(None, None, None))
        self._expire_lease(first)

        inner = second.resume_claim("ex1")
        self.assertTrue(inner.__enter__(),
                        "an expired lease bounds a crashed holder")
        self.addCleanup(lambda: inner.__exit__(None, None, None))

        # The stale owner must not clear the NEW holder's claim.
        outer.__exit__(None, None, None)
        third = second.resume_claim("ex1")
        self.assertFalse(third.__enter__(),
                         "the new owner's live lease must survive a stale "
                         "release")
        third.__exit__(None, None, None)

    def test_legacy_database_gains_the_lease_columns(self):
        path = Path(tempfile.mkdtemp()) / "legacy.sqlite3"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE executions (id TEXT PRIMARY KEY, "
            "question TEXT NOT NULL, status TEXT NOT NULL, "
            "created_at TEXT NOT NULL)")
        conn.commit()
        conn.close()

        store = ExecutionStore(path)
        self.addCleanup(store.close)
        store.create_execution("ex1", QUESTION)
        with store.resume_claim("ex1") as claimed:
            self.assertTrue(claimed)


# ---------------------------------------------------------------------------
# 2. Gate: pending creation (incl. the books fallback) and validation
# ---------------------------------------------------------------------------


class TestGateCreatesPendingRequests(_GraphPatchCase):
    def test_books_fallback_graph_run_creates_a_pending_request(self):
        """The REAL compiled graph: books refuses -> the fallback edge fires
        -> the gate persists a pending request and answers with its id.
        Zero model calls (score below the recovery floor) and zero web."""
        store = _tmp_store()
        self.addCleanup(store.close)
        store.create_execution("exfb", QUESTION)   # FK: the bound execution

        class _Resp:
            def __str__(self_inner):
                return f"{NOT_FOUND_PHRASE} (best similarity score 0.30)"

        orig_engine = ba._get_query_engine
        orig_llm = ba.get_llm
        ba._get_query_engine = lambda: SimpleNamespace(
            query=lambda _q: SimpleNamespace(
                source_nodes=[SimpleNamespace(
                    score=0.30,
                    metadata={"file_name": "guide.pdf", "page_label": "1"},
                    text="irrelevant chunk")]))
        ba.get_llm = mock.Mock(
            side_effect=AssertionError("recovery must not run at score 0.30"))
        self.addCleanup(setattr, ba, "_get_query_engine", orig_engine)
        self.addCleanup(setattr, ba, "get_llm", orig_llm)

        with gm.approval_scope(store, "exfb"):
            result = gm.build_graph().invoke({"question": QUESTION})

        pending = store.list_approvals(APPROVAL_PENDING)
        self.assertEqual(len(pending), 1)
        row = pending[0]
        self.assertEqual(row["execution_id"], "exfb")
        self.assertEqual(row["payload"]["question"], QUESTION)
        self.assertEqual(row["payload"]["books_refusal"]["reason"],
                         "low_confidence")
        self.assertTrue(row["payload"]["local_answer"].startswith(
            NOT_FOUND_PHRASE))
        self.assertEqual(result["route"], "books")
        self.assertIn(row["approval_id"], result["answer"])
        self.assertIn("Approval request ID:", result["answer"])
        self.assertIn("NOT searched", result["answer"])
        self.assertEqual(result["sources"], ["books"])
        self.assertEqual(result.get("qa_flags") or [], [])
        self.assertEqual(self.recorder.calls, [],
                         "a pending request must never execute research")

    def test_repeated_gate_delivery_reuses_one_request(self):
        store = _tmp_store()
        self.addCleanup(store.close)
        store.create_execution("exdup", QUESTION)
        state = {"question": QUESTION, "route": "books",
                 "books_refusal": REFUSAL}
        with gm.approval_scope(store, "exdup"):
            first = gm.research_entry(state)
            second = gm.research_entry(state)
        self.assertEqual(len(store.list_approvals()), 1)
        first_id = first["answer"].split("Approval request ID: ")[1].split("\n")[0]
        second_id = second["answer"].split("Approval request ID: ")[1].split("\n")[0]
        self.assertEqual(first_id, second_id)
        self.assertEqual(self.recorder.calls, [])

    def test_scope_is_restored_and_isolated(self):
        store = _tmp_store()
        self.addCleanup(store.close)
        store.create_execution("exa", QUESTION)
        store.create_execution("exb", QUESTION)
        state = {"question": QUESTION, "route": "books",
                 "books_refusal": REFUSAL}
        with gm.approval_scope(store, "exa"):
            with gm.approval_scope(store, "exb"):
                gm.research_entry(state)          # bound to exb
            gm.research_entry(state)              # bound back to exa
        self.assertIsNone(gm.current_approval_scope())
        by_exec = {a["execution_id"]: a["approval_id"]
                   for a in store.list_approvals()}
        self.assertEqual(set(by_exec), {"exa", "exb"})
        self.assertNotEqual(by_exec["exa"], by_exec["exb"])

    def test_store_failure_denies_instead_of_executing(self):
        store = _tmp_store()
        self.addCleanup(store.close)
        store.create_execution("exfail", QUESTION)
        state = {"question": QUESTION, "route": "books",
                 "books_refusal": REFUSAL}
        broken = mock.Mock(side_effect=OSError("disk full"))
        with mock.patch.object(store, "create_or_reuse_approval", broken):
            with gm.approval_scope(store, "exfail"):
                update = gm.research_entry(state)
        broken.assert_called_once()
        self.assertEqual(self.recorder.calls, [],
                         "a persistence failure must never grant permission")
        self.assertIn("could not register the approval request",
                      update["answer"])
        self.assertIn("NOT searched", update["answer"])

    def test_without_a_scope_the_gate_denies_silently(self):
        state = {"question": QUESTION, "route": "books",
                 "books_refusal": REFUSAL}
        update = gm.research_entry(state)
        self.assertEqual(self.recorder.calls, [])
        self.assertIn("NOT searched", update["answer"])
        self.assertNotIn("Approval request ID:", update["answer"])


class TestGateResumeValidation(_GraphPatchCase):
    def _scope_state(self, **overrides):
        state = {"question": QUESTION, "route": "books",
                 "sources": ["books"],
                 "agent_answers": {"books": LOCAL_ANSWER},
                 "books_refusal": REFUSAL}
        state.update(overrides)
        return state

    def _approved(self, store, execution_id="exval"):
        store.create_execution(execution_id, QUESTION)
        with gm.approval_scope(store, execution_id):
            gm.research_entry(self._scope_state())
        row = [a for a in store.list_approvals()
               if a["execution_id"] == execution_id][0]
        store.decide_approval(row["approval_id"], "approved")
        return row["approval_id"]

    def test_valid_approval_runs_exactly_its_bound_operation(self):
        store = _tmp_store()
        self.addCleanup(store.close)
        aid = self._approved(store)
        state = self._scope_state(
            research_approval={"approval_id": aid})
        with gm.approval_scope(store, "exval"):
            update = gm.research_entry(state)
        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(update["answer"], "EXTERNAL_RESEARCH_ANSWER")
        self.assertIn("research", update["sources"])
        self.assertTrue(store.get_approval(aid)["executed_at"])
        # second processing: the claim refuses — nothing runs again
        with gm.approval_scope(store, "exval"):
            again = gm.research_entry(state)
        self.assertEqual(len(self.recorder.calls), 1,
                         "one approval may execute its operation only once")
        self.assertIn("already executed", again["answer"])
        self.assertNotIn(RESEARCH_URL, again["answer"])

    def test_missing_scope_unknown_and_foreign_approvals_are_denied(self):
        cases = [
            ("no scope", None, self._scope_state(
                research_approval={"approval_id": "x"})),
        ]
        # 1) resume attempted with no store in scope
        update = gm.research_entry(cases[0][2])
        self.assertEqual(self.recorder.calls, [])
        self.assertIn("no approval store", update["answer"])

        store = _tmp_store()
        self.addCleanup(store.close)
        aid = self._approved(store, "exone")

        # 2) unknown approval id
        state = self._scope_state(research_approval={"approval_id": "ffffffffffff"})
        with gm.approval_scope(store, "exone"):
            update = gm.research_entry(state)
        self.assertIn("unknown approval request", update["answer"])

        # 3) approval bound to ANOTHER execution
        store.create_execution("extwo", QUESTION)
        state = self._scope_state(research_approval={"approval_id": aid})
        with gm.approval_scope(store, "extwo"):
            update = gm.research_entry(state)
        self.assertIn("belongs to another execution", update["answer"])

        self.assertEqual(self.recorder.calls, [],
                         "no mismatch may execute research")

    def test_non_approved_states_are_denied(self):
        store = _tmp_store()
        self.addCleanup(store.close)
        store.create_execution("exs", QUESTION)

        with gm.approval_scope(store, "exs"):
            gm.research_entry(self._scope_state())
        aid = store.list_approvals()[0]["approval_id"]
        state = self._scope_state(research_approval={"approval_id": aid})

        # pending -> denied
        with gm.approval_scope(store, "exs"):
            update = gm.research_entry(state)
        self.assertIn("approval is pending", update["answer"])

        # rejected -> denied
        store.decide_approval(aid, "rejected")
        with gm.approval_scope(store, "exs"):
            update = gm.research_entry(state)
        self.assertIn("approval is rejected", update["answer"])
        self.assertEqual(self.recorder.calls, [])

    def test_mismatched_bound_operation_is_denied(self):
        store = _tmp_store()
        self.addCleanup(store.close)
        aid = self._approved(store, "exm")

        variants = [
            dict(question="A different question entirely"),
            dict(route="research"),
            dict(books_refusal={"reason": "grounding", "score": 0.9,
                                "nodes": 2, "recovery": "rejected"}),
        ]
        for overrides in variants:
            with self.subTest(overrides=sorted(overrides)):
                state = self._scope_state(
                    research_approval={"approval_id": aid})
                state.update(overrides)
                with gm.approval_scope(store, "exm"):
                    update = gm.research_entry(state)
                self.assertIn("mismatch", update["answer"])
        self.assertEqual(self.recorder.calls, [],
                         "an approval authorizes only its bound operation")

    def test_malformed_approval_reference_is_denied(self):
        store = _tmp_store()
        self.addCleanup(store.close)
        store.create_execution("exmal", QUESTION)
        for bad in ({"approval_id": ""}, {"approval_id": None}, "just-a-string",
                    {}):
            with self.subTest(ref=bad):
                state = self._scope_state(
                    research_approval=bad if isinstance(bad, dict)
                    else {"approval_id": bad})
                if not isinstance(bad, dict) or not bad.get("approval_id"):
                    # treated as NO reference -> pending path (still no run)
                    with gm.approval_scope(store, "exmal"):
                        update = gm.research_entry(state)
                self.assertEqual(self.recorder.calls, [])
                self.assertIn("NOT searched", update["answer"])


# ---------------------------------------------------------------------------
# 3. Runner resume
# ---------------------------------------------------------------------------


class TestRunnerResume(_GraphPatchCase):
    def setUp(self):
        super().setUp()
        self.store = _tmp_store()
        self.addCleanup(self.store.close)
        self.runner = GraphRunner(self.store)

    def _pending_approval(self, with_local=True):
        # runner-created, like production: the timeline starts at
        # execution.created and the gate appends to it.
        self.eid = self.runner.create(QUESTION)["id"]
        self.store.update_execution(
            self.eid, status="completed", route="books",
            answer="I have NOT searched the web for this yet.",
            sources=["books"])
        state = {"question": QUESTION, "route": "books",
                 "sources": ["books"],
                 "agent_answers": {"books": LOCAL_ANSWER},
                 "books_refusal": REFUSAL}
        if not with_local:
            state.pop("agent_answers")
        with gm.approval_scope(self.store, self.eid):
            gm.research_entry(state)
        return self.store.list_approvals()[0]["approval_id"]

    def test_pending_resume_is_refused_before_any_decision(self):
        aid = self._pending_approval()
        with self.assertRaises(ApprovalDecisionRequired):
            self.runner.resume_approval(aid)
        self.assertEqual(self.recorder.calls, [])

    def test_expired_resume_fails_closed(self):
        self.store.approval_ttl_seconds = -1
        aid = self._pending_approval()
        with self.assertRaises(ApprovalExpired):
            self.runner.resume_approval(aid)
        self.assertEqual(self.store.get_approval(aid)["status"], "expired")
        self.assertEqual(self.recorder.calls, [])

    def test_approved_resume_executes_and_persists_the_real_result(self):
        aid = self._pending_approval()
        self.store.decide_approval(aid, "approved")
        result = self.runner.resume_approval(aid)
        self.assertEqual(result["outcome"], "executed")
        self.assertEqual(len(self.recorder.calls), 1)
        execution = result["execution"]
        self.assertEqual(execution["answer"], "EXTERNAL_RESEARCH_ANSWER")
        self.assertEqual(execution["sources"],
                         ["books", "research", RESEARCH_URL])
        self.assertEqual(execution["status"], "completed")

        # duplicate processing: nothing runs, the real answer stays
        dup = self.runner.resume_approval(aid)
        self.assertEqual(dup["outcome"], "denied")
        self.assertIn("already executed", dup["reason"])
        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(self.store.get_execution(self.eid)["answer"],
                         "EXTERNAL_RESEARCH_ANSWER")

    def test_rejected_resume_closes_with_honest_local_only_answer(self):
        aid = self._pending_approval()
        self.store.decide_approval(aid, "rejected")
        result = self.runner.resume_approval(aid)
        self.assertEqual(result["outcome"], "rejected")
        self.assertTrue(result["finalized"])
        self.assertEqual(self.recorder.calls, [],
                         "a rejected request must never execute research")
        execution = result["execution"]
        self.assertEqual(execution["answer"], LOCAL_ANSWER)
        self.assertEqual(execution["sources"], ["books"])
        # This legacy-style test fixture starts from a completed snapshot;
        # production executions are now waiting_for_approval and are covered
        # by test_rejecting_waiting_execution_closes_it_as_failed_once.
        self.assertEqual(execution["status"], "completed")

        # idempotent: a second resolve changes nothing, appends no event
        events_before = len(self.store.get_events(self.eid))
        repeat = self.runner.resume_approval(aid)
        self.assertEqual(repeat["outcome"], "rejected")
        self.assertFalse(repeat["finalized"])
        self.assertEqual(len(self.store.get_events(self.eid)), events_before)
        self.assertEqual(self.store.get_execution(self.eid)["answer"],
                         LOCAL_ANSWER)

    def test_rejecting_waiting_execution_closes_it_as_failed_once(self):
        aid = self._pending_approval(with_local=False)
        self.store.update_execution(self.eid, status="waiting_for_approval")
        self.store.decide_approval(aid, "rejected")

        result = self.runner.resume_approval(aid)
        self.assertEqual(result["outcome"], "rejected")
        self.assertEqual(result["execution"]["status"], "failed")
        self.assertEqual(result["execution"]["error"]["category"],
                         "approval_rejected")
        self.assertEqual(self.recorder.calls, [])

        events = len(self.store.get_events(self.eid))
        repeated = self.runner.resume_approval(aid)
        self.assertFalse(repeated["finalized"])
        self.assertEqual(len(self.store.get_events(self.eid)), events)
        self.assertEqual(self.store.get_execution(self.eid)["status"], "failed")

    def test_plan_bound_approval_resume_keeps_partial_result_and_admits_incomplete(self):
        """Approved research on a PAUSED plan run executes the research, but
        the execution may only say `completed` when the whole plan finished.

        This plan stopped for a reason that is NOT the missing approval, so it
        is not resumed (see TestPlanContinuation for the resumed case): the run
        keeps the real research answer and reports the honest partial state.
        """
        aid = self._pending_approval()
        self.store.update_execution(self.eid, plan={
            "source": "default",
            "tasks": [
                {"task_id": "t1", "capability": "research",
                 "depends_on": [], "status": "failed",
                 "error": "RuntimeError: search backend unavailable"},
                {"task_id": "t2", "capability": "synthesis",
                 "depends_on": ["t1"], "status": "blocked",
                 "block_reason": "a dependency did not complete successfully"},
            ],
        })
        self.store.decide_approval(aid, "approved")

        result = self.runner.resume_approval(aid)

        self.assertEqual(result["outcome"], "executed")
        self.assertEqual(len(self.recorder.calls), 1)
        execution = result["execution"]
        self.assertEqual(execution["answer"], "EXTERNAL_RESEARCH_ANSWER")
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["error"]["category"], "incomplete_plan")
        self.assertEqual(
            execution["error"]["details"]["incomplete_tasks"][0]["task_id"],
            "t1")

        # without a plan (direct research route) the same resume completes
        store2 = _tmp_store()
        self.addCleanup(store2.close)
        runner2 = GraphRunner(store2)
        eid2 = runner2.create(QUESTION)["id"]
        store2.update_execution(
            eid2, status="waiting_for_approval", route="books",
            answer="I have NOT searched the web for this yet.", sources=["books"])
        state = {"question": QUESTION, "route": "books", "sources": ["books"],
                 "agent_answers": {"books": LOCAL_ANSWER},
                 "books_refusal": REFUSAL}
        with gm.approval_scope(store2, eid2):
            gm.research_entry(state)
        aid2 = store2.list_approvals()[0]["approval_id"]
        store2.decide_approval(aid2, "approved")
        direct = runner2.resume_approval(aid2)
        self.assertEqual(direct["execution"]["status"], "completed")
        self.assertEqual(len(self.recorder.calls), 2)

    def test_rejected_resume_without_local_answer_keeps_run1_answer(self):
        aid = self._pending_approval(with_local=False)
        self.store.decide_approval(aid, "rejected")
        result = self.runner.resume_approval(aid)
        self.assertEqual(result["outcome"], "rejected")
        self.assertFalse(result["finalized"])
        self.assertIn("NOT searched",
                      self.store.get_execution(self.eid)["answer"])
        self.assertEqual(self.store.get_execution(self.eid)["status"], "completed")
        self.assertEqual(self.recorder.calls, [])

    def test_unknown_approval_and_unbound_rows_are_refused(self):
        with self.assertRaises(LookupError):
            self.runner.resume_approval("ffffffffffff")
        # an approval created outside an execution cannot be resumed — the
        # pending check comes first (decide first), so decide it and then
        # the missing binding is what refuses:
        row = self.store.create_or_reuse_approval(
            operation="research", description="d",
            payload=_payload(), execution_id=None)
        with self.assertRaises(ApprovalDecisionRequired):
            self.runner.resume_approval(row["approval_id"])
        self.store.decide_approval(row["approval_id"], "approved")
        with self.assertRaises(ApprovalConflict):
            self.runner.resume_approval(row["approval_id"])
        self.assertEqual(self.recorder.calls, [])

    def test_timeline_records_the_whole_lifecycle(self):
        aid = self._pending_approval()
        self.store.decide_approval(aid, "approved")
        self.runner.resume_approval(aid)
        types = [e["type"] for e in self.store.get_events(self.eid)]
        for expected in ("execution.created", "approval.decided",
                         "approval.executed"):
            self.assertIn(expected, types)


# ---------------------------------------------------------------------------
# 3b. Plan continuation after approval (GUI-10 follow-up)
# ---------------------------------------------------------------------------


class TestPlanContinuation(_GraphPatchCase):
    """Approving the head research task of a paused plan really finishes it.

    The remaining tasks run through the EXISTING dispatcher: completed tasks
    are skipped, a dependent task consumes the approved research answer, a
    later task's failure is reported honestly, a second research task pauses
    for its OWN decision, and a retried resume adds no work.
    """

    def setUp(self):
        super().setUp()
        self.store = _tmp_store()
        self.addCleanup(self.store.close)
        self.runner = GraphRunner(self.store)

        # The planner's terminal `synthesis` task now dispatches to the
        # synthesis node (2026-10-28) — that is what a continuation runs.
        self.synthesis = _ArgumentRecorder("synthesis", "SYNTHESIS_ANSWER")
        self._orig_synthesis = gm.synthesis_node
        gm.synthesis_node = self.synthesis
        self.addCleanup(self._restore_synthesis)

        self.books = _ArgumentRecorder("books", LOCAL_ANSWER)
        self._orig_books = gm.books_agent
        gm.books_agent = self.books
        self.addCleanup(self._restore_books)

    def _restore_synthesis(self):
        gm.synthesis_node = self._orig_synthesis

    def _restore_books(self):
        gm.books_agent = self._orig_books

    def _paused_plan(self, tasks) -> str:
        """A real paused execution: stored plan + a PENDING bound request."""
        self.eid = self.runner.create(QUESTION)["id"]
        self.store.update_execution(
            self.eid, status="waiting_for_approval", route="books",
            answer="I have NOT searched the web for this yet.",
            sources=["books"], plan={"source": "default", "tasks": tasks})
        state = {"question": QUESTION, "route": "books", "sources": ["books"],
                 "agent_answers": {"books": LOCAL_ANSWER},
                 "books_refusal": REFUSAL}
        with gm.approval_scope(self.store, self.eid):
            gm.research_entry(state)
        return self.store.list_approvals()[0]["approval_id"]

    def _plan_tasks(self, execution) -> dict:
        return {t["task_id"]: t for t in (execution.get("plan") or {})["tasks"]}

    def test_plan_completes_after_approval(self):
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        result = self.runner.resume_approval(aid)

        self.assertEqual(result["outcome"], "executed")
        self.assertEqual(len(self.recorder.calls), 1,
                         "the approved research must run exactly once")
        self.assertEqual(len(self.synthesis.calls), 1,
                         "the blocked synthesis task must now run")
        self.assertEqual(
            self.synthesis.calls[0]["agent_answers"]["research"],
            "EXTERNAL_RESEARCH_ANSWER",
            "the dependent task must see the approved research result")

        execution = result["execution"]
        self.assertEqual(execution["status"], "completed")
        self.assertIsNone(execution["error"])
        self.assertEqual(execution["answer"], "SYNTHESIS_ANSWER")
        self.assertIn("synthesis", execution["sources"])
        self.assertEqual(execution["qa_flags"], [],
                         "a finished plan must not keep the stale run-1 flags")
        tasks = self._plan_tasks(execution)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t1"]["result"], "EXTERNAL_RESEARCH_ANSWER")
        self.assertNotIn("error", tasks["t1"])
        self.assertEqual(tasks["t2"]["status"], "completed")

        # exactly ONE terminal status event for the continued run
        types = [e["type"] for e in self.store.get_events(self.eid)]
        self.assertEqual(types.count("execution.approval_resolved"), 1)
        self.assertEqual(types.count("execution.completed"), 1)
        self.assertNotIn("execution.failed", types)

    def test_resume_keeps_the_task_objective_and_searches_the_request(self):
        """An objective may narrow the work, never the approved external call."""
        objective = "Search the EU AI Act enforcement deadlines"
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", objective=objective,
                  error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        result = self.runner.resume_approval(aid)

        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(self.recorder.calls[0]["question"], QUESTION,
                         "the approved search is the request, never the "
                         "plan's narrower objective")
        tasks = self._plan_tasks(result["execution"])
        self.assertEqual(tasks["t1"]["objective"], objective,
                         "the plan keeps the objective as data")
        self.assertEqual(tasks["t1"]["result"], "EXTERNAL_RESEARCH_ANSWER")
        self.assertEqual(result["execution"]["status"], "completed")

    def test_completed_tasks_are_not_executed_again(self):
        aid = self._paused_plan([
            _task("t0", "books", [], "completed", result=LOCAL_ANSWER),
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t0", "t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        result = self.runner.resume_approval(aid)

        self.assertEqual(result["execution"]["status"], "completed")
        self.assertEqual(self.books.calls, [],
                         "a task completed in run 1 must never run again")
        self.assertEqual(len(self.synthesis.calls), 1)
        self.assertEqual(
            self.synthesis.calls[0]["agent_answers"]["books"], LOCAL_ANSWER,
            "earlier task output must be replayed to the dependent task")

    def test_later_task_failure_is_reported_honestly(self):
        self.synthesis.error = RuntimeError("synthesis exploded")
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        result = self.runner.resume_approval(aid)

        execution = result["execution"]
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["error"]["category"], "incomplete_plan")
        tasks = self._plan_tasks(execution)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t2"]["status"], "failed")
        self.assertIn("RuntimeError", tasks["t2"]["error"])
        self.assertEqual(len(self.recorder.calls), 1)

    def test_second_research_task_pauses_for_its_own_approval(self):
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            # run 1 recorded t2 as BLOCKED (its dependency t1 did not
            # complete) — that is what the dispatcher actually writes.
            _task("t2", "research", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
            _task("t3", "synthesis", ["t2"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        result = self.runner.resume_approval(aid)

        self.assertEqual(result["outcome"], "executed")
        self.assertEqual(len(self.recorder.calls), 1,
                         "a second research task must NOT search")
        self.assertEqual(self.synthesis.calls, [])

        execution = result["execution"]
        self.assertEqual(execution["status"], "waiting_for_approval")
        self.assertIsNone(execution["error"])
        self.assertIn("Approval request ID:", execution["answer"])
        tasks = self._plan_tasks(execution)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t2"]["status"], "failed")
        self.assertEqual(tasks["t3"]["status"], "blocked")

        pending = self.store.list_approvals("pending")
        self.assertEqual(len(pending), 1)
        self.assertNotEqual(pending[0]["approval_id"], aid,
                            "the second operation needs its OWN decision")

    def test_interrupted_continuation_is_recovered(self):
        """A crash AFTER the approved research and BEFORE the continuation
        must be recoverable without repeating the external operation."""
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        # Simulate the process dying inside the continuation: the approved
        # research HAS run (and its result is already persisted), the
        # remaining tasks have not.
        def _crash(*_args, **_kwargs):
            raise RuntimeError("process died mid-continuation")

        orig = gm.continue_plan
        gm.continue_plan = _crash
        try:
            with self.assertRaises(RuntimeError):
                self.runner.resume_approval(aid)
        finally:
            gm.continue_plan = orig

        self.assertEqual(len(self.recorder.calls), 1,
                         "the external operation ran exactly once")
        self.assertEqual(self.synthesis.calls, [])
        crashed = self.store.get_execution(self.eid)
        self.assertEqual(crashed["status"], "waiting_for_approval")
        self.assertEqual(crashed["answer"], "EXTERNAL_RESEARCH_ANSWER",
                         "the research result must be durably persisted")
        tasks = self._plan_tasks(crashed)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t1"]["result"], "EXTERNAL_RESEARCH_ANSWER")

        # The retry RECOVERS the continuation: no second search.
        recovered = self.runner.resume_approval(aid)

        self.assertEqual(recovered["outcome"], "executed")
        self.assertTrue(recovered.get("recovered"))
        self.assertEqual(len(self.recorder.calls), 1,
                         "recovery must never repeat the approved research")
        self.assertEqual(len(self.synthesis.calls), 1)
        execution = recovered["execution"]
        self.assertEqual(execution["status"], "completed")
        self.assertIsNone(execution["error"])
        self.assertEqual(execution["answer"], "SYNTHESIS_ANSWER")
        tasks = self._plan_tasks(execution)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t2"]["status"], "completed")

        # A further retry is still a safe no-op.
        repeat = self.runner.resume_approval(aid)
        self.assertEqual(repeat["outcome"], "denied")
        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(len(self.synthesis.calls), 1)
        self.assertEqual(self.store.get_execution(self.eid)["status"],
                         "completed")

    def test_interruption_mid_continuation_resumes_without_repeating_tasks(self):
        """A crash AFTER one continuation task finished must not replay it.

        Progress is persisted as each task completes, so the retry continues
        from the recorded statuses: the finished task is skipped (its result is
        replayed to its dependent), the pending one runs, and the external
        research is never repeated.
        """
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "books", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
            _task("t3", "synthesis", ["t2"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        real_dispatch = gm.dispatcher_node
        visits: list = []

        def _dying_dispatch(state):
            visits.append(len(visits))
            if len(visits) == 2:
                raise RuntimeError("process died during the continuation")
            return real_dispatch(state)

        gm.dispatcher_node = _dying_dispatch
        try:
            with self.assertRaises(RuntimeError):
                self.runner.resume_approval(aid)
        finally:
            gm.dispatcher_node = real_dispatch

        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(len(self.books.calls), 1)
        self.assertEqual(self.synthesis.calls, [])
        crashed = self.store.get_execution(self.eid)
        self.assertEqual(crashed["status"], "waiting_for_approval")
        tasks = self._plan_tasks(crashed)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t1"]["result"], "EXTERNAL_RESEARCH_ANSWER")
        self.assertEqual(tasks["t2"]["status"], "completed",
                         "a finished continuation task must be persisted "
                         "immediately")
        self.assertEqual(tasks["t2"]["result"], LOCAL_ANSWER)
        self.assertEqual(tasks["t3"]["status"], "pending")

        recovered = self.runner.resume_approval(aid)

        self.assertEqual(recovered["outcome"], "executed")
        self.assertTrue(recovered.get("recovered"))
        self.assertEqual(len(self.recorder.calls), 1,
                         "recovery never searches again")
        self.assertEqual(len(self.books.calls), 1,
                         "a task completed before the crash never runs again")
        self.assertEqual(len(self.synthesis.calls), 1)
        self.assertEqual(
            self.synthesis.calls[0]["agent_answers"]["books"], LOCAL_ANSWER,
            "the recovered task must still see the earlier result")
        execution = recovered["execution"]
        self.assertEqual(execution["status"], "completed")
        self.assertIsNone(execution["error"])
        self.assertEqual(self._plan_tasks(execution)["t3"]["status"],
                         "completed")

    def test_second_gate_interruption_is_recovered_without_repeating_research(self):
        """Crash AFTER the second approved research ran, BEFORE its tail.

        Recovery is keyed on the persisted released plan of THAT decision, so
        neither search is repeated, t1 (done before the first pause) is never
        re-run, and the remaining synthesis executes from persisted state.
        """
        first_aid = self._paused_plan([
            _task("t1", "research", [], "failed",
                  error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "research", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
            _task("t3", "synthesis", ["t2"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(first_aid, "approved")
        self.runner.resume_approval(first_aid)

        self.assertEqual(len(self.recorder.calls), 1)
        paused = self.store.get_execution(self.eid)
        self.assertEqual(paused["status"], "waiting_for_approval")
        pending = [a for a in self.store.list_approvals("pending")
                   if a["approval_id"] != first_aid]
        self.assertEqual(len(pending), 1,
                         "the second research task needs its own request")
        second_aid = pending[0]["approval_id"]
        self.store.decide_approval(second_aid, "approved")

        def _crash(*_args, **_kwargs):
            raise RuntimeError("process died mid-continuation")

        orig = gm.continue_plan
        gm.continue_plan = _crash
        try:
            with self.assertRaises(RuntimeError):
                self.runner.resume_approval(second_aid)
        finally:
            gm.continue_plan = orig

        self.assertEqual(len(self.recorder.calls), 2,
                         "the second operation ran exactly once")
        self.assertEqual(self.synthesis.calls, [])
        crashed = self.store.get_execution(self.eid)
        self.assertEqual(crashed["status"], "waiting_for_approval")
        tasks = self._plan_tasks(crashed)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t1"]["result"], "EXTERNAL_RESEARCH_ANSWER")
        self.assertEqual(tasks["t2"]["status"], "completed",
                         "the second approved result must be durably "
                         "persisted before the continuation")
        self.assertEqual(tasks["t3"]["status"], "pending")

        recovered = self.runner.resume_approval(second_aid)

        self.assertEqual(recovered["outcome"], "executed")
        self.assertTrue(recovered.get("recovered"))
        self.assertEqual(len(self.recorder.calls), 2,
                         "recovery never repeats either approved search")
        self.assertEqual(len(self.synthesis.calls), 1)
        execution = recovered["execution"]
        self.assertEqual(execution["status"], "completed")
        self.assertIsNone(execution["error"])
        self.assertEqual(execution["answer"], "SYNTHESIS_ANSWER")
        self.assertEqual(self._plan_tasks(execution)["t3"]["status"],
                         "completed")

    def test_unreleased_second_approval_is_not_recovered(self):
        """A spent approval whose release was never persisted is not provable.

        The plan still records the second research task as approval-blocked, so
        the earlier completed research result must NOT be taken as proof of
        this decision: the run closes honestly and nothing is searched again.
        """
        first_aid = self._paused_plan([
            _task("t1", "research", [], "failed",
                  error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "research", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
            _task("t3", "synthesis", ["t2"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(first_aid, "approved")
        self.runner.resume_approval(first_aid)
        second_aid = [a for a in self.store.list_approvals("pending")
                      if a["approval_id"] != first_aid][0]["approval_id"]
        self.store.decide_approval(second_aid, "approved")
        # The operation was claimed (it ran) but nothing recorded its result.
        self.assertTrue(self.store.claim_approval_execution(second_aid))

        result = self.runner.resume_approval(second_aid)

        self.assertEqual(result["outcome"], "denied")
        self.assertFalse(result.get("recovered"))
        self.assertEqual(len(self.recorder.calls), 1,
                         "a claimed approval never searches a second time")
        self.assertEqual(self.synthesis.calls, [])
        execution = result["execution"]
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["error"]["category"], "resume_interrupted")
        tasks = self._plan_tasks(execution)
        self.assertEqual(tasks["t1"]["result"], "EXTERNAL_RESEARCH_ANSWER",
                         "the first task's real result is preserved")
        self.assertNotEqual(tasks["t2"]["status"], "completed")

    def test_crash_after_continuation_finished_finalizes_on_retry(self):
        """The tail completed but the terminal write was lost: finalize it.

        Every task was persisted as completed by the progress callback, so the
        retry must record the truthful outcome from the persisted plan without
        dispatching any task again.
        """
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        with mock.patch.object(self.runner, "_persist_continued",
                               side_effect=RuntimeError("process died")):
            with self.assertRaises(RuntimeError):
                self.runner.resume_approval(aid)

        crashed = self.store.get_execution(self.eid)
        self.assertEqual(crashed["status"], "waiting_for_approval")
        tasks = self._plan_tasks(crashed)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t2"]["status"], "completed",
                         "the finished tail was persisted task by task")
        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(len(self.synthesis.calls), 1)

        recovered = self.runner.resume_approval(aid)

        self.assertEqual(recovered["outcome"], "executed")
        self.assertTrue(recovered.get("recovered"))
        self.assertEqual(len(self.recorder.calls), 1,
                         "finalizing never repeats the research")
        self.assertEqual(len(self.synthesis.calls), 1,
                         "finalizing never re-dispatches a task")
        execution = recovered["execution"]
        self.assertEqual(execution["status"], "completed")
        self.assertIsNone(execution["error"])
        self.assertEqual(execution["answer"], "SYNTHESIS_ANSWER")
        self.assertEqual(execution["qa_flags"], [],
                         "stale paused-run flags must not survive")
        types = [e["type"] for e in self.store.get_events(self.eid)]
        self.assertEqual(types.count("execution.completed"), 1)
        self.assertNotIn("execution.failed", types)

    def test_running_task_in_persisted_plan_is_not_replayed(self):
        """A task persisted mid-flight proves nothing about its side effects."""
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")
        self.assertTrue(self.store.claim_approval_execution(aid))
        self.store.update_execution(
            self.eid, plan={"source": "default", "tasks": [
                _task("t1", "research", [], "completed",
                      result="EXTERNAL_RESEARCH_ANSWER"),
                _task("t2", "synthesis", ["t1"], "running"),
            ]})

        result = self.runner.resume_approval(aid)

        self.assertEqual(result["outcome"], "denied")
        self.assertEqual(self.synthesis.calls, [])
        self.assertEqual(len(self.recorder.calls), 0)
        execution = result["execution"]
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["error"]["category"], "resume_interrupted")
        self.assertIn("still running", execution["error"]["why"])

    def test_unrecoverable_resume_is_closed_honestly(self):
        """Claimed approval + no persisted result: never search again, and
        never pretend the run can continue."""
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")
        # The operation was claimed (it ran) but its result was never
        # persisted: the plan still shows the research task as failed.
        self.assertTrue(self.store.claim_approval_execution(aid))

        result = self.runner.resume_approval(aid)

        self.assertEqual(result["outcome"], "denied")
        self.assertFalse(result.get("recovered"))
        self.assertEqual(self.recorder.calls, [],
                         "an already-executed approval never searches again")
        self.assertEqual(self.synthesis.calls, [])
        execution = result["execution"]
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["error"]["category"], "resume_interrupted")
        self.assertEqual(execution["error"]["recoverable"], True)

        # Same honesty for a paused run WITHOUT a plan: its answer was the
        # research result the interruption lost.
        store2 = _tmp_store()
        self.addCleanup(store2.close)
        runner2 = GraphRunner(store2)
        eid2 = runner2.create(QUESTION)["id"]
        store2.update_execution(
            eid2, status="waiting_for_approval", route="books",
            answer="I have NOT searched the web for this yet.", sources=["books"])
        with gm.approval_scope(store2, eid2):
            gm.research_entry({"question": QUESTION, "route": "books",
                               "sources": ["books"],
                               "books_refusal": REFUSAL})
        aid2 = store2.list_approvals()[0]["approval_id"]
        store2.decide_approval(aid2, "approved")
        store2.claim_approval_execution(aid2)
        direct = runner2.resume_approval(aid2)
        self.assertEqual(direct["outcome"], "denied")
        self.assertEqual(direct["execution"]["status"], "failed")
        self.assertEqual(direct["execution"]["error"]["category"],
                         "resume_interrupted")
        self.assertEqual(len(self.recorder.calls), 0)

    def test_concurrent_resume_does_not_continue_the_same_execution(self):
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")

        with self.store.resume_claim(self.eid) as claimed:
            self.assertTrue(claimed)
            second = self.runner.resume_approval(aid)

        self.assertEqual(second["outcome"], "denied")
        self.assertIn("another resume", second["reason"])
        self.assertEqual(self.recorder.calls, [])
        self.assertEqual(self.synthesis.calls, [])
        # The claim is released, so the same request can still succeed.
        ok = self.runner.resume_approval(aid)
        self.assertEqual(ok["outcome"], "executed")
        self.assertEqual(ok["execution"]["status"], "completed")
        self.assertEqual(len(self.recorder.calls), 1)

    def test_expired_lease_takeover_never_repeats_an_uncertain_operation(self):
        """A stale lease lets the retry OWN the execution — not re-search.

        The approval was already claimed but its result was never persisted,
        so taking over the expired lease must close the run honestly instead
        of treating the claim as permission to run the search again.
        """
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")
        self.assertTrue(self.store.claim_approval_execution(aid))
        # Another resolver's lease was left behind by a crash.
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE executions SET resume_lease_owner = 'dead', "
                "resume_lease_expires_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00+00:00", self.eid))
            self.store._conn.commit()

        result = self.runner.resume_approval(aid)

        self.assertEqual(result["outcome"], "denied")
        self.assertFalse(result.get("recovered"))
        self.assertEqual(self.recorder.calls, [],
                         "an expired lease is not permission to search again")
        self.assertEqual(self.synthesis.calls, [])
        execution = result["execution"]
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["error"]["category"], "resume_interrupted")

    def test_retried_resume_adds_no_duplicate_work(self):
        aid = self._paused_plan([
            _task("t1", "research", [], "failed", error=APPROVAL_BLOCKED_ERROR),
            _task("t2", "synthesis", ["t1"], "blocked",
                  block_reason=gm.DEPENDENCY_BLOCK_REASON),
        ])
        self.store.decide_approval(aid, "approved")
        first = self.runner.resume_approval(aid)
        plan_after_first = first["execution"]["plan"]

        repeat = self.runner.resume_approval(aid)

        self.assertEqual(repeat["outcome"], "denied")
        self.assertIn("already executed", repeat["reason"])
        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(len(self.synthesis.calls), 1)
        execution = self.store.get_execution(self.eid)
        self.assertEqual(execution["status"], "completed")
        self.assertEqual(execution["plan"], plan_after_first)


# ---------------------------------------------------------------------------
# 4. API endpoints (GUI-10 surface)
# ---------------------------------------------------------------------------


class TestApprovalApi(_GraphPatchCase):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        import app.api.server as server

        self.server = server
        self.store = _tmp_store()
        self.server._store = self.store
        self.client = TestClient(server.build_server())
        self.addCleanup(self.store.close)

    def _pending(self, execution_id="exapi"):
        self.store.create_execution(execution_id, QUESTION)
        self.store.update_execution(
            execution_id, status="completed", route="books",
            answer="I have NOT searched the web for this yet.",
            sources=["books"])
        state = {"question": QUESTION, "route": "books",
                 "sources": ["books"],
                 "agent_answers": {"books": LOCAL_ANSWER},
                 "books_refusal": REFUSAL}
        with gm.approval_scope(self.store, execution_id):
            gm.research_entry(state)
        return self.store.list_approvals()[0]["approval_id"]

    def test_list_and_get_pending_requests(self):
        aid = self._pending()
        r = self.client.get("/api/v1/approvals")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["approvals"][0]["approval_id"], aid)
        self.assertEqual(body["approvals"][0]["status"], "pending")

        r = self.client.get(f"/api/v1/approvals/{aid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["approval"]["approval_id"], aid)
        self.assertNotIn("fingerprint", r.json()["approval"])

        self.assertEqual(
            self.client.get("/api/v1/approvals/ffffffffffff").status_code, 404)
        self.assertEqual(
            self.client.get("/api/v1/approvals?status=bogus").status_code,
            422)

    def test_approval_mutations_reject_cross_site_origins_before_state_change(self):
        aid = self._pending()
        for action in ("approve", "reject", "resume"):
            with self.subTest(action=action):
                response = self.client.post(
                    f"/api/v1/approvals/{aid}/{action}",
                    headers={"Origin": "https://attacker.example"},
                )
                self.assertEqual(response.status_code, 403)
                self.assertEqual(self.store.get_approval(aid)["status"], "pending")
        self.assertEqual(self.recorder.calls, [])

    def test_local_control_center_origin_can_decide_approval(self):
        aid = self._pending()
        response = self.client.post(
            f"/api/v1/approvals/{aid}/approve",
            headers={"Origin": "http://localhost:3333"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.get_approval(aid)["status"], "approved")

    def test_approve_contract_idempotency_and_conflicts(self):
        aid = self._pending()
        r = self.client.post(f"/api/v1/approvals/{aid}/approve")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["approval"]["status"], "approved")
        self.assertEqual(body["decision"]["outcome"], "approved")
        self.assertFalse(body["decision"]["already_decided"])
        self.assertEqual(body["resume"], f"/api/v1/approvals/{aid}/resume")

        repeat = self.client.post(f"/api/v1/approvals/{aid}/approve")
        self.assertEqual(repeat.status_code, 200)
        self.assertTrue(repeat.json()["decision"]["already_decided"])
        self.assertEqual(repeat.json()["approval"]["decided_at"],
                         body["approval"]["decided_at"])

        conflict = self.client.post(f"/api/v1/approvals/{aid}/reject")
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(self.store.get_approval(aid)["status"], "approved")

        self.assertEqual(
            self.client.post("/api/v1/approvals/ffffffffffff/approve"
                             ).status_code, 404)

    def test_reject_and_expired_semantics(self):
        aid = self._pending()
        r = self.client.post(f"/api/v1/approvals/{aid}/reject")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["approval"]["status"], "rejected")

        self.store.approval_ttl_seconds = -1
        aid2 = self._pending(execution_id="exapi2")
        self.assertEqual(
            self.client.post(f"/api/v1/approvals/{aid2}/approve").status_code,
            410)

    def test_resume_pending_is_conflict_not_execution(self):
        aid = self._pending()
        r = self.client.post(f"/api/v1/approvals/{aid}/resume")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self.recorder.calls, [])
        self.assertEqual(
            self.client.post(
                "/api/v1/approvals/ffffffffffff/resume").status_code, 404)

    def test_resume_approved_runs_once_and_updates_the_execution(self):
        aid = self._pending()
        self.client.post(f"/api/v1/approvals/{aid}/approve")
        r = self.client.post(f"/api/v1/approvals/{aid}/resume")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["outcome"], "executed")
        self.assertEqual(len(self.recorder.calls), 1)

        execution = self.client.get("/api/v1/executions/exapi").json()
        self.assertEqual(execution["answer"], "EXTERNAL_RESEARCH_ANSWER")
        self.assertEqual(execution["sources"],
                         ["books", "research", RESEARCH_URL])

        dup = self.client.post(f"/api/v1/approvals/{aid}/resume")
        self.assertEqual(dup.status_code, 200)
        self.assertEqual(dup.json()["outcome"], "denied")
        self.assertEqual(len(self.recorder.calls), 1)

    def test_resume_rejected_finalizes_local_only(self):
        aid = self._pending()
        self.client.post(f"/api/v1/approvals/{aid}/reject")
        r = self.client.post(f"/api/v1/approvals/{aid}/resume")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["outcome"], "rejected")
        self.assertTrue(body["finalized"])
        self.assertEqual(self.recorder.calls, [])
        execution = self.client.get("/api/v1/executions/exapi").json()
        self.assertEqual(execution["answer"], LOCAL_ANSWER)
        self.assertEqual(execution["sources"], ["books"])


# ---------------------------------------------------------------------------
# 5. Unchanged behavior: local RAG, conversation, routing pins, lifecycle
# ---------------------------------------------------------------------------


class TestUnchangedBehavior(_GraphPatchCase):
    def test_local_rag_answers_with_no_approval_at_all(self):
        """A grounded local answer never reaches the gate: zero approval
        rows, zero research, real provenance intact."""
        store = _tmp_store()
        self.addCleanup(store.close)
        claim = ("Propolis is a resinous substance bees collect from plant "
                 "buds and use to seal small gaps and sterilize the interior "
                 "of the hive.")
        passage = ("Chapter 3: Hive Products. " + claim)
        node = SimpleNamespace(
            score=0.74,
            metadata={"file_name": "urban_beekeeping_field_guide.pdf",
                      "page_label": "3"},
            text=passage)

        class _Resp:
            def __str__(self_inner):
                return claim

        orig_engine = ba._get_query_engine
        ba._get_query_engine = lambda: SimpleNamespace(
            query=lambda _q: SimpleNamespace(source_nodes=[node]))
        self.addCleanup(setattr, ba, "_get_query_engine", orig_engine)

        runner = GraphRunner(store)
        created = runner.create("What does chapter 3 say about propolis?")
        finished = runner.run(created["id"])

        self.assertEqual(finished["route"], "books")
        self.assertIn(claim, finished["answer"])
        self.assertIn("References: urban_beekeeping_field_guide.pdf, page 3",
                      finished["answer"])
        self.assertEqual(self.recorder.calls, [])
        self.assertEqual(store.list_approvals(), [],
                         "local RAG must never create an approval request")
        self.assertEqual(finished["qa_flags"] or [], [])

    def test_conversation_and_routing_pins_still_hold(self):
        store = _tmp_store()
        self.addCleanup(store.close)
        self.assertEqual(sup.supervisor({"question": "hello"})["route"],
                         "conversational")
        self.assertEqual(sup.supervisor(
            {"question": "What is a vector database?"})["route"], "general")
        self.assertEqual(sup.supervisor(
            {"question": "Search the latest AI news"})["route"], "research")
        self.assertEqual(sup.supervisor(
            {"question": QUESTION})["route"], "books")
        # a message that forbids web access can never authorize it
        self.assertEqual(
            sup.research_authorization(
                "Do not search the web; use only this document", "research"),
            (False, "local_only_request"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
