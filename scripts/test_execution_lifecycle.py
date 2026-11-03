"""Lifecycle invariant: no execution stays non-terminal (2026-10-28).

Reference behavior adapted from the Mohamad AI OS archive
(`unified_agent_runtime.execute` always persists a terminal status; its
mission control only accepts COMPLETED/ESCALATED/BLOCKED as end states).
Agentica's runner can only write `failed` for exceptions it observes, so a
process killed mid-run previously left `queued`/`running` rows that the UI
showed as active forever (observed: execution a7bd96990f0b, `running` since
2026-10-25).

Covers:
  * ExecutionStore.reconcile_interrupted() drives queued/running rows to a
    terminal `failed` status with a structured, honest error;
  * append-only: answers/routes/plans and existing events are never
    rewritten; exactly one closing event is appended per fixed execution;
  * terminal rows are untouched;
  * idempotence (second call fixes nothing, adds no events);
  * event ids can never collide with however the runner numbered the
    timeline (tail-of-timeline allocation);
  * the API-startup hook in app.api.server.get_store() actually runs it,
    fail-soft, against its configured database — with DEFAULT_DB_PATH
    patched so the real storage/ database is never touched.

Fully offline: temp SQLite files only, no graph build, no network, no LLM.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.execution.store import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    ExecutionStore,
)


def make_store() -> ExecutionStore:
    return ExecutionStore(Path(tempfile.mkdtemp()) / "lifecycle.sqlite3")


def seed(store: ExecutionStore, eid: str, status: str, *, answer=None,
         events=(1, 2)) -> None:
    store.create_execution(eid, f"question for {eid}", status)
    if answer is not None:
        store.update_execution(eid, answer=answer, completed_at=True)
    for event_id in events:
        store.append_event(
            eid, event_id, "execution.created", "Execution created",
        )


class TestReconcileInterrupted(unittest.TestCase):
    def setUp(self):
        self.store = make_store()

    def tearDown(self):
        self.store.close()

    def test_running_row_reaches_terminal_failed_with_structured_error(self):
        seed(self.store, "stuckrun000001", STATUS_RUNNING)
        self.assertEqual(self.store.reconcile_interrupted(), 1)

        row = self.store.get_execution("stuckrun000001")
        self.assertEqual(row["status"], STATUS_FAILED)
        self.assertIsNotNone(row["completed_at"])
        err = row["error"]
        self.assertIsInstance(err, dict)
        self.assertEqual(err["category"], "interrupted")
        self.assertTrue(err["human_message"].strip())
        self.assertTrue(err["why"].strip())
        self.assertTrue(err["recovery"].strip())
        self.assertTrue(err["recoverable"])

    def test_queued_row_is_reconciled_too(self):
        seed(self.store, "neverstart001", STATUS_QUEUED)
        self.assertEqual(self.store.reconcile_interrupted(), 1)
        row = self.store.get_execution("neverstart001")
        self.assertEqual(row["status"], STATUS_FAILED)
        self.assertEqual(row["error"]["category"], "interrupted")

    def test_one_closing_event_appended_and_history_preserved(self):
        seed(self.store, "stuckrun000002", STATUS_RUNNING, events=(1, 2, 3))
        before = self.store.get_events("stuckrun000002")
        self.store.reconcile_interrupted()
        after = self.store.get_events("stuckrun000002")

        self.assertEqual(len(after), len(before) + 1)
        # The original timeline survives byte-for-byte (append-only).
        for old, new in zip(before, after):
            self.assertEqual(old, new)
        closing = after[-1]
        self.assertEqual(closing["type"], "execution.failed")
        self.assertEqual(closing["severity"], "error")
        self.assertEqual(closing["payload"]["previous_status"],
                         STATUS_RUNNING)
        self.assertEqual(closing["payload"]["category"], "interrupted")

    def test_terminal_rows_are_untouched(self):
        seed(self.store, "done0000000001", STATUS_COMPLETED, answer="ANSWER")
        seed(self.store, "failed00000001", STATUS_FAILED, answer=None,
             events=(1,))
        seed(self.store, "cancelled00001", STATUS_CANCELLED, events=(1,))
        counts = {
            eid: len(self.store.get_events(eid))
            for eid in ("done0000000001", "failed00000001", "cancelled00001")
        }

        self.assertEqual(self.store.reconcile_interrupted(), 0)

        done = self.store.get_execution("done0000000001")
        self.assertEqual(done["status"], STATUS_COMPLETED)
        self.assertEqual(done["answer"], "ANSWER")
        self.assertIsNone(done["error"])
        self.assertEqual(
            self.store.get_execution("failed00000001")["status"],
            STATUS_FAILED,
        )
        self.assertEqual(
            self.store.get_execution("cancelled00001")["status"],
            STATUS_CANCELLED,
        )
        for eid, n in counts.items():
            self.assertEqual(len(self.store.get_events(eid)), n)

    def test_reconcile_is_idempotent(self):
        seed(self.store, "stuckrun000003", STATUS_RUNNING)
        self.assertEqual(self.store.reconcile_interrupted(), 1)
        events_after_first = len(self.store.get_events("stuckrun000003"))
        self.assertEqual(self.store.reconcile_interrupted(), 0)
        self.assertEqual(
            len(self.store.get_events("stuckrun000003")), events_after_first,
        )

    def test_event_id_never_collides_with_existing_timeline(self):
        # Runner numbering is process-global, so a timeline can even be
        # non-contiguous; the closing id must come from THIS execution's
        # tail, not from a global counter.
        seed(self.store, "gap00000000001", STATUS_RUNNING, events=(1, 2, 7))
        self.store.reconcile_interrupted()
        ids = [e["event_id"] for e in self.store.get_events("gap00000000001")]
        self.assertEqual(ids, sorted(set(ids)))       # unique + ordered
        self.assertEqual(ids[-1], 8)                  # MAX(id) + 1

    def test_answers_and_routes_are_never_rewritten(self):
        seed(self.store, "halfdone00001", STATUS_RUNNING, answer="PARTIAL")
        self.store.update_execution("halfdone00001", route="books")
        self.store.reconcile_interrupted()
        row = self.store.get_execution("halfdone00001")
        self.assertEqual(row["answer"], "PARTIAL")
        self.assertEqual(row["route"], "books")
        self.assertEqual(row["status"], STATUS_FAILED)


class TestStartupHook(unittest.TestCase):
    """app.api.server.get_store() runs the reconciliation on first open."""

    def setUp(self):
        import app.api.server as server_mod
        self.server = server_mod
        self.store = make_store()
        seed(self.store, "stuckrun000004", STATUS_RUNNING)
        self._patcher_db = mock.patch.object(
            server_mod, "DEFAULT_DB_PATH", Path(self.store.db_path),
        )
        self._patcher_store = mock.patch.object(server_mod, "_store", None)
        self._patcher_db.start()
        self._patcher_store.start()

    def tearDown(self):
        self._patcher_store.stop()   # restores the original _store value
        self._patcher_db.stop()
        self.store.close()

    def test_get_store_reconciles_stuck_rows_on_first_open(self):
        opened = self.server.get_store()
        try:
            row = opened.get_execution("stuckrun000004")
            self.assertEqual(row["status"], STATUS_FAILED)
            self.assertEqual(row["error"]["category"], "interrupted")
        finally:
            opened.close()

    def test_reconcile_failure_never_blocks_store_opening(self):
        # Bookkeeping must be fail-soft: a broken reconciliation cannot
        # stop the API from obtaining its store.
        with mock.patch.object(
            ExecutionStore, "reconcile_interrupted",
            side_effect=RuntimeError("boom"),
        ):
            opened = self.server.get_store()
        try:
            self.assertIsInstance(opened, ExecutionStore)
            # The row is left as-is for the next startup, not corrupted.
            self.assertEqual(
                self.store.get_execution("stuckrun000004")["status"],
                STATUS_RUNNING,
            )
        finally:
            opened.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
