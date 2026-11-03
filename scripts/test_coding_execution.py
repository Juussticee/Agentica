"""Workspace patch through the REAL runner lifecycle (2026-10-30) — offline.

Proves the full arc with real routing, real graph nodes, the REAL approval
lifecycle and the real single-execution claim:

  create + run     -> routes to `coding`, prepares a patch, pauses
                      (waiting_for_approval) without writing any file
  resume too early -> refused; nothing runs, the file is untouched
  approve + resume -> the reviewed patch is hash-rechecked and applied
  resume again     -> denied (already executed); the file is not rewritten
  stale file       -> nothing is applied, the refusal is structured
  absent workspace -> the refusal is structured (workspace_unavailable)
  rejected         -> the execution closes failed, nothing was ever written
  second patch     -> a later execution prepares its OWN new request

Two seams only: `app.llm_client.get_llm` (fixed patch reply for the coding
node; anything else fails the suite) and `app.config.CODING_WORKSPACE_ROOT`
(pointed at a temporary workspace). No app startup, no live model, no
network, no Chroma.
"""

import hashlib
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

import app.graph as gm  # noqa: E402
from app.execution.events import (  # noqa: E402
    EXECUTION_APPROVAL_COMPLETED,
    EXECUTION_WAITING_APPROVAL,
)
from app.execution.runner import GraphRunner  # noqa: E402
from app.execution.store import (  # noqa: E402
    ExecutionStore,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_WAITING_APPROVAL,
)
from app.supervisor import keyword_route  # noqa: E402

QUESTION = "Fix the off-by-one bug in calc.py"

CALC = (
    "def cumulative(n):\n"
    "    total = 0\n"
    "    for i in range(n):\n"
    "        total += i\n"
    "    return total\n"
)
CALC_FIXED = (
    "def cumulative(n):\n"
    "    total = 0\n"
    "    for i in range(n + 1):\n"
    "        total += i\n"
    "    return total\n"
)

REPLY = json.dumps({
    "summary": "range(n) excludes n",
    "changes": [{
        "file": "calc.py",
        "find": "for i in range(n):",
        "replace": "for i in range(n + 1):",
        "reason": "off-by-one",
    }],
})


class _FakeLLM:
    def invoke(self, prompt, *args, **kwargs):
        return SimpleNamespace(content=REPLY)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8")
                          .encode("utf-8")).hexdigest()


class PatchExecutionCase(unittest.TestCase):
    """One fresh workspace + store + runner per test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "project"
        self.root.mkdir(parents=True)
        self.calc = self.root / "calc.py"
        self.calc.write_text(CALC, encoding="utf-8")
        self.patcher = mock.patch("app.config.CODING_WORKSPACE_ROOT",
                                  str(self.root))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.llm = mock.patch("app.llm_client.get_llm",
                              lambda *a, **k: _FakeLLM())
        self.llm.start()
        self.addCleanup(self.llm.stop)
        self.store = ExecutionStore(Path(self._tmp.name) / "store.sqlite3")
        self.addCleanup(self.store.close)
        self.runner = GraphRunner(self.store)

    # -- shared steps ------------------------------------------------------

    def prepare(self, question=QUESTION):
        eid = self.runner.create(question)["id"]
        row = self.runner.run(eid)
        approvals = self.store.list_approvals()
        self.assertEqual(len(approvals), 1)
        return eid, row, approvals[0]

    def pause_is_honest(self, row, approval):
        self.assertEqual(row["route"], "coding")
        self.assertEqual(row["status"], STATUS_WAITING_APPROVAL)
        self.assertIn("NOT modified any file", row["answer"])
        self.assertIn("```diff", row["answer"])
        self.assertEqual(approval["operation"], "workspace_patch")
        self.assertEqual(approval["status"], "pending")
        payload = approval["payload"]
        self.assertEqual(payload["question"], QUESTION)
        self.assertEqual(payload["workspace"], str(self.root))
        self.assertEqual(payload["files"][0]["before_sha256"],
                         file_hash(self.calc), "bound to the pre-change hash")
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC,
                         "preparation never writes")

    def execute(self, approval):
        self.store.decide_approval(approval["approval_id"], "approved")
        return self.runner.resume_approval(approval["approval_id"])


class TestApprovedApplication(PatchExecutionCase):
    def test_full_lifecycle_prepare_approve_apply(self):
        eid, row, approval = self.prepare()
        self.pause_is_honest(row, approval)
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC)

        result = self.execute(approval)
        self.assertEqual(result["outcome"], "executed")
        self.assertEqual(result["applied"], ["calc.py"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC_FIXED)

        execution = result["execution"]
        self.assertEqual(execution["status"], STATUS_COMPLETED)
        self.assertIn("Applied the approved workspace patch", execution["answer"])
        self.assertIn("calc.py", execution["answer"])
        self.assertIn("does not prove the change is correct",
                      execution["answer"])

        types = [e["type"] for e in self.store.get_events(eid)]
        for expected in (EXECUTION_WAITING_APPROVAL,
                         EXECUTION_APPROVAL_COMPLETED):
            self.assertIn(expected, types)

    def test_resume_is_single_execution_and_replay_is_denied(self):
        _, _, approval = self.prepare()
        aid = approval["approval_id"]
        first = self.execute(approval)
        self.assertEqual(first["outcome"], "executed")

        replay = self.runner.resume_approval(aid)
        self.assertEqual(replay["outcome"], "denied")
        self.assertIn("already executed", replay["reason"])
        # The applied result is never overwritten by the denial.
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC_FIXED)
        execution = replay["execution"]
        self.assertEqual(execution["status"], STATUS_COMPLETED)
        self.assertIn("Applied the approved workspace patch",
                      execution["answer"])

    def runner_exec_id(self):
        rows = self.store.list_executions()
        return rows[0]["id"]

    def test_resume_before_any_decision_is_refused(self):
        _, _, approval = self.prepare()
        aid = approval["approval_id"]
        from app.execution.store import ApprovalDecisionRequired

        with self.assertRaises(ApprovalDecisionRequired):
            self.runner.resume_approval(aid)
        self.assertEqual(
            self.store.get_approval(aid)["status"], "pending")
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC)


class TestRefusals(PatchExecutionCase):
    def test_stale_file_is_never_applied(self):
        eid, row, approval = self.prepare()
        self.pause_is_honest(row, approval)
        self.calc.write_text("# owner edited me\n", encoding="utf-8")

        result = self.execute(approval)
        self.assertEqual(result["outcome"], "failed")
        self.assertIn("changed since the patch was prepared", result["reason"])
        self.assertEqual(result["applied"], [])

        execution = result["execution"]
        self.assertEqual(execution["status"], STATUS_FAILED)
        self.assertEqual(execution["error"]["category"], "stale_patch")
        self.assertIn("NOT applied", execution["answer"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"),
                         "# owner edited me\n",
                         "the owner's edit wins; the review is void")

    def test_absent_workspace_refuses_the_apply(self):
        _, _, approval = self.prepare()
        self.patcher.stop()   # workspace motor disabled after approval
        result = self.execute(approval)
        self.assertEqual(result["outcome"], "failed")
        execution = result["execution"]
        self.assertEqual(execution["error"]["category"],
                         "workspace_unavailable")
        self.assertIn("NOT applied", execution["answer"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC)

    def test_rejection_closes_failed_without_writing(self):
        eid, row, approval = self.prepare()
        self.pause_is_honest(row, approval)
        aid = approval["approval_id"]
        from app.execution.store import ApprovalConflict

        self.store.decide_approval(aid, "rejected")
        with self.assertRaises(ApprovalConflict):
            self.store.decide_approval(aid, "approved")   # no overwrite
        result = self.runner.resume_approval(aid)
        self.assertEqual(result["outcome"], "rejected")
        execution = result["execution"]
        self.assertEqual(execution["status"], STATUS_FAILED)
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC)

    def test_second_execution_prepares_its_own_request(self):
        eid, _, approval = self.prepare()
        self.execute(approval)
        # A SECOND patch from an already-fixed file must not be auto-allowed:
        # the earlier approval is DECIDED, so it can never release new work.
        eid2 = self.runner.create(QUESTION)["id"]
        self.assertNotEqual(eid2, eid)
        row2 = self.runner.run(eid2)
        self.assertEqual(row2["status"], STATUS_COMPLETED,
                         "no change possible -> no patch is prepared, "
                         "the honest completion stays")
        self.assertEqual(self.store.list_approvals(), [],
                         "no NEW pending request is created")
        self.assertEqual(
            self.store.get_approval(approval["approval_id"])["status"],
            "approved", "the first decision is untouched")
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC_FIXED)


class TestInterruptedApplyRecovery(PatchExecutionCase):
    """A resume that claimed the approval and then died before its terminal
    write must be reconciled from DISK, never re-applied blindly."""

    def consume_claim(self, approval):
        """Reproduce the crash: the claim is taken, the terminal write is lost."""
        self.store.decide_approval(approval["approval_id"], "approved")
        self.assertTrue(
            self.store.claim_approval_execution(approval["approval_id"]))
        self.assertEqual(
            self.store.get_execution(self.store.list_executions()[0]["id"])
            ["status"], STATUS_WAITING_APPROVAL,
            "the execution is still paused: that is the interrupted state")

    def test_patch_present_on_disk_is_recovered_not_reapplied(self):
        _, _, approval = self.prepare()
        self.consume_claim(approval)
        # The earlier attempt WROTE the file, then died before its status.
        self.calc.write_text(CALC_FIXED, encoding="utf-8")

        result = self.runner.resume_approval(approval["approval_id"])
        self.assertEqual(result["outcome"], "recovered")
        self.assertEqual(result["applied"], ["calc.py"])
        self.assertEqual(result["execution"]["status"], STATUS_COMPLETED)
        self.assertIn("present on disk", result["answer"])
        self.assertIn("does not prove the change is correct",
                      result["answer"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC_FIXED)

    def test_nothing_written_is_closed_failed_not_reapplied(self):
        _, _, approval = self.prepare()
        self.consume_claim(approval)

        result = self.runner.resume_approval(approval["approval_id"])
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(result["execution"]["status"], STATUS_FAILED)
        self.assertEqual(result["execution"]["error"]["category"],
                         "patch_apply_unverified")
        self.assertIn("NOT re-applied", result["answer"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC,
                         "recovery never writes")

    def test_unproven_file_state_fails_honestly(self):
        _, _, approval = self.prepare()
        self.consume_claim(approval)
        # Neither the reviewed BEFORE nor the reviewed AFTER content.
        self.calc.write_text("# owner edited me\n", encoding="utf-8")

        result = self.runner.resume_approval(approval["approval_id"])
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(result["execution"]["error"]["category"],
                         "patch_apply_conflict")
        self.assertIn("unproven: calc.py", result["answer"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"),
                         "# owner edited me\n")


class TestRoutingSeparation(unittest.TestCase):
    def test_workspace_enabled_routes_file_requests_to_coding(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("app.config.CODING_WORKSPACE_ROOT", tmp):
                self.assertEqual(keyword_route(
                    {"question": QUESTION})["route"], "coding")
                self.assertEqual(keyword_route(
                    {"question": "Explain the bug in calc.py"})["route"],
                    "coding")

    def test_workspace_disabled_routes_here_only_to_explain(self):
        """2026-10-30: routing no longer hides a switched-off capability.

        With no workspace the request used to take the `unsupported` route and
        answer "I have no tool for it", which is untrue: the capability exists
        and is disabled. It now reaches the coding node, whose only behaviour
        in that state is to report exactly why it cannot run — and it prepares
        nothing, so no file can be touched.
        """
        from app.coding.agent import coding_node

        with mock.patch("app.config.CODING_WORKSPACE_ROOT", ""):
            self.assertEqual(keyword_route({"question": QUESTION})["route"],
                             "coding")
            update = coding_node({"question": QUESTION})
        self.assertIn("not enabled", update["answer"])
        self.assertIn("CODING_WORKSPACE_ROOT", update["answer"])
        self.assertEqual(update["coding"]["status"], "unavailable")
        self.assertIsNone(update["coding"].get("approval_id"))

    def test_other_source_cues_outrank_coding_even_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("app.config.CODING_WORKSPACE_ROOT", tmp):
                self.assertEqual(keyword_route({
                    "question": "Search the web for the latest changes "
                                "to app/graph.py"})["route"], "research")
                self.assertEqual(keyword_route({
                    "question": "What does the manual say about testing "
                                "in chapter 3?"})["route"], "books")

    def test_plain_knowledge_question_stays_general(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("app.config.CODING_WORKSPACE_ROOT", tmp):
                self.assertEqual(keyword_route({
                    "question": "How does Python's range function work?"}
                )["route"], "general")


if __name__ == "__main__":
    unittest.main(verbosity=2)
