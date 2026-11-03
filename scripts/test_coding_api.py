"""Workspace patch through the REAL API (2026-10-30) — offline.

The coding capability is only useful if the ordinary product surface can drive
it, so this suite goes through the live endpoints, not the runner directly:

  POST /api/v1/executions                -> routed to `coding`, a patch is
                                            PREPARED and persisted as an
                                            approval request; status pauses
  GET  /api/v1/approvals                 -> the pending `workspace_patch`
  POST /api/v1/approvals/{id}/resume     -> 409 while still pending (no gate
                                            bypass, nothing written)
  POST /api/v1/approvals/{id}/approve    -> decision recorded
  POST /api/v1/approvals/{id}/resume     -> the reviewed patch is applied
  GET  /api/v1/executions/{id}           -> status completed, real answer
  resume again                           -> denied, the file is NOT rewritten
  reject                                 -> nothing is ever written

Seams (provider boundary only): `app.llm_client.get_llm` answers the coding
node with a fixed change set and RAISES for any other prompt, and
`app.config.CODING_WORKSPACE_ROOT` points at a temporary workspace. Routing,
the graph, the approval gate, the store, the runner and the write path are the
real ones. No app startup, no model, no network, no Chroma.
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

from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

from app.execution.store import ExecutionStore  # noqa: E402

QUESTION = "Fix the off-by-one bug in calc.py"

CALC = (
    "def cumulative(n):\n"
    "    total = 0\n"
    "    for i in range(n):\n"
    "        total += i\n"
    "    return total\n"
)
CALC_FIXED = CALC.replace("range(n):", "range(n + 1):")

REPLY = json.dumps({
    "summary": "range(n) excludes n",
    "changes": [{
        "file": "calc.py",
        "find": "for i in range(n):",
        "replace": "for i in range(n + 1):",
        "reason": "off-by-one",
    }],
})


class _ScriptedLLM:
    """Answers only the coding prompt; anything else fails the suite."""

    def __init__(self):
        self.prompts: list = []

    def invoke(self, prompt, *args, **kwargs):
        self.prompts.append(str(prompt))
        if "preparing a MINIMAL patch" in str(prompt):
            return SimpleNamespace(content=REPLY)
        raise AssertionError(
            "unexpected model call in an offline suite: "
            + str(prompt)[:120].replace("\n", " "))


class CodingApiTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.api.server as server  # noqa: E402

        self.server = server
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "project"
        self.root.mkdir()
        self.calc = self.root / "calc.py"
        self.calc.write_text(CALC, encoding="utf-8")

        self.workspace = mock.patch("app.config.CODING_WORKSPACE_ROOT",
                                    str(self.root))
        self.workspace.start()
        self.addCleanup(self.workspace.stop)

        self._orig_store = server._store
        self.store = ExecutionStore(Path(self._tmp.name) / "coding.sqlite3")
        server._store = self.store
        self.addCleanup(self._restore_store)
        self.addCleanup(self.store.close)

        self.llm = _ScriptedLLM()
        self.llm_patch = mock.patch("app.llm_client.get_llm",
                                    side_effect=lambda **_kwargs: self.llm)
        self.llm_patch.start()
        self.addCleanup(self.llm_patch.stop)

        self.client = TestClient(server.build_server())

    def _restore_store(self):
        self.server._store = self._orig_store

    # -- helpers ----------------------------------------------------------

    def _prepare(self):
        created = self.client.post("/api/v1/executions",
                                   json={"question": QUESTION})
        self.assertEqual(created.status_code, 201, created.text)
        body = created.json()
        approvals = self.client.get("/api/v1/approvals").json()["approvals"]
        self.assertEqual(len(approvals), 1)
        return body, approvals[0]

    def _execution(self, eid: str) -> dict:
        response = self.client.get(f"/api/v1/executions/{eid}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    # -- the workflow -----------------------------------------------------

    def test_prepare_then_approved_resume_applies_the_patch(self):
        created, approval = self._prepare()
        eid = created["id"]

        # 1. The API ACCEPTED the work and paused honestly.
        execution = self._execution(eid)
        self.assertEqual(execution["route"], "coding")
        self.assertEqual(execution["status"], "waiting_for_approval")
        self.assertIn("NOT modified any file", execution["answer"])
        self.assertIn("```diff", execution["answer"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC)

        # 2. The pending request is a real workspace_patch, bound to this run.
        self.assertEqual(approval["operation"], "workspace_patch")
        self.assertEqual(approval["status"], "pending")
        self.assertEqual(approval["execution_id"], eid)
        self.assertEqual(approval["payload"]["question"], QUESTION)
        self.assertEqual(approval["payload"]["files"][0]["path"], "calc.py")

        # 3. Resuming before a decision cannot bypass the gate.
        early = self.client.post(
            f"/api/v1/approvals/{approval['approval_id']}/resume")
        self.assertEqual(early.status_code, 409, early.text)
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC)

        # 4. Approve, then resume: the reviewed patch is applied.
        approved = self.client.post(
            f"/api/v1/approvals/{approval['approval_id']}/approve")
        self.assertEqual(approved.status_code, 200, approved.text)
        resumed = self.client.post(
            f"/api/v1/approvals/{approval['approval_id']}/resume")
        self.assertEqual(resumed.status_code, 200, resumed.text)
        result = resumed.json()
        self.assertEqual(result["outcome"], "executed")
        self.assertEqual(result["applied"], ["calc.py"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC_FIXED)

        final = self._execution(eid)
        self.assertEqual(final["status"], "completed")
        self.assertIn("Applied the approved workspace patch", final["answer"])
        self.assertIn("does not prove the change is correct",
                      final["answer"])
        self.assertEqual(self.client.get("/api/v1/approvals")
                         .json()["approvals"], [],
                         "the decision is consumed, nothing stays pending")

        # 5. A replay is denied and NEVER rewrites the file.
        self.calc.write_text(CALC_FIXED + "# reviewed\n", encoding="utf-8")
        replay = self.client.post(
            f"/api/v1/approvals/{approval['approval_id']}/resume")
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json()["outcome"], "denied")
        self.assertEqual(self.calc.read_text(encoding="utf-8"),
                         CALC_FIXED + "# reviewed\n")

    def test_rejection_never_writes_and_closes_failed(self):
        created, approval = self._prepare()
        eid = created["id"]

        rejected = self.client.post(
            f"/api/v1/approvals/{approval['approval_id']}/reject")
        self.assertEqual(rejected.status_code, 200, rejected.text)
        resumed = self.client.post(
            f"/api/v1/approvals/{approval['approval_id']}/resume")
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.assertEqual(resumed.json()["outcome"], "rejected")

        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC,
                         "a rejected patch is never applied")
        execution = self._execution(eid)
        self.assertEqual(execution["status"], "failed")
        self.assertNotIn("Applied the approved workspace patch",
                         execution["answer"])

    def test_changed_file_is_refused_at_resume(self):
        created, approval = self._prepare()
        eid = created["id"]
        self.calc.write_text("# owner edited me\n", encoding="utf-8")

        self.client.post(f"/api/v1/approvals/{approval['approval_id']}/approve")
        resumed = self.client.post(
            f"/api/v1/approvals/{approval['approval_id']}/resume")
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.assertEqual(resumed.json()["outcome"], "failed")
        self.assertIn("changed since the patch was prepared",
                      resumed.json()["reason"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"),
                         "# owner edited me\n")

        execution = self._execution(eid)
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["error"]["category"], "stale_patch")
        self.assertIn("NOT applied", execution["answer"])

    def test_disabled_workspace_writes_nothing_and_says_so(self):
        # 2026-10-30: a workspace action with no configured root now routes to
        # `coding` on purpose, so the node itself states that the capability is
        # switched off ("there is no tool for it" would be untrue). What must
        # never happen is a write, an approval request, or an invented patch.
        with mock.patch("app.config.CODING_WORKSPACE_ROOT", ""):
            created = self.client.post("/api/v1/executions",
                                       json={"question": QUESTION})
            self.assertEqual(created.status_code, 201, created.text)
            self.assertEqual(created.json()["route"], "coding")
            execution = self._execution(created.json()["id"])
        self.assertIn("Workspace coding is not enabled", execution["answer"])
        self.assertEqual(execution["status"], "completed")
        self.assertEqual(self.client.get("/api/v1/approvals")
                         .json()["approvals"], [])
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC,)
        self.assertEqual(self.llm.prompts, [],
                         "the disabled capability must not call the model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
