"""The `coding` graph node (2026-10-30) — offline, model at the seam.

The node has two deterministic shapes decided by the request itself:

  * INSPECT — read-only. Files are read (redacted) and the model answers;
    NOTHING writes and NO approval request is ever created.
  * PATCH — the model proposes exact find/replace edits over the files that
    were shown; the node builds the unified diff, statically validates it,
    and persists ONE `workspace_patch` approval request through the EXISTING
    approval lifecycle. The workspace is never written here.

The model is faked exactly at `app.llm_client.get_llm`, so any unexpected
provider call fails the suite instead of reaching a provider. No app
startup, no network, no Chroma.
"""

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
from app.coding.patch import patch_digest  # noqa: E402
from app.coding.workspace import hash_text  # noqa: E402
from app.execution.store import ExecutionStore  # noqa: E402

INSPECT_QUESTION = "What is wrong with the cumulative(n) function in app/calc.py?"
PATCH_QUESTION = "Fix the off-by-one bug in app/calc.py"

CALC = (
    "def cumulative(n):\n"
    "    total = 0\n"
    "    for i in range(n):\n"
    "        total += i\n"
    "    return total\n"
)


class _FakeLLM:
    def __init__(self, reply):
        self._reply = reply

    def invoke(self, prompt, *args, **kwargs):
        return SimpleNamespace(content=self._reply)


def fake_llm(reply):
    """Patch app.llm_client.get_llm for one test (coding node imports it at call time)."""
    return mock.patch("app.llm_client.get_llm",
                      lambda *args, **kwargs: _FakeLLM(reply))


def js_reply(summary, changes):
    import json

    return json.dumps({"summary": summary, "changes": changes})


class CodingCase(unittest.TestCase):
    """One fresh workspace + store per test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "project"
        (self.root / "app").mkdir(parents=True)
        self.calc = self.root / "app" / "calc.py"
        self.calc.write_text(CALC, encoding="utf-8")
        self.patcher = mock.patch("app.config.CODING_WORKSPACE_ROOT",
                                  str(self.root))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.store = ExecutionStore(Path(self._tmp.name) / "store.sqlite3")
        self.addCleanup(self.store.close)
        # Approval rows are FK-bound to their owning execution.
        self.store.create_execution("exraft00001", PATCH_QUESTION)

    def run_node(self, question):
        """Run the coding node exactly like the graph does."""
        from app.coding.agent import coding_node

        with gm.approval_scope(self.store, "exraft00001"):
            return coding_node({"question": question})

    def approvals(self):
        return self.store.list_approvals()

    def assert_no_writes(self):
        # The only fixture file still has its exact original bytes.
        self.assertEqual(self.calc.read_text(encoding="utf-8"), CALC)


class TestInspect(CodingCase):
    def test_inspect_is_read_only_and_needs_no_approval(self):
        with fake_llm("The bug is in cumulative(): range(n) excludes n."):
            update = self.run_node(INSPECT_QUESTION)
        coding = update["coding"]
        self.assertEqual(coding["status"], "inspected")
        self.assertTrue(coding["read_only"])
        self.assertEqual(coding["files"], ["app/calc.py"])
        self.assertEqual(coding["workspace"], str(self.root))
        self.assertNotIn("approval_id", coding)
        self.assertEqual(self.approvals(), [])
        self.assert_no_writes()

    def test_inspect_headers_name_files_and_evidence(self):
        with fake_llm("analysis"):
            update = self.run_node(INSPECT_QUESTION)
        answer = update["answer"]
        self.assertIn("Workspace:", answer)
        self.assertIn("app/calc.py", answer)
        # Source evidence lines come from Workspace.search, path:line form.
        self.assertRegex(answer, r"app/calc\.py:\d+:")

    def test_inspect_never_counts_as_a_patch(self):
        with fake_llm("The bug is subtle."):
            update = self.run_node(
                "Explain the off-by-one bug in app/calc.py line by line.")
        self.assertEqual(update["coding"]["status"], "inspected")
        self.assertEqual(self.approvals(), [])
        self.assert_no_writes()


class TestPatchPreparation(CodingCase):
    def test_patch_is_prepared_not_applied(self):
        reply = js_reply("loop excludes n", [{
            "file": "app/calc.py",
            "find": "for i in range(n):",
            "replace": "for i in range(n + 1):",
            "reason": "off-by-one",
        }])
        with fake_llm(reply):
            update = self.run_node(PATCH_QUESTION)
        coding = update["coding"]
        self.assertEqual(coding["status"], "patch_prepared")
        self.assertIn("NOT modified any file", update["answer"])
        self.assertIn("```diff", update["answer"])
        self.assertIn("+    for i in range(n + 1):", update["answer"])
        self.assert_no_writes()

    def test_approval_request_binds_patch_workspace_and_question(self):
        reply = js_reply("s", [{
            "file": "app/calc.py",
            "find": "for i in range(n):",
            "replace": "for i in range(n + 1):",
        }])
        with fake_llm(reply):
            update = self.run_node(PATCH_QUESTION)
        rows = self.approvals()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["operation"], "workspace_patch")
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["approval_id"],
                         update["coding"]["approval_id"])
        payload = row["payload"]
        self.assertEqual(payload["question"], PATCH_QUESTION)
        self.assertEqual(payload["workspace"], str(self.root))
        (fr, ) = payload["files"]
        self.assertEqual(fr["path"], "app/calc.py")
        self.assertEqual(fr["before_sha256"], hash_text(CALC))
        self.assertIn("range(n + 1)",
                      [f["diff"] for f in payload["files"]][0])
        self.assertEqual(update["coding"]["digest"],
                         patch_digest(payload))

    def test_preparation_is_idempotent_one_pending_request(self):
        reply = js_reply("s", [{
            "file": "app/calc.py",
            "find": "for i in range(n):",
            "replace": "for i in range(n + 1):",
        }])
        with fake_llm(reply):
            first = self.run_node(PATCH_QUESTION)
            second = self.run_node(PATCH_QUESTION)
        self.assertEqual(
            first["coding"]["approval_id"],
            second["coding"]["approval_id"])
        self.assertEqual(len(self.approvals()), 1)
        self.assert_no_writes()

    def test_unusable_model_reply_blocks_the_patch(self):
        with fake_llm("I would totally fix that for you."):
            update = self.run_node(PATCH_QUESTION)
        self.assertEqual(update["coding"]["status"], "patch_blocked")
        self.assertIn("not prepare a patch", update["answer"])
        self.assertIn("JSON", update["answer"])
        self.assertEqual(self.approvals(), [])
        self.assert_no_writes()

    def test_find_not_in_file_blocks_the_patch(self):
        reply = js_reply("s", [{
            "file": "app/calc.py",
            "find": "for i in range(n << 2):",
            "replace": "for i in range(n + 1):",
        }])
        with fake_llm(reply):
            update = self.run_node(PATCH_QUESTION)
        self.assertEqual(update["coding"]["status"], "patch_blocked")
        self.assertIn("exactly once", update["answer"])
        self.assertEqual(self.approvals(), [])
        self.assert_no_writes()

    def test_model_cannot_reach_files_it_was_not_shown(self):
        # other.py resolves OUTSIDE the workspace, so it is never shown;
        # a reply that still targets it must be refused with no approval.
        reply = js_reply("s", [{
            "file": "../outside.py",
            "find": "x",
            "replace": "y",
        }])
        with fake_llm(reply):
            update = self.run_node(PATCH_QUESTION)
        self.assertEqual(update["coding"]["status"], "patch_blocked")
        self.assertIn("outside.py", update["answer"])
        self.assertEqual(self.approvals(), [])
        self.assert_no_writes()

    def test_validation_failure_never_offers_an_approval(self):
        reply = js_reply("s", [{
            "file": "app/calc.py",
            "find": "for i in range(n):",
            "replace": "for i in range(n:",   # syntactically broken
        }])
        with fake_llm(reply):
            update = self.run_node(PATCH_QUESTION)
        self.assertEqual(update["coding"]["status"], "patch_blocked")
        self.assertIn("FAILED offline validation", update["answer"])
        self.assertIn("line", update["answer"])
        self.assertEqual(self.approvals(), [])
        self.assert_no_writes()

    def test_without_approval_scope_nothing_is_registered_or_written(self):
        from app.coding.agent import coding_node

        reply = js_reply("s", [{
            "file": "app/calc.py",
            "find": "for i in range(n):",
            "replace": "for i in range(n + 1):",
        }])
        with fake_llm(reply):
            update = coding_node({"question": PATCH_QUESTION})
        self.assertEqual(update["coding"]["status"], "patch_blocked")
        self.assertIn("approval", update["answer"].lower())
        self.assertEqual(self.approvals(), [])
        self.assert_no_writes()


class TestUnavailableAndNoTarget(CodingCase):
    def test_disabled_workspace_answers_honestly(self):
        self.patcher.stop()   # CODING_WORKSPACE_ROOT back to "" (default)
        from app.coding.agent import coding_node

        update = coding_node({"question": PATCH_QUESTION})
        self.assertEqual(update["coding"]["status"], "unavailable")
        self.assertIn("not enabled", update["answer"])
        self.assertIn("CODING_WORKSPACE_ROOT", update["answer"])
        self.assert_no_writes()

    def test_no_named_source_file_asks_for_a_target(self):
        with fake_llm("anything"):
            update = self.run_node("Fix the off-by-one bug in the loop.")
        self.assertEqual(update["coding"]["status"], "no_target")
        self.assertIn("could not find a source file", update["answer"])
        self.assertEqual(self.approvals(), [])
        self.assert_no_writes()

    def test_denied_path_is_reported_not_read(self):
        (self.root / "app" / "secrets.py").write_text(
            "SECRET = \"saturn\"\n", encoding="utf-8")
        with fake_llm("analysis"):
            update = self.run_node("What does app/secrets.py contain?")
        # The denied file is never read; the honest refusal names it.
        self.assertIn("secrets.py", update["answer"])
        self.assertEqual(update["coding"]["status"], "no_target")
        self.assertEqual(self.approvals(), [])
        self.assertNotIn("saturn", update["answer"])
        self.assert_no_writes()


if __name__ == "__main__":
    unittest.main(verbosity=2)
