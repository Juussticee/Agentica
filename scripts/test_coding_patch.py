"""Patch value semantics (2026-10-30) — offline.

A patch is a REVIEWABLE value: content hashes, a diff, and static offline
validation. These tests prove that preparation never writes, that a patch
that changed the file is refused in full, that the persisted payload cannot
be tampered with, and that a partial write failure is reported honestly.

No app startup, no LLM, no network.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.coding.patch import (  # noqa: E402
    apply_patch,
    build_patch,
    patch_digest,
    sha256_text,
    static_validate,
    stale_files,
    unified_diff,
    validate_payload,
)
from app.coding.workspace import Workspace  # noqa: E402


def make_ws():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "project"
    (root / "app").mkdir(parents=True)
    calc = root / "app" / "calc.py"
    calc.write_text(
        "def cumulative(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += i\n"
        "    return total\n",
        encoding="utf-8",
    )
    util = root / "app" / "util.py"
    util.write_text("def double(x):\n    return x * 2\n", encoding="utf-8")
    return tmp, Workspace(root), calc, util


NEW_CALC = (
    "def cumulative(n):\n"
    "    total = 0\n"
    "    for i in range(n + 1):\n"
    "        total += i\n"
    "    return total\n"
)
NEW_UTIL = (
    "def double(x):\n"
    "    \"\"\"Doubles a number.\"\"\"\n"
    "    return x * 2\n"
)


class TestBuildPatch(unittest.TestCase):
    def setUp(self):
        self._tmp, self.ws, self.calc, self.util = make_ws()
        self.addCleanup(self._tmp.cleanup)
        self.calc_raw = self.calc.read_text(encoding="utf-8")

    def test_build_produces_hashes_diff_and_validation(self):
        patch = build_patch(self.ws, "fix off-by-one", [("app/calc.py", NEW_CALC)])
        (change,) = patch.files
        self.assertEqual(change.path, "app/calc.py")
        self.assertEqual(change.before_sha256, sha256_text(self.calc_raw))
        self.assertEqual(change.after_sha256, sha256_text(NEW_CALC))
        self.assertIn("-    for i in range(n):", change.diff)
        self.assertIn("+    for i in range(n + 1):", change.diff)
        self.assertEqual(
            [v["status"] for v in patch.validation], ["passed"])
        # Preparation is a pure read: the workspace is untouched.
        self.assertEqual(self.calc.read_text(encoding="utf-8"), self.calc_raw)

    def test_empty_change_is_refused(self):
        with self.assertRaises(Exception) as ctx:
            build_patch(self.ws, "noop", [("app/calc.py", self.calc_raw)])
        self.assertIn("empty", str(ctx.exception))

    def test_duplicate_path_refused(self):
        with self.assertRaises(Exception) as ctx:
            build_patch(self.ws, "dup", [
                ("app/calc.py", NEW_CALC), ("app/calc.py", "x + 1")])
        self.assertIn("twice", str(ctx.exception))

    def test_too_many_files_refused(self):
        with self.assertRaises(Exception):
            build_patch(self.ws, "big", [
                (f"app/f{i}.py", "x") for i in range(4)])

    def test_missing_file_refused(self):
        with self.assertRaises(Exception):
            build_patch(self.ws, "gone", [("app/ghost.py", "x = 1\n")])

    def test_multi_file_patch(self):
        patch = build_patch(self.ws, "two", [
            ("app/calc.py", NEW_CALC), ("app/util.py", NEW_UTIL)])
        self.assertEqual([c.path for c in patch.files],
                         ["app/calc.py", "app/util.py"])
        self.assertEqual(patch.paths, ["app/calc.py", "app/util.py"])
        # Every target still has the RAW pre-change bytes on disk.
        self.assertEqual(self.calc.read_text(encoding="utf-8"), self.calc_raw)

    def test_validation_failure_is_recorded_not_raised(self):
        patch = build_patch(
            self.ws, "broken",
            [("app/calc.py", "for i in range(n\n")])
        self.assertEqual(
            [v["status"] for v in patch.validation], ["failed"])
        self.assertIn("syntaxerror", patch.validation[0]["detail"].lower())


class TestStaticValidate(unittest.TestCase):
    def test_python_ok(self):
        self.assertEqual(
            static_validate("a/x.py", "x = 1\n")["status"], "passed")

    def test_python_syntax_error(self):
        v = static_validate("a/x.py", "def f(:\n    pass\n")
        self.assertEqual(v["status"], "failed")
        self.assertIn("line 1", v["detail"])

    def test_json_ok_and_broken(self):
        self.assertEqual(
            static_validate("a/x.json", '{"k": 1}')["status"], "passed")
        self.assertEqual(
            static_validate("a/x.json", "{oops}")["status"], "failed")

    def test_unknown_type_is_honest(self):
        v = static_validate("a/README.md", "***")
        self.assertEqual(v["status"], "not_validated")


class TestPayloadTamperChecks(unittest.TestCase):
    def setUp(self):
        self._tmp, self.ws, self.calc, self.util = make_ws()
        self.addCleanup(self._tmp.cleanup)

    def _payload(self):
        return build_patch(
            self.ws, "q", [("app/calc.py", NEW_CALC)]).to_payload()

    def test_payload_round_trips_through_validate(self):
        changes = validate_payload(self._payload())
        self.assertEqual(changes[0].after_content, NEW_CALC)

    def test_tampered_content_is_refused(self):
        payload = self._payload()
        payload["files"][0]["after_content"] = NEW_CALC.replace("range", "RANGE")
        with self.assertRaises(Exception) as ctx:
            validate_payload(payload)
        self.assertIn("hash", str(ctx.exception))

    def test_missing_files_refused(self):
        payload = self._payload()
        payload["files"] = []
        with self.assertRaises(Exception):
            validate_payload(payload)

    def test_no_workspace_refused(self):
        payload = self._payload()
        payload["workspace"] = ""
        with self.assertRaises(Exception):
            validate_payload(payload)

    def test_digest_tracks_the_file_set_only(self):
        patch = build_patch(self.ws, "q", [("app/calc.py", NEW_CALC)])
        payload = patch.to_payload()
        base = patch_digest(payload)
        # Same file set -> same digest, whatever the validation notes say.
        payload["validation"] = [{"path": "app/calc.py", "status": "failed"}]
        self.assertEqual(patch_digest(payload), base)
        # Different content -> different digest.
        other = build_patch(
            self.ws, "q", [("app/calc.py", "x = 2\n")]).to_payload()
        self.assertNotEqual(patch_digest(other), base)
        self.assertNotEqual(patch.digest, patch_digest(other))

    def test_unified_diff_shape(self):
        diff = unified_diff("a/x.py", "old\n", "new\n")
        self.assertTrue(diff.startswith("a/a/x.py") or bool(diff))
        self.assertIn("+new", diff)
        self.assertIn("-old", diff)


class TestStaleAndApply(unittest.TestCase):
    def setUp(self):
        self._tmp, self.ws, self.calc, self.util = make_ws()
        self.addCleanup(self._tmp.cleanup)

    def _two_file_changes(self):
        patch = build_patch(self.ws, "q", [
            ("app/calc.py", NEW_CALC), ("app/util.py", NEW_UTIL)])
        return patch, validate_payload(patch.to_payload())

    def test_stale_files_detects_external_edit(self):
        _, changes = self._two_file_changes()
        self.assertEqual(stale_files(self.ws, changes), [])
        self.calc.write_text("# owner edit\n", encoding="utf-8")
        self.assertEqual(
            stale_files(self.ws, changes), ["app/calc.py"])

    def test_apply_writes_both_files(self):
        _, changes = self._two_file_changes()
        result = apply_patch(self.ws, changes)
        self.assertTrue(result.ok)
        self.assertEqual(result.applied, ["app/calc.py", "app/util.py"])
        self.assertEqual(self.calc.read_text(encoding="utf-8"), NEW_CALC)
        self.assertEqual(self.util.read_text(encoding="utf-8"), NEW_UTIL)

    def test_apply_refuses_everything_when_one_file_went_stale(self):
        _, changes = self._two_file_changes()
        self.calc.write_text("# owner edit\n", encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            apply_patch(self.ws, changes)
        self.assertIn("app/calc.py", str(ctx.exception))
        # NOTHING was written — not even the untouched util.py.
        self.assertEqual(self.util.read_text(encoding="utf-8"),
                         "def double(x):\n    return x * 2\n")

    def test_apply_refuses_a_workspace_change(self):
        patch, changes = self._two_file_changes()
        payload = patch.to_payload()
        payload["workspace"] = "C:/somewhere/else"
        with self.assertRaises(Exception) as ctx:
            apply_patch(self.ws, changes, payload)
        self.assertIn("workspace", str(ctx.exception))
        self.assertEqual(self.calc.read_text(encoding="utf-8"),
                         "def cumulative(n):\n"
                         "    total = 0\n"
                         "    for i in range(n):\n"
                         "        total += i\n"
                         "    return total\n")

    def test_apply_refuses_a_digest_that_no_longer_matches_its_payload(self):
        patch, changes = self._two_file_changes()
        payload = patch.to_payload()
        payload["digest"] = "a" * 64
        with self.assertRaises(Exception) as ctx:
            apply_patch(self.ws, changes, payload)
        self.assertIn("digest", str(ctx.exception).lower())
        self.assertEqual(self.calc.read_text(encoding="utf-8"),
                         "def cumulative(n):\n"
                         "    total = 0\n"
                         "    for i in range(n):\n"
                         "        total += i\n"
                         "    return total\n")

    def test_partial_failure_reports_honestly(self):
        _, changes = self._two_file_changes()
        real_replace = self.ws.replace_text

        def flaky(relative, expected, content):
            if relative == "app/util.py":
                raise OSError("disk went away")
            return real_replace(relative, expected, content)

        with mock.patch.object(self.ws, "replace_text", side_effect=flaky):
            result = apply_patch(self.ws, changes)
        self.assertFalse(result.ok)
        self.assertEqual(result.applied, ["app/calc.py"])
        self.assertEqual([f["path"] for f in result.failed], ["app/util.py"])
        # calc.py genuinely landed (never hidden as "failed").
        self.assertEqual(self.calc.read_text(encoding="utf-8"), NEW_CALC)
        self.assertEqual(self.util.read_text(encoding="utf-8"),
                         "def double(x):\n    return x * 2\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
