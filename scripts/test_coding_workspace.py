"""The workspace boundary of the coding capability (2026-10-30) — offline.

Proves the contract in app/coding/workspace.py exactly as the capability is
used: one configured root, traversal/symlink escapes refused, credential
files and secret-looking values never exposed, reads bounded and read-only,
and the single write path refusing to touch a file whose content changed.

No app startup, no LLM, no network: temporary directories only.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.coding.workspace import (  # noqa: E402
    MAX_FILE_BYTES,
    PathDenied,
    PathOutsideWorkspace,
    Workspace,
    WorkspaceError,
    WorkspaceUnavailable,
    configured_workspace,
    hash_text,
    redact_secrets,
)


def make_dir_link(link: Path, target: Path) -> bool:
    """Create a REAL directory link, without mocking resolve().

    POSIX uses a symlink. Windows uses a directory junction, which needs no
    elevation (a Windows symlink does, so on an unprivileged host the symlink
    attempt fails and the junction is what actually gets tested).
    Returns False only when neither exists on this host.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name != "nt":
        return False
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True)
    return created.returncode == 0 and link.exists()


class WorkspaceFactory:
    """Builds a real directory tree; mixes into unittest.TestCase body via setUp()."""

    def _setup_workspace(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "project"
        (self.root / "app").mkdir(parents=True)
        self.calc = self.root / "app" / "calc.py"
        self.calc.write_text(
            "def cumulative(n):\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        total += i\n"
            "    return total\n",
            encoding="utf-8",
        )
        self.ws = Workspace(self.root)


class TestConstruction(unittest.TestCase):
    def test_empty_configuration_is_unavailable(self):
        with mock.patch("app.config.CODING_WORKSPACE_ROOT", ""):
            self.assertIsNone(configured_workspace())

    def test_read_at_call_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                    "app.config.CODING_WORKSPACE_ROOT", str(Path(tmp) / "x")):
                # A relative-looking absolute? No — tmp IS absolute; missing
                # dir raises the honest unavailable error.
                with self.assertRaises(WorkspaceUnavailable):
                    configured_workspace()
            (Path(tmp) / "y").mkdir()
            with mock.patch(
                    "app.config.CODING_WORKSPACE_ROOT", str(Path(tmp) / "y")):
                ws = configured_workspace()
        self.assertIsNotNone(ws)

    def test_relative_root_is_refused(self):
        with self.assertRaises(WorkspaceUnavailable):
            Workspace("some/relative/path")

    def test_missing_directory_is_refused(self):
        with self.assertRaises(WorkspaceUnavailable):
            Workspace(tempfile.TemporaryDirectory().name / ".." / "nope"
                      if False else Path(tempfile.gettempdir(), "missing-zzz"))

    def test_root_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            inner = Path(tmp) / "real"
            inner.mkdir()
            link = Path(tmp) / "alias"
            if not make_dir_link(link, inner):
                self.skipTest("no symlink or junction support on this host")
            ws = Workspace(link)
        self.assertEqual(ws.root, inner.resolve())


class TestPathBoundary(WorkspaceFactory, unittest.TestCase):
    def setUp(self):
        self._setup_workspace()
    def test_empty_path(self):
        with self.assertRaises(WorkspaceError):
            self.ws.resolve("")

    def test_dotdot_rejected(self):
        with self.assertRaises(PathOutsideWorkspace):
            self.ws.resolve("../outside.py")

    def test_absolute_path_rejected(self):
        with self.assertRaises(PathOutsideWorkspace):
            self.ws.resolve(str(Path(tempfile.gettempdir(), "calc.py")))

    def test_lingering_escape(self):
        # app/../../etc-passwd-style: the .. hides INSIDE the tree.
        with self.assertRaises(PathOutsideWorkspace):
            self.ws.resolve("app/../../etc/passwd-source.py")

    def test_symlink_escape_rejected(self):
        # A link inside the root pointing OUT is read through resolve() and
        # the REAL target is checked, so the escape is refused.
        target = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, target, True)
        (target / "leak.py").write_text("x = 1\n", encoding="utf-8")
        if not make_dir_link(self.root / "linked", target):
            self.skipTest("no symlink or junction support on this host")
        # Proven against a REAL reparse point: resolve() follows it outside.
        with self.assertRaises(PathOutsideWorkspace):
            self.ws.resolve("linked/leak.py")
        with self.assertRaises(PathOutsideWorkspace):
            self.ws.read("linked/leak.py")

    def test_denied_directory_rejected(self):
        (self.root / "node_modules" / "pkg").mkdir(parents=True)
        (self.root / "node_modules" / "pkg" / "index.js").write_text("x")
        with self.assertRaises(PathDenied):
            self.ws.resolve("node_modules/pkg/index.js")

    def test_git_internal_rejected(self):
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("[core]")
        with self.assertRaises(PathDenied):
            self.ws.resolve(".git/config")

    def test_denied_file_names(self):
        cases = [
            ".env", ".env.local", "secrets.json", "credentials.yaml",
            "id_rsa", "id_ed25519.pub", "cert.pem", "api.key", "db.sqlite3",
            "users.db",
        ]
        for name in cases:
            target = self.root / "app" / name
            target.write_text("content")
            with self.assertRaises(PathDenied, msg=name):
                self.ws.resolve(f"app/{name}")

    def test_legitimate_token_named_source_is_readable(self):
        # Name-based denial is narrow: a token named in a normal filename
        # must not get the whole file denied.
        target = self.root / "app" / "tokenizer.py"
        target.write_text("TOKENS = 100\n", encoding="utf-8")
        f = self.ws.read("app/tokenizer.py")
        self.assertEqual(f.content, "TOKENS = 100\n")

    def test_binary_extension_refused_on_read(self):
        (self.root / "app" / "logo.png").write_bytes(b"\x89PNG")
        with self.assertRaises(PathDenied):
            self.ws.read("app/logo.png")

    def test_missing_file(self):
        with self.assertRaises(WorkspaceError):
            self.ws.read("app/ghost.py")


class TestReadAndRedaction(WorkspaceFactory, unittest.TestCase):
    def setUp(self):
        self._setup_workspace()
    def test_read_returns_raw_hash_with_redacted_content(self):
        f = self.ws.read("app/calc.py")
        self.assertEqual(f.path, "app/calc.py")
        self.assertFalse(f.truncated)
        self.assertEqual(
            f.sha256,
            hashlib.sha256(
                self.calc.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest())

    def test_secret_values_in_read_files_are_redacted(self):
        target = self.root / "app" / "settings_example.py"
        target.write_text(
            'api_key = "AIzaSyDOCa-w0nderful-not-real-7xYzAbCdEfIlJm"\n'
            'PASSWORD = "correct horse battery staple 4000"\n'
            "sk_prefix = 'sk-' + filler_slice_16plus\n"
            "return None\n",
            encoding="utf-8",
        )
        f = self.ws.read("app/settings_example.py")
        self.assertNotIn("AIzaSyDOCa", f.content)
        self.assertNotIn("correct horse", f.content)
        self.assertIn("***REDACTED***", f.content)
        # The RAW file on disk is untouched (redaction is view-only).
        self.assertIn("AIzaSyDOCa", target.read_text(encoding="utf-8"))

    def test_redact_helpers(self):
        blob = ("ghp_0123456789abcdefghij0123456789abcd\n"
                "---BEGIN RSA PRIVATE KEY---\n"          # case-insensitive form
                "put sk-deadbeefdeadbeefdead into config\n")
        red = redact_secrets(blob)
        self.assertNotIn("ghp_", red)
        self.assertNotIn("sk-deadbeef", red)
        self.assertIn("***REDACTED***", red)

    def test_too_large_file_refused(self):
        big = self.root / "app" / "big.py"
        big.write_text("#" * (MAX_FILE_BYTES + 1), encoding="utf-8")
        with self.assertRaises(WorkspaceError):
            self.ws.read("app/big.py")

    def test_read_is_never_a_write(self):
        before = self.calc.read_text(encoding="utf-8")
        self.ws.read("app/calc.py")
        self.assertEqual(self.calc.read_text(encoding="utf-8"), before)

    def test_hash_text_helper(self):
        self.assertEqual(hash_text("abc"),
                         hashlib.sha256(b"abc").hexdigest())


class TestSearch(WorkspaceFactory, unittest.TestCase):
    def setUp(self):
        self._setup_workspace()
    def test_search_returns_bounded_hits_with_line_numbers(self):
        (self.root / "app" / "other.py").write_text(
            "def cumulative(n):\n"
            "    return n\n",
            encoding="utf-8",
        )
        hits = self.ws.search("cumulative")
        paths = [h[0] for h in hits]
        self.assertIn("app/calc.py", paths)
        self.assertIn("app/other.py", paths)
        # 1-based: the `def cumulative(n):` line is line 1 in both files.
        self.assertEqual(min(h[1] for h in hits), 1)
        self.assertGreaterEqual(min(h[1] for h in hits), 1)

    def test_search_never_scans_denied_directories_or_files(self):
        secret_dir = self.root / "vendor" / "node_modulesOutside"
        nm = self.root / "node_modules"
        nm.mkdir()
        (nm / "leak.py").write_text("def leak(): pass\n", encoding="utf-8")
        (self.root / "app" / ".env").write_text("SECRET=saturn\n")
        hits = self.ws.search("leak")
        self.assertEqual(hits, [])
        hits = self.ws.search("saturn")                # denied name never read
        self.assertEqual(hits, [])

    def test_search_redacts_matched_lines(self):
        (self.root / "app" / "keys_used.py").write_text(
            'API_KEY = "sk-abcd1234abcd1234abcd1234abcd1234"\n',
            encoding="utf-8",
        )
        hits = self.ws.search("API_KEY")
        self.assertTrue(hits)
        self.assertNotIn("sk-abcd", hits[0][2])

    def test_short_needle_is_noop(self):
        self.assertEqual(self.ws.search("x"), [])


class TestSingleWritePath(WorkspaceFactory, unittest.TestCase):
    def setUp(self):
        self._setup_workspace()
    def test_replace_writes_exact_content_and_returns_new_hash(self):
        new = self.calc.read_text(encoding="utf-8").replace(
            "range(n)", "range(n + 1)")
        new_hash = self.ws.replace_text(
            "app/calc.py", hash_text(self.calc.read_text(encoding="utf-8")),
            new)
        self.assertEqual(self.calc.read_text(encoding="utf-8"), new)
        self.assertEqual(new_hash, hash_text(new))

    def test_replace_refuses_a_stale_hash(self):
        stale = hash_text(self.calc.read_text(encoding="utf-8"))
        self.calc.write_text("# owner edited me in the meantime\n",
                             encoding="utf-8")
        with self.assertRaises(WorkspaceError) as ctx:
            self.ws.replace_text("app/calc.py", stale, "# new\n")
        self.assertIn("changed since the patch was prepared",
                      str(ctx.exception))
        self.assertEqual(self.calc.read_text(encoding="utf-8"),
                         "# owner edited me in the meantime\n")

    def test_replace_denied_paths_fail_closed(self):
        with self.assertRaises(PathDenied):
            self.ws.replace_text("app/.env", "0" * 64, "SECRET=x\n")

    def test_replace_outside_root_fail_closed(self):
        with self.assertRaises(PathOutsideWorkspace):
            self.ws.replace_text("../victim.py", "0" * 64, "boom")


if __name__ == "__main__":
    unittest.main(verbosity=2)
