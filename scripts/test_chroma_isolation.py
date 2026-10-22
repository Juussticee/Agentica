"""Chroma production-store isolation tests (offline, fail-closed, zero quota).

Covered behaviors (no Ollama, no network, no production Chroma open — every
production-path construction attempt must raise BEFORE chromadb runs):

  1. Production path is rejected before client construction.
  2. Equivalent normalized path spellings are rejected.
  3. Temporary disposable paths are permitted (system temp).
  4. Direct persistent-client construction is guarded (constructor wrapper).
  5. Lazy query-engine initialization is guarded (`books_agent` seam).
  6. Import-time initialization cannot bypass the protection.
  7. Existing tests that mock the query engine continue to work.
  8. Rejected attempts do not change the protected production files.
  9. The guard fails closed if its configuration is missing or ambiguous.

    env\\Scripts\\python.exe -m pytest scripts/test_chroma_isolation.py -q

Self-bootstraps: runs identically under pytest or as a direct script.
"""

import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

import chromadb  # noqa: E402  (wrapped by install_persistent_client_wrapper)
import app.agents.books_agent as ba  # noqa: E402

PROD = chroma_guard.PRODUCTION_STORAGE_DIR


def _prod_signature() -> dict:
    """Filesystem-only snapshot (hash + size + mtime) — never opens the DB."""
    paths = [PROD / "chroma.sqlite3", PROD / ".index_manifest.json"]
    paths += sorted(PROD.glob("*/*.bin"))
    snap = {}
    for p in paths:
        st = p.stat()
        snap[str(p)] = (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            st.st_size,
            st.st_mtime_ns,
        )
    return snap


class _SentinelConstructionAttempted(AssertionError):
    pass


class TestActivation(unittest.TestCase):
    """Bullets 9 (+ activation contract): fail closed on missing/ambiguous config."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       (chroma_guard.GUARD_ENV, "PYTEST_CURRENT_TEST")}

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_empty_guard_env_still_arms(self):
        # Ambiguous (set but empty) configuration must arm, never bypass.
        os.environ[chroma_guard.GUARD_ENV] = ""
        self.assertTrue(chroma_guard.is_armed())
        with self.assertRaises(chroma_guard.ProductionStoreAccessError):
            chroma_guard.assert_store_path_allowed(PROD)

    def test_armed_rejects_production_even_disposable_list_ok(self):
        os.environ[chroma_guard.GUARD_ENV] = "banana-garbage-value"
        self.assertTrue(chroma_guard.is_armed())
        with self.assertRaises(chroma_guard.ProductionStoreAccessError):
            chroma_guard.assert_store_path_allowed(PROD)

    def test_disallowed_when_unconfigured_outside_test_context(self):
        # Production default: with NO configuration and no pytest marker the
        # guard is inert — normal app/CLI behavior is unchanged (requirement 7).
        os.environ.pop(chroma_guard.GUARD_ENV, None)
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        self.assertFalse(chroma_guard.is_armed())
        chroma_guard.assert_store_path_allowed(str(PROD))  # no-op, no open


class TestPathNormalizationAndPolicy(unittest.TestCase):
    """Bullets 2, 3, 4."""

    def setUp(self):
        chroma_guard.arm()
        self._saved_disposable = os.environ.get(chroma_guard.DISPOSABLE_ENV)

    def tearDown(self):
        if self._saved_disposable is None:
            os.environ.pop(chroma_guard.DISPOSABLE_ENV, None)
        else:
            os.environ[chroma_guard.DISPOSABLE_ENV] = self._saved_disposable

    def test_production_path_rejected(self):
        with self.assertRaises(chroma_guard.ProductionStoreAccessError):
            chroma_guard.assert_store_path_allowed(str(PROD))

    def test_equivalent_spellings_rejected(self):
        rel = os.path.relpath(PROD)
        variants = [
            str(PROD),
            rel,                                    # relative
            "./" + rel.replace("\\", "/") if not rel.startswith("..") else rel,
            str(PROD) + os.sep,                     # trailing separator
            os.path.join(str(PROD), "..", "storage"),  # dot-dot mixed
            str(PROD.parent / "storage" / "subdir"),    # inside production dir
        ]
        if sys.platform == "win32":
            variants.append(str(PROD).upper())      # case-different equivalent
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(chroma_guard.ProductionStoreAccessError):
                    chroma_guard.assert_store_path_allowed(variant)

    def test_symlink_equivalent_rejected(self):
        if sys.platform == "win32":
            self.skipTest("symlink creation typically needs privileges on Windows")
        with tempfile.TemporaryDirectory() as td:
            link = Path(td) / "storage-link"
            try:
                link.symlink_to(PROD, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(chroma_guard.ProductionStoreAccessError):
                chroma_guard.assert_store_path_allowed(str(link))

    def test_temp_disposable_path_permitted(self):
        with tempfile.TemporaryDirectory() as td:
            chroma_guard.assert_store_path_allowed(td)          # explicit tmp path
            chroma_guard.assert_store_path_allowed(
                str(Path(td) / "nested" / "store"))             # nested under tmp

    def test_explicitly_configured_disposable_root_permitted(self):
        scratch = PROJECT_ROOT / "reports"          # non-temp, non-production
        os.environ.pop(chroma_guard.DISPOSABLE_ENV, None)
        with self.assertRaises(chroma_guard.ChromaGuardError):
            chroma_guard.assert_store_path_allowed(str(scratch))  # unconfigured:
        os.environ[chroma_guard.DISPOSABLE_ENV] = str(scratch)
        chroma_guard.assert_store_path_allowed(str(scratch))      # allowed now

    def test_unknown_non_disposable_path_rejected(self):
        os.environ.pop(chroma_guard.DISPOSABLE_ENV, None)
        target = PROJECT_ROOT / "knowledge" / "books"
        with self.assertRaises(chroma_guard.ChromaGuardError) as cm:
            chroma_guard.assert_store_path_allowed(str(target))
        # Generic rejection, not the production error: the check is fail-closed.
        self.assertIs(type(cm.exception), chroma_guard.ChromaGuardError)

    def test_ambiguous_disposable_config_fails_closed(self):
        os.environ[chroma_guard.DISPOSABLE_ENV] = "  ;;  "  # entries all blank
        with self.assertRaises(chroma_guard.ChromaGuardConfigError):
            chroma_guard.assert_store_path_allowed(tempfile.gettempdir())
        with self.assertRaises(chroma_guard.ChromaGuardConfigError):
            chroma_guard.assert_store_path_allowed(str(PROD))


class TestConstructionGuards(unittest.TestCase):
    """Bullets 1, 4, 5: rejection strictly precedes any constructor call."""

    def setUp(self):
        chroma_guard.arm()
        self._wrapped = chromadb.PersistentClient
        self.assertTrue(getattr(self._wrapped, "_chroma_guard", False),
                        "conftest/standalone bootstrap must install the wrapper")

    def test_all_known_constructors_are_wrapped(self):
        import chromadb
        for name in ("PersistentClient", "Client", "AdminClient"):
            self.assertTrue(getattr(getattr(chromadb, name), "_chroma_guard", False),
                            f"chromadb.{name} is not guarded")

    def test_direct_persistent_client_rejected_before_construction(self):
        # Prove ordering against a RECORDING constructor: with the guard
        # active, the original constructor must never be reached for a
        # production path, and must be reached for a disposable path.
        fake = types.ModuleType("chromadb_ordering_probe")
        calls = []

        def recording_constructor(path=None, *args, **kwargs):
            calls.append(str(path))
            return "CLIENT-CONSTRUCTED"

        fake.PersistentClient = recording_constructor
        with mock.patch.dict(sys.modules, {"chromadb": fake}):
            n = chroma_guard.install_persistent_client_wrapper()
            self.assertEqual(n, 1)  # PersistentClient wrapped once (idempotent)
            with self.assertRaises(chroma_guard.ProductionStoreAccessError):
                fake.PersistentClient(path=str(PROD))
            self.assertEqual(calls, [],
                             "constructor must never run for a rejected path")
            with self.assertRaises(chroma_guard.ProductionStoreAccessError):
                fake.PersistentClient(path=str(PROD) + os.sep + "subdir")
            self.assertEqual(calls, [],
                             "constructor must never run for a prod-sub path")
            with tempfile.TemporaryDirectory() as td:
                made = fake.PersistentClient(path=td)   # positive control
            self.assertEqual(calls, [td], "exactly one allowed construction")
            self.assertEqual(made, "CLIENT-CONSTRUCTED")
        # Real chromadb module is restored by patch.dict; untouched by this test.

    def test_lazy_query_engine_rejected_before_construction(self):
        sentinel_calls = []
        orig_engine = ba._query_engine
        orig_attr = chromadb.PersistentClient

        def sentinel(*_args, **_kwargs):
            sentinel_calls.append(1)
            raise _SentinelConstructionAttempted("chromadb was invoked")

        ba._query_engine = None
        chromadb.PersistentClient = sentinel
        try:
            with mock.patch.object(ba, "STORAGE_DIR", str(PROD)):
                with self.assertRaises(chroma_guard.ProductionStoreAccessError):
                    ba._get_query_engine()
            self.assertIsNone(ba._query_engine)
        finally:
            chromadb.PersistentClient = orig_attr
            ba._query_engine = orig_engine
        self.assertEqual(sentinel_calls, [])

    def test_import_time_init_cannot_bypass(self):
        with tempfile.TemporaryDirectory() as td:
            mod_path = Path(td) / "import_time_probe.py"
            mod_path.write_text(
                "import chromadb\n"
                f"chromadb.PersistentClient(path={str(PROD)!r})\n",
                encoding="utf-8")
            spec = importlib.util.spec_from_file_location(
                "guard_import_time_probe", mod_path)
            module = importlib.util.module_from_spec(spec)
            with self.assertRaises(chroma_guard.ProductionStoreAccessError):
                spec.loader.exec_module(module)

    def test_rejected_attempts_do_not_change_protected_files(self):
        before = _prod_signature()
        with self.assertRaises(chroma_guard.ProductionStoreAccessError):
            chromadb.PersistentClient(path=str(PROD))
        with mock.patch.object(ba, "STORAGE_DIR", str(PROD)):
            orig = ba._query_engine
            ba._query_engine = None
            try:
                with self.assertRaises(
                        chroma_guard.ProductionStoreAccessError):
                    ba._get_query_engine()
            finally:
                ba._query_engine = orig
        self.assertEqual(before, _prod_signature(),
                         "a rejected attempt must leave production files "
                         "byte- and mtime-identical")


class _FakeEngineResponse:
    source_nodes = ()

    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


class TestMockedEngineContinuity(unittest.TestCase):
    """Bullet 7: suites that mock the seam keep working under the guard."""

    def test_mocked_query_engine_never_touches_guard(self):
        from app.config import NOT_FOUND_PHRASE

        calls = []

        def fake_engine():
            calls.append("engine-resolved")
            return type("E", (), {
                "query": lambda _self, q: _FakeEngineResponse(NOT_FOUND_PHRASE)
            })()

        orig = ba._get_query_engine
        ba._get_query_engine = fake_engine
        try:
            update = ba.books_agent(
                {"question": "anything", "agent_answers": {}})
        finally:
            ba._get_query_engine = orig
        self.assertEqual(calls, ["engine-resolved"])
        self.assertEqual(set(update), {"answer", "sources", "agent_answers"})
        self.assertIn("books", update["sources"])


if __name__ == "__main__":
    unittest.main(verbosity=2)