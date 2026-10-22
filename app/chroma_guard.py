"""Fail-closed production-store isolation for automated tests and eval scripts.

Context: the RAG-1 closeout regression (commit
``df3e68347bc062287dccdea386f5e6c018254976``) passed all suites while
``storage/chroma.sqlite3`` changed bytes unexpectedly. This module makes any
test/eval access to the production store FAIL *before* a client is opened.

Activation (fail closed; there is deliberately NO disable switch):

* ``scripts/conftest.py`` arms it for every pytest run and wraps the
  ``chromadb`` constructors before any test module is imported;
* documented standalone test scripts call :func:`arm` themselves at startup;
* ``AGENTICA_CHROMA_GUARD`` set to ANY value — including the empty string —
  arms it (an ambiguous value arms rather than bypasses);
* with no configuration at all it falls back to the ``PYTEST_CURRENT_TEST``
  marker (set only while pytest executes a test).

While armed, every store path is checked BEFORE any client construction:

* the path is normalized with ``realpath`` + ``normcase``, so relative,
  absolute, ``..``-mixed, symlinked and case-different spellings compare equal;
* anything equal to or inside the production ``storage/`` directory raises
  :class:`ProductionStoreAccessError`;
* every other path must sit inside an allowed disposable root — the system
  temp directory (implicitly allowed) plus any roots listed in
  ``AGENTICA_TEST_STORAGE_DIRS`` (``os.pathsep``-separated, e.g.
  ``C:\\Temp;D:\\scratch``); a set-but-empty value raises
  :class:`ChromaGuardConfigError` before ANY path is allowed;
* any other path raises :class:`ChromaGuardError` (fail closed).

When NOT armed (normal production operation: app start, ingest CLI, manual
diagnostics), every check is a no-op — production behavior is unchanged.

What it does NOT guarantee: coverage ends at client constructions flowing
through ``chromadb.PersistentClient`` / ``chromadb.Client`` /
``chromadb.AdminClient`` plus the explicit checks at this repository's known
construction sites. It does not stop raw sqlite file writes, protects only
invocations that are pytest-based or explicitly armed, and it does NOT
establish the root cause of the earlier unexpected store mutation.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

GUARD_ENV = "AGENTICA_CHROMA_GUARD"          # any value (even "") arms the guard
DISPOSABLE_ENV = "AGENTICA_TEST_STORAGE_DIRS"  # os.pathsep-separated allowed roots

# Derived independently of app.config on purpose: a test that patches
# app.config.STORAGE_DIR must not be able to redirect what is protected.
# app.config.STORAGE_DIR is PROJECT_ROOT/"storage" and is not env-overridable.
PRODUCTION_STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"


class ChromaGuardError(RuntimeError):
    """Base: a store access was rejected by the fail-closed guard."""


class ProductionStoreAccessError(ChromaGuardError):
    """A path at or inside the production ``storage/`` directory was rejected."""


class ChromaGuardConfigError(ChromaGuardError):
    """Guard configuration is set but unusable -> reject everything (fail closed)."""


def arm() -> None:
    """Arm the guard process-wide (environment-based; inherited by subprocesses)."""
    os.environ[GUARD_ENV] = "1"


def is_armed() -> bool:
    if GUARD_ENV in os.environ:
        # Set-but-empty / garbage values arm the guard: ambiguous config can
        # only fail closed, never open a bypass.
        return True
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _normalize(path) -> Path:
    """Absolute, symlink-resolved, case-normalized form for equality checks."""
    if path is None:
        # chromadb.PersistentClient's own default path.
        path = "./chroma"
    real = os.path.realpath(os.path.expanduser(str(path)))
    return Path(os.path.normcase(real))


def _configured_disposable_roots() -> "list[Path]":
    raw = os.environ.get(DISPOSABLE_ENV)
    if raw is None:
        return []
    entries = [e for e in raw.split(os.pathsep) if e.strip()]
    if not entries:
        raise ChromaGuardConfigError(
            f"{DISPOSABLE_ENV} is set but contains no usable path entries "
            f"({raw!r}); refusing ALL store access. Unset the variable or "
            f"list disposable test-store roots separated by {os.pathsep!r}."
        )
    return [_normalize(e) for e in entries]


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def assert_store_path_allowed(path) -> None:
    """Reject production/unknown store paths before any client may open.

    No-op when the guard is not armed. Raises ChromaGuardError subclasses
    (fail closed) otherwise.
    """
    if not is_armed():
        return
    # Configuration is validated first: ambiguous config must not allow even
    # an apparently-safe path through.
    _configured_disposable_roots()
    norm = _normalize(path)
    prod = _normalize(PRODUCTION_STORAGE_DIR)
    if _inside(norm, prod):
        raise ProductionStoreAccessError(
            f"refusing to open the production Chroma store from a "
            f"test/eval context: {path!r} resolves to {norm} (production "
            f"root {prod}). Use a disposable path under the system temp "
            f"directory, or add an explicit root via {DISPOSABLE_ENV}."
        )
    temp_root = _normalize(tempfile.gettempdir())
    if _inside(norm, temp_root):
        return
    for root in _configured_disposable_roots():
        if _inside(norm, root):
            return
    raise ChromaGuardError(
        f"refusing store access to {norm}: it is neither the production "
        f"store (always blocked), the system temp directory, nor a "
        f"disposable root configured via {DISPOSABLE_ENV}. Point the test "
        f"at a temp/allowed path (fail-closed default)."
    )


def path_from_call(args, kwargs):
    """Best-effort store-path extraction from a chromadb constructor call."""
    if "path" in kwargs:
        return kwargs["path"]
    if args and isinstance(args[0], (str, os.PathLike)):
        return os.fspath(args[0])
    settings = kwargs.get("settings") if "settings" in kwargs else (
        args[0] if args else None)
    persist = getattr(settings, "persist_directory", None)
    return persist if persist is not None else "./chroma"


def install_persistent_client_wrapper() -> int:
    """Wrap every chromadb constructor so calls are checked before construction.

    Idempotent (already-wrapped constructors are skipped). Returns the number
    of constructors wrapped.
    """
    if "chromadb" not in sys.modules and \
            importlib.util.find_spec("chromadb") is None:
        # chromadb is not installed in this interpreter: no persistent client
        # can be constructed here, so there is nothing to guard. This keeps
        # scripts that never touch Chroma importable under any Python.
        # (A module already in sys.modules — real or a test double — skips
        # find_spec, which refuses modules without a __spec__.)
        return 0
    import chromadb

    wrapped = 0
    for name in ("PersistentClient", "Client", "AdminClient"):
        original = getattr(chromadb, name, None)
        if original is None or getattr(original, "_chroma_guard", False):
            continue

        def guarded(*call_args, __original=original, **call_kwargs):
            assert_store_path_allowed(path_from_call(call_args, call_kwargs))
            return __original(*call_args, **call_kwargs)

        guarded._chroma_guard = True
        guarded._chroma_guard_original = original
        guarded.__name__ = name
        setattr(chromadb, name, guarded)
        wrapped += 1
    return wrapped