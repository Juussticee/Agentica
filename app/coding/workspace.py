"""The workspace boundary for the coding capability (2026-10-30).

One root, chosen by configuration (`CODING_WORKSPACE_ROOT`), and NOTHING
outside it. Every path the capability touches goes through `Workspace.resolve`
so the boundary is enforced in ONE place:

  * the root is `resolve()`d once, so the check is on real paths;
  * absolute paths, `..` segments, NUL bytes and empty input are refused;
  * symlinks are followed by `resolve()` and then checked — a link that
    points outside the root is refused exactly like a `..` escape;
  * credential-shaped paths (`.env*`, key material, `id_rsa*`, secrets/
    credentials files, databases) are DENIED even inside the root, and so are
    dependency/build/VCS directories;
  * only a small allowlist of text source extensions can be read at all, and
    every read is size-capped.

`redact_secrets` is defence in depth: a credential pasted into a source file
must not reach a model prompt or an error message even if the file itself is
legitimately readable.

Deliberately NOT here: recursive filesystem walks of the home directory, any
shell, any write that is not a reviewed patch application. The workflow is
read-only by default.
"""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Source/text extensions the capability will read. Anything else (binaries,
# archives, images, notebooks) is refused rather than guessed at.
TEXT_EXTENSIONS = frozenset({
    ".py", ".pyi", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg",
    ".ini", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".css", ".scss",
    ".html", ".htm", ".sql", ".sh", ".ps1", ".bat", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs",
})

# Directories never entered, even if they are inside the root: they hold
# installed dependencies, build output, local databases or VCS internals.
DENIED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "storage", "logs", ".idea", ".vscode",
})

# Credential-shaped FILE names. Deliberately name-based only: a project
# source file called `tokenizer.py` is legitimate, so nothing is denied for
# merely containing "token".
_DENIED_NAME_RE = re.compile(
    r"^(?:\.env(?:\..*)?|\.npmrc|\.netrc|\.pgpass|"
    r"id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?|"
    r"credentials?(?:\..*)?|secrets?(?:\..*)?|"
    r".*\.(?:pem|key|pfx|p12|keystore|jks|sqlite3?|db))$",
    re.IGNORECASE,
)

_SECRET_PATTERNS = (
    # Quoted values MAY contain spaces (passphrases) — mask everything up to
    # the closing quote after a credential-shaped keyword.
    re.compile(
        r"\b((?:api[_-]?key|apikey|secret|client[_-]?secret|password|passwd|"
        r"access[_-]?token|refresh[_-]?token|auth[_-]?token|"
        r"private[_-]?key)\b\s*[:=]\s*)(['\"])([^'\"\n]{6,})\2",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)\b((?:api[_-]?key|apikey|secret|client[_-]?secret|password|"
        r"passwd|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
        r"private[_-]?key)\b\s*[:=]\s*)(['\"]?)([^\s'\"]{6,})\2"
    ),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
        r"-----END [A-Z ]*PRIVATE KEY-----"
    ),
)
_REDACTED = "***REDACTED***"

# Hard bounds. A patch workflow never needs a whole repository in one prompt.
MAX_FILE_BYTES = 400_000          # per file, before reading
MAX_PROMPT_FILES = 3              # files shown to the model per request
MAX_PROMPT_CHARS_PER_FILE = 12_000
MAX_SEARCH_FILES = 400            # files scanned by search_symbol
MAX_SEARCH_HITS = 20


class WorkspaceError(Exception):
    """Base class for every refused workspace operation."""


class WorkspaceUnavailable(WorkspaceError):
    """No workspace is configured — the capability is disabled."""


class PathOutsideWorkspace(WorkspaceError):
    """The requested path escapes the configured root."""


class PathDenied(WorkspaceError):
    """The path exists inside the root but is never exposed."""


def redact_secrets(text: str) -> str:
    """Mask credential-shaped values before text leaves this module.

    Defence in depth only: sensitive FILES are already denied. A key pasted
    into a readable source file must still not reach a prompt, a log line or
    an error message.
    """
    out = str(text or "")
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            out = pattern.sub(
                lambda m: m.group(1) + m.group(2) + _REDACTED + m.group(2),
                out,
            )
        else:
            out = pattern.sub(_REDACTED, out)
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkspaceFile:
    """One bounded, redacted view of a source file."""

    path: str            # workspace-relative, POSIX separators
    content: str         # redacted text (never bytes)
    sha256: str          # hash of the RAW text on disk
    truncated: bool


class Workspace:
    """A strict view of exactly one configured root directory."""

    def __init__(self, root: Path | str):
        if not str(root or "").strip():
            raise WorkspaceUnavailable(
                "no coding workspace is configured (CODING_WORKSPACE_ROOT)")
        candidate = Path(root).expanduser()
        if not candidate.is_absolute():
            raise WorkspaceUnavailable(
                "the coding workspace root must be an absolute path")
        resolved = candidate.resolve()
        if not resolved.is_dir():
            raise WorkspaceUnavailable(
                f"the coding workspace root is not a directory: {resolved}")
        self.root = resolved

    # -- identity ---------------------------------------------------------

    def as_str(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:                       # never leaks content
        return f"Workspace({self.root})"

    # -- path boundary ----------------------------------------------------

    def resolve(self, relative: object) -> Path:
        """Map a workspace-relative path to an absolute path INSIDE the root.

        Raises PathOutsideWorkspace / PathDenied / WorkspaceError. Never
        touches the filesystem outside the root.
        """
        raw = str(relative if relative is not None else "").strip()
        if not raw:
            raise WorkspaceError("a workspace path is required")
        if "\x00" in raw:
            raise WorkspaceError("workspace paths cannot contain NUL bytes")
        candidate = Path(raw)
        if candidate.is_absolute() or candidate.drive:
            raise PathOutsideWorkspace(
                "workspace paths must be relative to the configured root")
        parts = [p for p in candidate.parts if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise PathOutsideWorkspace(
                "workspace paths cannot contain '..' segments")
        if not parts:
            raise WorkspaceError("a workspace path is required")

        resolved = (self.root / Path(*parts)).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            # Includes the symlink case: resolve() followed the link, and the
            # real target is outside the root.
            raise PathOutsideWorkspace(
                f"path escapes the workspace: {raw}")
        relative_parts = resolved.relative_to(self.root).parts
        for part in relative_parts[:-1]:
            if part.lower() in DENIED_DIRS:
                raise PathDenied(
                    f"directory is outside the coding boundary: {part}")
        name = relative_parts[-1]
        if name.lower() in DENIED_DIRS or _DENIED_NAME_RE.match(name):
            raise PathDenied(f"file type is not readable here: {name}")
        return resolved

    @staticmethod
    def relative_name(path: Path, root: Path) -> str:
        return path.relative_to(root).as_posix()

    def is_text_path(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTENSIONS

    # -- reads ------------------------------------------------------------

    def read(self, relative: object) -> WorkspaceFile:
        """Read one allowed text file; redacted content + raw-content hash."""
        path = self.resolve(relative)
        if not path.is_file():
            raise WorkspaceError(f"not a readable file: {relative}")
        if not self.is_text_path(path):
            raise PathDenied(
                f"not a text source file: {path.suffix or '(none)'}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise WorkspaceError(
                f"file is too large to inspect ({size} bytes > "
                f"{MAX_FILE_BYTES})")
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise WorkspaceError(f"could not read {relative}: {exc}") from exc
        return WorkspaceFile(
            path=self.relative_name(path, self.root),
            content=redact_secrets(raw),
            sha256=_sha256_text(raw),
            truncated=False,
        )

    def read_raw(self, relative: object) -> Tuple[str, str]:
        """Raw (unredacted) text + its hash — used ONLY to build a patch.

        The raw text is hashed and diffed locally; it is never placed in a
        prompt, an event payload or an answer.
        """
        path = self.resolve(relative)
        if not path.is_file():
            raise WorkspaceError(f"not a readable file: {relative}")
        if not self.is_text_path(path):
            raise PathDenied("not a text source file")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise WorkspaceError("file is too large to patch")
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise WorkspaceError(f"could not read {relative}: {exc}") from exc
        return raw, _sha256_text(raw)

    def search(
        self, needle: str, limit: int = MAX_SEARCH_HITS
    ) -> List[Tuple[str, int, str]]:
        """Literal symbol search inside the boundary (read-only, bounded).

        Returns (relative path, 1-based line, redacted line text). Skips
        denied directories and non-text files; gives up after
        MAX_SEARCH_FILES files so one request can never walk a large tree.
        """
        term = str(needle or "").strip()
        if len(term) < 2:
            return []
        hits: List[Tuple[str, int, str]] = []
        scanned = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                d for d in dirnames if d.lower() not in DENIED_DIRS
            )
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.is_symlink():
                    continue
                if (not self.is_text_path(path)
                        or _DENIED_NAME_RE.match(filename)):
                    continue
                scanned += 1
                if scanned > MAX_SEARCH_FILES:
                    return hits
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                    lines = path.read_text(
                        encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                for number, line in enumerate(lines, start=1):
                    if term in line:
                        hits.append((
                            self.relative_name(path, self.root),
                            number,
                            redact_secrets(line.strip())[:240],
                        ))
                        if len(hits) >= limit:
                            return hits
        return hits

    # -- the ONLY write path ----------------------------------------------

    def replace_text(self, relative: object, expected_sha256: str,
                     content: str) -> str:
        """Atomically replace one file, but ONLY if it still has the hash the
        reviewed patch was prepared against. Returns the new content hash."""
        path = self.resolve(relative)
        if not path.is_file():
            raise WorkspaceError(f"not a readable file: {relative}")
        if not self.is_text_path(path):
            raise PathDenied("not a text source file")
        current = _sha256_text(
            path.read_text(encoding="utf-8", errors="replace"))
        if current != str(expected_sha256 or ""):
            raise WorkspaceError(
                "the file changed since the patch was prepared")
        directory = path.parent
        handle, temp_name = tempfile.mkstemp(
            dir=str(directory), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as fh:
                fh.write(content)
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return _sha256_text(
            path.read_text(encoding="utf-8", errors="replace"))


def configured_workspace() -> Optional[Workspace]:
    """The workspace from configuration, or None when it is not enabled.

    Read at CALL TIME (like the other feature switches) so tests can point it
    at a temporary directory without reloading the module.
    """
    from app.config import CODING_WORKSPACE_ROOT

    if not str(CODING_WORKSPACE_ROOT or "").strip():
        return None
    return Workspace(CODING_WORKSPACE_ROOT)


def describe_workspace_problem(exc: Exception) -> str:
    """One honest sentence for a refused/unavailable workspace operation."""
    if isinstance(exc, WorkspaceUnavailable):
        return (
            "Workspace coding is not enabled: no coding workspace root is "
            "configured. Ask the operator to set CODING_WORKSPACE_ROOT."
        )
    return redact_secrets(str(exc)) or exc.__class__.__name__


def hash_text(text: str) -> str:
    return _sha256_text(text)


def iter_extensions() -> Iterable[str]:
    return sorted(TEXT_EXTENSIONS)
