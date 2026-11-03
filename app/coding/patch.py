"""Deterministic patch preparation, validation and hash-checked application.

The coding capability never edits a file to "see if it works". A change is
represented as a VALUE — for every target file: its workspace-relative path,
the sha256 of the content the patch was prepared against, the proposed new
content, its sha256, and a unified diff for review. That value is what the
approval request binds to (see app/coding/agent.py) and what the resume path
re-verifies before a single byte is written (app/execution/runner.py).

Validation is STATIC and offline: Python content is compiled in memory, JSON
is parsed, and nothing is imported, executed or shelled out. A patch that
fails validation is never offered for approval.
"""

import ast
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.coding.workspace import (
    Workspace,
    WorkspaceError,
    redact_secrets,
)

OPERATION = "workspace_patch"

MAX_PATCH_FILES = 3
MAX_FILE_CHARS = 120_000
MAX_DIFF_CHARS = 20_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PatchError(Exception):
    """The proposed patch is malformed, unsafe or no longer applicable."""


@dataclass(frozen=True)
class FileChange:
    path: str
    before_sha256: str
    after_sha256: str
    after_content: str
    diff: str


@dataclass(frozen=True)
class Patch:
    """A reviewable, not-yet-applied change set."""

    workspace: str
    request: str
    files: Tuple[FileChange, ...] = field(default_factory=tuple)
    validation: Tuple[Dict[str, str], ...] = field(default_factory=tuple)

    @property
    def digest(self) -> str:
        return patch_digest(self.to_payload())

    def to_payload(self) -> Dict[str, object]:
        return {
            "question": self.request,
            "workspace": self.workspace,
            "files": [
                {
                    "path": c.path,
                    "before_sha256": c.before_sha256,
                    "after_sha256": c.after_sha256,
                    "after_content": c.after_content,
                    "diff": c.diff,
                }
                for c in self.files
            ],
            "validation": [dict(v) for v in self.validation],
        }

    def display_diff(self, limit: int = MAX_DIFF_CHARS) -> str:
        return "\n".join(c.diff for c in self.files)[:limit]

    @property
    def paths(self) -> List[str]:
        return [c.path for c in self.files]

    def validation_failed(self) -> List[Dict[str, str]]:
        return [v for v in self.validation if v.get("status") == "failed"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def unified_diff(path: str, before: str, after: str) -> str:
    """A unified diff with `before/`+`after/` prefixes (review only)."""
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
    )
    return "\n".join(lines)


def static_validate(path: str, content: str) -> Dict[str, str]:
    """Offline check of the PROPOSED content. Never executes anything.

    * `.py` / `.pyi` — `compile()` in memory (syntax errors are caught);
    * `.json` — parsed;
    * anything else — reported honestly as not validated rather than
      pretending the change is safe.
    """
    suffix = "." + str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else ""
    try:
        if suffix in (".py", ".pyi"):
            compile(content, str(path), "exec")
            return {"path": str(path), "status": "passed",
                    "detail": "python syntax check passed (compiled in memory)"}
        if suffix == ".json":
            json.loads(content or "null")
            return {"path": str(path), "status": "passed",
                    "detail": "json parsed"}
    except SyntaxError as exc:
        return {"path": str(path), "status": "failed",
                "detail": redact_secrets(
                    f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno})")}
    except ValueError as exc:
        return {"path": str(path), "status": "failed",
                "detail": redact_secrets(f"{exc.__class__.__name__}: {exc}")}
    return {"path": str(path), "status": "not_validated",
            "detail": "no offline check exists for this file type"}


def build_patch(workspace: Workspace, request: str,
                replacements: List[Tuple[str, str]]) -> Patch:
    """Build a Patch from exact (path, new_content) pairs.

    The caller (the coding agent) resolved the NEW content from the model's
    find/replace instructions; here every change is re-read from disk, hashed,
    diffed and statically validated, and a file whose content did not change
    is refused (an "empty" patch is never offered for approval).
    """
    if not replacements:
        raise PatchError("no changes were proposed")
    if len(replacements) > MAX_PATCH_FILES:
        raise PatchError(
            f"a patch may touch at most {MAX_PATCH_FILES} files")
    seen: set = set()
    files: List[FileChange] = []
    validations: List[Dict[str, str]] = []
    for raw_path, new_content in replacements:
        before, before_hash = workspace.read_raw(raw_path)
        if raw_path in seen:
            raise PatchError(f"{raw_path} was changed twice in one patch")
        seen.add(raw_path)
        if len(new_content) > MAX_FILE_CHARS:
            raise PatchError(f"the new content for {raw_path} is too large")
        after = str(new_content)
        if after == before:
            raise PatchError(f"the proposed change to {raw_path} is empty")
        files.append(FileChange(
            path=str(raw_path),
            before_sha256=before_hash,
            after_sha256=sha256_text(after),
            after_content=after,
            diff=unified_diff(str(raw_path), before, after)[:MAX_DIFF_CHARS],
        ))
        validations.append(static_validate(str(raw_path), after))
    return Patch(
        workspace=workspace.as_str(),
        request=str(request or ""),
        files=tuple(files),
        validation=tuple(validations),
    )


def patch_digest(payload: Dict[str, object]) -> str:
    """A stable digest of the reviewed change set (files + content hashes)."""
    files = payload.get("files")
    basis = [
        {"path": f.get("path"),
         "before_sha256": f.get("before_sha256"),
         "after_sha256": f.get("after_sha256")}
        for f in (files if isinstance(files, list) else [])
        if isinstance(f, dict)
    ]
    return hashlib.sha256(
        json.dumps(basis, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_payload(payload: object) -> List[FileChange]:
    """Structurally validate a persisted patch payload (fail closed).

    Used by the store before a request is persisted and again by the resume
    path before anything is written: a client can never inject a file list.
    """
    if not isinstance(payload, dict):
        raise PatchError("the patch payload is not an object")
    workspace = payload.get("workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        raise PatchError("the patch payload has no workspace")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise PatchError("the patch payload has no files")
    if len(files) > MAX_PATCH_FILES:
        raise PatchError("the patch payload touches too many files")
    changes: List[FileChange] = []
    for item in files:
        if not isinstance(item, dict):
            raise PatchError("a patch file entry is not an object")
        path = item.get("path")
        after = item.get("after_content")
        before_hash = str(item.get("before_sha256") or "")
        after_hash = str(item.get("after_sha256") or "")
        if not isinstance(path, str) or not path.strip():
            raise PatchError("a patch file entry has no path")
        if not isinstance(after, str) or not after:
            raise PatchError(f"the proposed content for {path} is missing")
        if len(after) > MAX_FILE_CHARS:
            raise PatchError(f"the proposed content for {path} is too large")
        if not _SHA256_RE.match(before_hash) or not _SHA256_RE.match(after_hash):
            raise PatchError(f"invalid content hash for {path}")
        if sha256_text(after) != after_hash:
            raise PatchError(f"the proposed content for {path} does not "
                             "match its recorded hash")
        diff = item.get("diff")
        changes.append(FileChange(
            path=path.strip(),
            before_sha256=before_hash,
            after_sha256=after_hash,
            after_content=after,
            diff=str(diff or "")[:MAX_DIFF_CHARS],
        ))
    return changes


def stale_files(workspace: Workspace,
                changes: List[FileChange]) -> List[str]:
    """Files whose current content no longer matches the reviewed patch."""
    stale: List[str] = []
    for change in changes:
        try:
            _raw, current = workspace.read_raw(change.path)
        except (WorkspaceError, OSError):
            stale.append(change.path)
            continue
        if current != change.before_sha256:
            stale.append(change.path)
    return stale


@dataclass
class ApplyResult:
    applied: List[str]
    failed: List[Dict[str, str]]
    errors: List[str]

    @property
    def ok(self) -> bool:
        return not self.failed and not self.errors


def apply_patch(workspace: Workspace, changes: List[FileChange],
                payload: Optional[Dict[str, object]] = None) -> ApplyResult:
    """Write the reviewed content, refusing the WHOLE patch if it went stale.

    Re-checks the workspace identity, every file's pre-change hash and (when
    present) the patch digest before the first write, then verifies each
    file's post-change hash. A write that fails midway is reported honestly:
    what was written stays written (never silently reverted) and is named.
    """
    if payload is not None:
        expected_workspace = str(payload.get("workspace") or "")
        if expected_workspace and expected_workspace != workspace.as_str():
            raise PatchError(
                "the configured workspace differs from the workspace the "
                "patch was prepared in")
        expected_digest = str(payload.get("digest") or "")
        if expected_digest and expected_digest != patch_digest(payload):
            raise PatchError(
                "the persisted patch no longer matches its own digest")
    stale = stale_files(workspace, changes)
    if stale:
        raise PatchError(
            "these files changed since the patch was prepared: "
            + ", ".join(stale))

    applied: List[str] = []
    failed: List[Dict[str, str]] = []
    for change in changes:
        try:
            new_hash = workspace.replace_text(
                change.path, change.before_sha256, change.after_content)
        except Exception as exc:  # noqa: BLE001 — reported, never hidden
            failed.append({"path": change.path,
                           "reason": redact_secrets(str(exc))
                           or exc.__class__.__name__})
            continue
        if new_hash != change.after_sha256:
            failed.append({
                "path": change.path,
                "reason": "the file was written but its content hash does "
                          "not match the reviewed patch",
            })
            continue
        applied.append(change.path)
    return ApplyResult(applied=applied, failed=failed, errors=[])
