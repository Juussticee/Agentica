"""Workspace coding capability (2026-10-30).

The smallest end-to-end coding workflow Agentica needs:

  * `workspace` — the STRICT boundary: one configured root, traversal and
    symlink escapes rejected, credential-shaped files denied, redaction of
    secret-looking values before any model sees file text.
  * `patch` — deterministic unified-diff generation, static validation of the
    proposed content (no execution), and hash-checked application.
  * `agent` — the read-only graph node: inspect/explain code, or PREPARE a
    patch (never apply it) and ask for approval through the EXISTING
    approval lifecycle (`operation="workspace_patch"`).

Nothing here runs arbitrary shell commands, and nothing outside the
configured workspace is ever read.
"""

from app.coding.workspace import (
    PathDenied,
    PathOutsideWorkspace,
    Workspace,
    WorkspaceError,
    WorkspaceUnavailable,
)

__all__ = [
    "PathDenied",
    "PathOutsideWorkspace",
    "Workspace",
    "WorkspaceError",
    "WorkspaceUnavailable",
]
