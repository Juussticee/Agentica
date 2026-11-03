"""The `coding` graph node: inspect code, or PREPARE a patch for approval.

Read-only by default. Two shapes, decided deterministically:

  * INSPECT ("explain/why/what is wrong with the code in <file>") — the node
    reads the bounded, redacted files, answers with the model, and cites
    `path:line` evidence taken from `Workspace.search`. No file is touched.
  * PATCH ("fix/patch/change <file>") — the model proposes exact
    find/replace edits inside the SAME files that were shown, this module
    builds the unified diff and statically validates the proposed content,
    and then the EXISTING approval lifecycle persists one
    `workspace_patch` request bound to the patch. Nothing is written here:
    application happens only in `GraphRunner.resume_approval` after the user
    approves, and only if every target file still has the hash the patch was
    prepared against.

Every failure is answered honestly: no workspace configured, a denied path,
a missing file, an unusable model response, or a patch that fails static
validation. A patch is never presented as applied.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from app.coding.patch import (
    OPERATION,
    FileChange,
    PatchError,
    build_patch,
    validate_payload,
)
from app.coding.workspace import (
    MAX_PROMPT_CHARS_PER_FILE,
    MAX_PROMPT_FILES,
    PathDenied,
    Workspace,
    WorkspaceError,
    configured_workspace,
    describe_workspace_problem,
    redact_secrets,
)
from app.state import AgentState, with_agent_output

# Intent: an explicit request to CHANGE code vs a request to explain it.
_PATCH_INTENT_RE = re.compile(
    r"\b(?:fix|patch|change|update|modify|refactor|rename|add|remove|"
    r"replace|correct|repair|implement)\b",
    re.IGNORECASE,
)

# A path-looking token: `app/graph.py`, `frontend/src/App.tsx`.
_PATH_RE = re.compile(r"\b[\w][\w./\\-]*\.([A-Za-z0-9]{1,8})\b")
_TEXT_SUFFIXES = {
    "py", "pyi", "md", "txt", "json", "yml", "yaml", "toml", "cfg", "ini",
    "js", "jsx", "mjs", "cjs", "ts", "tsx", "css", "scss", "html", "htm",
    "sql", "sh", "ps1", "bat", "java", "go", "rs", "rb", "php", "c", "h",
    "cpp", "hpp", "cs",
}

# An identifier to look up in the workspace: `parse_plan`, `validate_plan`.
_SYMBOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\(")

_INSPECT_PROMPT = (
    "You are inspecting source files from a local workspace. Answer the "
    "user's question using ONLY the files below. Name the concrete defect or "
    "behaviour and cite evidence as path:line whenever you refer to code. "
    "If the files do not show what is asked, say so plainly instead of "
    "guessing. You cannot modify files.\n\nUser request: {request}\n\n{files}"
)

_PATCH_PROMPT = (
    "You are preparing a MINIMAL patch for source files from a local "
    "workspace. Return ONLY a JSON object:\n"
    '{{"summary": "one sentence", "changes": [{{"file": "<path exactly as '
    'shown>", "find": "<exact existing text>", "replace": "<new text>", '
    '"reason": "why"}}]}}\n'
    "Rules: use only the files shown; `find` must appear EXACTLY ONCE in "
    "that file; keep changes minimal; do not invent new files; keep the "
    "existing style. Do not include code fences or commentary.\n\n"
    "User request: {request}\n\n{files}"
)


# -- small helpers ------------------------------------------------------------

def _extract_paths(question: str) -> List[str]:
    """Workspace-relative paths named in the request (allowlisted suffixes)."""
    found: List[str] = []
    for match in _PATH_RE.finditer(str(question or "")):
        if match.group(1).lower() not in _TEXT_SUFFIXES:
            continue
        token = match.group(0).replace("\\", "/").lstrip("./")
        if token and token not in found:
            found.append(token)
    return found


def _extract_symbols(question: str) -> List[str]:
    return [m.group(1) for m in _SYMBOL_RE.finditer(str(question or ""))][:3]


def _read_targets(workspace: Workspace, question: str
                  ) -> Tuple[List[str], List[str], List[str]]:
    """Resolve target files + evidence lines. Returns (paths, evidence, notes)."""
    notes: List[str] = []
    paths: List[str] = []
    for candidate in _extract_paths(question):
        try:
            workspace.resolve(candidate)
        except PathDenied as exc:
            notes.append(redact_secrets(str(exc)))
            continue
        except WorkspaceError:
            notes.append(f"not a workspace file: {candidate}")
            continue
        if candidate not in paths:
            paths.append(candidate)

    evidence: List[str] = []
    for symbol in _extract_symbols(question):
        for path, line, text in workspace.search(symbol, limit=6):
            evidence.append(f"{path}:{line}: {text}")
            if path not in paths:
                paths.append(path)
    return paths[:MAX_PROMPT_FILES], evidence, notes


def _format_files(workspace: Workspace, paths: List[str]) -> Tuple[str, Dict[str, str]]:
    blocks: List[str] = []
    raws: Dict[str, str] = {}
    for path in paths:
        try:
            raw, _hash = workspace.read_raw(path)
        except WorkspaceError:
            blocks.append(f"### {path}\n(not readable)")
            continue
        raws[path] = raw
        shown = redact_secrets(raw)[:MAX_PROMPT_CHARS_PER_FILE]
        blocks.append(f"### {path}\n```\n{shown}\n```")
    return "\n\n".join(blocks), raws


def _parse_changes(text: object) -> Optional[dict]:
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", body, re.DOTALL)
        if brace:
            body = brace.group(0)
    try:
        parsed = json.loads(body)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _apply_edits(raws: Dict[str, str], changes: object
                 ) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Apply exact find/replace edits in memory. Returns (replacements, why)."""
    if not isinstance(changes, list) or not changes:
        return [], ["the model proposed no changes"]
    updated = dict(raws)
    touched: List[str] = []
    for item in changes:
        if not isinstance(item, dict):
            return [], ["the model returned a malformed change"]
        path = str(item.get("file") or "").strip()
        find = item.get("find")
        replace = item.get("replace")
        if path not in raws:
            return [], [f"the model tried to change a file that was not "
                        f"shown: {path or '(missing)'}"]
        if not isinstance(find, str) or not find:
            return [], [f"the change for {path} has no `find` text"]
        if not isinstance(replace, str):
            return [], [f"the change for {path} has no `replace` text"]
        content = updated[path]
        if content.count(find) != 1:
            return [], [f"`find` does not appear exactly once in {path}"]
        updated[path] = content.replace(find, replace, 1)
        if path not in touched:
            touched.append(path)
    return [(p, updated[p]) for p in touched], []


def _completion_update(state: AgentState, answer: str, coding: dict) -> dict:
    update = with_agent_output(state, "coding", answer, [])
    # Provenance stays EMPTY on purpose: a workspace file is not an external
    # source, and the answer names every inspected path with line evidence.
    update["coding"] = coding
    return update


def _llm_text(prompt: str) -> str:
    from app.llm_client import get_llm, llm_text

    return str(llm_text(get_llm().invoke(prompt)) or "").strip()


# -- the node -----------------------------------------------------------------

def coding_node(state: AgentState) -> dict:
    """Read-only inspection, or patch preparation + approval request."""
    question = str(state.get("question") or "")
    try:
        workspace = configured_workspace()
    except WorkspaceError as exc:
        return _completion_update(state, describe_workspace_problem(exc),
                                  {"status": "unavailable"})
    if workspace is None:
        return _completion_update(
            state,
            "Workspace coding is not enabled: no coding workspace root is "
            "configured. Ask the operator to set CODING_WORKSPACE_ROOT.",
            {"status": "unavailable"},
        )

    paths, evidence, notes = _read_targets(workspace, question)
    if not paths:
        answer = (
            "I could not find a source file for this request inside the "
            "workspace. Name the file (for example `app/graph.py`) or a "
            "symbol so I can look it up.")
        if notes:
            answer += "\n\n" + "\n".join(f"- {note}" for note in notes)
        return _completion_update(
            state, answer, {"status": "no_target", "workspace": workspace.as_str()})

    files_text, raws = _format_files(workspace, paths)
    wants_patch = bool(_PATCH_INTENT_RE.search(question))

    if not wants_patch:
        return _inspect(state, workspace, question, paths, evidence, files_text)
    return _prepare_patch(state, workspace, question, paths, raws, files_text)


def _inspect(state: AgentState, workspace: Workspace, question: str,
             paths: List[str], evidence: List[str], files_text: str) -> dict:
    coding = {
        "status": "inspected",
        "workspace": workspace.as_str(),
        "files": list(paths),
        "read_only": True,
    }
    prompt = _INSPECT_PROMPT.format(request=question, files=files_text)
    try:
        answer = _llm_text(prompt)
    except Exception as exc:  # noqa: BLE001 — fail-soft, never a fake answer
        answer = (
            "I read the workspace files but could not analyse them "
            f"(the AI provider is unavailable: {exc.__class__.__name__}). "
            "Files inspected: " + ", ".join(paths))
    if not answer:
        answer = ("I read the workspace files but could not produce an "
                  "analysis. Files inspected: " + ", ".join(paths))
    header = "Workspace: " + workspace.as_str() + "\nFiles inspected: " + \
             ", ".join(paths)
    if evidence:
        header += "\nMatched evidence:\n" + "\n".join(
            f"- {line}" for line in evidence)
    return _completion_update(
        state, redact_secrets(header + "\n\n" + answer), coding)


def _prepare_patch(state: AgentState, workspace: Workspace, question: str,
                   paths: List[str], raws: Dict[str, str],
                   files_text: str) -> dict:
    coding: Dict[str, object] = {
        "status": "patch_blocked",
        "workspace": workspace.as_str(),
        "files": list(paths),
    }
    try:
        raw_response = _llm_text(_PATCH_PROMPT.format(
            request=question, files=files_text))
    except Exception as exc:  # noqa: BLE001 — honest, nothing was changed
        return _completion_update(
            state,
            "I did not prepare a patch: the AI provider is unavailable "
            f"({exc.__class__.__name__}). No file was modified.",
            coding)
    proposed = _parse_changes(raw_response)
    if proposed is None:
        return _completion_update(
            state,
            "I did not prepare a patch: the model's response was not a "
            "usable JSON change set. No file was modified.",
            coding)

    replacements, problems = _apply_edits(raws, proposed.get("changes"))
    if problems:
        return _completion_update(
            state,
            "I did not prepare a patch: " + "; ".join(problems)
            + ". No file was modified.",
            coding)

    try:
        patch = build_patch(workspace, question, replacements)
    except (PatchError, WorkspaceError) as exc:
        coding["status"] = "patch_blocked"
        coding["reason"] = redact_secrets(str(exc))
        return _completion_update(
            state,
            "I did not prepare a patch: " + redact_secrets(str(exc))
            + ". No file was modified.",
            coding)

    failed = patch.validation_failed()
    if failed:
        coding["validation"] = [dict(v) for v in patch.validation]
        return _completion_update(
            state,
            "I prepared a change but it FAILED offline validation, so it "
            "was not offered for approval and nothing was written:\n"
            + "\n".join(f"- {v['path']}: {v['detail']}" for v in failed),
            coding)

    summary = str(proposed.get("summary") or "").strip()
    approval_id = _request_approval(patch, question, summary)
    diff = patch.display_diff()
    if approval_id is None:
        coding["status"] = "patch_blocked"
        coding["validation"] = [dict(v) for v in patch.validation]
        return _completion_update(
            state,
            "I prepared this patch but could NOT register an approval "
            "request, so it was NOT applied:\n\n```diff\n" + diff
            + "\n```\n\nNo file was modified.",
            coding)

    coding.update({
        "status": "patch_prepared",
        "approval_id": approval_id,
        "validation": [dict(v) for v in patch.validation],
        "digest": patch.digest,
    })
    answer = (
        ("Prepared a patch — " + summary + "\n" if summary else
         "Prepared a patch.\n")
        + "Files: " + ", ".join(patch.paths)
        + "\nValidation: offline only ("
        + "; ".join(f"{v['path']}: {v['status']}" for v in patch.validation)
        + ") — applying it does not prove the change is correct.\n\n"
        "```diff\n" + diff + "\n```\n\n"
        "I have NOT modified any file. This patch is pending your decision.\n"
        f"Approval request ID: {approval_id}\n"
        "Approve it with POST /api/v1/approvals/" + approval_id +
        "/approve (or /reject), then POST /api/v1/approvals/" + approval_id +
        "/resume to apply it. The approval is bound to this exact patch "
        "and workspace, and cannot apply it if any target file changed.")
    return _completion_update(state, redact_secrets(answer), coding)


def _request_approval(patch, question: str, summary: str) -> Optional[str]:
    """Persist (or reuse) the ONE pending request bound to this exact patch."""
    from app.graph import current_approval_scope

    scope = current_approval_scope()
    if not scope:
        return None
    store, execution_id = scope
    changes: List[FileChange] = list(patch.files)
    paths = ", ".join(patch.paths)
    description = (
        f"Apply a {len(changes)}-file patch to the workspace ({paths}) for: "
        + " ".join(str(question).split())[:160]
    )
    if summary:
        description += f" — {summary[:120]}"
    try:
        payload = patch.to_payload()
        validate_payload(payload)
        created = store.create_or_reuse_approval(
            operation=OPERATION,
            description=description,
            payload=payload,
            execution_id=execution_id,
        )
    except Exception as exc:  # noqa: BLE001 — no permission on failure
        print(f"[coding] could not register the patch approval "
              f"({exc.__class__.__name__}) — nothing is applied")
        return None
    return str(created.get("approval_id") or "") or None
