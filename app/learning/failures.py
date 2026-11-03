"""Execution-failure learning (2026-10-30).

A failed run is NOT a lesson by itself. Turning every failure into a memory row
would accumulate noise and teach nothing, so this module records ONE candidate
lesson only when the SAME failure signature has repeated:

    signature = route + error category + the QA findings (normalised ids)

The second occurrence is the evidence that a behaviour, not bad luck, is
involved. Everything here is deterministic: no model call, no invented root
cause. The corrective behaviour reads as a procedure to investigate, never as a
fabricated fix, and the lesson is always `candidate` — it cannot apply to a
prompt until the user reviews it (`lessons.decide` applies the same rule).

Nothing in this module can change a permission, approval rule, credential or
configuration: it only inserts/merges a lesson row.
"""

import re
import uuid
from typing import Any, Dict, Iterable, List, Optional

from app.learning import lessons as L

MIN_OCCURRENCES = 2
MAX_SCAN = 300
MAX_FINDINGS = 5

_UUID_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\b\d+\b")
# Plan/task/step labels carry a counter that changes between runs ("t1" vs
# "t12"); the finding is the same, so the label is normalised away.
_LABEL_RE = re.compile(r"\b(?:t|task|step|exec(?:ution)?)[-_ ]?\d+\b",
                       re.IGNORECASE)


def _flag_text(flag: Any) -> str:
    """QA text without ids/counts, so the same finding matches next time."""
    text = _LABEL_RE.sub("<id>", str(flag or ""))
    text = _DIGITS_RE.sub("#", _UUID_RE.sub("<id>", text))
    return re.sub(r"\s+", " ", text).strip().lower()[:180]


def signature(route: Optional[str], error: Optional[dict],
              qa_flags: Optional[Iterable[Any]] = None) -> str:
    """Stable identity of one kind of failure ('' when there is none)."""
    category = str((error or {}).get("category") or "").strip().lower()
    flags = sorted({_flag_text(f) for f in (qa_flags or []) if _flag_text(f)})
    if not category and not flags:
        return ""
    parts = [str(route or "unknown"), category or "qa_findings"]
    parts.extend(flags[:MAX_FINDINGS])
    return "|".join(parts)


def _row_signature(row: Dict[str, Any]) -> str:
    return signature(row.get("route"), row.get("error"), row.get("qa_flags"))


def occurrences(store, key: str, *, exclude_id: Optional[str] = None) -> int:
    """How many recorded executions already show this signature."""
    try:
        rows = store.list_executions()
    except Exception:  # noqa: BLE001 — an unreadable store means no evidence
        return 0
    count = 0
    for row in rows[:MAX_SCAN]:
        if exclude_id and row.get("id") == exclude_id:
            continue
        if _row_signature(row) == key:
            count += 1
    return count


def _existing(store, capability: str, key: str) -> Optional[Dict[str, Any]]:
    """An open lesson already tracking this failure signature."""
    try:
        rows = store.list_lessons(capability=capability)
    except Exception:  # noqa: BLE001
        return None
    for row in rows:
        if row.get("status") in (L.REJECTED, L.DISABLED, L.SUPERSEDED):
            continue
        for item in row.get("evidence") or []:
            if isinstance(item, dict) and item.get("signature") == key:
                return row
    return None


def observe(store, *, execution_id: str, question: str, route: Optional[str],
            error: Optional[dict] = None,
            qa_flags: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """Record or merge a candidate lesson when a failure repeats.

    Returns a small dict that says exactly what happened — including
    `recorded: False` with the reason when there is nothing to learn yet.
    """
    from app.learning import writes_allowed

    if not writes_allowed():
        return {"recorded": False, "signature": "", "occurrences": 0,
                "reason": "learning writes are disabled for this run "
                          "(evaluation)"}

    flags = [str(f) for f in (qa_flags or []) if str(f).strip()]
    key = signature(route, error, flags)
    if not key:
        return {"recorded": False, "signature": "",
                "reason": "the run finished without a failure or QA finding"}

    capability = str(route or "any")
    if capability not in L.CAPABILITIES:
        capability = "any"
    seen = occurrences(store, key, exclude_id=execution_id) + 1
    if seen < MIN_OCCURRENCES:
        return {"recorded": False, "signature": key, "occurrences": seen,
                "reason": (f"first occurrence ({seen} of {MIN_OCCURRENCES}); a "
                           "lesson needs a repeated failure, not a single one")}

    category = str((error or {}).get("category") or "qa_findings")
    problem = (f"Repeated {category} failure on the {route or 'unknown'} route "
               f"({seen} occurrences)")
    cause = str((error or {}).get("why") or "unknown")[:300]
    behavior = (
        f"Before re-running the {route or 'unknown'} route, investigate the "
        f"recurring {category} failure ({seen} occurrences) and fix its cause; "
        "do not present a retry as progress."
    )
    evidence = {
        "kind": "execution",
        "execution_id": execution_id,
        "signature": key,
        "category": category,
        "flags": flags[:MAX_FINDINGS],
        "question": question[:200],
    }

    existing = _existing(store, capability, key)
    if existing is not None:
        merged = list(existing.get("evidence") or []) + [evidence]
        updated = store.update_lesson(
            existing["lesson_id"],
            evidence=merged,
            confidence=min(0.6, float(existing.get("confidence") or 0.0) + 0.05),
        )
        return {"recorded": True, "action": "merged", "signature": key,
                "occurrences": seen, "lesson": updated}

    lesson_id = f"lsn_{uuid.uuid4().hex[:12]}"
    row = {
        "lesson_id": lesson_id,
        "status": L.CANDIDATE,
        "category": "experience",
        "capability": capability,
        "scope": "capability",
        "problem": problem[:300],
        "root_cause": cause,
        "corrective_behavior": behavior[:400],
        "validation": ("A later run of this route ends completed with no QA "
                       "findings and no repeat of this signature."),
        "confidence": 0.4,
        "evidence": [evidence],
        "source_execution_id": execution_id,
        "source_text": None,
        "origin": L.ORIGIN_FAILURE,
    }
    stored = store.create_lesson(row)
    return {"recorded": True, "action": "created", "signature": key,
            "occurrences": seen, "lesson": stored}
