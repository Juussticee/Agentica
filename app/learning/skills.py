"""Self-improving skills: versioned procedures with deterministic gates.

A skill here is a REUSABLE PROCEDURE (how to do a kind of task), which is not
an agent (who does the work) and not a tool (an operation). The documents in
`app.agents.registry.SKILLS` stay the floor; a learned version is a candidate
derived from ACTIVE lessons for one capability.

SAFETY RULES BAKED INTO THE CODE
  * a candidate is created as `candidate` and is never applied in that state;
  * activation requires a PASSED deterministic evaluation — the version's own
    text must contain the required sections, carry its source lessons, and
    contain no privilege/credential/security directive;
  * the previous active version is retired (kept, inspectable), never
    overwritten, so rollback is exact;
  * one active version per skill, enforced by `activate`.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from app.execution.store import utc_now_iso

from app.learning import lessons as L

REQUIRED_SECTIONS = ("Scope", "Steps", "Validation")
MAX_INSTRUCTIONS = 5000
MIN_INSTRUCTIONS = 120

CAPABILITY_SKILLS: Dict[str, str] = {
    "research": "research-comparison",
    "books": "document-grounding",
    "data": "dataset-analysis",
    "coding": "workspace-coding",
}

_SKILL_PURPOSE: Dict[str, str] = {
    "research-comparison": "Compare options using cited evidence and explicit "
                           "trade-offs.",
    "document-grounding": "Answer from the indexed corpus with citations, and "
                          "refuse when the corpus does not support it.",
    "dataset-analysis": "Compute over an actual dataset and report what was "
                        "computed, including limitations.",
    "workspace-coding": "Inspect, patch and validate workspace code behind the "
                        "approval gate.",
}

_FORBIDDEN_RE = re.compile(
    r"\b(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions"
    r"|disable\s+(?:the\s+)?approval|bypass\s+(?:the\s+)?(?:approval|security)"
    r"|grant\s+(?:full|unrestricted)\s+access|rm\s+-rf|sudo\b"
    r"|(?:print|expose|reveal|log)\s+(?:the\s+)?(?:api[_\s-]?key|token|"
    r"secret|password)"
    r"|(?:read|upload|send)\s+(?:the\s+)?(?:\.env|credentials?|private\s+key))\b",
    re.IGNORECASE,
)


def skill_for(capability: str) -> Optional[str]:
    return CAPABILITY_SKILLS.get(str(capability or ""))


def build_instructions(skill_id: str, capability: str,
                       lessons: List[Dict[str, Any]]) -> str:
    """Compose the procedure text from the source lessons (deterministic)."""
    purpose = _SKILL_PURPOSE.get(skill_id,
                                 f"Handle {capability or 'this kind of'} tasks.")
    steps = []
    for index, lesson in enumerate(lessons, 1):
        behavior = str(lesson.get("corrective_behavior") or "").strip()
        if not behavior:
            continue
        steps.append(f"{index}. {behavior} "
                     f"(learned from {lesson.get('origin')} "
                     f"#{lesson.get('lesson_id')})")
    if not steps:
        steps.append("1. Answer the request using the capability's existing "
                     "procedure.")
    checks = [f"- {str(l.get('validation')).strip()}"
              for l in lessons if str(l.get("validation") or "").strip()]
    if not checks:
        checks = ["- The answer addresses every requirement stated in the "
                  "request."]
    return (
        f"# Skill: {skill_id}\n"
        f"Scope: {capability or 'general'} tasks — {purpose}\n\n"
        f"Steps:\n" + "\n".join(steps) + "\n\n"
        "Validation:\n" + "\n".join(checks) + "\n"
    )


def evaluate(instructions: str,
             source_lessons: List[Any]) -> Dict[str, Any]:
    """Deterministic gate for a candidate version. No model involved."""
    text = str(instructions or "")
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    missing = [s for s in REQUIRED_SECTIONS
               if not re.search(rf"^{s}\s*:", text, re.IGNORECASE | re.MULTILINE)]
    add("required_sections", not missing,
        "all sections present" if not missing else f"missing: {missing}")
    add("length", MIN_INSTRUCTIONS <= len(text) <= MAX_INSTRUCTIONS,
        f"{len(text)} chars (allowed {MIN_INSTRUCTIONS}-{MAX_INSTRUCTIONS})")
    sourced = "learned from" in text.lower()
    add("carries_lessons", sourced and bool(source_lessons),
        f"{len(source_lessons)} source lesson(s) referenced"
        if sourced else "no lesson provenance in the text")
    unsafe = _FORBIDDEN_RE.search(text)
    add("no_privilege_directive", unsafe is None,
        "no permission/credential directive"
        if unsafe is None else f"forbidden instruction: {unsafe.group(0)!r}")
    reproducible = all(
        str(lesson.get("corrective_behavior") or "").strip() in text
        for lesson in source_lessons if isinstance(lesson, dict)
    ) if source_lessons else False
    add("lesson_coverage", reproducible,
        "every source lesson's behaviour appears verbatim" if reproducible
        else "a source behaviour is not reflected in the procedure")
    passed = all(c["ok"] for c in checks)
    return {"evaluated_at": utc_now_iso(), "passed": passed, "checks": checks}


def propose(store, skill_id: str, capability: str,
            lessons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a candidate version from ACTIVE lessons. Never activates."""
    existing = store.list_skill_versions(skill_id)
    version = max([int(v.get("version") or 0) for v in existing] or [0]) + 1
    instructions = build_instructions(skill_id, capability, lessons)
    evaluation = evaluate(instructions, lessons)
    row = store.create_skill_version({
        "version_id": f"skv_{uuid.uuid4().hex[:12]}",
        "skill_id": skill_id,
        "version": version,
        "status": "candidate",
        "instructions": instructions,
        "validation": "deterministic gate: sections, length, provenance, "
                      "no privilege directive, lesson coverage",
        "source_lesson_ids": [l.get("lesson_id") for l in lessons],
        "evaluation": evaluation,
    })
    return row


def activate(store, version_id: str) -> Dict[str, Any]:
    """Activate a candidate version IFF its evaluation passed.

    Re-evaluates from the STORED text and the stored source lessons, so a
    version cannot be activated by editing its evaluation blob.
    """
    version = store.get_skill_version(version_id)
    if version is None:
        raise KeyError(version_id)
    sources = [store.get_lesson(lid)
               for lid in version.get("source_lesson_ids") or []]
    sources = [s for s in sources if isinstance(s, dict)]
    evaluation = evaluate(version.get("instructions"), sources)
    if not evaluation["passed"]:
        store.set_skill_version_status(version_id, "rejected",
                                       evaluation=evaluation)
        return {"activated": False, "version": store.get_skill_version(version_id),
                "evaluation": evaluation,
                "reason": "deterministic evaluation did not pass"}
    # One active version per skill: retire the current one first.
    for other in store.list_skill_versions(version["skill_id"]):
        if other["version_id"] != version_id and other.get("status") == "active":
            store.set_skill_version_status(other["version_id"], "retired")
    activated = store.set_skill_version_status(version_id, "active",
                                              evaluation=evaluation)
    return {"activated": True, "version": activated,
            "evaluation": evaluation, "reason": "evaluation passed"}


def rollback(store, skill_id: str) -> Dict[str, Any]:
    """Re-activate the newest non-active version (previous behaviour)."""
    versions = store.list_skill_versions(skill_id)
    current = next((v for v in versions if v.get("status") == "active"), None)
    candidates = [v for v in versions
                  if v.get("status") in ("retired", "validated")
                  and (current is None
                       or int(v.get("version") or 0) < int(current["version"]))]
    if not candidates:
        return {"rolled_back": False, "reason": "no earlier version to restore"}
    target = candidates[0]  # list is version DESC
    if current is not None:
        store.set_skill_version_status(current["version_id"], "retired")
    restored = store.set_skill_version_status(target["version_id"], "active")
    return {"rolled_back": True, "version": restored,
            "replaced": current["version_id"] if current else None}


def improve_from_lessons(store, capability: str) -> Dict[str, Any]:
    """End-to-end: propose, then activate only when the gate passes."""
    skill_id = skill_for(capability)
    if not skill_id:
        return {"proposed": False,
                "reason": f"no learned skill is defined for {capability!r}"}
    lessons = [l for l in L.active_lessons(store, capability=capability)
               if l.get("capability") == capability]
    if not lessons:
        return {"proposed": False,
                "reason": "no active lesson for this capability"}
    version = propose(store, skill_id, capability, lessons[:6])
    outcome = activate(store, version["version_id"])
    return {"proposed": True, "skill_id": skill_id, **outcome}
