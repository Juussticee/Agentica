"""Calibration: what KIND of feedback is this, and how wide may it apply?

The learning loop can improve behaviour, or it can quietly make the agent
cautious: one "don't touch that file" turning into "never modify files", one
frustrated message turning into a standing refusal, a repeated complaint
turning into a new approval gate. This module is the deterministic guard
between a user's words and a stored lesson.

It answers four questions about ONE message (no model call):

  * `feedback_type` — specific correction, contextual preference, general
    preference, task constraint, temporary instruction, frustration, quality
    feedback, or a safety/permission constraint. These are NOT interchangeable:
    only a preference the user stated as such may generalise.
  * `impact` — does the lesson improve quality, adjust a preference, improve
    execution, restrict within a scope, or enforce a safety boundary? A
    restriction needs more evidence than an improvement.
  * `overgeneralization` — does the lesson read WIDER than the user's own
    words (a restriction without its object, a capability scope the user never
    named, a temporary instruction made permanent)? Blocking findings hold the
    lesson for review.
  * `scope` — the narrowest honest scope: `task`, `workspace`, `capability`,
    or `any`, with conditions/exceptions/duration kept as provenance.

Two rules that follow from the milestone and are enforced elsewhere:

  * a temporary instruction is stored for the record and NEVER applied to a
    future task (`retrieval` skips it; `lessons.decide` never activates it);
  * pure frustration is not a learning signal at all (`learnable=False`), so
    the system asks what was wrong instead of inventing a rule.

Nothing here writes, executes or reads a file: it is text analysis over the
user's own message, used as a proposal that `lessons.decide` still has to
accept.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

# --- feedback types -------------------------------------------------------- #

SPECIFIC_CORRECTION = "specific_correction"
CONTEXTUAL_PREFERENCE = "contextual_preference"
GENERAL_PREFERENCE = "general_preference"
TASK_CONSTRAINT = "task_constraint"
TEMPORARY_INSTRUCTION = "temporary_instruction"
FRUSTRATION = "frustration"
QUALITY_FEEDBACK = "quality_feedback"
SAFETY_CONSTRAINT = "safety_constraint"

FEEDBACK_TYPES = (SPECIFIC_CORRECTION, CONTEXTUAL_PREFERENCE,
                  GENERAL_PREFERENCE, TASK_CONSTRAINT, TEMPORARY_INSTRUCTION,
                  FRUSTRATION, QUALITY_FEEDBACK, SAFETY_CONSTRAINT)

# --- impact classes -------------------------------------------------------- #

QUALITY = "quality_improvement"
PREFERENCE = "preference_adjustment"
EXECUTION = "execution_improvement"
CONSTRAINT = "constraint"
SAFETY = "safety_requirement"

IMPACTS = (QUALITY, PREFERENCE, EXECUTION, CONSTRAINT, SAFETY)

# --- scopes ---------------------------------------------------------------- #

SCOPE_TASK = "task"
SCOPE_WORKSPACE = "workspace"
SCOPE_CAPABILITY = "capability"
SCOPE_ANY = "any"

# --- patterns -------------------------------------------------------------- #
# Every pattern below describes the USER'S message, not a proposed behaviour.
# They are deliberately literal: a false negative means "hold for review", a
# false positive means "apply a rule the user never stated", so the safe side
# is to under-detect.

_TEMPORARY_RE = re.compile(
    r"\b(?:just\s+this\s+once|this\s+time\s+only|only\s+this\s+time|for\s+now|"
    r"for\s+this\s+(?:task|run|request|one|time)|temporar(?:y|ily)|"
    r"until\s+(?:i\s+say|further\s+notice|i\s+tell\s+you)|one[-\s]?off|"
    r"in\s+this\s+session)\b",
    re.IGNORECASE)

_CONTEXT_RE = re.compile(
    r"\b(?:in|for|on)\s+this\s+(?:project|repo|repository|codebase|workspace|"
    r"spreadsheet|file|document|task|client|company)\b"
    r"|\bthis\s+(?:project|repo|repository|codebase|workspace)\b"
    r"|\bfor\s+(?:this\s+)?(?:client|team|company)\b"
    r"|\bin\s+this\s+case\b|\bhere\b",
    re.IGNORECASE)

# A standing rule, stated as one: the user marks it as applying beyond this
# task. "next time" is deliberately NOT here (2026-10-31): it names the next
# COMPARABLE task ("include trade-offs next time"), not every request, and
# treating it as global turned a content requirement into a rule applied to
# unrelated questions.
_GENERAL_RE = re.compile(
    r"\b(?:from\s+now\s+on|always|never|every\s+time|whenever|in\s+general|"
    r"all\s+(?:my|of\s+my)\s+\w+|every\s+answer|every\s+task|"
    r"across\s+the\s+board)\b",
    re.IGNORECASE)

_SAFETY_RE = re.compile(
    r"\b(?:permission|permissions|access|sudo|admin|root|credential|"
    r"credentials|api\s+key|token|secret|password|firewall|approval|"
    r"security|boundary|production|prod\s+environment)\b"
    r"[\s\S]{0,80}?\b(?:grant|allow|disable|remove|bypass|ignore|turn\s+off|"
    r"stop\s+asking|change|deploy|modify)\b"
    r"|\b(?:grant|allow)\b[\s\S]{0,40}?\b(?:full|unrestricted|unlimited)\b"
    r"|\bdisable\s+(?:the\s+)?(?:approval|security|safety)\b"
    r"|\b(?:be\s+)?careful\s+(?:with|around)\b[\s\S]{0,40}"
    r"\b(?:production|prod|live|credentials|secrets)\b",
    re.IGNORECASE)

_RESTRICTION_RE = re.compile(
    r"\b(?:don'?t|do\s+not|never|stop|avoid|must\s+not|should\s+not|"
    r"shouldn'?t|refuse\s+to|decline\s+to|leave\s+(?:it|them|that)\s+"
    r"unchanged|without\s+changing)\b",
    re.IGNORECASE)

# The user's own object/context narrows a restriction ("this file", "the
# original dates"). A behaviour that drops it has widened the restriction.
_NARROW_OBJECT_RE = re.compile(
    r"\b(?:this|that|these|those|the\s+original|the\s+existing|current)\s+"
    r"(?:file|files|column|columns|row|rows|date|dates|sheet|spreadsheet|"
    r"project|repo|repository|document|passage|config|settings|branch|"
    r"task|request|answer|result)\b",
    re.IGNORECASE)

_FRUSTRATION_RE = re.compile(
    r"\b(?:useless|frustrat\w*|annoy\w*|terrible|awful|ridiculous|"
    r"not\s+helpful|you'?re\s+not\s+helping|waste\s+of\s+time|"
    r"disappoint\w*|fed\s+up|sick\s+of|still\s+(?:wrong|broken)|"
    r"this\s+is\s+(?:bad|awful|terrible))\b",
    re.IGNORECASE)

# A behaviour this module treats as a restriction when the user did not ask
# for one. "Include X" is an improvement; "never do Y" removes a capability.
_INSTRUCTION_RESTRICTION_RE = re.compile(
    r"\b(?:never|do\s+not|don'?t|avoid|refuse|decline|stop|must\s+not)\b",
    re.IGNORECASE)

_ABSOLUTE_RE = re.compile(
    r"\b(?:never|always|all|every|any|entirely|completely|"
    r"under\s+no\s+circumstances)\b",
    re.IGNORECASE)

_CORRECTIVE_CUE_RE = re.compile(
    r"\b(?:include|add|cover|cite|verify|preserve|check|use|format|"
    r"explain|show|list|give|provide|keep|report|mention|compare)\b",
    re.IGNORECASE)

# A defect named in the message is already a specific complaint, even when no
# imperative verb follows it ("your research is incomplete").
_DEFECT_RE = re.compile(
    r"\b(?:incomplete|missing|omitted|omits?|lacks?|lacking|wrong|incorrect|"
    r"inaccurate|shallow|superficial|outdated|unsupported)\b",
    re.IGNORECASE)

_CAPABILITY_CUES = (
    ("research", re.compile(
        r"\bbenchmark|vram|licen[cs]ing|citation|sources?|evidence|"
        r"competitor|pricing|research|web\s+search", re.IGNORECASE)),
    ("data", re.compile(
        r"\bspreadsheet|\.csv\b|\.xlsx?\b|column|row|dataset|dates?\b|"
        r"chart|aggregation", re.IGNORECASE)),
    ("coding", re.compile(
        r"\bpatch|diff|repo|repository|codebase|unit\s+test|refactor|"
        r"\bfunction\b|\bfile\b", re.IGNORECASE)),
    ("books", re.compile(
        r"\bdocument|chapter|\bbook\b|manual|knowledge\s+base|passage",
        re.IGNORECASE)),
    ("erp", re.compile(r"\berp\b|\border\b|\binvoice|inventory", re.IGNORECASE)),
)


def _capability_of(text: str) -> str:
    for capability, pattern in _CAPABILITY_CUES:
        if pattern.search(text or ""):
            return capability
    return ""


def _conditions(text: str) -> List[str]:
    """Clauses the user attached to the instruction (kept as provenance)."""
    found = []
    for match in re.finditer(
            r"\b(?:unless|except|only\s+if|only\s+when|when|if)\b[^.!?;]{3,120}",
            text or "", re.IGNORECASE):
        clause = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;")
        if 4 <= len(clause) <= 140 and clause.lower() not in (
                item.lower() for item in found):
            found.append(clause)
    return found[:4]


def _restricts(text: str) -> bool:
    return bool(_RESTRICTION_RE.search(text or ""))


def _substantive(text: str, missing: Sequence[str], kind: str) -> bool:
    """Is there something to learn, or only a complaint?

    A corrective KIND is already the signal ("that's wrong, the figure is 24 "
    "GB"): it names a fact or a format even when it states no requirement in
    the words this module looks for. Only a bare expression of displeasure
    with no corrective reading is treated as nothing to learn.
    """
    if list(missing or []):
        return True
    if _CORRECTIVE_CUE_RE.search(text or ""):
        return True
    if _DEFECT_RE.search(text or ""):
        return True
    return str(kind or "") in ("incorrect", "misunderstood", "repeated",
                                "failure", "format", "future_directive",
                                "constraint")


def _feedback_type(text: str, *, kind: str, unsafe: bool,
                   missing: Sequence[str] = ()) -> str:
    if unsafe or _SAFETY_RE.search(text or ""):
        return SAFETY_CONSTRAINT
    if _TEMPORARY_RE.search(text or ""):
        return TEMPORARY_INSTRUCTION
    contextual = bool(_CONTEXT_RE.search(text or ""))
    general = bool(_GENERAL_RE.search(text or ""))
    if _restricts(text or "") and (contextual or _NARROW_OBJECT_RE.search(text or "")):
        return TASK_CONSTRAINT
    if _FRUSTRATION_RE.search(text or "") and not _substantive(text, missing,
                                                              kind):
        return FRUSTRATION
    if kind == "incorrect" and not contextual:
        return SPECIFIC_CORRECTION
    if kind == "format":
        # A format wish with no context is a general preference; only a wish
        # explicitly bounded to this project/task is contextual.
        return CONTEXTUAL_PREFERENCE if contextual else GENERAL_PREFERENCE
    if contextual and not general:
        return CONTEXTUAL_PREFERENCE
    if general:
        return GENERAL_PREFERENCE
    if kind in ("insufficient", "misunderstood", "repeated", "failure"):
        return QUALITY_FEEDBACK
    return QUALITY_FEEDBACK


def _impact(feedback_type: str, text: str) -> str:
    if feedback_type == SAFETY_CONSTRAINT:
        return SAFETY
    if feedback_type in (TASK_CONSTRAINT, TEMPORARY_INSTRUCTION):
        return CONSTRAINT
    if feedback_type == GENERAL_PREFERENCE and _restricts(text or ""):
        return CONSTRAINT
    if feedback_type in (QUALITY_FEEDBACK, SPECIFIC_CORRECTION, FRUSTRATION):
        return QUALITY
    if _restricts(text or ""):
        return CONSTRAINT
    return PREFERENCE


def _scope(feedback_type: str) -> str:
    if feedback_type == TEMPORARY_INSTRUCTION:
        return SCOPE_TASK
    if feedback_type == TASK_CONSTRAINT:
        return SCOPE_WORKSPACE
    if feedback_type == CONTEXTUAL_PREFERENCE:
        return SCOPE_WORKSPACE
    if feedback_type == GENERAL_PREFERENCE:
        return SCOPE_ANY
    return SCOPE_CAPABILITY


def _duration(feedback_type: str) -> str:
    if feedback_type == TEMPORARY_INSTRUCTION:
        return "single_task"
    if feedback_type in (TASK_CONSTRAINT, CONTEXTUAL_PREFERENCE):
        return "project"
    return "standing"


def findings(user_text: str, behavior: str, *, feedback_type: str,
             capability: str, scope: str) -> List[Dict[str, Any]]:
    """Ways the proposed behaviour reads WIDER than the user's own words.

    Each finding has `kind`, `detail` and `blocking`. Blocking findings do not
    reject the lesson — they send it to review, because the analysis may be
    right and the user may genuinely have meant the wider rule.
    """
    text = str(user_text or "")
    action = str(behavior or "")
    out: List[Dict[str, Any]] = []

    user_capability = _capability_of(text)
    if (capability == "any" and user_capability
            and feedback_type in (QUALITY_FEEDBACK, SPECIFIC_CORRECTION,
                                  CONTEXTUAL_PREFERENCE, TASK_CONSTRAINT)):
        out.append({
            "kind": "capability_widened",
            "detail": (f"the message is about {user_capability!r} work but the "
                       "lesson would apply to every capability"),
            "blocking": True,
        })

    if (feedback_type in (TASK_CONSTRAINT, CONTEXTUAL_PREFERENCE,
                          TEMPORARY_INSTRUCTION)
            and scope in (SCOPE_ANY, SCOPE_CAPABILITY)):
        out.append({
            "kind": "scope_widened",
            "detail": ("the message is scoped to this task/project but the "
                       "lesson would generalise"),
            "blocking": True,
        })

    if (_INSTRUCTION_RESTRICTION_RE.search(action)
            and not _restricts(text)
            and feedback_type not in (SAFETY_CONSTRAINT, TASK_CONSTRAINT,
                                      TEMPORARY_INSTRUCTION)
            and not _FRUSTRATION_RE.search(text)):
        out.append({
            "kind": "unrequested_restriction",
            "detail": ("the proposed behaviour removes a capability but the "
                       "user only asked for a quality change"),
            "blocking": True,
        })

    if (_INSTRUCTION_RESTRICTION_RE.search(action) and _restricts(text)
            and _NARROW_OBJECT_RE.search(text)
            and not _NARROW_OBJECT_RE.search(action)):
        out.append({
            "kind": "restriction_broadened",
            "detail": ("the user restricted one object ('this file', 'the "
                       "original dates') but the lesson has no such limit"),
            "blocking": True,
        })

    if (feedback_type == TEMPORARY_INSTRUCTION
            and not _TEMPORARY_RE.search(action)):
        out.append({
            "kind": "temporary_made_permanent",
            "detail": ("the user said this applies now/once, so the lesson "
                       "must not become a standing rule"),
            "blocking": True,
        })

    absolute = _ABSOLUTE_RE.search(action)
    if absolute and feedback_type not in (GENERAL_PREFERENCE, SAFETY_CONSTRAINT):
        out.append({
            "kind": "absolute_quantifier",
            "detail": (f"the behaviour says {absolute.group(0)!r} although the "
                       "message states no general rule"),
            "blocking": False,
        })
    return out


def assess(text: str, *, kind: str = "", unsafe: bool = False,
           missing: Sequence[str] = (), behavior: str = "",
           capability: str = "") -> Dict[str, Any]:
    """Classify one user message (and, when given, its proposed behaviour).

    Returns a plain dict so it can be stored as lesson provenance, rendered
    in the Control Center, and asserted in tests without a schema.
    """
    message = re.sub(r"\s+", " ", str(text or "")).strip()
    missing = list(missing or [])
    feedback_type = _feedback_type(message, kind=kind, unsafe=unsafe,
                                   missing=missing)
    substantive = _substantive(message, missing, kind) or feedback_type in (
        SAFETY_CONSTRAINT, TEMPORARY_INSTRUCTION, TASK_CONSTRAINT)
    learnable = feedback_type != FRUSTRATION and substantive
    scope = _scope(feedback_type)
    impact = _impact(feedback_type, message)
    restricts = impact in (CONSTRAINT, SAFETY)
    widening = findings(message, behavior, feedback_type=feedback_type,
                        capability=capability or _capability_of(message),
                        scope=scope)
    review = [f["detail"] for f in widening if f["blocking"]]
    if feedback_type == TEMPORARY_INSTRUCTION:
        review.append("a temporary instruction is never applied to a future "
                      "task")
    if impact == CONSTRAINT and not review:
        review.append("a restriction needs review before it can apply")
    if impact == SAFETY:
        review.append("a safety/permission request is never applied "
                      "automatically")
    return {
        "feedback_type": feedback_type,
        "impact": impact,
        "restricts": restricts,
        "learnable": learnable,
        "temporary": feedback_type == TEMPORARY_INSTRUCTION,
        "scope": scope,
        "scope_detail": {
            "kind": _duration(feedback_type),
            "capability": capability or _capability_of(message) or "any",
            "conditions": _conditions(message),
            "narrow_object": bool(_NARROW_OBJECT_RE.search(message)),
        },
        "overgeneralization": widening,
        "review_reasons": review,
        "escalate": ("investigate_recurring_failure" if kind == "repeated"
                     else None),
    }


def blocking(assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [f for f in assessment.get("overgeneralization") or []
            if f.get("blocking")]


def summary_line(assessment: Dict[str, Any]) -> str:
    """One-line explanation for a reply or a table cell (no jargon)."""
    kind = str(assessment.get("feedback_type") or "feedback").replace("_", " ")
    impact = str(assessment.get("impact") or "").replace("_", " ")
    if assessment.get("temporary"):
        return f"{kind} — applies to this task only"
    return f"{kind} — {impact}"


__all__ = [
    "SPECIFIC_CORRECTION", "CONTEXTUAL_PREFERENCE", "GENERAL_PREFERENCE",
    "TASK_CONSTRAINT", "TEMPORARY_INSTRUCTION", "FRUSTRATION",
    "QUALITY_FEEDBACK", "SAFETY_CONSTRAINT", "FEEDBACK_TYPES",
    "QUALITY", "PREFERENCE", "EXECUTION", "CONSTRAINT", "SAFETY", "IMPACTS",
    "SCOPE_TASK", "SCOPE_WORKSPACE", "SCOPE_CAPABILITY", "SCOPE_ANY",
    "assess", "blocking", "findings", "summary_line",
]
