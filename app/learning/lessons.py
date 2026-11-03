"""Structured lessons: analysis, dedupe, activation policy, retrieval text.

A lesson is a small record with provenance, not a memory blob:

    problem / root_cause / corrective_behavior / validation
    category (preference | project | procedure | experience)
    capability scope + origin (user_feedback | execution_failure | success)
    confidence, evidence, status, hits/successes/failures

UNDERSTANDING vs TRUST
`analyze()` asks the model what the correction MEANS (a keyword match cannot
tell "I needed benchmarks" from "benchmarks are irrelevant"). The model's
output is a PROPOSAL: it is validated field by field, and the deterministic
analysis in `signals.py` is used whenever the model is unavailable or returns
something unusable. A model never decides that a lesson is active — the
policy in `decide()` does, from the message's own shape and the lesson's
category. Nothing here can grant permissions, change an approval rule or alter
credentials; such requests are stored as reviewed proposals only.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from app.execution.store import utc_now_iso

from app.learning import calibration as C
from app.learning import signals as S

# Statuses. `candidate` waits for review, `active` is applied to new work,
# `rejected`/`superseded`/`disabled` are kept for provenance and never applied.
CANDIDATE = "candidate"
ACTIVE = "active"
VALIDATED = "validated"
REJECTED = "rejected"
SUPERSEDED = "superseded"
DISABLED = "disabled"

ORIGIN_USER = "user_feedback"
ORIGIN_FAILURE = "execution_failure"

# Capabilities a lesson may scope itself to. `any` is allowed for preferences
# (how the user likes answers) but NOT for factual corrections, where a
# capability-specific rule is the only honest scope.
CAPABILITIES = ("any", "general", "research", "books", "erp", "data",
                "coding", "feasibility", "followup")

_MAX_BEHAVIOR = 400
_MAX_PROBLEM = 300

# A learning call must never block a run forever: on 2026-10-31 a local
# qwen3:0.6b call stalled the whole feedback execution with no transport
# timeout. Bounded here, and the deterministic reading is used on timeout.
ANALYSIS_TIMEOUT = 60.0

_ANALYZE_PROMPT = """You are analysing ONE user correction so an agent can
avoid repeating the mistake. Read the earlier work and the user's correction.

Return ONLY these lines (no preamble, no markdown):
PROBLEM: <what went wrong, one sentence>
CAUSE: <likely cause, or "unknown">
BEHAVIOUR: <the concrete behaviour to follow next time, one imperative sentence>
VALIDATION: <how a future answer could be checked for this, one sentence>
KEYWORDS: <3-5 comma-separated keywords describing when this applies>

Do not invent facts about the earlier work. If the correction is only a
preference, describe it as a preference.

Earlier request: {prior_question}
Earlier result: {prior_answer}
User correction: {text}
Detected kind: {kind}; stated missing items: {missing}
"""

_FALLBACK_BEHAVIOR = {
    S.INSUFFICIENT: "Cover every requirement the user names, and say plainly "
                    "which ones could not be supported.",
    S.INCORRECT: "Re-check the specific claim the user disputed against the "
                 "evidence before restating it.",
    S.MISUNDERSTOOD: "Restate the request in the user's own terms before "
                     "answering, and answer that reading.",
    S.FORMAT: "Follow the requested output format.",
    S.FUTURE_DIRECTIVE: "Apply the user's stated procedure to comparable tasks.",
    S.REPEATED: "Check whether this correction has been given before and "
                "follow the recorded behaviour instead of repeating it.",
    S.FAILURE: "Report the failure honestly, name the failing step, and do "
               "not present an incomplete result as complete.",
    S.CONSTRAINT: "Respect the stated restriction, and say what was left "
                  "unchanged.",
    S.FRUSTRATION: "Ask what specifically was wrong before changing "
                   "behaviour.",
}


def _clean(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    return text[:limit]


# A small local model sometimes answers the analysis prompt with the SCHEMA
# instead of the analysis: "PROBLEM: <what went wrong, one sentence>". Such a
# value is a template, not a finding — storing it would inject a meaningless
# rule ("<the concrete behaviour to follow next time>") into every later
# prompt. A placeholder is treated exactly like a missing field, so the
# deterministic reading of the user's own words is used instead.
_PLACEHOLDER_RE = re.compile(r"<[^<>]{2,}>")
_SCHEMA_HINTS = (
    "what went wrong", "likely cause",
    "the concrete behaviour", "the concrete behavior",
    "how a future answer", "comma-separated keywords", "3-5 keywords",
    "one sentence",
)


def _placeholder(value: Any) -> bool:
    """True when a field holds the prompt's template text, not an answer."""
    text = str(value if value is not None else "").strip().strip("\"'")
    if not text:
        return False
    if _PLACEHOLDER_RE.fullmatch(text):
        return True
    low = text.lower()
    return any(hint in low for hint in _SCHEMA_HINTS)


def _field(raw: str, name: str) -> str:
    match = re.search(rf"^\s*{name}\s*:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    value = _clean(match.group(1), 400).strip("\"'\u201c\u201d\u2018\u2019")
    return "" if _placeholder(value) else value


def _keywords(raw: str) -> List[str]:
    value = _field(raw, "KEYWORDS")
    return [k.strip().lower() for k in value.split(",")
            if k.strip() and not _placeholder(k)][:6]


# Readable one-line statement of what went wrong, used when the model is
# unusable. The user's own missing items are appended by `analyze`.
_PROBLEM_PHRASE = {
    S.INSUFFICIENT: "The earlier result did not go far enough",
    S.INCORRECT: "Something in the earlier result was wrong",
    S.MISUNDERSTOOD: "The earlier answer addressed a different question",
    S.REPEATED: "The same mistake happened again",
    S.FAILURE: "The earlier run failed",
    S.FORMAT: "The earlier output format was not what the user wanted",
    S.FUTURE_DIRECTIVE: "The user wants comparable tasks handled differently",
    S.CONSTRAINT: "The user stated a constraint for comparable work",
}


_DIRECTIVE_RE = re.compile(
    r"\b(?:include|add|cover|cite|verify|preserve|keep|use|format|explain|"
    r"show|list|give|provide|report|mention|compare|remember|always|never|"
    r"from\s+now\s+on|next\s+time|check|confirm)\b", re.IGNORECASE)


# A directive that only points ("Include those next time") is not a rule on
# its own: the prompt receives the behaviour, not the evidence. When it points
# at items the user already named, the behaviour states them instead.
_DEICTIC_RE = re.compile(
    r"\b(?:those|these|that|this|them|it)\b", re.IGNORECASE)
_NAMED_DIRECTIVE_RE = re.compile(
    r"^\s*(?:include|add|use|apply|follow|do)\b", re.IGNORECASE)
_DEICTIC_POINTER_RE = re.compile(
    r"^\s*(?:those|these|that|this|them|it)\b", re.IGNORECASE)


def named_items(missing: Any) -> List[str]:
    """The user's named requirements, without the instruction itself."""
    out: List[str] = []
    for item in missing or []:
        phrase = str(item or "").strip()
        if not phrase or _NAMED_DIRECTIVE_RE.match(phrase):
            continue
        # "those next time" names nothing: it is the pointer, not the item.
        if _DEICTIC_POINTER_RE.match(phrase):
            continue
        if phrase.lower() not in (existing.lower() for existing in out):
            out.append(phrase)
    return out[:4]


# A behaviour that only POINTS ("include those next time") cannot be applied
# in a later prompt: after the directive verb and the pointer words, nothing
# is left that names a requirement.
_POINTER_ONLY_WORDS = frozenset(
    "include add use also please next time those these that this them it and "
    "the a an to in of for them accordingly".split())


def _pointer_only(value: str) -> bool:
    words = [w for w in re.findall(r"[a-z0-9-]+", str(value or "").lower())
             if w not in _POINTER_ONLY_WORDS]
    return len(words) < 2


def _directive(text: str, missing: Any = ()) -> str:
    """The user's own instruction sentence, used when the model is unusable.

    A weak local model frequently returns the schema; the correction itself
    still says what to do ("Include a short trade-off comparison ...").
    Quoting it is honest provenance and is strictly better than a template.

    A purely deictic directive ("Include those next time") would be applied
    to later prompts without saying what "those" are, so the named items are
    substituted when the user supplied them.
    """
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        candidate = sentence.strip(" \t.;:\"'")
        if len(candidate) < 10 or not _DIRECTIVE_RE.search(candidate):
            continue
        named = named_items(missing)
        if named and _DEICTIC_RE.search(candidate):
            return _clean("Ensure the answer covers: " + ", ".join(named)
                          + ".", 240)
        return _clean(candidate[:1].upper() + candidate[1:], 240)
    return ""


def analyze(text: str, signal: S.FeedbackSignal,
            prior: Optional[Dict[str, Any]] = None,
            llm_call: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Propose problem/cause/behaviour/validation for one correction.

    `llm_call(prompt) -> str` is injected in tests; production passes None and
    the client is resolved lazily. Any failure falls back to the
    deterministic reading, and the fallback always names the user's own
    missing items, so the lesson is never empty.
    """
    prior = prior or {}
    missing = ", ".join(signal.missing) or "none stated"
    prompt = _ANALYZE_PROMPT.format(
        prior_question=_clean(prior.get("question"), 600) or "(none recorded)",
        prior_answer=_clean(prior.get("answer"), 2500) or "(none recorded)",
        text=signal.text,
        kind=signal.kind,
        missing=missing,
    )
    raw = ""
    if llm_call is None:
        try:
            from app.llm_client import get_llm, llm_text

            llm = get_llm(timeout=ANALYSIS_TIMEOUT)
            llm_call = lambda p: str(llm_text(llm.invoke(p)) or "")  # noqa: E731
        except Exception:  # noqa: BLE001 — provider unavailable, use fallback
            llm_call = None
    if llm_call is not None:
        try:
            raw = str(llm_call(prompt) or "")
        except Exception:  # noqa: BLE001 — fail-soft, deterministic reading
            raw = ""

    model_problem = _field(raw, "PROBLEM")
    model_cause = _field(raw, "CAUSE")
    model_behavior = _field(raw, "BEHAVIOUR")
    model_validation = _field(raw, "VALIDATION")
    # Only the items the user actually NAMED: a pointer ("those next time")
    # is not a requirement, and storing it would render as one.
    named = named_items(signal.missing)
    # A model behaviour that only points at the user's own words is discarded,
    # so the deterministic path below states those words instead
    # (2026-10-31 live: BEHAVIOUR came back as "include those next time").
    if model_behavior and named and _pointer_only(model_behavior):
        model_behavior = ""
    problem = model_problem or (
        _PROBLEM_PHRASE.get(signal.kind,
                            "The user reported a problem with the earlier "
                            "result")
        + (f"; the user named: {', '.join(named)}" if named else "."))
    cause = model_cause or "unknown"
    behavior = (model_behavior or _directive(signal.text, named)
                or _FALLBACK_BEHAVIOR.get(
                    signal.kind, "Follow the user's correction on comparable tasks."))
    validation = model_validation or (
        ("Check that the next answer explicitly covers: "
         + ", ".join(named) + ".") if named else
        "Every requirement named in the correction is visibly addressed.")
    keywords = _keywords(raw)
    if not keywords:
        keywords = [w.lower() for w in re.findall(r"[A-Za-z]{4,}",
                                                  signal.text)][:5]
    model_fields = [model_problem, model_cause, model_behavior,
                    model_validation]
    if all(model_fields):
        source = "model"
    elif any(model_fields):
        source = "model+fallback"
    else:
        source = "deterministic"
    return {
        "problem": _clean(problem, _MAX_PROBLEM),
        "root_cause": _clean(cause, _MAX_PROBLEM),
        "corrective_behavior": _clean(behavior, _MAX_BEHAVIOR),
        "validation": _clean(validation, 240),
        "keywords": keywords,
        "analysis_source": source,
    }


def scope_for(signal: S.FeedbackSignal) -> str:
    """`capability`-wide for preferences, `task` for factual corrections.

    A formatting preference generalises; "this number was wrong" does not, so
    it must never be replayed into an unrelated answer as if it were a fact.
    """
    return "workspace" if signal.category == S.CATEGORY_PROJECT else "capability"


def capability_for(signal: S.FeedbackSignal) -> str:
    cap = signal.capability or "any"
    if cap not in CAPABILITIES:
        cap = "any"
    return cap


def assess(signal: S.FeedbackSignal, analysis: Optional[Dict[str, Any]] = None
           ) -> Dict[str, Any]:
    """Calibrate one correction (type, impact, scope, widening findings).

    The proposed behaviour is included so the widening check can compare it
    with the user's own words; when the analysis is not available yet the
    message alone is classified.
    """
    return C.assess(
        signal.text,
        kind=signal.kind,
        unsafe=signal.unsafe_change,
        missing=signal.missing,
        behavior=str((analysis or {}).get("corrective_behavior") or ""),
        capability=capability_for(signal),
    )


def build(signal: S.FeedbackSignal, analysis: Dict[str, Any], *,
          source_execution_id: Optional[str],
          origin: str = ORIGIN_USER,
          assessment: Optional[Dict[str, Any]] = None
          ) -> Dict[str, Any]:
    """Assemble a candidate lesson row (no id, no status — see `decide`).

    The calibration is stored as the lesson's own explanation: which KIND of
    feedback produced it, what impact it has, how wide it may apply, and any
    way it reads wider than the user's words. That is what makes "why does it
    behave like this?" answerable later.
    """
    capability = capability_for(signal)
    assessment = assessment or assess(signal, analysis)
    scope = assessment.get("scope") or scope_for(signal)
    # A lesson whose capability already limits it is not globally scoped, even
    # when the user stated a standing preference: `capability` is the bound.
    if scope == C.SCOPE_ANY and capability != "any":
        scope = C.SCOPE_CAPABILITY
    if scope == C.SCOPE_CAPABILITY:
        scope = scope_for(signal)
    assessment = {**assessment, "scope": scope}
    return {
        "category": signal.category,
        "capability": capability,
        "scope": scope_for(signal),
        "problem": analysis.get("problem") or "unspecified",
        "root_cause": analysis.get("root_cause") or "unknown",
        "corrective_behavior": analysis.get("corrective_behavior")
                              or _FALLBACK_BEHAVIOR.get(signal.kind, ""),
        "validation": analysis.get("validation") or "",
        "confidence": round(float(signal.confidence), 2),
        "evidence": [{
            "kind": "user_correction",
            "kind_of_feedback": signal.kind,
            "missing": list(signal.missing),
            "keywords": list(analysis.get("keywords") or []),
            "text": _clean(signal.text, 800),
            "analysis_source": analysis.get("analysis_source", "deterministic"),
            "at": utc_now_iso(),
        }],
        "source_execution_id": source_execution_id,
        "source_text": _clean(signal.text, 800),
        "origin": origin,
        "feedback_type": assessment.get("feedback_type"),
        "impact": assessment.get("impact"),
        "temporary": bool(assessment.get("temporary")),
        "calibration": {
            "feedback_type": assessment.get("feedback_type"),
            "impact": assessment.get("impact"),
            "restricts": bool(assessment.get("restricts")),
            "learnable": bool(assessment.get("learnable")),
            "temporary": bool(assessment.get("temporary")),
            "scope": scope,
            "scope_detail": assessment.get("scope_detail") or {},
            "overgeneralization": assessment.get("overgeneralization") or [],
            "review_reasons": assessment.get("review_reasons") or [],
            "escalate": assessment.get("escalate"),
            "assessed_at": utc_now_iso(),
        },
    }


# Text that tries to become an instruction rather than a requirement. A
# correction is data; when the analysed behaviour reads like an injected
# command it is held for review and can never auto-activate.
_INSTRUCTION_INJECTION_RE = re.compile(
    r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|"
    r"above|earlier|system)\s+(?:instructions?|prompts?|rules?)\b"
    r"|\byou\s+are\s+now\b|\bnew\s+(?:system\s+)?instructions?\b"
    r"|\b(?:reveal|print|expose)\s+(?:the\s+)?(?:system\s+)?prompt\b"
    r"|\bdisable\s+(?:the\s+)?(?:approval|security|safety)"
    r"|\bignoring\s+(?:your|all)\s+(?:rules|guidelines)",
    re.IGNORECASE,
)


def is_injection(text: str) -> bool:
    """Whether a proposed behaviour reads like an injected instruction."""
    return bool(_INSTRUCTION_INJECTION_RE.search(str(text or "")))


# --- dedupe / conflict ----------------------------------------------------- #

_STOP = frozenset(
    "the a an and or of to in on for with is are was were be been this that "
    "it its user answer result should must not no".split())


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
            if w not in _STOP}


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _missing_tokens(lesson: Dict[str, Any]) -> set:
    """What the user named as missing, across the lesson's evidence."""
    tokens: set = set()
    for item in lesson.get("evidence") or []:
        if isinstance(item, dict):
            for phrase in item.get("missing") or []:
                tokens |= _tokens(str(phrase))
    return tokens


def _same_topic(candidate: Dict[str, Any], lesson: Dict[str, Any], *,
                require_category: bool = True) -> bool:
    """Whether two lessons concern the same behaviour.

    `require_category=False` is used for CONFLICT detection: "always cite a "
    "source" (a procedure) and "skip citations in this project" (a
    preference) are the same topic with different intended scopes, and a
    conflict must not be missed because the classifier filed them differently.

    When both corrections NAME what was missing, those names decide it. The
    fallback problem sentence is boilerplate ("The earlier result did not go
    far enough"), so text similarity alone merged two different complaints
    ("trade-offs and numbers" with "caching and payload size") into one lesson
    in the 2026-10-31 live check, burying the second requirement.
    """
    if candidate["capability"] != lesson.get("capability"):
        return False
    if require_category and candidate["category"] != lesson.get("category"):
        return False
    named_a, named_b = _missing_tokens(candidate), _missing_tokens(lesson)
    if named_a and named_b:
        overlap = len(named_a & named_b) / len(named_a | named_b)
        if overlap < 0.34:
            return False
    text_a = f"{candidate.get('problem')} {candidate.get('corrective_behavior')}"
    text_b = f"{lesson.get('problem')} {lesson.get('corrective_behavior')}"
    return similarity(text_a, text_b) >= 0.34


def decide(lesson: Dict[str, Any], signal: S.FeedbackSignal,
           assessment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Choose the lesson's status and record WHY (deterministic policy).

    Auto-activation is limited to corrections the user stated as behaviour:
      * any correction that asks to change privileges/security is never active;
      * an execution-failure lesson is only active after evidence review;
      * a factual correction without a named capability is scoped `any`, which
        would leak into unrelated answers — held for review instead;
      * since 2026-10-30 a TEMPORARY instruction is never applied to a future
        task, and a lesson that reads wider than the user's own words (a
        restriction without its object, a widened capability, a one-off made
        permanent) is held for review;
      * repeated criticism raises the priority of investigating the failure
        and never by itself widens or restricts behaviour.
    """
    behavior = str(lesson.get("corrective_behavior") or "").strip()
    assessment = assessment or (lesson.get("calibration") or {})
    if is_injection(behavior) or is_injection(lesson.get("problem")):
        return {"status": CANDIDATE, "auto_applied": False,
                "decision_reason": "the proposed behaviour reads like an "
                                   "injected instruction; held for review"}
    if signal.unsafe_change or assessment.get("feedback_type") == C.SAFETY_CONSTRAINT:
        return {"status": CANDIDATE, "auto_applied": False,
                "decision_reason": "requests a permission/security change — "
                                   "stored as a proposal, never applied"}
    if not behavior:
        return {"status": CANDIDATE, "auto_applied": False,
                "decision_reason": "no corrective behaviour could be derived"}
    if lesson.get("origin") != ORIGIN_USER:
        return {"status": CANDIDATE, "auto_applied": False,
                "decision_reason": "learned from an execution failure; needs "
                                   "evidence review before it is applied"}
    if assessment.get("temporary"):
        return {"status": CANDIDATE, "auto_applied": False,
                "decision_reason": "temporary instruction — kept for this "
                                   "task's record and never applied to a "
                                   "future task"}
    blocked = C.blocking(assessment)
    if blocked:
        kinds = ", ".join(str(f.get("kind")) for f in blocked)
        return {"status": CANDIDATE, "auto_applied": False,
                "decision_reason": "the lesson reads wider than the "
                                   f"correction ({kinds}); held for review "
                                   "instead of generalising it"}
    if assessment.get("impact") in (C.CONSTRAINT, C.SAFETY):
        return {"status": CANDIDATE, "auto_applied": False,
                "decision_reason": "introduces a restriction; a restriction "
                                   "needs review before it applies"}
    if signal.kind == S.INCORRECT and lesson.get("capability") == "any":
        return {"status": CANDIDATE, "auto_applied": False,
                "decision_reason": "factual correction with no capability "
                                   "scope would apply too broadly"}
    kind = str(assessment.get("feedback_type") or signal.kind)
    return {"status": ACTIVE, "auto_applied": True,
            "decision_reason": f"explicit user correction ({kind}) "
                               f"scoped to {lesson.get('capability')}"}


# --- persistence helpers --------------------------------------------------- #

def active_lessons(store, capability: Optional[str] = None
                   ) -> List[Dict[str, Any]]:
    """ACTIVE lessons for a capability (or all when None). Fail-soft: an
    unreadable store means no lessons, never a crashed answer."""
    try:
        return store.list_lessons(status=ACTIVE, capability=capability)
    except Exception:  # noqa: BLE001
        return []


def record(store, signal: S.FeedbackSignal, *,
           source_execution_id: Optional[str],
           prior: Optional[Dict[str, Any]] = None,
           llm_call: Optional[Callable[[str], str]] = None,
           origin: str = ORIGIN_USER) -> Dict[str, Any]:
    """Analyse, dedupe, decide and persist one lesson. Returns the outcome.

    The result is always a dict describing what happened:
    {action: created|merged|superseded, lesson, decision}. A duplicate message
    (the same correction twice) strengthens the existing lesson instead of
    piling up rows — that is what makes "learned twice" visible rather than
    invisible noise.
    """
    # Observation mode (evaluation harness): measure, do not teach. Checked
    # first so not even the model call is made.
    from app.learning import writes_allowed

    if not writes_allowed():
        return {"action": "skipped", "lesson": {},
                "decision": {"status": None, "auto_applied": False,
                             "decision_reason": "learning writes are disabled "
                                                "for this run (evaluation)"}}

    analysis = analyze(signal.text, signal, prior, llm_call=llm_call)
    assessment = assess(signal, analysis)
    if not assessment.get("learnable"):
        # Frustration is a signal to ASK, not material for a rule. Nothing is
        # stored, so nothing can later be applied: the reply explains that the
        # message was heard but names no reusable requirement.
        return {"action": "not_learned", "lesson": {}, "decision": {
            "status": None, "auto_applied": False,
            "feedback_type": assessment.get("feedback_type"),
            "impact": assessment.get("impact"),
            "temporary": bool(assessment.get("temporary")),
            "escalate": assessment.get("escalate"),
            "decision_reason": "dissatisfaction without a stated requirement: "
                               "nothing to learn until the user names what "
                               "was wrong"}}
    candidate = build(signal, analysis,
                      source_execution_id=source_execution_id, origin=origin,
                      assessment=assessment)
    decision = {**decide(candidate, signal, assessment),
                "feedback_type": assessment.get("feedback_type"),
                "impact": assessment.get("impact"),
                "temporary": bool(assessment.get("temporary")),
                "escalate": assessment.get("escalate")}

    existing = _find_existing(store, candidate)
    if existing is not None and _conflicts_with(existing, candidate):
        # A DIFFERENT procedure/preference for the same topic is not a
        # duplicate: merging it would leave two contradictory behaviours in
        # one lesson. It goes down the create path, which supersedes the old.
        existing = None
    if existing is not None:
        evidence = list(existing.get("evidence") or []) + candidate["evidence"]
        # A repeat can PROMOTE a candidate, never DEMOTE an active lesson: a
        # second, more cautious reading of the same correction must not switch
        # off behaviour that already earned activation.
        status = existing.get("status")
        if status == CANDIDATE and decision["status"] == ACTIVE:
            status = ACTIVE
        elif status not in (CANDIDATE, ACTIVE):
            status = existing.get("status")
        updated = store.update_lesson(
            existing["lesson_id"],
            evidence=evidence,
            confidence=min(0.99, float(existing.get("confidence") or 0) + 0.05),
            status=status,
        )
        return {"action": "merged", "lesson": updated, "decision": {
            **decision, "auto_applied": status == ACTIVE,
            "decision_reason":
            "same topic as an existing lesson; evidence appended"
            + (" (a repeat does not change the scope or add a restriction)"
               if status == ACTIVE else "")}}

    narrowed = _context_conflicts(store, candidate)
    if narrowed and decision["status"] == ACTIVE:
        decision = {**decision, "status": CANDIDATE,"auto_applied": False,
                    "decision_reason":
                    "contradicts a lesson with a different context ("
                    + ", ".join(r["lesson_id"] for r in narrowed)
                    + "); both are kept and this one waits for review"}
        candidate["calibration"] = {
            **(candidate.get("calibration") or {}),
            "conflict_with": [r["lesson_id"] for r in narrowed],
        }

    superseded = _supersede_conflicts(store, candidate, decision)
    lesson_id = f"lsn_{uuid.uuid4().hex[:12]}"
    row = {
        "lesson_id": lesson_id,
        "created_at": utc_now_iso(),
        "status": decision["status"],
        "activated_at": utc_now_iso() if decision["status"] == ACTIVE else None,
        **candidate,
    }
    stored = store.create_lesson(row)
    return {"action": "superseded" if superseded else "created",
            "lesson": stored,
            "decision": {**decision,
                         "superseded": [s["lesson_id"] for s in superseded]}}


def _find_existing(store, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        rows = store.list_lessons(capability=candidate["capability"])
    except Exception:  # noqa: BLE001
        return None
    for row in rows:
        if row.get("status") in (REJECTED, DISABLED, SUPERSEDED):
            continue
        if _same_topic(candidate, row):
            return row
    return None


def _conflicts_with(existing: Dict[str, Any], candidate: Dict[str, Any]
                    ) -> bool:
    if existing.get("category") not in (S.CATEGORY_PREFERENCE,
                                        S.CATEGORY_PROCEDURE):
        return False
    return similarity(str(existing.get("corrective_behavior") or ""),
                      str(candidate.get("corrective_behavior") or "")) < 0.7


def _context_kind(lesson: Dict[str, Any]) -> str:
    """How wide a lesson was meant to apply (from its calibration)."""
    calibration = lesson.get("calibration") or {}
    detail = calibration.get("scope_detail") or {}
    kind = str(detail.get("kind") or "").strip()
    if kind:
        return kind
    scope = str(lesson.get("scope") or "capability")
    return "project" if scope == "workspace" else "standing"


def _behaviours_conflict(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return similarity(str(a.get("corrective_behavior") or ""),
                      str(b.get("corrective_behavior") or "")) < 0.7


def _same_problem(candidate: Dict[str, Any], lesson: Dict[str, Any]) -> bool:
    """Same observed problem, even when the remedy differs.

    Conflict detection compares what the user complained about, not the two
    proposals: "always cite a source" and "skip citations in this project"
    share a problem and disagree, which is exactly the pair a merge would
    corrupt.
    """
    return similarity(str(candidate.get("problem") or ""),
                      str(lesson.get("problem") or "")) >= 0.34


def _same_topic_rows(store, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Active lessons on the same problem (category may differ — see above)."""
    try:
        rows = store.list_lessons(status=ACTIVE,
                                 capability=candidate["capability"])
    except Exception:  # noqa: BLE001
        return []
    return [row for row in rows
            if row.get("lesson_id") != candidate.get("lesson_id")
            and row.get("capability") == candidate["capability"]
            and _same_problem(candidate, row)]


def _context_conflicts(store, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Active lessons on the same topic with a DIFFERENT intended scope.

    "Keep the tabs in this project" and "always use spaces" can both be true.
    A preference stated for one context must not silently replace a standing
    one (or the reverse), so the pair is kept and the new lesson waits for
    review instead of superseding anything.
    """
    if candidate["category"] not in (S.CATEGORY_PREFERENCE, S.CATEGORY_PROCEDURE):
        return []
    mine = _context_kind(candidate)
    return [row for row in _same_topic_rows(store, candidate)
            if _context_kind(row) != mine
            and _behaviours_conflict(row, candidate)]


def _supersede_conflicts(store, candidate: Dict[str, Any],
                         decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A NEW active preference for the same topic replaces the old one.

    Only a lesson on the SAME topic and the SAME context is replaced: "use a
    table" -> "use bullets" supersedes, while "include benchmarks" is left
    alone. The earlier lesson stays in the table (status `superseded`) so the
    change of mind is inspectable and revertible.
    """
    if decision["status"] != ACTIVE:
        return []
    if candidate["category"] not in (S.CATEGORY_PREFERENCE, S.CATEGORY_PROCEDURE):
        return []
    out = []
    for row in _same_topic_rows(store, candidate):
        if not _behaviours_conflict(row, candidate):
            continue
        if _context_kind(row) != _context_kind(candidate):
            continue
        updated = store.update_lesson(
            row["lesson_id"], status=SUPERSEDED,
            superseded_by=candidate.get("lesson_id"))
        out.append(updated)
    return out


# --- prompt rendering ------------------------------------------------------ #

MAX_RENDERED = 6


def render(lessons: List[Dict[str, Any]]) -> str:
    """Compact block of applicable behaviours for a prompt ('' when none).

    Lessons are Agentica's OWN validated text, but they are still inserted as
    quoted constraints under an explicit heading, and capped, so a long lesson
    list can neither flood the context nor read like user instructions.
    """
    lines = []
    for lesson in lessons[:MAX_RENDERED]:
        behavior = _clean(lesson.get("corrective_behavior"), _MAX_BEHAVIOR)
        if not behavior:
            continue
        check = _clean(lesson.get("validation"), 160)
        suffix = f" (check: {check})" if check else ""
        lines.append(f"- [{lesson.get('capability')}] {behavior}{suffix}")
    if not lines:
        return ""
    return ("\n\nLearned requirements for this kind of task — apply them:\n"
            + "\n".join(lines) + "\n")
