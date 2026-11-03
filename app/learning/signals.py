"""Deterministic feedback detection (2026-10-30) — no model call.

The learning system must not treat every message as feedback (an ordinary
question is not a correction), and it must not need a model to notice that one
IS feedback: the wording of a correction is recognisable. This module is the
cheap, testable first stage. It answers two questions only:

  * is this message feedback about earlier work?  (`detect` -> None or signal)
  * which kind, which capability, and — when stated — which requirements were
    missing?

WHY THE CUES ARE NARROW
A user message reaches this module only after the routing policy decided it is
a correction route, so precision here protects two things at once:
  * a genuine question ("is that enough detail?") is not learning material;
  * an external document's text can never become a lesson, because detection
    runs on the USER MESSAGE only, never on retrieved passages.
Evidence text is stored for provenance; it is data, and nothing in this module
turns it into an instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- kinds ---------------------------------------------------------------- #
INSUFFICIENT = "insufficient"          # "not enough", "too shallow", "missed X"
INCORRECT = "incorrect"                # "that's wrong", "the number is wrong"
MISUNDERSTOOD = "misunderstood"        # "not what I asked"
FORMAT = "format"                      # "use a table", "be more concise"
FUTURE_DIRECTIVE = "future_directive"  # "next time, also cite sources"
REPEATED = "repeated"                  # "you made the same mistake again"
FAILURE = "failure"                    # "the run keeps failing" / incomplete
CONSTRAINT = "constraint"              # "don't change this file" (2026-10-30)
FRUSTRATION = "frustration"            # "this is useless" — no rule in it

CATEGORY_PREFERENCE = "preference"
CATEGORY_PROJECT = "project"
CATEGORY_PROCEDURE = "procedure"
CATEGORY_EXPERIENCE = "experience"

# Messages that ask for a change to Agentica's own privileges or security
# posture are NOT learnable instructions. They are recorded as a proposal for
# the user to review; nothing is applied. Kept here (not in the lesson module)
# because it is a property of the message, not of the lesson.
_UNSAFE_CHANGE_RE = re.compile(
    r"\b(?:permission|permissions|access|sudo|admin|root|credential|"
    r"credentials|api\s+key|token|secret|password|firewall|"
    r"approval\s+requirement|security|boundary)\b"
    r"(?:[\s\S]{0,60}?\b(?:grant|allow|disable|remove|bypass|ignore|turn\s+off|"
    r"stop\s+asking|change)\b)?"
    r"|\b(?:grant|allow)\b[\s\S]{0,40}?\b(?:full|unrestricted|unlimited)\b",
    re.IGNORECASE,
)

_KIND_RULES: List[Tuple[str, str, "re.Pattern[str]"]] = [
    (REPEATED, CATEGORY_EXPERIENCE, re.compile(
        r"\b(?:same|another)\s+(?:mistake|error|problem|issue)\b"
        r"|\bagain\b[\s\S]{0,40}\b(?:mistake|wrong|forgot|missed)\b"
        r"|\byou\s+keep\s+(?:doing|making|forgetting|missing)\b",
        re.IGNORECASE)),
    (INCORRECT, CATEGORY_EXPERIENCE, re.compile(
        # An operation the user did not want performed counts as a factual
        # correction: "you changed the spreadsheet dates".
        r"\byou\s+(?:changed|overwrote|overwrit|dropped|deleted|renamed|"
        r"reformatted|converted|replaced|removed)\b"
        r"|"
        r"\b(?:that|this|it|your\s+(?:answer|result|number|figure|statement))"
        r"\s*(?:'s|is|was)\s+(?:incorrect|wrong|inaccurate|not\s+(?:right|correct))\b"
        r"|\byou\s+(?:got|have)\s+(?:it|that)\s+wrong\b"
        r"|\bthat\s+is\s+not\s+(?:true|correct|accurate)\b",
        re.IGNORECASE)),
    (MISUNDERSTOOD, CATEGORY_EXPERIENCE, re.compile(
        r"\byou\s+misunderstood\b|\bnot\s+what\s+i\s+(?:asked|wanted|meant)\b"
        r"|\bthat'?s\s+not\s+what\s+i\s+(?:asked|wanted|meant)\b"
        r"|\bi\s+(?:asked|wanted)\s+(?:for|you\s+to)\b",
        re.IGNORECASE)),
    # A report that an execution kept failing or came back incomplete. Used
    # by the failure-learning path (origin=execution_failure); it still has to
    # name a run/task so an ordinary "errors in Python" question never
    # matches.
    (FAILURE, CATEGORY_EXPERIENCE, re.compile(
        r"\b(?:run|execution|task|agent|job)\b[\s\S]{0,60}?\b(?:failed|"
        r"failing|error|incomplete|broke|blocked)\b"
        r"|\bkeeps?\s+failing\b|\bfailed\s+again\b",
        re.IGNORECASE)),
    # A restriction stated as an instruction ("don't modify this file").
    # Recognised so the calibration layer can scope it narrowly instead of
    # letting it become a standing rule; the anaphor requirement keeps
    # "never use tabs in code" (a new request) out of the learning path.
    (CONSTRAINT, CATEGORY_PREFERENCE, re.compile(
        r"\b(?:don'?t|do\s+not|never|stop|avoid)\s+(?:modify|change|touch|"
        r"delete|overwrite|rename|edit|remove|drop|reformat|rewrite|move)\b",
        re.IGNORECASE)),
    (INSUFFICIENT, CATEGORY_EXPERIENCE, re.compile(
        r"\b(?:not|isn'?t|wasn'?t|aren'?t)\s+(?:enough|sufficient|"
        r"detailed\s+enough|specific\s+enough)\b"
        r"|\binsufficient\b|\btoo\s+(?:short|shallow|vague|generic|brief)\b"
        r"|\bmissing\s+(?:several|some|important|the)\b"
        r"|\bdidn'?t\s+(?:cover|include|answer|address)\b"
        r"|\bi\s+need\s+(?:actual|real|more|proper)\b",
        re.IGNORECASE)),
    (FORMAT, CATEGORY_PREFERENCE, re.compile(
        r"\b(?:use|format\s+(?:it|this|as))\b[\s\S]{0,30}\b(?:table|bullets?|"
        r"list|paragraphs?|sections?|markdown)\b"
        r"|\bmore\s+(?:concise|verbose|detail|detail\s+please)\b"
        r"|\bshorter\b|\blonger\b|\b(?:bullet|ordered|numbered)\s+points\b",
        re.IGNORECASE)),
    (FUTURE_DIRECTIVE, CATEGORY_PROCEDURE, re.compile(
        r"\bnext\s+time\b|\bfrom\s+now\s+on\b|\bin\s+future\b"
        r"|\balways\s+(?:include|use|cite|check|verify|preserve|ask|format)\b"
        r"|\bnever\s+(?:drop|change|delete|overwrite|invent|guess)\b"
        r"|\bplease\s+(?:also\s+)?(?:include|add|cover|cite|verify)\b",
        re.IGNORECASE)),
    # Dissatisfaction with nothing reusable in it. Ordered LAST because it is
    # the least specific reading; `calibration.assess` marks it not learnable,
    # so it can never become a rule (the agent asks what was wrong instead).
    (FRUSTRATION, CATEGORY_EXPERIENCE, re.compile(
        r"\b(?:useless|frustrat\w*|annoy\w*|terrible|awful|ridiculous|"
        r"not\s+helpful|waste\s+of\s+time|disappoint\w*|fed\s+up|"
        r"still\s+(?:wrong|broken))\b",
        re.IGNORECASE)),
]

# A correction refers BACK to something. Without one of these the message is
# most likely a new (possibly short) request, not feedback about prior work.
_ANAPHOR_RE = re.compile(
    r"\b(?:that|this|it|these|those|your|you|again|"
    r"(?:the|these|my)\s+(?:results?|answer|output|spreadsheet|file|patch|"
    r"summary|report))\b",
    re.IGNORECASE,
)

# Named requirements inside a correction ("I need benchmarks, VRAM and
# licensing"). Extracted verbatim for provenance and for the skill candidate;
# never executed.
_REQUIREMENT_RE = re.compile(
    r"(?:\bi\s+(?:(?:also|still|really|just|only|actually)\s+)*"
    r"(?:need|needed|want|wanted|expected|require|required)\b"
    r"|\b(?:include|add|cover|cite|verify|preserve|check)\b"
    r"|\bmiss(?:ed|ing)\b"
    r"|\bshould\s+(?:also\s+)?(?:include|cover|have)\b"
    r"|\bshould\s+have\b)"
    r"([^.!?;]{3,160})",
    re.IGNORECASE,
)

_QUESTION_SHAPE_RE = re.compile(
    r"^\s*(?:what|why|how|who|whom|whose|when|where|which|"
    r"is|are|was|were|do|does|did|can|could|would|should|will|"
    r"explain|describe|define|list|tell\s+me|show\s+me|compare)\b",
    re.IGNORECASE,
)

_CAPABILITY_CUES: List[Tuple[str, "re.Pattern[str]"]] = [
    ("research", re.compile(
        r"\bbenchmark|vram|licen[cs]ing|citation|sources?|evidence|"
        r"competitor|pricing|web\s+search|research", re.IGNORECASE)),
    ("data", re.compile(
        r"\bspreadsheet|\.csv\b|\.xlsx?\b|column|row|dataset|dates?\b|"
        r"aggregation|chart", re.IGNORECASE)),
    ("coding", re.compile(
        r"\bpatch|diff|repo|repository|codebase|unit\s+test|refactor|"
        r"\bfunction\b|\bfile\b", re.IGNORECASE)),
    ("books", re.compile(
        r"\bdocument|chapter|\bbook\b|manual|knowledge\s+base|passage",
        re.IGNORECASE)),
    ("erp", re.compile(r"\berp\b|\border\b|\binvoice|inventory", re.IGNORECASE)),
]


@dataclass(frozen=True)
class FeedbackSignal:
    """One detected correction. `text` is the user's own words (data only)."""

    kind: str
    category: str
    capability: str = "any"
    confidence: float = 0.5
    future_directive: bool = False
    missing: List[str] = field(default_factory=list)
    unsafe_change: bool = False
    text: str = ""


def is_feedback(text: str) -> bool:
    """True when the message is shaped like feedback about earlier work."""
    return detect(text) is not None


def _requirements(text: str) -> List[str]:
    found: List[str] = []
    for match in _REQUIREMENT_RE.finditer(text):
        chunk = match.group(1).strip(" ,:;.-\t")
        # Split "benchmarks, VRAM requirements and licensing" into its parts.
        parts = re.split(r",\s*|\s+and\s+|\s*/\s*", chunk)
        for part in parts:
            part = re.sub(r"^(?:and|or)\s+", "", part.strip(" ,:;.-\t"))
            if 2 <= len(part) <= 80 and part.lower() not in (
                    p.lower() for p in found):
                found.append(part)
    return found[:8]


def _capability(text: str) -> str:
    for capability, pattern in _CAPABILITY_CUES:
        if pattern.search(text):
            return capability
    return "any"


def detect(text: str, *, capability: str = "") -> Optional[FeedbackSignal]:
    """Classify `text` as feedback, or None when it is not one.

    `capability` (the route that produced the earlier answer) is used as a
    fallback hint when the message itself names no subject — a bare "this
    isn't enough" is about the work the user just saw.
    """
    message = (text or "").strip()
    if not message:
        return None
    # A genuine question is a new request, not a correction: "why is it not
    # enough to use a checklist?" contains an insufficiency cue and an
    # anaphor, and routing it to feedback would replace a real question with
    # a lesson recording. Question-shaped messages are therefore excluded.
    if _QUESTION_SHAPE_RE.match(message) and message.rstrip().endswith("?"):
        return None
    kind, category, confidence = "", CATEGORY_EXPERIENCE, 0.0
    for rule_kind, rule_category, pattern in _KIND_RULES:
        if pattern.search(message):
            kind, category = rule_kind, rule_category
            # First matching rule wins: kind rules are ordered by specificity.
            confidence = 0.8 if rule_kind == REPEATED else 0.7
            break
    if not kind:
        return None
    # A bare correction ("that's wrong") is feedback; a question that merely
    # contains a cue word but refers to nothing earlier is not. A FAILURE
    # report names its own subject ("the run keeps failing"), so it does not
    # need an anaphor.
    # A frustration message often names nothing else, and its own wording is
    # the whole signal ("this is useless"), so `this`/`it` is enough.
    if kind != FAILURE and not _ANAPHOR_RE.search(message) and not re.search(
            r"\bnext\s+time\b|\bfrom\s+now\s+on\b", message, re.IGNORECASE):
        return None
    future = category == CATEGORY_PROCEDURE or bool(
        re.search(r"\bnext\s+time\b|\bfrom\s+now\s+on\b|\balways\b", message,
                  re.IGNORECASE))
    return FeedbackSignal(
        kind=kind,
        category=category,
        capability=capability or _capability(message),
        confidence=confidence,
        future_directive=future,
        missing=_requirements(message),
        unsafe_change=bool(_UNSAFE_CHANGE_RE.search(message)),
        text=message,
    )
