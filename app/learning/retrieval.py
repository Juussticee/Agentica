"""Relevance: which ACTIVE lessons a given task must respect (2026-10-30).

Two failure modes this module exists to prevent:

  * injecting every stored lesson into every prompt — noisy, expensive, and
    it would apply a research-depth rule to a coding task;
  * dropping lessons that were learned on one capability but are stated as
    global preferences.

Selection is deterministic: capability must match (or the lesson is scoped
`any`), then lessons are ranked by keyword overlap with the current request,
then by confidence, then by recency. Only the top few are rendered, and the
caller gets their ids back so usage can be counted.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.learning import calibration as C
from app.learning import lessons as L

MAX_APPLIED = 4

# Function words AND the lesson schema's own vocabulary. Overlap is a proxy
# for "same subject", so a lesson could match an unrelated request through
# boilerplate: "write a Python function that reverses a string" matched a
# trade-off lesson on the word "that" (2026-10-31 live audit, case A6), and
# the same lesson reached "what is a checksum?" through "which"/"actual".
# Only words that name WHAT the lesson is about may count.
_STOPWORDS = frozenset("""
about above after again also and answer answered answered any are been
before being both but can could did does doing done down each either else
ensure enough even every from further had has have having here how into
its itself just like made make many more most much must need needed next
named names neither never none not now off once only other our ours out
over own same shall she should since some such than that the their theirs
them then there these they this those through time too under until very
was were what when where which while who whom whose why will with within
without would your yours user users result results earlier missing
plainly supported ones covers ensures""".split())


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (text or "").lower())
            if w not in _STOPWORDS}


def _lesson_words(lesson: Dict[str, Any]) -> set:
    """Words that describe the lesson's SUBJECT.

    `validation` is deliberately excluded: it is a QA procedure ("check that
    the next answer covers ..."), and its generic verbs made a lesson match
    requests about anything containing the word "check".
    """
    words = set()
    for item in lesson.get("evidence") or []:
        if isinstance(item, dict):
            for keyword in item.get("keywords") or []:
                words.update(_words(str(keyword)))
    text = (f"{lesson.get('problem')} {lesson.get('corrective_behavior')}")
    words.update(_words(text))
    return words


def _temporary(lesson: Dict[str, Any]) -> bool:
    """A one-off instruction ("just this once") never reaches a new task.

    Stored for the record and visible in the Control Center, but applying it
    to the next unrelated request is exactly the overcorrection this milestone
    forbids.
    """
    if lesson.get("temporary"):
        return True
    detail = (lesson.get("calibration") or {}).get("scope_detail") or {}
    return str(detail.get("kind") or "") == "single_task"


def _overlaps(lesson_words: set, question_words: set) -> int:
    """Count shared words, matching on a 4-character stem.

    "comparison" and "compare" are the same requirement, and a plain set
    intersection would treat them as unrelated — which matters because a
    content lesson is only applied to a related request.
    """
    stems = {w[:4] for w in question_words}
    return sum(1 for w in lesson_words if w in question_words or w[:4] in stems)


def _impact(lesson: Dict[str, Any]) -> str:
    return str((lesson.get("calibration") or {}).get("impact") or "")


def score(lesson: Dict[str, Any], capability: str, question: str,
          require_relevance: bool = False) -> float:
    """Relevance of one lesson for this request (0 when it must not apply).

    Two kinds of lesson behave differently by design:

      * a PREFERENCE or CONSTRAINT is a standing instruction inside its scope
        ("use numbered steps", "leave the original dates alone"), so it
        applies to anything in that scope;
      * a QUALITY lesson is about the CONTENT of one kind of answer ("include
        trade-offs with figures"). Applying it to an unrelated question
        produces a rule-shaped non-answer — the 2026-10-31 live check had
        "include trade-offs" injected into "what does HTTP 429 mean". It is
        therefore applied only when the request shares a word with it.
    """
    if _temporary(lesson):
        return 0.0
    lesson_cap = str(lesson.get("capability") or "any")
    if lesson_cap != "any" and capability and lesson_cap != capability:
        return 0.0
    overlap = _overlaps(_lesson_words(lesson), _words(question))
    if (require_relevance and not overlap and lesson_cap != "any"
            and _impact(lesson) == C.QUALITY):
        return 0.0
    base = 1.0 if lesson_cap == capability else 0.6
    return base + min(overlap, 5) * 0.5 + float(lesson.get("confidence") or 0)


def select(lessons: List[Dict[str, Any]], capability: str,
           question: str = "", limit: int = MAX_APPLIED
           ) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for lesson in lessons or []:
        if lesson.get("status") not in (L.ACTIVE, None):
            continue
        value = score(lesson, capability, question, require_relevance=True)
        if value <= 0:
            continue
        scored.append((value, str(lesson.get("created_at") or ""), lesson))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [lesson for _value, _at, lesson in scored[:limit]]


def applicable(state: Dict[str, Any], capability: str,
               question: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lessons for this state's task (reads `state["lessons"]` only).

    The runner loads the active set once per run; a node never reads the
    store itself, so what a node can apply is exactly what the run was given
    and tests can inject a precise set.
    """
    lessons = state.get("lessons") if isinstance(state, dict) else None
    if not isinstance(lessons, list):
        return []
    return select(lessons, capability,
                  question if question is not None
                  else str(state.get("question") or ""))


def notes(state: Dict[str, Any], capability: str,
          question: Optional[str] = None) -> str:
    """Prompt block for this task ('' when no lesson applies)."""
    return L.render(applicable(state, capability, question))


def applied_ids(state: Dict[str, Any], capability: str,
                question: Optional[str] = None) -> List[str]:
    return [str(lesson.get("lesson_id"))
            for lesson in applicable(state, capability, question)]
