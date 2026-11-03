"""The `feedback` route (2026-10-30): correct now, learn for later.

WHAT THE NODE DOES, IN ORDER
  1. read the correction deterministically (signals.detect) and the recorded
     earlier result the user is talking about;
  2. compose an IMPROVED answer to the current request from that recorded
     result plus the correction — naming what it could not add, because this
     reply runs no tools and performs no new research;
  3. record a structured lesson (lessons.record) with the earlier execution as
     provenance, and — only when the deterministic gate passes — refresh the
     learned skill version for that capability;
  4. report what was learned in `state["learning"]` so the run, the event log
     and the Control Center all say the same thing.

WHAT IT CANNOT DO
It never re-runs the original task, never claims a search or a computation
happened, and never modifies a skill that fails validation (skills.activate
re-evaluates from stored text). With no recorded prior execution it says so
and still records the lesson, because the correction is about behaviour.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.execution.events import LESSON_RECORDED, SKILL_VERSION
from app.learning import lessons as L
from app.learning import retrieval, signals, skills
from app.state import AgentState, with_agent_output

MAX_PRIOR_CHARS = 6000

# The revision is one model call inside a synchronous run: bound it, so an
# unresponsive provider falls back to the deterministic reply instead of
# leaving the execution running (2026-10-31).
COMPOSE_TIMEOUT = 90.0

# Words that describe the complaint rather than its subject. They are ignored
# when checking whether the revised answer actually covers what was missing.
_COVERAGE_STOPWORDS = frozenset({
    "with", "that", "this", "from", "your", "have", "were", "been", "more",
    "into", "they", "their", "than", "then", "also", "just", "very",
    "next", "time", "real", "short", "actual", "properly", "done", "make",
    "need", "needs", "wanted", "should", "would", "could", "each", "some",
})


def _uncovered(missing: List[str], answer: str) -> List[str]:
    """Named items the revised answer does not visibly mention.

    The feedback route cannot re-run the task, so it must not claim it added
    something it did not — but it also must not claim a requirement is still
    missing when the revision in front of the user contains it. Coverage is a
    plain token check over the answer, deliberately conservative.
    """
    text = str(answer or "").lower()
    if not text.strip():
        return list(missing)
    out: List[str] = []
    for item in missing or []:
        words = {w for w in re.findall(r"[a-z0-9]{4,}", str(item).lower())
                 if w not in _COVERAGE_STOPWORDS}
        if not words or not any(word in text for word in words):
            out.append(item)
    return out

NO_PRIOR_ANSWER_NO_RECORD = (
    "I don't have the earlier result in front of me (no completed "
    "execution is recorded in this Control Center), and I could not record "
    "this correction for future tasks. Send the request again with this "
    "requirement stated and I'll run it with it applied."
)
NO_PRIOR_ANSWER = (
    "Thanks — I've recorded that correction so comparable tasks follow it. "
    "I don't have the earlier result in front of me (no completed execution "
    "is recorded in this Control Center), so I can't revise it here. Send the "
    "request again and I'll run it with this requirement applied."
)

# Frustration alone is not a rule. Saying so (and asking for the missing
# detail) is better than inventing a standing restriction from a complaint.
ASK_WHAT_WAS_WRONG = (
    "I hear that this didn't work for you, and I haven't changed my behaviour "
    "over it — I can't turn dissatisfaction into a rule without knowing what "
    "to change. Tell me the specific thing that was wrong or missing (for "
    "example “the dates should have been preserved” or “I needed benchmarks "
    "and licensing”) and I'll fix this answer and remember it for comparable "
    "tasks. If it was a temporary wish, say so and I'll apply it to this task "
    "only."
)

# The RULES half only. It is passed to `strip_instruction_echo` as the text a
# reply must not parrot — never the whole prompt, because the earlier answer
# and the learned block are content the revised answer may legitimately
# restate (2026-10-31: passing the whole prompt made the benchmark's learned
# behaviour line look like an echoed instruction and removed it).
_COMPOSE_RULES = """You are revising an earlier answer after the user told you
what was wrong with it.

Rules:
- You ran no tools for this reply. Do not claim or imply a new search,
  retrieval, computation or file change happened.
- Fix what the user named. Where the earlier result does not contain the
  missing material and you are adding general knowledge, mark it as such.
- If something cannot be supplied without running the task again, say so in
  one sentence at the end.
- Keep the format the user asked for."""

_COMPOSE_PROMPT = _COMPOSE_RULES + """

Earlier request: {prior_question}
Earlier answer:
{prior_answer}

What the user said was wrong or missing: {correction}
Items they named: {missing}
{learned}

Revised answer:"""


def _scope():
    """The run's store binding, or None outside a run.

    Reuses the existing run-scoped `(store, execution_id)` context bound by
    GraphRunner (app.graph.approval_scope) instead of introducing a second
    mechanism: inside a run there is exactly one store handle a node may
    touch, and outside a run there is none.
    """
    try:
        from app.graph import current_approval_scope

        return current_approval_scope()
    except Exception:  # noqa: BLE001
        return None


def _prior_capability(store, prior: Dict[str, Any]) -> str:
    """The capability the earlier answer belongs to (the lesson's scope).

    A chain of corrections must stay scoped to the work being corrected:
    when the previous run was itself a `feedback`/`followup` execution, its
    route says nothing about the subject, so the capability of the lesson it
    recorded is used instead. Without this, correcting a correction would
    scope the second lesson to `feedback` and never apply either to research.
    """
    prior_id = prior.get("id")
    if store is None or not prior_id:
        return ""
    route = ""
    try:
        row = store.get_execution(str(prior_id))
        route = str((row or {}).get("route") or "")
    except Exception:  # noqa: BLE001 — scope falls back to the signal
        return ""
    if route not in ("feedback", "followup"):
        return route
    # The lesson a feedback execution recorded names the capability the chain
    # is about; its own event is the durable link to it (the lesson's
    # source_execution_id points at the ORIGINAL execution, not at the
    # correction).
    try:
        for event in store.get_events(str(prior_id)):
            if event.get("type") != LESSON_RECORDED:
                continue
            lesson_id = (event.get("payload") or {}).get("lesson_id")
            lesson = store.get_lesson(str(lesson_id)) if lesson_id else None
            if lesson:
                return str(lesson.get("capability") or "")
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _emit(event_type: str, message: str, payload: Dict[str, Any]) -> None:
    try:
        from app.execution.instrumentation import emit

        emit(event_type, message, agent_id="feedback", step_id="learning",
             payload=payload)
    except Exception:  # noqa: BLE001 — events are never load-bearing
        pass


def _compose(prior: Dict[str, Any], question: str,
             signal: signals.FeedbackSignal,
             learned: str, recorded: bool = True) -> str:
    answer = ""
    prompt = ""
    try:
        from app.llm_client import get_llm, llm_text, strip_instruction_echo

        prompt = _COMPOSE_PROMPT.format(
            prior_question=str(prior.get("question") or "").strip(),
            prior_answer=(str(prior.get("answer") or "").strip()
                          [:MAX_PRIOR_CHARS]),
            correction=signal.text,
            missing=", ".join(signal.missing) or "none stated",
            learned=learned,
        )
        answer = strip_instruction_echo(
            str(llm_text(
                get_llm(timeout=COMPOSE_TIMEOUT).invoke(prompt)) or "").strip(),
            _COMPOSE_RULES)
    except Exception as exc:  # noqa: BLE001 — fail-soft, never fabricates
        answer = ""
        note = f" (the AI provider is unavailable: {type(exc).__name__})"
    else:
        note = ""
    if not answer:
        # Never claim the correction was saved when the store was unavailable:
        # the fallback wording follows what actually happened.
        opener = (
            "I've recorded the correction so comparable tasks follow it"
            if recorded else
            "I understood the correction but could not record it for future "
            "tasks (no execution store in this run)"
        )
        answer = (
            opener
            + note + ". I could not compose a revised answer just now; here "
            "is what I understood: " + _explain(signal) + "\n\nEarlier "
            "result, verbatim:\n\n" + str(prior.get("answer") or "").strip()
        )
    named = L.named_items(signal.missing)
    if named:
        outstanding = _uncovered(named, answer)
        if outstanding:
            answer += ("\n\nNot covered here (needs the task re-run): "
                       + ", ".join(outstanding) + ".")
    return answer


def _explain(signal: signals.FeedbackSignal) -> str:
    base = {
        signals.INSUFFICIENT: "the earlier result did not go far enough",
        signals.INCORRECT: "something in the earlier result was wrong",
        signals.MISUNDERSTOOD: "I answered a different question than the one "
                              "you asked",
        signals.FORMAT: "the output format was not what you wanted",
        signals.FUTURE_DIRECTIVE: "you want this handled a specific way next "
                                  "time",
        signals.REPEATED: "this is a mistake I have made before",
    }.get(signal.kind, "the earlier result needed fixing")
    named = L.named_items(signal.missing)
    if named:
        return base + "; you named: " + ", ".join(named)
    return base


def feedback_node(state: AgentState) -> dict:
    """Graph node for the `feedback` route (see the module contract)."""
    question = str(state.get("question") or "").strip()
    prior = state.get("prior_execution")
    prior = prior if isinstance(prior, dict) else {}
    scope = _scope()
    store = scope[0] if scope else None
    execution_id = scope[1] if scope else None

    capability = _prior_capability(store, prior)
    signal = signals.detect(question, capability=capability)
    if signal is None:  # routed here but not shaped like feedback: be honest
        return with_agent_output(
            state, "feedback",
            "I read that as a correction but could not tell what should "
            "change. Tell me what was wrong or missing and I'll apply it.")
    if not capability and signal.capability != "any":
        capability = signal.capability

    learning: Dict[str, Any] = {
        "signal": signal.kind,
        "category": signal.category,
        "capability": capability or signal.capability,
        "missing": L.named_items(signal.missing),
        "learned": False,
        "reason": "no store in this run",
    }
    sources: List[str] = []
    from app.learning import writes_allowed

    if not writes_allowed():
        # Observation mode (evaluation harness): the answer is still revised,
        # but nothing is recorded and the reply says exactly that.
        learning["reason"] = ("observation mode: lessons are not recorded "
                             "during an evaluation run")
    elif store is not None:
        try:
            outcome = L.record(
                store, signal,
                source_execution_id=prior.get("id"),
                prior=prior,
            )
            lesson = outcome["lesson"]
            if outcome["action"] == "not_learned":
                learning.update({
                    "learned": False,
                    "action": "not_learned",
                    "feedback_type": outcome["decision"].get("feedback_type"),
                    "reason": outcome["decision"].get("decision_reason"),
                    "decision": outcome["decision"],
                })
            else:
                learning.update({
                    "learned": True,
                    "action": outcome["action"],
                    "lesson_id": lesson.get("lesson_id"),
                    "status": lesson.get("status"),
                    "feedback_type": outcome["decision"].get("feedback_type"),
                    "impact": outcome["decision"].get("impact"),
                    "temporary": outcome["decision"].get("temporary"),
                    "decision": outcome["decision"],
                })
            sources.append(f"learning:{lesson.get('lesson_id')}")
            _emit(LESSON_RECORDED,
                  f"Lesson {lesson.get('lesson_id')} {lesson.get('status')}",
                  {"lesson_id": lesson.get("lesson_id"),
                   "status": lesson.get("status"),
                   "capability": lesson.get("capability"),
                   "action": outcome["action"],
                   "reason": outcome["decision"].get("decision_reason")})
            # A learned procedure version is refreshed only for a capability
            # that owns a skill AND has enough active lessons to encode.
            capability_skill = skills.skill_for(capability)
            if capability_skill:
                active = [l for l in L.active_lessons(store, capability)
                          if l.get("capability") == capability]
                if len(active) >= 2:
                    improved = skills.improve_from_lessons(store, capability)
                    learning["skill"] = {
                        "skill_id": capability_skill,
                        "activated": bool(improved.get("activated")),
                        "version": (improved.get("version") or {}).get("version"),
                        "reason": improved.get("reason"),
                    }
                    _emit(SKILL_VERSION,
                          f"Skill {capability_skill} "
                          + ("activated" if improved.get("activated")
                             else "candidate rejected"),
                          learning["skill"])
        except Exception as exc:  # noqa: BLE001 — feedback must still answer
            learning.update({"learned": False,
                             "reason": f"storage error: {type(exc).__name__}"})

    # Compose AFTER recording, so the answer can state what was actually
    # stored (and cannot claim a save that failed). The prior learned set is
    # still the run's `lessons`; the lesson just stored is new and is not
    # applied to the same request that produced it.
    learned_notes = L.render(retrieval.applicable(state, capability or "any",
                                                  question))
    recorded = bool(learning.get("learned"))
    if learning.get("action") == "not_learned":
        update = with_agent_output(state, "feedback", ASK_WHAT_WAS_WRONG, [])
        update["learning"] = learning
        return update
    if str(prior.get("answer") or "").strip():
        answer = _compose(prior, question, signal, learned_notes,
                          recorded=recorded)
    else:
        answer = NO_PRIOR_ANSWER if recorded else NO_PRIOR_ANSWER_NO_RECORD

    update = with_agent_output(state, "feedback", answer, sources)
    update["learning"] = learning
    return update
