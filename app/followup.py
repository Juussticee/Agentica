"""Contextual follow-up: answer a request from the RECORDED earlier result.

This is the first capability in Agentica that consumes prior work instead of
starting from nothing. A request such as "What about the one you recommended?"
refers back to an answer the user already received; answering it from model
knowledge invents a recommendation that may never have been given.

CONTRACT
  ACCEPTS  an anaphoric reference to earlier work (routing decides this:
           app/supervisor._FOLLOWUP_RE) AND a prior completed execution that
           the runner loaded from the execution store.
  REFUSES  inventing that earlier work. With no recorded prior execution the
           answer says so plainly and asks for the details; it never guesses
           what "the one" was.
  READS    state["prior_execution"] — {id, question, answer, completed_at},
           written ONLY by GraphRunner.run from the store. This module never
           reads the store itself, so it cannot see anything the run was not
           given, and tests can inject the record directly.
  WRITES   answer / sources / agent_answers["followup"] (the existing
           with_agent_output channel — no new state keys).
  PROVENANCE ["execution:<id>"], so the answer names the record it used and
           QA/UI can attribute it. No source is claimed when no record exists.

The model is used ONLY to compose the answer from the supplied record; it is
never asked what happened. When the provider is unavailable the recorded
answer is returned verbatim with an honest note, so the capability degrades to
"show me what I recorded" instead of failing.
"""

from typing import Any, Dict, List

from app.state import AgentState, with_agent_output

# The recorded answer is quoted to the model when composing; a huge earlier
# answer is cut with an explicit marker rather than silently truncated.
MAX_RECORD_CHARS = 6000

NO_PRIOR_ANSWER = (
    "I don't have an earlier result to refer to: no completed execution is "
    "recorded in this Control Center, and I don't keep conversation history "
    "of my own. Send the question or the details again and I'll run it."
)

_COMPOSE_PROMPT = """You are answering a FOLLOW-UP question from a user.

Use ONLY the earlier result below. Do not add facts that are not in it, and
do not claim any new work was performed: you did not search, retrieve, or
compute anything for this reply.

If the earlier result does not contain what the follow-up asks for, say so
plainly and name what is missing.

Earlier request:
{prior_question}

Earlier result:
{prior_answer}

Follow-up question:
{question}

Answer:"""


def _recorded_answer(prior: Dict[str, Any]) -> str:
    """The earlier answer, cut with an explicit marker when oversized."""
    text = str(prior.get("answer") or "").strip()
    if len(text) <= MAX_RECORD_CHARS:
        return text
    return text[:MAX_RECORD_CHARS] + "\n[…earlier answer truncated for this reply]"


def _format_prior(prior: Dict[str, Any]) -> str:
    return (
        f"Earlier execution: #{prior.get('id')}\n"
        f"Earlier request: {str(prior.get('question') or '').strip()}\n"
        f"Earlier result:\n{_recorded_answer(prior)}"
    )


def followup_node(state: AgentState) -> dict:
    """Graph node for the `followup` route (see the module contract)."""
    question = str(state.get("question") or "").strip()
    prior = state.get("prior_execution")
    if not isinstance(prior, dict) or not str(prior.get("answer") or "").strip():
        # Nothing recorded: refuse to invent, and claim no source.
        return with_agent_output(state, "followup", NO_PRIOR_ANSWER, [])

    sources: List[str] = [f"execution:{prior.get('id')}"]
    answer = ""
    try:
        from app.llm_client import get_llm, llm_text

        answer = str(llm_text(get_llm().invoke(
            _COMPOSE_PROMPT.format(
                prior_question=str(prior.get("question") or "").strip(),
                prior_answer=_recorded_answer(prior),
                question=question,
            )
        )) or "").strip()
    except Exception as exc:  # noqa: BLE001 — fail-soft, never fabricates
        answer = (
            "I could not compose a follow-up answer right now (the AI "
            f"provider is unavailable: {type(exc).__name__}). Here is the "
            "result I recorded earlier, verbatim:\n\n" + _format_prior(prior)
        )

    if not answer:
        answer = (
            "I could not compose a follow-up answer, so here is the result I "
            "recorded earlier, verbatim:\n\n" + _format_prior(prior)
        )

    return with_agent_output(state, "followup", answer, sources)
