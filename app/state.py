"""AgentState — the single state contract passed between LangGraph nodes."""

from typing import Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    route: str              # "books" | "research" | "feasibility" | "erp" | "multi"
    sub_routes: List[str]   # when route == "multi" (feasibility sets ["research", "books"])
    sources: List[str]
    answer: str
    qa_flags: List[str]     # conflicts/uncertainty noted by the QA node (Phase 4)

    # Phase 7 addition: supervisor marks genuinely multi-objective requests
    # (never written False, so pre-Phase-7 states unchanged).
    complex: bool
    # Phase 2 addition: per-agent answers, so a downstream agent (feasibility,
    # later qa_node) can consume multiple prior outputs from one state.
    agent_answers: Dict[str, str]

    # Phase 7 (MASTER_PLAN §5.2): the structured plan produced by the planner
    # node — the SOLE channel between planner and (Phase 8) executor nodes;
    # justified in MASTER_PLAN §5.2. It is plan DATA, not executed results.
    # Absent on simple/multi routes, so all pre-Phase-7 behavior is unchanged.
    plan: Optional[dict]

    # 2026-10-25 (orchestrated knowledge fallback): present ONLY on a books
    # refusal, this structured diagnosis — {reason, score, nodes, recovery},
    # scalars only, never passage text — is what the orchestrator's
    # books->research fallback decision reads (app.supervisor.
    # fallback_decision). A similarity score or the refusal sentence alone
    # is deliberately NOT the decision input. Absent on every answered run.
    books_refusal: Optional[dict]

    # GUI-10 (2026-10-28): present ONLY when a run resumes an already
    # DECIDED approval — {approval_id: str}. The research gate revalidates
    # it against the persisted approval (status, expiry, bound question /
    # route / diagnosis, single-execution claim) before running anything;
    # without it the gate never executes, it only creates/reuses a pending
    # request. Set by GraphRunner.resume_approval, never by a graph node.
    research_approval: Optional[dict]

    # Contextual follow-up (2026-10-30): set by GraphRunner.run BEFORE the
    # graph starts — the most recent COMPLETED execution that actually
    # produced an answer ({id, question, answer, completed_at}), or absent
    # when there is none. Only the `followup` route reads it; no node writes
    # it, so it can never be invented by a model or a specialist.
    prior_execution: Optional[dict]

    # Workspace coding (2026-10-30): written ONLY by the `coding` node —
    # {status: inspected | patch_prepared | patch_blocked | no_target |
    #  unavailable, workspace, files, approval_id?, validation?, digest?}.
    # `patch_prepared` means a REVIEWED patch is persisted as an approval
    # request; it never means a file was modified.
    coding: Optional[dict]

    # Continuous learning (2026-10-30): set by GraphRunner.run BEFORE the graph
    # starts — the ACTIVE lessons read from the store, so a node applies
    # exactly what the run was given and tests can inject a precise set. No
    # node writes it.
    lessons: Optional[list]
    # Written ONLY by the `feedback` node: {signal, category, capability,
    # missing, learned, action?, lesson_id?, status?, decision?, skill?}. It
    # reports what was actually recorded (or why nothing was), never a claim
    # that behaviour changed.
    learning: Optional[dict]
    # Written by nodes that actually inserted lessons into a prompt (the
    # lesson ids), so the runner can count their use. Never set otherwise.
    lessons_applied: Optional[list]

    # Evidence grounding (2026-11-01): written ONLY by the `general` node —
    # {status: grounded|partial|none|error, answered_from: evidence|model|error,
    #  passages, best_score, coverage, missing, conflict, reason, grounding,
    #  claims, files, used, text}. `text` is the consulted passage text, used
    # by QA to check figures and licence claims; it stays in memory only (the
    # runner does not persist it), while `files`/`used` are the citation
    # allow-list. Declared here because LangGraph persists exactly the keys
    # the state type declares: an undeclared channel is silently DROPPED
    # between the node and QA (found in the 2026-11-01 live run — the licence
    # check never saw the evidence record and stayed silent).
    evidence: Optional[dict]

    # Books-route evidence (2026-11-02): written ONLY by the `books` node —
    # the passages the query engine actually consulted, in the same shape the
    # general route uses, so the runner persists both with one snapshot
    # writer. A SEPARATE key by design: `evidence` drives QA's general-route
    # claim checks, and a second answer path must not change that behavior.
    books_evidence: Optional[dict]


def with_agent_output(state: AgentState, agent: str, answer: str, sources: List[str]) -> dict:
    """Build a node update that records the agent's answer and appends provenance.

    Nodes return only the keys they change (spec 0); this helper merges prior
    state so chained agents (feasibility route) accumulate rather than overwrite.
    """
    prior_sources = list(state.get("sources") or [])
    prior_answers = dict(state.get("agent_answers") or {})
    prior_answers[agent] = answer
    return {
        "answer": answer,
        "sources": prior_sources + list(sources),
        "agent_answers": prior_answers,
    }
