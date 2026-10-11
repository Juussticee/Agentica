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
