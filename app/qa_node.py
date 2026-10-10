"""Production QA node (Phase 5): deterministic workflow validation.

Moved from the Phase 4 test-suite copy (scripts/test_phase4_workflow_qa.py)
into production per docs/MASTER_PLAN.md Phase 5: qa_flags was declared in
AgentState but never written by production code.

Responsibilities (frozen by the 30 Phase 4 QA tests):
  * evaluates whether the completed workflow produced the required output
  * checks route-aware expectations
  * never decides routing / never invokes agents / never replaces the
    supervisor or the graph / never uses an LLM
"""

from typing import List

from app.state import AgentState


def run_qa(state: dict) -> List[str]:
    """Deterministic QA. Takes the completed workflow output and flags issues."""
    flags = []
    route = state.get("route", "research")
    sources = state.get("sources") or []
    answers = state.get("agent_answers") or {}

    if route in ("feasibility", "multi"):
        # Feasibility and multi-agent routes require research + books.
        has_research = "research" in sources or "research" in answers
        has_books = "books" in sources or "books" in answers
        if not has_research:
            flags.append("QA: feasibility route missing required research output")
        if not has_books:
            flags.append("QA: feasibility route missing required books output")

        missing = [name for name in ("research", "books") if name not in answers]
        if missing:
            flags.append(
                "QA: feasibility route incomplete — missing upstream agent output(s): "
                + ", ".join(missing)
            )
    elif route == "erp":
        # ERP route must NOT be forced to need research/books.
        if "erp" not in sources:
            flags.append("QA: ERP route missing required erp provenance")

    if not str(state.get("answer") or "").strip():
        flags.append("QA: final answer is empty — workflow produced no output")

    return flags


def qa_node(state: AgentState) -> dict:
    """Graph node wrapper: append QA findings to state["qa_flags"].

    Merges with any existing flags (never clears accumulated ones), returns
    only the key it changes, and is fail-soft: a QA crash can never fail a
    run — it appends a QA error note instead (MASTER_PLAN Phase 5 failure
    case).
    """
    flags = state.get("qa_flags") or []
    try:
        findings = run_qa(state)
    except Exception as exc:  # fail-soft by design
        findings = [f"QA: internal error while validating output ({exc})"]
    return {"qa_flags": list(flags) + list(findings)}
