"""Graph wiring (spec Section 3): START -> supervisor -> agents -> qa -> END.

Phase 2: the feasibility route runs the spec's sub_routes chain first
(research -> books -> feasibility), so both prior answers are in state when
the feasibility agent synthesizes the study. Single-agent routes end directly.
Phase 5: every route's terminal hop passes through the deterministic QA
gate, which only appends state["qa_flags"] (never blocks, never routes).
"""

from langgraph.graph import END, START, StateGraph

from app.agents.books_agent import books_agent
from app.agents.erp_agent import erp_agent
from app.agents.feasibility_agent import feasibility_agent
from app.agents.research_agent import research_agent
from app.qa_node import qa_node
from app.state import AgentState
from app.supervisor import supervisor

# supervisor -> first node per route; feasibility enters via its sub_routes
# chain (research first, per the feasibility_agent contract).
_ROUTES = {
    "books": "books",
    "research": "research",
    "erp": "erp",
    "feasibility": "research",
}


def _after_research(state: AgentState):
    """Chain hand-off inside feasibility; direct research route ends here."""
    return "books" if state.get("route") == "feasibility" else END


def _after_books(state: AgentState):
    """Chain hand-off inside feasibility; direct books route ends here."""
    return "feasibility" if state.get("route") == "feasibility" else END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("books", books_agent)
    graph.add_node("research", research_agent)
    graph.add_node("erp", erp_agent)
    graph.add_node("feasibility", feasibility_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state.get("route", "research"),
        _ROUTES,
    )
    graph.add_conditional_edges(
        "research",
        _after_research,
        # Phase 5: direct-research terminal passes through the QA gate.
        {"books": "books", END: "qa"},
    )
    graph.add_conditional_edges(
        "books",
        _after_books,
        # Phase 5: direct-books terminal passes through the QA gate.
        {"feasibility": "feasibility", END: "qa"},
    )
    graph.add_edge("feasibility", "qa")
    graph.add_edge("erp", "qa")
    graph.add_edge("qa", END)

    graph.add_node("qa", qa_node)

    return graph.compile()
