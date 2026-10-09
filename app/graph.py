"""Graph wiring (spec Section 3): START -> supervisor -> agents -> END.

Phase 2: the feasibility route runs the spec's sub_routes chain first
(research -> books -> feasibility), so both prior answers are in state when
the feasibility agent synthesizes the study. Single-agent routes end directly.
"""

from langgraph.graph import END, START, StateGraph

from app.agents.books_agent import books_agent
from app.agents.erp_agent import erp_agent
from app.agents.feasibility_agent import feasibility_agent
from app.agents.research_agent import research_agent
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
        "research", _after_research, {"books": "books", END: END}
    )
    graph.add_conditional_edges(
        "books", _after_books, {"feasibility": "feasibility", END: END}
    )
    graph.add_edge("feasibility", END)
    graph.add_edge("erp", END)

    return graph.compile()
