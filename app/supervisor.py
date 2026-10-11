"""Supervisor: decides routing only — it NEVER calls an agent node directly.

Phase 1: keyword-based routing (books-related words -> books, else -> research).
Phase 2: feasibility-style questions matched first (sub_routes chain).
Phase 3: a tight ERP keyword list routes to the erp agent. Checked BEFORE
books because ERP terms (invoices, inventory, ...) are specific while book
questions usually self-identify ("book", "manual", "chapter").

Phase 6: optional LLM-assisted classification behind ROUTING_MODE=llm
(config; default "keyword" keeps byte-identical behavior). The LLM label is
restricted to the four valid routing classes; any LLM error, malformed
output, or unknown label falls back to the keyword result — default mode
never depends on a working AI provider. The function signature and return
contract stay identical so nothing downstream changes.

Phase 7: additionally marks genuinely multi-objective requests as
"complex" (new key: \'complex\': bool — only ever True, never written False
so pre-Phase-7 test states are untouched). The planner (app/planner.py)
decides whether to actually run via PLANNER_MODE; the supervisor only
detects the need so the graph can take one conditional hop.
"""

import re

from app.config import ROUTING_MODE
from app.planner import needs_planning
from app.state import AgentState

_ROUTING_PROMPT = """Classify the user question into exactly one of these labels:
books: the question can be answered from the user's own book library
research: the question needs general web research
feasibility: the question asks for a business feasibility study / go-no-go analysis
erp: the question is about ERP business data (invoices, inventory, orders, customers)

Answer with ONLY the label (one word).

Question: {question}

Label:"""

_VALID_LLM_ROUTES = {"books", "research", "feasibility", "erp"}

_BOOKS_RE = re.compile(
    r"\b(books?|chapter|pages?|page \d|textbook|handbook|manual|guide|"
    r"author|bibliography|section|orbital|orbit|spacecraft|speed|altitude|"
    r"bee|beekeeping|urban|hive|queen|drone|nectar|pollen|swarm)\b",
    re.IGNORECASE,
)

_FEASIBILITY_RE = re.compile(
    r"\b(feasibility|feasible|business case|go/no-go|go no go|"
    r"market study|market analysis|risk assessment|venture)\b",
    re.IGNORECASE,
)

_ERP_RE = re.compile(
    r"\b(erp|inventory|stock|invoices?|purchase orders?|sales orders?|"
    r"customer records?|vendor records?|procurement)\b",
    re.IGNORECASE,
)


def _llm_route(question: str) -> str | None:
    """Ask the LLM for a routing label; return None on ANY failure.

    Failures are swallowed here by design (fail-soft): the caller falls back
    to keyword routing, so a missing/broken/billing-dead provider can never
    crash routing.
    """
    try:
        from app.llm_client import get_llm, llm_text

        response = get_llm(fast=True).invoke(
            _ROUTING_PROMPT.format(question=question)
        )
        label = llm_text(response).strip().lower()
    except Exception:
        return None
    return label if label in _VALID_LLM_ROUTES else None


def keyword_route(state: AgentState) -> dict:
    """Deterministic keyword routing (Phases 1–5 behavior, frozen by tests)."""
    question = state.get("question", "")
    if _FEASIBILITY_RE.search(question):
        # Spec (feasibility_agent): Supervisor sets sub_routes=["research", "books"]
        # first so both agents produce prior answers in state.
        return {"route": "feasibility", "sub_routes": ["research", "books"]}
    if _ERP_RE.search(question):
        return {"route": "erp"}
    if _BOOKS_RE.search(question):
        return {"route": "books"}
    return {"route": "research"}


def supervisor(state: AgentState) -> dict:
    """Return only the keys this node changes: route (and sub_routes if multi).

    ROUTING_MODE="keyword" (default): pure keyword routing, unchanged.
    ROUTING_MODE="llm": consult the LLM first; deterministic keyword result is
    the fallback whenever the LLM fails, returns garbage, or an unknown label.

    Phase 7: also sets "complex": True when app.planner.needs_planning()
    detects a multi-objective request (planner gating itself stays behind
    PLANNER_MODE in app/planner.py — keyword routing values are unchanged).
    """
    question = state.get("question", "")

    base = None
    if ROUTING_MODE == "llm":
        label = _llm_route(question)
        if label == "feasibility":
            base = {"route": "feasibility", "sub_routes": ["research", "books"]}
        elif label in _VALID_LLM_ROUTES:
            base = {"route": label}
        # LLM missing/malformed/unknown -> deterministic keyword fallback.

    if base is None:
        base = keyword_route(state)

    if needs_planning(state):
        base["complex"] = True
    return base
