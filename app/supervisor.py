"""Supervisor: decides routing only — it NEVER calls an agent node directly.

Phase 1: keyword-based routing (books-related words -> books, else -> research).
Phase 2: feasibility-style questions matched first (sub_routes chain).
Phase 3: a tight ERP keyword list routes to the erp agent. Checked BEFORE
books because ERP terms (invoices, inventory, ...) are specific while book
questions usually self-identify ("book", "manual", "chapter"). This keyword
routing is provisional; Phase 4 replaces it with an LLM classification call;
the function signature and return contract stay identical so nothing
downstream changes.
"""

import re

from app.state import AgentState

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


def supervisor(state: AgentState) -> dict:
    """Return only the keys this node changes: route (and sub_routes if multi)."""
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
