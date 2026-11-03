"""Phase 1 routing test — proves supervisor decisions and graph wiring without
any LLM call: both agent nodes are faked before build_graph(), then two
questions flow through the real compiled graph (supervisor -> route -> node).

Pass criteria mirror spec step 6: correct routing for books-style and
research-style questions, and each agent's answer/source keys merged into state.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Fail-closed Chroma isolation: arm the guard at startup so a direct
# `python scripts\test_phase1_routing.py` run is protected exactly like a
# pytest run (no-op plan: this suite never opens Chroma itself).
from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()


def main():
    import app.graph as graph_module
    from app.supervisor import supervisor

    # --- supervisor unit checks (pure logic, no LLM) ---
    assert supervisor({"question": "What does chapter 3 say about propolis?"})["route"] == "books"
    assert supervisor({"question": "Find pages about Hohmann transfers in my books"})["route"] == "books"
    assert supervisor({"question": "What is the latest news on fusion energy startups?"})["route"] == "research"
    # 2026-10-28 (Priority A): a plain knowledge question no longer falls
    # through to the research DEFAULT — it is answered from model knowledge
    # (scripts/test_routing_policy.py owns the full table).
    assert supervisor({"question": "Who won the 2024 F1 championship?"})["route"] == "general"

    # --- document-deictic routing (2026-10-24 approved RAG fix) ---
    # Natural document-grounded questions must reach books WITHOUT artificial
    # trigger words (verified GUI failures: strict wordings misrouted to web
    # research because no keyword matched). Each case mirrors the actual
    # failure shapes from docs/GUI_RAG_ROOT_CAUSE_2026-10-24_2349.md.
    deictic_books = [
        # natural four-functions question (T1 without "section/page")
        "According to the document, what are the four core functions of the "
        "AI Risk Management Framework? Explain each briefly.",
        # the exact strict wording that misrouted live (T3)
        "Using only this document, explain the difference between the Map and "
        "Measure functions.",
        # evidence-from-the-pdf phrasing
        "Where in the PDF can the four functions of the AI RMF be found?",
        "Show me material in that document on trustworthiness traits.",
    ]
    for q in deictic_books:
        assert supervisor({"question": q})["route"] == "books", q

    # Precedence and fallback must stay intact:
    assert supervisor({
        "question": "Using only this document, do a feasibility study for the "
                    "beekeeping venture"
    })["route"] == "feasibility", "feasibility keeps precedence over deictic books"
    assert supervisor({
        "question": "Show me open invoices for this document"
    })["route"] == "erp", "erp keeps precedence over deictic books"
    # Ambiguous question: since 2026-10-28 (Priority A) it is answered from
    # model knowledge instead of costing a web search.
    assert supervisor({"question": "Tell me about functions."})["route"] == "general"
    assert supervisor({"question": "What is the latest on fusion?"})["route"] == "research"
    print("[supervisor] routing decisions OK (incl. document-deictic fix)")

    # --- corpus-identity anchors + current-info precedence (2026-10-25 fix cycle) ---
    # Verified failures from docs/GUI_RAG_FIX_VALIDATION_2026-10-25_1415.md
    # Phase 6B: A2/A3 (and the natural T6) contain no deictic wording and
    # misrouted to research. General behavior only — no exact-question rules:
    identity_books = [
        # A2 / A3 exact GUI wording
        "How does the framework explain the relationship between trustworthy "
        "AI characteristics and managing AI risks?",
        "Explain the four core functions of the AI Risk Management Framework "
        "and what each one is meant to accomplish.",
        # varied wording and paraphrases of the same intent
        "Explain trustworthy AI characteristics as the RMF describes them.",
        "What does the RMF say about measuring AI risk?",
        "Summarize the NIST guidance on governing AI systems.",
        "How do MAP and MEASURE relate according to the AI RMF?",
        # natural T6 paraphrase (misrouted live on 2026-10-25)
        "If a team wants to identify and understand the context of possible "
        "AI harms before measuring them, which AI RMF function is most "
        "relevant, and why?",
    ]
    for q in identity_books:
        assert supervisor({"question": q})["route"] == "books", q

    research_stays = [
        # current-information cues outrank corpus anchors (incl. one that
        # names NIST and the AI RMF — must stay research)
        "What is the latest news about the NIST AI framework?",
        "When is the next update to the AI RMF due today?",
        # current-information cues keep keyword-free requests on research
        "What is the weather in Berlin today?",
    ]
    for q in research_stays:
        assert supervisor({"question": q})["route"] == "research", q
    # 2026-10-28 (Priority A): these three named NO source cue, so they used
    # to hit the research default by accident. They are general-knowledge
    # questions now — the whole point of the policy change — and must never
    # become DOCUMENT questions either (that is what the anchors above guard).
    general_stays = [
        "Explain how transformer neural networks work in modern AI systems.",
        "What are people saying about AI art these days?",
        "Who won the 2024 F1 championship?",
    ]
    for q in general_stays:
        assert supervisor({"question": q})["route"] == "general", q
    # not every question naming a corpus owner becomes a document question
    # when it asks for current information — precedence pinned above.
    # Existing precedence must still hold over the new anchors:
    assert supervisor({
        "question": "Feasibility study of an AI RMF-based consulting venture"
    })["route"] == "feasibility"
    assert supervisor({
        "question": "Show me open invoices about trustworthy AI"
    })["route"] == "erp"
    print("[supervisor] corpus-identity anchors + current-info precedence OK "
          "(2026-10-25)")

    # --- end-to-end graph wiring with faked agent nodes ---
    calls = []

    def fake_books(state):
        calls.append("books")
        return {"answer": "FAKE_BOOKS_ANSWER", "sources": ["books"]}

    def fake_research(state):
        calls.append("research")
        return {"answer": "FAKE_RESEARCH_ANSWER", "sources": ["research", "https://example.com"]}

    graph_module.books_agent = fake_books
    graph_module.research_agent = fake_research

    graph = graph_module.build_graph()

    r1 = graph.invoke({"question": "Summarize chapter 2 of the beekeeping manual"})
    assert calls[-1] == "books", "books-style question must reach the books node"
    assert r1["route"] == "books" and r1["answer"] == "FAKE_BOOKS_ANSWER"
    assert r1["sources"] == ["books"]

    r2 = graph.invoke({"question": "What happened in AI research this week?"})
    assert calls[-1] == "research", "research-style question must reach the research node"
    assert r2["route"] == "research" and r2["answer"] == "FAKE_RESEARCH_ANSWER"
    assert "https://example.com" in r2["sources"]

    print("[graph] supervisor -> route -> node -> END wiring OK")
    print("\nSTEP6_WIRING_OK")


def test_phase1_routing_decisions():
    """pytest entry point (the historical entry is running this file
    directly). Restores the graph module's agent nodes afterwards so other
    suites in the same session build the real graph, not these fakes."""
    import app.graph as graph_module

    saved = (graph_module.books_agent, graph_module.research_agent)
    try:
        main()
    finally:
        (graph_module.books_agent,
         graph_module.research_agent) = saved


if __name__ == "__main__":
    main()
