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


def main():
    import app.graph as graph_module
    from app.supervisor import supervisor

    # --- supervisor unit checks (pure logic, no LLM) ---
    assert supervisor({"question": "What does chapter 3 say about propolis?"})["route"] == "books"
    assert supervisor({"question": "Find pages about Hohmann transfers in my books"})["route"] == "books"
    assert supervisor({"question": "What is the latest news on fusion energy startups?"})["route"] == "research"
    assert supervisor({"question": "Who won the 2024 F1 championship?"})["route"] == "research"
    print("[supervisor] routing decisions OK")

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


if __name__ == "__main__":
    main()
