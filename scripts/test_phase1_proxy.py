"""Phase 1 proxy test — validates the full agent wiring with a deterministic
fake LLM injected through the SAME factory the agents use (app.llm_client.get_llm).

This proves everything except the real Gemini key itself:
  - books_agent: Chroma load, retrieval on indexed content, synthesis call,
    not-found refusal, provenance
  - research_agent: SearXNG search, crawl, synthesis call, URL listing
No real API key is needed; no secrets are printed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeGemini:
    """Deterministic stand-in for ChatGoogleGenerativeAI (same .invoke/.content)."""

    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = str(prompt)
        low = self.last_prompt.lower()
        if "quantum tunneling" in low:
            return _FakeMessage("That information was not found in your books.")
        if "delta-v" in low and "hohmann" in low:
            return _FakeMessage(
                "A LEO-to-GTO Hohmann transfer needs about 2.4 km/s, plus ~1.5 km/s "
                "to circularize at GEO."
            )
        if "bee space" in low or "60,000" in low:
            return _FakeMessage(
                "A healthy midsummer hive can contain up to 60,000 worker bees; the "
                "Langstroth design relies on a 6-9 mm bee space."
            )
        return _FakeMessage("Working from the provided excerpts: (fake answer)")


def main():
    import app.llm_client as llm_client_module

    fake = FakeGemini()
    llm_client_module.get_llm = lambda: fake

    # Re-import agents AFTER the patch so they bind the patched factory.
    from app.agents.books_agent import books_agent
    from app.agents.research_agent import research_agent

    # --- books_agent: answerable-from-books question ---
    r = books_agent({"question": "How much delta-v does a Hohmann transfer need?"})
    print("\n[books/hohmann] answer head:", r["answer"][:120].replace("\n", " "))
    assert "2.4" in r["answer"], "expected the delta-v figure in the answer"
    assert r["sources"] == ["books"], "books_agent must attach 'books' to sources"
    assert "References:" in r["answer"], "expected file/page provenance"

    # --- books_agent: not-found refusal path ---
    r = books_agent({"question": "What is quantum tunneling?"})
    print("[books/tunneling] answer:", r["answer"][:100])
    assert r["answer"].startswith("That information was not found in your books")

    # --- research_agent: real search + crawl, fake synthesis ---
    r = research_agent({"question": "what is langgraph"})
    print("[research] answer head:", r["answer"][:100].replace("\n", " "))
    print("[research] sources count:", len(r["sources"]))
    assert r["sources"][0] == "research"
    assert sum(1 for s in r["sources"] if s.startswith("http")) >= 1
    assert fake.last_prompt and "Web excerpts:" in fake.last_prompt, (
        "LLM prompt must include crawled excerpts"
    )
    assert "langgraph" in fake.last_prompt.lower()

    print("\nPHASE1_PROXY_OK — all wiring seams validated")


if __name__ == "__main__":
    main()
