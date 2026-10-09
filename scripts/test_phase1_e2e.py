"""Phase 1 REAL end-to-end test (spec steps 3 and 5) — live Gemini via get_llm().

  step 3a: books_agent answers an indexed question, cites file/page provenance
  step 3b: books_agent refuses on out-of-corpus question (never invents)
  step 5:  research_agent searches, crawls, synthesizes, lists source URLs
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.books_agent import books_agent  # noqa: E402
from app.agents.research_agent import research_agent  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    r = books_agent({"question": "How much delta-v does a Hohmann transfer need?"})
    print(f"\n[step3a books] ({time.perf_counter() - t0:.0f}s)")
    print(r["answer"][:400])
    assert r["sources"] == ["books"], "books_agent must attach 'books' provenance"
    assert "References:" in r["answer"], "expected file/page provenance"
    assert "not found" not in r["answer"].lower(), "unexpected refusal on indexed content"

    t0 = time.perf_counter()
    r = books_agent({"question": "What is quantum tunneling?"})
    print(f"\n[step3b books-refusal] ({time.perf_counter() - t0:.0f}s)")
    print(r["answer"][:200])
    assert r["answer"].startswith("That information was not found in your books"), (
        "out-of-corpus question must be refused, not invented"
    )

    t0 = time.perf_counter()
    r = research_agent({"question": "What is LangGraph and what is it used for?"})
    print(f"\n[step5 research] ({time.perf_counter() - t0:.0f}s)")
    print(r["answer"][:400])
    assert r["sources"][0] == "research", "research_agent must attach 'research' provenance"
    assert any(s.startswith("http") for s in r["sources"]), "expected source URLs"

    print("\nSTEPS_3_5_REAL_OK")


if __name__ == "__main__":
    main()
