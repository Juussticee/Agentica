"""Graph smoke test: invokes the real compiled graph for the three routing
shapes (books, out-of-corpus refusal, research).

Each call runs the REAL agents — the research scenario performs live web
search/crawl/synthesis and the books scenarios each consume one Gemini
request. For a zero-quota wiring check use scripts/test_phase1_routing.py.

Run:  python scripts/verify_graph.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.graph import build_graph  # noqa: E402


def scenario(label: str, question: str) -> None:
    print("=" * 78)
    print(f"  SCENARIO: {label}")
    print(f"  QUESTION: {question}")
    print("-" * 78)
    try:
        result = build_graph().invoke({"question": question})
    except Exception as exc:  # noqa: BLE001 - report honestly, keep going
        print(f"  [!] graph.invoke raised: {type(exc).__name__}: {exc}")
        return
    print(f"\n  [route: {result.get('route', '?')}]")
    print(f"  [sources: {result.get('sources', [])}]")
    print(f"\n  --- ANSWER ---\n  {result.get('answer', '(no answer)')}\n  (end answer)")


def main() -> None:
    print("GRAPH SMOKE TEST (live services + Gemini quota)")
    print("=" * 78)

    scenario(
        "BOOKS — answerable from the indexed books",
        "What does the book say about the orbital speed of a spacecraft in a "
        "circular low Earth orbit at 400 km altitude?",
    )
    scenario(
        "OUT-OF-CORPUS — should refuse rather than hallucinate",
        "What is quantum tunneling?",  # genuinely absent from the corpus
    )
    scenario(
        "RESEARCH — SearXNG + Crawl4AI + synthesis",
        "What is the current weather in Beirut, Lebanon today?",
    )

    print("\n" + "=" * 78)
    print("SMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
