"""CLI entry point: build the graph once, then chat (spec Section 3).

Optional flags (all optional, any combination):
  --once        Run a single question, then quit (suitable for scripted demos).
  --demo        Run the Phase 1 canned scenarios (books / out-of-corpus /
                research) and exit.

When neither flag is given the process drops into an interactive loop.
"""
import argparse
import sys

# Windows consoles (e.g. cp1256) can choke on unicode in answers; replace
# unencodable characters instead of crashing.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.graph import build_graph
from app.llm_client import llm_text

try:
    from langchain_google_genai.chat_models import GoogleRateLimitError
except ImportError:  # provider-specific; degrade to a generic catch below
    GoogleRateLimitError = ()


def _answer(graph, question: str) -> dict:
    result = graph.invoke({"question": question})
    return result


def _pretty(result: dict) -> None:
    answer = result.get("answer", "(no answer)")
    route = result.get("route", "?")
    sources = result.get("sources", [])
    print(f"\n{answer}")
    print(f"\n[route: {route}] [sources: {sources}]")


def run_demo() -> None:
    graph = build_graph()
    print("Master Agent System (Phase 1: books + research) — demo mode.\n")

    scenarios = [
        (
            "What does the book say about the orbital speed of a spacecraft in a "
            "circular low Earth orbit at 400 km altitude?",
            "BOOKS — answerable from the indexed books",
        ),
        (
            "Who won the 2026 FIFA World Cup?",
            "OUT-OF-CORPUS — should refuse rather than hallucinate",
        ),
        (
            "What is the current weather in Beirut, Lebanon today?",
            "RESEARCH — SearXNG + Crawl4AI + Gemini synthesis",
        ),
    ]

    for question, label in scenarios:
        print("=" * 70)
        print(f"SCENARIO: {label}")
        print(f"QUESTION: {question}")
        print("-" * 70)
        try:
            result = _answer(graph, question)
        except GoogleRateLimitError:
            print(
                "\n[!] Gemini quota exhausted (free tier: 20 requests/day for "
                "gemini-3-flash). Wait for the daily reset, enable billing, "
                "or set GEMINI_MODEL in .env to a different model.\n"
            )
            continue
        except Exception as exc:  # noqa: BLE001 — report and keep going
            print(f"\n[!] graph.invoke raised: {type(exc).__name__}: {exc}\n")
            continue
        _pretty(result)

    print("\n" + "=" * 70)
    print("Demo complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Agent System (Phase 1: books + research)."
    )
    parser.add_argument(
        "--once",
        metavar="QUESTION",
        help="Run a single question then quit (script-friendly).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the canned Phase 1 scenarios and exit.",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    graph = build_graph()
    print("Master Agent System (Phase 1: books + research). Ctrl+C or empty line to quit.")

    if args.once:
        try:
            result = _answer(graph, args.once)
        except GoogleRateLimitError:
            print(
                "[!] Gemini quota exhausted (free tier: 20 requests/day for "
                "gemini-3-flash). Wait for the daily reset, enable billing, "
                "or set GEMINI_MODEL in .env to a different model."
            )
            return
        _pretty(result)
        return

    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not question:
            break
        try:
            result = _answer(graph, question)
        except GoogleRateLimitError:
            print(
                "[!] Gemini quota exhausted (free tier: 20 requests/day for "
                "gemini-3-flash). Wait for the daily reset, enable billing, "
                "or set GEMINI_MODEL in .env to a different model."
            )
            continue
        _pretty(result)


if __name__ == "__main__":
    main()
