"""Direct books-agent check (no graph): one answerable question end to end.

Uses the real Chroma index and the real LLM, so it consumes ONE Gemini
request. For a zero-quota wiring check use scripts/test_phase1_proxy.py.

Run:  python scripts/check_books_agent.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.books_agent import books_agent  # noqa: E402
from app.state import AgentState  # noqa: E402


def main() -> None:
    question = (
        "What is the orbital speed of a spacecraft in a circular low Earth "
        "orbit at 400 km altitude?"
    )
    state: AgentState = {"question": question}
    result = books_agent(state)
    print("ANSWER:")
    print(result.get("answer"))
    print()
    print("SOURCES:", result.get("sources"))


if __name__ == "__main__":
    main()
