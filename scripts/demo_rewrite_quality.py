"""Before/after demo: raw question vs rewritten query on live SearXNG.

Cost: exactly ONE fast Gemini call (the rewrite). No crawls, no synthesis.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.research_agent import _is_noisy, _rewrite_query  # noqa: E402
from app.tools import searxng_client  # noqa: E402


def show(label, question_or_query):
    results = searxng_client.search(question_or_query, num_results=5)
    print(f"\n--- {label} ---")
    print(f"query: {question_or_query!r}")
    for r in results:
        flag = "NOISY " if _is_noisy(r) else "clean "
        print(f"  [{flag}] {r['url'][:90]}")
    clean = [r for r in results if not _is_noisy(r)]
    kept = [r for r in results if not _is_noisy(r)][:3]
    print(f"  => {len(clean)}/{len(results)} clean; research_agent would crawl {len(kept)}")
    return results


def main():
    question = "What is LangGraph used for?"

    before = show("BEFORE (raw question, pre-change behavior)", question)

    rewritten = _rewrite_query(question)
    print(f"\n[one fast Gemini call] rewrite -> {rewritten!r}")

    after = show("AFTER (rewritten query + noise filter)", rewritten)

    def noise_ratio(rs):
        return (sum(1 for r in rs if _is_noisy(r)) / len(rs)) if rs else 0.0

    print("\n=== verdict ===")
    print(f"before: {len(before)} results, noise ratio {noise_ratio(before):.0%}")
    print(f"after:  {len(after)} results, noise ratio {noise_ratio(after):.0%}")
    kept_before = [r["url"] for r in before if not _is_noisy(r)][:3]
    kept_after = [r["url"] for r in after if not _is_noisy(r)][:3]
    print(f"crawled before: {kept_before}")
    print(f"crawled after:  {kept_after}")


if __name__ == "__main__":
    main()
