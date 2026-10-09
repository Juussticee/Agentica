"""Query-rewrite step tests — fake LLM + mocked SearXNG, zero Gemini quota.

Covers:
  1. The rewrite output is what reaches SearXNG (not the raw question)
  2. Synthesis still receives the ORIGINAL question
  3. Noisy hosts (youtube/dictionary/support.google) are filtered from results
  4. Rewrite failure degrades gracefully to the raw question
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class _Msg:
    def __init__(self, text):
        self.content = text


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        if "Rewrite the question" in str(prompt):
            return _Msg("urban beekeeping business startup costs")
        return _Msg("synthesized answer from excerpts")


def run(fake, monkeypatched_search, question):
    import app.llm_client as llm_client_module
    from app.agents import research_agent as ra

    # Patch BOTH namespaces: research_agent imported get_llm by name, so its
    # module global must be rebound too (or the real model gets called).
    llm_client_module.get_llm = lambda fast=False: fake
    ra.get_llm = lambda fast=False: fake
    ra.searxng_client = type(
        "S", (), {"search": staticmethod(lambda q, num_results=5: monkeypatched_search(q, num_results))}
    )
    # hermetic: no real network crawls either
    ra.crawl_extractor = type(
        "C", (), {"extract": staticmethod(lambda url: f"CRAWLED CONTENT OF {url}")}
    )
    return ra.research_agent({"question": question})


GOOD = {"title": "Beekeeping startup guide", "url": "https://example.com/", "snippet": "costs"}
NOISY = [
    {"title": "Video", "url": "https://www.youtube.com/watch?v=abc", "snippet": "video"},
    {"title": "Definition", "url": "https://dictionary.cambridge.org/do", "snippet": "def"},
    {"title": "Support", "url": "https://support.google.com/youtube/?hl=en", "snippet": "help"},
]


def main():
    fake = FakeLLM()

    searches = []

    def search_recorder(q, num_results=5):
        searches.append((q, num_results))
        return NOISY + [GOOD]

    r = run(fake, search_recorder, "What are urban beekeeping startup costs for a beginner?")

    # 1. rewritten query reached SearXNG
    assert searches and searches[0][0] == "urban beekeeping business startup costs", searches
    # 2. synthesis prompt carries the ORIGINAL question
    synthesis_prompt = fake.prompts[-1]
    assert "urban beekeeping startup costs for a beginner" in synthesis_prompt
    assert "Rewrite the question" not in synthesis_prompt
    # exactly two LLM interactions: one rewrite, one synthesis (both fake)
    assert len(fake.prompts) == 2, fake.prompts
    # 3. noisy hosts filtered, good result kept
    assert "https://example.com/" in r["answer"]
    assert all("youtube.com" not in s and "dictionary" not in s for s in r["sources"])
    assert r["sources"][0] == "research"

    # 4. rewrite failure -> graceful fallback to raw question
    class BoomLLM(FakeLLM):
        def invoke(self, prompt):
            self.prompts.append(str(prompt))
            if "Rewrite the question" in str(prompt):
                raise RuntimeError("quota")
            return _Msg("answer")

    searches.clear()
    boom = BoomLLM()
    run(boom, search_recorder, "raw question fallback test")
    assert searches[0][0] == "raw question fallback test", searches
    # exactly two LLM interactions: the failed rewrite attempt, then a
    # synthesis prompt that carries the ORIGINAL question (no rewrite text)
    assert len(boom.prompts) == 2, boom.prompts
    assert boom.prompts[0].startswith("Rewrite"), boom.prompts[0][:60]
    assert "Rewrite" not in boom.prompts[1]
    assert "raw question fallback test" in boom.prompts[1]

    print("REWRITE_STEP_OK — all 4 behaviors verified (zero quota)")


if __name__ == "__main__":
    main()
