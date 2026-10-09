"""Research agent: SearXNG search -> crawl top results -> LLM synthesis with URLs."""

import time

from app.llm_client import get_llm, llm_text
from app.state import AgentState, with_agent_output
from app.tools import crawl_extractor, searxng_client

_TOP_RESULTS_TO_CRAWL = 3
_MAX_CHARS_PER_PAGE = 4000

_BAD_RESULT_HOSTS = (
    "youtube.com",
    "youtu.be",
    "dictionary",
    "support.google.com",
    "play.google.com",
)

_REWRITE_PROMPT = """Rewrite the question below as ONE tight web-search query.
Rules:
- Output ONLY the query text, nothing else. No quotes, no explanation.
- 3 to 8 words, topic keywords only.
- Drop filler words like "what", "is", "used for", "this week".

Question: {question}

Search query:"""


def _rewrite_query(question: str) -> str:
    """Turn the user question into a tight SearXNG query (one fast LLM call).

    Never fails the pipeline: any error or empty result falls back to the raw
    question, exactly as before this step existed.
    """
    try:
        raw = llm_text(get_llm(fast=True).invoke(_REWRITE_PROMPT.format(question=question)))
        query = " ".join(raw.strip().split())
        query = query.strip('"\'')
        if 3 <= len(query) <= 120 and "\n" not in query:
            return query
        print(f"[research_agent] rewrite rejected ({query[:60]!r}...); using raw question")
    except Exception as exc:  # noqa: BLE001 — rewrite is an optimization, not a dependency
        print(f"[research_agent] rewrite failed ({type(exc).__name__}); using raw question")
    return question


def _is_noisy(result: dict) -> bool:
    url = (result.get("url") or "").lower()
    return any(host in url for host in _BAD_RESULT_HOSTS)


_PROMPT_TEMPLATE = """You are a research assistant. Answer the question using ONLY
the provided web excerpts. If the excerpts do not contain enough information, say so
clearly instead of guessing. Be concise and factual.

Question:
{question}

Web excerpts:
{context}

Write a short synthesized answer. Do not invent sources."""


def research_agent(state: AgentState) -> dict:
    question = state.get("question", "")
    start = time.perf_counter()

    search_query = _rewrite_query(question)
    if search_query != question:
        print(f"[research_agent] search query: {search_query!r}")

    results = [
        r for r in searxng_client.search(search_query, num_results=8) if not _is_noisy(r)
    ][:5]
    if not results:
        print("[research_agent] no search results available")
        return with_agent_output(
            state,
            "research",
            "Web search returned no results, so I could not research this question.",
            ["research"],
        )

    context_parts, cited_urls = [], []
    for item in results[:_TOP_RESULTS_TO_CRAWL]:
        page_text = crawl_extractor.extract(item["url"])
        if not page_text:
            continue  # empty extraction handled gracefully, per spec
        cited_urls.append(item["url"])
        context_parts.append(
            f"[Source: {item['title']}] ({item['url']})\n"
            f"{page_text[:_MAX_CHARS_PER_PAGE]}"
        )
    if not context_parts:
        # Fall back to snippets when every crawl failed.
        context_parts = [
            f"[Source: {r['title']}] ({r['url']})\n{r['snippet']}" for r in results
        ]
        cited_urls = [r["url"] for r in results if r["url"]]

    prompt = _PROMPT_TEMPLATE.format(
        question=question, context="\n\n---\n\n".join(context_parts)
    )
    answer = llm_text(get_llm().invoke(prompt)).strip()
    elapsed = time.perf_counter() - start

    print(f"[research_agent] synthesized in {elapsed:.1f}s using {len(cited_urls)} URL(s)")
    listing = "\n".join(f"- {u}" for u in cited_urls)
    return with_agent_output(
        state,
        "research",
        f"{answer}\n\nSources:\n{listing}",
        ["research"] + cited_urls,
    )
