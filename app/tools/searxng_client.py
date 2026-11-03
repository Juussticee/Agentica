"""SearXNG client tool — one function, per spec: search(query, num_results)."""

import time
from typing import Dict, List

import requests

from app.config import SEARXNG_URL
from app.execution.events import TOOL_COMPLETED, TOOL_FAILED
from app.execution.instrumentation import emit


def search(query: str, num_results: int = 5) -> List[Dict]:
    """Query the SearXNG JSON API; returns [{title, url, snippet}], [] on failure.

    Never raises — the research agent handles empty results gracefully.
    W2: the outcome (count / duration / coarse error category) is recorded
    as an event while it happens. The QUERY is deliberately not part of the
    payload — it is the user's own question, already stored on the run.
    """
    started = time.perf_counter()
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            headers={"User-Agent": "master-agent-system/0.1"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — tool boundary, must not raise
        print(f"[searxng_client] search failed: {exc}")
        # Category only: the raw exception string can echo the request URL,
        # which carries the query — that never belongs in an event payload.
        emit(
            TOOL_FAILED,
            "Web search failed",
            payload={
                "tool": "searxng.search",
                "error_type": type(exc).__name__,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return []

    results = []
    for item in data.get("results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
        )
    emit(
        TOOL_COMPLETED,
        f"Web search returned {len(results)} result(s)",
        payload={
            "tool": "searxng.search",
            "result_count": len(results),
            "duration_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return results
