"""SearXNG client tool — one function, per spec: search(query, num_results)."""

from typing import Dict, List

import requests

from app.config import SEARXNG_URL


def search(query: str, num_results: int = 5) -> List[Dict]:
    """Query the SearXNG JSON API; returns [{title, url, snippet}], [] on failure.

    Never raises — the research agent handles empty results gracefully.
    """
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
    return results
