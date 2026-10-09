"""Crawl4AI extractor tool — one function, per spec: extract(url) -> str.

Crawl4AI uses an async API internally, so this sync wrapper runs it on a
private event loop. Returns "" on any failure (never raises).

Note: log messages are ASCII-sanitized because this project runs on Windows
consoles (cp1256) where printing raw exception text can itself raise
UnicodeEncodeError.
"""

import asyncio
import sys
from contextlib import redirect_stderr
from io import StringIO


def _ascii(text: str, limit: int = 200) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")[:limit]


def _crawl(url: str) -> str:
    from crawl4ai import AsyncWebCrawler

    async def run() -> str:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        return (result.markdown or "") if getattr(result, "success", False) else ""

    return asyncio.run(run())


def extract(url: str) -> str:
    """Fetch and return clean text for a URL; "" on failure — never raises."""
    try:
        with redirect_stderr(StringIO()):  # crawl4ai logs startup noise to stderr
            text = _crawl(url)
        return (text or "").strip()
    except Exception as exc:  # noqa: BLE001 — tool boundary, must not raise
        print(f"[crawl_extractor] extract failed ({type(exc).__name__}): {_ascii(str(exc))}")
        return ""
