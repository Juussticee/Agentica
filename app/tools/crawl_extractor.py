"""Crawl4AI extractor tool — one function, per spec: extract(url) -> str.

Crawl4AI uses an async API internally, so this sync wrapper runs it on a
private event loop. Returns "" on any failure (never raises).

Note: log messages are ASCII-sanitized because this project runs on Windows
consoles (cp1256) where printing raw exception text can itself raise
UnicodeEncodeError.
"""

import asyncio
import sys
import time
from contextlib import redirect_stderr
from io import StringIO
from urllib.parse import urlparse

from app.execution.events import TOOL_COMPLETED, TOOL_FAILED
from app.execution.instrumentation import emit


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
    """Fetch and return clean text for a URL; "" on failure — never raises.

    W2: records the outcome as an event while it happens — the host and the
    amount of text obtained, never the page content itself.
    """
    started = time.perf_counter()
    host = urlparse(url).netloc[:80]
    try:
        with redirect_stderr(StringIO()):  # crawl4ai logs startup noise to stderr
            text = _crawl(url)
        text = (text or "").strip()
        emit(
            TOOL_COMPLETED,
            f"Page extraction finished ({len(text)} chars)",
            payload={
                "tool": "crawl.extract",
                "host": host,
                "chars": len(text),
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return text
    except Exception as exc:  # noqa: BLE001 — tool boundary, must not raise
        print(f"[crawl_extractor] extract failed ({type(exc).__name__}): {_ascii(str(exc))}")
        emit(
            TOOL_FAILED,
            "Page extraction failed",
            payload={
                "tool": "crawl.extract",
                "host": host,
                "error_type": type(exc).__name__,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return ""
