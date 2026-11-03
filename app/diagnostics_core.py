"""Shared diagnostic core (Increment 2, 2026-10-28).

Two underlying checks were previously answered independently by more than
one endpoint, so the three status surfaces could disagree:

1. ``provider_readiness()`` — "is the configured AI provider configured?"
   Was duplicated in ``server.ai_ready()`` (/api/v1/health) and in the
   /api/v1/services ``ai_provider`` rollup, with DIFFERENT logic: an unset
   or empty ``llm_provider`` alongside a saved Gemini key read as "ready"
   in /health but "not_configured" in /services. It is config-only by
   contract: no network call, no model call — /health must never depend on
   live inference.

2. ``fetch_ollama_tags()`` + ``model_present()`` — one bounded read of
   Ollama's model list and ONE definition of "the configured chat model is
   installed". ``services_health.check_ollama`` used to issue THREE separate
   GET /api/tags requests per call (probe, latency, list) and re-implemented
   the bare-tag matching rule that ``ai_diagnostics.check_ollama`` already
   had. The dashboard polls /api/v1/services, so the triple probe was paid
   on every poll.

This module returns plain facts. Each endpoint keeps its own vocabulary,
extra fields, timeouts, config source and side effects — presentation and
policy stay with the caller, and no response schema changes.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.execution.errors import (
    AI_AUTH_REJECTED,
    AI_BAD_RESPONSE,
    AI_CONNECTION_REFUSED,
    AI_DNS_UNREACHABLE,
    AI_MODEL_MISSING,
    AI_TIMEOUT,
    AI_WRONG_SERVICE,
)

# --- check 1: provider readiness (config only, no network) -----------------


def provider_readiness(eff: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Single definition of "the configured AI provider is configured".

    Pure config inspection — NO network request and NO model call, so /health
    stays free of live inference. Pass ``eff`` to supply the merged settings
    explicitly (tests); by default they are read through
    ``ai_settings.effective_settings()`` (session > saved > env/config) via a
    module-attribute lookup so existing patches keep applying.

    Returns plain facts; each caller adds its own response fields.
    """
    if eff is None:
        from app import ai_settings
        eff = ai_settings.effective_settings()

    provider = eff.get("llm_provider") or "gemini"
    if provider == "gemini":
        configured = bool(eff.get("gemini_api_key"))
        what = "Gemini"
        model = eff.get("gemini_model")
    elif provider == "ollama":
        configured = bool(eff.get("ollama_base_url"))
        what = f"Ollama ({eff.get('ollama_base_url')})"
        model = eff.get("main_model")
    else:
        configured = False
        what = f"unknown provider {provider!r}"
        model = ""

    return {
        # Normalized provider actually in effect (matches /health's display).
        "provider": provider,
        # The raw stored setting (matches /services' display of what is saved).
        "provider_setting": eff.get("llm_provider"),
        "configured": configured,
        "state": "ready" if configured else "not_configured",
        "what_will_be_tested": f"{what}, model {model}",
        "state_hint": (
            "Configured. Agentica will use it for AI-backed steps."
            if configured
            else
            "No API key or endpoint set. AI-backed steps will fail until you "
            "configure a provider in Settings."
        ),
    }


# --- check 2: Ollama model list (one fetch, one presence rule) -------------


def ollama_model_names(payload: Any) -> List[str]:
    """Model names from a parsed GET /api/tags body (never raises)."""
    if not isinstance(payload, dict):
        return []
    names = []
    for m in payload.get("models", []) or []:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model")
            if name:
                names.append(name)
    return names


def model_present(model: str, names: List[str]) -> bool:
    """ONE definition of "the configured chat model is installed".

    Exact tag match, or bare tag against its tagged variant
    ('qwen3' matches 'qwen3:0.6b'; 'qwen3:0.6b' matches exactly).
    """
    if not model:
        return False
    return model in names or any(
        (n or "").split(":")[0] == model for n in names
    )


def fetch_ollama_tags(base_url: str, *, timeout: float
                      ) -> Tuple[Optional[int], Any, Optional[str]]:
    """Single bounded GET ``{base_url}/api/tags`` — never raises.

    Returns ``(status, payload, category)``:

    * ``status``  — HTTP status when the endpoint answered (including an
      HTTP error response), ``None`` when nothing answered;
    * ``payload`` — the parsed body when it was readable, else ``None``;
    * ``category`` — an ``AI_*`` failure category when the HTTP exchange
      failed, ``None`` when the exchange itself succeeded.

    Body-shape validation is deliberately left to the caller: the dashboard
    row and the readiness phases interpret a wrong-shaped body in their own
    vocabulary.
    """
    url = f"{(base_url or '').rstrip('/')}/api/tags"
    try:
        status, body = _http_json(url, timeout=timeout)
        return status, body, None
    except urllib.error.HTTPError as exc:
        # Something answered: keep the status so callers can still
        # distinguish "HTTP 500 (degraded)" from "no answer".
        return exc.code, None, _classify_url_error(exc)
    except Exception as exc:  # noqa: BLE001 — a probe never raises
        return None, None, _classify_url_error(exc)


# --- shared low-level HTTP + error classification (moved from ai_diagnostics) ---

def _http_json(url: str, method: str = "GET", body: Optional[dict] = None,
               timeout: float = 10.0) -> Tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        try:
            return resp.status, json.loads(raw)
        except Exception:
            return resp.status, raw


def _classify_url_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return AI_AUTH_REJECTED
        if exc.code == 404:
            return AI_MODEL_MISSING
        return AI_BAD_RESPONSE
    if isinstance(exc, urllib.error.URLError):
        reason = str(getattr(exc, "reason", exc)).lower()
        if "timed out" in reason:
            return AI_TIMEOUT
        if "connection refused" in reason or "connectionreset" in reason:
            return AI_CONNECTION_REFUSED
        if "getaddrinfo failed" in reason or "name or service not known" in reason:
            return AI_DNS_UNREACHABLE
        return AI_DNS_UNREACHABLE
    if isinstance(exc, socket.timeout):
        return AI_TIMEOUT
    return AI_BAD_RESPONSE
