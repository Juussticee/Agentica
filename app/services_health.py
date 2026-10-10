"""Service inventory + bounded health checks for the Control Center.

Dashboard rule (recovery spec §A + §33 of CONTROL_CENTER_PLAN): no hardcoded
statuses, no fabricated latency. Every state here comes from a real probe or
from configuration reality. "Not configured" is explicitly distinct from
"unreachable", and optional services never turn the whole system red.

Checks are lightweight availability probes (bounded timeout), NOT functional
tests; the expensive generation/model checks stay behind explicit user
actions (POST /api/v1/ai/test) and are surfaced as a separate unknown state
until verified.
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import PROJECT_ROOT

PROBE_TIMEOUT = 3.5   # bounded; no retry loops

# State vocabulary shown verbatim by the UI.
RUNNING = "running"
STOPPED = "stopped"
UNREACHABLE = "unreachable"
DEGRADED = "degraded"
MISCONFIGURED = "misconfigured"
NOT_CONFIGURED = "not_configured"
UNKNOWN = "unknown"

_OPTIONAL_BECAUSE = {
    "research": "Web research is optional: answers degrade to feasibility "
                "web-only or refusal paths when it is down.",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_status(url: str, timeout: float = PROBE_TIMEOUT) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code  # something answered — the service IS reachable
    except Exception:
        return None


def _latency_ms(url: str) -> Optional[int]:
    started = time.perf_counter()
    status = _http_status(url)
    if status is None:
        return None
    return int((time.perf_counter() - started) * 1000)


def _classify_probe(status: Optional[int]) -> Dict[str, Any]:
    """Map a lightweight HTTP probe to the state vocabulary + reason."""
    if status is None:
        return {"state": UNREACHABLE,
                "reason": "No answer at the configured address (nothing "
                          "listening or network error)."}
    if 500 <= status <= 599:
        return {"state": DEGRADED,
                "reason": f"Service answered with an error (HTTP {status})."}
    return {"state": RUNNING, "reason": None}


def check_api(store_ok: bool) -> Dict[str, Any]:
    """The Agentica API itself: implied running if this request executes."""
    return {
        "id": "agentica_api",
        "name": "Agentica backend",
        "purpose": "Serves the Control Center and runs executions",
        "required": True,
        "state": RUNNING if store_ok else DEGRADED,
        "reason": None if store_ok else "Storage opening failed; execution "
                                        "history is unavailable.",
        "latency_ms": None,
        "last_checked": _now(),
        "next_action": (None if store_ok else
                        "Check the storage/ database, then restart the API."),
        "config": {},
    }


def check_storage() -> Dict[str, Any]:
    """SQLite execution persistence: open it and read one row."""
    import sqlite3

    from app.execution.store import DEFAULT_DB_PATH

    started = time.perf_counter()
    try:
        conn = sqlite3.connect(str(PROJECT_ROOT / DEFAULT_DB_PATH), timeout=2)
        try:
            n = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
            ok = True
            reason = None
        except sqlite3.OperationalError as exc:
            n = None
            ok = False
            reason = f"Database tables unavailable ({type(exc).__name__})."
        finally:
            conn.close()
        return {
            "id": "storage",
            "name": "Execution storage (SQLite)",
            "purpose": "Persists executions and events across restarts",
            "required": True,
            "state": RUNNING if ok else MISCONFIGURED,
            "reason": reason,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "last_checked": _now(),
            "next_action": (None if ok else
                            "The schema is created on server start; restart "
                            "the API once to initialize it."),
            "config": {"path": str(DEFAULT_DB_PATH), "executions": n},
        }
    except Exception as exc:
        return {
            "id": "storage",
            "name": "Execution storage (SQLite)",
            "purpose": "Persists executions and events across restarts",
            "required": True,
            "state": STOPPED,
            "reason": f"Could not open the database ({type(exc).__name__}).",
            "latency_ms": None,
            "last_checked": _now(),
            "next_action": "Check the storage/ folder permissions or restart "
                           "the API to create the database.",
            "config": {"path": str(DEFAULT_DB_PATH)},
        }


def check_ollama() -> Dict[str, Any]:
    """Split: endpoint availability vs configured-model availability."""
    from app.config import MAIN_MODEL, OLLAMA_BASE_URL

    if not OLLAMA_BASE_URL:
        return _not_configured("ollama", "Ollama",
                               "Local AI model runtime", required=False,
                               what_now="Configure the Ollama address in "
                                        "Settings → AI connection.")

    url = OLLAMA_BASE_URL.rstrip("/")
    result: Dict[str, Any] = {
        "id": "ollama",
        "name": "Ollama",
        "purpose": "Runs the configured AI model for execution steps",
        "required": False,   # gemini default path does not use it
        "config": {"base_url": OLLAMA_BASE_URL, "model": MAIN_MODEL},
    }
    status = _http_status(f"{url}/api/tags")
    probe = _classify_probe(status)
    latency = _latency_ms(f"{url}/api/tags")
    result.update({
        "state": probe["state"],
        "reason": probe["reason"],
        "latency_ms": latency,
        "last_checked": _now(),
    })
    if probe["state"] == RUNNING:
        # Cheap model-list read (same endpoint) — distinguish reachability
        # from "the configured model is actually installed".
        try:
            req = urllib.request.Request(f"{url}/api/tags")
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                tags = json.loads(resp.read().decode("utf-8", "replace"))
            names = [m.get("name") or m.get("model") for m in tags.get("models", [])]
            names = [n for n in names if n]
            result["config"]["available_models"] = names[:20]
            model_ok = MAIN_MODEL in names or any(
                (n or "").split(":")[0] == MAIN_MODEL for n in names)
            if not model_ok:
                result["state"] = DEGRADED
                result["reason"] = (f"Endpoint is up, but the configured model "
                                    f"{MAIN_MODEL!r} is not installed there.")
                result["next_action"] = ("Install the model on the Ollama "
                                         "machine or select an available one "
                                         "in Settings → AI connection.")
            else:
                result["next_action"] = ("Model availability confirmed by list; "
                                         "run 'Test connection' in Settings "
                                         "for a full generation check.")
        except Exception as exc:
            result["state"] = UNKNOWN
            result["reason"] = ("Endpoint answered but the model list could "
                                "not be read.")
            result["config"]["detail"] = type(exc).__name__
    else:
        result["next_action"] = ("Start the Ollama service or fix the address "
                                 "in Settings → AI connection.")
    return result


def check_searxng() -> Dict[str, Any]:
    """SearXNG: optional research dependency; not configured ≠ outage."""
    from app.config import SEARXNG_URL

    result: Dict[str, Any] = {
        "id": "searxng",
        "name": "SearXNG (web research)",
        "purpose": "Search backend used by the research agent",
        "required": False,
        "config": {"base_url": SEARXNG_URL},
        "last_checked": _now(),
    }
    if not SEARXNG_URL:
        return _not_configured("searxng", "SearXNG (web research)",
                               "Search backend used by the research agent",
                               what_now="Set SEARXNG_URL to enable web "
                                        "research.")
    url = f"{SEARXNG_URL.rstrip('/')}/search?q=agentica+ping&format=json"
    started = time.perf_counter()
    status = _http_status(url)
    latency = int((time.perf_counter() - started) * 1000) if status else None
    result["latency_ms"] = latency
    if status is None:
        result.update({
            "state": UNREACHABLE,
            "reason": "No answer from the search service (research requests "
                      "will fall back to degraded web-only paths).",
            "next_action": ("Start the SearXNG container (docker compose up) "
                            "or update SEARXNG_URL. Research is optional; "
                            "other routes keep working."),
        })
        return result
    if status == 403:
        result.update({
            "state": MISCONFIGURED,
            "reason": "Search answered but refused the JSON API request "
                      "(format=enforced is likely disabled).",
            "next_action": "Enable 'formats: [html, json]' in the SearXNG "
                           "settings, or point SEARXNG_URL at a instance "
                           "that permits JSON.",
        })
        return result
    if status == 429:
        result.update({
            "state": DEGRADED,
            "reason": "Search rate-limited the probe (HTTP 429).",
            "next_action": "Wait before requesting more web research.",
        })
        return result
    probe = _classify_probe(status)
    result.update({"state": probe["state"], "reason": probe["reason"]})
    return result


def check_erp() -> Dict[str, Any]:
    """ERP mock adapter: in-process, deterministic, no network."""
    try:
        from app.erp import get_erp_adapter
        from app.config import ERP_ADAPTER

        adapter_name = get_erp_adapter().name() if callable(
            getattr(get_erp_adapter(), "name", None)) else "mock"
        return {
            "id": "erp",
            "name": "ERP integration",
            "purpose": "Business-data lookups via the read-only adapter",
            "required": False,
            "state": RUNNING if ERP_ADAPTER == "mock" else UNKNOWN,
            "reason": None if ERP_ADAPTER == "mock" else
                      f"Unknown adapter type {ERP_ADAPTER!r}; runs as mock "
                      "until configured.",
            "latency_ms": None,
            "last_checked": _now(),
            "next_action": ("Mock adapter is by design read-only placeholder "
                            "data (real ERP intentionally deferred per the "
                            "MASTER_PLAN ERP gate)."
                            if ERP_ADAPTER == "mock" else
                            "Verify the configured adapter module."),
            "config": {"adapter": ERP_ADAPTER, "adapter_name": adapter_name},
        }
    except Exception as exc:
        return {
            "id": "erp",
            "name": "ERP integration",
            "purpose": "Business-data lookups via the read-only adapter",
            "required": False,
            "state": UNKNOWN,
            "reason": f"Adapter load failed ({type(exc).__name__}).",
            "latency_ms": None,
            "last_checked": _now(),
            "next_action": "Check the ERP_ADAPTER configuration value.",
            "config": {},
        }


def _not_configured(sid: str, name: str, purpose: str, what_now: str,
                    required: bool = False) -> Dict[str, Any]:
    return {
        "id": sid,
        "name": name,
        "purpose": purpose,
        "required": required,
        "state": NOT_CONFIGURED,
        "reason": "No address configured.",
        "latency_ms": None,
        "last_checked": _now(),
        "next_action": what_now,
        "config": {},
    }


def services_snapshot() -> Dict[str, Any]:
    """Run every bounded check; assemble the dashboard-ready payload."""
    from app.execution.store import ExecutionStore, DEFAULT_DB_PATH

    storage = check_storage()
    payload = {
        "checked_at": _now(),
        "services": [
            {
                "id": "agentica_api",
                "name": "Agentica backend",
                "purpose": "Serves the Control Center and runs executions",
                "required": True,
                "state": RUNNING,
                "reason": None,
                "latency_ms": 0,
                "last_checked": _now(),
                "next_action": None,
                "config": {},
            },
            storage,
            check_ollama(),
            check_searxng(),
            check_erp(),
        ],
    }
    # Lightweight "AI provider readiness" rollup mirrors /health semantics
    from app.ai_settings import effective_settings
    eff = effective_settings()
    payload["ai_provider"] = {
        "provider": eff.get("llm_provider"),
        "state": ("ready" if (eff.get("llm_provider") == "gemini"
                              and eff.get("gemini_api_key"))
                  or (eff.get("llm_provider") == "ollama"
                      and eff.get("ollama_base_url")) else "not_configured"),
        "settings_url": "/#/settings",
        "note": "Availability only; functional verification via "
                "Settings → Test connection.",
    }
    return payload
