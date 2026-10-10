"""GUI-16 bounded AI connectivity diagnostics.

The seven-phase readiness check (each phase must pass for `ready`):
  1. configured      — a provider endpoint/key exists
  2. endpoint_parsed — URL sane before any network call
  3. reachable       — HTTP responds from the right service
  4. authorized      — provider accepts the credential
  5. model_present   — requested chat model exists on the endpoint
  6. generation_ok   — a tiny bounded generation succeeds
  7. verified        — (timestamped) end-to-end proof

Ollama-specific rule learned the hard way: model availability MUST come from
GET /api/tags (a POST to /api/tags is not a valid connectivity test), and
"connectivity" must be distinguished from "working chat generation".
"""

import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.execution.errors import (
    AI_AUTH_REJECTED,
    AI_BAD_RESPONSE,
    AI_CONNECTION_REFUSED,
    AI_DNS_UNREACHABLE,
    AI_MALFORMED_ENDPOINT,
    AI_MODEL_MISSING,
    AI_NOT_CONFIGURED,
    AI_TIMEOUT,
    AI_WRONG_SERVICE,
)

TEST_TIMEOUT = 10.0          # seconds, bounded; no retry loops
GEN_TIMEOUT = 120.0          # generation can be slower on modest hardware
GEN_MAX_TOKENS = 8           # "Return exactly: OLLAMA_TEST_OK"

OLLAMA_TEST_PROMPT = "Return exactly: OLLAMA_TEST_OK"
OLLAMA_TEST_EXPECT = "OLLAMA_TEST_OK"

PHASES = ("configured", "endpoint_parsed", "reachable", "authorized",
          "model_present", "generation_ok", "verified")


def _http_json(url: str, method: str = "GET", body: Optional[dict] = None,
               timeout: float = TEST_TIMEOUT) -> Tuple[int, Any]:
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


def _fail(state: str, category: str, extra: Optional[dict] = None) -> Dict[str, Any]:
    result = {
        "state": state,
        "phases": {},
        "category": category,
    }
    if extra:
        result.update(extra)
    return result


def check_ollama(base_url: str, model: str,
                 with_generation: bool = True) -> Dict[str, Any]:
    """Bounded Ollama readiness. Connectivity ≠ generation — reported apart."""
    phases: Dict[str, Dict[str, Any]] = {}

    if not base_url or not model:
        phases["configured"] = {"ok": False}
        return _fail("not_configured", AI_NOT_CONFIGURED, {"phases": phases})
    phases["configured"] = {"ok": True}

    url = base_url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        phases["endpoint_parsed"] = {"ok": False}
        return _fail("error", AI_MALFORMED_ENDPOINT, {"phases": phases})
    phases["endpoint_parsed"] = {"ok": True}

    # Phase 3+4: GET /api/tags — reachability + service identity in one call.
    try:
        status, tags = _http_json(f"{url}/api/tags", timeout=TEST_TIMEOUT)
        if not isinstance(tags, dict) or "models" not in tags:
            phases["reachable"] = {"ok": False, "why": "unexpected response format"}
            return _fail("error", AI_WRONG_SERVICE, {"phases": phases})
        phases["reachable"] = {"ok": True}
        phases["authorized"] = {"ok": True}  # Ollama has no auth; 200 suffices
    except Exception as exc:
        category = _classify_url_error(exc)
        phases["reachable"] = {"ok": False, "category": category}
        state = "not_configured" if category in (AI_CONNECTION_REFUSED,
                                                 AI_DNS_UNREACHABLE, AI_TIMEOUT) else "error"
        return _fail(state, category, {"phases": phases})

    # Phase 5: requested chat model installed?
    models = tags.get("models", [])
    names = []
    for m in models:
        name = m.get("name") or m.get("model")
        if name:
            names.append(name)
    # bare tag form matches its tagged variant ('llama3' -> 'llama3:latest')
    model_present = model in names or any(n.split(":")[0] == model for n in names)
    is_embedding_only = (
        all("embed" in n.lower() for n in names) if names
        else (not model_present and any("embed" in n.lower() for n in names))
    )
    phases["model_present"] = {
        "ok": model_present,
        "available_models": names[:20],
        "expected_model": model,
    }
    if not model_present:
        result = _fail("model_missing", AI_MODEL_MISSING, {"phases": phases})
        if is_embedding_only:
            result["hint"] = (
                "Only embedding models are installed. Chat generation needs a "
                "chat-capable model installed on the Ollama machine."
            )
        return result

    # Phase 6: tiny bounded generation against the actual chat model.
    if not with_generation:
        result = {
            "state": "partial",
            "phases": phases,
            "category": None,
            "note": "Generation check skipped; connectivity and model verified only.",
        }
        return result

    started = time.perf_counter()
    try:
        gstatus, gen = _http_json(
            f"{url}/api/generate",
            method="POST",
            body={
                "model": model,
                "prompt": OLLAMA_TEST_PROMPT,
                "stream": False,
                # qwen3-family models emit reasoning into a separate
                # "thinking" field when enabled; a tiny num_predict is then
                # consumed entirely by thinking and `response` comes back
                # empty (done_reason=length). Disabling thinking gives an
                # honest, bounded generation probe for ANY model that
                # supports the flag; models without it just ignore it.
                "think": False,
                "options": {"num_predict": GEN_MAX_TOKENS,
                            "temperature": 0.0},
            },
            timeout=GEN_TIMEOUT,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        text = ""
        if isinstance(gen, dict):
            text = str(gen.get("response") or "").strip()
        # Models may wrap the answer in markdown emphasis; accept the token
        # when it appears in the first short line of output.
        first_line = text.splitlines()[0] if text else ""
        expected = OLLAMA_TEST_EXPECT in first_line or text == OLLAMA_TEST_EXPECT
        phases["generation_ok"] = {
            "ok": bool(text) and expected,
            "elapsed_ms": elapsed_ms,
            "response_excerpt": text[:80],
        }
        if not (text and expected):
            return _fail("error", AI_BAD_RESPONSE, {"phases": phases})
        phases["verified"] = {"ok": True}
        return {
            "state": "ready",
            "phases": phases,
            "category": None,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        category = _classify_url_error(exc)
        phases["generation_ok"] = {"ok": False, "category": category}
        return _fail("error", category, {"phases": phases})


def check_gemini(api_key: Optional[str], model: str) -> Dict[str, Any]:
    """Bounded Gemini readiness against the public model list endpoint.

    Distinguishes: no key / bad key (401-403) / endpoint down / plausible OK.
    A full generation check requires the SDK path used by get_llm(); the list
    call proves endpoint + key acceptance, not chat quality — that gap is
    reported honestly in `note`.
    """
    phases: Dict[str, Any] = {}
    if not api_key:
        phases["configured"] = {"ok": False}
        return _fail("not_configured", AI_NOT_CONFIGURED, {"phases": phases})
    phases["configured"] = {"ok": True}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        f"?key={api_key}&pageSize=1"
    )
    try:
        status, body = _http_json(url, timeout=TEST_TIMEOUT)
        phases["reachable"] = {"ok": True}
        phases["authorized"] = {"ok": True}
        phases["model_list_ok"] = {"ok": True, "note": f"model list status {status}"}
        return {
            "state": "partial",
            "phases": phases,
            "category": None,
            "note": (
                "Gemini endpoint reachable and key accepted (model list OK). "
                f"Live chat with {model!r} is exercised by real executions, not "
                "by this bounded test."
            ),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
    except urllib.error.HTTPError as exc:
        category = AI_AUTH_REJECTED if exc.code in (401, 403) else AI_BAD_RESPONSE
        phases["authorized"] = {"ok": False, "code": exc.code, "category": category}
        return _fail("error", category, {"phases": phases})
    except Exception as exc:
        category = _classify_url_error(exc)
        phases["reachable"] = {"ok": False, "category": category}
        return _fail("error", category, {"phases": phases})
