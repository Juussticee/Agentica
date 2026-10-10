"""AI provider settings + readiness (GUI-16 recovery).

Configuration precedence (highest wins):
  1. Session override (in-process, this server run only)
  2. Persisted settings file: storage/ai_settings.json (explicit "Save for
     this server")
  3. Environment variables / app/config.py defaults (never rewritten)

Secrets are NEVER returned in full — only masked previews. Saved settings
are invalidated from the LLM cache on change so the next call uses them.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import PROJECT_ROOT

SETTINGS_PATH = PROJECT_ROOT / "storage" / "ai_settings.json"

ALLOWED_KEYS = ("llm_provider", "gemini_api_key", "gemini_model",
                "main_model", "ollama_base_url")

_lock = threading.Lock()


def _mask(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}…{value[-4:]} (saved)"


def load_persisted() -> Dict[str, Any]:
    """Read the persisted settings file; empty dict when absent/corrupt."""
    try:
        raw = SETTINGS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in ALLOWED_KEYS and v}
    except FileNotFoundError:
        pass
    except Exception:
        pass  # corrupt file treated as "no saved settings" (fail-soft)
    return {}


def persist(settings: Dict[str, Any]) -> None:
    """Save allowed keys to storage/ai_settings.json (server persistence)."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in settings.items() if k in ALLOWED_KEYS and v}
    with _lock:
        SETTINGS_PATH.write_text(
            json.dumps(clean, indent=2), encoding="utf-8"
        )
    try:
        # Windows has no chmod 0600; POSIX tightens to owner-only. Best-effort.
        os.chmod(SETTINGS_PATH, 0o600)
    except OSError:
        pass


def clear_persisted() -> None:
    try:
        SETTINGS_PATH.unlink()
    except FileNotFoundError:
        pass


# Session override: process-lifetime, wins over persisted + env.
_session_overrides: Dict[str, Any] = {}


def set_session_overrides(overrides: Dict[str, Any]) -> None:
    _session_overrides.update(
        {k: v for k, v in overrides.items() if k in ALLOWED_KEYS}
    )


def effective_settings() -> Dict[str, Any]:
    """Merged view: env/config defaults -> persisted -> session."""
    eff: Dict[str, Any] = {}
    from app import config as cfg

    eff["llm_provider"] = cfg.LLM_PROVIDER
    eff["gemini_api_key"] = cfg.GOOGLE_API_KEY
    eff["gemini_model"] = cfg.GEMINI_MODEL
    eff["main_model"] = cfg.MAIN_MODEL
    eff["ollama_base_url"] = cfg.OLLAMA_BASE_URL
    eff.update(load_persisted())
    eff.update({k: v for k, v in _session_overrides.items() if v})
    return eff


def masked_settings() -> Dict[str, Any]:
    """What the API returns: everything except full secrets."""
    eff = effective_settings()
    return {
        "llm_provider": eff.get("llm_provider"),
        "gemini_api_key_masked": _mask(eff.get("gemini_api_key")),
        "gemini_model": eff.get("gemini_model"),
        "main_model": eff.get("main_model"),
        "ollama_base_url": eff.get("ollama_base_url"),
        "source": {
            "llm_provider": _source_of("llm_provider", eff, bool(eff.get("llm_provider"))),
            "gemini_api_key": _source_of("gemini_api_key", eff, bool(eff.get("gemini_api_key"))),
            "gemini_model": _source_of("gemini_model", eff, bool(eff.get("gemini_model"))),
            "main_model": _source_of("main_model", eff, bool(eff.get("main_model"))),
            "ollama_base_url": _source_of("ollama_base_url", eff, bool(eff.get("ollama_base_url"))),
        },
    }


def _source_of(key: str, eff: Dict[str, Any], present: bool) -> str:
    if not present:
        return "unset"
    if key in _session_overrides:
        return "session"
    if key in load_persisted():
        return "saved"
    return "environment"
