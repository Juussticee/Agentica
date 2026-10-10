"""Central configuration — the ONLY place models/paths/ports are defined (spec 0/3).

Environment variables come from .env at the project root via python-dotenv.
NEVER hardcode an API key in any file; .env is git-ignored.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Load .env FIRST, with override=True, then resolve the API key in priority
# order: a key explicitly written in .env (GEMINI_API_KEY or GOOGLE_API_KEY)
# must win over any stale value inherited from the OS environment — dotenv
# comparisons on this machine found exactly that shadowing scenario.
_dotenv_values = dotenv_values(PROJECT_ROOT / ".env")
for _k, _v in _dotenv_values.items():
    if _v is not None:
        os.environ[_k] = _v


def _resolve_google_api_key() -> str | None:
    # 1) key written in the .env file (either name) always wins
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = _dotenv_values.get(name)
        if val:
            return val
    # 2) fall back to the process environment (either name)
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


GOOGLE_API_KEY = _resolve_google_api_key()

# Normalize the in-process environment so every library (langchain-google-genai
# included) sees the SAME valid key under either name — otherwise a stale
# OS-level GOOGLE_API_KEY leaks into libraries that read env directly.
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# --- LLM provider switch (user-requested deviation from original Ollama-only spec) ---
# "gemini" now, or "ollama" later. get_llm() in app/llm_client.py reads this.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Gemini settings (used when LLM_PROVIDER == "gemini").
# User-approved model choice (2026-09): Gemini 3 Flash preview.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
# NOTE: the API key itself is resolved ONCE above via _resolve_google_api_key()
# (.env values win over inherited OS env vars) — do not re-assign it here.

# --- Ollama models (original spec names; used when LLM_PROVIDER == "ollama") ---
MAIN_MODEL = os.getenv("MAIN_MODEL", "qwen2.5:7b-instruct-q4_K_M")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "qwen2.5:3b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embeddings still run locally via Ollama (KNOWN GAP: Gemini embeddings need
# separate setup — flagged to user, intentionally not worked around).
EMBED_MODEL = "nomic-embed-text"

# --- Services ---
# Local SearXNG container maps host port 8888 -> container 8080 (measured on
# this machine; the spec's default 8080 had nothing listening).
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")

# --- ERP boundary (Phase 3) ---
# Which ERPAdapter implementation get_erp_adapter() returns. Phase 3 ships
# ONLY the deterministic vendor-neutral "mock"; the real ERP (vendor still
# unspecified) will be added later as a new adapter module behind this switch.
ERP_ADAPTER = os.getenv("ERP_ADAPTER", "mock")

# --- Supervisor routing mode (Phase 6, MASTER_PLAN) ---
# "keyword" (default): deterministic keyword routing, byte-identical to
# Phase 1..5 behavior. "llm": opt-in — the supervisor additionally asks the
# configured LLM (get_llm(fast=True)) to classify the question; any LLM
# error, malformed output, or unknown label falls back to keywords
# deterministically. Default mode NEVER depends on a working AI provider.
ROUTING_MODE = os.getenv("ROUTING_MODE", "keyword")
if ROUTING_MODE not in ("keyword", "llm"):
    raise ValueError("ROUTING_MODE must be 'keyword' or 'llm'")

# --- Paths / storage ---
BOOKS_DIR = PROJECT_ROOT / "knowledge" / "books"
COMPANY_DIR = PROJECT_ROOT / "knowledge" / "company"
STORAGE_DIR = PROJECT_ROOT / "storage"            # created automatically by ingest
INDEX_MANIFEST = STORAGE_DIR / ".index_manifest.json"
ERP_LOG_PATH = PROJECT_ROOT / "logs" / "erp_actions.log"
REPORTS_DIR = PROJECT_ROOT / "reports"            # feasibility_agent .docx output

# --- Books retrieval tuning ---
CHROMA_COLLECTION = "books"
# Below this cosine similarity, books_agent answers "not found in your books".
BOOKS_MIN_SCORE = 0.35
BOOKS_TOP_K = 3
NOT_FOUND_PHRASE = "That information was not found in your books."
