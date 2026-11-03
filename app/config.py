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

# --- Planner (Phase 7, MASTER_PLAN §5.2) ---
# "auto": planner runs only for requests the supervisor classifies complex
# (MASTER_PLAN Phase 7 complex detection). "off": planner disabled entirely
# (graph takes the direct execution path; graph behavior identical to
# Phase ≤6). "always" (debug/testing only): bypass complex detection.
PLANNER_MODE = os.getenv("PLANNER_MODE", "auto")
if PLANNER_MODE not in ("auto", "off", "always"):
    raise ValueError("PLANNER_MODE must be 'auto', 'off' or 'always'")

# Hard cap on plan size (MASTER_PLAN §5.2: recommend 6). Larger LLM plans are
# deterministically rejected (fallback default plan is used instead).
MAX_PLAN_TASKS = int(os.getenv("MAX_PLAN_TASKS", "6"))

# Per-task retry budget (MASTER_PLAN §5.6). Phase 8 does NOT retry (a failed
# task is recorded and the plan continues); the constant exists so Phase 8's
# hard dispatcher iteration cap is expressed as MAX_PLAN_TASKS *
# (1 + MAX_TASK_RETRIES) and rises when Phase 12 adds retries.
MAX_TASK_RETRIES = int(os.getenv("MAX_TASK_RETRIES", "2"))

# --- Services (continued after planner block so config stays grouped) ---
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

# --- Workspace coding capability (2026-10-30) -----------------------------
# ABSOLUTE path of the ONE directory the coding workflow may read (and, only
# after an approval, patch). Empty (the default) DISABLES the capability: no
# routing decision changes and no file is ever read. There is deliberately no
# default workspace — the operator must choose one explicitly.
CODING_WORKSPACE_ROOT = os.getenv("CODING_WORKSPACE_ROOT", "")

# --- Books retrieval tuning ---
CHROMA_COLLECTION = "books"
# Below this cosine similarity, books_agent refuses per the RAG grounding
# rules (spec section 2.1). NOT_FOUND_PHRASE is the MANDATED user-facing
# refusal: it must explicitly state what happened when the requested
# information is absent, and must never carry a references footer (a
# refusal presents no evidence, so it cites none).
BOOKS_MIN_SCORE = 0.35
BOOKS_TOP_K = 3

# --- General-route evidence (2026-11-01) ----------------------------------
# The general route checks the SAME persisted index before answering from
# model knowledge. A passage is only treated as evidence when its similarity
# clears the floor AND it shares enough of the question's own content words
# (a score alone is not support). Below either bound the answer says it is
# from model knowledge instead of citing a passage that does not cover it.
EVIDENCE_TOP_K = int(os.getenv("EVIDENCE_TOP_K", "4"))
EVIDENCE_MIN_SCORE = float(os.getenv("EVIDENCE_MIN_SCORE", "0.45"))
EVIDENCE_MIN_COVERAGE = float(os.getenv("EVIDENCE_MIN_COVERAGE", "0.6"))
# Bound on the passage text placed in one prompt (and kept in state for QA).
EVIDENCE_PASSAGE_CHARS = int(os.getenv("EVIDENCE_PASSAGE_CHARS", "6000"))
# Per-PASSAGE bound for the persisted evidence snapshot (2026-11-02). This is
# deliberately NOT the prompt budget: using one cumulative budget for the
# prompt AND for persistence silently dropped the passages that did not fit
# (the live 3-of-4 truncation). Persistence keeps every consulted passage with
# its own cap, and marks a passage `truncated` instead of dropping it.
EVIDENCE_SNAPSHOT_CHARS = int(os.getenv("EVIDENCE_SNAPSHOT_CHARS", "8000"))
NOT_FOUND_PHRASE = "The requested information was not found in the books."

# --- Orchestrated knowledge fallback (2026-10-25 additional phase) --------
# "1" (default): when the books agent REFUSES on the direct books route and
# the question is not local-only / document-scoped, the graph hands the
# ORIGINAL question to the existing research agent (exactly one transition —
# books -> research -> END in app/graph.py, so fallback loops are impossible).
# "0": pre-fallback behavior, a books refusal is terminal.
# Local-only / document-scoped questions never fall back regardless of this
# switch (app/supervisor.py _LOCAL_ONLY_RE / _DOC_DEICTIC_RE).
BOOKS_WEB_FALLBACK = os.getenv("BOOKS_WEB_FALLBACK", "1") not in (
    "0", "false", "off", "no"
)

# --- Ingest integrity / resource limits (M2) -------------------------------
# Both caps are enforced BEFORE a file's text is decoded: the size check runs
# on stat() alone, and the page cap runs on the PDF xref/trailer (pypdf, already
# present via llama-index-readers-file) before any page content is extracted.
# The sample corpus is under 40 KB per file, so these are crash guards, not
# tuning knobs. A value <= 0 disables the corresponding cap.
INGEST_MAX_FILE_MB = int(os.getenv("INGEST_MAX_FILE_MB", "50"))
INGEST_MAX_PAGES = int(os.getenv("INGEST_MAX_PAGES", "500"))
# Document types the ingest understands (the reader has no OCR/vision path).
INGEST_EXTENSIONS = (".pdf", ".docx")
# Manifest schema version. v1 was a flat {absolute_path: mtime_ns} map that
# recorded a file as done even when the reader had silently skipped it. v2
# records per-file status + persisted chunk counts; v1 entries are migrated
# explicitly and their chunk counts verified against the vector store instead
# of being trusted.
INGEST_MANIFEST_VERSION = 2

# --- M4B semantic verification (experimental) -----------------------------
# "off" (default): the books agent behaves byte-identically to M3 — the
# deterministic token/metadata gate is the only support check.
# "local": every RETAINED claim is additionally classified by a bounded
# verifier (app/verifier.py) through the same get_llm() factory. The
# verifier is monotone: it can only REMOVE claims (never add, rewrite, or
# bless content) and it never authors citations. Read at call time so tests
# can toggle it per-test.
BOOKS_VERIFIER = os.getenv("BOOKS_VERIFIER", "off")
if BOOKS_VERIFIER not in ("off", "local"):
    raise ValueError("BOOKS_VERIFIER must be 'off' or 'local'")
# Hard fail-safe bounds. Each is enforced inside app/verifier.py (claims
# beyond the cap are NOT verified — fail-closed: removed and disclosed).
BOOKS_VERIFIER_MAX_CLAIMS = int(os.getenv("BOOKS_VERIFIER_MAX_CLAIMS", "6"))
BOOKS_VERIFIER_EVIDENCE_MAX_CHARS = int(
    os.getenv("BOOKS_VERIFIER_EVIDENCE_MAX_CHARS", "2000")
)
BOOKS_VERIFIER_CLAIM_TIMEOUT = float(
    os.getenv("BOOKS_VERIFIER_CLAIM_TIMEOUT", "10")
)
BOOKS_VERIFIER_TOTAL_DEADLINE = float(
    os.getenv("BOOKS_VERIFIER_TOTAL_DEADLINE", "30")
)
# M4B-H hardening: process-wide bound on CONCURRENT verifier model calls.
# Claims within one answer are verified sequentially (one permit at a time),
# but a call that exceeds its timeout cannot be cancelled: its daemon thread
# keeps the underlying HTTP request alive until the transport gives up. This
# budget bounds that lingering work to BOOKS_VERIFIER_MAX_INFLIGHT requests
# across the WHOLE process (all answers, not just one). Once exhausted, any
# further verification fails fast as `unavailable(busy)` — fail-closed, never
# a false "supported". A value < 1 is invalid.
BOOKS_VERIFIER_MAX_INFLIGHT = int(os.getenv("BOOKS_VERIFIER_MAX_INFLIGHT", "6"))
if BOOKS_VERIFIER_MAX_INFLIGHT < 1:
    raise ValueError("BOOKS_VERIFIER_MAX_INFLIGHT must be >= 1")
