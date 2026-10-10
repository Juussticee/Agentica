# Master Agent System — Project Status

Factual status after the recovery session (post Phase 6 + Control Center
foundation). No aspirational features are listed as present.

```text
Phase 1: COMPLETE
Phase 2: COMPLETE
Phase 3: ERP-AGNOSTIC BOUNDARY COMPLETE
Phase 3: REAL ERP INTEGRATION NOT STARTED
Phase 4: COMPLETE (workflow/orchestration + deterministic QA test suite)
Phase 5: COMPLETE (production qa_node in the graph; 12 tests)
Phase 6: COMPLETE (opt-in LLM routing; 11 tests; keyword fallback intact)
Control Center GUI-1: COMPLETE (events + runner + 18 tests)
Control Center GUI-2: RESTORED (SQLite store + FastAPI v1 + 14 tests)
Control Center GUI-3: RESTORED (React shell, dashboard, history, detail)
Control Center GUI-16: PARTIAL (AI settings + 7-phase readiness + 21 tests)
GUI-4..19: NOT STARTED
```

## Completed

### Phase 1

- **Environment**: Python 3.11 venv; Ollama (embeddings) and a SearXNG Docker
  container (host port 8888 → container 8080) verified working.
- **Book ingestion** (`ingest_books.py`): PDF + DOCX via SimpleDirectoryReader
  with `llama-index-readers-file`; LlamaIndex chunking; Ollama
  `nomic-embed-text` embeddings; binary-garbage guard (NUL bytes, magic
  signatures, printable/letter ratios) fails loudly instead of indexing bytes.
- **Chroma persistence**: cosine-similarity collection in `storage/`,
  incremental re-ingest driven by `storage/.index_manifest.json`
  (filename → mtime).
- **Books Agent**: LlamaIndex query engine over Chroma; shared-LLM bridge
  (`_GetLlmBridge`) because the LangChain↔LlamaIndex bridge package is broken
  with modern LangChain; `References: file, page` provenance; refusal with the
  exact phrase "That information was not found in your books." on out-of-corpus
  questions (prompt-level instruction + `BOOKS_MIN_SCORE` score floor).
- **SearXNG client** (`app/tools/searxng_client.py`): JSON API search,
  fail-soft (never raises).
- **Crawl4AI extractor** (`app/tools/crawl_extractor.py`): sync wrapper over
  the async crawler, fail-soft, ASCII-sanitized logging for cp1256 consoles.
- **Research Agent**: SearXNG → top-result crawl → Gemini synthesis with
  source URLs; snippet fallback when all crawls fail.
- **Supervisor**: deterministic keyword routing (books-pattern vs research
  fallback).
- **LangGraph wiring**: single compiled StateGraph, supervisor → agent → END.
- **CLI** (`main.py`): interactive, `--demo`, `--once`; UTF-8 reconfigure;
  `GoogleRateLimitError` caught with a quota message.

### Phase 2

- **Feasibility routing**: supervisor matches feasibility-style questions
  first and sets `route="feasibility"`, `sub_routes=["research", "books"]`.
- **State accumulation**: `agent_answers` + `with_agent_output()` so research
  and books outputs coexist in state (neither overwrites the other) and
  feasibility can consume both.
- **Feasibility synthesis**: Market / Financial / Risk study written to a
  timestamped `.docx` in `reports/`; section extraction from the model's
  markdown with a "(section not produced by the model)" placeholder guard.
- **Graph chain**: research → books → feasibility via conditional edges
  (`_after_research` / `_after_books`); single-agent routes unchanged.
- **Web-only fallback**: when books output is missing, a placeholder, or the
  not-found refusal, feasibility switches to a dedicated web-only prompt
  (told the library had no relevant material; asked to note where internal
  data is required), stamps a **Limitations** section in the `.docx`, and
  prefixes the answer with `⚠ Web-only study: ...`. Missing internal/company
  data is represented as a limitation, never fabricated.
- **Query rewrite**: `get_llm(fast=True)` (minimal thinking on Gemini 3)
  rewrites the question into a tight search query before SearXNG; any failure
  falls back to the raw question; synthesis still uses the ORIGINAL question.
- **Search noise filtering**: known noisy hosts (youtube, dictionary sites,
  Google support/store pages) filtered from results before crawling.

### Phase 3

- **ERP-agnostic boundary** (`app/erp/`): vendor-neutral READ-ONLY
  `ERPAdapter` interface (`name` / `search` / `get_record` — no write
  methods exist by construction), frozen `ERPRecord`, an `ERPError` taxonomy
  (invalid_request / adapter_unavailable / permission_denied /
  record_not_found / timeout), and `get_erp_adapter()` selected by the
  `ERP_ADAPTER` switch in `app/config.py` (Phase 3 ships only `"mock"`).
- **Mock adapter** (`app/erp/mock_adapter.py`): deterministic, read-only,
  fictional fake records, no network, no credentials.
- **ERP agent** (`app/agents/erp_agent.py`): replaces the Phase 1 stub;
  talks only to the adapter interface, deterministic formatting with NO LLM
  call (zero Gemini quota), fixed safe messages per error category (no
  credential/internal-detail leakage), provenance via the existing
  `sources` mechanism + `agent_answers["erp"]`, explicit mock-data note on
  data answers.
- **Minimal ERP action logging** (`app/erp/log.py`): one line per operation
  (timestamp, op, adapter, status, duration, error category or record
  count) to the git-ignored `logs/erp_actions.log`; never secrets, query
  text, or record data; fail-soft.
- **Routing**: `_ERP_RE` keyword list added to the supervisor (order:
  feasibility → ERP → books → research fallback); graph wiring unchanged
  (supervisor → erp → END existed since Phase 1).
- **AgentState: no changes** — existing keys were sufficient (explicitly
  documented in PHASE3_ERP_PLAN.md).
- **Tests**: `scripts/test_phase3_erp.py` — 35 deterministic zero-quota
  tests (routing, mock adapter, reads, error paths, permission denied,
  provenance, write protection, factory, logging).
- **No real ERP connection, no vendor selection, no MCP, no ERP SDKs, no
  credentials/API keys, no network calls.** The actual ERP has not been
  specified; see [PHASE3_ERP_PLAN.md](PHASE3_ERP_PLAN.md),
  "ERP-Specific Details Deferred".

## Current Architecture

Single `AgentState` (`app/state.py`), single compiled graph (`app/graph.py`),
single LLM factory (`app/llm_client.py`), single config source
(`app/config.py`). Routing decisions only in the supervisor; agents never
call agents; tool clients never raise. Full details and diagrams:
[ARCHITECTURE.md](ARCHITECTURE.md).

```text
START → supervisor → books_agent ────────────────────────→ END
        supervisor → research_agent ─────────────────────→ END
        supervisor → research → books → feasibility ─────→ END   (feasibility route)
        supervisor → erp_agent → ERP adapter interface → MockERPAdapter → END   (ERP route;
                                   the mock is replaced by the real adapter later)
```

## Known Limitations

1. **Gemini free-tier quota** (20 requests/day for gemini-3-flash-preview) is
   easily consumed by repeated multi-agent testing; a feasibility run costs 5
   calls (1 rewrite + 1 research synthesis + 1 books + 1 feasibility + margin).
   Most test suites therefore run on fakes at zero quota.
2. **SearXNG result quality varies**; the local container may return sparse or
   low-relevance results depending on upstream engines.
3. **Research answers depend on successfully crawled web content**; when
   extraction fails the system falls back to short snippets, and answers are
   not guaranteed authoritative.
4. **Books Agent is limited to the ingested corpus** (currently two small
   sample documents); anything outside it is refused by design.
5. **Ollama `nomic-embed-text` score distributions are narrow**; refusal
   therefore relies on the prompt-level exact phrase plus the
   `BOOKS_MIN_SCORE` floor, not on similarity alone. Embeddings are NOT on
   Gemini (known gap, intentionally not worked around).
6. **Real ERP integration does not exist** — only the vendor-neutral
   boundary plus the deterministic mock (Phase 3); see below.
7. **QA node and LLM-based routing do not exist** — see below.

## Not Yet Implemented

### Phase 3 — Real ERP integration

The ERP-agnostic boundary is COMPLETE (see Completed above); the REAL ERP
integration has NOT been started. The ERP vendor is unspecified — the
deterministic mock adapter (`app/erp/mock_adapter.py`) provides the data
until a real adapter is implemented. No MCP, no vendor SDK, no credentials.
See [PHASE3_ERP_PLAN.md](PHASE3_ERP_PLAN.md).

### Phase 5 note
The deterministic QA logic (Phase 4's test-suite function) now lives in
`app/qa_node.py` as a production graph node; every route's terminal edge
passes through it. `test_phase4_workflow_qa.py` imports the production
implementation instead of a local copy.

### Phase 6 note
Routing mode is controlled by `ROUTING_MODE` in `.env`
(`keyword` default; `llm` enables the LLM classifier with deterministic
fallback on any error/malformed/unknown reply). Default behavior is
byte-identical to keyword routing.

### Control Center status (GUI)

* **Restored and verified**: execution lifecycle, append-only event log,
  SQLite persistence (`storage/control_center.sqlite3`), FastAPI v1
  endpoints (health/executions/events + AI settings/test), React frontend
  (dashboard/new/executions/detail/settings), honest failure reporting.
* **Live checks performed**: executions answered via the real graph (ERP
  mock route, research route via SearXNG + Ollama `qwen3:0.6b`); AI
  connection test passed all seven phases against the live Ollama endpoint;
  browser UI rendered persisted runs.
* **Not implemented yet**: GUI-4..GUI-19 per [CONTROL_CENTER_PLAN.md]
  (CONTROL_CENTER_PLAN.md) — the preview is an initial GUI-1/2/3 slice plus
  a partial GUI-16 only.

### Legacy live-test scripts

`verify_graph.py`, `test_phase1_e2e.py`, `test_phase1_proxy.py`
`check_books_agent.py`, `probe_books_prompt.py`,
`demo_rewrite_quality.py` are historical live scripts; they may need
updates for the Phase 5 QA-terminated graph (qa node now always runs after
the last agent).

## Current Git History

```text
<this commit>   Phase 3: establish ERP-agnostic integration boundary
<docs commit>   docs: finalize phase 2 documentation and cleanup
e69ec5c         Phase 2: feasibility chain with graceful web-only degradation and query rewrite
f0a051f         Phase 1: LangGraph books + research agents with SearXNG/Crawl4AI tools
```
