# Master Agent System — Project Status

Factual status as of Phase 3 (ERP-agnostic boundary) completion. No
aspirational features are listed as present.

```text
Phase 1: COMPLETE
Phase 2: COMPLETE
Phase 3: ERP-AGNOSTIC BOUNDARY COMPLETE
Phase 3: REAL ERP INTEGRATION NOT STARTED
Phase 4: NOT STARTED
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

### Phase 4

QA node and LLM-based Supervisor routing have NOT been implemented.
Routing remains deterministic keyword matching; `qa_flags` exists in
`AgentState` as reserved space only.

## Current Git History

```text
<this commit>   Phase 3: establish ERP-agnostic integration boundary
<docs commit>   docs: finalize phase 2 documentation and cleanup
e69ec5c         Phase 2: feasibility chain with graceful web-only degradation and query rewrite
f0a051f         Phase 1: LangGraph books + research agents with SearXNG/Crawl4AI tools
```
