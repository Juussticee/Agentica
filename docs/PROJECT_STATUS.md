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
Phase 7: COMPLETE (conditional planner; 35 tests; plan data only)
Phase 8: COMPLETE (dynamic plan execution dispatcher; 21 tests)
Control Center GUI-1: COMPLETE (events + runner + 18 tests)
Control Center GUI-2: RESTORED (SQLite store + FastAPI v1 + 14 tests)
Control Center GUI-3: RESTORED (React shell, dashboard, history, detail)
Control Center GUI-16: PARTIAL (AI settings + 7-phase readiness + 21 tests)
Operations dashboard (GUI-5/GUI-7 slice): RESTORED (real service health,
  event-derived agent activity, execution sequence; 14 ops tests)
RAG-1 (retrieval baseline & evaluation): COMPLETE (gold set + fixtures +
  evaluator + frozen baseline v3; 38 offline tests; audit + completion
  report; answer-level evaluation = RAG-AE, NOT STARTED)
GUI-4, GUI-6, GUI-8..19: NOT STARTED
```

## Completed

### Phase 1

- **Environment**: Python 3.11 venv; Ollama (embeddings) and a SearXNG Docker
  container (host port 8888 → container 8080) verified working.
- **Book ingestion** (`ingest_books.py`): PDF + DOCX via SimpleDirectoryReader
  with `llama-index-readers-file`; LlamaIndex chunking; Ollama
  `nomic-embed-text` embeddings; binary-garbage guard (NUL bytes, magic
  signatures, printable/letter ratios) fails loudly instead of indexing bytes.
  M2 integrity: per-file `indexed` / `partial` / `failed` status backed by
  store-verified chunk counts (a reader that silently skips a file can no
  longer be recorded as a success); pages with no extractable text reported by
  page rather than assumed blank (no OCR in this build); size and PDF page caps
  enforced before the reader loads the file; replacement inserts before it
  deletes, so a failed re-ingest never drops the previous version; exit code 1
  when any file failed.
- **Chroma persistence**: cosine-similarity collection in `storage/`,
  incremental re-ingest driven by `storage/.index_manifest.json` (schema v2:
  per-file mtime, status, extracted page counts, pages without extractable
  text, and the chunk count *verified against the vector store*; schema-v1
  entries are migrated and re-verified rather than trusted).
- **Books Agent**: LlamaIndex query engine over Chroma; shared-LLM bridge
  (`_GetLlmBridge`) because the LangChain↔LlamaIndex bridge package is broken
  with modern LangChain; `References: file, page` provenance; refusal with the
  exact phrase "That information was not found in your books." on out-of-corpus
  questions (prompt-level instruction + `BOOKS_MIN_SCORE` score floor).
  M3 citation integrity: the footer is rebuilt only from the retrieved nodes'
  own metadata, and only for nodes that actually supported a retained claim;
  model-written filenames/pages/tables are validated against the retrieved
  metadata (mismatch ⇒ the claim is dropped), any References/Sources block
  emitted by the model is stripped, prompt-injection cues in retrieved text
  are dropped as content, missing metadata yields an explicit "attribution
  unavailable" marker, and refusals never carry a footer. Claim support is a
  deterministic token/metadata gate, NOT semantic entailment. M4B adds an
  OPT-IN bounded verifier (`BOOKS_VERIFIER=local`, default `off` = byte-
  identical M3 behavior): each retained claim is classified
  (supported / contradicted / insufficient_evidence / uncertain) against
  only its assigned passages via the shared LLM factory, with hard bounds
  (6 claims, 2000 evidence chars, 10 s/claim, 30 s total, zero retries,
  daemon-thread timeout — langchain_ollama has none). M4B-H hardening: a
  process-wide 6-permit in-flight budget with fail-fast `busy` (no thread
  spawned when exhausted; permits released exactly once including zombie
  time), a factory-level httpx transport bound that reclaims abandoned
  requests at claim-timeout + 5 s (non-streaming invoke on ollama — the
  installed ChatOllama streams by default, under which the read timeout
  bounds only inter-chunk gaps; SDK retries pinned to 0 on gemini, whose
  installed default is six). The verdict is
  monotone: it can only REMOVE claims (fail-closed), never add, rewrite or
  bless content, and its reasons never reach users. The confirmed D1
  (relation reversal) and D2 (numeric conflict) defects are closed on this
  path; a version-controlled benchmark (38 cases, three layers per case:
  deterministic baseline, scripted verifier verdict, integrated design
  target) is green on both drift gates, with 17 annotated baseline-gaps
  that only the verifier path closes; the optional local-model experiment
  at the production 10 s bound shows model latency (not verdict quality)
  is the binding constraint — and it is a small-sample latency run, not a
  measurement of semantic accuracy; real-model entailment quality remains
  unverified. G1: absolute `file_path` metadata no longer
  reaches the LLM context (query-time allowlist; stored metadata and
  citation provenance untouched).
  M4C claim-level evidence attribution (verification-enabled mode only):
  every retained claim keeps its OWN evidence set — the deterministic
  covering set for its sentence, narrowed by the verifier's per-claim
  `evidence_ids`, which are whitelist-validated against exactly the
  evidence supplied for that claim (an unknown id, or a valid id
  belonging to a different claim, degrades to malformed → fail-closed
  removal + fixed disclosure — strict JSON validation and id whitelisting
  not weakened). The footer in this mode is one numbered line per
  retained claim, `[n] <ref>[; <ref>…] (verifier-supported: <claim>)`,
  built from that claim's retained nodes only, so a source supporting one claim is
  never presented as supporting another. The parenthetical is
  deliberately ATTRIBUTED wording: it reports the bounded verifier's
  per-claim judgment on the shown passages — a heuristic/small-model
  verdict — not independently established semantic entailment; a
  single claim genuinely backed by several passages keeps all of them
  on its own line; a claim with no
  adequate attributable evidence is removed/disclosed (or refused
  outright when nothing survives), and refusals still carry no footer.
  Contradiction handling is claim-scoped and kept separate from
  attribution (a valid citation can still contradict its own claim): a
  reversed or numerically conflicting claim is removed with the conflict
  note while sibling claims keep exactly their own citations. This is
  claim-level attribution, NOT semantic entailment: the deterministic
  gate stays token/metadata coverage, clause-level splitting and general
  NLI/arithmetic are deliberately absent, and unresolved cases resolve to
  removal + fixed disclosure instead of a confident guess.
  `BOOKS_VERIFIER=off` (still the default) is unchanged: byte-identical
  M3 answer-level footer. Evidence-ID validation, citation construction,
  covered/unresolved contradiction cases, exact commands, and the
  deterministic-vs-scripted-vs-live test distinction are documented in
  docs/TESTING.md ("M4C — what the tests prove"); 35 dedicated offline
  tests in `scripts/test_m4c_attribution.py` assert the actual
  claim→evidence mapping, and the benchmark checks an exact
  `expected_mapping` on the verified layer (6/6).
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

### RAG-1 (retrieval baseline & evaluation — implemented)

Deterministic retrieval evaluation for the books pipeline, measured
against a frozen reference instead of intuition:

- **Gold set** `benchmarks/rag_gold_v1.json` — 32 labeled cases (25
  answerable / 7 unanswerable; 17 synthetic) with `required_facts`,
  `citation_expectation`, per-case provenance and a 9-item gap list;
  SHA-256 pinned (`16e25677…`) and immutable to the code under test.
- **Fixture corpus** `benchmarks/rag_fixtures/` built by
  `scripts/make_rag_fixtures.py` — byte-deterministic synthetic documents
  (table, columns, scan, headings, adversarial) so retrieval behavior can
  actually discriminate.
- **Harness** `scripts/eval_retrieval.py` — throwaway temp eval collection
  with instrumented Chroma clients (fails hard on any non-temp path),
  stat + SHA-256 guards on the production store and gold set before/after,
  embedding-model identity + config + git metadata recorded, and
  **output safety**: an existing `--out` report is refused with exit 3
  before any work unless `--force` is passed.
- **Frozen baseline** `benchmarks/rag_baseline_v3.json` (`aa89c37b…`,
  `artifact_schema.version=3` / `ordered-top5-per-case`): real hit@1
  0.9091 (10/11), hit@3 1.0, MRR 1.0, nDCG@5 1.0; synthetic hit@1 0.8571
  (12/14), hit@3 1.0, MRR 0.9643, nDCG@5 0.9736 — denominators and
  caveats in `docs/TESTING.md` ("RAG-1 frozen baseline (v3)").
- **Tests** `scripts/test_rag_eval_harness.py` — 38 offline tests (metrics
  math, guards, fixture determinism, isolation proof, artifact-evidence
  recalculation, overwrite safety, SHA-256 pins for gold + v1/v2/v3); the
  frozen-baseline gate `benchmark_verification.py --check` reports
  `MATCH`.
- **Docs** `docs/RAG_BASELINE_AUDIT.md` (v1 audit preserved + §7 v2/v3
  addendum), `docs/RAG_1_COMPLETION_REPORT.md` (phase report),
  `docs/TESTING.md` (recorded metrics).
- **Not measured:** answer-level correctness and refusal — that is
  **RAG-AE** (see "Not Yet Implemented"). No production code changed:
  `app/`, `ingest_books.py`, prompts and thresholds are untouched.

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
   boundary plus the deterministic mock (Phase 3); see Completed above.
7. **LLM-based routing is opt-in and off by default** (`ROUTING_MODE=keyword`;
   see the Phase 6 note below). The Phase 7 planner is also fail-soft and
   never blocks a request (see the Phase 7 note below); per-execution
   approval modes remain unimplemented (GUI-10).
8. **Phase 8 dispatch executes only whitelisted capabilities inside the
   hard visit cap.** `synthesis` currently dispatches to the existing
   feasibility agent (terminal consumer) — the dedicated Synthesizer is
   Phase 10. With a LIVE Ollama server, the qwen3:0.6b planner can emit
   valid but minimal LLM plans (e.g. research+synthesis without a books
   task) for complex questions, so dispatched runs can differ from the old
   fixed feasibility chain — QA flags any missing/not-completed parts;
   see docs/TESTING.md "Known environment-dependent flake".
9. **RAG-1 measures retrieval ranking only.** The abstention proxy is a
   score-floor check (0.35 unreachable — every top-1 score ≥ 0.65) and says
   nothing about answer-level refusal; real-corpus retrieval metrics cannot
   discriminate (2 single-chunk documents); and coverage-family metrics
   changed meaning with the page-metadata-aware relevance rule (v1 vs
   v2/v3 values are not comparable — `docs/RAG_BASELINE_AUDIT.md` §7).
   Answer quality is **RAG-AE**, not started.

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

### Phase 7 note (planner — implemented)
The conditional planner ships per MASTER_PLAN §5.2. `app/planner.py`
contains `needs_planning()` (complex detection: PLANNER_MODE gate →
deterministic multi-objective heuristic → optional LLM tie-break when
ROUTING_MODE=llm), `default_plan()` (deterministic route-derived fallback),
`parse_plan()` (defensive JSON-in-text decoding), `validate_plan()` (pure
deterministic validator: whitelist capabilities, unique/well-formed
task_ids, existing + acyclic depends_on, satisfiable required_inputs,
MAX_PLAN_TASKS=6 cap), and `planner_node` (graph node returning ONLY
`{"plan": ...}` — plan data, never executed here). The supervisor marks
genuinely multi-objective requests with the new `complex: True` state key
(never written False); the graph adds one conditional hop
supervisor → planner → same route table. Simple/multi requests bypass the
planner entirely — behavior for them is byte-identical to Phase ≤6.
Any planner failure (LLM down, malformed JSON, failed validation) falls
back to `default_plan()`; the request is still answered. NO task execution:
planned tasks are NOT run yet — Phase 8's dispatcher consumes the plan in
state. Config additions: `PLANNER_MODE` (auto/off/always; auto default),
`MAX_PLAN_TASKS` (6). New state key: `plan` (justified in MASTER_PLAN §5.2).
35 new deterministic tests in `scripts/test_phase7_planner.py` (zero
quota); full regression 185 passed + 1 skipped.

### Phase 8 note (dispatcher — implemented)
`app/graph.py` now executes validated plans: a `dispatcher` node picks ONE
task whose `depends_on` are all completed, runs its whitelisted node via
the literal code mapping `CAPABILITY_NODES` (`research→research`,
`books→books`, `erp→erp`, `synthesis→feasibility` — plan data can never
name a node), records the honest per-task status/result ON the plan dict
(`pending/completed/failed/blocked/unsupported`; failed/blocked deps make
dependents blocked, never executed), and returns to itself until every
task is terminal — then routes to QA, which now also verifies plan
completion and explicitly flags an unavailable AI planning service
(user-requested server-down visibility) instead of degrading silently.
Revisits never re-execute completed tasks; a raising node is recorded
failed without crashing the run; and the hard visit cap
`MAX_PLAN_TASKS * (1 + MAX_TASK_RETRIES)` (`app/config.py`) fail-safes any
pathological plan instead of looping. Simple and multi requests keep
byte-identical pre-Phase-7 edges. 21 new deterministic tests in
`scripts/test_phase8_execution.py` (zero quota). Full regression: 184
passed + 1 skipped (see TESTING.md known-flake note for the
Ollama-running environment). Config addition: `MAX_TASK_RETRIES` (2).

### RAG-AE note (answer-quality evaluation — NOT started)

RAG-1 proves retrieval ranking only. **RAG-AE** is the separately
registered track (defined in `docs/MASTER_PLAN.md` §5.1; deliberately kept
outside the numbered RAG-1…RAG-6 sequence because the name "RAG-2" already
means hybrid retrieval there and structure-aware chunking in
`docs/RAG_IMPLEMENTATION_PLAN.md`) that will measure end-to-end answers:
factual correctness against `required_facts`, evidence fidelity per claim,
citation correctness, abstention precision/recall on the 7 unanswerable
cases, and false-answer rate — reported as five separate metric families,
never one composite score. It must not change the production answer path,
`BOOKS_MIN_SCORE`, prompts, the gold set or the frozen baselines, and it
needs its own authorization plus a model budget (local Ollama preferred on
4 GB VRAM; Gemini spends free-tier quota). Plan: one new offline suite +
new `scripts/eval_answers.py` + a recorded artifact — see
[RAG_1_COMPLETION_REPORT.md](RAG_1_COMPLETION_REPORT.md) §7.

### Control Center status (GUI)

* **Restored and verified**: execution lifecycle, append-only event log,
  SQLite persistence (`storage/control_center.sqlite3`), FastAPI v1
  endpoints (health/executions/events + AI settings/test + the three
  operations endpoints below), React frontend (overview/new/executions/
  detail/settings + Services/Agents/Tasks & Sequence pages with grouped
  sidebar and mobile-responsive layout), honest failure reporting.
* **Operations dashboard restored (partial GUI-5/GUI-7)**: real bounded
  service-health probes (`app/services_health.py` — agentica_api, SQLite
  storage, Ollama `/api/tags`, SearXNG, ERP adapter in-process; states
  running/stopped/unreachable/degraded/misconfigured/not_configured/unknown
  with latency, failure reason, and a next diagnostic action),
  event-derived agent activity (`app/execution/events.py` now records
  `routing.decided` and `qa.completed` with `agent_id`; the Agents page
  shows live activity while an execution runs and stays visible on failed
  runs), and a truthful execution sequence endpoint
  (`/api/v1/executions/{id}/sequence`: created → supervisor → routed →
  agents → qa → completed, built from the recorded events — this build has
  no queue/planner dependency graph; that arrives with MASTER_PLAN
  Phase 7/8).
* **Live checks performed**: executions answered via the real graph (ERP
  mock route, research route via SearXNG + Ollama `qwen3:0.6b`); AI
  connection test passed all seven phases against the live Ollama endpoint;
  browser UI rendered persisted runs; Services page matched the real
  machine state (backend/SQLite/Ollama running, SearXNG shown per its
  actual availability); Agents page shows the honest idle/empty state
  between runs; Tasks & Sequence rendered the full recorded stage list for
  real execution IDs including a fresh run with the routed + qa stages;
  desktop and 375 px mobile viewports verified with no horizontal overflow.
* **Not implemented yet**: planner/dispatcher dependency-graph views (GUI-5
  core, blocked on MASTER_PLAN Phase 7/8), and the rest of GUI-4..GUI-19
  per [CONTROL_CENTER_PLAN.md](CONTROL_CENTER_PLAN.md) — the operations
  dashboard is a GUI-5/GUI-7 slice, not the full workspaces.

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
