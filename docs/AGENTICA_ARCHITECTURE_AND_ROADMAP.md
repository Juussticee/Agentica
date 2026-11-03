# Agentica — Architecture Assessment and Implementation Roadmap

**Date:** 2026-10-27 · **Status:** planning artifact (read-only inspection) ·
**Branch:** `main` @ `08d44b6`

> This document is the product of an inspection-only task. Nothing in it was
> implemented, and nothing outside this file (and the companion proposal
> [docs/execution_record.schema.json](execution_record.schema.json)) was
> created or modified. Every statement about the current system is traced to
> source code or to a recorded report; every forward-looking statement is
> labelled **[PROPOSAL]**. Read the "Evidence" column before trusting any
> capability claim.

---

## 0. Baseline inventory (Phase 0)

Recorded before any inspection work; used later for the integrity check.

**Repository state**

| Item | Value (verified 2026-10-27) |
|---|---|
| Repository root | project working copy (this directory) |
| Branch | `main` (no branch switch performed) |
| HEAD | `08d44b6` — *docs(rag): eval-gap inventory, gold v2 candidate, manual test guide* |
| Staged files | **0** (`git diff --cached` empty) |
| Commits / pushes | none performed |

**Pre-existing working-tree changes (19 modified files — all preserved):**

```
M app/agents/books_agent.py            M scripts/test_books_grounding_paraphrase.py
M app/agents/research_agent.py         M scripts/test_chroma_isolation.py
M app/config.py                        M scripts/test_citation_integrity.py
M app/graph.py                         M scripts/test_phase1_routing.py
M app/state.py                         M scripts/test_phase6_routing_llm.py
M app/supervisor.py                    M scripts/test_rag_gold_v2_candidate.json
M benchmarks/rag_gold_v2_candidate.json M scripts/test_self_refusal_recovery.py
M docs/RAG_EVAL_GAPS.md                M scripts/make_rag_fixtures.py
M docs/TESTING.md                      M scripts/test_agent_architecture.py
                                       M scripts/test_books_grounding.py
```

**Pre-existing untracked work (all preserved):** `.diag/`,
`benchmarks/rag_ae_{unit,integration}_{v1,v3}.json`,
`benchmarks/rag_gold_v2.json`, `benchmarks/rag_gold_v3.json`,
`benchmarks/rag_fixtures/fixture_multipage_table.pdf`,
`docs/GUI_RAG_TEST_REPORT_2026-10-24_1114.md`,
`docs/GUI_RAG_ROOT_CAUSE_2026-10-24_2349.md`,
`docs/GUI_RAG_FIX_VALIDATION_2026-10-25_1415.md`,
`docs/GUI_RAG_FIX_VALIDATION_2026-10-25_1702.md`, `docs/RAG_AE_EVALUATION.md`,
`knowledge/books/*.pdf` (3 PDFs), `scripts/eval_answers.py`,
`scripts/test_eval_answers.py`, `scripts/test_knowledge_fallback.py`.

**Frozen/production artifacts (read-only, untouched):** gold datasets
`benchmarks/rag_gold_v2.json` (2026-10-23 20:20), `rag_gold_v3.json`
(2026-10-23 21:46), `rag_gold_v2_candidate.json` (2026-10-23 21:52);
`knowledge/books/` source PDFs; `storage/.index_manifest.json`
(all `last_processed_at` = 2026-10-24, every `mtime_ns` matches disk).

**Verification run during this task (offline, no network):**

```
env\Scripts\python.exe -m pytest scripts -q -p no:cacheprovider
→ 661 passed, 2 skipped, 138 subtests passed in 206.66s, EXIT=0
```

This re-runs the canonical documented suite against the *current* tree and
reproduces the count recorded in `docs/GUI_RAG_FIX_VALIDATION_2026-10-25_1702.md`
§9.4 exactly, so the reports below can be trusted as describing the code that
exists today.

---

## 1. Executive summary — what Agentica can do today

Agentica is a local-first, single-process AI agent platform: a LangGraph
orchestrator behind a FastAPI server, a React control-centre GUI, a
Chroma-backed document RAG pipeline with grounding and citations, an
opt-in web-research agent, a deterministic ERP boundary (mock adapter), a
conditional planner/dispatcher, and a persisted execution/event store.

**Verified working today** (each backed by source + recorded GUI evidence):

1. **Document-grounded Q&A with citations.** Natural-language document
   questions route to the Books agent without trigger words, retrieve from
   the local Chroma index, run a deterministic token-grounding gate, and emit
   a `References: <file>, page <label>` footer built from retrieved chunk
   metadata. Evidence: 16/16 GUI runs routed `books` in the 2026-10-25 fix
   cycle, citations verified page-by-page against the PDFs.
2. **Honest abstention.** Unanswerable questions return the mandated phrase
   *"The requested information was not found in the books."* with no
   fabricated number and no footer (GUI T5, C, B1; proof-grade `gate_drop:`
   diagnostics).
3. **Bounded self-refusal recovery.** Exactly one evidence-primed retry for
   provable pure self-refusals; never twice, never after the grounding gate.
4. **Books → internet fallback with guards.** A structured
   `books_refusal` diagnosis (never a score alone, never a refusal sentence
   alone) lets the orchestrator hand the *original* question to the research
   agent exactly once; local-only and document-scoped requests never fall
   back; loops are structurally impossible (`books → research → END`).
5. **Execution history.** Every run is persisted in SQLite with an
   append-only event log (`execution.created → agent.* → routing.decided →
   qa.completed → execution.completed/failed`), and the GUI renders only
   that log (no invented progress).
6. **Planner + dispatcher.** Multi-objective requests are decomposed into a
   validated plan executed through a literal capability→node whitelist with
   a hard iteration cap; plans are persisted and rendered in the GUI.
7. **Evaluation harness.** Frozen gold datasets, an answer-quality harness
   (unit: scripted evidence through the real `books_agent`; integration:
   real retrieval into a temp store) with isolation guards, sha pins, and
   `network_attempts: 0`.
8. **Service/agent observability.** Bounded live probes (`/api/v1/services`),
   event-derived agent activity, per-execution sequence views.

**Not implemented (planned or aspirational):** JSON export of runs,
file-system/desktop capabilities, spreadsheet handling, any skill or agent
builder, a dynamic agent registry (explicitly prohibited by
`docs/AGENT_DEVELOPMENT.md`), a real ERP adapter, approval workflow,
knowledge management, chat, reports (all four shown as "Planned" in the GUI
sidebar), live answer-quality scoring, and any positive externally-sourced
answer observed end-to-end (the chain runs; the synthesis has always ended
in an honest abstention in this environment).

---

## 2. Verified architecture and agent/tool inventory

### 2.1 Stack

| Layer | Implementation | Evidence |
|---|---|---|
| CLI | `main.py` — builds the graph once, `--demo`/`--once`/interactive | `main.py` |
| Orchestration | LangGraph 1.2.12 `StateGraph(AgentState)`, `app/graph.py` | `app/graph.py`, pip list |
| State | `app/state.py` — one `TypedDict(total=False)` + `with_agent_output()` | `app/state.py` |
| Routing | `app/supervisor.py` — keyword router (default) or opt-in `ROUTING_MODE=llm` with deterministic fallback | `app/config.py` |
| Agents | `books`, `research`, `feasibility`, `erp` (+ test double) | `app/agents/` |
| QA gate | `app/qa_node.py` — deterministic, appends `qa_flags`, never routes | `app/qa_node.py` |
| Planner | `app/planner.py` — complex detection, defensive JSON plan parse, `validate_plan()`, deterministic fallback plan | `app/planner.py` |
| LLM | `app/llm_client.py` — single factory, `LLM_PROVIDER` = `gemini`\|`ollama`, `get_llm(fast=, timeout=)`, `llm_text()` | `app/config.py`, `.env.example` |
| RAG store | Chroma 1.5.9 (`storage/chroma.sqlite3`), collection `books`, LlamaIndex 0.14.25 query engine | `ingest_books.py`, `app/agents/books_agent.py` |
| Embeddings | Ollama `nomic-embed-text` (local) | `app/config.py` |
| Ingestion | `ingest_books.py` — manifest v2, incremental, per-file honest status, page caps, partial-write rollback | `ingest_books.py`, `app/config.py` |
| Web search | `app/tools/searxng_client.py` — SearXNG JSON API at `http://localhost:8888`, 15 s hardcoded read timeout, never raises | source |
| Web crawl | `app/tools/crawl_extractor.py` — Crawl4AI sync wrapper, `""` on failure | source |
| ERP | `app/erp/adapter.py` read-only ABC + `MockERPAdapter` (fictional records, no network) + `app/erp/log.py` | source |
| API | FastAPI 0.143 (`app/api/server.py`), v1 prefix, localhost bind, CORS limited to the Vite dev origin | source |
| Persistence | SQLite WAL `storage/control_center.sqlite3` — `executions` + `events` tables | `app/execution/store.py` |
| Runner | `app/execution/runner.py` — wraps every node, records real events, honest `ExecutionError` mapping | source |
| Frontend | Vite + React + TS, hash routing, pages: Overview, New Execution, Executions, Execution Detail, Tasks & Sequence, Services, Agents, Settings | `frontend/src/App.tsx` |
| Tests | 45 scripts under `scripts/` (pytest; 661 passed / 2 skipped / 138 subtests today) + frontend vitest suites | run in §0 |

### 2.2 The real execution path (traced in source)

```text
GUI "Start execution"
  → POST /api/v1/executions            (app/api/server.py, runs synchronously, 201)
  → GraphRunner.create/run             (event: execution.created / execution.started)
  → build_graph(node_wrapper=…)        (every node wrapped: agent.started / agent.completed /
                                        agent.failed + routing.decided after supervisor)
  → START → supervisor                 (keyword_route: feasibility > erp > local-only >
                                        current-info/explicit-web > books anchors > research)
      ├─ books      → Chroma top-k → grounding gate → (M5 bounded retry) → footer
      │                 └─ refusal + books_refusal diagnosis
      │                      └─ _after_books → fallback_decision() → research (at most once) → END
      ├─ research   → rewrite → SearXNG → noise filter → Crawl4AI → synthesis + URLs → END
      ├─ erp        → MockERPAdapter (read-only, no LLM) → END
      ├─ feasibility→ research → books → feasibility_agent → reports/*.docx → END
      └─ complex    → planner → dispatcher ⇄ {research|books|erp|feasibility}
                       (capability whitelist, hard visit cap) → QA
  → qa node appends qa_flags (plan-completion checks included)
  → GraphRunner persists status/route/answer/qa_flags/result/plan (+ events)
  → GUI Execution Detail renders the event-derived sequence, plan, flags, answer
```

### 2.3 Capability status: verified / partial / planned

| Capability | Status | Evidence |
|---|---|---|
| Keyword routing incl. natural document questions + current-info precedence | **Verified** | `app/supervisor.py`; 16/16 GUI runs in the fix cycle |
| LLM routing mode | **Implemented, opt-in, not exercised here** | `ROUTING_MODE=llm`, `scripts/test_phase6_routing_llm.py` (offline) |
| Grounding gate + abstention | **Verified** | `gate_drop:` diagnostics; GUI T5/C/B1 |
| Bounded recovery (M5) | **Verified bounded; effectiveness limited by model** | GUI A4/A6 second refusals |
| Citation footer from metadata, noise filtered | **Verified (GUI + tests)** | fix-cycle report §5–§7; `test_citation_integrity.py` |
| Books→research fallback, loop-safe, local-only respected | **Verified (GUI G1–G5b)** | fix-cycle report §9 |
| Positive externally-sourced answer | **Partial — chain proven, no positive synthesis observed** | report §9.6 bullet 2 PARTIAL, §9.7 |
| Execution persistence + event timeline | **Verified** | `test_gui1_events.py`, GUI |
| Plan persistence + GUI rendering | **Verified** | `test_gui12_plan_render.py` (20 tests) |
| JSON export of runs | **Not implemented** | no export endpoint/route/download in `frontend/src` |
| Tool-call-level tracing (search/crawl/retrieval events) | **Not implemented** (only `print()` to stdout logs) | `searxng_client.py`, `research_agent.py`, `books_agent.py` |
| Provenance (`sources`) persisted per execution | **Not persisted** (state-only; `update_execution` has no `sources` column) | `app/execution/store.py` |
| File-system / desktop tools | **Not implemented** | `app/tools/` contains only search + crawl |
| Spreadsheets (CSV/XLSX/Sheets) | **Not implemented** | no `openpyxl`/`pandas` usage in `app/` |
| Skill/agent builder, registry, versioning | **Not implemented**; dynamic registry explicitly prohibited | `docs/AGENT_DEVELOPMENT.md` §3, §10 |
| Real ERP | **Not implemented (vendor unspecified)** | `app/erp/adapter.py`, `ERP_ADAPTER=mock` |
| Approval / Knowledge / Chat / Reports GUI | **Planned only** | GUI sidebar "Planned (not implemented yet)" |

---

## 3. Current limitations and known failures (evidence-backed)

**RAG quality**

1. **Five false refusals remain** after the 2026-10-25 fix cycle (T3, A1,
   A2, A4, A6); three others were flipped to answers (T2, T4, A5). Each is
   diagnosed: `gate_drop: unsupported (missing tokens: main|essential|highlight)`
   (FR-1 class, stop-list not yet extended), T3's coverage block on abstract
   task nouns (needs a *design* decision, not a stop word), and the
   **model capability floor** — `qwen3:0.6b` re-refuses after an eligible,
   evidence-primed retry (A4, A6, both B2 attempts).
2. **Research positive-path never succeeded live.** G5's search hit the
   hardcoded 15 s client timeout because the local `searxng` container is
   `unhealthy` (~23 s `/search` latency); G5b's search succeeded (5 on-topic
   sources) but all 3 page crawls returned empty and the 0.6B model declined
   to synthesize from snippets alone.
3. **Corpus is tiny and partly broken:** 5 files in `knowledge/books/`;
   manifest `version 2` reports `NIST.AI.100-1.pdf` indexed (48 chunks),
   `orbital_kinetics_principles.docx` (1), `urban_beekeeping_field_guide.pdf`
   (1), and **`NIST.AI.600-1.pdf.pdf` + `nasa_systems_engineering_handbook_0.pdf`
   = `failed`, 0 chunks**. Repair/ingestion is explicitly out of scope until
   authorized. With `BOOKS_TOP_K=3` over ~50 chunks and two single-chunk
   documents, retrieval metrics are known to be weak discriminators
   (`docs/RAG_EVAL_GAPS.md`).
4. **No live answer-quality metric.** The harness's answer layer is
   deterministic/scripted; `local-model` mode exists but is opt-in and was
   not run. GUI categories are evidence-based, not numeric.

**Platform**

5. **No structured export** (deliverable of Phase 3 below) and no persisted
   provenance: `state["sources"]` never reaches the database.
6. **No tool-level observability**: search/crawl/retrieval/recovery/fallback
   are observable only as `print()` lines in a backend log file.
7. **Synchronous single-run API** — `POST /executions` blocks until the graph
   finishes (GUI shows "Starting…" for 40–250 s). `STATUS_CANCELLED` exists
   in the store but no cancellation path.
8. **No file-system capability at all** — by design today; the agent contract
   even forbids an agent reading a file it did not create.
9. **ERP is a mock**; approval workflow and knowledge management are GUI
   placeholders.

---

## 4. RAG stabilization and internet-fallback plan

### 4.1 Where each requirement stands (verified against source + reports)

**A. Natural-language routing — PASS, with known edges.**
`keyword_route()` precedence: feasibility → ERP → explicit local-only →
current-info/explicit-web → `_BOOKS_RE` (lexical + document-deictic) or
`_BOOKS_IDENTITY_RE` (AI RMF / trustworthy AI / NIST) → research.
Pinned offline by `scripts/test_phase1_routing.py` (incl. paraphrases and
research-retained cases: "latest news about NIST…", "transformers in modern
AI", "AI art", weather → research). *Residual risk:* the anchors are
corpus-specific regexes; a second corpus would need a data-driven identity
source rather than more regex (see §9 W1).

**B. Evidence-grounded answering — PARTIAL.**
Gates are in place and never bypassed; recovery is bounded to one retry;
abstention is preserved (G3, T5, C, B1). *Residual:* the three stop-list
tokens, T3's coverage arithmetic, and the model floor.

**C. Citation integrity — PASS as measured.**
Two-tier `_covering_nodes`, footer built only from validated retrieved-node
metadata, model-emitted reference blocks stripped before handling,
cross-document noise pinned by tests including the *rendered* answer.
*Residual:* the answered-path footer could not be re-observed in GUI after
the B2 fix (post-fix B2 runs refused), and page labels are printed folios
(verified) rather than printed+viewer pairs (documented in
`docs/GUI_RAG_ROOT_CAUSE_2026-10-24_2349.md` as an optional metadata
enrichment).

**D. Books→internet orchestration — implemented and GUI-verified.**
The decision table (9 states) is in the fix report §9.3; guards:
`BOOKS_WEB_FALLBACK` switch, structured diagnosis, direct-books-route only,
local-only and deictic blocks, `research_already_ran`, topology-limited to
one transition, query = original question, scalars-only diagnosis (no
passage text leaves the process), provenance note separating local from
external claims, offline tests with a failing-closed network guard.

### 4.2 Staged validation plan **[PROPOSAL]**

Do not change datasets. Each stage has an objective, a measure, and a gate.

| Stage | Objective | Measure / acceptance | Gate |
|---|---|---|---|
| **R1 — Stop-list + coverage decision** | Remove the three diagnosed false-refusal tokens (`main`, `essential`, `highlight`) and decide T3's coverage design | Focused suites (`test_books_grounding*`, `test_self_refusal_recovery`, `test_citation_integrity`) stay green; **abstention cases T5/C/B1 must still refuse**; polarity test (`not`-claims) still refuses | Full suite ≥ 661 passed, exit 0; no gold-data edits |
| **R2 — Model capability** | Decide the serving model (0.6B floor is the dominant residual cause) | Re-run the 7 original + 6 A-questions: measure flip count of the 5 remaining false refusals **without** any abstention regression | Human decision (§11 Q1) before any change |
| **R3 — Research path health** | SearXNG latency vs the 15 s client timeout; empty crawl bodies | Live probe recorded; one authorized fallback run must produce a *positively sourced* answer or a recorded honest failure | Approval for external calls (§11 Q3) |
| **R4 — Corpus growth** | Repair/ingest the two `failed` PDFs and add real multi-chunk documents | Manifest integrity (`test_ingest_integrity.py`), retrieval baseline re-recorded as a **new** baseline file | Explicit ingestion authorization (§11 Q4) |
| **R5 — Live answer scoring** | Opt-in `local-model` RAG-AE mode over the frozen gold sets | Report false-refusal rate and correct-abstention rate as numbers; never claim a score the harness did not compute | Runs offline against Ollama only, artifacts immutable (`--out` exit-3 protection) |
| **R6 — GUI end-to-end regression** | The 7 original questions + 6 A-questions + G1–G5 fallback set | Categorised as in the fix report: correct / incorrect citation / unsupported / false refusal / correct abstention / routing failure / not-run | Recorded in a dated validation report; no code edits during verification |

**Baseline metrics to compare against (recorded, byte-verified):**
unit 17/17/0 with all sub-scores 1.0 and abstention P/R 1.0/1.0;
integration 10/10/0 with 7 skipped, hit@1 0.8, hit@3 1.0, coverage 1.0,
precision@1 0.9, precision@3 0.3667, rr 0.9333, ndcg@5 0.95,
fact_coverage 1.0, `guard_unchanged=True`, `network_attempts: 0`;
GUI: 16/16 books routing, 4/4 abstentions held, 3/7 false refusals fixed.

---

## 5. Proposed execution-record schema and JSON export **[PROPOSAL]**

### 5.1 What exists (inspect first — no new storage needed for the base)

`storage/control_center.sqlite3` already holds, **as the run happens**:

* `executions(id, question, status, route, answer, qa_flags, error,
  created_at, started_at, completed_at, result, plan)`
* `events(execution_id, event_id, timestamp, type, message, step_id,
  agent_id, severity, payload)` — append-only, restart-surviving, cursor-paged.

Event types today: `execution.created/started/planned/completed/failed/cancelled`,
`step.*`, `agent.started/completed/failed`, `routing.decided`, `qa.completed`.
The GUI is already a strict projection of this log (fake progress is
forbidden by `app/execution/runner.py`).

### 5.2 Gaps that block the required exports (honest list)

| Required by spec | Available today? | Gap |
|---|---|---|
| Run ID, timestamps, status, request, final answer | **Yes** | — |
| Citations / source provenance | **No** | `state["sources"]` is never persisted; citations live only inside the answer string |
| Errors / partial-result status | **Yes** | `error` column + `qa_flags` |
| Routing & delegation decisions | **Yes** | `routing.decided` payload |
| Agent invocations + statuses + timings | **Yes** | `agent.*` events with timestamps |
| **Tool** names, safe input/output summaries, timings, errors | **No** | search/crawl/retrieval are inside agents and only `print()` |
| Retrieval summaries / source identifiers | **No** | `books_agent` logs `score`/`nodes` to stdout only |
| Recovery / fallback decisions *and why* | **No (as events)** | `[books_agent]` and `[graph] knowledge fallback …` are log lines; `books_refusal` state is discarded at run end |
| Evaluation outcomes attached to a run | **No** | eval artifacts are separate JSON files with no run linkage |

### 5.3 Design

1. **One versioned document, two projections.** `agentica.execution.v1`
   carries everything; `view: "result"` is a small projection for a single
   answer, `view: "full"` is the whole lifecycle. Consumers ignore unknown
   fields; producers fill `null` for missing values (never omit, never
   invent).
2. **Record, never reconstruct.** New event types are emitted at the source
   (`[PROPOSAL]` additions to the existing emitter):
   `retrieval.completed`, `tool.started/completed/failed`,
   `recovery.outcome`, `fallback.decided`, `citation.validated`,
   `provenance.persisted`. Payloads are scalars and identifiers only —
   the same no-passage-text rule already enforced for `books_refusal`.
3. **Storage: keep SQLite** (recommendation). Reasons: it is already
   local-first, WAL, thread-locked, private (git-ignored), survives restarts,
   is trivially backup-able, and the expected volume (hundreds of runs ×
   <100 events) is far below any threshold that would justify files or a
   server. Add columns/tables (`sources`, `events.payload` reuse) with the
   same back-compat migration pattern as `plan` (`_migrate_schema`). Exports
   are **generated on demand**, not stored, so no stale export can be
   associated with another run: the export always re-reads by `execution_id`
   and stamps `generated_at`.
4. **Schema versioning:** `schema_version: "agentica.execution.v1"`;
   additive changes bump the minor (`v1.1`) inside the same field; a
   breaking change starts `v2` and the exporter keeps emitting `v1` for old
   rows by upgrading on read. Unknown/legacy rows export with
   `partial: true` and an `incomplete_fields[]` list rather than failing.
5. **Redaction (hard rules):** never export `.env` values, API keys,
   `os.environ`, raw HTTP headers, session settings (the masked-settings
   rule in `app/api/server.py` is the precedent), model hidden reasoning,
   or full retrieved passage text (export chunk identifiers, file + page
   labels, scores, and a bounded excerpt only). Diagnostic payload strings
   stay length-capped exactly as `ExecutionError` already does.
6. **GUI:** a `Download JSON` action on Execution Detail (single run) and on
   the evaluation artifact viewer, implemented as a client-side
   `Blob`/`createObjectURL` download of the API response — no filesystem
   write on the server.

A machine-readable draft is published alongside this document as
[docs/execution_record.schema.json](execution_record.schema.json) (JSON
Schema, draft 2020-12) with a synthetic example in §5.4 below.

### 5.4 Synthetic example (`view: "result"`; Arabic preserved)

```json
{
  "schema_version": "agentica.execution.v1",
  "view": "result",
  "generated_at": "2026-10-27T09:14:03+00:00",
  "run": {
    "id": "40c2206212ba",
    "status": "completed",
    "partial": false,
    "created_at": "2026-10-27T09:12:41+00:00",
    "started_at": "2026-10-27T09:12:41+00:00",
    "completed_at": "2026-10-27T09:13:31+00:00",
    "request": {
      "question": "ما هي الوظائف الأربع لإطار إدارة مخاطر الذكاء الاصطناعي؟",
      "language_hint": "ar"
    },
    "routing": { "route": "books", "decision_event_id": 4, "mode": "keyword" },
    "source_selection": {
      "initial_source": "books",
      "fallback_used": false,
      "fallback_reason": null,
      "local_only": false
    }
  },
  "result": {
    "answer": "الوظائف الأربع هي GOVERN وMAP وMEASURE وMANAGE.",
    "citations": [
      { "document": "NIST.AI.100-1.pdf", "page_label": "3",
        "chunk_ids": ["chunk:8f21"], "support": "verified" }
    ],
    "provenance": ["books"],
    "qa_flags": [],
    "refusal": null,
    "errors": []
  },
  "evaluation": [
    { "suite": "rag_ae_unit", "artifact": "rag_ae_unit_v3.json",
      "executed": 17, "passed": 17, "failed": 0 }
  ]
}
```

`view: "full"` additionally embeds `execution.agents[]` (name, status,
started/completed, duration_ms), `execution.tools[]` (name, purpose,
duration_ms, outcome, error_category, `input_summary` redacted),
`execution.retrieval[]` (query_hash, top_k, per-chunk file/page/score),
`execution.decisions[]` (`kind`: `routing|recovery|fallback|gate`,
`code`, `reason`, `at`), `plan`, and the raw `events[]` array.

---

## 6. Safe Windows / file-operation architecture **[PROPOSAL]**

**Finding:** the backend runs on Windows but has **no computer-control
capability whatsoever** today — `app/tools/` contains only SearXNG and
Crawl4AI, and `docs/AGENT_DEVELOPMENT.md` §5 states an agent "never reads a
file it did not create". Nothing should be inferred from the OS.

**Recommended shape — mirror the ERP boundary that already works:**

```text
app/fs/adapter.py        WorkspaceAdapter ABC (READ-ONLY methods first)
                          list / stat / read_text / read_bytes(bounded) /
                          mkdir / write / copy / move  — no delete in v1
app/fs/windows_adapter.py canonicalizing implementation
app/agents/workspace_agent.py  the ONLY caller (same rule as erp_agent)
app/fs/log.py            audit line: timestamp, op, path-role, size, status,
                          duration, coarse error category — never file content
```

**Permission model (v1):**

1. **Single allowlisted root** — e.g. `storage/workspace/`. Every path is
   resolved with `Path.resolve()` and checked against the root with
   `os.path.commonpath`; reject `..`, absolute paths outside the root,
   symlink/junction reparse points (`os.path.islink` + Windows
   `FILE_ATTRIBUTE_REPARSE_POINT`), and NTFS alternate data streams (`:` in
   components).
2. **Hard deny-list regardless of root:** `.env`, `storage/*.sqlite3*`,
   `storage/.index_manifest.json`, `env/`, `.git/`, `logs/`, and anything
   matching secret-shaped names — this is the "agents cannot read secrets"
   control, enforced by path, not by prompt.
3. **Size and type caps** reuse the M2 precedent (`INGEST_MAX_FILE_MB`,
   `INGEST_MAX_PAGES`): enforced on `stat()` before any read; explicit
   allowlist of extensions (`.txt .csv .md .json .xlsx` later).
4. **Two-phase mutation:** every write/copy/move first returns a *plan*
   (source, target, bytes, conflict?); execution happens only after an
   explicit confirmation. No in-place overwrite by default; writes create
   `*.new` then swap. **No delete in v1** — it is the operation with no
   undo path in the current design.
5. **Read-only default.** A capability flag (`read_only: true`) is the
   default state; enabling writes is an operator setting, not an agent
   decision.
6. **Audit + failure recovery.** Every attempt (allowed and denied) logs a
   line; a denied attempt is a normal, reported outcome, not a crash — the
   agent converts it into the same structured error shape the ERP adapter
   uses, never leaking absolute paths outside the workspace to the answer.
7. **Explicit local-only behaviour:** file tools never call the network, and
   file contents reach an external search query only under an explicit
   authorization rule (see §11 Q9).

**Smallest useful first implementation:** `list` + `stat` + `read_text`
(bounded) behind a `workspace` capability on the existing dispatcher
whitelist, exercised with synthetic files, read-only end-to-end. Estimated
**M**, and it is a prerequisite for spreadsheets.

**Future capabilities** (only after the above is proven): bounded write/copy/
move, spreadsheet formats (§8), then application APIs (COM/Office scripts)
— and only then, if still justified, GUI-level automation. Desktop
automation should never be the first choice where a structured API exists.

---

## 7. Agent Builder vs Skill Builder — design recommendation

**Recommendation: a hybrid, but built as "builder agent + deterministic
creation tools", not as a new self-modifying agent — and skills first.**

Why, from the architecture (not naming):

* Agents are **literal graph wiring**. `docs/AGENT_DEVELOPMENT.md` §3/§10
  requires five explicit steps and explicitly forbids a BaseAgent, plugin
  marketplace, DI container, or dynamic registry; `app/graph.py` is the
  documentation. A "new agent" is therefore a **code change** that must go
  through tests and review — it cannot honestly be a runtime registration.
* **Skills are different.** A skill can be pure *data* consumed by agents
  that already exist: name, purpose, instructions, tool allowlist,
  input/output contract, permissions, example tasks, and fixture-based
  acceptance tests. Nothing new has to be wired into the graph; the
  orchestrator stays untouched.

**Proposed architecture:**

```text
skills/<skill-id>/<version>/
   skill.json        declarative spec (validated by a deterministic validator)
   SKILL.md          instructions injected into the consuming agent's prompt
   fixtures/         synthetic inputs + expected outputs (offline)
   tests/test_skill.py   must pass before the version can be enabled
   MANIFEST.json     version, parent version, approvals, hashes

agent_specs/<agent-id>/  [later] spec + generated app/agents/<id>.py diff
```

**Creation flow (both paths share steps 1–6):**

1. **Inspect** existing skills/tools/agents (deduplication check) — a pure
   read tool over `skills/`, `app/tools/`, `app/agents/`.
2. **Propose** a capability spec from the natural-language request
   (the Builder agent's only creative step).
3. **Detect gaps**: an requested tool that does not exist is reported as
   *"needs new integration"* — never silently stubbed.
4. **Validate + test offline** with a deterministic validator and the
   skill's own fixtures (no network, no model, synthetic data).
5. **Present** spec, permissions, test evidence, limitations — then
   **require explicit approval** before anything is enabled.
6. **Activate** by writing a new version directory; **rollback** = re-enable
   the previous version (versions are immutable once approved).
7. **Discoverability:** enabled skills are loaded at startup into a
   read-only map that the supervisor/QA layer can consult for routing hints
   and that agents can merge into their prompt context.
8. **Recorded:** every proposal/approval/activation is an execution event so
   §5's export covers it.

**Path A — Skill (recommended first):** small reusable tasks. No graph
change, testable offline, rollback trivial. **Effort M.**

**Path B — Agent (second):** for tasks needing independent routing or a
broader workflow. The Builder may generate the module and the exact wiring
diff, but activation requires a human to accept the diff and the five-step
wiring (plus the deterministic test). This keeps "improve through tested,
reviewable changes" true and keeps the "no dynamic registry" rule intact.
**Effort L**, and gated on §11 Q7.

**Explicitly out of scope, permanently:** unrestricted self-modifying code,
auto-execution of generated scripts, generated code running outside the test
harness without review, and any builder-granted filesystem escape (§6 rules
apply to the Builder too).

*Key architectural decision to approve first:* whether skills may be loaded
dynamically at startup (a read-only, validated registry — a small, honest
departure from the current "no registry" doctrine, scoped to *data*, not
graph nodes) or must still be activated by a code change. See §11 Q7.

---

## 8. Spreadsheet capability plan

**Environment comparison (verified):**

| | Local CSV/XLSX | Google Sheets |
|---|---|---|
| Libraries present | `python-docx` yes; **`openpyxl` NOT installed**; `pandas` 2.3.3 present *transitively* (not a declared dependency — do not rely on it) | none |
| Auth | none needed | OAuth/service account + new dependency + consent; **no integration exists** |
| Data leaving the machine | no | yes — needs explicit authorization |
| Fitting architecture | workspace adapter (§6) | new adapter beside `app/erp/adapter.py` pattern |

**Recommendation:** local **CSV first** (stdlib `csv`, zero new
dependencies), XLSX second (requires adding `openpyxl` — approval needed),
Google Sheets last and only with explicit authorization and a connected
credential (§11 Q5, Q6).

**First practical capability (a Skill, per §7):** *spreadsheet-inspect-and-validate*

1. **Inspect** — sheets/encoding, headers, row/column counts, inferred
   types, sample rows (bounded), duplicate keys.
2. **Validate** — declarative rules (required columns, type/range checks,
   uniqueness, referential rules supplied in the request).
3. **Transform** — requested operation only (dedupe, update selected rows,
   derive a column), implemented deterministically, never by the LLM
   editing cell values directly.
4. **Preview** — a diff (rows added/removed/changed with before → after),
   rendered before any write.
5. **Write** — only to a *new* file inside the workspace root, only after
   confirmation, then **report exactly what changed** (counts per
   operation + the diff artifact).

**Safety:** synthetic fixtures only in tests; size caps; no in-place
overwrite in v1; audit lines for every read/write; the LLM never sees or
emits bulk cell data — it only chooses between validated operations.
**Effort M** (after §6), testable entirely offline.

---

## 9. Prioritized implementation roadmap

Sizes are relative (**S** ≤ 1–2 days, **M** ≤ ~1 week, **L** multi-week)
and deliberately conservative; no performance numbers are promised.

| # | Workstream | Size | Depends on | Priority |
|---|---|---|---|---|
| W1 | Stabilize RAG (stop-list, coverage decision, citation audit, model decision) | **M** | — | **P0** |
| W2 | Structured execution events + persisted provenance | **M** | — | **P0** |
| W3 | JSON export (result + full execution) + GUI download | **S–M** | W2 | P1 |
| W4 | Restricted Windows workspace tools (read-only first) | **M** | W2 (audit events) | P1 |
| W5 | Initial spreadsheet skill (CSV → XLSX) | **M** | W4, §7 Skill path | P2 |
| W6 | Agent/Skill Builder (skill path first) | **L** | W2, W4, W5 | P2 |
| W7 | Skill persistence, versioning, lifecycle + approval workflow | **M** | W6 | P2 |
| W8 | Desktop/application automation expansion | **L** | W4 proven, real justification | P3 (deferred) |

**W1 — Stabilize RAG**
*Status:* partial (§4). *Changes:* the three diagnosed stop-token additions;
a design decision on T3-style coverage; optional printed+viewer page labels
(metadata-only ingest change, requires re-ingest authorization); serving-model
decision. *Files:* `app/agents/books_agent.py`, `app/config.py`,
`ingest_books.py` (metadata only), `scripts/test_books_grounding*.py`,
`scripts/test_self_refusal_recovery.py`. *Acceptance:* full suite ≥ 661
green with **zero abstention regressions** (T5/C/B1 still refuse); the 7+6
GUI questions re-categorized with named remaining failures; gold datasets
byte-identical. *Risks/rollback:* stop-list edits can silently weaken the
gate — mitigated by the polarity test and an A/B behavior script
(`.diag/fix_cycle_baseline/ab_polarity.py` pattern); every change is
independently revertible. *Uncertainty:* high — outcome depends on the model
decision (Q1).

**W2 — Structured execution events + provenance**
*Status:* not implemented (§5.2). *Changes:* emit `retrieval.completed`,
`tool.*`, `recovery.outcome`, `fallback.decided`, `citation.validated`
events at their real source; persist `sources` (new column, migration like
`plan`); persist `books_refusal` scalars as a run attribute. *Files:*
`app/execution/events.py`, `runner.py`, `store.py`, `app/agents/books_agent.py`,
`app/agents/research_agent.py`, `app/tools/*`, `app/graph.py`,
`scripts/test_gui1_events.py`. *Acceptance:* a run's event stream contains
every decision the logs currently show as `print()`; export can be built
with no log scraping; legacy rows still load (`plan`-style migration test);
suite green. *Risks:* payload bloat/leak — mitigated by the scalars-only
rule already enforced for `books_refusal`. *Uncertainty:* low.

**W3 — JSON export**
*Status:* not implemented. *Changes:* `GET /api/v1/executions/{id}/export?view=result|full`
returning the versioned document; GUI download button; redaction pass;
schema file. *Files:* `app/api/server.py`, new small `app/execution/export.py`,
`frontend/src/pages/ExecutionDetail.tsx`, `frontend/src/api/client.ts`, tests.
*Acceptance:* exported JSON validates against the published schema; Arabic
round-trips; a failed run exports `status=failed` with `errors[]`; no key/
env/secret string appears (automated grep assertion in tests); two runs never
share a record (id stamping test). *Risks:* leaking internals — mitigated by
redaction tests. *Uncertainty:* low.

**W4 — Workspace tools**
*Status:* not implemented (§6). *Changes:* `WorkspaceAdapter` + read-only
first capability + audit log + path-confusion tests. *Files:* new `app/fs/`,
`app/agents/workspace_agent.py`, `app/graph.py`, `app/supervisor.py`,
`scripts/test_workspace_safety.py`. *Acceptance:* traversal/symlink/junction/
deny-list attempts all denied **and** logged; legitimate read works; zero
network in the tool; suite green. *Risks:* path-escape classes of bugs —
mitigated by an adversarial test battery (Windows-specific: reparse points,
8.3 short names, case-insensitivity, alternate data streams). *Uncertainty:* medium.

**W5 — Spreadsheet skill** — see §8. *Acceptance:* synthetic fixture
workbook → inspect/validate/transform/preview/write with a diff report; no
in-place overwrite; rollback = delete the output file. *Uncertainty:* medium
(dependency decision Q5).

**W6/W7 — Builder + lifecycle** — see §7. *Acceptance:* end-to-end §10
scenario offline; a rejected proposal changes nothing on disk; rollback
restores the prior version byte-for-byte; no unapproved activation path
exists (asserted by test). *Uncertainty:* high (Q7 architectural decision).

**W8 — Desktop automation:** deferred; re-evaluate only after W4/W5 prove
the permission model. No work now.

**Independence:** W1 and W2 can proceed **in parallel** (disjoint files).
W3 strictly follows W2. W4 follows W2 but not W1. W5 follows W4. W6/W7
follow W2+W5. Nothing in W3–W8 may be used to mask a W1 regression: keep
their test batches separate, as the fix cycle did.

---

## 10. End-to-end acceptance scenario (future; **not executed now**)

> A user asks Agentica to create a spreadsheet-management skill, supplies a
> sample spreadsheet, asks it to validate and transform the data, and
> requests a JSON export of the execution.

| # | Step | Component | Observable evidence |
|---|---|---|---|
| 1 | Natural-language intent recognized | supervisor + Builder entry | `routing.decided` event shows the chosen path; no keyword required |
| 2 | Builder generates a capability spec | W6 Builder agent | `skill.json` proposal artifact + `proposal.created` event |
| 3 | Tools & permissions selected | spec validator | permissions = workspace root only, read/write scoped to `*.csv`, deny-list intact |
| 4 | Isolated tests on synthetic data | skill fixtures + pytest | test results attached to the proposal; zero network |
| 5 | User approval, then registration | approval gate | `approval.granted` event with actor + timestamp; version directory written **only after** approval |
| 6 | Execution through the orchestrator | graph + dispatcher | plan/task events; skill used by an existing agent; no dynamic node invention |
| 7 | Spreadsheet output validated | W5 skill | preview diff matches expectations; new file in workspace root; audit log lines |
| 8 | JSON export (result + full) | W3 | both files validate against `agentica.execution.v1`; Arabic/Unicode intact |
| 9 | Auditable trail | W2 event log | tool calls, timings, results, errors, approvals all present as events — none reconstructed afterwards |
| 10 | No unauthorized access | W4 guard | attempted read outside workspace denied **and logged**; no secret-shaped path readable; `network_attempts: 0` for local-only steps |

**Negative controls required in the same scenario:** a rejected approval
leaves the system unchanged; a skill whose tests fail cannot be activated;
a file outside the workspace cannot be read even by an approved skill; and
if research is not permitted for the request, no external call occurs.

---

## 11. Open questions and decisions requiring your approval

1. **Serving model (blocks W1/R2).** `qwen3:0.6b` is the dominant residual
   cause of false refusals (A4/A6/B2). Options: keep 0.6B and accept the
   floor; move to a larger local model (RAM/VRAM constraint); or enable the
   existing Gemini path (paid, needs `.env` key). *Recommendation:* decide
   before authorizing any further stop-list/coverage edits, otherwise model
   noise is confounded with gate behavior.
2. **Authorize the next RAG micro-fix** (stop tokens `main`/`essential`/
   `highlight`, T3 coverage design) — or accept the 5 remaining refusals as
   documented.
3. **SearXNG environment:** container unhealthy, `/search` ~23 s vs a
   hardcoded 15 s client timeout. Approve raising/making configurable the
   timeout and repairing the container? Also: crawl returned empty for all
   3 URLs — authorize investigation of `crawl_extractor`?
4. **Ingestion:** repair/re-ingest the two `failed` PDFs
   (`NIST.AI.600-1.pdf.pdf`, `nasa_systems_engineering_handbook_0.pdf`)?
   This is the only way retrieval metrics become discriminating, but it
   touches the frozen-ish corpus and manifest (would require a *new* baseline).
5. **New dependencies:** `openpyxl` for XLSX (declared, pinned) — yes/no?
   (Do not rely on the transitive `pandas`.)
6. **Google Sheets:** authorize external data flow + credential setup, defer,
   or drop? (Recommend: defer until local spreadsheets are proven.)
7. **Dynamic skill registration vs the documented "no registry" rule**
   (§7). Approve a validated, read-only, data-only skill loader, or require
   skills to be code-reviewed files like agents?
8. **Event instrumentation (W2)** is a source-code change to agents/tools.
   Approve adding the new event types (recommended; it is the foundation of
   the export deliverable)?
9. **External-search privacy policy:** what may leave the machine? Current
   behavior sends the *original question only* (verified). Confirm that as
   the standing rule, and confirm whether file/spreadsheet contents may ever
   be included (recommend: never without per-request consent).
10. **GUI export surface:** Execution Detail only, or also evaluation
    artifacts and bulk "download all"?
11. **Real ERP adapter** — vendor still unspecified; is it in scope at all?
12. **Approval Center / Knowledge / Chat / Reports** (GUI placeholders):
    which, if any, should be scheduled — Approval is a prerequisite for W7.

---

### Inspection notes (how this was verified)

Git state, manifest statuses, installed dependency versions, and the absence
of file/export/skill code were read directly from the working tree; the
canonical test suite was re-run offline in this task (661 passed / 2 skipped
/ 138 subtests, exit 0); RAG/GUI claims are cross-checked against
`docs/GUI_RAG_TEST_REPORT_2026-10-24_1114.md`,
`docs/GUI_RAG_ROOT_CAUSE_2026-10-24_2349.md`,
`docs/GUI_RAG_FIX_VALIDATION_2026-10-25_1415.md`,
`docs/GUI_RAG_FIX_VALIDATION_2026-10-25_1702.md`, `docs/RAG_AE_EVALUATION.md`
and `docs/RAG_EVAL_GAPS.md`, and verified against the source they cite.
No source file, configuration, dataset, PDF, manifest, or branch was
modified; no network or paid-model call was made; no commit was created.
