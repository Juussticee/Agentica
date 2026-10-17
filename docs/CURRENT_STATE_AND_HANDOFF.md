# Recovery Handoff — Current State and Next Steps

Factual snapshot after the reconstruction session. Read this before any
further work; MASTER_PLAN.md and CONTROL_CENTER_PLAN.md remain the roadmaps.

## What exists now (all verified by tests this session)

| Component | Files | Tests |
|---|---|---|
| Phase 5 production QA | `app/qa_node.py`, `app/graph.py` (qa terminal node) | `scripts/test_phase5_qa_node.py` — 12 |
| Phase 6 LLM routing | `app/supervisor.py`, `app/config.py` (ROUTING_MODE) | `scripts/test_phase6_routing_llm.py` — 11 |
| Execution foundation | `app/execution/{events,store,runner,errors}.py`, `build_graph(node_wrapper)` hook | `scripts/test_gui1_events.py` — 18 |
| API v1 | `app/api/server.py` | `scripts/test_api_preview.py` — 14 |
| AI settings/readiness | `app/ai_settings.py`, `app/ai_diagnostics.py`, API endpoints | `scripts/test_gui16_settings.py` — 21 (+1 live) |
| Frontend | `frontend/` (Vite+React+TS) | `npm run build` (tsc) green |
| RAG-1 retrieval baseline (closeout 2026-10-17) | `benchmarks/rag_gold_v1.json`, `benchmarks/rag_fixtures/`, `benchmarks/rag_baseline_v{1,2,3}.json`, `scripts/eval_retrieval.py`, `scripts/make_rag_fixtures.py`, `scripts/test_rag_eval_harness.py`, `docs/RAG_BASELINE_AUDIT.md`, `docs/RAG_1_COMPLETION_REPORT.md` | `scripts/test_rag_eval_harness.py` — **38 passed + 11 subtests** (offline); gate `benchmark_verification.py --check` = **MATCH** |

Total deterministic suites this session: **177 passed + 1 skipped (live-optional) + 24 subtests.**

## How to run (Windows)

```bat
:: backend tests (venv at env\)
env\Scripts\python.exe -m pytest scripts\ -q

:: API on :8000 (localhost-bound)
env\Scripts\python.exe -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000

:: frontend on :3000 (proxies /api)
cd frontend && npm install && npm run dev

:: AI settings live check of the real Ollama endpoint
set OLLAMA_LIVE=1 && env\Scripts\python.exe -m pytest scripts\test_gui16_settings.py::TestLiveOllama
```

## AI providers

- Default: `LLM_PROVIDER=gemini` (key from `.env`, never committed).
- Ollama: `LLM_PROVIDER=ollama` + `OLLAMA_BASE_URL=http://localhost:11434`
  + `MAIN_MODEL=qwen3:0.6b` (verified working: endpoint → tags → generation).
- Settings precedence: session override > `storage/ai_settings.json`
  (only written by explicit "Save for this server") > env/config.
- The diagnostics lesson recorded in `app/ai_diagnostics.py`: qwen3-family
  thinking models can return an EMPTY response when `num_predict` is small
  because thinking consumed the budget; the test generation call disables
  thinking (`"think": false`).

## Environment notes

- `env/` is the working Python 3.11.9 venv (the older `venv/` is broken and
  git-ignored).
- Verified live this session: fake agent runs through the whole graph,
  ERP mock route (INV-1001), research route (SearXNG + crawl + Ollama
  synthesis), seven-phase Ollama readiness, browser UI flows.

## Blockers / limitations

- Gemini live path was NOT exercised (free-tier quota protection); keyword
  routing default never needs it, and LLM routing is opt-in.
- GUI-4..GUI-19 not started (see CONTROL_CENTER_PLAN.md); Settings page
  honestly lists what is missing.
- MASTER_PLAN Phases 7+ (planner, dispatcher, hybrid RAG, reviser) are
  future work; do not start them without reading the plan templates.

## RAG-1 closeout (2026-10-17)

**Verified completion state.** HEAD `7a44a9a7f747…`, nothing staged or
committed. Retrieval baseline frozen at
`benchmarks/rag_baseline_v3.json` (SHA-256
`aa89c37bc23b0dc748255df9410a28ba43a6f8d8043070ad54cafd6e175de1cd`,
`artifact_schema.version=3` / `ordered-top5-per-case`). Harness suite **38
passed + 11 subtests** (offline), answer-level grounding/citation suites
**46 passed + 8 subtests**, frozen-baseline gate **MATCH** — all with the
production store byte-stable (`cb0e2564…`, size and mtime unchanged);
`scripts/check_chroma.py` was never run, no model/API/network call was
made, and the gold/v1/v2/v3 SHA-256 pins are unchanged. The evaluator now
refuses to overwrite an existing `--out` report (**exit 3**) unless
`--force` is passed, so no documented command can silently clobber a
pinned artifact. Full record: `docs/RAG_1_COMPLETION_REPORT.md`.

**Limitations (must travel with the numbers).** The abstention proxy is a
score-floor check only (0.35 unreachable; all top-1 scores ≥ 0.65) —
answer-level refusal is **unmeasured**. Real-corpus metrics cannot
discriminate (2 single-chunk documents). `hit@1` vs MRR divergence is
definitional (distinct-doc `min_docs` vs first covering node).
Coverage-family metrics under the page-metadata-aware rule are not
comparable between v1 and v2/v3 (`docs/RAG_BASELINE_AUDIT.md` §7). Latency
is machine-local. `git_dirty: true` is recorded inside v3 (evaluated code
was uncommitted at run time). The full 383-test regression was **not**
re-run this session (last recorded green 2026-10-16) — reported, not
claimed.

**Outstanding work.** (1) Commit authorization for the RAG-1 change set.
(2) **RAG-AE** — answer-quality evaluation (spec in `docs/MASTER_PLAN.md`
§5.1; registered outside the numbered RAG-1…RAG-6 sequence to end the
three-way "RAG-2" collision — MASTER_PLAN RAG-2 = hybrid retrieval,
implementation-plan RAG-2 = structure-aware chunking; mapping documented
in both). (3) `BOOKS_MIN_SCORE` calibration report (plan RAG-3) — user
approval required, never silent. (4) Optional gold extension for
partial-answer cases in a NEW gold file (gold v1 stays pinned immutable).
(5) Re-run the full regression floor before commit if a wider green bar is
wanted.

## Recommended next task

GUI-4 (execution history page hardening) or MASTER_PLAN Phase 7 (conditional
planner) — GUI-4 is the smaller increments-first choice; Phase 7 unlocks
GUI-12 later. Either agent should read `docs/MASTER_PLAN.md` /
`docs/CONTROL_CENTER_PLAN.md` phase templates before coding.
