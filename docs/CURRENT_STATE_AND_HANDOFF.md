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

## Recommended next task

GUI-4 (execution history page hardening) or MASTER_PLAN Phase 7 (conditional
planner) — GUI-4 is the smaller increments-first choice; Phase 7 unlocks
GUI-12 later. Either agent should read `docs/MASTER_PLAN.md` /
`docs/CONTROL_CENTER_PLAN.md` phase templates before coding.
