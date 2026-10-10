# Testing Guide

All commands run from the project root with the venv active
(`venv\Scripts\activate`). Suites marked **zero-quota** use fake LLMs and
never call Gemini; only the explicitly marked live scripts consume quota.

## A. Environment checks

```bat
python --version                 :: 3.11.x expected
python scripts\check_dependencies.py   :: all imports OK -> ALL_DEPENDENCIES_OK
ollama list                      :: nomic-embed-text must be present
docker ps                        :: searxng container should be listed (unhealthy status is common; the JSON API is what matters)
curl "http://localhost:8888/search?q=test&format=json"   :: JSON with a "results" key
python scripts\check_chroma.py   :: collection "books", count >= 1, no null bytes
```

## B. Book ingestion

1. Put `.pdf` / `.docx` files into `knowledge\books\`.
2. Run:

```bat
python ingest_books.py
```

Incremental: `storage\.index_manifest.json` tracks filename → mtime;
re-running skips already-indexed files. A binary-garbage guard fails loudly
if a file parses as bytes instead of text (missing file readers).
3. Verify: `python scripts\check_chroma.py` (count grows; manifest lists your
   files; printable ratio ~100%).

## C. Test suites

| Command | What it tests | Live services | Gemini quota | Expected output |
|---|---|---|---|---|
| `python scripts\test_phase1_routing.py` | supervisor keyword decisions + graph wiring with faked nodes | no | 0 | `STEP6_WIRING_OK` |
| `python scripts\test_phase1_proxy.py` | full books+research wiring: real Chroma retrieval, real SearXNG, real crawl; fake synthesis LLM | SearXNG + crawled sites | 0 | `PHASE1_PROXY_OK` (takes ~1 min; external sites) |
| `python scripts\test_phase2.py` | feasibility `.docx` contract (Market/Financial/Risk) + chain order research→books→feasibility, fake LLM | no | 0 | `OK` (2 tests) |
| `python scripts\test_rewrite_step.py` | rewrite prompt behavior, fallback on failure, noisy-host filter, synthesis still uses original question | no | 0 | `REWRITE_STEP_OK` (4 behaviors) |
| `python scripts\test_feasibility_degradation.py` | web-only degradation: refusal / missing key / placeholder → flagged study + Limitations; real content → normal path | no | 0 | `OK` (4 tests) |
| `python scripts\test_phase3_erp.py` | Phase 3 ERP boundary: routing (keywords + graph), mock adapter semantics, successful read, record-not-found, adapter failure, permission denied, provenance, write protection, adapter factory, action logging | no | 0 | `OK` (35 tests) |
| `env\Scripts\python.exe -m pytest scripts/test_phase4_workflow_qa.py scripts/test_phase5_qa_node.py scripts/test_phase6_routing_llm.py -q` | Phase 4 workflow/QA suite (30), Phase 5 production QA node (12), Phase 6 LLM routing fallback (11) — all deterministic | no | 0 | pytest: 53 passed, 24 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_gui1_events.py scripts/test_api_preview.py -q` | GUI-1 execution events + runner lifecycle (18); GUI-2 API endpoints, validation, event cursor, honest 404s (14) | no | 0 | pytest: 32 passed |
| `env\Scripts\python.exe -m pytest scripts/test_gui16_settings.py -q` | AI settings precedence/masking/persistence (7), readiness diagnostics vs a fake Ollama server (8), API endpoints (5) + one optional LIVE provider test (runs only with `OLLAMA_LIVE=1`) | no (fake server) | 0 | pytest: 21 passed, 1 skipped |
| `env\Scripts\python.exe -m pytest scripts/test_agent_architecture.py -q` | Phase 4 agent-architecture contract (no BaseAgent, single factory, isolation) | no | 0 | 30 passed |
| `python scripts\check_dependencies.py` | every required package imports | no | 0 | `ALL_DEPENDENCIES_OK` |
| `python scripts\check_chroma.py` | persisted collection non-empty and text is prose | no | 0 | count + printable ratio, exit 0 |
| `python scripts\verify_graph.py` | **LIVE** smoke test: real graph, 3 scenarios | SearXNG + web | **~3 calls** | three scenario blocks; refusal on out-of-corpus |
| `python scripts\test_phase1_e2e.py` | **LIVE** spec steps 3 & 5 end to end | SearXNG + web | **~2 calls** | `STEPS_3_5_REAL_OK` |
| `python scripts\check_books_agent.py` | one real books answer with provenance | Ollama embeds | **1 call** | answer + `SOURCES: ['books']` |
| `python scripts\probe_books_prompt.py` | captures the exact prompt sent to the LLM by the query engine | Ollama embeds | **1 call** | answer + captured prompts |
| `python scripts\demo_rewrite_quality.py` | before/after search-quality demo of the rewrite step | SearXNG + web | **1 fast call** | raw vs rewritten result lists |

### Regression order after any code change

```bat
python -m py_compile main.py ingest_books.py app\*.py app\agents\*.py app\tools\*.py app\erp\*.py scripts\*.py
python scripts\test_phase1_routing.py
python scripts\test_phase2.py
python scripts\test_feasibility_degradation.py
python scripts\test_rewrite_step.py
python scripts\test_phase3_erp.py
env\Scripts\python.exe -m pytest scripts/test_phase4_workflow_qa.py scripts/test_phase5_qa_node.py scripts/test_phase6_routing_llm.py scripts/test_agent_architecture.py scripts/test_gui1_events.py scripts/test_api_preview.py scripts/test_gui16_settings.py -q
```

Live suites are optional and consume free-tier quota (20 requests/day) — run
them deliberately.

## D. Manual CLI testing

```bat
python main.py --once "What does the orbital mechanics book say about delta-v?"
python main.py --once "What is LangGraph used for?"
python main.py --once "Do a feasibility study for launching an urban beekeeping venture."
```

Expected observations:

1. **Books** — an answer citing Hohmann-transfer delta-v figures, ending with
   `References: orbital_kinetics_principles.docx...`; footer
   `[route: books] [sources: ['books']]`.
2. **Research** — a synthesized paragraph plus a `Sources:` URL list; footer
   shows `[route: research]` with URLs in sources.
3. **Feasibility** — `[route: feasibility]`; console line
   `[feasibility_agent] ... study written ... -> feasibility_*.docx`; the
   answer contains the report path and Market/Financial/Risk content. With
   the sample corpus (beekeeping book present), the normal path runs; if the
   library had nothing relevant, the answer starts with
   `⚠ Web-only study: ...` and the `.docx` contains a Limitations section.
4. **ERP** — `python main.py --once "Show me open invoices"` routes to
   `[route: erp]` and returns deterministic mock records (INV-1001, INV-1002)
   with an explicit `(Data source: mock ERP adapter ...)` note; a question
   matching nothing (e.g. "Show me unicorns in the ERP") answers
   "The ERP returned no records matching this request." Zero Gemini quota:
   the ERP agent makes no LLM call.

Interactive mode: `python main.py` (empty line or Ctrl+C quits).

## E. Failure testing

| Scenario | How to simulate | Expected behavior |
|---|---|---|
| Missing books index | rename `storage\` away, run a books question | books_agent creates an empty collection and answers with the exact not-found phrase (code path verified by reading; not yet covered by an automated test) |
| Unrelated books question | "What is quantum tunneling?" | exact not-found refusal — never invented content (`test_phase1_e2e.py` covers this live) |
| SearXNG down | `docker stop searxng`, run a research question | search returns `[]` (fail-soft client), answer: "Web search returned no results, so I could not research this question." |
| Crawl failure | offline network or a blocking site during research | extraction returns `""`; synthesis falls back to search snippets; URLs still listed |
| Query rewrite failure | quota exhausted / key invalid during rewrite | logged, falls back to the raw question; pipeline continues (`test_rewrite_step.py` behaviors 2–3) |
| Books unavailable during feasibility | books refusal/missing/placeholder (see `test_feasibility_degradation.py`) | web-only study: dedicated prompt, `⚠` answer flag, Limitations section in the `.docx` |
| Gemini quota exhausted mid-question | run after 429s start | `main.py` catches `GoogleRateLimitError` and prints a quota message instead of a traceback |
| ERP adapter unavailable / timeout / permission denied | stub adapter raising the matching `ERPError` subclass (see `TestAgentErrorPaths` in `scripts/test_phase3_erp.py`) | fixed safe user message per category ("currently unavailable", "timed out", "insufficient permissions"); exception details never reach the answer; log line carries `error_category=<category>` |
| ERP unexpected failure | stub adapter raising a non-ERP `Exception` | "An unexpected error occurred while accessing the ERP."; logged as `error_category=unexpected_failure` |
