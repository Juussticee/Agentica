# Architecture

Status: reflects the code after Phase 5 (production QA gate) and Phase 6
(opt-in LLM routing). Phase 3 REAL ERP integration is NOT implemented — the
ERP vendor is intentionally unspecified.

Recovery additions (Control Center preview): `app/execution/` (events,
store, runner, errors), `app/api/server.py` (FastAPI v1),
`app/ai_settings.py` + `app/ai_diagnostics.py` (GUI-16), and a Vite+React
frontend under `frontend/`. The graph itself is unchanged except that every
route now terminates through the deterministic `qa` node.

## Topology

```text
                        ┌───────────────┐
                        │    main.py    │  CLI (--demo / --once / interactive)
                        └───────┬───────┘
                                │ build_graph().invoke({"question": ...})
                                ▼
                        ┌───────────────┐
                        │  LangGraph    │  compiled StateGraph (app/graph.py)
                        │    START      │
                        └───────┬───────┘
                                ▼
                        ┌───────────────┐
                        │  Supervisor   │  keyword routing (app/supervisor.py)
                        └───────┬───────┘
                                │ sets state["route"]
        ┌───────────────────────┼──────────────────────────┐
        │                       │                          │
        ▼                       ▼                          ▼
   route="books"          route="research"        route="feasibility" / route="erp"
        │                       │                  (sub_routes chained)   │
        ▼                       ▼                          ▼               ▼
  ┌────────────┐         ┌──────────────┐         ┌──────────────┐  ┌───────────┐
  │ books_agent│         │research_agent│────────▶│ books_agent  │  │ erp_agent │
  └─────┬──────┘         └──────┬───────┘         └──────┬───────┘  └─────┬─────┘
        │                       │                        ▼                │ ONLY caller
        ▼                       │                 ┌───────────────────┐    ▼
  ChromaDB ◀─ ingest_books.py   │                 │ feasibility_agent │  ┌─────────────────────────┐
  (LlamaIndex,                  │                 └───────┬───────────┘  │ ERP Adapter Interface   │
   Ollama embeds)               ▼                         ▼              │ (app/erp/adapter.py,    │
                          SearXNG + Crawl4AI        reports/*.docx       │  READ-ONLY ABC)         │
        │                       │                                        └───────────┬─────────────┘
        └───────────┬───────────┘                                                    ▼
                    ▼                                                    ┌────────────────────────┐
                   END                                                   │ MockERPAdapter         │
                                                                         │ (deterministic fake    │
                                                                         │  data; will be         │
                                                                         │  REPLACED by the real  │
                                                                         │  ERP adapter later)    │
                                                                         └────────────────────────┘
```

Condensed route traces, as the code actually executes them:

```text
Normal Phase 1 route:
START → supervisor → books_agent (or research_agent) → END

Phase 2 feasibility route:
START → supervisor → research_agent → books_agent → feasibility_agent → END

Phase 3 ERP route:
START → supervisor → erp_agent → ERP Adapter Interface → MockERPAdapter → END
(the mock adapter is eventually replaced by the real ERP adapter — the agent
and the interface stay unchanged)
```

The supervisor never calls an agent node; it only sets `route`. LangGraph's
conditional edges perform all hops. The feasibility chain is wired with
`_after_research` / `_after_books` conditional edges in `app/graph.py`, which
hand off to the next node only while `state["route"] == "feasibility"`;
single-agent routes fall through to `END` after their node.

## Component reference

| Component | Phase | Responsibility | Inputs | Outputs | Key failure behavior |
|---|---|---|---|---|---|
| `main.py` | 1 | CLI: builds the graph once, interactive loop, `--demo`, `--once` | user question | printed answer + `[route] [sources]` | catches `GoogleRateLimitError` with a quota hint; UTF-8 reconfigure for cp1256 consoles |
| `app/graph.py` | 1+2 | builds and compiles the single `StateGraph`; supervisor → node edges; feasibility chain conditionals | — | compiled graph | — |
| `app/state.py` | 1+2 | the ONLY definition of `AgentState` + `with_agent_output()` merge helper | — | — | — |
| `app/supervisor.py` | 1 | deterministic keyword routing (`_FEASIBILITY_RE` first, then `_BOOKS_RE`) | `question` | `{"route": ...}` (+ `sub_routes` for feasibility) | unknown question → `research` fallback |
| `app/llm_client.py` | 1+2 | `get_llm()` shared factory; `get_llm(fast=True)` for short calls; `llm_text()` normalizer | — | LangChain chat model / plain text | raises with a clear message when the provider is gemini but no key is configured |
| `app/agents/books_agent.py` | 1 | RAG answer from the Chroma index via a LlamaIndex query engine; provenance; refusal | `question` | answer + `sources=["books"]` | not-found refusal (prompt-level exact phrase + `BOOKS_MIN_SCORE` hard floor) instead of guessing |
| `app/agents/research_agent.py` | 1+2 | query rewrite → SearXNG → noise filter → Crawl4AI → synthesis with URLs | `question` | answer + `sources=["research", ...urls]` | every tool boundary is fail-soft: rewrite falls back to the raw question, empty search returns a "no results" answer, failed crawls fall back to snippets |
| `app/agents/feasibility_agent.py` | 2 | synthesizes Market/Financial/Risk study from research + books answers; writes `.docx` | `agent_answers` from state | answer containing the report path; file in `reports/` | web-only degradation when books output is missing/placeholder/refusal; `Limitations` section + `⚠` answer flag |
| `app/agents/erp_agent.py` | 3 | the ONLY caller of the ERP boundary: formats adapter results; no LLM call (zero quota) | `question`, prior `sources` | answer + `sources=["erp"]` + `agent_answers["erp"]` via `with_agent_output` | every adapter error mapped to a fixed safe message; never raises; no internal/credential detail reaches the answer |
| `app/erp/adapter.py` | 3 | vendor-neutral READ-ONLY `ERPAdapter` ABC (`name`/`search`/`get_record`) + frozen `ERPRecord` + `ERPError` taxonomy | — | — | no write methods exist by construction |
| `app/erp/mock_adapter.py` | 3 | deterministic fake ERP (5 fictional records; token search; no network) | query/entity+id | `ERPRecord`s | invalid input → `ERPInvalidRequestError`; unknown record → `ERPRecordNotFoundError` |
| `app/erp/log.py` | 3 | minimal action log (timestamp, op, adapter, status, duration, error category) → `logs/erp_actions.log` | — | log line | fail-soft; never logs secrets, query text, or record data |
| `app/tools/searxng_client.py` | 1 | `search(query, num_results)` against the SearXNG JSON API | query | `[{title, url, snippet}]` | never raises; returns `[]` on any failure |
| `app/tools/crawl_extractor.py` | 1 | `extract(url)` → clean page text via Crawl4AI (sync wrapper over its async API) | url | text or `""` | never raises; `""` on failure; ASCII-sanitized log lines |
| `ingest_books.py` | 1 | offline: parse PDF/DOCX → embed → persist Chroma; manifest-driven incremental ingest | files in `knowledge/books/` | `storage/` collection + manifest | binary-garbage guard raises loudly instead of silently indexing raw bytes |

Dependencies worth noting: `books_agent` bridges the shared factory into
LlamaIndex with a small `_GetLlmBridge(CustomLLM)` subclass — LlamaIndex
expects its own LLM interface, and the dedicated bridge package is broken with
modern LangChain, so the bridge forwards each flat prompt to `get_llm()`.

## State flow

`AgentState` (defined exactly once, in `app/state.py`) is a `TypedDict,
total=False` passed through every node. Nodes return **only the keys they
change**; LangGraph merges them.

```text
{"question": ...}                      ← main.py
   → supervisor:      route, sub_routes?
   → research_agent:  answer, sources, agent_answers["research"]
   → books_agent:     answer, sources, agent_answers["books"]
   → feasibility_agent: answer (report path + study), sources, agent_answers["feasibility"]
   → erp_agent:        answer, sources, agent_answers["erp"]   (ERP route; no new keys)
```

**Phase 3 state note: `AgentState` was NOT changed.** The ERP agent reads
`question` and prior `sources` and writes only existing keys through the
existing `with_agent_output()` helper. The supervisor now also emits
`route: "erp"` (a value `AgentState` already documented since Phase 1).
AgentState remains the single state definition — no second state system.

Why `agent_answers` exists (Phase 2): the feasibility agent needs the research
answer AND the books answer simultaneously. Without it, each agent would
overwrite `state["answer"]` — the second agent's output would erase the
first's, and feasibility would see only books. `agent_answers` accumulates one
entry per agent; `with_agent_output()` (the single merge helper in
`app/state.py`) records the agent's own answer and appends provenance to
`sources` instead of replacing them. The top-level `answer` remains "what the
last contributing agent said", which is what the CLI prints.

## Why agents never call other agents directly

Spec rule (Section 0): *no node calls another node directly; all routing goes
through the Supervisor.* This keeps the graph the single place where execution
order is visible, lets routing decisions change without touching agent code,
and makes every hop inspectable/traceable by LangGraph. The feasibility route
reuses the existing `research_agent` and `books_agent` as graph nodes rather
than importing and invoking them — data flows between agents exclusively
through `AgentState`.

## Standardized agent pattern (Phase 4)

Future agents follow one documented contract ([docs/AGENT_DEVELOPMENT.md](AGENT_DEVELOPMENT.md)):

1. **One module** in `app/agents/`, one public entry point:

   ```python
   def agent_name(state: AgentState) -> dict:
       ...
   ```

2. **Returns only the keys it changes** — usually `answer`, `sources`, and
   `agent_answers`. Nodes never touch unrelated state.
3. **Records provenance** through the existing `state["sources"]` + the
   shared `with_agent_output(state, agent, answer, sources)` helper. There is
   no second provenance system.
4. **Never calls another agent** — data flows between agents exclusively
   through `AgentState` (the feasibility route already relies on this).
5. **Uses the shared LLM factory** `get_llm()` from `app/llm_client.py` for
   every model call. No agent instantiates a model.
6. **Never raises** — failures at tool, adapter, or model boundaries are
   translated into fixed safe messages or graceful fallbacks inside the
   agent; the agent is the last safe boundary.

Adding an agent is a **five-line explicit wiring change** — the graph stays
readable, with no plugin system, DI container, or dynamic registry:

```text
agent module  ->  graph.add_node("name", agent)  ->  supervisor route map  ->  conditional edges  ->  END
```

See [docs/AGENT_DEVELOPMENT.md](AGENT_DEVELOPMENT.md) "How to add a new agent" for
the exact step list. A test-only double (`app/agents/test_dummy_agent.py` +
`scripts/test_agent_architecture.py`) proves the pattern without adding a
permanent production agent.

## Configuration rule

All model names, paths, ports, and thresholds live in `app/config.py` — the
only file allowed to know them. The provider switch (`LLM_PROVIDER = "gemini" |
"ollama"`) and `get_llm()` live in `app/llm_client.py`; agents never build a
model themselves. API keys are resolved from `.env` only and normalized into
the process environment so libraries that read `os.environ` directly cannot
pick up a stale OS-level key.
