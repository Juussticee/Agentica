# Architecture

Status: reflects the code after Phase 5 (production QA gate), Phase 6
(opt-in LLM routing), Phase 7 (conditional planner) and Phase 8 (dynamic
plan execution). Phase 3 REAL ERP integration is NOT implemented — the
ERP vendor is intentionally unspecified.

Recovery additions (Control Center preview): `app/execution/` (events,
store, runner, errors), `app/api/server.py` (FastAPI v1),
`app/ai_settings.py` + `app/ai_diagnostics.py` (GUI-16),
`app/services_health.py` (bounded live service probes), and a Vite+React
frontend under `frontend/`. The graph itself is unchanged except that every
route now terminates through the deterministic `qa` node, and the runner
records two additional event types (`routing.decided`, `qa.completed`) so
the operations dashboard can show the real execution sequence.

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

Phase 7+8 complex route:
START → supervisor → planner (complex request) → dispatcher ⇄ (one whitelisted
node per visit: research/books/erp/feasibility) → QA when every task terminal
START → supervisor → (route table) → agents → END
Simple and multi requests share the exact pre-Phase-7 edges (their conditional
edges and route table are unchanged; they never reach planner/dispatcher).

Phase 3 ERP route:
START → supervisor → erp_agent → ERP Adapter Interface → MockERPAdapter → END
(the mock adapter is eventually replaced by the real ERP adapter — the agent
and the interface stay unchanged)

2026-10-28 routing policy + outbound research approval:
START → supervisor → general (model knowledge) / unsupported (guidance) /
conversational (fixed reply) → QA     — no agent node, no tools, empty provenance
2026-10-29: code/prose AUTHORING is `general`, not `unsupported` — "Write a
Python function that parses a CSV file" is answered in the reply (the agent
cannot create files or run commands), while an operation on the user's
machine ("write a script to a file on disk", "install", "deploy") stays
refused, so the action rule never refuses knowledge work and never grants a
tool that does not exist.
Every path into research_agent (direct route, books->research fallback,
plan-dispatched task) crosses research_entry: without in-message
authorization the answer is an approval REQUEST — proposed action, why,
and how to authorize exactly this operation — and NO search/crawl runs.
GUI-10 backend (2026-10-28): the unauthorized gate also PERSISTS the one
`ApprovalRequest` (store, fingerprint-idempotent create/reuse) bound to the
owning execution via the `approval_scope` ContextVar the runner sets around
invoke/resume, and the reply carries its approval_id. `POST
/api/v1/approvals/{id}/approve|reject` persists the decision (terminal
states never overwrite; a repeat of the same decision is idempotent, a
conflicting one is 409, expiry 410); `POST .../resume` →
`GraphRunner.resume_approval()` revalidates the PERSISTED decision, the
bound payload and a single-execution claim inside the real gate before any
search/crawl runs — rejection finalizes the execution with the honest
local-only answer. A store failure at gate time means denial, never
execution. Events: approval.created / decided / executed / finalized.
2026-10-28 follow-up: when the paused execution has a persisted PLAN, the
approval CONTINUES it — `released_plan()` marks the approval-blocked
research task completed and re-opens dependency-blocked tasks, then
`continue_plan()` replays the remainder through the same `dispatcher_node`
(completed tasks never re-run; a later research task must pass the gate
again and pauses the execution for its own decision). The run is
`completed` only when the whole plan finished. Recovery: the released plan
and the research answer are persisted BEFORE the continuation, so if the
process stops in between, a retried resume continues the remaining tasks
from that durable proof instead of re-searching (`resume_claim` on the store
serialises resumes of one execution — a DURABLE lease held in
`executions.resume_lease_owner` / `resume_lease_expires_at`, acquired by one
atomic conditional UPDATE, so a second API process on the same SQLite file
cannot continue the same execution at the same time; an expired lease is
taken over, a stale owner's release never clears the new holder's claim, and
the lease never authorizes repeating an external operation: the persisted
`approvals.executed_at` remains that durable guard). A plan may contain MORE than one research task: each one creates its own
bound request, so the execution pauses again after the first is executed and
only the decision on that second request continues it (proved end to end in
`scripts/test_multi_approval_api.py`). Replaying an already-spent resume is a
denied no-op that appends a single `approval.finalized` note: the recovery
path below is skipped whenever ANOTHER pending request is already bound to the
same execution — a plan that legitimately stopped at the next gate is not an
interrupted one.
`continue_plan(on_progress=…)`
reports the plan after EVERY task and the runner persists it, so a crash
mid-continuation also resumes from the last finished task instead of
replaying it (a task persisted as `running` cannot be proven either way and
closes the run `failed/resume_interrupted`). If no persisted result proves
what the approved operation returned, the run is closed `failed`
(`resume_interrupted`) rather than guessed at.
2026-10-29 completion of that path: a crash AFTER the last task but before
the terminal write leaves a COMPLETE plan on a paused run — the retry then
finalizes `completed` from the persisted plan/answer (QA recomputed) without
dispatching anything again; and a plan that still records a research task as
approval-blocked proves the decision was never released, so an earlier
gate's result is NOT accepted as proof and the run closes honestly instead
of continuing. Both cases are covered at the runner level and through the
API (`scripts/test_approval_lifecycle.py`,
`scripts/test_multi_approval_api.py`).
Conversation (2026-10-29): short acknowledgements ("thanks, that helps",
"great, thank you so much") are conversation, not underspecified requests —
`is_conversational` accepts them through an ALL-WORDS rule
(`_is_acknowledgement`: <=6 words, only acknowledgement filler plus at
least one anchor), so "Thanks, how do I sort a list?" keeps its real route.
The same predicate still guards QA against a sourced answer to a greeting.
Task-scoped execution (2026-10-29): a planned sub-request now runs as its
OWN instruction — `_dispatch_task` hands the specialist a state whose
`question` is the task's `objective`, with the original request beside it as
`state["request"]` (no new graph key: the copy lives only inside the task
call). Two tasks with different objectives therefore do different work
instead of every specialist answering the whole request. Three boundaries
keep this safe: (a) a `research` task ALWAYS runs the original request, so a
plan-generated objective can never widen the approved external operation, and
`research_authorization`/the approval payload/`_pending_approval_update` read
`request_text(state)` (request first, question as the fallback) so a
sub-request can never self-authorize research; (b) the planner's TEMPLATE
objectives (`app.planner.GENERIC_PLAN_OBJECTIVES`, pinned by a drift test)
describe a step rather than asking anything, so they keep the original
request; (c) `synthesis_node` composes its answer to the original request,
not to its own step description. Single-step routes never enter the
dispatcher, so their state is byte-identical.
Objective-level QA (2026-10-29): beyond the per-task status checks, the QA
gate reports structural CONTRADICTIONS between what the plan claims and what
it recorded — a task recorded `completed` while a prerequisite did not
complete, a task still `running` when the execution ended, an approval
REQUEST (`app.supervisor.APPROVAL_PENDING_MARKER`, one definition shared with
the runner) recorded as a completed step's result, and two DIFFERENT
capabilities returning indistinguishable long results (reported as a
possibility, never proof; steps of the same capability and short results are
never compared). Deliberately no entity or substring matching: a result that
names an entity differently, or explains it instead of repeating the words,
can never fail a task, and when the evidence cannot decide, nothing is
reported. QA findings never change an execution's status by themselves.
Numerical grounding (2026-10-31): a `general` or `feedback` answer states
figures no source in the run supplies — percentages, multipliers, data and
link rates — so QA names them ("this run cannot support … treat them as
unverified"). It is not a fact check: arithmetic on the user's own numbers and
figures the user or a cited source supplied are never flagged, and flags stay
warnings.
Plan honesty (2026-10-29): a dispatched task is `completed` only when its
node returned a non-empty answer (a blank/missing answer is `failed` with
"the task produced no output"), QA flags any completed task whose recorded
result is missing or blank, and `CAPABILITY_AGENT_KEYS` maps `synthesis` to
the `synthesis` channel that `synthesis_node` actually writes."
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
| `app/graph.py` | 1+2 | builds and compiles the single `StateGraph`; supervisor → node edges; feasibility chain conditionals; (2026-10-28) `general`/`unsupported` answer nodes and `research_entry`, the approval gate every path into the research node crosses (direct edge, books fallback, plan dispatch — `_dispatch_task` marks an unauthorized research task `failed` with a reason instead of `completed`) | — | compiled graph | — |
| `app/state.py` | 1+2 | the ONLY definition of `AgentState` + `with_agent_output()` merge helper | — | — | — |
| `app/supervisor.py` | 1 | deterministic routing policy: conversational > feasibility > ERP > explicit local-only > current-info / explicit web > books; then (2026-10-28 Priority A) `unsupported` (no such tool, no conversation memory, underspecified) > `general` (model knowledge; 2026-10-29 also code/draft authoring that needs no tool — see `_TEXT_AUTHORING_RE`/`_SYSTEM_ACTION_RE`) | `question` | `{"route": ...}` (+ `sub_routes` for feasibility) | no silent research default since 2026-10-28 — and `research_authorization()` fail-closes the research node: only an explicit web request, a current-information request, or a feasibility study in the USER'S OWN MESSAGE authorizes the external step (per-request, never a blanket permission) |
| `app/llm_client.py` | 1+2 | `get_llm()` shared factory; `get_llm(fast=True)` for short calls; `llm_text()` normalizer | — | LangChain chat model / plain text | raises with a clear message when the provider is gemini but no key is configured |
| `app/agents/books_agent.py` | 1 | books facade + orchestrator (Increment 2 split): one run's retrieve → strip → refuse/recover → ground → verify → footer sequence; owns the patchable seams (`_get_query_engine`, `_query_engine`, `_embedding_model`, `get_llm`, `STORAGE_DIR`, `CHROMA_COLLECTION`, `BOOKS_VERIFIER`, …) and re-exports the three modules below so every existing import/patch keeps working | `question` | answer + `sources=["books"]` | not-found refusal (prompt-level exact phrase + `BOOKS_MIN_SCORE` hard floor) instead of guessing |
| `app/agents/books_retrieval.py` | 1 | retrieval: builds the LlamaIndex query engine over the PERSISTED Chroma index (embedding model, LLM bridge, LLM-context path filter); never re-ingests | — | query engine | `chroma_guard.assert_store_path_allowed` fails closed before any store open |
| `app/agents/books_grounding.py` | 1 | grounding: refusal-shape detection, citation/footer validation, the deterministic token-support gate (polarity/numeric/injection), bounded paraphrase families, claim evidence + the bounded verifier step — pure functions over retrieved nodes | answer text + retrieved nodes | `(mode, kept, claims)` or refusal | unsupported content dropped + disclosed; nothing substantiated → mandated refusal with no footer |
| `app/agents/books_recovery.py` | 1 | recovery: the ONE bounded M5 regeneration for a provable model self-refusal + its question/evidence eligibility screen | question + retrieved nodes | recovered text or `None` | at most one retry; a second refusal abstains; recovered text re-enters the unchanged grounding gates |
| `app/agents/research_agent.py` | 1+2 | query rewrite → SearXNG → noise filter → Crawl4AI → synthesis with URLs | `question` | answer + `sources=["research", ...urls]` | every tool boundary is fail-soft: rewrite falls back to the raw question, empty search returns a "no results" answer, failed crawls fall back to snippets |
| `app/agents/feasibility_agent.py` | 2 | synthesizes Market/Financial/Risk study from research + books answers; writes `.docx` | `agent_answers` from state | answer containing the report path; file in `reports/` | web-only degradation when books output is missing/placeholder/refusal; `Limitations` section + `⚠` answer flag |
| `app/agents/erp_agent.py` | 3 | the ONLY caller of the ERP boundary: formats adapter results; no LLM call (zero quota) | `question`, prior `sources` | answer + `sources=["erp"]` + `agent_answers["erp"]` via `with_agent_output` | every adapter error mapped to a fixed safe message; never raises; no internal/credential detail reaches the answer |
| `app/erp/adapter.py` | 3 | vendor-neutral READ-ONLY `ERPAdapter` ABC (`name`/`search`/`get_record`) + frozen `ERPRecord` + `ERPError` taxonomy | — | — | no write methods exist by construction |
| `app/erp/mock_adapter.py` | 3 | deterministic fake ERP (5 fictional records; token search; no network) | query/entity+id | `ERPRecord`s | invalid input → `ERPInvalidRequestError`; unknown record → `ERPRecordNotFoundError` |
| `app/erp/log.py` | 3 | minimal action log (timestamp, op, adapter, status, duration, error category) → `logs/erp_actions.log` | — | log line | fail-soft; never logs secrets, query text, or record data |
| `app/tools/searxng_client.py` | 1 | `search(query, num_results)` against the SearXNG JSON API | query | `[{title, url, snippet}]` | never raises; returns `[]` on any failure |
| `app/tools/crawl_extractor.py` | 1 | `extract(url)` → clean page text via Crawl4AI (sync wrapper over its async API) | url | text or `""` | never raises; `""` on failure; ASCII-sanitized log lines |
| `ingest_books.py` | 1 | offline: parse PDF/DOCX → embed → persist Chroma; manifest-driven incremental ingest | files in `knowledge/books/` | `storage/` collection + manifest | binary-garbage guard raises loudly instead of silently indexing raw bytes |
| `app/services_health.py` | ops | `services_snapshot()`: bounded (≈3.5 s budget) live probes for agentica_api, SQLite storage, Ollama `/api/tags`, SearXNG, and the ERP adapter (in-process); each result carries state, optional latency, failure reason, and a next diagnostic action | config/readiness state | JSON snapshot for `GET /api/v1/services` | each probe is fail-soft: a broken service yields `unreachable`/`misconfigured`/`degraded` with a reason, never an exception |
| `app/execution/events.py` + `runner.py` | ops | append-only event log; also records `routing.decided` (with `agent_id`, after the supervisor decides) and `qa.completed` (with `agent_id`) so the sequence endpoint and Agents page report event-derived truth; (2026-10-28 GUI-10) emits `approval.created/decided/executed/finalized` and `GraphRunner.resume_approval(approval_id)` resolves a decided approval through the real gate with the bound payload | runner lifecycle | persisted events | events are written even on failed executions; failed agents remain visible |

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

## Operations endpoints (GUI-5/GUI-7 slice)

`app/api/server.py` exposes three read-only endpoints behind the existing v1
prefix. All derive from real state — no synthetic or cached-status data:

* `GET /api/v1/services` — `services_health.services_snapshot()`: one bounded
  probe round for every configured service (backend, SQLite storage, Ollama,
  SearXNG, ERP adapter).
* `GET /api/v1/agents/activity` — per-agent state (`idle`/`running`/`failed`)
  derived exclusively from recorded events; agents idle between executions
  because this build runs one request at a time in one process.
* `GET /api/v1/executions/{id}/sequence` — the execution's stages rebuilt
  from its event log (created → supervisor → routed → agents → qa →
  completed; complex requests additionally record a "Planning the work"
  step and now EXECUTE the plan through the Phase 8 dispatcher — one
  whitelisted capability→node hop per dispatcher visit
  (`research→research`, `books→books`, `erp→erp`, `synthesis→synthesis`;
  the plan data can never name a node directly), looping until every task
  has a terminal status (pending/completed/failed/blocked/unsupported) or
  the hard visit cap `MAX_PLAN_TASKS*(1+MAX_TASK_RETRIES)` fires and the
  plan fails safe. Per-task status/result bookkeeping lives ON the plan
  dict (no new AgentState keys), and deterministic QA now also verifies
  plan completion and explicitly flags an unavailable AI planning service
  instead of degrading silently.

The terminal `synthesis` task is the answer to the request (2026-10-28):
`synthesis_node` composes it from the results the OTHER tasks actually
produced (`agent_answers`), attributes each finding to its step, and — when
the AI provider is down — returns those results verbatim with an explicit
note instead of inventing a conclusion. It writes no file: before this the
`books`/`erp`/`research` default plans dispatched `synthesis` to the
feasibility agent, so an ordinary multi-step question ended with a
Market/Financial/Risk business study and a stray `.docx`. The feasibility
route is unchanged — there the terminal task IS the study, so
`synthesis_node` delegates to `feasibility_agent` for `route ==
"feasibility"`.

The frontend pages (Services, Agents, Tasks & Sequence) consume exactly these
endpoints, and label anything not implemented (planner views, approval modes)
as planned rather than absent-silently.

## Specialist registry (2026-10-30)

[`app/agents/registry.py`](../app/agents/registry.py) states, for every agent,
its node, entry point, routes, planner capability, **tools**, **permissions**,
the state keys it reads and writes, what it REFUSES, its model use and its
tests. It is passive data — nothing at runtime imports it — and
`scripts/test_agent_registry.py` fails if it drifts from the graph, the planner
whitelist or the dispatcher map. The human-readable companion is
[docs/AGENT_REGISTRY.md](AGENT_REGISTRY.md).

## Dataset analysis — the `data` specialist (2026-10-30)

A request naming a `.csv`/`.tsv` file routes to `data` **only when a workspace
is configured** (the same `CODING_WORKSPACE_ROOT`), because the analyst reads
through that same boundary — it adds no new filesystem permission.

Two layers, deliberately separated:

1. **`app/data/dataset.py`** — a bounded, read-only reader: size cap before
   reading, row cap with the truncation recorded, delimiter sniffing, unique
   column names (duplicates and blanks are renamed, never silently merged),
   ragged rows preserved, numeric typing by INFERENCE over the loaded sample
   (whole numbers with thousands separators are integers; a text column stays
   text).
2. **`app/data/analysis.py`** — a CLOSED set of operations whose numbers are
   computed in Python: `profile`, `value_counts`, `group_agg` (sum/mean/min/
   max/count), `filter_count`. Anything it cannot compute exactly (a text
   column asked for a sum, an unknown column, an unsupported operator, a
   blank-but-present measure) is REFUSED with a reason — never approximated,
   never shown as a zero.

The model sits at two narrow seams in `app/agents/data_agent.py`: it CHOOSES
which operations to run (JSON, validated against the closed set) and it
EXPLAINS the computed tables. It never sees a row of data — only the schema —
and it never produces a figure. Consequences the tests pin:

* with the provider down, `profile` is still computed and the answer says that
  no further analysis was planned;
* a step the model invents outside the set appears in the answer as
  "`<op>` was NOT run: <reason>", so the user can see what did not happen;
* every table states its coverage (whole file, or the first N of many rows).

QA adds warnings (never failures) when the data route produced no computed
results, or when part of the analysis was refused.

## Workspace coding (2026-10-30)

`route == "coding"` is the first capability that touches real files. It is
**off unless a workspace root is configured** (`CODING_WORKSPACE_ROOT`, an
absolute path, read at call time — see `app/config.py`); with no root the
supervisor never routes to it and `app/coding/workspace.py` refuses every
operation with a `WorkspaceUnavailable` message that says what to configure.

Smallest end-to-end workflow, no second orchestration layer:

1. **Inspect / explain** (`app/coding/agent.py`, read-only): the node reads
   only the files the request NAMES (`app/graph.py`) plus the files where a
   named symbol occurs (bounded literal search, max 3 files, 12k chars each),
   REDACTED, and answers from that evidence. With no resolvable file it says
   so and asks for the path — it never walks or guesses the tree. Nothing is
   written and no approval request exists.
2. **Prepare a patch**: the model returns exact find/replace edits over the
   files it was shown; `app/coding/patch.py` re-reads every target from disk,
   builds the unified diff, statically validates the PROPOSED content (Python
   compiled in memory, JSON parsed — never executed, never imported), and
   persists ONE approval request with `operation="workspace_patch"` through
   the existing approval lifecycle.
3. **Apply only with approval**: `GraphRunner.resume_approval` dispatches on
   the PERSISTED operation and re-verifies the bound question, the workspace
   identity, the payload digest, and EVERY file's pre-change sha256 before a
   single byte is written (`os.replace` per file). Any mismatch →
   `stale_patch`, nothing written. The approval's `executed_at` claim keeps the
   write single-shot; if a resume consumed the claim and then died before its
   terminal write, recovery compares each file against the reviewed
   before/after hashes and reports `completed` / `not applied` / `unproven`
   from that evidence — it never re-applies.

Boundary rules: workspace-relative paths only (`..`, absolute paths, NUL bytes
refused), `resolve()` followed and re-checked so symlink/junction escapes are
refused, credential-shaped files (`.env*`, `*secret*`, `id_rsa*`, keys, DBs)
and dependency/VCS directories denied, source-extension allowlist, size caps,
and `redact_secrets()` on everything that leaves the module (prompts, answers,
errors, events). No shell, no arbitrary execution, no walk of the filesystem.

QA never treats a prepared patch as an applied one, and flags a prepared patch
with no approval binding it or with failed offline validation. Ordinary code
questions still go to `general`; the coding route needs an explicit
workspace/file intent AND a configured root, and other source cues
("search the web", "the manual") still win.

## Continuous learning (2026-10-30)

Agentica improves from corrections and failures through four separated layers
(`app/learning/`), stored in the existing SQLite store — not a second memory
system and not a new orchestration framework:

| layer | file | what it decides |
| --- | --- | --- |
| signal detection | `signals.py` | is this message a correction, and what kind? Deterministic, no model call. Kinds: `insufficient`, `incorrect`, `misunderstood`, `format`, `future_directive`, `repeated`, `failure`, with categories `preference` / `project` / `procedure` / `experience`. Also flags `unsafe_change` (privilege/security wording). |
| lesson extraction | `lessons.py` | one model call UNDERSTANDS the correction (problem, cause, corrective behaviour, validation, keywords); the output is parsed into fixed fields and clamped. A field that holds the prompt's own TEMPLATE (`<what went wrong, one sentence>`, observed live with qwen3:0.6b on 2026-10-31) counts as missing, never as a finding: that field, and any deterministic reading the provider could not supply, comes from the user's own words (`_directive`), so a weak local model cannot store a placeholder as an active rule. The call is bounded (`ANALYSIS_TIMEOUT`) so a stalled provider falls back instead of blocking the run. Then dedupe — when both corrections NAME what was missing, those names decide the topic, so two different requirements never merge into one lesson — conflict handling (a different preference/procedure for the same topic supersedes the old one, which stays inspectable), and the activation policy. |
| retrieval | `retrieval.py` | which ACTIVE lessons a task must respect, scored by stem-matched question overlap + confidence, capped at 4, rendered as a short quoted block. Capability must match (or the lesson is `any`-scoped). A CONTENT lesson (`impact: quality_improvement`) additionally needs a word overlap with the request: applying "include trade-offs with figures" to "what does HTTP 429 mean" produced a rule-shaped non-answer in the 2026-10-31 live check. A preference or constraint inside the scope still applies to anything in it. |
| failure observation | `failures.py` | a failed run is not a lesson by itself. The execution's signature (route + error category + normalised QA findings) is compared with recorded executions, and ONE `candidate` lesson is created (or the existing one merged) only at the second occurrence. Deterministic, no model call, never auto-applied. |
| skill versions | `skills.py` | a capability's procedure (`research-comparison`, …) rebuilt from its active lessons. A candidate is activated only when a deterministic gate passes (`Scope`/`Steps`/`Validation` sections, length, lesson coverage, no privilege directive, provenance) and is re-evaluated from the STORED text at activation; the previous version is retained so a rollback is exact. |

**Activation policy** (`lessons.decide`, deterministic). Auto-active only for an
explicit user correction. Held as `candidate` for review when it asks for a
permission/security change, reads like an injected instruction, is learned from
an execution failure (needs evidence), or is a factual correction with no
capability scope (would leak into unrelated answers). Nothing here can change a
tool permission, a security boundary, an approval rule, credentials, provider
configuration or code.

**Execution feedback.** `GraphRunner` calls `failures.observe` after every terminal state — an incomplete plan, a graph/provider exception, or a completed run whose QA reported findings — and emits `learning.lesson_recorded` (severity warning) when it stores something. A clean completed run records nothing, and the same signature on another route is a different failure.

**Where learning acts.** `GraphRunner.run` reads the ACTIVE set once per run and
places it in `state["lessons"]` (no node reads the store, so a test can inject a
precise set). Nodes that build a prompt (`general`, `synthesis`, `research`, the
planner) insert the applicable block and report the ids they used in
`state["lessons_applied"]`; the runner counts those on the execution and emits
`learning.lessons_applied` with the QA outcome. The `feedback` node records the
lesson, emits `learning.lesson_recorded`, refreshes the capability's skill
version when it has ≥2 active lessons (`learning.skill_version`), and composes
its revised answer only AFTER recording — so it can state what was stored and
never claims a save that failed.

**Routing.** A message detected as a correction (or a revision of the previous
result) routes to `feedback`, which revises the recorded answer and learns,
rather than being treated as a fresh question. An ordinary request is not
feedback.

**QA.** `qa_node` proves two things deterministically: a feedback answer may not
claim a correction was recorded when `state["learning"]["learned"]` is false,
and a reported `lessons_applied` id must be one the run was given. It makes no
judgement about whether a lesson is *good* — that is the skill gate and the
user's review.

**Control Center.** `GET /api/v1/learning` projects the `lessons` and
`skill_versions` tables (counts by status, active/candidate/rejected/disabled/
superseded, repeated-failure patterns computed from stored capability+category
pairs, skill versions, awaiting-approval). `POST /api/v1/learning/lessons/{id}/
review` and `POST /api/v1/learning/skills/{skill_id}/rollback` are the only
paths that change learning state after recording; both are audited in the
lesson's evidence. The Memory page reads the same rows (preferences, learned
procedures), so there is one source of truth.

**Limits.** Learning is not an evaluator of answer quality: a lesson is useful
only if subsequent prompts carry it and the run's QA does not regress, which is
what `learning.lessons_applied` (with `success`) records. No automatic
before/after benchmark runs; lesson confidence is a bookkeeping value, never
proof.

## Learning evaluation — does learning actually help? (2026-10-30)

`app/learning/evaluation.py` (scoring, comparison, report) and
`app/learning/benchmark.py` (A/B execution) measure the learning loop without
changing it. `scripts/eval_learning.py` is the offline CLI.

**Conditions.** Each task runs twice through the REAL `GraphRunner`:

| condition | runner | lessons |
| --- | --- | --- |
| baseline | `GraphRunner(store, lessons_enabled=False)` | none supplied — nothing is disabled or deleted |
| learning | `GraphRunner(store)` | the seeded ACTIVE lessons, selected by the real relevance function and inserted by the real nodes |

Every task gets its own throwaway store seeded only with the fixtures it
declares, so one task's lessons cannot change another's measurement. Both
conditions use the same request, provider and evaluation criteria.

**Observation, not mutation.** Each measured run executes inside
`app.learning.writes_disabled()` (a contextvar): `lessons.record`,
`feedback_node` and `failures.observe` record nothing while it is set, and the
harness asserts the lesson id set is unchanged afterwards. A benchmark can
never teach the system it is measuring, and it never activates, rejects or
edits a lesson, skill or permission.

**Dataset.** `benchmarks/learning_eval_v1.json` (version-controlled) declares
the lesson fixtures and, per task: `task_id`, `capability`, `request`, the
`baseline_answer` a scripted provider returns, the requirement checks, the
lesson fixtures in scope and an `expected_outcome`. Requirement checks are
declared, deterministic predicates (`contains_all`, `contains_any`,
`not_contains_all`, `regex`, `min_words`) evaluated on the STORED ANSWER — never
on the prompt. `validate_dataset` refuses a file with an unknown check, an
unknown fixture, a duplicate id or a case without requirements.

**Provider.** The mode is `offline_deterministic`: a scripted double answers
with the case's baseline text, appends the learned requirement it was given,
and obeys an explicit one-sentence directive. No live model, no network. The
report says `live_model: false`; live comparisons would be a separate, labelled
mode.

**Scoring.** Per condition: requirement completeness (satisfied/total, missing
ids), QA findings (count and text — the store records no severity, so none is
invented), execution reliability (status, failure kind, timeout, empty answer),
and the applied/retrieved lesson ids (applied comes from the execution's own
`learning.lessons_applied` event; retrieved is computed with the same selector
the nodes use). `compare_task` names each regression explicitly:
`requirement_lost`, `qa_regression`, `execution_regression`, `output_lost`; a
case with both a gain and a loss is `mixed`, never rounded up to "improved".
Per-lesson credit comes only from the outcome diff of a task where the lesson
was applied; a retrieved-but-not-applied lesson is listed and NOT credited.

**Report.** JSON + Markdown with the benchmark version, mode, provider,
timestamps, totals, deltas (percentage points, QA, executions, empty answers,
regressions), per task, per capability, per lesson and the caveats.
`scripts/eval_learning.py` writes both under `storage/learning_eval/`.

**Control Center.** `GET /api/v1/learning/evaluation` reads those report files
only (newest 10, malformed files skipped and named) and the Learning page shows
the baseline-vs-learning table, per-task outcomes with links to both executions,
regressions, per-lesson effectiveness and earlier runs. No report means "no
benchmark has been run yet", with the command to run it — never placeholder
numbers.

## Evidence-grounded general answering (2026-11-01)

The `general` route no longer answers from model knowledge alone. Before
composing an answer it queries the SAME persisted Chroma index the books route
uses (one index, one embedding wiring, one `chroma_guard` call — `build_retriever`
in `app/agents/books_retrieval.py`, cached and patchable as `_get_retriever` on
the `books_agent` facade) and asks the deterministic coverage question in
`app/evidence.py`: do these passages cover the question's own content words, not
merely score highly?

Three outcomes, each reported honestly in `state["evidence"]`
(`{status, answered_from, coverage, best_score, missing, conflict, files, used,
text, …}`; `text` stays in memory for QA, the runner never persists it):

* `grounded` — composed from the passages only, then run through the EXISTING
  `books_grounding._grounded_repair` gate: unsupported and fabricated-citation
  sentences are dropped, the References list is built from the nodes that
  supported a retained sentence, and provenance is published as `evidence:<file>,
  page N`. Adversarial prompt text inside a passage is still never treated as an
  instruction.
* `partial` — same, plus an explicit "Not stated by the passages: …" disclosure;
  conflicting licence evidence is disclosed instead of silently resolved.
* `none` — the answer comes from `memory with an unconditional label ("Answered
  from my own knowledge…") and NO sources; if the model composed something the
  passages do not support, the gate refuses it and the labelled memory answer is
  used instead. No path invents a citation or turns an unsupported answer into a
  refusal of the whole question.

QA (`app/qa_node.py`) gained the checks this record makes possible, and only
those: figures the answered-from passages literally state are supported (figures
labelled as estimates, and plain arithmetic on user numbers, are not flagged);
licence claims are classified per subject (unsupported / contradicted / silent);
citation filenames are matched WHOLE against the retrieved set (a dotted name is
not split); an evidence-built answer with no `evidence:` provenance, and a memory
answer that points at "your documents" when nothing relevant was found, are
flagged. Nothing else is claimed to be verified — version, date, security and
general factual claims remain unverifiable here and are documented as such in
`docs/EVIDENCE_GROUNDED_ANSWERING.md`.

`evidence` is a DECLARED `AgentState` key. That is load-bearing, not tidiness:
LangGraph persists only declared keys, so an undeclared channel is dropped
between the node and QA (found live 2026-11-01 — the licence check silently saw
no evidence record).

## Retrieval reranking, claim verification and evidence snapshots (2026-11-01)

Three additions to the evidence layer, all deterministic and offline:

* **Reranking** (`app/evidence.py::select`) — `similarity + 0.40 × question-word
  coverage + 0.20 × exact technical terms`, then near-duplicate removal (token
  Jaccard ≥ 0.9) and a diversity bonus for passages that add question words no
  better-ranked passage covers. The per-passage decision (similarity, rerank,
  coverage, exact share, position) is recorded in `state["evidence"]["ranking"]`.
* **Claim verification** (`app/claim_check.py`) — every substantive sentence is
  checked against the passage IT CITES, not the union of retrieved material:
  `supported | partial | contradicted | unsupported | unable`. Numbers must match
  exactly (an unstated version/port/percentage is unsupported); a licence
  contradiction needs the same subject AND no supporting passage (two documents
  disagreeing is a disclosed conflict, not a contradiction); a cited file that
  was never retrieved is fabricated attribution. No model is consulted — a
  model must not be the sole judge of its own claims. QA reports only the
  non-clean verdicts (`qa_node._claim_verification_findings`) and never fails an
  execution for one weak claim.
* **Evidence snapshots** (`app/evidence_store.py`) — one immutable JSON sidecar
  per execution, `storage/evidence_snapshots/<id>.json`, written best-effort by
  `GraphRunner.run`: passage text as retrieved, chunk ids, file/page, retrieval
  and rerank scores, the ranking, the coverage verdict, the claim verdicts, and
  the ingest-manifest digest as the index identity. Never `file_path` or any
  absolute path. Read-only endpoint
  `GET /api/v1/executions/{id}/evidence` (200 with `evidence: null` and a reason
  when nothing was consulted). **No schema change or migration.**
* **Completeness and both routes** (2026-11-02) — `passage_details` records
  every retrieved passage with a PER-PASSAGE cap (`EVIDENCE_SNAPSHOT_CHARS`);
  the prompt budget (`EVIDENCE_PASSAGE_CHARS`) still bounds what the model
  reads, so each row carries `supplied`, and the snapshot reports
  `passages_retrieved`, `passages_supplied`, `complete` and `partial_reason`
  (false only for a truncated/empty passage, naming the rank). The `books`
  route writes the same record through `state["books_evidence"]` (a separate
  channel so the general route's QA claim checks keep one answer path);
  `runner.snapshot_evidence()` selects whichever channel holds passages.

Design, measurements and limits: `docs/RETRIEVAL_AND_CLAIM_VERIFICATION.md`.
