# Master Plan — Master Agent System (Phase 5 onward)

Status: authoritative roadmap for all remaining development. Phases 1–4 are
complete and committed (HEAD at time of writing: `c527ff3`). **This document
plans future work; it does not implement it.**

How a future coding agent should use this document:

> "Implement Phase X from `docs/MASTER_PLAN.md`."

…must be enough to start work without guessing the architecture. Every phase
below carries its own goal, prerequisites, affected files, exact tests,
acceptance criteria, failure cases, and explicit non-goals. If an
implementation would contradict the "Non-negotiable rules" (Section 3) or the
"What must never be added" list (Section 13), stop and report instead of
proceeding.

---

## 1. Current state (authoritative — inspected at `c527ff3`)

Everything in this section was verified against the actual code, not against
older documentation. Where documentation disagrees with code, the code wins
and the discrepancy is listed.

### 1.1 Inventory

| Area | File(s) | Current reality |
|---|---|---|
| State | `app/state.py` | ONE `AgentState(TypedDict, total=False)`: `question`, `route`, `sub_routes`, `sources`, `answer`, `qa_flags`, `agent_answers`. Merge helper `with_agent_output(state, agent, answer, sources)` is the only provenance mechanism. |
| Supervisor | `app/supervisor.py` | Pure keyword routing, order: feasibility → erp → books → research (fallback). Returns ONLY `route` (+ `sub_routes = ["research","books"]` for feasibility). Never calls agents. |
| Graph | `app/graph.py` | Single `StateGraph(AgentState)`; nodes: `supervisor`, `books`, `research`, `erp`, `feasibility`. Conditional edges: supervisor→node by route; `_after_research`/`_after_books` chain research→books→feasibility only when `route=="feasibility"`. Exposes `build_graph()`. |
| LLM | `app/llm_client.py` | ONE factory `get_llm(fast=False)` dispatching on `LLM_PROVIDER` (`gemini` \| `ollama`); `llm_text()` normalizes Gemini block lists and plain strings. Agents never construct models. |
| Config | `app/config.py` | Single config source: provider, models, `OLLAMA_BASE_URL` (env-driven), `SEARXNG_URL`, `ERP_ADAPTER`, paths, `BOOKS_MIN_SCORE=0.35`, `BOOKS_TOP_K=3`, `NOT_FOUND_PHRASE`. `.env` values win over OS env. |
| Books agent | `app/agents/books_agent.py` | Vector-only RAG: Chroma persisted index via LlamaIndex query engine; embeddings from Ollama `nomic-embed-text`; `_GetLlmBridge(CustomLLM)` forwards flat prompts to `get_llm()`; exact refusal phrase + score floor; `References: file, page` provenance. |
| Research agent | `app/agents/research_agent.py` | `get_llm(fast=True)` query rewrite → SearXNG → noise-host filter → Crawl4AI top-3 → synthesis with cited URLs; every step fail-soft. |
| Feasibility agent | `app/agents/feasibility_agent.py` | Consumes `agent_answers["research"]` + `["books"]`; Market/Financial/Risk study; writes `.docx` to `reports/`; web-only degradation with `⚠` flag + Limitations section. De-facto synthesizer today. |
| ERP agent | `app/agents/erp_agent.py` | ONLY caller of the `ERPAdapter` boundary; no LLM call; fixed safe messages per `ERPError` category; provenance via `sources`/`agent_answers["erp"]`. |
| ERP boundary | `app/erp/` | Read-only ABC (`name`/`search`/`get_record`), frozen `ERPRecord`, `ERPError` taxonomy, deterministic `MockERPAdapter`, fail-soft action log (`logs/erp_actions.log`). `ERP_ADAPTER` config switch (only `mock` exists). |
| Tools | `app/tools/` | `searxng_client.search()` and `crawl_extractor.extract()` — never raise; return `[]`/`""` on failure. |
| QA | *none in `app/`* | Deterministic QA exists ONLY as `run_qa(state) -> list[str]` inside `scripts/test_phase4_workflow_qa.py`. `state["qa_flags"]` is declared but **never written by production code**. There is no `app/qa_node.py`. |
| Tests | `scripts/test_*.py` | 7 deterministic suites, **101 tests + 10 subtests, all green, zero quota** (phase1 routing, phase2, feasibility degradation, rewrite, phase3 ERP ×35, agent architecture ×30, phase4 workflow/QA ×30). Live suites (`verify_graph`, `test_phase1_e2e`, `test_phase1_proxy`, `check_books_agent`, `probe_books_prompt`, `demo_rewrite_quality`) are separate and consume quota/network. |
| CLI | `main.py` | `--once`, `--demo`, interactive loop; catches `GoogleRateLimitError`. |
| Hardware | `docs/Plan.md` | Windows, GTX 1660 Super, **4 GB VRAM budget** — all future local models/embeddings/rerankers must fit this. |

### 1.2 Documented discrepancies found during this review

These are facts, not speculation; they drive Phase 5:

1. `README.md` "Current phase" block says Phase 3 and Phase 4 **NOT STARTED**
   and calls the ERP node a "stub" — both stale (Phase 3 boundary committed
   `8e2b18e`; Phase 4 tests committed `c527ff3`; ERP agent is real).
2. `docs/PROJECT_STATUS.md` says `Phase 4: NOT STARTED` and "QA node … do not
   exist" — half-stale: deterministic QA exists as a *test-suite function*,
   not as a graph node.
3. `docs/ARCHITECTURE.md` header says Phase 4 not implemented — stale in the
   same way.
4. `app/supervisor.py` docstring: "Phase 4 replaces it with an LLM
   classification call" — **not done**; routing is still keyword-only.
5. `docs/Plan.md` §3 specifies `app/qa_node.py` ("Phase 4 only … Appends
   findings to `state["qa_flags"]`") — **the file does not exist**.
6. `.gitignore` does not cover `env/` (the dependency venv is untracked but
   NOT ignored — `git check-ignore env/` exits 1).
7. `README.md` testing section lists only 4 quick suites — missing the
   phase3/architecture/phase4 suites.

### 1.3 What Phase 4 actually delivered (scope reconciliation)

The original spec's Phase 4 had two halves: (a) `qa_node.py` in the graph,
(b) LLM supervisor routing. What was delivered instead was **workflow/
orchestration correctness testing + a minimal deterministic QA layer**, both
as test-suite artifacts, deliberately without touching production routing or
graph. That was the approved scope ("Implement Phase 4 as
workflow/orchestration correctness testing + a minimal QA layer").

Consequence for the roadmap: the *unimplemented* halves of the original spec
become Phase 5 (graph-integrated deterministic QA) and Phase 6 (LLM-assisted
routing) below. This document supersedes `docs/Plan.md` §4 for anything after
Phase 4; `docs/Plan.md` remains the historical build spec.

---

## 2. Target architecture

### 2.1 The proposed diagram, evaluated

The direction under consideration was:

```text
USER → SUPERVISOR → PLANNER → GRAPH/EXECUTOR → {RESEARCH, BOOKS, ERP}
                                          → SYNTHESIZER → REVIEWER
                                          → PASS: response / FAIL: bounded re-plan
```

**Verdict: right components, wrong topology as a mandatory linear pipeline.**

Problems with accepting it as-is:

1. **Planning on every request is waste and risk.** Most requests are single
   capability ("what does chapter 3 say…"). Forcing them through an LLM
   planner adds a quota/latency cost, a new failure surface (malformed plans),
   and destroys the determinism that the Phase 4 tests just proved — for zero
   benefit.
2. **A Planner sitting *between* Supervisor and Graph makes plan machinery
   load-bearing for trivial routes.** The existing architecture is strongest
   exactly where it is now: supervisor decides, graph executes, one agent
   runs, deterministic QA validates.
3. **The diagram hides who owns orchestration.** If Planner → Executor is a
   separate stage outside the graph, the plan can drift from what the graph
   actually executes, and we get two execution stories again. The graph must
   remain the *only* executor, plan or no plan.

### 2.2 Recommended target architecture

```text
                          USER
                            │
                            ▼
                   ┌─────────────────┐
                   │   SUPERVISOR    │  route + mode
                   │ keyword (now) / │  mode: simple | multi | complex
                   │ LLM-assisted    │  (Phase 6)
                   │ (Phase 6)       │
                   └───────┬─────────┘
            ┌──────────────┼──────────────────────┐
            ▼              ▼                      ▼
       mode=simple    mode=multi            mode=complex
            │              │                      │
            │     ┌────────────────┐              ▼
            │     │ existing chain │      ┌────────────────┐
            │     │ research→books │      │   PLANNER      │  Phase 7
            │     │ →feasibility   │      │ structured plan│  (capability
            │     └───────┬────────┘      │ from whitelist)│   whitelist,
            │             │               └───────┬────────┘   validated
            │             │                       ▼            deterministically)
            ▼             ▼             ┌────────────────────┐
   single agent node   (same)           │ GRAPH / EXECUTOR   │  Phase 8
   books|research|     │                │ dispatches tasks by│
   erp|feasibility     │                │ dependency order;  │
            │          │                │ retries are bounded│
            │          │                │ (Phase 12)         │
            │          │                └───┬─────┬─────┬────┘
            │          │                    ▼     ▼     ▼
            │          │               RESEARCH BOOKS  ERP   (existing agents,
            │          │                 AGENT  AGENT  AGENT  unchanged contract)
            │          │                    │     │     │
            │          └────────────┬───────┘     │     │
            │                       ▼             │     │
            │              ┌────────────────┐     │     │
            │              │  SYNTHESIZER   │◄────┘     │   Phase 10
            │              │ (evolved from  │◄──────────┘
            │              │  feasibility)  │
            │              └───────┬────────┘
            │                      ▼
            │              ┌────────────────┐
            │              │ DETERMINISTIC  │   always on, zero quota,
            │              │ QA (run_qa)    │   structural checks → qa_flags
            │              └───────┬────────┘   (Phase 5)
            │                      ▼
            │              ┌────────────────┐
            │              │ SEMANTIC       │   Phase 11; LLM; verdict only;
            │              │ REVIEWER       │   appends to qa_flags; may NOT
            │              └───────┬────────┘   call agents or re-route
            │                 PASS │ │ FAIL ──► bounded correction / re-plan
            │                      │ │          (Phase 12, ≤ N attempts)
            └───────────┬──────────┴─┘
                        ▼
                    RESPONSE
```

Design commitments in this topology:

- **The graph is still the only executor.** The Planner is a *node* that
  writes a structured plan into state; the Executor is the graph itself
  (a dispatcher + the existing agent nodes). Plan data never executes
  anything directly; a code-side capability whitelist maps plan entries to
  nodes.
- **Planning is conditional** (`mode=complex`), not mandatory. Simple and
  multi routes keep their current deterministic paths untouched — which means
  all 101 existing tests remain valid unchanged.
- **QA is a gate, not a stage that can steer.** Deterministic QA always runs
  before END; the semantic reviewer follows it. Neither can route or invoke
  agents. Reviewer FAIL feeds *data* (feedback strings) into the bounded
  correction loop; the graph decides what to do with it.
- **Evolution over duplication**: the feasibility agent's synthesis logic is
  generalized into the Synthesizer (Phase 10) rather than adding a parallel
  synthesis agent; the ERP agent, tools, and all four existing agent
  contracts stay as-is.

Why this beats the alternative for *this* codebase: correctness (existing
behavior paths untouched until each phase proves otherwise), debuggability
(one graph, one state, explicit edges), deterministic orchestration (planner
output validated by code, retries counted in code), testability (every new
seam maps 1:1 to a zero-quota suite), provider independence (planner/reviewer
use `get_llm()` like everyone else), and controlled complexity (each phase
adds exactly one seam).

### 2.3 Route/mode semantics (target)

| Mode | Trigger (Phase 6 target) | Path |
|---|---|---|
| `simple` | single-capability intent | supervisor → one agent node → QA → END (today's behavior) |
| `multi` | feasibility/composite intent | existing research→books→feasibility chain → QA → END (today's behavior) |
| `complex` | genuinely multi-objective request needing decomposition | supervisor → planner → executor → synthesizer → QA → reviewer → END |

Phase 6 may implement mode as an internal supervisor decision while keeping
`route` values identical to today (`books|research|feasibility|erp`), so no
downstream test changes. Only Phase 7 introduces plan-driven execution.

---

## 3. Non-negotiable architecture rules

These rules are binding for every phase in this document.

### 3.1 LLM

- The LLM is an interchangeable reasoning engine. No phase may depend on
  Gemini-specific or Ollama-specific behavior (formats, thinking modes,
  tool-calling quirks). Anything model-specific belongs behind `llm_text()`
  or the factory, never in agents/nodes.
- All model construction continues through `get_llm()` in
  `app/llm_client.py`. **No second factory, ever.**

### 3.2 Supervisor

- Owns high-level routing/workflow selection only. It never executes agents,
  never performs domain work, never fabricates answers.
- Its output contract (`route`, `sub_routes`) must remain stable; new modes
  are additive.

### 3.3 Planner

- Decomposes complex requests into a structured plan of *tasks* (see
  Section 5). It never performs specialist work itself, never calls tools,
  never calls agents.
- Its LLM output is **data**: validated deterministically against a
  capability whitelist before it can influence the graph. It can never
  introduce arbitrary function names, code, or node names.

### 3.4 Graph / Executor

- The graph owns execution: dependency order, conditional execution, bounded
  retries, state propagation. Nothing outside the graph advances a plan.
- Execution order must stay visible as literal wiring/dispatch logic in
  `app/graph.py` — no dynamic node discovery.

### 3.5 Agents

- Unchanged contract (see `docs/AGENT_DEVELOPMENT.md`): `def agent(state:
  AgentState) -> dict`, return only changed keys, provenance via
  `with_agent_output`, never call another agent, never raise, LLM only via
  `get_llm()`.

### 3.6 State

- One canonical `AgentState` in `app/state.py`. No second state object, no
  parallel trace/history structures.
- New keys require written justification in this document *and* in the phase
  that introduces them, and must carry runtime data (not test scaffolding).
  Execution tracing for tests lives in test fixtures, never production state.
- Plan representation ≠ runtime state ≠ agent output (Section 5.4).

### 3.7 RAG

- RAG is a capability/tooling layer (retrieval + grounding), not an
  autonomous agent swarm. No "retrieval agents" that plan or synthesize on
  their own. Evolution is staged per Section 4; no new infrastructure
  without justification in Section 4.

### 3.8 QA vs Reviewer

- Deterministic QA validates structure/workflow of the *completed* result;
  it never routes, never calls agents, never modifies answers — only appends
  `qa_flags`.
- A semantic reviewer (if added) is an LLM judge with a bounded verdict
  contract; it is explicitly NOT a second orchestrator (Section 7).

### 3.9 ERP

- Remain ERP-agnostic until real ERP requirements are explicitly provided
  (vendor, API, auth, data model). No vendor SDK, API client, MCP server, or
  credentials may be introduced before Phase 15's gate is satisfied.

### 3.10 Testing

- Deterministic suites: zero quota, zero network, zero real ERP, zero
  Chroma/SearXNG/Crawl4AI dependency for orchestration claims. Existing 101
  tests are the regression floor for every phase.
- Live/integration tests stay separate, explicitly labeled, and are never
  deleted or weakened to make a new phase pass.

---

## 4. RAG: current state, options, and the first target

### 4.1 What exists today (Option A — vector-only)

```text
chunk (LlamaIndex) → embedding (Ollama nomic-embed-text) → Chroma (cosine)
→ similarity top_k=3 → prompt-with-context (exact refusal phrase)
→ get_llm() → answer + References
```

Known weaknesses (from code + `docs/PROJECT_STATUS.md`):

- **Narrow score band**: nomic score distributions are tight, so a single
  fixed floor (`BOOKS_MIN_SCORE=0.35`) is unreliable; refusal currently leans
  on the prompt-level exact phrase.
- **No lexical retrieval**: exact terms — part numbers, invoice IDs, named
  procedures, rare jargon — are precisely where dense-only retrieval fails,
  and they are common in a company library.
- **No reranking**: top-3 by cosine is accepted as-is.
- **No grounding check**: nothing verifies the answer's claims against the
  retrieved chunks (QA is structural, not evidential).

### 4.2 Option comparison

| | A: vector-only (today) | B: hybrid dense+lexical | C: advanced (decompose/multi-hop/iterative) |
|---|---|---|---|
| Pipeline | chunk→embed→Chroma→top-k→LLM | query → dense **and** BM25 → fusion → rerank → context → LLM | query decomposition → multi-query → hybrid → rerank → iterative hops → synthesis → grounding validation |
| New infra needed | none | none (LlamaIndex `BM25Retriever`/`rank-bm25` = pure Python; Chroma unchanged) | new dependencies (query router, iteration policy), more LLM calls per query, harder to test deterministically |
| Fits 4 GB VRAM | yes | yes (reranker optional, small cross-encoder or CPU) | questionable (adds models + calls) |
| Testability | high | high (fusion/rerank order assertable with fakes) | low (iteration depth is stochastic) |
| Failure modes | silent misses on exact terms | score-merge tuning | unbounded cost, non-termination risk, over-engineering |

### 4.3 Recommendation

**Implement Option B (hybrid) first — but only after RAG-1 stabilization —
and defer Option C until metrics prove B insufficient.**

Reasons:

1. It attacks the *measured* failure class (exact-term / ID lookups and
   unreliable score bands) with zero new infrastructure: LlamaIndex's BM25
   retriever is pure Python, Chroma stays the vector store, embeddings stay
   on Ollama.
2. Fusion + reranking is deterministic and unit-testable with fakes — no new
   stochastic surface.
3. Option C's query decomposition and iterative hops multiply LLM calls
   (quota is 20/day on the free tier) and make behavior harder to debug for
   a corpus of currently two documents. Revisit only when RAG-2/RAG-3
   metrics still show misses on realistic multi-hop evaluation questions.
4. Every new retrieval component must justify itself against the 4 GB VRAM /
   local-only budget in Section 3.7.

---

## 5. Design contracts (referenced by the phase templates)

### 5.1 RAG roadmap (RAG-1 … RAG-6)

Each RAG step states problem, scope, dependencies, tests, acceptance,
and why it must NOT come earlier. Steps are delivered inside the development
phases where noted (RAG-2/3 = Phase 9).

#### RAG-1 — Current vector RAG stabilization (do first; no new retrieval)

- **Problem**: refusal reliability depends on a prompt phrase + one global
  score floor; retrieval quality is unmeasured.
- **Scope**: evaluation corpus of ~20 labeled questions (answerable /
  out-of-corpus / exact-term) with expected refusal or expected chunk;
  retrieval-metric harness reporting hit-rate and refusal correctness;
  tune/document `BOOKS_MIN_SCORE` from measurements, not intuition.
- **Dependencies**: none. Uses existing Chroma index + `scripts/check_chroma.py`.
- **Tests**: `scripts/test_rag1_evaluation.py` — deterministic assertions on
  the harness itself (fake retriever/scores); metric run is a labeled
  integration script (quota).
- **Acceptance**: harness runs end-to-end; baseline hit-rate and
  refusal-precision recorded in `docs/TESTING.md`; refusal behavior unchanged
  (regression 101 green).
- **Why not earlier / why not later**: earlier — no data existed to evaluate
  against; later — every retrieval change after this must beat a recorded
  baseline, so the baseline must exist before RAG-2 code is written.

#### RAG-2 — Hybrid dense + lexical retrieval (Phase 9 core)

- **Problem**: exact-term queries miss under dense-only retrieval.
- **Scope**: dual retrieval (Chroma dense + LlamaIndex BM25 over the same
  chunks) behind the books agent's existing retrieval seam; query unchanged.
- **Dependencies**: RAG-1 baseline; new dependency `rank-bm25` (pure Python —
  the single justified new package; justify again if anything heavier).
- **Tests**: `scripts/test_rag2_hybrid.py` — deterministic: fusion output
  ordering, BM25 hits on exact tokens with a fake in-memory corpus, dense
  path unchanged when lexical finds nothing, no-network guarantee.
- **Acceptance**: exact-term suite (from RAG-1) shows measurable hit-rate
  gain vs recorded baseline; refusal suite unchanged; zero-quota.
- **Why not earlier**: needs the RAG-1 baseline to prove the gain; adding
  retrieval without measurement is unverifiable.

#### RAG-3 — Candidate fusion + reranking (Phase 9, second half)

- **Problem**: two candidate lists must be merged sensibly; top-k by raw
  cosine is noisy.
- **Scope**: deterministic score fusion (e.g., reciprocal rank fusion) +
  optional small local cross-encoder reranker behind a config switch
  (`BOOKS_RERANK=off|local`, default `off` until measured).
- **Dependencies**: RAG-2; reranker model must fit the 4 GB budget (or CPU).
- **Tests**: `scripts/test_rag3_fusion.py` — fusion math unit tests with
  fixed inputs; reranker-off path byte-identical to RAG-2; integration run
  against real index labeled as such.
- **Acceptance**: fusion is pure and deterministic; enabling rerank changes
  neither refusal semantics nor provenance format; documented latency cost.
- **Why not earlier**: fusion without a second candidate source has nothing
  to fuse; rerank without fusion has no stable candidate set.

#### RAG-4 — Source/metadata-aware retrieval

- **Problem**: answers cite `file, page` only when metadata happens to
  survive; there is no notion of document type, date, or authority.
- **Scope**: richer chunk metadata (doc, chapter/page, section headings) at
  ingest time; retrieval filters (e.g., prefer company docs over books);
  provenance carries structured metadata into `sources`.
- **Dependencies**: RAG-2/3; ingest re-run over the same corpus.
- **Tests**: deterministic ingest-metadata unit tests (fake readers);
  provenance-format regression; `check_chroma.py` extended.
- **Acceptance**: citations include chapter/page consistently; existing
  `References:` format remains backward-compatible (Phase 1 live test still
  passes).
- **Why not earlier**: metadata must exist before it can be filtered on;
  requires a corpus worth filtering (grow beyond the two samples first).

#### RAG-5 — Grounding / citation validation

- **Problem**: nothing verifies claims against retrieved context; QA is
  structural only.
- **Scope**: deterministic claim-support *checks where possible* (citation
  presence, quoted span existence in retrieved chunks) + reviewer (Phase 11)
  handles the semantic remainder. Unsupportable claims get flagged in
  `qa_flags`, never silently emitted.
- **Dependencies**: Phase 5 (QA node exists), Phase 11 (reviewer) for the
  semantic half; RAG-4 metadata for precise span checks.
- **Tests**: deterministic: fabricated citation detected, valid citation
  passes, missing citation flags; fake chunks, no LLM.
- **Acceptance**: `qa_flags` gains grounding findings on constructed bad
  answers; good answers stay flag-free; zero quota for structural checks.
- **Why not earlier**: needs a QA node to attach to and stable citations;
  needs the reviewer to split deterministic span checks from semantic
  judgment.

#### RAG-6 — Query decomposition / multi-hop / iterative retrieval (ONLY if needed)

- **Problem (conditional)**: multi-hop questions still fail after RAG-2..5.
- **Scope (if ever)**: bounded decomposition (≤3 sub-queries), bounded hops
  (≤2), all counts hard-capped in config; every hop logged.
- **Dependencies**: RAG-1..5 metrics proving the gap; Phase 12's bounded-retry
  machinery to prevent runaway loops.
- **Tests**: deterministic bounds enforcement (fake LLM); termination proof
  tests (max call counts); labeled integration set.
- **Acceptance**: strictly better scores on the multi-hop subset of the RAG-1
  corpus; per-query LLM-call budget respected.
- **Why not earlier**: complexity and cost are unjustified until simpler
  stages measurably fail. **This step may legitimately never be implemented.**

### 5.2 Planner contract (Phase 7)

> **GUI note (recovery, 2026-10):** the Control Center's Tasks & Sequence
> page and `/executions/{id}/sequence` endpoint currently model executions
> as a **sequential graph route** (created → supervisor → routed → agents →
> qa → completed) and say so explicitly in the UI. Planner-derived
> dependency graphs (the `depends_on` structure below) are intentionally
> absent until this phase lands.

**Input**: `state["question"]`, current `state` (route, prior
`agent_answers` if any), and the static capability catalog.

**Output**: a structured plan — data only:

```text
Plan = {
  "goal": <string restating the request>,
  "tasks": [
    {
      "task_id": "t1",
      "capability": "research" | "books" | "erp" | "synthesis",
      "objective": <string — what this task must produce>,
      "depends_on": ["t0", ...],          # task_ids only
      "required_inputs": ["research", ...],  # keys of agent_answers it needs
      "expected_output": <string — description of the artifact>,
      "done_when": <string — checkable completion condition>
    }, ...
  ],
  "replan_count": 0                         # mutated only by Phase 12
}
```

Rules:

- `capability` must come from a **whitelist defined in code** (the dispatcher
  mapping in `app/graph.py`). Unknown capability ⇒ plan rejected
  deterministically.
- Deterministic validator (pure function, unit-tested): task_ids unique and
  well-formed; `depends_on` references exist and the graph is acyclic;
  `required_inputs` satisfiable from capabilities in the plan; task count ≤
  `MAX_PLAN_TASKS` (config, recommend 6).
- The LLM's plan is parsed defensively (JSON-in-text, provider-neutral); on
  parse/validation failure the planner falls back to a **deterministic
  default plan** derived from the current route — the request still gets
  answered, with a `qa_flags` note that planning degraded.
- **Plan representation vs runtime state**: the plan itself is plan data; it
  becomes runtime state only via a single new `AgentState` key `plan`
  (justified below). Task *results* do NOT get a new structure — they flow
  through the existing `agent_answers` / `sources`. Test traces stay in test
  fixtures.
- The Planner node returns only `{"plan": ...}` (plus nothing else). It
  executes no task.

**Justification for the one new state key** (required by Section 3.6): the
planner node and executor nodes must exchange the plan through `AgentState` —
that is the architecture. No alternative exists without a second state
object, which is forbidden. Key: `plan: Optional[dict]` — absent on
simple/multi routes, so all existing tests remain valid.

### 5.3 Task execution (Phase 8)

```text
planner writes state["plan"]
        │
        ▼
executor dispatcher (graph node):
  pick next task with satisfied depends_on AND not completed
        │
        ├── capability → WHITELISTED node (research|books|erp|synthesizer)
        │        │
        │        └── node writes agent_answers[<capability>] as today
        │
        ├── task done_when satisfied (deterministic check where possible)
        │
        ├── failure → bounded retry hook (Phase 12; before Phase 12: fail
        │             the task and record it, plan continues with rest)
        │
        └── all tasks done → synthesizer → deterministic QA → END
```

- Dispatch is a **literal mapping table in code** — plan entries select
  capabilities, never node names, never callables.
- Dispatch loop caps at `MAX_PLAN_TASKS * (1 + MAX_TASK_RETRIES)` iterations
  (hard guard against planner-induced infinite loops), then fails safe:
  best partial answer + `qa_flags`.
- Existing simple/multi routes bypass the dispatcher entirely (their edges
  are unchanged), so Phase 8 cannot regress Phases 1–4 behavior.
- Planner LLM output never executes Python — only validated capability
  strings influence routing.

### 5.4 Synthesis (Phase 10)

A dedicated Synthesizer is necessary **once Phase 8 can produce N≥2 task
outputs for non-feasibility routes** — today, feasibility is the only
multi-output consumer and the feasibility agent already synthesizes.

Design decision: **evolve `feasibility_agent`'s synthesis core into a shared
synthesizer capability rather than adding a second synthesis agent.**

- Extract the "consume `agent_answers` + `sources`, produce a coherent
  answer with per-input attribution" logic; the feasibility agent keeps its
  Market/Financial/Risk template and `.docx` side-effect as a *profile* of
  the same synthesis step.
- Synthesizer inputs: completed task outputs + provenance already in state.
- Synthesizer MUST: combine evidence, surface contradictions explicitly
  (list both claims + sources rather than silently picking one), preserve
  citations, mark gaps as gaps.
- Synthesizer MUST NOT: perform research itself, call agents, bypass the
  graph, invent evidence (any claim must trace to an `agent_answers` entry
  or a cited URL/chunk).
- It is a node like any other: `def synthesizer(state) -> dict` returning
  only changed keys.

### 5.5 Reviewer vs QA (explicit distinction)

| | Deterministic QA (Phase 5) | Semantic Reviewer (Phase 11) |
|---|---|---|
| Nature | pure function, no LLM | LLM judge via `get_llm()` |
| Checks | structure: answer non-empty; route-required outputs exist (feasibility ⇒ research+books); required provenance tags present; plan completion (post-Phase 8); no contradictory route/state | semantics: internal consistency, answer supported by provided evidence, no fabricated specifics, requirements of the question actually addressed |
| Output | appends to `state["qa_flags"]` | appends to `state["qa_flags"]` (prefix `review:`) + returns verdict `pass`/`fail` + feedback |
| May it route? | no | no |
| May it call agents/tools? | no | no |
| May it modify the answer? | no | no |
| Runs | always, before END | on complex routes (and on demand later); after deterministic QA |
| On fail | flags only (informational) | feeds bounded correction (Phase 12); after cap is exhausted, flags only and ships best answer |
| Tests | zero-quota structural cases (already 30 in Phase 4 suite) | fake LLM with canned verdicts; zero quota; one labeled live smoke |

The reviewer is deliberately **not** a second orchestrator: it produces
verdict + feedback *data*; only the graph acts on it, only within Phase 12's
fixed budgets.

### 5.6 Bounded retries / replanning (Phase 12)

- Budgets (config constants, documented):
  - `MAX_TASK_RETRIES = 2` per task (initial attempt + 2 retries);
  - `MAX_REPLANS = 1` (reviewer-triggered replan happens at most once per
    request); hard iteration guard from Section 5.3 remains.
- Retry policy: identical inputs, fresh `get_llm()` call (transient model
  errors are the primary target); NO hidden prompt mutations between
  retries — if a correction changes the prompt, it comes from explicit
  reviewer feedback and counts against `MAX_REPLANS`.
- Counters live **on the plan structure** (`replan_count`, per-task attempt
  counts) — no new state keys (Section 3.6).
- Exhaustion behavior: keep the best partial result, append a `qa_flags`
  entry (e.g., `retry budget exhausted for task t2`), deliver with the gap
  stated — never hang, never raise, never loop.
- Deterministic tests: fake nodes that fail N times and then succeed (retry
  works); always-fail node (exhausts budget, terminates, flags, answer still
  produced); replan cap (second reviewer FAIL does not trigger a second
  replan).

### 5.7 Real ERP integration gate (Phase 15)

Real ERP integration begins **only when all of the following are explicitly
provided by the user**: vendor identity, API/transport (REST/GraphQL/OData/
other), authentication method, data model/entities, permission model, and a
test/sandbox target. Until then: no vendor SDK, no MCP, no credentials, no
endpoints in code — as already mandated by `docs/PHASE3_ERP_PLAN.md`. The
"Future integration process" in that document is the implementation checklist
for Phase 15; this plan adds only the gate and the test requirements
(adapter contract tests + error-taxonomy tests against a stubbed vendor
client, plus one explicitly-labeled live smoke test).

---

## 6. Phase roadmap overview and dependencies

```text
Phase 5   Graph-integrated deterministic QA  (+ doc truth sync)      ─┐
Phase 6   LLM-assisted supervisor routing    (keyword fallback)      ─┤ foundation,
                                                                     ─┘ sequential
Phase 7   Planner (structured decomposition, conditional)  ◄── needs 6 (complex detection)
Phase 8   Dynamic plan execution in the graph             ◄── needs 7
Phase 9   Hybrid RAG (RAG-2 + RAG-3)                       ◄── needs RAG-1 baseline
                                                            (PARALLEL track with 7–8)
Phase 10  Synthesis evolution (feasibility → synthesizer)  ◄── needs 8
Phase 11  Semantic reviewer                                ◄── needs 10
Phase 12  Bounded retry / replanning                       ◄── needs 8 + 11
Phase 13  Observability / execution tracing                ◄── needs 8 (parallel w/ 10–12)
Phase 14  Memory (evaluate; likely minimal/deferred)       ◄── only if multi-turn is real
Phase 15  Real ERP integration                             ◄── GATED on Section 5.7
Phase 16  Final hardening / evaluation                     ◄── last
```

Ordering rationale: 5–6 are small, independent, and fix known
doc/graph gaps before any complexity is added. 7–8 are the core architectural
addition and must land before anything that consumes multi-task results
(10–12). 9 is deliberately placed as a parallel track because retrieval
quality does not depend on planning, and hybrid RAG benefits every route
immediately. 14 may be skipped entirely; 15 is externally gated; 16 closes.

---

## 7. Phase specifications

Template per phase (identical for every phase):

### Phase 5 — Graph-integrated deterministic QA + documentation truth sync

### Goal
Wire the proven deterministic `run_qa()` checks into the production graph as
a node that populates `state["qa_flags"]`, and bring all stale Phase 4
documentation in line with reality.

### Why this phase exists
`qa_flags` exists in `AgentState` but production code never writes it;
`docs/Plan.md` specifies `app/qa_node.py`; README/PROJECT_STATUS/ARCHITECTURE
claim Phase 4 unimplemented; the supervisor docstring promises an LLM
classifier (delivered in Phase 6). This phase makes the system's behavior and
its documentation agree before any new machinery is added.

### Current prerequisite
Phase 4 complete (`c527ff3`); `run_qa()` behavior fixed by 30 passing tests.

### Components affected
New QA node; graph wiring (all routes pass through it); docs.

### Architecture changes
Every route's terminal hop becomes `… → qa_node → END`. QA is a node but
remains a *gate*: it only appends `qa_flags`. Supervisor/graph/agents
unchanged semantically.

### Files likely affected
- NEW `app/qa_node.py` (`def qa_node(state) -> dict` returning
  `{"qa_flags": [...]}` — merges/extends existing list, never clears it)
- `app/graph.py` (add node; route all four terminal edges through it)
- `scripts/test_phase4_workflow_qa.py` (import QA from `app/qa_node.py`
  instead of defining it locally — single source of truth)
- NEW `scripts/test_phase5_qa_node.py`
- Docs sync: `README.md`, `docs/PROJECT_STATUS.md`,
  `docs/ARCHITECTURE.md`, `docs/TESTING.md`, `app/supervisor.py` docstring
  (promise moves to Phase 6 wording), `.gitignore` (+`env/`)

### New state requirements
None. Uses existing `qa_flags`.

### Tests required
### Deterministic tests

(`scripts/test_phase5_qa_node.py`) QA runs on
every route end-to-end (`graph.invoke` output contains `qa_flags` key);
valid feasibility workflow ⇒ `qa_flags == []`; feasibility missing books
output ⇒ expected flag; erp route does not demand research/books flags;
empty answer ⇒ flag; QA never alters `answer`/`sources`/`agent_answers`;
QA node returns ONLY `qa_flags`.
### Integration tests

none required (behavior is structural); existing
live scripts must still pass untouched.

### Acceptance criteria
1. `graph.invoke(...)` returns `qa_flags` populated for all four routes.
2. All prior suites green unchanged (baseline **101 tests + 10 subtests**),
   with `test_phase4_workflow_qa.py` importing QA from `app/qa_node.py`.
3. New suite green, zero quota.
4. Doc discrepancies listed in Section 1.2 resolved or explicitly re-scoped
   to their owning phase.
5. `env/` ignored; `git diff --check` clean.

### Failure cases
- QA raising ⇒ wrap so a QA failure can never crash a run (QA appends a
  `qa_flags` error note instead).
- Flag ordering instability ⇒ define deterministic flag order in
  `app/qa_node.py` and assert it.

### What this phase explicitly does NOT implement
No LLM anywhere in QA; no reviewer; no routing changes; no new state keys;
no changes to agent contracts; no prompts touched.

### Definition of done
Regression + new suite green; docs describe reality; uncommitted diff
reviewed (per Section 12 process rules).

---

### Phase 6 — LLM-assisted supervisor routing with deterministic fallback

### Goal
Replace keyword-only routing with LLM-assisted classification *behind a
config switch*, keeping keyword routing as the always-available fallback, so
phrasing variations route correctly without changing any downstream contract.

### Why this phase exists
The supervisor docstring and `docs/Plan.md` both promise this upgrade; it is
also the prerequisite for reliably detecting `mode=complex` (planner
trigger). Routing mistakes are currently silent (a mis-routed question still
"answers").

### Current prerequisite
Phase 5 (QA gate exists so mis-routing is at least flagged).

### Components affected
Supervisor only (plus config switch).

### Architecture changes
`supervisor(state)` internally: keyword result first (cheap, deterministic);
an LLM classification (`get_llm(fast=True)`) runs for uncertain cases **or**
when `ROUTING_MODE=llm` — exact policy chosen at implementation time, but the
function's I/O contract (`route`, `sub_routes`) is frozen. Any LLM error,
parse failure, or unknown label ⇒ fall back to keywords. New config:
`ROUTING_MODE = "keyword" | "llm"` (default **`keyword`** — zero behavior
change for existing tests).

### Files likely affected
- `app/supervisor.py`, `app/config.py`
- NEW `scripts/test_phase6_routing_llm.py`
- `docs/TESTING.md`, `.env.example` (placeholder only)

### New state requirements
None. (`mode` if needed is derived inside the supervisor, not stored, unless
Phase 7 documents its own key.)

### Tests required
### Deterministic tests

fake LLM returning each label ⇒ correct route for all
four classes; malformed LLM output ⇒ keyword fallback; LLM exception ⇒
keyword fallback; `ROUTING_MODE=keyword` (default) ⇒ byte-identical
behavior to today for the whole Phase 1 suite; `sub_routes` contract
unchanged for feasibility; LLM route decision recorded (observable via
test double) but NOT stored in state.
### Integration tests

one labeled live run comparing LLM vs keyword routes over
a fixed 10-question set (quota-marked).

### Acceptance criteria
1. Default config ⇒ existing 101 tests pass with zero modifications.
2. New suite proves fallback on error/malformed output.
3. No prompt in any *agent* changed (routing prompt lives in supervisor).
4. Provider independence: suite passes with fake regardless of `LLM_PROVIDER`.

### Failure cases
- LLM returns nonsense ⇒ keyword fallback + no crash.
- Quota exhausted mid-routing ⇒ caught, keyword fallback (mirror
  `GoogleRateLimitError` handling pattern).
- Ambiguous question ⇒ deterministic tie-break order identical to today's
  keyword precedence (feasibility > erp > books > research).

### What this phase explicitly does NOT implement
No planner; no new routes; no state keys; no removal of keyword routing
(it remains the fallback forever); no graph changes.

### Definition of done
Both modes green; default mode unchanged; docs updated.

---

### Phase 7 — Planner: structured task decomposition (conditional)

### Goal
For genuinely complex requests, produce a validated, structured plan of
capability tasks as data — without executing anything.

### Why this phase exists
Complex multi-objective requests ("research X, check our books for Y, then
assess feasibility") currently collapse into one keyword route and silently
skip objectives. A planner makes decomposition explicit and testable.

### Current prerequisite
Phase 6 (reliable detection of when planning is needed).

### Components affected
New planner node; supervisor marks complex requests; graph gets one
conditional hop (complex ⇒ planner).

### Architecture changes
Planner is a graph node implementing the contract in Section 5.2. It returns
`{"plan": ...}` only. Deterministic validator + capability whitelist + default
plan fallback. Simple/multi routes skip it entirely.

### Files likely affected
- NEW `app/planner.py` (planner node + `validate_plan()` pure function)
- `app/state.py` (**one** new key: `plan: Optional[dict]`, documented)
- `app/supervisor.py` (complex detection), `app/config.py`
  (`MAX_PLAN_TASKS`), `app/graph.py` (conditional hop)
- NEW `scripts/test_phase7_planner.py`

### New state requirements
`plan` — JUSTIFIED in Section 5.2: the sole channel between planner node and
executor nodes. Absent for simple/multi routes.

### Tests required
### Deterministic tests

valid plan parsed from canned LLM JSON; malformed JSON ⇒
default plan + `qa_flags` note; unknown capability ⇒ rejected ⇒ default
plan; cyclic `depends_on` ⇒ rejected; missing task_id refs ⇒ rejected;
task-count cap enforced; planner node returns ONLY `plan`; simple route
never invokes planner (test double proves zero calls); plan does not
mutate `answer`/`sources`/`agent_answers`.
### Integration tests

none required (fake LLM); one optional live smoke with
labeled quota.

### Acceptance criteria
1. `validate_plan()` 100% branch coverage via unit tests.
2. Existing 101 tests green unchanged (plan key absent on their routes).
3. Planner LLM output demonstrably cannot introduce a capability outside the
   whitelist (test with adversarial plan JSON).
4. Zero quota in all deterministic tests.

### Failure cases
Covered above: malformed plan, adversarial capability, oversized plan,
cyclic plan, LLM exception — every one falls back to the deterministic
default plan; the request is always answered.

### What this phase explicitly does NOT implement
No task execution (Phase 8); no retries (Phase 12); no plan→Python mapping;
no second state structure; no planning for simple/multi routes.

### Definition of done
Plan contract + validator implemented, tested, documented; regression green.

---

### Phase 8 — Dynamic plan execution in the graph

### Goal
Execute a validated plan through the existing graph: dispatch tasks in
dependency order to the existing agent nodes, propagate results through
`agent_answers`, then synthesize.

### Why this phase exists
A plan that cannot execute is documentation. This is the phase where
"complex" requests actually run multiple capabilities in a controlled order.

### Current prerequisite
Phase 7 (validated `plan` in state).

### Components affected
Graph (dispatcher + task-node reuse), synthesizer entry point, QA (now also
checks plan completion).

### Architecture changes
Dispatcher node per Section 5.3: literal capability→node mapping, dependency
readiness, hard iteration cap, task failure recorded (retries come in
Phase 12; here a failed task is marked failed and the plan continues).
Existing simple/multi edges untouched.

### Files likely affected
- `app/graph.py` (dispatcher node + wiring), `app/state.py` (no new keys —
  task results use `agent_answers`), `app/qa_node.py` (plan-completion check)
- NEW `scripts/test_phase8_execution.py`

### New state requirements
None beyond `plan` (Phase 7). Task attempt bookkeeping lives on the plan
dict (Section 5.6).

### Tests required
### Deterministic tests

3-task plan executes in dependency order (recorded in
fixture); independent tasks order-flexible but both before dependent task
(assert partial order, not incidental order — same discipline as Phase 4);
capability dispatch reaches the correct existing node (fakes); missing
required input ⇒ task marked failed, plan continues, final answer still
produced + flagged; iteration cap triggers fail-safe on a pathological
plan; simple/multi routes never enter dispatcher; no agent-to-agent calls
(existing isolation tests extended to dispatcher path).
### Integration tests

one end-to-end complex request with fake LLMs at every
seam, zero quota; one quota-labeled live run.

### Acceptance criteria
1. New deterministic suite green; isolation tests cover dispatcher path.
2. Existing 101 tests green unchanged (their routes bypass the dispatcher).
3. Fail-safe: no input (valid or adversarial plan) can make the graph loop
   forever or raise — proven by cap tests.
4. Final answer exists for every tested plan outcome; `qa_flags` records any
   failed/skipped task.

### Failure cases
Task node raises (wrapper catches, marks failed); plan references impossible
inputs (validator caught it in Phase 7; dispatcher re-checks defensively);
LLM outage mid-plan (agent fail-soft behavior already exists).

### What this phase explicitly does NOT implement
No retry/replan (Phase 12); no semantic review (Phase 11); no RAG changes;
no parallel execution — sequential dispatch first (parallelism adds
determinism risk for no current benefit; revisit only with measured need).

### Definition of done
Complex route works end-to-end deterministically; regression green; docs
(graph diagram) updated.

---

### Phase 9 — Hybrid RAG (RAG-2 + RAG-3)

### Goal
Add lexical (BM25) retrieval alongside dense retrieval with deterministic
fusion, plus an optional reranker switch — following Sections 4 and 5.1.

### Why this phase exists
Exact-term lookups (IDs, part numbers, rare jargon) are the known failure
class of dense-only RAG; refusal reliability rests on fragile score tuning.

### Current prerequisite
RAG-1 baseline harness must exist (created as Phase 9's first commit: the
harness itself is in-scope "before" work, described in Section 5.1 — if
split out, it is RAG-1 and lands immediately before this phase's code).

### Components affected
Books retrieval internals only; ingest (metadata capture is RAG-4, NOT this
phase); config.

### Architecture changes
Inside `books_agent`'s retrieval seam: dense top-k + BM25 top-k over the same
chunks → reciprocal rank fusion → (optional) rerank → context assembly. The
agent's contract, refusal semantics, and provenance format are unchanged.

### Files likely affected
- `app/agents/books_agent.py`, `app/config.py`
  (`BOOKS_RERANK=off|local`, `BOOKS_FUSION_*`), `requirements.txt`
  (`rank-bm25`)
- NEW `scripts/test_rag1_evaluation.py`, `scripts/test_rag2_hybrid.py`,
  `scripts/test_rag3_fusion.py`
- `docs/TESTING.md` (baseline metrics recorded)

### New state requirements
None.

### Tests required
### Deterministic tests

fusion math with fixed inputs; BM25 hits exact tokens in
a fake in-memory corpus; dense-only degradation path when lexical is empty;
rerank-off path identical to pre-phase behavior; no network (fake
retrievers).
### Integration tests

real-index run over the RAG-1 labeled corpus; hit-rate and
refusal metrics compared against recorded baseline; quota-marked if the
synthesis LLM is exercised.

### Acceptance criteria
1. Measured hit-rate on the exact-term subset strictly improves vs RAG-1
   baseline; refusal suite precision unchanged or better.
2. Deterministic suites zero-quota and green; full regression green.
3. One new dependency total (`rank-bm25`), justified in `docs/ARCHITECTURE.md`.
4. Latency impact measured and documented.

### Failure cases
Empty index (existing not-found path); BM25 over empty corpus (degrades to
dense); reranker model unavailable (config flag falls back to off, logged).

### What this phase explicitly does NOT implement
No query decomposition/multi-hop (RAG-6, conditional); no metadata filtering
(RAG-4); no grounding validation (RAG-5); no changes to research agent's web
pipeline; no new vector DB.

### Definition of done
Metrics recorded, suites green, docs updated.

---

### Phase 10 — Synthesis and evidence aggregation (evolve feasibility)

### Goal
Generalize the feasibility agent's synthesis into a reusable synthesizer
capability so any multi-output plan produces one coherent, attributed,
contradiction-aware answer.

### Why this phase exists
After Phase 8, multi-task answers would otherwise be the last agent's output
overwriting everything else (exactly the problem `agent_answers` solved for
two inputs).

### Current prerequisite
Phase 8 (plans that produce multiple outputs).

### Components affected
Synthesis logic extraction; feasibility agent becomes a profile; QA gains
plan-completion expectations.

### Architecture changes
Per Section 5.4: extract shared synthesis; `feasibility_agent` keeps its
template/`.docx` behavior by delegating to the shared core (its tests must
stay green — behavior-preserving refactor, not a rewrite); synthesizer node
serves the Phase 8 dispatcher's `synthesis` capability.

### Files likely affected
- `app/agents/feasibility_agent.py` (refactor to use shared core),
  NEW `app/agents/synthesizer.py` (or `app/synthesis.py` if not an agent —
  decide by contract: it IS an agent node ⇒ `app/agents/`, same contract)
- `app/graph.py` (wire `synthesis` capability)
- NEW `scripts/test_phase10_synthesis.py`

### New state requirements
None (inputs: `agent_answers` + `sources`; outputs via `with_agent_output`).

### Tests required
### Deterministic tests

synthesis of 2 and N inputs preserves all citations;
contradiction case (research says X, books say Y) ⇒ both surfaced in
answer, not silently merged; missing input ⇒ gap stated; synthesizer never
invents a claim absent from inputs (assert against canned inputs);
synthesizer returns only changed keys; feasibility `.docx` contract tests
(Phase 2 + degradation suites) still pass byte-for-behavior.
### Integration tests

one live feasibility run (quota-labeled) to confirm no
quality regression.

### Acceptance criteria
1. Phase 2 + feasibility-degradation suites green WITHOUT modification
   (proves behavior preservation).
2. New synthesis suite green, zero quota.
3. Contradiction-handling covered by a deterministic test.

### Failure cases
All inputs missing ⇒ synthesizer produces "insufficient inputs" answer +
flag (never a hallucinated study).

### What this phase explicitly does NOT implement
No reviewer; no retries; no changes to research/books/erp internals; no
`.docx` format changes.

### Definition of done
Shared synthesis exists, feasibility behavior preserved, docs updated.

---

### Phase 11 — Semantic reviewer

### Goal
Add an LLM-based reviewer that judges the synthesized result against its
evidence and appends semantic findings + a bounded verdict — strictly after
deterministic QA, with the separation defined in Section 5.5.

### Why this phase exists
Deterministic QA proves structure, not truthfulness. The original spec's
"does the answer cite a source? do outputs contradict?" questions are partly
semantic (contradiction detection) and need a model — but must remain
bounded and non-orchestrating.

### Current prerequisite
Phase 10 (there is a final synthesis to review).

### Components affected
New reviewer node; QA/reviewer ordering; qa_flags vocabulary.

### Architecture changes
`… → synthesizer → qa_node (deterministic) → reviewer (complex routes) → END`.
Reviewer returns `{"qa_flags": [ ...prefixed findings... ]}` plus its verdict
carried for Phase 12 (verdict representation designed at implementation
time, minimal — see "no new keys" discussion below; a reviewer verdict on
`plan` structure or an existing key is preferred over a new state field; if
a key is unavoidable it must be justified in the phase's diff the same way
`plan` was).

### Files likely affected
- NEW `app/agents/reviewer.py` (agent contract, LLM via `get_llm()`)
- `app/graph.py`, `app/config.py` (`REVIEWER_MODE=off|on`, default `off`
  until Phase 12 exists), `app/qa_node.py` (no change to its checks)
- NEW `scripts/test_phase11_reviewer.py`

### New state requirements
Prefer none; if the Phase 12 handoff genuinely needs it, exactly one
justified key — decided and documented during implementation, not before.

### Tests required
### Deterministic tests

fake reviewer LLM returns PASS ⇒ flags unchanged;
returns FAIL ⇒ `review:` findings appended; reviewer output never alters
`answer`/`sources`/`agent_answers`; reviewer never calls agents/tools
(static + runtime double); deterministic QA still runs and is unaffected;
reviewer disabled by default (existing suites prove zero behavior change).
### Integration tests

one labeled live review of a canned research answer with
a deliberately planted contradiction (quota-marked).

### Acceptance criteria
1. Default config ⇒ existing suites green unchanged (`REVIEWER_MODE=off`).
2. Reviewer verdict format provider-neutral (works with fake and real LLM).
3. Zero quota for all deterministic tests.

### Failure cases
Reviewer raises/returns garbage ⇒ caught, treated as `pass` with a
`review:` flag noting review failure (never blocks delivery).

### What this phase explicitly does NOT implement
No correction loop (Phase 12); no reviewer-driven routing; no answer
rewriting by the reviewer; no grounding span checks (RAG-5).

### Definition of done
Reviewer bounded, tested, documented; regression green.

---

### Phase 12 — Bounded correction, retry, and replanning

### Goal
Turn reviewer/task failures into a bounded correction loop with hard budgets
per Section 5.6.

### Why this phase exists
Without it, Phase 11's FAIL verdict has nowhere to go except a flag; with
it, quality failures get exactly one controlled improvement attempt — and
guaranteed termination.

### Current prerequisite
Phases 8 (task retries) and 11 (reviewer feedback).

### Components affected
Dispatcher retry hooks; replan path; qa_flags budget-exhaustion vocabulary.

### Architecture changes
Per Section 5.6: per-task retries, single reviewer-triggered replan, hard
iteration cap, fail-safe exhaustion. Counters live on the `plan` structure.

### Files likely affected
- `app/graph.py`, `app/planner.py` (replan consumes reviewer feedback),
  `app/config.py` (`MAX_TASK_RETRIES`, `MAX_REPLANS`),
  `app/agents/reviewer.py` (feedback format fixed)
- NEW `scripts/test_phase12_retry.py`

### New state requirements
None (counters on `plan`).

### Tests required
### Deterministic tests

transient failure retried then succeeds (exactly N+1
calls observed); permanent failure exhausts retries, plan continues,
flag appended, answer produced; reviewer FAIL triggers exactly one replan
(second FAIL ⇒ flags only, ships answer); caps cannot be exceeded by any
input (property-style loop over adversarial plans/fakes); no sleep/backoff
dependencies (retries are immediate — deterministic timing).
### Integration tests

one live run forcing a reviewer FAIL (quota-marked).

### Acceptance criteria
1. Termination proven: every test path ends in finite steps (caps asserted).
2. Default configs preserve current behavior when retries/replans not
   triggered.
3. All prior suites green; new suite green zero-quota.

### Failure cases
Reviewer always fails (live model disagreement) ⇒ still terminates at
`MAX_REPLANS` with flagged best answer — asserted by test.

### What this phase explicitly does NOT implement
No adaptive/prompt-mutating retries; no exponential backoff (no
transient-network-heavy calls here); no multi-pass self-improvement loops.

### Definition of done
Budgets enforced and tested; docs updated with budget table.

---

### Phase 13 — Observability / execution tracing

### Goal
Make every run inspectable: one structured log line/record per node
execution with coarse metadata, extending the proven `app/erp/log.py`
pattern to the whole workflow.

### Why this phase exists
Phase 8+ runs multi-node plans; without tracing, a wrong dispatch is
debuggable only by re-running. This is production observability (NOT test
scaffolding — test traces stay in fixtures).

### Current prerequisite
Phase 8 (multi-node execution worth tracing).

### Components affected
Graph wrapper around node invocation; a `logs/` JSONL file (git-ignored).

### Architecture changes
A thin node-wrapper in `app/graph.py` records: run id, node name, status,
duration_ms, task_id (plan routes only), error category — **never** question
text, answers, credentials, or record data (same privacy rule as the ERP
log). Fail-soft: logging failure never breaks a run.

### Files likely affected
- `app/graph.py`, NEW `app/tracing.py`, `app/config.py`
  (`TRACE_LOG_PATH`, `TRACE_ENABLED` default on — file writes are cheap and
  local)
- NEW `scripts/test_phase13_observability.py`

### New state requirements
None (traces go to a file, not `AgentState`).

### Tests required
### Deterministic tests

trace lines contain required fields; no question/answer
text or secrets in trace content (adversarial input test); ordering
matches node execution order; disabled mode writes nothing; logging
exception swallowed (run still succeeds).
### Integration tests

none required.

### Acceptance criteria
1. Privacy assertion tests prove no user text in traces.
2. All prior suites green; new suite green zero-quota.
3. Overhead measured (documented; target: negligible vs LLM latency).

### Failure cases
Log path unwritable ⇒ fail-soft; disk full ⇒ fail-soft (asserted).

### What this phase explicitly does NOT implement
No external APM/vendor; no state-key trace fields; no PII capture; no
distributed tracing.

### Definition of done
Trace format documented in `docs/TESTING.md`; tests green.

---

### Phase 14 — Memory (evaluate first; likely minimal or deferred)

### Goal
Determine whether the system needs conversational memory at all; if yes,
add the minimal multi-turn context the CLI genuinely requires.

### Why this phase exists
Listed as a candidate phase, but inspection shows the CLI is stateless
per-invocation by design and no use case has been demonstrated. Memory is
the classic over-engineering trap — this phase may resolve as "documented
decision: not needed".

### Current prerequisite
A concrete, user-stated multi-turn requirement. **If none exists, the phase
resolves to a short decision record in `docs/PROJECT_STATUS.md` and no code.**

### Components affected (if implemented)
`main.py` turn handling; possibly one bounded state key
(`history: List[...]` with a hard cap) — justified only if implemented.

### Architecture changes (if implemented)
CLI keeps a bounded rolling window of prior (question, answer) pairs and
prepends a compact context block to the supervisor input. Agents remain
stateless; memory is a CLI/presentation concern, NOT agent-internal memory.

### Files likely affected (if implemented)
`main.py`, NEW `scripts/test_phase14_memory.py`.

### New state requirements
Prefer none (CLI-level list). Any `AgentState` key requires the full
justification ritual.

### Tests required
### Deterministic tests

window cap enforced; context block built from window
only; agents still receive a single question; forgetting old turns;
zero-state default identical to today.
### Integration tests

two-turn conversation smoke (quota-labeled).

### Acceptance criteria
Without a real requirement: decision recorded, zero code, regression green.
With a requirement: windowed memory works, cap enforced, no agent contract
changes, regression green.

### Failure cases
Oversized history ⇒ capped; malformed stored turn ⇒ dropped.

### What this phase explicitly does NOT implement
No vector memory store; no per-agent memory; no RAG-over-history; no new
infra (Chroma is NOT repurposed for chat memory in this phase).

### Definition of done
Decision documented either way; if built, tests green.

---

### Phase 15 — Real ERP integration (externally gated)

### Goal
Connect the real ERP behind the existing vendor-neutral boundary once — and
only once — the gate in Section 5.7 is satisfied.

### Why this phase exists
Phase 3 deliberately shipped the boundary without the vendor. This phase
consumes that preparation exactly as `docs/PHASE3_ERP_PLAN.md` prescribes.

### Current prerequisite
**External**: user provides vendor, API/transport, auth method, data model,
permissions, sandbox. Nothing in the codebase can proceed without these.

### Components affected
New adapter module; factory switch; tests. **Nothing else** — that is the
point of Phase 3.

### Architecture changes
Add `app/erp/<vendor>_adapter.py` implementing `ERPAdapter` (read-only),
translating vendor errors into the `ERPError` taxonomy; add its branch in
`get_erp_adapter()`; set `ERP_ADAPTER=<vendor>` in `.env`. Agents, graph,
supervisor, state: unchanged.

### Files likely affected
NEW `app/erp/<vendor>_adapter.py`, `app/erp/__init__.py`,
`app/config.py` (non-secret settings only), NEW
`scripts/test_phase15_erp_real.py`, `docs/PHASE3_ERP_PLAN.md` (status),
`requirements.txt` (only if the vendor needs a client lib — justify).

### New state requirements
None.

### Tests required
### Deterministic tests

stubbed vendor transport (fake HTTP/client) covering
every `ERPError` category mapping; record mapping to `ERPRecord` (no
vendor types leak); no secrets in exceptions/logs/answers (adversarial
secret-injection tests, mirroring Phase 3 suite); read-only surface still
enforced; factory selects new adapter when configured, mock remains
default.
### Integration tests

one explicitly-labeled live read against the sandbox
(credentials via `.env` only, never committed).

### Acceptance criteria
1. Full Phase 3 suite (35 tests) still green with mock as default.
2. Zero vendor detail in tracked files (endpoints/credentials/SDK config
   live in `.env` + docs placeholders).
3. Read-only enforcement tests still pass unchanged.

### Failure cases
Vendor down/timeout/401 ⇒ existing fixed safe messages (already tested in
Phase 3 — reuse, don't reinvent).

### What this phase explicitly does NOT implement
No write operations (needs a separate explicit approval per
`PHASE3_ERP_PLAN.md`); no MCP; no second provenance system; no changes to
other agents; no premature SDK sprawl.

### Definition of done
Adapter live-tested in sandbox, all suites green, `PHASE3_ERP_PLAN.md`
status updated.

---

### Phase 16 — Final hardening and evaluation

### Goal
Prove the mature system end-to-end: a maintained evaluation set, adversarial
checks, performance budgets, and documentation that matches the code exactly.

### Why this phase exists
Every prior phase adds seams; this phase verifies the assembled system and
freezes the contract for maintenance.

### Current prerequisite
Phases 5–12 complete (13–15 as scoped); whatever of 14 resolved.

### Components affected
Test/eval infrastructure and docs only — no architectural change expected.

### Architecture changes
None by design. Any architectural defect found here is reported and fixed in
its owning phase, not "hardened around".

### Files likely affected
NEW `scripts/eval_harness.py` (+ eval dataset file), `docs/TESTING.md`,
`docs/PROJECT_STATUS.md`, `README.md`, `docs/ARCHITECTURE.md`,
`docs/MASTER_PLAN.md` (mark phases done).

### New state requirements
None.

### Tests required
### Deterministic tests

full regression (all suites) green in one command;
eval-harness self-tests (metric math, dataset schema) zero-quota;
adversarial prompt corpus routed safely with fake LLMs (no crashes, no
leaked internals); performance budget assertions on orchestration overhead
(node dispatch < budget, no LLM calls on deterministic paths).
### Integration tests

one full live pass over the eval set (quota-budgeted,
documented).

### Acceptance criteria
1. Single documented command runs the entire deterministic regression.
2. Eval set covers: all 4 routes, refusal case, contradiction case,
   multi-objective complex case, degraded-tools case, ERP case.
3. Docs 100% in sync with code (the Section 1.2 class of drift must be
   impossible to reintroduce — add a doc-sync checklist to TESTING.md).
4. Zero uncommitted changes at phase close.

### Failure cases
Eval regressions are routed back to the owning phase's tests.

### What this phase explicitly does NOT implement
No new features; no new dependencies; no "quick fixes" outside phase specs.

### Definition of done
Eval green, docs synced, repository clean, `docs/MASTER_PLAN.md` updated
with completion marks.

---

## 8. How RAG, planning, execution, synthesis, and review evolve together

One-page synthesis of Sections 5–7:

```text
TODAY:    keyword route → 1 agent (or 2-step chain) → answer → [QA in tests]

PHASE 5:  keyword route → agents → answer → deterministic QA → qa_flags → END
PHASE 6:  (LLM-assisted) route, same downstream; keyword fallback forever
PHASE 7:  complex? → validated plan (data only) into state
PHASE 8:  plan → dispatcher → existing agents → outputs in agent_answers
PHASE 9:  (parallel) retrieval: dense → dense+BM25 → fused → (reranked)
PHASE 10: outputs → synthesizer (feasibility evolves into it) → one answer
PHASE 11: answer → deterministic QA → semantic reviewer → verdict + feedback
PHASE 12: feedback/failures → bounded retries & ONE replan → guaranteed end
PHASE 13: everything above emits a privacy-safe trace line per node
PHASE 14: multi-turn only if a real requirement exists (else: documented "no")
PHASE 15: ERP vendor lands behind the existing read-only boundary (gated)
PHASE 16: eval set + doc-sync discipline freeze the system
```

---

## 9. The final system (target end-state)

- **One** `AgentState`, **one** compiled graph, **one** LLM factory, **one**
  config file, **one** provenance mechanism — as today, plus exactly one new
  state key (`plan`) and possibly one more only if Phase 11/14 justify it.
- Supervisor (keyword + LLM-assisted, fallback-protected) → conditional
  planner → graph dispatcher → the same four agent capabilities + synthesizer
  → deterministic QA gate → bounded semantic review → response, with hard
  budgets on retries/replans and a privacy-safe trace per node.
- RAG: hybrid (dense + BM25 + fusion, optional rerank) with measured
  baselines; grounding validation where deterministic; multi-hop only if
  metrics demand it.
- ERP: vendor-neutral read-only boundary, mock by default, real adapter only
  behind the Phase 15 gate; writes still impossible by construction.
- Tests: every behavior above provable at zero quota; live suites explicitly
  labeled and budgeted; a single regression command; evaluation set as the
  quality contract.
- Documentation: README/PROJECT_STATUS/ARCHITECTURE/TESTING and this plan
  continuously in sync — no more drift like Section 1.2.

What the final system deliberately does NOT contain: BaseAgent/inheritance
hierarchy, plugin registry, DI container, MCP (unless the future ERP
explicitly requires it), vendor SDKs before the gate, a second state or
provenance system, a second LLM factory, per-agent memory, autonomous
sub-agents, or any LLM step that can execute code.

---

## 10. Test strategy summary (deterministic vs LLM)

| Layer | Deterministic (zero quota, always in CI) | LLM/integration (labeled, budgeted) |
|---|---|---|
| Routing | keyword matrix; LLM routing with fake + fallback tests | 10-question live route comparison (Phase 6) |
| Planning | validator branches; adversarial plan JSON; default-plan fallback | one canned live plan (Phase 7) |
| Execution | dispatch order (partial-order asserts); caps; isolation | one live complex run (Phase 8) |
| RAG | fusion math; BM25 fake corpus; refusal suites | eval-corpus metrics vs baseline (Phase 9) |
| Synthesis | attribution/contradiction/gap tests with canned inputs | one live feasibility (Phase 10) |
| QA | all structural cases (existing 30 + Phase 5 additions) | none (never needs a model) |
| Reviewer | fake verdicts; flag format; non-interference | one planted-contradiction review (Phase 11) |
| Retry | counters, caps, termination, adversarial loops | one forced-FAIL live (Phase 12) |
| Tracing | field/privacy/order tests | none |
| ERP (real) | stubbed transport, error taxonomy, secret-leak tests | one sandbox read (Phase 15) |
| Whole system | full regression (baseline 101 + additions) | eval-set live pass (Phase 16) |

Baselines and ordering rules live in `docs/TESTING.md`; this table is the
contract for what each future phase must add there.

---

## 11. What must never be added (explicit anti-goals)

Re-asserted here so any phase diff can be checked against one list:

1. `BaseAgent` or any agent inheritance hierarchy — the contract stays
   functional.
2. Agent registry / plugin architecture / dynamic node discovery — wiring
   stays literal in `app/graph.py`.
3. Dependency-injection framework or generic orchestration framework.
4. A second `AgentState`, second provenance system, or trace fields in
   production state for testing.
5. A second LLM factory or provider-specific code inside agents/nodes.
6. Real ERP details (vendor/SDK/API/MCP/credentials) before Phase 15's gate.
7. LLM output that executes code, names nodes, or bypasses the whitelist.
8. QA or reviewer that routes, calls agents, or edits answers.
9. Prompts rewritten to make one provider pass — provider-neutral fixes only.
10. New infrastructure (vector DBs, queues, services) without a written
    justification against the 4 GB/local budget.
11. Weakened or deleted existing tests; quota-consuming tests presented as
    deterministic.
12. Scratch/probe files committed to the repository.

---

## 12. Process rules for future implementers

1. `git status --short` before starting; never disturb unrelated changes.
2. Implement **one phase at a time**, in dependency order (Section 6).
3. Every phase lands with: new tests + full regression green +
   `git diff --check` clean + docs touched only where the phase says.
4. Do not commit until the user reviews; report per Section 7's
   "Definition of done".
5. If an implementation would require violating Section 11, STOP and report
   the conflict — do not improvise architecture.
6. Keep `.env`, `venv/`, `env/`, `storage/`, `logs/`, `reports/` out of
   commits; never hardcode endpoints, keys, or model IPs in tracked files.
7. Update `docs/PROJECT_STATUS.md`, `docs/ARCHITECTURE.md`,
   `docs/TESTING.md`, and this file when — and only when — a phase changes
   what they describe.

---

## 13. Traceability checklist (what each property depends on)

| Property | Proven today? | Owning/next phase |
|---|---|---|
| Routing correctness | yes (Phase 4 suite) | Phase 6 extends it |
| Multi-agent invocation | yes | Phase 8 extends to plan dispatch |
| Dependency/order correctness | yes (feasibility chain) | Phase 8 for plan DAGs |
| State propagation | yes | Phase 8 keeps `agent_answers` as the channel |
| Agent isolation | yes (static + runtime) | every phase must keep it green |
| Failure/degradation behavior | yes (research/books/ERP) | Phase 12 adds bounded budget semantics |
| Final output existence | yes | Phase 5 adds the QA gate to production |
| QA behavior | yes (test-level) | Phase 5 (node), Phase 11 (semantic) |
| Phase 1–3 regression | yes (101 + 10 subtests) | frozen floor for all phases |
| Plan validity/execution | no | Phases 7–8 |
| Hybrid retrieval quality | no | Phase 9 (RAG-1 baseline first) |
| Review/repair loop | no | Phases 11–12 |
| Observability | no (ERP log only) | Phase 13 |
| Real ERP | no (by design) | Phase 15 (gated) |
