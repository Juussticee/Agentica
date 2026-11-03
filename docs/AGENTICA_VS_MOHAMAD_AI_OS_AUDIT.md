# Agentica vs. Mohamad AI Operating System v1.70.0 — Architecture & Overengineering Audit

**Date:** 2026-10-28
**Type:** Read-only audit and recommendation. No source, config, dependency, or data changes were made.
**Status of this document:** report only. Nothing in it is implemented.

---

## 0. Scope, method, and evidence classes

**Subjects**

| | Agentica | Mohamad AI OS |
|---|---|---|
| Location | working tree (branch `main`, HEAD `08d44b6`, 51 uncommitted changes — all pre-existing, untouched) | `docs/mohamad_ai_operating_system_v1_70_0 (1).zip` (17.4 MB, 2,441 entries), inspected **in-place via `zipfile` — never extracted** |
| Stated identity | README: *"Agentica … local-first, multi-agent assistant for business questions"* | README: *"Mohamad AI Operating System v1.72.0 — Integrated Product Edition"*; `product_core.py`: `VERSION = "1.72.0"`. The archive name says v1.70.0; `README_v1_70_0.md` describes the v1.70 line. **Version label mismatch: archive name ≠ code version (v1.72.0).** |

Note on the earlier confusion: `docs/Agentic_System-main.zip` is a same-day snapshot of **Agentica itself** (188 files, 150 byte-identical to the current tree, 0 files unique to it), not a second system. The real comparison target is the `mohamad_…` archive.

**Evidence classes used below**

* **[V] Verified** — read directly from source/DB today, or reproduced by running an offline test.
* **[A] Architected/inferred** — follows from structure, not executed.
* **[U] Unverified** — cannot be settled offline with the endpoint unavailable (no live execution, no extraction/execution of the other system).

**Environment constraint honored:** no live agent execution, no endpoint start/repair, no network, no credentials, no state-changing tests. The only writes performed were this document.

**Reproduced today [V]** (commands run in this session, exit code 0):

```
pytest scripts/test_conversational_routing.py -q      → 12 passed, 20 subtests
pytest phase1_routing + phase5_qa + phase4_workflow + agent_architecture → 73 passed
python .diag/w2_network_audit.py                      → 717 passed, 2 skipped,
                                                        164 subtests, external attempts = 0
```

The reported figures (12 / 73 / 717-with-2-skipped-164-subtests) are therefore **reproduced, not taken on trust**. They are offline tests only and say nothing about live GUI or external execution.

---

## A. Executive verdict

**Simpler, by area**

| Area | Simpler system | Basis |
|---|---|---|
| Runtime decision path (per request) | **Mohamad** [V] | 0 LLM calls in the decision path; deterministic `classify_intent` → plan → select → run. Agentica: 1–4 LLM calls per answer. |
| Code that actually runs | **Agentica** [V] | ~100% of `app/` is reachable from `main`/`server` (34 modules, 6,731 LOC; only `test_dummy_agent.py` is test-only) vs **47/130 modules reachable (41%)** in Mohamad after including registry-loaded agents. |
| Permission / safety model | **Mohamad** [V] | One-shot, mission-scoped, non-persisted, deny-by-default approvals (`ALLOWED_ONE_SHOT_SERVICES = {web_search, erp_read}`; unknown approval → `PermissionError`). Agentica has **no approval concept at all** — any unmatched input can trigger a live web search. |
| Persistence | **Agentica** [V] | 1 SQLite DB, 2 tables. Mohamad: 1 SQLite + ≥5 integrity-hashed JSON state files + ≥3 `audit.jsonl`, each with `fsync` + atomic-replace ceremony replicated in four engines. |
| Test evidence | **Agentica** [V] | 33 test files, 717 tests, network-guarded, frozen gold datasets, eval harness. Mohamad: 18 ad-hoc `selftest_*` scripts (963 LOC), **zero `pytest`/`unittest` imports, 1 `def test_` in the whole project**. |
| RAG depth | **Agentica** [V] | Chroma + embeddings + top-k + grounding/polarity/coverage gates + per-claim citations with page numbers + abstention + bounded recovery. Mohamad: JSON chunk index, toy lexical scorer, and a "rerank" layer whose constants make it a **mathematical no-op** (see §D). |
| Repository hygiene | **Agentica** [V] | No vendored interpreter. The Mohamad archive ships `.venv/` (1,914 files) and `__pycache__/` (148 files) — 86% of its own archive. |

**Stronger evidence of correctness:** Agentica, by a wide margin — automated, offline, network-guarded tests plus frozen evaluation datasets versus self-reported `TEST_REPORT_*.md` files. But see §A.3: test volume did not prevent a production intent bug.

### A.1 Where Agentica's complexity is justified

1. **Grounding, abstention and citation integrity** (`app/agents/books_agent.py`) — the product's core promise is "answer from my documents with provenance and refuse when the evidence is not there." The gates are exercised by dedicated suites (`test_books_grounding*`, `test_citation_integrity`, `test_self_refusal_recovery`, `test_knowledge_fallback`).
2. **`app/chroma_guard.py` (188 LOC)** — fail-closed protection of the production vector store during tests/evals. A real failure mode was hit historically; this is a correctness boundary, not decoration.
3. **Execution events + provenance (`app/execution/*`, W2)** — recorded as operations happen (ContextVar-isolated sink), fail-soft, privacy-filtered. Needed for any future JSON export and for debugging.
4. **The network guard itself** (`.diag/w2_network_audit.py`) — proves the suite makes zero external calls. Neither system has an equivalent check for the other.
5. **QA gate (`app/qa_node.py`)** — 105 LOC, deterministic, runs once per execution. Cheap for what it catches (it just caught the greeting/wrong-answer class of bug).

### A.2 Where Agentica is probably overengineered [V unless noted]

| # | Item | Evidence | Verdict |
|---|---|---|---|
| O1 | **Planner + dispatcher + plan-QA stack (Phase 7/8)** — `app/planner.py` (437 LOC), dispatcher loop in `app/graph.py` (~200 LOC), plan checks in `qa_node.py`, GUI plan rendering, ~975 LOC of tests (`test_phase7_planner.py` 369, `test_phase8_execution.py` 606) | Read-only DB: **1 of 67 executions has a plan** (`plan IS NOT NULL`); routes: books 54, research 7, erp 3, feasibility 2, null 1 | **Proven underuse.** Three layers of machinery for a request class that has occurred once. Keep, but freeze. |
| O2 | **`app/verifier.py` (355 LOC)** reachable from `books_agent` but `BOOKS_VERIFIER = "off"` by default | `app/config.py:157` | **Off-by-default parallel implementation** of "is this claim supported", alongside the always-on grounding gates. Cost is code surface and reader confusion, not runtime. |
| O3 | **Three diagnostics surfaces** — `services_health.py` (386) + `ai_diagnostics.py` (255) + `ai_settings.py` (125) = 766 LOC of provider probing/status, with overlapping "is the endpoint up / is the model there" logic exposed as `/api/v1/services`, `/api/v1/ai/test`, `/api/v1/health` | server routes `app/api/server.py:100,264,336` | **Proven duplication** — two independent sources of truth for provider status. Structural risk: they can disagree. |
| O4 | **`books_agent.py` = 1,349 LOC = 20% of `app/`** — retrieval, prompt build, polarity, numeric, coverage, citation, refusal diagnosis, recovery, fallback diagnosis in one file | `wc -l app/agents/books_agent.py` | **Concentration risk**, not runtime waste. Maintainability only. |
| O5 | **Route keyword vocabularies defined twice** — `app/supervisor.py` regexes vs `app/planner.py:_ROUTING_HINTS/_COMPLETENESS_SIGNALS` | both files | **Proven drift risk**: a concept added to routing can be missed by the planner (or vice versa). |
| O6 | **Infrastructure ahead of intent reliability** [V, historical] | The greeting defect shipped while Phase 7/8, W2 events, and 717 tests already existed: execution `3fc3bc8460c9` ("hello") routed to research and returned `hellosearch.io` / `google.com/xhtml/search` / `hallo.zone` as "sources". Fixed 2026-10-27. | **Proven**: test volume was aimed at structure, not representative intent behavior. |
| O7 | **Stale execution lifecycle** [V] | DB row `a7bd96990f0b` is still `status='running'` from 2026-10-25T17:11 | **Proven gap**: no reconciliation of interrupted runs (small, real). |

### A.3 What cannot be established while the endpoint is offline

* Any claim about **answer quality, synthesis success rate, recovery benefit, latency, or token cost** — [U].
* Whether `test_full_feasibility_workflow`'s one-off failure is a genuine ordering race or an artifact of my earlier combined run — [U] (documented in §E.3, not dismissed).
* Everything about Mohamad's *runtime* behavior: its selftests were **not executed** (running them would extract files into the repository and create `runtime/` state, both forbidden). All Mohamad findings are static.

---

## B. Side-by-side architecture comparison

### B.1 Agentica — actual request path [V]

```
POST /api/v1/executions  (FastAPI, app/api/server.py)
  → ExecutionStore.insert (SQLite: executions + events)
  → GraphRunner.run (app/execution/runner.py)
      → build_graph() (app/graph.py)   [LangGraph StateGraph]
      → supervisor   : deterministic keyword route (ROUTING_MODE=keyword default;
                       greeting → "conversational"; optional LLM mode = +1 call)
      → [planner]    : only if PLANNER_MODE/complex heuristic fires (1/67 observed)
      → [dispatcher] : only for plans (1/67)
      → agent node   : books | research | erp | feasibility | conversational
      → books→research fallback edge (1 transition max, decision recorded)
      → qa_node      : deterministic flags (provenance, emptiness, plan, greeting)
  → store result + events
```

* **LLM calls per request [V]:** greeting 0 · books 1 (+1 bounded recovery; verifier off) · research 2 (query rewrite + synthesis) · `ROUTING_MODE=llm` +1 · complex plan +1.
* **Agent handoffs:** none — agents never talk to each other; the graph routes state. Only one conditional hop exists (books → research) and it is structurally one-way.
* **Memory:** **none**. No conversation history anywhere in `app/`; `app/supervisor.py:198` says so to the user. Persistence = executions/events (SQLite) + vector store (Chroma) + `storage/ai_settings.json`.
* **Tools:** SearXNG search, page crawl/extract, Chroma retrieval, ERP mock adapter, docx report writer.
* **Safety:** `chroma_guard` (production-store write guard), network guard in tests, fail-soft conventions. **No user approval gate for external search.**

### B.2 Mohamad AI OS — actual mission path [V, static]

```
run_command_center.py → CommandCenterV6 (stdlib http.server, single-file RTL UI, CSRF token)
  → ProductCore.process (product_core.py)
      → MasterCognitivePlanningEngine.plan
            classify_intent (adaptive_planner.py — term lists, no LLM)
            → requirements → DAG → gaps → evidence_plan  (state.json + audit.jsonl, sha256+fsync)
      → IntelligentAgentSelectionEngine.select  (capability ∩ trust/load scoring,
                                                  state.json + audit.jsonl)
      → AutonomousMissionControlEngine.run loop (checkpoints, replan ≤ 2 → ESCALATE)
      → per node: ProductCore._executor → UnifiedAgentRuntime.execute
            registry.load(agent) via importlib → builtin adapter → agent.handle()
            (state.json + audit.jsonl, sha256+fsync)
      → EvidenceAwareDecisionSynthesisEngine (FACT/EVIDENCE/ASSUMPTION/INFERENCE,
            conflict detection, coverage, confidence)  (state.json + audit.jsonl)
      → decision gate GREEN/YELLOW/RED → ProductStore.record_mission (SQLite)
      → LocalLLMAdvisory (Ollama, POST-decision, non-blocking, off by default)
```

* **LLM calls in the decision path [V]:** **0.** `ProductCore._local_llm_inference` literally `return None` ("Deprecated in v1.71.0"); advisory is submitted *after* the decision is persisted and is disabled by default (`enrich_agent_results=false`, README_v1_70_0).
* **Intent interpretation [V]:** pure term lists in `adaptive_planner.classify_intent` — research/production/feasibility/financial/personal domains. **No greeting handling** (no greeting term appears anywhere in the classifier).
* **RAG in the mission path [V]:** **none.** No specialist agent calls `rag_read`; `KnowledgeManager`/`IntelligentRAG` are used by the UI knowledge panel, `product_core.status()`, and selftests only.
* **Memory [V]:** no conversation history. Per-agent JSON notes (`knowledge/work.json`, `knowledge/shared.json`), mission history in SQLite, plan/selection/run state in JSON files.
* **Tools/services [V]:** `rag_read` (read-only), `web_search` (approval-gated), `erp_read` (approval-gated, read-only), `ollama_local`, `secure_tool_gateway` (token-based, unreachable path).
* **Safety [V]:** approvals are one-shot, mission-scoped, never persisted, destroyed at mission completion; unknown approval → `PermissionError`; external actions structurally off; local-only bind enforced (`PermissionError` if host not loopback); CSP/X-Frame-Options/nosniff headers.

### B.3 Overlapping responsibilities — who does what

| Responsibility | Agentica | Mohamad |
|---|---|---|
| Intent/routing | `supervisor` (regex, optional LLM) + `planner` complexity heuristic | `adaptive_planner.classify_intent` (term lists) |
| Planning | `planner.py` LLM plan + validation | `MasterCognitivePlanningEngine` deterministic DAG |
| Delegation | graph `_ROUTES` table (literal, code-defined) | `IntelligentAgentSelectionEngine` (capability fit + trust + load) |
| Execution | LangGraph nodes + Phase-8 dispatcher | `UnifiedAgentRuntime` + adapters |
| QA/verification | `qa_node` + books grounding + optional `verifier` | synthesis engine (conflicts/coverage/confidence) + specialist `decision_gate` |
| Recovery | bounded regeneration + books→research fallback | replan ≤ 2 → ESCALATE |
| Observability | SQLite `events` + ContextVar `emit` sink + logs | `audit.jsonl` ×N + `state.json` integrity hashes |
| Health/status | `services_health` + `ai_diagnostics` + `/health` (**overlapping**) | `LiveProviderValidator` + `product_doctor` |

Both systems have **the same eight responsibilities**. The difference is where the weight sits: Agentica spreads them over a 4-node graph with per-node gates; Mohamad spreads them over ~6 named "engines" plus 59% unreachable code.

---

## C. Ranked simplification opportunities (Agentica)

Rank = impact × confidence ÷ risk.

| # | Opportunity | Category | Impact | Risk | Confidence | Effort | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | **Consolidate provider/readiness probing** behind one internal function used by `/health`, `/services`, `/ai/test` (keep the three endpoints, share one core) | Simplify/consolidate | High (removes divergent-truth class of bugs) | Low | High | **S–M** | O3: 766 LOC, 3 overlapping surfaces |
| 2 | **Freeze the planner/dispatcher stack** — no new features, no new plan fields; decide its default only after real usage data | Defer until usage justifies | High (stops investment in the least-used path) | Low | High | **S** | 1/67 executions carried a plan |
| 3 | **Split `books_agent.py` into retrieval / grounding / recovery modules with byte-identical behavior** (move-only refactor + existing suites as the oracle) | Simplify | Medium (maintainability of the 20%-of-app file) | Medium | High | **M** | 1,349 LOC single file |
| 4 | **Deduplicate route keyword vocabularies** into one module consumed by both `supervisor` and `planner` | Simplify | Medium | Low | High | **S** | O5 |
| 5 | **Quarantine `verifier.py`**: document it as experimental with an explicit decision ("enable with measurement" or "remove") instead of reachable-but-off | Investigate further | Medium | Low | Medium | **S** | O2 |
| 6 | **Reconcile stale executions** (interrupted runs never reach a terminal status) | Simplify/fix | Medium | Low | High | **S** | DB row `a7bd96990f0b` |
| 7 | **Adopt a one-shot approval gate for external research** (borrow Mohamad's model verbatim: mission-scoped, non-persisted, deny-by-default) | Do NOT remove — **add** (safety boundary) | High | Medium | High | **M** | Agentica searches the web with no consent concept; Mohamad's `runtime_adapters.HandleAgentAdapter` gates it |
| 8 | Reduce structural test weight in favor of a handful of representative end-to-end intent cases | Investigate further | Medium | Low | Medium | **M** | 717 tests, yet the greeting bug shipped |
| 9 | Keep `chroma_guard`, grounding gates, citations, `qa_node`, execution events, network guard **unchanged** | Keep unchanged | — | — | High | — | correctness/safety boundaries |
| 10 | Do not delete planner/dispatcher/verifier outright | Do not remove | — | — | — | — | responsibilities are legitimate; only the *timing* of the investment is wrong |

**What Agentica should NOT copy from Mohamad:** the 59%-dead module set, per-engine integrity-hash ceremony, no-test-framework selftests, or the vendored interpreter.

**What Agentica SHOULD copy from Mohamad:** the approval boundary (item 7), and the discipline of a single deterministic intent classifier with **zero LLM calls on the routing path** (Agentica already has this by default — it must not be silently turned on to "llm").

---

## D. Proposed minimal architecture

Smallest practical shape that keeps RAG, memory, tool execution, permissions, and verification — **incremental, no rewrite**:

```
UI / CLI
   └─ API (FastAPI) ── SQLite: executions + events   ← keep as-is (single store)
        └─ Graph (4 nodes, unchanged today):
             supervisor ──► agent ──► qa ──► END
             (keyword route, greeting route, 1 one-way fallback edge)
        └─ planner/dispatcher: FROZEN behind PLANNER_MODE, default unchanged
        └─ books agent (split into 3 modules, behavior identical)
        └─ research agent + approval gate (new, one-shot, mission-scoped)
        └─ ONE health/probe core → three thin endpoints
```

Deliberately **not** proposed: a rewrite, an engine/broker/fabric layer, integrity-hashed JSON state, a registry, or LLM-based routing. Agentica's current 4-node path is already close to minimal; the excess is in *unused branches and duplicated diagnostics*, not in the shape of the path.

---

## E. Validation plan

### E.1 Runnable now, offline, endpoint-free

1. The three suites reproduced above (exit 0, external attempts = 0).
2. New: a **reachability check** as a test (the analysis in §A/§C) — flags modules drifting out of the active path.
3. New: **DB invariant check** (read-only): no execution left in `running` past a bound; plan-usage counter reported.
4. Existing eval harnesses (`scripts/eval_retrieval.py`, `scripts/eval_answers.py`) against frozen gold datasets — no re-ingestion.

### E.2 Requires the endpoint / live execution (defer)

* Answer quality and false-refusal rate on the gold sets with the real serving model.
* Whether bounded recovery and books→research fallback measurably improve completion (today's claim rests on docs, not live metrics).
* Latency and token cost per route (instrumentation needed: spans around `get_llm().invoke` — none exist today).
* Approval-gate UX for external research.

### E.3 Known anomaly to carry forward (documented, not dismissed)

`scripts/test_phase4_workflow_qa.py::TestEndToEndFeasibility::test_full_feasibility_workflow` failed **once** in a combined run (recorded `calls == ['research', 'feasibility']`, i.e. the fake books node was not observed), and passed when run standalone and on an identical rerun. Not attributable to the greeting change from source inspection, but **not proven benign either** — it must be re-checked in any future combined run before being called flaky.

---

## F. First three recommended changes

1. **One probe core for provider/readiness (S–M, low risk, highest confidence).**
   `services_health`, `ai_diagnostics`, and `/health` currently answer the same question three different ways. One shared probing function, three thin serializers. This removes the only *operationally dangerous* duplication found (status surfaces disagreeing), and it is testable entirely offline. It precedes everything else because every other decision in this document depends on trusting the status surfaces.

2. **Freeze the planner/dispatcher/QA-plan stack and stop extending it (S, zero-risk).**
   Three layers plus ~975 test lines exist for a path used by **1 of 67** recorded executions. The responsibility is legitimate and should not be deleted; the correct move is to stop spending here and to re-open the default-setting question only after real usage data exists. It precedes other work because it redirects effort from the least-exercised code to the intent/routing layer that actually shipped a bug.

3. **Decompose `books_agent.py` + centralize route keywords (M, behavior-identical).**
   A 1,349-LOC file holds retrieval, grounding, citation, refusal diagnosis and recovery; route vocabulary is written twice (`supervisor.py`, `planner.py`). A move-only split with the existing grounding/citation/fallback suites as the oracle reduces the single largest maintenance risk without touching semantics. It precedes new features because every future RAG change lands in that file.

*Adjacent safety recommendation (not a simplification, scheduled after the three):* adopt Mohamad's one-shot, mission-scoped, deny-by-default approval for outbound web research. Agentica currently performs live external searches with no consent concept — the single clearest lesson the other system teaches.

---

## Appendix 1 — Measured counts

**Agentica [V]**
- `app/`: 34 modules, 6,731 LOC; ~100% reachable from `main`/`app.api.server` (only `agents/test_dummy_agent.py` is test-only).
- Tests: 33 files, 717 tests + 164 subtests, 2 skipped; network guard: 0 external attempts.
- DB (`storage/control_center.sqlite3`, read-only): 67 executions; routes books 54 / research 7 / erp 3 / feasibility 2 / null 1; **plans 1/67**; tables `executions`, `events`; max events/execution 13; 1 stale `running` row.
- Decision stages per simple request: 3 (supervisor → agent → qa); 5 with a plan.
- LLM calls: 0 (greeting) / 1 (books) / 2 (research) / +1 (llm-routing) / +1 (plan).

**Mohamad AI OS [V, static]**
- Zip: 2,441 files = 1,914 `.venv` + 148 `__pycache__` + 365 project files.
- Project Python: 130 modules, 13,808 LOC (root: 123 files, 13,403 LOC).
- **Reachable from entry points + registry agents: 47 modules / 5,654 LOC (41%). Unreachable: 83 modules / 8,154 LOC (59%)** — incl. `legacy_main` (965), `master` (439), `orchestrator` (220), `agent_supervisor` (168), `agent_dispatch_engine` (112), `command_center` v2–v5, and every `*_engine/fabric/broker/gateway`-named module not wired into `product_core`.
- Docs: 70 README/TEST_REPORT files spanning v1.15 → v1.72 (the archive label says v1.70.0).
- Tests: 18 selftest scripts, 963 LOC; 0 `pytest`/`unittest` imports; 1 `def test_`.
- Decision stages per mission: ~8–9 named layers; LLM calls: 0 (advisory off by default, post-decision).
- Persistence: 1 SQLite + ≥5 hashed JSON state files + ≥3 `audit.jsonl`.

## Appendix 2 — Ten realistic request scenarios (source-traced)

| # | Scenario | Agentica | Mohamad AI OS |
|---|---|---|---|
| 1 | `"hello"` | **Handled** [V]: exact full-string opener match → `conversational` route → fixed reply, 0 LLM, 0 network; QA flags any sourced answer to a greeting (`test_conversational_routing`, 12 passed). Pre-fix behavior documented in DB exec `3fc3bc8460c9`. | **No greeting path** [V]: `classify_intent` has no greeting terms → domain `general` → single `work_agent` node → deterministic number-extraction agent. Actual output shape [U] (not executed). |
| 2 | `"hello, what does the NIST AI RMF say about …"` | **Handled** [V]: not a greeting (full-string match) → `books` route; test asserts route, no QA flags | Term matching still applies, but a general question has no domain signal → `work_agent` [V]; document lookup never happens (no agent consults RAG) |
| 3 | Question about an uploaded document | **Core path** [V]: Chroma retrieval → synthesis → grounding → page citations; 54/67 recorded executions | **Not in the mission path** [V]: RAG exists only as the UI knowledge panel / `rag_read` service that no agent calls |
| 4 | Needs previous conversation context | **Unsupported** [V] — no history; the reply states it explicitly | **Unsupported** [V] — no conversation history; per-agent JSON notes only |
| 5 | Current-information research | **Automatic** [V]: `_CURRENT_INFO_RE` → rewrite LLM → SearXNG → crawl 3 pages → synthesis | **Approval-gated** [V]: research agent returns `SERVICE_APPROVAL_REQUIRED` unless the user ticks `web_search` for that one mission; then search (SearXNG→DDG) + deterministic extraction |
| 6 | Request needing one tool | ERP route with **mock adapter** [V] (3/67 executions) — canned data, honest about being a mock | `erp_read` adapter, read-only, approval-gated, `PermissionError` otherwise [V] |
| 7 | Several dependent steps | Planner (only if ≥12 words and ≥2 completeness signals) → validated plan → dispatcher loop → plan QA [V]; **used 1/67** | Always builds a DAG from intent; nodes run in dependency order; replan ≤ 2 [V] |
| 8 | Ambiguous request | **Single-shot**: routes deterministically and answers or abstains — no clarification turn [V] | **Single-shot**: records `knowledge_gaps` and agents emit `missing_sections`, but gaps never block the plan status [V] — surfaces gaps instead of asking |
| 9 | Task fails halfway | Agent nodes fail-soft; dispatcher marks failed/blocked/unsupported; QA flags plan completion; runner records the error [V] | Runtime returns `FAILED` with risk entry; mission control replans ≤ 2 then `ESCALATE`/`BLOCKED`; checkpoints persisted [V] |
| 10 | Verification before reporting success | `qa_node` + always-on grounding/citation gates + optional verifier (off) [V] | Deterministic `verification.passed` gate inside specialists + conflict/coverage/confidence in the synthesis engine + GREEN/YELLOW/RED decision gate [V] |

**Reading of the table:** neither system asks clarifying questions and neither has conversation memory. Mohamad is stricter about *permission to act*; Agentica is stricter about *whether the answer is supported by evidence*.

## Appendix 3 — Findings about the comparison target itself (for the record)

* The archive contains a `mohamad_ai_operating_system_v1_70_1_HOTFIX/` copy of 18 files, four `command_center` generations, `legacy_main.py`, `master.py`, and 13 `selftest_v1_*` files — i.e., **historical versions shipped alongside the current code**, all unreachable.
* `intelligent_rag.py`'s reranking computes `rank = score*0.75 + freshness*0.10 + source_quality*0.15` with `freshness = source_quality = 1.0` constants → `rank = 0.75*score + 0.25`, a strictly monotonic transform: **ordering is identical to the base score; the reranker has no effect** [V by arithmetic].
* `research_pipeline` assigns confidence from a hard-coded `PRIMARY_DOMAINS` allowlist (includes `google.com`, `github.com`, `openai.com`) and flags "conflicts" whenever any number appears in ≥1 source — a deliberately conservative, non-LLM heuristic.
* The archive ships `.venv/` and `__pycache__/`, and its README/version label (`v1.70.0`) disagrees with the code (`v1.72.0`).
