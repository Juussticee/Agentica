# GUI RAG Fix Validation — 2026-10-25 17:02 (Controlled Fix Cycle)

Validation of the three authorized fixes from
`docs/GUI_RAG_FIX_VALIDATION_2026-10-25_1415.md` — false refusals (Phase 1),
natural-language routing (Phase 2), citation noise (Phase 3) — with automated
validation (Phase 4) and controlled GUI verification (Phase 5). Local Ollama
`qwen3:0.6b` only; no paid/hosted models; **zero external/SearXNG requests in
this cycle** (`[research_agent]` lines in the Phase-5 backend log: 0).

**Overall verdict: PARTIAL (majority of acceptance criteria met).**
Routing: PASS (16/16 GUI runs routed books, including A2/A3/T6). Abstention:
PASS (all four unanswerable/insufficient-evidence questions still refuse, now
with proof-grade diagnostics). Citations: PASS on every answered run + the
noise filter pinned by tests (no post-fix GUI footer contained a noise
citation — every post-fix B2 output was a refusal, so the answered-path footer
was not re-observable in the GUI this cycle). False refusals: **3 of 7 flipped
to real answers (T2, T4, A5)**, 1 eligibility mechanism fixed (A6 recovery now
runs), and every remaining refusal is now precisely diagnosed by new
gate-drop diagnostics — but 5 refusals remain.

The three fixes are kept independently identifiable below so a routing success
cannot conceal a grounding or citation result.

---

## 1. Confirmed root causes (all empirically reproduced before editing)

### Issue 1 — False refusals (three distinct causes)

- **FR-1, grounding gate stop-list gaps.** `_grounded_repair` requires every
  content token of every sentence to appear in the retrieved passages.
  Non-assertive meta/reporting scaffolding was counted as required content.
  Offline reproduction on the real A5 output (score 0.736, rank-1 chunk =
  the verbatim answer): the fully evidenced sentence *"It emphasizes that
  GOVERN is a cross-cutting function integrates compliance and evaluation..."*
  was dropped on the single token `emphasiz`; `like` and `content` killed two
  more evidenced sentences. Nothing substantiated survived -> mandated
  refusal despite rank-1 verbatim evidence.
- **FR-2, M5 coverage-gate arithmetic.** `_question_covered_by_evidence`
  counted interrogatives and task verbs ("how", "what", "summarize") that
  evidence never echoes: A6's coverage was 0.45 (blocked) with exactly those
  words among the missing tokens, so the one bounded retry never ran for an
  answerable question.
- **FR-3, recovery prompt missing the partial clause.** The M5 retry prompt
  lacked the main QA prompt's *"state ONLY the covered facts... do not claim
  the whole topic is absent"* clause, so a weak model could only choose
  between full answer and NOT_FOUND: A1/A4 eligible retries re-refused twice
  (observed live and offline).

### Issue 2 — Natural-language routing

- **RN-1.** `_BOOKS_RE` only matched deictic/lexical triggers; the corpus's
  own identity terms (AI Risk Management Framework / AI RMF / RMF /
  trustworthy AI / NIST) were absent, so A2/A3 (and the natural T6) fell
  through to research. There was also no current-information precedence, so
  any corpus anchor would have hijacked "latest news about NIST..." queries.

### Issue 3 — Irrelevant citation leakage

- **CN-1.** `_covering_nodes` attributed a claim via literal token-subset of
  a node, else any node sharing >=1 token (intersection fallback). For the
  B2 sentence, "thrust"/"changes" were group-covered by the orbital chunk's
  "rocket"/"delta", so no node matched literally, and the NIST disclaimer
  chunk joined the citation purely because its text contains `chang`
  (reproduced offline: claim nodes = [orbital, NIST page i]; footer printed
  `orbital_kinetics_principles.docx; NIST.AI.100-1.pdf, page i`).

## 2. Exact changed files and rationale

| File | Change (fix id) | Rationale |
|---|---|---|
| `app/agents/books_agent.py` | **FR-1**: added non-assertive forms to `_STOP_TOKENS` (`like content emphasizes emphasize emphasized emphasizing summarize* summary`, interrogatives/demonstratives `what how why when where which who whom whose this these those`). Comment block moved OUTSIDE the triple-quoted string (an in-string comment previously turned comment words — including `not` — into stop words, silently weakening the gate; caught by a polarity regression test and reverted). | Symmetric normalization: only non-assertive requirement tokens stop being required; numbers, content words and polarity gates untouched. |
| same | **FR-3**: `_RECOVERY_PROMPT` now carries the main prompt's partial-coverage clause (byte-identical sentence). | A bounded retry must be able to answer partially instead of choosing full-answer-or-NOT_FOUND. |
| same | **FR-1 (observability, req 6)**: `_grounded_repair(..., diagnostics=[])` + `[books_agent] gate_drop:` lines printed at the refusal stage, computed at decision time from a caller-owned list (never an overwritten variable). | Requirement: refusal diagnostics must identify the actual stage and reason. |
| same | **CN-1**: `_covering_nodes` is now two-tier — tier 1: one node covers EVERY sentence token literally or via that token's paraphrase-group sibling **in this node**; tier 2 (splices): contributing node must cover >= half the tokens. | Cite only a source that actually supports the claim; a one-incident-token noise chunk can never be cited; genuine multi-passage answers keep both citations (tested). |
| `app/supervisor.py` | **RN-1**: `_BOOKS_IDENTITY_RE` (risk management framework / ai rmf / rmf / trustworthy ai / nist) checked with `_BOOKS_RE`; new `_CURRENT_INFO_RE` (latest/breaking/today/this week/this month/overnight/news/weather/stock price/exchange rate) checked AFTER feasibility+ERP but BEFORE books anchors. Docstring updated. | General corpus-identity anchors, no exact-question rules, no bare "AI"/"framework"; current-information queries keep research even when they name a corpus document. |
| `scripts/test_phase1_routing.py` | +13 routing cases (A2/A3 exact, 4 paraphrases, T6 natural, 6 research-retained incl. current-info-over-anchor, precedence pins) + pytest wrapper that restores graph nodes. | Phase 2 requirements 2-7. |
| `scripts/test_books_grounding.py` | `TestRealSourceFalseRefusalFixes` (8 tests) with VERBATIM NIST fixtures (printed p22 GOVERN / p25 MAP / p29 MEASURE / front-matter i) + captured A5 raw output; meta-word, Map/Measure, multi-passage, unanswerable-C, supers-insufficient, diagnostics-print tests. | Phase 1 test list (Govern verbatim, Map/Measure, paraphrase, multi-part, relevant-but-insufficient, unanswerable, diagnostics). |
| `scripts/test_citation_integrity.py` | `TestCitationNoiseAcrossDocuments` (3 tests) with the real orbital docx chunk + real NIST disclaimer/p36 noise; high-sim (0.75) and low-sim noise subtests; multi-document citation preservation; noise-only refusal. | Phase 3 requirements 1-6 incl. final rendered answer. |
| `scripts/test_self_refusal_recovery.py` | 4 tests: question-structure words no longer block eligibility (A6 shape), B1-shape still blocked (not over-granted), recovery prompt carries the partial clause. | Phase 1 reqs 2, 4, 5, 7. |

No other source files were touched. No dependencies, datasets, prompts
(outside the two authorized prompt clauses), PDFs, manifest, or configs.

## 3. Tests added and results

Commands (exact, all offline/deterministic, no network, no Ollama in tests):

- Baseline focused: `pytest scripts/test_phase1_routing.py scripts/test_citation_integrity.py scripts/test_books_grounding.py scripts/test_books_grounding_paraphrase.py scripts/test_self_refusal_recovery.py scripts/test_books_grounding_adversarial.py -q`
  -> before: **120 passed, 30 subtests, exit 0**; after: **135 passed, 34 subtests, exit 0** (+15 tests, +4 subtests = exactly the new tests).
- Full documented suite: `pytest scripts -q` -> before **629 passed, 2 skipped, 121 subtests (304s), exit 0**; after **644 passed, 2 skipped, 125 subtests (225s), exit 0**.
- `python scripts/test_phase1_routing.py` (direct) -> `STEP6_WIRING_OK`, exit 0.
- One mid-cycle regression was caught and fixed, not suppressed: an
  in-string comment in the stop list temporarily disabled the `not`
  requirement and broke `test_recovered_polarity_reversal_is_rejected`;
  the comment was moved outside the string and the A/B script
  (`.diag/fix_cycle_baseline/ab_polarity.py`) confirmed old==new behavior
  restored (negated claim still refused).

## 4. Before/after offline evaluation metrics

Exact commands (exit codes captured without pipes):

```
env\Scripts\python.exe scripts\eval_answers.py --mode unit --gold benchmarks/rag_gold_v3.json --out .diag/fix_cycle_baseline/unit_after2.json        -> exit 0
env\Scripts\python.exe scripts\eval_answers.py --mode integration --gold benchmarks/rag_gold_v3.json --out .diag/fix_cycle_baseline/integration_after2.json -> exit 0
```

| Metric | Before (unit_before.json) | After | Change |
|---|---|---|---|
| unit: executed/passed/failed | 17 / 17 / 0 (2 unresolved C101/C103 excluded) | 17 / 17 / 0 | identical |
| unit: abstention, citation, contradiction, disclosure, error, evidence_use, facts_present, groundedness, recovery, relevance | all 1.0 | all 1.0 | identical |
| unit: abstention P/R | 1.0 / 1.0 (tp=5 fp=0 fn=0) | identical | none |
| integration: executed/passed/failed/skipped | 10 / 10 / 0 / 7 | 10 / 10 / 0 / 7 | identical |
| integration: hit@1 / hit@3 / coverage | 0.8 / 1.0 / 1.0 | 0.8 / 1.0 / 1.0 | identical |
| integration: precision@1 / precision@3 / rr / ndcg@5 / fact_coverage | 0.9 / 0.3667 / 0.9333 / 0.95 / 1.0 | identical | none |
| isolation | temp store, guard_unchanged=True, network_attempts=0 | temp store, guard_unchanged=True, network_attempts=0 | identical |

Gold datasets unchanged (sha pins enforced by `test_eval_answers.py`, which
passes in both runs). Exit-3 overwrite protection also observed working.

## 5-6. GUI verification: questions, routes, answers, evidence, citations

Setup: backend restarted AFTER the edits — old PID 3872 terminated (verified
port free), new uvicorn started (`Started server process [34648]`, log
`.diag/gui_rag_fix_2026-10-25/backend_phase5.log`), health
`{"status":"ok",...,"provider":"ollama","model qwen3:0.6b"}`, frontend :3333
HTTP 200 through the Vite proxy. Every execution was created through the GUI
New Execution form; all routes below are GUI-observed (`routing.decided`),
never predicted. **All 16 distinct questions routed `books`; 17 executions
total (one same-wording B2 retry); `[research_agent]` lines: 0.**

Retrieval is deterministic across cycles: scores match the pre-fix dumps
exactly (e.g. A1 0.8252168258459255, T3 0.7185562618372222, T5
0.6547125008967275), so the read-only evidence dumps remain valid for these
runs.

### Original seven NIST regression questions (exact wording, unchanged)

| # | Exec | Route | Answer (verbatim key part) / outcome | Citations (printed labels) | Category | Before -> After |
|---|---|---|---|---|---|---|
| T1 | `1e04af96ea5e` | books | "The four core functions ... 1. **GOVERN** – designed to inform and be infused throughout other functions. 2. 3. 4. Note: ..." (partial + disclosure) | 3; 20; 42 — all footer-verified correct | correct partial + valid citations | answered -> answered (parity) |
| T2 | `dfe1f088b9f4` | books | **answered**: "- **Accountable and Transparent** ... - **Fair with Harmful Bias** ... Note: ..." | 12; 16; 13 — §2.2/§3 trustworthiness pages, verified | correct partial + valid citations | **false refusal -> ANSWERED** |
| T3 | `c5b2887f3bf6` | books | refusal (model self-refusal; NO recovery attempt line — coverage still <0.5: "difference/concrete/activity" absent from Table 2/3 chunks) | none (correct on refusal) | false refusal (remaining) | refusal -> refusal |
| T4 | `aa3248aff06d` | books | **answered**: "The function also facilitates transparency, documentation, and accountability in AI systems. Note: ..." | 22 — the GOVERN cross-cutting page (viewer 27), verified | correct partial + valid citation | **false refusal -> ANSWERED** |
| T5 | `fc6ab3668457` | books | "The requested information was not found in the books." (fast path, no recovery — correct) | none | correct abstention | abstain -> abstain |
| T6 | `797d7e743d39` | **books** | **answered**: "The relevant AI RMF function is MAP. Note: ..." (recovery succeeded) | 29 — real retrieved page whose text discusses "the MAP function" (strongest support is printed 25/28; attribution nuance, not a fabrication) | correct answer + adequate citation | **research+external -> books+ANSWERED** |
| T7 | `cf2821976ff1` | books | **answered**: "The AI RMF's purpose is to address AI risks by being risk-based, resource-efficient, pro-innovation, and voluntary. Its four core functions are GOVERN, MAP, MEASURE, and MANAGE ... Note: ..." | 42; 3; 20 — purpose/functions pages verified | correct partial + valid citations | partial -> partial (parity; limitation part still evidence-limited) |

### Realistic AI-document questions (Phase 6B set, exact wording)

| # | Exec | Route | Outcome | Citations | Category | Before -> After |
|---|---|---|---|---|---|---|
| A1 | `37b67c5ee1ed` | books | refusal — **recovery ran and succeeded** (270 chars) but the gate dropped the near-perfect sentence on one missing token: `gate_drop: unsupported (missing tokens: main): "The NIST AI Risk Management Framework is about managing AI risks, with its main parts including the Core and functions GOVERN, MAP, MEASURE, and MANAGE."` | none | false refusal (diagnosed to the token) | refusal -> refusal w/ exact cause |
| A2 | `ba4877526574` | **books** | refusal — gate dropped both sentences on `essential` and `highlight` (same non-assertive class as the fixed `emphasiz`) | none | false refusal (diagnosed) | **not-run (safety) -> GUI-observed books route** |
| A3 | `f47286b1122e` | **books** | **answered**: "...4. **MANAGE**: Manages AI risks and activities in practice. Note: ..." | 3; 20; 4 — correct | correct partial + valid citations | **not-run -> books+ANSWERED** |
| A4 | `598998aabb0b` | books | refusal — recovery ran (score 0.865), `second_refusal` -> bounded rejection | none | false refusal (model capability; bounded recovery intact) | refusal -> refusal |
| A5 | `1e5186abd2f1` | books | **ANSWERED**: "The document states that the GOVERN function is a cross-cutting function integrated into AI risk management processes. Note: ..." | **22** — the verbatim rank-1 evidence page | correct answer + exact citation | **false refusal -> ANSWERED (flagship fix)** |
| A6 | `e38b19b1a147` | books | refusal — **coverage now eligible: recovery ATTEMPTED (0.808) where it was previously blocked**, then `second_refusal` -> bounded rejection | none | false refusal (eligibility fixed; model capability remains) | coverage-blocked -> recovery ran |
| C | `17f059b23be7` | books | "The requested information was not found in the books." (12.1s, no invented number) | none | correct abstention | abstain -> abstain |
| B1 | `50395da60fc6` | books | refusal with proof-grade diagnostics: `gate_drop: unsupported (missing tokens: add, inspect, recommend, super): "The field guide recommends using a corridor of 6 to 9 millimeters for inspecting a hive before adding honey supers."` — the bee chunk really lacks those terms | none | correct abstention (relevant-looking but insufficient) | abstain -> abstain (now evidenced) |
| B2 | `51fa11564e72` (+ retry `f74344b2d1f7`) | books | refusal both attempts (recovery attempted, bounded second refusal) — **no footer at all: no NIST noise citation appeared, and the unevidenced "thrust" premise was NOT repeated as documented fact** | none | correct-for-premise abstention; answered-path footer not re-observable in GUI | answered-with-noise -> refusal (premise-safe) |

### Phase-5 backend diagnostics (new observability, verbatim excerpts)

```
[books_agent] grounding refusal (25.4s, score=0.8252168258459255)
[books_agent] gate_drop: unsupported (missing tokens: main): The NIST AI Risk Management Framework is about managing AI risks, with its main parts including the "Core" and functions GOVERN, MAP, MEASURE, and MANAGE.
[books_agent] grounding refusal (27.6s, score=0.8578194111835772)
[books_agent] gate_drop: unsupported (missing tokens: essential): The framework explains that trustworthy AI characteristics (like valid and reliable) are essential for trustworthiness, ...
[books_agent] grounding refusal (21.4s, score=0.7834581536086234)
[books_agent] gate_drop: unsupported (missing tokens: add, inspect, recommend, super): The field guide recommends using a corridor of 6 to 9 millimeters for inspecting a hive before adding honey supers.
```

Every refusal now names its stage and exact reason; the user-facing phrase is
unchanged.

## 7. Remaining failures and limitations

1. **Stop-list candidates confirmed but NOT changed (Phase 5 forbids code
   changes during verification):** `main` (A1 — its sentence was otherwise
   fully evidenced) and `highlight`/`essential` (A2 — reporting/meta class
   identical to the fixed `emphasiz`). Next-cycle candidates, awaiting
   approval.
2. **T3 coverage block remains:** abstract task nouns
   ("difference", "concrete", "activity") are absent from the retrieved
   Table 2/3 chunks, so eligibility still denies the retry. Fixing this
   needs a coverage-design decision (not a stop word).
3. **Model capability floor (qwen3:0.6b):** A4 and A6 (and both B2
   attempts) produced a second refusal after an eligible, evidence-primed
   retry — bounded by contract (one retry, no loops). A1/A2/A6 show the
   pipeline working while the model or the last un-stopped token still
   blocks the outcome.
4. **B2 answered-path footer** could not be re-observed in the GUI (refusals
   only); the CN-1 fix is pinned by
   `test_group_covered_claim_cites_only_the_supporting_document` (rendered
   answer asserted, high- and low-sim noise subtests) and multi-document
   citation preservation is pinned separately.
5. **Formatting thinness** (empty "2. 3. 4." list markers in T1/A3) is the
   model's output plus sentence-drop disclosure — cosmetic, unchanged.
6. **Nondeterminism:** qwen3:0.6b output varies run-to-run; each test is a
   single execution (B2 retried once, same wording, recorded separately).
7. **No live-answer-quality scoring** was invented; categories above are
   evidence-based, not numeric.

Acceptance criteria check:
- Natural document questions route to books without keywords: **PASS** (16/16 GUI-observed; A2/A3 exact wording included).
- Genuine research questions retain research: **PASS** (offline-pinned: latest-news-over-NIST, transformers, AI-art, weather, F1 -> research; feasibility/ERP precedence intact).
- Answerable questions not falsely refused: **PARTIAL** (T2/T4/A5 flipped; 5 remain, each diagnosed).
- Unavailable information still refused: **PASS** (T5, C, B1 with proof, premise-unsafe B2).
- Citations correct, not fabricated: **PASS on all 7 answered runs** (printed-label verification), noise filter test-pinned.
- Recovery bounded, no bypass: **PASS** (attempted -> second_refusal -> rejected; no loops; max one retry).
- Existing tests pass / failures explained: **PASS** (644 passed; every residual explained above).

## 8. Final Git status and integrity checks

- Branch `main`, HEAD `08d44b6` — unchanged; **staged files: 0**; no commit,
  push, branch switch, reset, clean, stash, or deploy.
- Working tree vs the cycle-start snapshot (`.diag/fix_cycle_baseline/git_status_before.txt`):
  identical pre-existing entries, plus this cycle's edits to
  `app/agents/books_agent.py`, `app/supervisor.py`,
  `scripts/test_books_grounding.py`, `scripts/test_citation_integrity.py`,
  `scripts/test_phase1_routing.py`, `scripts/test_self_refusal_recovery.py`
  (the first two were already modified pre-cycle and received the authorized
  additions), plus new files: this report, `.diag/fix_cycle_baseline/*`
  (baseline snapshots, scripts, before/after artifacts),
  `.diag/gui_rag_fix_2026-10-25/backend_phase5.log` and `P5_*.json`
  execution records.
- Frozen gold datasets untouched: `rag_gold_v2.json` (2026-10-23 20:20),
  `rag_gold_v2_candidate.json` (2026-10-23 21:52), `rag_gold_v3.json`
  (2026-10-23 21:46) mtimes unchanged; sha pins pass.
- Source PDFs / manifest untouched: NIST PDF `mtime_ns` byte-matches the
  manifest; all `last_processed_at` values remain 2026-10-24; no
  ingest/re-ingest command run; failed-extraction PDFs not modified.
- External calls: none — Phase-5 backend log has 0 `[research_agent]`
  lines; all tests/evals report `network_attempts: 0`; only local Ollama
  (GUI answers) and localhost HTTP were used.
- Production data: no writes; Chroma opened read-only (retriever path) for
  evidence dumps; evals used isolated temp stores with
  `guard_unchanged=True`.

**Stopping here as instructed: no commit, no further fix cycle.**

---

# ADDITIONAL WORKSTREAM (2026-10-25 evening) — Orchestrated Knowledge Fallback: Books → Internet Research

Sections 1–8 above document the controlled fix cycle (verdict PARTIAL). This
section extends the same report with the separately authorized fallback
workstream: its own implementation, orchestration decision table, regression
results, live GUI acceptance results and limitations — kept independently
identifiable so the fallback workstream cannot conceal or be concealed by the
fix-cycle results.

**Workstream verdict: PARTIAL** — 4 of the 5 GUI acceptance bullets PASS;
bullet 2 (books-missing → research → *properly sourced answer*) is PARTIAL:
the fallback chain, live search and live crawling are proven end-to-end, but
no positive externally-sourced answer was synthesized in this environment
(honest abstentions instead; causes in §9.7).

## 9.1 Scope and approach

- Read the existing orchestrator (`app/supervisor.py` routing + decision
  layer, `app/graph.py` topology) and `app/agents/research_agent.py`
  interfaces first; then implemented the smallest architecture-consistent
  change (no orchestrator rewrite, no new search provider).
- Not changed: grounding/abstention thresholds and gates, frozen gold
  datasets, source PDFs, ingestion manifest, dependencies, deployment
  settings, unrelated features. No re-ingest. No commit, push, branch switch,
  reset, clean, stash or deploy. All pre-existing changes preserved.
- Automated tests stayed offline (scripted doubles, `network_attempts: 0`);
  live external calls were limited to the three explicitly identified GUI
  acceptance runs in §9.6 (G2, G5, G5b) plus one manual localhost SearXNG
  diagnostic probe.

## 9.2 Implementation — exact files and rationale

| File | Change | Rationale |
|---|---|---|
| `app/state.py` | New optional channel `books_refusal: Optional[dict]`, documented as present ONLY on a books refusal. | The decision must read a structured diagnosis; a similarity score or the refusal sentence alone is explicitly not a sufficient input. |
| `app/config.py` | `BOOKS_WEB_FALLBACK` env switch (default ON; `"0"` restores pre-fallback terminal-refusal behavior). | Operator control; loop-safety comes from graph topology (exactly one books→research→END transition), not from this switch. |
| `app/agents/books_agent.py` | `_refusal_update()` attaches the scalar diagnosis (refusal stage/reason, score, node count, bounded-recovery outcome `not_attempted`/`skipped_not_eligible`/`attempted`/`succeeded`/`rejected`) to `books_refusal` on refusals. User-facing refusal text stays byte-identical (mandated phrase, no references footer). | Decision needs stage + recovery state; scalars only — never passage text, never model output (privacy / no-leak contract). |
| `app/supervisor.py` | `fallback_decision()` + `_FAILURE_CLASS` (maps the five refusal reasons to failure classes); guards: switch, diagnosis present, direct books route only, `_LOCAL_ONLY_RE` explicit local-only, `_DOC_DEICTIC_RE` document-scoped, `research_already_ran`. New `_LOCAL_ONLY_RE` / `_DOC_DEICTIC_RE` precedence: explicit local-only outranks every source cue in `keyword_route`. | The orchestrator's books→research decision, distinguishing the seven required failure states; a document-only request is never silently converted into an internet-sourced answer. |
| `app/graph.py` | `_after_books` consults `fallback_decision()`; one conditional edge books→research; research's own edge is `END` for every non-feasibility route; logs `[graph] knowledge fallback books->research (<code>)`. | Fallback transitions bounded to ONE per run — books↔research loops are structurally impossible; the log line carries the diagnostic code. |
| `app/agents/research_agent.py` | When `state["books_refusal"]` is present, prepend the provenance note ("the local books were consulted first … reason … Any answer below is based on external web research, not the local documents"); no-results path returns an honest abstention under that note; the query stays the ORIGINAL question via the existing rewrite step. | Provenance separation required by the spec; the refusal sentence and retrieved passage text never become the search query. |
| `scripts/test_knowledge_fallback.py` | **NEW — 17 tests / 12 classes.** | The ten required regression scenarios (mapping in §9.4). |
| `scripts/test_agent_architecture.py`, `scripts/test_chroma_isolation.py` | Key-shape assertions updated to legitimately include `books_refusal` on refusal-path outputs; `AgentState` expected-key set gains `books_refusal`. | Contract updates caused by the new state channel — assertion changes are additive, none weakened. |

No other source files were touched by this workstream.

## 9.3 Orchestration decision table

Decision inputs (never a score alone, never a refusal sentence alone): the
structured `books_refusal` diagnosis; the ORIGINAL question's source
constraints; run context (direct route only, research not already run).

| # | Failure state | Detected by | Orchestration behavior | Evidence |
|---|---|---|---|---|
| 1 | Sufficient local evidence | books answers (no diagnosis recorded) | Answer from Books/RAG; research never invoked | GUI **G1**, **G4**: only `[books_agent] answered …`, no `[graph]`/`[research_agent]` lines; offline: `TestAnswerableLocalNoWeb` (zero research calls, citations kept) |
| 2 | Relevant evidence but failed synthesis/validation | reason `self_refusal_after_recovery` (recovery already exhausted inside books' bounded M5: never twice, never for low score, never after the grounding gate) | Bounded local recovery FIRST (inside books), then `fallback_decision` may hand the ORIGINAL question to research | GUI **G2/G5/G5b** log chains: `self_refusal_recovery_attempted` → `self_refusal_recovery_rejected (reason=second_refusal)` → `not-found path` → `[graph] knowledge fallback …`; offline: `TestRecoveryBeforeFallback` |
| 3 | No relevant local evidence | reason `low_confidence` (score below the hard floor) → class `no_local_evidence` | Fallback allowed | offline: `TestFallbackToResearch` (query = original question) |
| 4 | Ambiguous retrieval | eligibility logic inside books (one bounded attempt) | Recovery attempt inside books before any source switch; fallback only after a structured refusal exists | GUI G5/G5b (attempted → rejected); offline `TestRecoveryBeforeFallback` |
| 5 | Explicit local-only / no-internet request | routing: `_LOCAL_ONLY_RE` → books; decision guard: `local_only_request` | Abstain locally; **zero** research | GUI **G3**: not-found path only — no fallback line, no research line; offline `TestLocalOnlyAbstains` + `test_local_only_constraints_outrank_web_and_current_info` |
| 6 | Document-scoped (deictic) request | decision guard `_DOC_DEICTIC_RE` ("this/that/the document") | Fallback blocked; books abstention survives | offline-pinned (T5-class questions, `TestLocalOnlyAbstains` / fix-cycle T5) |
| 7 | Explicit web-research or current-info request | routing `_EXPLICIT_WEB_RE` / `_CURRENT_INFO_RE` → route=research as PRIMARY (books never runs) | Research primary source | offline-pinned: `TestExplicitWebRouting` (3 tests) + existing `test_phase1_routing.py` pins |
| 8 | Research unavailable / inconclusive | research agent: search returned nothing, or the synthesis declines from the excerpts | Transparent limitation + provenance note; never invent an answer | GUI **G5** (search timeout → "Web search returned no results…") and **G5b** (model: "excerpts do not contain…" + 5 sources listed); offline `TestResearchFailureTransparent` |
| 9 | Research succeeded | research synthesis | Answer with external sources, provenance note kept distinguishable | offline `test_real_research_success_labels_external_provenance`; live chain proven in G2/G5b (search+crawl+synthesize ran) but no positive synthesis this environment — see §9.7 |

**Loop and limit guarantees.** Exactly ONE books→research transition is
possible per run (research's edge is `END` for non-feasibility routes; the
`research_already_ran` guard refuses a second); the research query is always
the original user question (`research_agent` reads `state["question"]`),
never a Books refusal sentence and never retrieved passage text; local
recovery stays bounded to M5's single retry. Diagnostics preserved: the
`books_refusal` dict, the `[graph] knowledge fallback … (<code>)` log line
(code = `books_insufficient:<failure_class>:<reason>`), and the
`[books_agent]` recovery-outcome lines.

**Privacy / network controls.** `books_refusal` carries scalars only — no
passage text, no model output, no conversation context. Observed live search
queries were built from the original question alone: `'NIST AI RMF on
quantum computing safeguards'` (G2) and `'NIST AI RMF EU AI Act'`
(G5/G5b) — no local document content was sent externally. Local-only
requests never reach the research path (G3: zero external). The provenance
note tells the user when an answer is external, and external claims are kept
separate from local-document claims (`TestProvenanceDistinguishable`). All
automated tests run offline with scripted doubles and a failing-closed
network guard (`TestPrivacyGuards.test_network_guard_is_armed_and_failing_closed`).

## 9.4 Offline regression results (exact commands, counts, exit codes)

All runs use `env\Scripts\python.exe`, offline, no Ollama, no SearXNG.

| Command | Result | Exit |
|---|---|---|
| `pytest scripts -q` (canonical full documented suite) | **661 passed, 2 skipped, 138 subtests (184s)** — exactly the fix-cycle 644 + the 17 new fallback tests + their 13 subtests | 0 |
| Focused batch: `pytest scripts/test_knowledge_fallback.py scripts/test_phase1_routing.py scripts/test_self_refusal_recovery.py scripts/test_phase4_workflow_qa.py scripts/test_phase5_qa_node.py scripts/test_phase6_routing_llm.py scripts/test_agent_architecture.py scripts/test_gui1_events.py scripts/test_api_preview.py scripts/test_gui16_settings.py scripts/test_gui5_operations.py scripts/test_phase7_planner.py scripts/test_phase8_execution.py scripts/test_books_grounding.py scripts/test_ingest_integrity.py scripts/test_citation_integrity.py scripts/test_verifier.py scripts/test_m4c_attribution.py scripts/test_books_grounding_paraphrase.py scripts/test_books_grounding_adversarial.py scripts/test_gui12_plan_render.py -q` | **492 passed, 1 skipped, 77 subtests (205s)** | 0 |
| `pytest scripts/test_eval_answers.py scripts/test_rag_eval_harness.py scripts/test_chroma_isolation.py scripts/test_rag_gold_v2_candidate.py -q` | **128 passed, 1 skipped, 51 subtests (17.8s)** | 0 |
| `python scripts/test_phase2.py` / `test_feasibility_degradation.py` / `test_rewrite_step.py` / `test_phase3_erp.py` | all completed, `OK`/`*_OK` markers | 0 / 0 / 0 / 0 |

The 17 new tests map to the ten required scenarios:

| Required scenario | Test class(es) in `test_knowledge_fallback.py` |
|---|---|
| 1. Indexed-PDF question answered locally, citations, no web call | `TestAnswerableLocalNoWeb` |
| 2. Absent from books → fallback → externally supported answer | `TestFallbackToResearch`, `TestResearchFailureTransparent::test_real_research_success_labels_external_provenance` |
| 3. Relevant passages, failed initial synthesis → bounded recovery before fallback | `TestRecoveryBeforeFallback` |
| 4. Grounding failure not bypassed by fallback | `TestGroundingGateNotBypassed` (unsupported claim never reaches the answer **or** the search query) |
| 5. Local-only request abstains without research | `TestLocalOnlyAbstains` |
| 6. Explicit web request → research primary | `TestExplicitWebRouting` (3 tests, incl. precedence and corpus-questions-still-books) |
| 7. Research failure/inconclusive → transparent abstention | `TestResearchFailureTransparent` |
| 8. Mixed local+research provenance stays distinguishable | `TestProvenanceDistinguishable` |
| 9. Retry/fallback limits, no loops | `TestFallbackLimits` (topology = exactly one transition; `_after_books` guards; config switch) + `TestInvocationCounts` (each node exactly once per run) |
| 10. Privacy + zero real network calls in tests | `TestPrivacyGuards` (no document content/refusal text in search state; network guard armed and failing-closed) |

**RAG-AE offline evaluations** (run this workstream, artifacts
`.diag/knowledge_fallback_2026-10-25/unit_v3.json` + `integration_v3.json`,
run AFTER the last source edit — source mtimes 18:19–18:33, evals 18:44–18:45):
metrics are byte-identical to the §4 after-column: unit 17/17/0, all
sub-scores 1.0, abstention P/R 1.0/1.0 (tp=5 fp=0 fn=0); integration
10/10/0/7 skipped, hit@1 0.8, hit@3 1.0, coverage 1.0, precision@1 0.9,
precision@3 0.3667, rr 0.9333, ndcg@5 0.95, fact_coverage 1.0;
isolation `guard_unchanged=True`, `network_attempts: 0`, gold sha pins pass.
**No evaluation regression from the fallback workstream.**

## 9.5 Backend/frontend verification (before any GUI result was trusted)

- First GUI submission this workstream failed with a visible error card:
  `No module named 'langgraph'` — the running backend had been started with
  the system Python instead of the project venv. Process evidence:
  old PID 62592 terminated; restarted as
  `nohup env/Scripts/python.exe -m uvicorn … --port 8000` →
  `.diag/knowledge_fallback_2026-10-25/backend2.log` line 1
  `Started server process [62820]`, `Application startup complete.`,
  `Uvicorn running on http://127.0.0.1:8000`.
- Health (recorded, but deliberately NOT treated as proof of the fixes):
  `GET /api/v1/health` → 200
  `{"status":"ok","backend":"ok","ai":{"provider":"ollama","state":"ready","what_will_be_tested":"Ollama (http://localhost:11434), model qwen3:0.6b"},"version":"0.1.0"}`.
- Frontend `http://localhost:3333` → 200 through the Vite proxy; all six
  executions below were created through the real GUI New Execution form
  (`POST /api/v1/executions … 201` lines in `backend2.log`), and every route
  and answer shown is what the GUI/API returned — never a predicted route.
- Records: `.diag/knowledge_fallback_2026-10-25/EX_*.json` (execution
  results), `EV_*.json` (event streams), `backend2.log` (decision lines).

## 9.6 GUI acceptance results (live; exact wording preserved)

| # | Exec | Exact question | Route (GUI) | Decision-chain evidence (`backend2.log`) | External calls | Outcome → category |
|---|---|---|---|---|---|---|
| G1 | `40c2206212ba` | “Explain the four core functions of the AI Risk Management Framework and what each one is meant to accomplish.” | books | `[books_agent] answered in 49.8s from 1 supporting chunk(s)`; no fallback/research lines | **0** | Answer: “The four core functions … are GOVERN, MAP, MEASURE, and MANAGE.” + `References: NIST.AI.100-1.pdf, page 3` — cited printed p.3 verified to contain the four-functions text → **PASS (bullet 1)** |
| G2 | `9f9fc6206085` | “What does the NIST AI RMF say about quantum computing safeguards?” | books | `not-found (24.6s, score=0.7529, reason=self_refusal_after_recovery)` → recovery had run → `[graph] knowledge fallback books->research (books_insufficient:recovery_exhausted:self_refusal_after_recovery)` → `search query: 'NIST AI RMF on quantum computing safeguards'` → `synthesized in 119.1s using 2 URL(s)` | 1 live batch (search + 2 crawls) | Provenance note + honest limitation (“web excerpts do not contain information … specifically tied to the NIST AI RMF”) + 2 sources listed → **PASS (bullets 4 & 5: recovery before fallback; both-sources-fail transparency)** |
| G3 | `8e70d7e4637a` | “Using only this document, what exact chunk size in characters does the ingestion pipeline require?” | books | `not-found (14.1s, score=0.6727, reason=self_refusal_after_recovery)` — **no `[graph] knowledge fallback` line, no `[research_agent]` line** | **0** | Exact mandated abstention “The requested information was not found in the books.” → **PASS (bullet 3: local-only stays local)** |
| G4 | `ddcdb35b1f95` | “How does the framework explain the relationship between trustworthy AI characteristics and managing AI risks?” (the original A2 wording) | books | `answered in 48.2s from 1 supporting chunk(s)`; no fallback/research | **0** | Answer + `References: NIST.AI.100-1.pdf, page 12` — printed p.12 = PDF page 17, section “3. AI Risks and Trustworthiness”, verbatim contains “Creating trustworthy AI requires balancing each of these characteristics based on the AI system’s …” and “Approaches which enhance AI trustworthiness can reduce negative AI risks” → answer and citation **verified correct** |
| G5 | `a360994640ad` | “What does the NIST AI RMF say about the EU AI Act?” (0 occurrences in the PDF — verified before running) | books | `recovery_attempted (score=0.7729, nodes=3)` → `recovery_rejected (reason=second_refusal)` → `not-found (25.8s)` → `[graph] knowledge fallback books->research (…)` → `search query: 'NIST AI RMF EU AI Act'` → `[searxng_client] search failed: Read timed out. (read timeout=15)` → `no search results available` | attempted (timed out; no crawl) | Provenance note + “Web search returned no results, so I could not research this question.” → **PASS (bullet 5: research unavailable → abstain transparently); fallback decision itself correct** |
| G5b | `cc618977941f` | Same question — one deliberate retry (SearXNG latency is variable; recorded as a separate execution, same wording) | books | `recovery_attempted` → `rejected(second_refusal)` → `not-found (22.3s)` → fallback → `search query: 'NIST AI RMF EU AI Act'` → `synthesized in 251.9s using 5 URL(s)` | 1 live batch (search + 3 crawl attempts) | Provenance note + honest decline (“The provided web excerpts do not contain information about the NIST AI RMF's specific stance on the EU AI Act.”) + 5 sources incl. an exactly on-topic first hit (“EU AI Act Vs NIST AI RMF: Comparing AI Risk Governance”). Crawl extraction returned empty for all 3 top URLs → snippet-only context (§9.7) → **bullet 2 PARTIAL: chain proven live, positive sourced answer not achieved** |

**Acceptance bullets:** (1) PDF question answered locally without web —
**PASS** (G1, G4); (2) missing → research → properly sourced answer when
research succeeds — **PARTIAL** (fallback + live search + crawl + synthesis
pipeline proven in G2/G5b, but every live run ended in an honest abstention:
causes §9.7); (3) local-only remains unanswered — **PASS** (G3, zero
external); (4) initial grounding refusal attempts bounded local recovery
before falling back — **PASS** (G2/G5/G5b log chains show
`attempted → rejected` BEFORE the fallback line); (5) both sources fail →
transparent limitation message — **PASS** (G2, G5).

## 9.7 Limitations and unresolved issues (this workstream)

1. **SearXNG environment is unhealthy.** Container `searxng` reports
   `Up 2 days (unhealthy)`: `/` answers in 0.1s but `/search` took **23s** in
   a direct probe, exceeding the research client's **hardcoded 15s** read
   timeout (`app/tools/searxng_client.py`) — G5 failed on exactly this.
   The timeout constant was NOT changed (pre-existing, outside this
   workstream's authorization). Recommended, awaiting approval: raise it or
   make it configurable.
2. **Crawl extraction returned empty for all 3 top URLs in G5b** → the
   agent's documented snippet-fallback path supplied only search snippets;
   `crawl_extractor` is pre-existing and was not modified.
3. **Model capability floor (`qwen3:0.6b`)** declined to synthesize from
   snippets in G2 and G5b (its prompt instructs “say so clearly instead of
   guessing” — the abstentions are honest, not hallucinations). A stronger
   model or better excerpts are needed for bullet 2's positive case; no
   hardcoded answers or test-specific logic were used.
4. **Research-primary routing and mocked-fallback scenarios were verified
   offline only** (§9.4) — live GUI exercised the fallback path, not the
   explicit-web-primary path, to keep identified external calls minimal.
5. **Single execution per GUI test**; `qwen3:0.6b` output is
   nondeterministic (G5/G5b shown as two separate records, same wording).
6. **G5b is a retry of G5** — recorded transparently rather than replacing
   the failed run; SearXNG's flakiness is an environment fact, not hidden.
7. No numeric answer-quality score was invented; categories in §9.6 are
   evidence-based (log lines, PDF page verification, saved records).

## 9.8 Final integrity check (this workstream)

- Branch `main`, HEAD `08d44b6` — unchanged; staged files: 0; **no commit,
  push, branch switch, reset, clean, stash, or deploy.**
- Working tree vs the cycle-start snapshot
  (`.diag/fix_cycle_baseline/git_status_before.txt`): every pre-existing and
  fix-cycle entry still present and unmodified in kind; NEW this workstream —
  modified `app/config.py`, `app/graph.py`, `app/state.py`,
  `app/agents/research_agent.py` (all clean at cycle start), plus authorized
  additions to the already-modified `app/agents/books_agent.py` and
  `app/supervisor.py`, contract updates in `scripts/test_agent_architecture.py`
  and `scripts/test_chroma_isolation.py`, and the new
  `scripts/test_knowledge_fallback.py`; new artifacts under
  `.diag/knowledge_fallback_2026-10-25/` (logs, EX/EV records, eval
  outputs) and this report's section 9.
- Frozen gold datasets untouched: `rag_gold_v2.json` (2026-10-23 20:20:36),
  `rag_gold_v3.json` (2026-10-23 21:46:21), `rag_gold_v2_candidate.json`
  (2026-10-23 21:52:03) — mtimes byte-identical to §8; sha pins pass in
  every evaluation.
- Source PDFs / manifest untouched: every manifest entry's `mtime_ns`
  byte-matches disk (`ALL mtime_ns match = True`, including
  `NIST.AI.100-1.pdf`); `last_processed_at` values remain 2026-10-24; no
  ingest/re-ingest command was run; failed-extraction PDFs not modified.
- External-call inventory (live): G2 (1 search + 2 crawls), G5 (1 search
  attempt, timed out), G5b (1 search + 3 crawl attempts), 1 manual localhost
  SearXNG probe — all explicitly identified above. Everything automated:
  `network_attempts: 0`. LLM: local Ollama `qwen3:0.6b` only — no
  paid/hosted models.
- Production data: no writes; Chroma accessed read-only through the
  retriever path during GUI runs; evals used isolated temp stores with
  `guard_unchanged=True`.

**Stopping as instructed: validation report extended; no commit, no further
fix cycle, no deployment work.**
