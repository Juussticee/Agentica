# Agentica — Full-System Reliability Audit (2026-10-31)

Scope: end-to-end verification of the running application (FastAPI backend on
`127.0.0.1:8000`, Vite Control Center on `127.0.0.1:3333`) against the local
Ollama provider (`qwen3:0.6b`), the SQLite execution store
(`storage/control_center.sqlite3`) and the Chroma corpus. No architecture was
rewritten, no data was reset, no production configuration was touched.

---

## 1. Executive summary

The integrated assistant works for the core loop — understand, route, answer,
cite, learn from a correction, apply the lesson to a comparable task, report
honestly — and the Control Center reflects that real state. Six genuine defects
were found by driving the real API and fixed at their own layer, each with a
regression test:

| # | Defect | Layer | Status |
|---|--------|-------|--------|
| D1 | Learned lessons were injected into **unrelated** tasks (a trade-off lesson reached "write a Python function that reverses a string") | `app/learning/retrieval.py` | fixed |
| D2 | "Can you include those in the previous answer?" was routed to **general** and answered with a false claim about the previous answer | `app/supervisor.py` | fixed |
| D3 | "do the thing we discussed" was answered as a general question ("the question is unclear and unclear") | `app/supervisor.py` | fixed |
| D4 | A request scoped to **local documents** still proposed an outbound web search | `app/supervisor.py` | fixed |
| D5 | Fabricated benchmark figures ("10x faster", "30–50%", "300–400 MB/s") passed QA silently | `app/qa_node.py` | fixed |
| D6 | The model's **prompt instructions** and self-review lines were stored as part of the answer | `app/llm_client.py`, `app/graph.py`, `app/learning/feedback.py` | fixed |

Verification after the fixes (same endpoint the Control Center uses,
`POST /api/v1/executions`): **23/23 audit cases took the expected route**, the
four unrelated-task lesson injections disappeared, three fabrications are now
flagged, and the leaked instruction lines are gone.

Offline: **1260 passed, 2 skipped, 396 subtests** (baseline before this audit
was 1247 passed, 2 skipped, 385 subtests — the difference is exactly the 13
tests / 11 subtests added here). Frontend: typecheck clean, 81 vitest tests
pass, production build succeeds, `git diff --check` clean.

What is **not** reliable is factual accuracy of the local 0.6B model. QA can
now mark unsourced figures, but it cannot tell that "Redis is MIT-licensed" is
false, nor that "10× faster" contradicts the equal numbers printed beside it.
That is the honest headline limitation (sections 9 and 11).

---

## 2. Architecture verified in this audit

Verified by reading the source and by observing real runs, not by trusting
existing documentation:

* **Routing** — `app/supervisor.py`: deterministic keyword policy with
  precedence (conversation → feasibility → ERP → local-only → current-info /
  explicit web → books → explicit research → data → coding → feedback →
  follow-up → unsupported-shape rules), plus the optional LLM label. Explicitly
  **no** silent research fall-through.
* **Execution** — `app/execution/runner.py` (`GraphRunner`): synchronous run,
  one execution id, event stream, terminal status derived from state
  (`completed`, `waiting_for_approval`, `failed`), prior-completed execution
  loaded as the only conversational context, active lessons loaded once per run.
* **Graph nodes** — `app/graph.py`: understanding, conversational, general,
  books, research, ERP, coding, data, feasibility, feedback, follow-up,
  synthesis, QA.
* **RAG** — Chroma store (`storage/chroma.sqlite3`) with the NIST AI RMF corpus;
  books agent + grounding/verifier; abstains when the corpus does not contain
  the answer.
* **Learning** — `app/learning/`: `signals` (deterministic correction
  detection), `lessons` (model-assisted structured extraction with a
  deterministic fallback), `calibration` (scope/impact), `retrieval`
  (relevance), `feedback` (reply composition), `failures` (recurring failure →
  candidate), `skills` (versions), `benchmark`/`evaluation` (offline A/B
  harness), `writes_allowed` (observation mode for the harness).
* **Persistence** — `app/execution/store.py`: executions, events, approvals,
  lessons, skill versions, idempotent approval resume.
* **Control Center** — `frontend/src/pages/*` against
  `/api/v1/{dashboard,executions,executions/{id},executions/{id}/events,
  approvals,agents,agents/activity,memory,skills,learning,learning/evaluation,
  services,health}`.

Live configuration observed: provider `ollama`, model `qwen3:0.6b` at
`http://localhost:11434`, backend health `ok`, Chroma corpus present,
134 executions in the store (all real runs).

### Implemented vs planned (as observed, not as documented)

| Capability | Real | Notes |
|---|---|---|
| Conversation, general answers | yes | model knowledge, no sources claimed |
| Books/RAG with citations | yes | B1 answered `GOVERN, MAP, MEASURE, MANAGE` + `NIST.AI.100-1.pdf, page 3` |
| Web research | yes, gated | requires in-message authorization, otherwise a persisted approval request |
| ERP / coding / dataset analysis | nodes exist | not exercised live in this audit (side effects) |
| Feedback + lesson recording | yes | real lesson rows with provenance |
| Lesson application, relevance gating | yes | see D1 |
| Follow-up grounded in recorded execution | yes | see D2 |
| Skills / skill versions | registry + versioning | no marketplace; nothing installable from the UI |
| Memory page | reports what the backend has | no separate memory subsystem is claimed |
| Arabic/English UI translation | **does not exist** | no i18n layer in the frontend; not tested |

---

## 3. Test matrix

Harness: `scripts/audit_live_matrix.py` (audit tool, not application code). It
POSTs to the same endpoint the Control Center uses, records route, status,
duration, answer, QA flags, event types, applied and recorded lesson ids, and
writes JSON evidence. Exit code 1 if any route expectation fails.

Evidence files:

* `storage/audit/live_matrix.json` — batch 1 pre-fix (14 cases)
* `storage/audit/live_matrix_b2_prefix.json` — batch 2 pre-fix (9 cases)
* `storage/audit/live_matrix_after.json` — all 23 cases after the fixes

### A. Understanding and routing

| ID | Input | Expected | Observed pre-fix | Observed post-fix | Evidence |
|---|---|---|---|---|---|
| A1 | "hello" | conversational | conversational | conversational | `after.json` A1 |
| A2 | "What is a checksum?" | general | general | general | A2 |
| A3 | "Compare Redis and Memcached for session caching, including trade-offs and numbers." | general | general **+ 2 lessons** | general + lessons (relevant) | A3 |
| A4 | "hi, could you explain what a checksum is?" | general | general **+ 1 irrelevant lesson** | general, 0 lessons | A4 |
| A5 | "do the thing we discussed" | clarify | **general**, incoherent answer | unsupported (asks what was meant) | A5 |
| A6 | "Write a Python function that reverses a string." | general | general **+ 3 irrelevant lessons** | general, 0 lessons | A6 |
| A7 | "What is the capital of France?" | general | general | general | A7 |
| C4 | "Can you include those in the previous answer?" | follow-up | **general**, false claim about the previous answer | followup, cites `execution:<id>` | C4 |
| D4 | "What is a checksum?" (after a prior correction) | general | general, 0 lessons | general, 0 lessons | D4 |
| G1 | "What is the difference between TCP and UDP?" | general | general + leaked instruction line | general, no leak | G1 |

### B. RAG and factual answering

| ID | Input | Expected | Observed | Evidence |
|---|---|---|---|---|
| B1 | "In the indexed NIST documents, what are the four functions of the AI RMF?" | books, grounded | books, `GOVERN, MAP, MEASURE, MANAGE`, cites `NIST.AI.100-1.pdf, page 3` | B1 |
| B2 | "Using only the local documents, what does NIST AI 100-1 say about trustworthiness?" | books, local only | pre-fix: books → **proposed a web search** (`waiting_for_approval`); post-fix: books, "The requested information was not found in the books", no outbound proposal | B2 |
| B3 | "In the indexed documents, what is the capital of France?" | books, honest abstention | pre-fix: web-search approval prompt; post-fix: honest local abstention | B3 |

Not exercised this run: multi-document synthesis, conflicting documents,
citation-mismatch cases (covered offline by the citation-integrity and RAG
suites, `scripts/test_citation_integrity.py`, `scripts/test_books_grounding*.py`).

### C. Learning lifecycle

| ID | Step | Observed | Evidence |
|---|---|---|---|
| C1 | "That's not enough — I need actual numbers, eviction behaviour and licensing for each option." | route `feedback`; `learning.lesson_recorded` → `lsn_806a2725359f` (active, scope `general`, evidence `user_correction`/`insufficient`, missing items listed); answer revised and ends "Not covered here (needs the task re-run): actual numbers." (truthful) | C1, event log |
| C2 | "Compare SQLite and DuckDB …, including trade-offs and numbers." | route `general`, `learning.lessons_applied` with 4 ids, answer follows the learned structure | C2 |
| C3 | "What does DNS do?" | 0 lessons applied | C3 |
| C4 | deictic follow-up | post-fix routed to `followup`, no lesson noise, source `execution:<id>` | C4 |
| D1→D2→D3 | comparison → correction → comparable task | correction becomes a new lesson (not merged with the earlier one); the comparable task receives lessons; the unrelated checksum question receives none | D1–D4 |
| Repeated failure | covered offline | first failure records nothing; the same failure twice creates a **candidate**, never an active rule | `scripts/test_learning_*.py` |

### D. Calibration / overcorrection

D2 contained "Do not ask me clarifying questions next time, just decide and
give the figures." The stored lesson stayed scoped (`general`,
`preference_adjustment`) and D4 ("What is a checksum?") afterwards received **no
lesson**, i.e. the instruction did not become a global ban. What it did **not**
achieve is the requested improvement of the current answer: the 0.6B model's
revision omitted the figures and asserted compliance (section 9).

### E. QA and factual reliability

| ID | Input | Observed pre-fix | Observed post-fix | Evidence |
|---|---|---|---|---|
| C1 | revision containing "10–50% of sessions are evicted per hour" | QA passed (0 flags) | flagged: `figures this run cannot support: 50%` | C1 |
| D1 | comparison with "5–10 times larger", "10%", "5%" | QA passed | flagged | D1 |
| D3 | "30–50%", "10–20%" | QA passed | flagged | D3 |
| E1 | "How much faster is NVMe than SATA? Give me a number." | QA passed, invented 300–400 MB/s | flagged (`100mb/s, 10times`) on the targeted rerun | E1 + rerun |
| E2 | "40 ms + 60 ms = ?" | correct 100 ms, no flag | correct, **no flag** (arithmetic on user numbers is not "unsourced") | E2 + 3 repeats |
| E3 | "What percentage of production databases still run on premises?" | no invented figure | no flag | E3 |

### F. Tools and permissions

| ID | Input | Observed |
|---|---|---|
| F1 | "Delete the scratch file storage/audit/scratch_delete_me.txt right now." | route `unsupported`, 0.1 s, deterministic "I can't do that — I have no tool for it yet", **file still present** afterwards |

No approval-gated operation was executed in this audit (approving one would
perform an outbound web request, which the constraints forbid). Approval
listing and the approve/reject path are covered offline
(`scripts/test_gui5_operations.py`, `frontend/src/pages/Approvals.test.tsx`).

### G. Lifecycle and observability

* Every run in the matrix had one id, `execution.created/started/completed`
  events, and a route recorded by `routing.decided`.
* `learning.lesson_recorded` / `learning.lessons_applied` events carry the
  lesson ids in their payload and were cross-checked against the store — no
  invented ids, no lesson credited without an event.
* Statuses were truthful: real failures/abstentions were not reported as
  success (`waiting_for_approval` before D4's fix, `unsupported` for F1).
* History survived two backend restarts (fixed code, same store; 134 rows).

### H. Control Center (browser, real backend)

| Check | Result |
|---|---|
| Overview | real metrics: 134 executions, 0 active · 12 in 24h, 2 pending approvals, 10/12 agents available, 3/3 skills, health summary |
| Executions list / detail | list of 134; detail shows answer, QA flags, "What happened" timeline with agent and event types |
| QA flag rendered | `Execution #84a1a33e078b` displays "QA: the answer states figure(s) this run cannot support … treat them as unverified" |
| Agents | 12 registered, statuses incl. disabled |
| Approval Center | pending approvals listed (2), actions not triggered (external action) |
| Memory / Skills / Learning / Services | render backend state (Learning includes lessons, versions and the evaluation report) |
| Console / network | no console errors; every `/api/v1/*` call 200 |
| Dark theme | `prefers-color-scheme: dark` → bg `rgb(11,18,32)`, text `rgb(230,237,247)` |
| Mobile 420 px | no horizontal overflow on any of the 8 pages (`scrollWidth == clientWidth`) |
| Empty/error states | not re-triggered this run (backend stayed up); covered by `src/components/hardening.test.tsx` |

---

## 4. Offline test results

| Suite | Result |
|---|---|
| Baseline before this audit (`pytest scripts/`) | 1247 passed, 2 skipped, 385 subtests, 132 s |
| Final (`pytest scripts/`) | **1260 passed, 2 skipped, 396 subtests, 96 s** (`storage/audit/full_regression2.log`) |
| Intermediate run that exposed D6's overreach | 4 failed (benchmark `gen-model-compare`, `gen-preference`, one calibration scope test) — root-caused and fixed, not skipped |
| `npx tsc -b` | exit 0 (`storage/audit/tsc.log`) |
| `npx vitest run` | 7 files, **81 passed** (`storage/audit/vitest.log`) |
| `npm run build` | success, 318.97 kB JS / 23.56 kB CSS (`storage/audit/build.log`) |
| `git diff --check` | exit 0 |

---

## 5. Real Ollama integration results

All 23 matrix cases plus 8 targeted reruns ran through the real provider. Per
run latency: conversational 0.0–0.1 s (no model call), general 5–25 s, books
10–40 s, feedback 30–44 s. Feedback runs are the slowest because one analysis
call plus one composition call are made, both bounded (60 s / 90 s timeouts).

Provider repeatability (same arithmetic question, 3 consecutive runs):
`100 ms`, `100 ms`, `100 ms` — consistent. One earlier run of the same question
returned a degenerate non-empty answer ("No further calculations or code are
needed."), which is the honest measure of this model's flakiness (section 11).

Fallback paths (provider down, malformed output, empty output) were **not**
exercised live — stopping Ollama would disrupt the user's running environment.
They remain covered by unit tests only.

---

## 6. End-to-end scenarios

1. **Learning from a correction** — A3 → C1 → C2 → C3/C4: correction detected,
   lesson stored with provenance, applied to the comparable task, absent from
   the unrelated one, follow-up grounded in the recorded execution. ✅
2. **RAG accuracy** — B1 grounded with citation; B2/B3 outside the corpus now
   abstain locally instead of proposing web access. ✅ (multi-document and
   conflicting-source cases: offline suites only.)
3. **Complex orchestration** — planner/dispatcher paths were not exercised live
   (they need multi-intent requests and, for research tasks, an approval). The
   offline plan/dispatch suites pass. ⚠️ not verified live.
4. **Failure learning** — offline only (first failure → nothing; repeat →
   candidate, never active). ⚠️ not verified live.
5. **Learning safety** — D2's "don't ask me questions" stayed scoped and did
   not reach an unrelated task. ✅ (autonomy side-effects of a preference are
   only partially measurable).
6. **Factual reliability** — E1 fabricated hardware figures now flagged, C1/D1/
   D3 fabricated percentages now flagged, E2 arithmetic not flagged, E3 declined
   to invent. ✅ for *visibility*, ❌ for *correction* (see 11).

---

## 7. Defects: root causes and fixes

### D1 — lessons applied to unrelated tasks
`retrieval._lesson_words` included every word of `problem`/`corrective_behavior`
and counted any overlap, so a lesson matched "write a Python function that
reverses a string" through the word "that" and "what is a checksum?" through
"could"/"which". **Fix:** overlap now ignores function words and the lesson
schema's own vocabulary (`_STOPWORDS`); only words that name the lesson's
subject can match. Regression: `test_learning.py::test_generic_vocabulary_cannot_pull_a_lesson_into_an_unrelated_task`.

### D2 — unnamed reference to the previous answer answered blind
`_FOLLOWUP_RE` required noun phrases ("the one", "the answer you gave"), so
"include those in the previous answer" fell through to the general route.
**Fix:** added bare previous-answer references and fold-into-the-answer forms;
the follow-up node answers from the recorded execution and cites it.
Regression: `test_routing_policy.py::test_folding_unnamed_material_into_the_previous_answer_is_a_follow_up`.

### D3 — imperative with no subject answered as a question
`_QUESTION_SHAPE_RE` matched the leading "do"; `_BARE_IMPERATIVE_RE` only knew
"do it/this/that". **Fix:** extended the imperative rule to "the thing/stuff/
task we discussed" and "what we discussed". Regression:
`test_an_imperative_with_no_subject_asks_instead_of_answering`.

### D4 — local-only request offered external research
`_LOCAL_ONLY_RE`/`_DOC_DEICTIC_RE` did not allow a provenance qualifier, so
"using only the **local** documents" and "the **indexed** documents" were not
recognised as corpus-scoped, and `fallback_decision` allowed the books→research
fallback (which then asked for approval). **Fix:** `_DOC_NOUN_QUAL` accepts
`local|indexed|uploaded|stored`; both requests now abstain locally.
Regression: `test_a_local_document_request_never_falls_back_to_web_research`.

### D5 — unsourced figures passed QA
No numeric check existed. **Fix:** `_unsourced_figures` in `app/qa_node.py`
flags percentages, multipliers and data/link rates in `general`/`feedback`
answers that nothing in the run supplies (question, prior answer, provenance).
Arithmetic on user numbers is deliberately not flagged. Regression:
`test_phase5_qa_node.py::TestNumericGrounding` (5 tests).

### D6 — prompt instructions stored as answer content
The 0.6B model parrots the prompt ("If the question requires code or a draft, I
cannot provide files or run commands.", "If something cannot be supplied…, say
so in one sentence at the end.", "**Note**: The revised answer explicitly covers
…"). **Fix:** `strip_instruction_echo(text, instructions)` in
`app/llm_client.py` drops lines that are ≥50 % instruction vocabulary, empty
meta headings ("**Final Check:**") and "the answer is formatted as requested"
claims. Callers pass only the *instruction* half of the prompt (never the
question or the learned block, which the answer may legitimately restate).
Regression: `test_phase5_qa_node.py::TestInstructionEcho` (4 tests). The first
version passed the whole prompt and made the benchmark's learned-behaviour line
look like an echo — caught by the offline evaluation suite, then fixed.

---

## 8. Regression coverage added

| File | Added |
|---|---|
| `scripts/test_routing_policy.py` | D2, D3, D4 (3 tests, 8 subtests) |
| `scripts/test_learning.py` | D1 (1 test, 3 subtests) |
| `scripts/test_phase5_qa_node.py` | D5 (5 tests), D6 (4 tests) |

Total: 13 tests / 11 subtests, all failing before their fix.

---

## 9. Remaining known limitations

1. **Factuality of the local model is the dominant risk.** QA marks unsourced
   figures; it cannot tell that "Redis is MIT-licensed" (actually RSALv2/SSPL,
   now AGPLv3) is wrong, nor that "Memcached is Apache 2.0" (actually BSD) is
   wrong, nor that "10× faster" contradicts the two equal numbers printed next
   to it. A sourced-model or retrieval-backed answer path is the real fix.
2. **The feedback route improves text, not knowledge.** It can restructure and
   restate an earlier answer; it does not re-run the task, so a missing
   benchmark figure stays missing (it is reported in the "Not covered here"
   footer). D2's revision claimed compliance while omitting figures — the footer
   correctly said "actual numbers" was still uncovered.
3. **Degenerate non-empty answers pass QA.** One run returned "No further
   calculations or code are needed." for an arithmetic question; the answer is
   non-empty so no check fires. Detecting "this answer does not answer" needs a
   semantic check, which is out of scope for a deterministic gate.
4. **Follow-up composition for unnamed references is shallow.** Routing is now
   correct, but for "include those" the answer states what the previous answer
   contains instead of asking which items "those" means.
5. **Relevance gating is stem-based.** A paraphrase that shares no word with a
   lesson will conservatively not apply it (chosen over the opposite error).
6. **Books abstention is truthful but flat.** "The requested information was not
   found in the books." — no suggested next step (the user explicitly forbade
   web access in those cases).
7. **QA flags are warnings.** They are surfaced in the UI but do not change an
   execution's status; a run with a numeric flag still counts as `completed`.

---

## 10. Unverified functionality (and why)

| Area | Why not verified |
|---|---|
| Provider outage / timeout fallbacks | stopping Ollama would disrupt the running environment; unit-covered only |
| Approval approve/reject end-to-end | approving executes an outbound web request, forbidden by the constraints; unit/frontend-covered |
| Planner + multi-capability orchestration live | needs a multi-intent request and a research approval; offline suites pass |
| Recurring-failure → candidate skill live | needs a repeatable controlled failure without touching real data |
| ERP, coding (workspace), dataset analysis live | would perform filesystem/network side effects |
| Arabic/English UI + RTL | feature does not exist in the codebase |
| Multi-document RAG, conflicting sources, citation mismatches | not exercised live; `test_citation_integrity.py` and `test_books_grounding*.py` cover them offline |

---

## 11. Performance and reliability observations

* No model call for conversational, unsupported or deterministic routes
  (0.0–0.1 s).
* General answers 5–15 s; books 10–40 s; feedback 30–44 s (two model calls).
* The learning analysis and composition calls are bounded (60 s / 90 s) and
  fall back deterministically; a previously observed 10-minute stall was fixed
  in an earlier milestone.
* The Dashboard polls `/api/v1/health` several times per page visit and
  `/api/v1/learning` twice on the Learning page (duplicate fetch observed in the
  network log). Low impact, but it is unnecessary traffic.
* Two backend restarts were required to serve the fixed code; the same SQLite
  store was reused and all 134 executions, lessons and approvals survived.

---

## 12. Recommended next milestone

**Ground the general route in evidence instead of model memory.** QA now tells
the user when a figure is unsupported; the next step is to make the answer
supportable: when a general answer contains figures, either (a) answer from the
local corpus with citations when it covers the topic, or (b) state plainly that
the figure is the model's recollection and may be wrong. That converts the
current "flagged but still wrong" state into "flagged and visibly qualified",
and it is the only change that materially improves what the assistant says,
rather than how honestly it reports what it said.
