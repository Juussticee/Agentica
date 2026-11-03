# GUI RAG Fix Validation — 2026-10-25 14:15

Phase 6 real-GUI verification of the approved RAG fixes (routing, citation integrity,
false refusals, answer completeness), run against the unchanged NIST AI RMF 1.0 PDF
with the local Ollama `qwen3:0.6b` model. Companion to (does not replace)
`docs/GUI_RAG_TEST_REPORT_2026-10-24_1114.md` (original) and
`docs/GUI_RAG_ROOT_CAUSE_2026-10-24_2349.md` (root-cause).

**Final verdict: PARTIAL.** Two acceptance criteria pass (routing 6/7, citations
verified, abstention preserved); two fail (answerable questions are still falsely
refused; natural-wording T6 still misroutes to research). Details below.

---

## 1. Backend / frontend verification

| Check | Evidence |
|---|---|
| Backend process | uvicorn, PID 3872, started 2026-10-25 *after* the Phase 1–4 source edits: `.diag/gui_rag_fix_2026-10-25/backend.log` → `Started server process [3872] … Application startup complete.` |
| Health | `GET /api/v1/health` → `{"status":"ok","backend":"ok","ai":{"provider":"ollama","state":"ready","what_will_be_tested":"Ollama (http://localhost:11434), model qwen3:0.6b"}}` |
| Model/provider | Ollama `qwen3:0.6b` only (health + dashboard "AI connection" panel). No paid/hosted model was called. |
| Frontend | `http://localhost:3333` → HTTP 200; `frontend/vite.config.ts` uses `strictPort: 3333` with `'/api' → http://127.0.0.1:8000` proxy. |
| Frontend↔backend | Every execution below was created through the browser New Execution form (`#/new`); the backend access log shows the corresponding `127.0.0.1:*` `POST/GET /api/v1/executions…` traffic. |
| Edited code is live | Behavioral proof: the supervisor's document-deictic routing fix is observed in this process — the exact strict question that escaped to web research in the original run (`Using only this document, …`) now returns `route: books`. A healthy endpoint alone was not treated as proof. |

## 2. What was executed

All seven original NIST questions ran through the actual GUI, unmodified wording
(the natural/original wordings as recorded for T1–T3; T3's strict natural
Map/Measure question explicitly included). The original report additionally
recorded *variant* wordings for T3–T7; those five recorded variants were also run
byte-exact from `docs/GUI_RAG_TEST_REPORT_2026-10-24_1114.md` so the comparison
is like-for-like. No question was pre-filtered, rewritten, or omitted.

Execution IDs, routes, durations (from `created_at`→`completed_at`):

| Run | Execution ID | Route | Duration | Result class |
|---|---|---|---|---|
| T1 natural | `afc300c64b78` | books ✓ | 62.3 s | answered (partial) + disclosure |
| T2 natural | `ccafef259c20` | books ✓ | 39.4 s | **false refusal** (grounding gate) |
| T3 natural (strict) | `42e7dbf30b81` | books ✓ (orig: research) | 26.0 s | **false refusal** (model self-refusal, recovery gated) |
| T4 natural | `141c79e0a89c` | books ✓ | 27.4 s | **false refusal** (grounding gate) |
| T5 natural | `f5bb51c9ab6d` | books ✓ | 19.6 s | correct abstention ✓ |
| T7 natural | `768704949b0b` | books ✓ | 29.1 s | answered (partial) + disclosure |
| T6 natural | `2217601ef1df` | **research ✗** | 115.1 s | routing failure |
| T3v recorded | `6ed500d4de90` | books ✓ | 68.8 s | answered but **semantically garbled** |
| T4v recorded | `24664ea415a4` | books ✓ | 78.6 s | false refusal (recovery exhausted) |
| T5v recorded | `a18de6ab9c6e` | books ✓ | 20.0 s | correct abstention ✓ |
| T6v recorded | `e0a6bee95fa1` | books ✓ | 70.8 s | false refusal (recovery exhausted) |
| T7v recorded | `2479a0276c80` | books ✓ | 44.8 s | answered (partial) + disclosure |

Raw records: `.diag/gui_rag_fix_2026-10-25/natural_wording_results.json`,
`T3v_result.json` … `T7v_result.json`, `recorded_variants.json`.

## 3. Per-test detail (exact wording, full response, evidence, assessment)

**Citation semantics (applies throughout):** chunk `page_label` values are the
PDF's *printed* page footers, not the viewer index. Verified by extracting footers
with pypdf: viewer p8 → footer "Page 3"; viewer p25 → "Page 20"; viewer p27 →
"Page 22"; viewer p47 → "Page 42" (body offset = +5; front matter uses roman
labels, e.g. `page_label: i` = viewer p3). Every citation below was checked
against the printed label of the source chunk, i.e. the scholarly page a reader
cites.

### T1 — Direct factual retrieval (natural, exact as recorded) — PASS (partial)

> "According to the document, what are the four core functions of the AI Risk Management Framework? Explain each briefly and cite the supporting section or page."

Route: books. Response (verbatim):

> The four core functions of the AI Risk Management Framework are:
> 1. **GOVERN** – A cross-cutting function that governs and infuses AI risk management across all AI lifecycle stages. 2. 3. **MEASURE** – A function for measuring AI risks in practice. 4. **MANAGE** – A function for managing AI risks in a multidisciplinary and diverse way. Note: the answer above is limited to the parts supported by the retrieved book passages; the rest was not found in the books.
>
> References: NIST.AI.100-1.pdf, page 3; NIST.AI.100-1.pdf, page 20; NIST.AI.100-1.pdf, page 42

Evidence (read-only retrieval dump, top-3): printed p20 chunk (viewer 25)
contains verbatim "The AI RMF Core provides outcomes and actions … composed of
four functions: GOVERN, MAP, MEASURE, and MANAGE", plus "cross-cutting" and
"multidisciplinary"; printed p3 chunk (viewer 8) lists the four functions and
trustworthiness characteristics; printed p42 chunk (viewer 47) is Appendix D
"AI RMF strives to: …".

- Support: GOVERN/MEASURE/MANAGE descriptions and the four-function list are
  directly supported by the cited chunks ✓. MAP's bullet was dropped by the
  grounding gate (disclosure shown), leaving a dangling "2." — a formatting
  artifact of sentence-drop + preserved model numbering.
- Citations: pages 3/20/42 all verified correct to their printed footers ✓.
  Minor imprecision: the GOVERN/MANAGE description sentence cites page 3 where the
  descriptions themselves sit in the printed-20 chunk (page 3 carries the list).
- Original score: 6/10 (answered + disclosure). Now: same class, citations verified.

### T2 — Trustworthy AI (natural, exact as recorded) — FAIL (false refusal)

> "How does the document discuss trustworthy AI and the characteristics of trustworthiness? Distinguish the document's wording from your own explanation, and cite the supporting sections or pages."

Route: books. Response: `The requested information was not found in the books.`
Backend: `[books_agent] grounding refusal (39.4s, score=0.8418)`.

Evidence: top-3 chunks are printed p12 (viewer 17, §2.2/§3 "AI Risks and
Trustworthiness … characteristics of trustworthy AI"), printed p16 (viewer 21,
§4.1 operator/user notification), printed p13 (viewer 18, tradeoffs) — directly
relevant evidence was retrieved. The grounding gate dropped every model sentence
(`mode=refusal`); the dropped sentences are **not observable** (not persisted in
events or logs). Recovery did not run (M5 covers model self-refusals only, not
gate refusals — by design). Classification: **false refusal** despite supporting
retrieval (score 0.84). Original score: 6/10 (answered with disclosure) —
answer availability regressed vs the original run.

### T3 — Map vs Measure, natural strict wording (exact as recorded) — FAIL (false refusal), routing PASS

> "Using only this document, explain the difference between the Map and Measure functions. Give one concrete activity associated with each and cite the supporting evidence."

Route: **books** — the original run routed this exact question to research
(original T3 strict: 0/10). Routing fix confirmed on the original failure case.

Response: `The requested information was not found in the books.`
Backend: `not-found path (26.0s, score=0.7186, reason=self_refusal_after_recovery)`
— the model emitted the not-found phrase, and **no**
`self_refusal_recovery_attempted` line exists for this run: M5's topical-coverage
gate (`≥50%` of question content tokens present in evidence; "difference",
"concrete activity" are absent from table chunks) denied the one bounded retry.

Evidence: printed p28 (viewer 33, Table 2 MAP categories), printed p25 (viewer 30,
MAP→MEASURE transition), printed p29 (viewer 34, Table 3 MEASURE categories) —
directly relevant. Classification: **false refusal** (model synthesis failure +
recovery-eligibility gate). Original: 0/10 (misroute) / T3v 2/10 (floor refusal).

### T4 — Govern role, natural wording — FAIL (false refusal)

> "What role does the Govern function play in the AI RMF, and how does it relate to the other three functions? Cite evidence from the document."

Route: books. Response: `The requested information was not found in the books.`
Backend: `grounding refusal (27.4s, score=0.8180)`.

Evidence: printed p22 (viewer 27) chunk states verbatim "GOVERN is a
cross-cutting function that is infused throughout AI risk management and enables
the other functions …"; printed p41 (viewer 46) and printed p20 chunks also
retrieved. The document answers this question directly; the gate still refused.
Dropped sentences not observable. Classification: **false refusal**.
Original score: 2/10 (floor refusal) — unchanged outcome class.

### T5 — Unanswerable chunk-size question, natural wording — PASS (correct abstention)

> "What exact chunk size in tokens does this document require for a retrieval-augmented generation system?"

Route: books. Response: `The requested information was not found in the books.`
Backend: `not-found path (19.6s, score=0.6547, reason=self_refusal_after_recovery)`
(no recovery attempt — coverage gate denied it, correctly here).

The document prescribes no RAG chunk size; **no number was invented**.
Recorded variant T5v (byte-exact: "According to the sections of this document,
… Cite the page if such a requirement exists.") also returned the mandated
abstention (20.0 s, score 0.6693). Abstention is intact. Original: 10/10 ✓ parity.

### T6 — Paraphrase robustness, natural wording — FAIL (routing failure)

> "If a team wants to identify and understand the context of possible AI harms before measuring them, which AI RMF function is most relevant, and why?"

Route: **research** ✗ — execution completed before it could be stopped; there is
no cancel/stop API in the backend (documented limitation), so one SearXNG search
was fired: `[research_agent] search query: 'Identify AI harms and understand context before measuring.'`
→ `synthesized in 115.0s using 1 URL(s)`. This is the only external request in
the whole validation run.

Response (verbatim, for the record):

> The most relevant AI RMF function for identifying and understanding AI harms before measuring is the **Free AI Adoption Readiness Assessment**. This assessment evaluates readiness to implement AI strategies, which directly relates to context analysis and harm identification. No other functions are explicitly mentioned in the provided excerpts.
>
> Sources:
> - https://www.aiawareness.ai/ai-resources/responsible-and-ethical-ai-use/ai-harm-prevention-strategies/

Classification: **routing failure** + unsupported/off-topic answer (it does not
even name a real AI RMF function). The natural question contains no
document-deictic phrase ("this document", "the pdf") and no `_BOOKS_RE` term, so
the fixed router still falls through to research. The recorded variant
("In this document, … which section-based AI RMF function …") *does* route books:
T6v → books, model self-refused, recovery attempted (score 0.803) → rejected
(`second_refusal`) → bounded refusal. So with keyword-bearing wording the route
holds; the natural paraphrase remains misrouted. Original T6 variant: 2/10.

### T7 — Multi-part, natural wording — PASS (partial)

> "Summarize the purpose of the AI RMF, identify its four core functions, and explain one limitation or implementation consideration discussed by the document itself. Cite the evidence for each part."

Route: books. Response (verbatim):

> The purpose of the AI RMF is to be risk-based, resource-efficient, pro-innovation, and voluntary, developed through consensus and open processes. The four core functions are GOVERN, MAP, MEASURE, and MANAGE. - **Four Core Functions:** GOVERN, MAP, MEASURE, and MANAGE. Note: the answer above is limited to the parts supported by the retrieved book passages; the rest was not found in the books.
>
> References: NIST.AI.100-1.pdf, page 20; NIST.AI.100-1.pdf, page 42; NIST.AI.100-1.pdf, page 3

- Purpose part: verified against printed p42 (viewer 47) "AI RMF strives to …
  Be risk-based … pro-innovation … voluntary" ✓. Functions part: printed p20 ✓.
- Limitation/implementation part: dropped by the gate with disclosure. The
  retrieved printed-p42 chunk does contain implementation attributes ("strives to
  be feasible to implement…"), so evidence existed for *an* implementation
  consideration; the model's sentence for it was not kept (dropped text not
  observable). The other two retrieved chunks contain no limitation statement.
- Citations: pages 20/42/3 verified correct to printed footers ✓ (minor: page 3
  carries the function list, not the purpose statement).
- Original: 6/10 — parity (same partial class).

### Recorded variants T3v / T4v / T7v (byte-exact from the original report)

- **T3v** "…difference between the MAP and MEASURE functions described in their
  sections…" → books, answered:
  `**Answer:**  - **MAP (Measure)**: Measures AI risks and related impacts. Page: Table 2. Note: …`
  References: pages 28; 25; 29 — citations point to the correct MAP/MEASURE
  table pages ✓, but the content **misattributes** MEASURE's wording to MAP and
  drops the MEASURE bullet. Token-level grounding cannot catch entity misbinding;
  this is an unsupported/garbled answer that passed the gate. Original T3v: 2/10
  (floor refusal) — no longer a floor refusal, but not a correct answer either.
- **T4v** → books; model self-refused; recovery **attempted** (score 0.827,
  coverage passed) → rejected `second_refusal` → mandated refusal. Recovery is
  bounded (one retry, no loop) ✓ but the question remains falsely refused.
  Original: 2/10.
- **T7v** → books, answered partial (purpose ✓ + functions ✓, limitation part
  dropped + disclosure), References pages 20; 42; 3 ✓ same class as natural T7.
  Original: 6/10.

## 4. Failure-mode classification vs the original report

| Run | Original (score/class) | Now | Changed? |
|---|---|---|---|
| T1 | 6/10, answered+disclosure, citations called "misattributed" | answered+disclosure, **citations verified correct** (printed-label semantics) | citation finding overturned; answer class unchanged |
| T2 | 6/10, answered (contained ISO-9000 misattribution) | **false refusal** (gate) | worse: available evidence, no answer |
| T3 strict | 0/10, **research misroute** | books route ✓, false refusal (self-refusal, recovery gated) | routing fixed; answer still missing |
| T3v | 2/10 floor refusal | answered but semantically garbled (citations correct) | different failure: entity misbinding |
| T4 | 2/10 floor refusal | **false refusal** (gate) | unchanged class |
| T5 | 10/10 correct abstention | correct abstention ✓ (natural + variant) | parity ✓ |
| T6 | variant 2/10 books-route refusal | natural **routes research** (external call fired); variant routes books → bounded refusal | natural wording still fails routing |
| T7 | 6/10 partial | answered partial (purpose+functions; limitation dropped) | parity |

**Citation finding — correction to the original report.** The original report's
"critical citation failures" (pages 3/20/42 "not containing the claims") were
checked against *viewer* PDF indices. The pipeline stores the document's *printed*
page labels (footer-verified: viewer 25 = "Page 20", viewer 47 = "Page 42"). Under
printed-page semantics — what a reader citing the document would use — every
citation in every answering run here maps to the chunk whose text supports the
claim. There is no off-by-one in the pipeline; there is an **undocumented label
ambiguity**: the GUI does not tell the user whether "page 20" is the printed page
or the viewer page (viewer page for printed 20 is 25). Remaining real imprecision:
two sentences cite printed p3 for content whose strongest support is printed p20.

## 5. Acceptance criteria

| Criterion | Result |
|---|---|
| Original natural questions work through the actual GUI | **PARTIAL** — 6/7 route books; T6 natural still routes research |
| Citations point to supporting evidence | **PASS** — all cited printed pages verified against the source; label ambiguity is cosmetic but real |
| Answerable questions not falsely refused | **FAIL** — T2, T3, T4 (+T4v, T6v) refused despite directly supporting retrieval |
| Unavailable information still refused | **PASS** — T5/T5v abstain with no invented number |

**Overall: PARTIAL.** The fixes demonstrably landed for routing (6/7), citation
integrity (overturned a false original finding and verified the real mapping),
and abstention. False refusals were *not* resolved end-to-end: the unit suites
(feed well-formed sentences into `_grounded_repair`) stay green — 87 passed + 12
subtests re-run after this GUI work — but the live `qwen3:0.6b` synthesis + gate +
coverage-gated recovery combination still withholds answers the corpus supports.

Observed mechanism split (from backend logs, per run):
- Gate refusals (`grounding refusal`): T2, T4 natural — model produced an answer,
  every sentence failed token support (dropped text not persisted → exact trigger
  not observable). Recovery does not apply by design.
- Model self-refusals with recovery **gated out**: T3 natural (coverage <50%),
  T5 (correctly gated).
- Model self-refusals with recovery **attempted and exhausted**: T4v, T6v
  (attempted → `second_refusal` → rejected; bounded, no loops) ✓ mechanism works.
- Entity misbinding passed the gate: T3v ("MAP (Measure): Measures AI risks…").

## 6. Limitations, side effects, and unobservable checks

- One SearXNG/web request fired (T6 natural, 1 search, 1 URL, 115 s). No cancel
  API exists; the run was recorded and no further research-routed test was
  submitted. All other 11 runs were books-routed with zero external calls.
- Grounding-gate *dropped sentences* for T2/T4 are not persisted anywhere
  (events store only `qa_flags: []`; logs only the outcome line) → the precise
  rejected text is **not observable**; classification rests on route, score,
  outcome line, and the retrieved-evidence dump.
- Model outputs are nondeterministic run-to-run (same question, different
  failure mode across runs: T4 natural = gate refusal, T4v = self-refusal).
  Single execution per question; no averaging claimed.
- Live/web-search answer quality was not scored (out of scope); T6's answer is
  recorded verbatim and classified as unsupported.
- No numeric scores are invented here; only the original report's recorded
  scores are quoted for comparison.

## 7. Final integrity check

- Git: branch `main`, HEAD `08d44b6` — **unchanged**; nothing staged
  (`git diff --cached` empty); no commit, push, branch switch, reset, clean, or
  stash performed.
- Working tree: exactly the pre-task snapshot — the 12 pre-existing modified
  files (`app/agents/books_agent.py`, `app/supervisor.py`,
  `benchmarks/rag_gold_v2_candidate.json`, `docs/RAG_EVAL_GAPS.md`,
  `docs/TESTING.md`, and the seven test/fixture scripts) plus the same untracked
  prior-task outputs. New from this task only: this report and new files under
  `.diag/gui_rag_fix_2026-10-25/`.
- Frozen datasets untouched: `benchmarks/rag_gold_v2.json` mtime 2026-10-23
  20:20, `benchmarks/rag_gold_v2_candidate.json` mtime 2026-10-23 21:52 (its
  ` M` status predates this task). No dataset was written during this validation.
- Source PDFs untouched / no re-ingestion: `knowledge/books/NIST.AI.100-1.pdf`
  size 1,946,127 B and `mtime_ns` byte-match `storage/.index_manifest.json`
  (`last_processed_at: 2026-10-24T07:49:32+00:00`); SHA-256
  `7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1`.
  No ingest/re-ingest command was run; Chroma was opened read-only (retriever
  path) for the evidence dump.
- Production data: no writes, deletes, or schema changes anywhere; evidence
  dump used the app's own query engine only (no LLM synthesis).
- Model calls: local Ollama `qwen3:0.6b` only; no paid/hosted provider.
- Focused regressions re-run after GUI findings: `pytest
  scripts/test_phase1_routing.py scripts/test_citation_integrity.py
  scripts/test_books_grounding.py scripts/test_books_grounding_paraphrase.py
  scripts/test_self_refusal_recovery.py -q` → **87 passed, 12 subtests passed,
  exit 0**.

## 8. Remaining gaps (no fix cycle started — awaiting approval)

1. **False refusals** (T2/T3/T4): root cause is end-to-end model synthesis vs
   the token-level gate plus M5 coverage-gate eligibility; unit tests do not
   exercise this path with live model output. Needs an approved design decision
   (e.g., persist dropped sentences for diagnosis; evidence-primed retry wording;
   or a model-capability floor) — not a threshold tweak.
2. **T6 natural paraphrase misroute**: router lacks cues for RMF-paraphrase
   questions with no document-deictic wording; research fallback persists.
3. **Entity misbinding** (T3v): token-level grounding accepts sentences whose
   entity pairing is wrong.
4. **Label ambiguity**: GUI should distinguish printed page vs viewer index.
5. **Dropped-sentence observability**: gate refusals leave no audit trail.*Proposed next tests (not executed): persistence of rejected sentences; a two-execution determinism probe per question; router regression corpus including T6-natural paraphrases.*

---

# Phase 6B — Exploratory AI document-research tests (appended 2026-10-25 14:5x)

A separate, exploratory suite run later the same day through the same live GUI,
backend (PID 3872), model (Ollama `qwen3:0.6b`), and unmodified routing/retrieval
pipeline. **These are new exploratory questions, distinct from the seven original
regression questions in Sections 1–8 above.** Nothing in the application, gold
datasets, prompts, routing rules, or source files was changed during this phase;
all source inspection was read-only. **Zero external/web requests were made in
this phase** (no research-routed execution was submitted — see A2/A3 below).
Raw records: `.diag/gui_rag_fix_2026-10-25/P6B_*.json`, `P6B_retrieval.json`.

## B. Document inventory (read-only, `storage/.index_manifest.json`)

| File | Status | Chunks | Note |
|---|---|---|---|
| `NIST.AI.100-1.pdf` (AI RMF 1.0) | indexed | 48 (unit=page) | processed 2026-10-24T07:49:32Z — target document |
| `NIST.AI.600-1.pdf.pdf` (GenAI Profile) | **failed** | 0 | "parsed as binary data … Install the file readers: pip install llama-index-readers-file docx2txt pypdf" — **not retrievable; recorded, not modified** |
| `nasa_systems_engineering_handbook_0.pdf` | **failed** | 0 | same extraction failure — **not retrievable; recorded, not modified** |
| `orbital_kinetics_principles.docx` | indexed | 1 | manifest v1 entry, verified against vector store |
| `urban_beekeeping_field_guide.pdf` | indexed | 1 | 1 page, 814 chars |

Consequence for Section B: the only readable *AI* document in the collection is
`NIST.AI.100-1.pdf` (the GenAI Profile and NASA handbook cannot be selected —
extraction previously failed). Cross-document selection was therefore tested
against the two other genuinely indexed sources (beekeeping, orbital) with one
natural question each.

## A + C + B — Executions (all through the GUI, natural wording, no added keywords)

| Test | Exec ID | Route (GUI-observed) | Top-1 document (score) | Outcome category | Result |
|---|---|---|---|---|---|
| A1 document discovery | `140f60ed575d` | books ✓ | NIST.100-1 (0.825) | **False refusal** (self-refusal; recovery attempted → second refusal; evidence supports a partial answer) | FAIL |
| A2 core concepts | — | **not run (safety)** | — | predicted route `research` → would fire external request | NOT RUN |
| A3 framework structure | — | **not run (safety)** | — | predicted route `research` → would fire external request | NOT RUN |
| A4 practical application | `b79dfb21fb80` | books ✓ | NIST.100-1 (0.865) | **False refusal** (self-refusal; recovery attempted → second refusal) | FAIL |
| A5 specific evidence (Govern) | `e967373090f0` | books ✓ | NIST.100-1 (0.736, printed p22 verbatim) | **False refusal** (grounding gate) | FAIL |
| A6 cross-section synthesis | `7df13a931782` | books ✓ | NIST.100-1 (0.808) | **False refusal** (self-refusal; recovery gated out by coverage check) | FAIL |
| C abstention (chunk size in characters) | `058fbd42315c` | books ✓ | NIST.100-1 (0.662, back-matter) | **Correct abstention** — no number invented | PASS |
| B1 hive inspection (bee guide) | `87a19dbb2d5a` | books ✓ | **urban_beekeeping_field_guide.pdf (0.783)** | **Correct abstention + correct document selection** — source lacks "supers" entirely | PASS |
| B2 thrust/speed (orbital) | `f9b78274895c` | books ✓ | **orbital_kinetics_principles.docx (0.762)** | Partially correct answer (recovery **succeeded** — first observed success) with one unsupported detail and one spurious citation | PARTIAL |

**A2/A3 not-run basis (not a silent omission):** `ROUTING_MODE=keyword` (verified
from config and runtime), and a read-only call of the unmodified
`keyword_route()` with the exact question strings returns `research` for both —
neither contains any document-deictic or book term (`How does the framework
explain…`, `Explain the four core functions…`). The API has no
cancel/stop/abort endpoint (verified: routes are health, create/list/get/events/
plan/sequence for executions, ai settings, services, activity — no execution
cancel), and the research agent calls SearXNG right after its query-rewrite LLM
step, so a submitted execution would make an external request before it could be
stopped. Per the safety rule they are recorded as **not run**, category
"incorrect route (predicted) + not run for safety", and are **not counted as
passing**. The keyword router is deterministic, so this prediction is the route
the GUI would have reported — but it is labeled a prediction, not a
GUI-observed route.

### Exact wording, full responses, evidence, and source verification

All refusals below are the mandated sentence verbatim:
`The requested information was not found in the books.` (no references footer, no
invented content). Backend classification and retrieval evidence per run:

- **A1** "I want to understand the NIST AI Risk Management Framework. What is this document about, and what are its main parts?" — books, 52.6 s wall (agent 37.5 s, score 0.825). Backend: `self_refusal_recovery_attempted → rejected (second_refusal) → not-found`. Retrieved: printed p3 (trustworthiness list + four functions), printed p4 ("Part 1: Foundational Information 1. Framing Risk"), printed p9 (accountability/culture). Source-verified: "Part 1: Foundational Information" on viewer 9 (= printed 4) ✓ in evidence; "Part 2: Core and Profiles" on viewer 25 (= printed 20) — **not** in the retrieved top-3, so a complete "main parts" answer needed evidence the retriever did not surface. Still, "what it is about" + Part 1 were supportable → refusal withheld an answer the evidence partly supports.
- **A4** "According to this document, how should an organization approach identifying and managing risks associated with an AI system?" — books, 53.9 s (agent 40.5 s, score 0.865). Backend: recovery attempted → second refusal → not-found. Retrieved: printed p5 (harms/risks framing), printed p8 ("risks which the organization determines to be highest … most urgent prioritization"), printed p32 (Table 4, MANAGE categories — source-verified: "Table 4: Categories and sub…" on viewer 37/38 = printed 32/33 ✓). Directly relevant evidence → false refusal.
- **A5** "What does the document say about the Govern function, and how does it relate to the other functions?" — books, 41.4 s, **grounding refusal** (score 0.736). Retrieved top-1 is the exact source sentence — printed p22 (viewer 27 verified): "GOVERN is a cross-cutting function that is infused throughout AI risk management and enables the other functions of the process" — plus printed p23 (Table 1 GOVERN) and printed p41. This is the clearest false refusal of the phase: the document's verbatim answer was rank-1 evidence and the gate still refused. Dropped model text not persisted (not observable).
- **A6** "Using this document, summarize how the framework connects its overall purpose, its core functions, and the characteristics of trustworthy AI." — books, 31.3 s (score 0.808). Backend: `not-found` **with no recovery attempt** (coverage gate denied). Retrieved: printed p12 (viewer 17 verified: "For AI systems to be trustworthy …" §3), printed p42 (viewer 47: Appendix D attributes/"strives to" = purpose), plus a list-of-figures page. Three-part evidence present → false refusal.
- **C** "What exact chunk size in characters does this document's ingestion pipeline use?" — books, 12.2 s (score 0.662; top-3 are back-matter/cover: DOI page, title page, Fig. 3 page). No recovery (coverage correctly denied), **no number invented** → correct abstention. Also confirms the question is unanswerable from *any* indexed source: neither the bee (814 chars) nor orbital (1,353 chars) text nor the NIST PDF describes an ingestion pipeline.
- **B1** "What does the field guide recommend for inspecting a hive before adding honey supers?" — books, 26.9 s (score 0.783). Top-1 = the bee guide's single chunk (**correct document**, ahead of two NIST chunks at 0.647/0.643). Full source read (1 page, 814 chars): mentions "inspections" (smokers) but contains **no "supers" text at all** → the requested recommendation does not exist in the source; refusal = correct abstention, and document selection passed.
- **B2** "What does the orbital kinetics material say about how thrust changes a spacecraft's speed?" — books, 35.6 s, **answered after successful recovery** (`self_refusal_recovery_attempted → succeeded (chars=326) → answered in 21.5s from 2 supporting chunk(s)`). Response verbatim:

  > Thus, thrust changes the spacecraft's speed as per the equation. Note: the answer above is limited to the parts supported by the retrieved book passages; the rest was not found in the books.
  >
  > References: orbital_kinetics_principles.docx; NIST.AI.100-1.pdf, page i

  Assessment: correct document cited (`orbital_kinetics_principles.docx`, rank-1
  at 0.762; source read-only-verified: speed/equation/velocity present, rocket
  equation and delta-v content real). Two defects: (1) the word **"thrust"
  appears nowhere in the source document** (verified by full-text scan) — the
  answer carries the question's premise through a partial-support pass rather
  than noting the source discusses delta-v/mass-ratio instead; (2) the second
  citation, `NIST.AI.100-1.pdf, page i` (the disclaimer page), is a real page but
  **irrelevant to the claim** — an unsupported citation drawn from low-rank
  noise chunks (0.603) that were also in the top-3.

## D. Scores against the outcome categories

| Test | Category | Pass? |
|---|---|---|
| A1 | False refusal (recovery exhausted) | FAIL |
| A2 | Incorrect route (predicted research) — not run for safety | FAIL / NOT RUN |
| A3 | Incorrect route (predicted research) — not run for safety | FAIL / NOT RUN |
| A4 | False refusal (recovery exhausted) | FAIL |
| A5 | False refusal (grounding gate, rank-1 verbatim evidence) | FAIL |
| A6 | False refusal (recovery gated out) | FAIL |
| C | Correct abstention | PASS |
| B1 | Correct abstention + correct document selection | PASS |
| B2 | Partially correct/incomplete answer; correct document; 1 unsupported detail, 1 irrelevant citation | PARTIAL |

No run produced an unsupported *confident* answer to a document-grounded NIST
question — the failure mode this phase exposed is systematic withholding, not
hallucination. The one synthesis that did occur (B2) came from the M5 recovery
path, which therefore works end-to-end when the model cooperates.

## E. Acceptance criteria — Phase 6B

| # | Capability | Result | Concrete evidence |
|---|---|---|---|
| 1 | Natural-language AI questions without special routing keywords | **FAIL** | A2/A3 contain no book/document term and predict-research (deterministic router); keyword-free natural questions still cannot reach books. Only "this/the document" deictics routed (A1/A4/A5/A6). |
| 2 | Select the appropriate existing PDF | **PASS** | NIST.100-1 rank-1 for every A/C question; bee guide rank-1 for B1; orbital docx rank-1 for B2 — scores match the executions exactly (dump reproduces every run's retrieval). GenAI Profile/NASA could not compete (extraction failed, recorded). |
| 3 | Retrieve relevant rather than merely similar passages | **PASS (with a gap)** | A4/A5/A6 top chunks verified verbatim against source pages; gap: A1's "main parts" evidence missed the "Part 2: Core and Profiles" page, and B1/B2 top-3 carried irrelevant NIST chunks (0.60–0.65) that leaked into B2's citation footer. |
| 4 | Synthesize a faithful answer from passages | **FAIL** | 4 of 5 answerable A-tests refused (A1, A4, A5, A6) despite relevant evidence; only B2 synthesized, with a premise-carrying detail. |
| 5 | Cite correct source pages, avoid fabricated citations | **PARTIAL** | No fabricated page numbers (printed-label semantics verified in Phase 6); B2 cited the correct file plus one *irrelevant* real page (NIST p.i). Refusals cite nothing (correct). |
| 6 | Acknowledge uncertainty/missing information | **PASS** | C and B1 abstained without inventing numbers/recommendations; B1's source provably lacks "supers". |
| 7 | Readable, complete answer through the real GUI | **PARTIAL** | Both displayed answers are readable and carry the disclosure note; but 6 of 9 tests displayed only the refusal sentence, which is honest yet uninformative for answerable questions. |

**Phase 6B summary:** capabilities 2 and 6 pass; 3 and 5 partially pass; 1 and 4
fail; 7 partially passes. Document selection and abstention are trustworthy;
natural routing (keyword-free phrasing) and answer synthesis are not. Overall
**FAIL-leaning PARTIAL**, dominated by the same false-refusal root cause found in
Phase 6 — this phase shows it is the *systematic* outcome (4/5 answerable
questions withheld), not an occasional one.

Integrity for this phase: HEAD `08d44b6` unchanged, nothing staged; no source,
dataset, prompt, routing, config, or PDF changes (manifest `last_processed_at`
values unchanged; no ingest command run; Chroma opened read-only); no external
or paid model calls (local Ollama only; no research execution submitted); no
commits. Per instructions, **no fix cycle was started — evidence collected,
reporting stopped for review.**
