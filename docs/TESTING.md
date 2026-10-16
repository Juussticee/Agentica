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

Incremental and per-file: `storage\.index_manifest.json` (schema v2) records,
for each file, its mtime, status (`indexed` / `partial` / `failed`), extracted
document/page counts, pages with no extractable text, and the number of chunks
actually persisted in Chroma. Re-running skips unchanged healthy files without
re-embedding, and the manifest is rewritten byte-identically when nothing
changed. A legacy v1 manifest (flat filename → mtime) is migrated in place and
each entry's chunk count is verified against the vector store instead of being
trusted.

A file is **never** recorded as `indexed` without persisted chunks. Corrupt or
unreadable files are reported as `failed`; a PDF whose pages partly yield no
text layer is `partial` and names those pages (they are not called "blank" —
this build has no OCR, so "no extractable text" is as far as the evidence goes).
Resource caps are enforced before the reader loads anything:
`INGEST_MAX_FILE_MB` (default 50, `stat` only) and `INGEST_MAX_PAGES`
(default 500, PDF only — `.docx` has no page concept; needs pypdf, already
present). Encrypted PDFs fail explicitly; no password is guessed. A
binary-garbage guard fails a file that parses as bytes instead of text (missing
file readers). Files that are not supported document types are listed as
ignored (never silently invisible, never recorded). `python ingest_books.py`
exits 1 when any file failed.
3. Verify: `python scripts\check_chroma.py` (count grows; per-file status and
   chunk count verified against the store; printable ratio ~100%; see the row
   in section C).

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
| `env\Scripts\python.exe -m pytest scripts/test_phase7_planner.py -q` | Phase 7 conditional planner (35): complex detection + PLANNER_MODE gating, validate_plan branches (whitelist, duplicate/malformed ids, unknown/forward deps, unsatisfiable inputs, task cap, garbage never raises), defensive JSON parsing, planner_node fallback + only-`plan` contract, supervisor complex marking, adversarial capability | no | 0 | pytest: 35 passed |
| `env\Scripts\python.exe -m pytest scripts/test_phase8_execution.py -q` | Phase 8 dynamic plan execution (21): dispatcher dependency ordering + partial-order flexibility, failed/blocked deps never execute dependents, unsupported capability fails safe to no node, completed tasks never re-executed on re-entry, hard iteration cap (`MAX_PLAN_TASKS*(1+MAX_TASK_RETRIES)`) fail-safe, simple/multi routes bypass planner+dispatcher byte-identically, provenance/agent_answers preserved through dispatch, QA runs after planned execution | no | 0 | pytest: 21 passed |
| `env\Scripts\python.exe -m pytest scripts/test_gui1_events.py scripts/test_api_preview.py -q` | GUI-1 execution events + runner lifecycle (18); GUI-2 API endpoints, validation, event cursor, honest 404s (14) | no | 0 | pytest: 32 passed |
| `env\Scripts\python.exe -m pytest scripts/test_gui16_settings.py -q` | AI settings precedence/masking/persistence (7), readiness diagnostics vs a fake Ollama server (8), API endpoints (5) + one optional LIVE provider test (runs only with `OLLAMA_LIVE=1`) | no (fake server) | 0 | pytest: 21 passed, 1 skipped |
| `env\Scripts\python.exe -m pytest scripts/test_gui5_operations.py -q` | Operations dashboard: service snapshot states (running/unreachable/misconfigured/degraded/not_configured JSON, fail-soft probes, ERP bound-method serialization), routing.decided + qa.completed event record, agents-activity endpoint (live + failed runs), sequence endpoint (stage order, routing/qa stage inclusion, honest 404) | no | 0 | pytest: 14 passed |
| `env\Scripts\python.exe -m pytest scripts/test_gui12_plan_render.py -q` | GUI-12 planner-visualization data contract (20): plan serialization projection (task fields, bookkeeping pruned, long results trimmed + flagged), store plan column roundtrip + legacy-DB migration + corrupt-blob degradation, list rows omit plan, runner persists the executed plan + emits `execution.planned` (payload carries renderable tasks; precedes `execution.completed`), simple routes get no plan, failed runs still load with plan null, `/plan` endpoint (real plan / null for legacy+simple / honest 404), detail response backward-compatible with and without plan | no | 0 | pytest: 20 passed |
| `cd frontend && npm test` | GUI-12 PlanSection rendering (11, vitest+jsdom, mock data): multi-task order + objectives, backend statuses only, dependencies only where the plan has them (no fabricated edges), completed result summaries vs failed/blocked reasons, honest truncation, fallback-plan source note with recorded reasons, null/empty/legacy plan renders nothing, distinct "plan vs event log" card, llm plans unlabeled / default plans labeled | no | 0 | vitest: 11 passed |
| `env\Scripts\python.exe -m pytest scripts/test_agent_architecture.py -q` | Phase 4 agent-architecture contract (no BaseAgent, single factory, isolation) | no | 0 | 30 passed |
| `env\Scripts\python.exe -m pytest scripts/test_books_grounding.py -q` | Spec 2.1 RAG grounding (24 + 4 subtests): shape-based refusal detection (exact/paraphrased refusals, no references footer on refusal-only answers, high similarity never equals evidence, low-score floor note), the deterministic support gate (unsupported claims dropped + disclosed, mandate refusal when nothing is substantiated, supported answers and references intact), and the ADVERSARIAL polarity gates: negation reversal refused in both directions (agreeing negation preserved), number lifted from negated evidence refused, conflicting-passage claim kept only when its own evidence affirms it, partial with unsupported figure, no-evidence refusal — plus one explicitly labeled limitation test (cross-passage token splicing is still not detected; fixing it deterministically would over-refuse legitimate combined summaries; real entailment needs a verifier model, deliberately not added). Drives the real books_agent with faked query-engine responses and FIXED passages: no embeddings, no LLM, no network | no | 0 | pytest: 24 passed, 4 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_citation_integrity.py -q` | M3 citation/evidence integrity (22 + 4 subtests): supported answer cites its valid retrieved source; unsupported claim removed + disclosed; partial answer discloses the missing detail; refusals (exact / paraphrased / score-floor / empty) never carry a References footer; a model-generated filename not among retrieved sources is dropped — inline (even when every token of the sentence is otherwise supported) and as a fabricated References/Sources/Citations block (stripped, or degraded to refusal when it is the whole output); a fabricated page number is dropped by the metadata gate even though the number exists in the passage text, while the genuinely retrieved page survives; missing / malformed / junk source metadata yields an explicit "attribution unavailable" marker and never an invented filename (docx without page metadata cites file only, junk page labels are not rendered); two contradicting sources make the agent abstain while agreeing evidence is still answered; multi-claim answers cite each supporting passage exactly once; prompt-injection text inside a retrieved passage is dropped with the shape contract intact (no state escalation) and cannot hijack the footer; a retrieved-but-unused passage is never cited. Drives the real books_agent with faked responses: no embeddings, no LLM, no network | no | 0 | pytest: 22 passed, 4 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_verifier.py -q` | M4B/M4B-H bounded semantic verification + privacy (49 + 12 subtests): strict verifier JSON contract (four verdicts; malformed JSON, extra/missing fields, invalid enums, unknown evidence ids and non-string reasons all rejected as `unavailable`), enforceable per-claim timeout via a daemon-thread wrapper (langchain_ollama 1.1.0 exposes no timeout parameter — verified in installed source), 6-claim cap + zero retries + 30 s total deadline + evidence-length bounding, D1 relation reversal and D2 numeric conflict removed with conflict disclosure when enabled, paraphrases kept only when verified (still subject to the deterministic gate), per-claim evidence isolation, malformed/provider/timeout fail-closed (all-unavailable refusals are marked "semantic verification unavailable"; verified negatives are not), footer rebuilt from RETAINED claims only, verifier reasons never reach the user, injection-framed untrusted-evidence prompts, model-emitted References/Sources/Citations blocks proven unable to override the validated footer (stripped before handling; removal path never cites), the G1 regression inspects the ACTUAL LLM context (absolute file_path absent, page_label citation metadata intact, stored node metadata untouched), and `BOOKS_VERIFIER=off` proven byte-identical to M3 (verifier provably never called). M4B-H hardening: process-wide 6-permit in-flight budget with fail-fast `busy` (no thread spawned when exhausted), timeout-lifecycle/late-result-isolation/concurrent-permit-race tests, exactly-once permit release across timeout/exception/malformed/success outcomes, and request-semantics pins — non-streaming invoke on ollama (the installed ChatOllama streams by default, under which the httpx read timeout bounds only gaps between chunks) and `max_retries=0` on gemini (installed langchain_google_genai defaults to six SDK retries); both verified against installed source + a fake-model recording test, not a paid API call. Mocked verifier boundary: no model, no network, no quota | no | 0 | pytest: 49 passed, 12 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_m4c_attribution.py -q` | M4C claim-level evidence attribution (35, offline: fake nodes + scripted verdicts, no Ollama/Gemini/SearXNG): two claims tied to DIFFERENT sources asserted as an exact claim→evidence mapping on the per-claim footer (`[n] <ref> (verifier-supported: <claim>)` lines — the parenthetical reports the bounded verifier's judgment, never an answer-level footer implying shared support); one claim genuinely supported by two passages keeps BOTH refs on its own line; the verifier's per-claim `evidence_ids` narrow a claim's citations to the confirmed passage; a VALID id attached to the wrong claim is rejected exactly like a fabricated one (that claim removed + disclosed, the sibling keeps its own source); unknown evidence id fail-closed; per-claim contradiction scoping (reversed relationship removed with the conflict note while the sibling claim keeps its own citation); supported+unsupported claims in one answer; compound mixed support; model-authored citation block stripped; missing/junk filename+page metadata → explicit "attribution unavailable", never an invented ref; a claim removed after verification never leaves its citation behind; the adversarial battery — keyword overlap asserting an OPPOSITE relationship removed, correct entity with the wrong number removed (siblings keep their sources), evidence ids swapped between two claims rejected on the misattached claim only, a valid id carrying an insufficient verdict removed, partially overlapping evidence kept per-claim, a multi-passage claim narrowed to its one relevant passage, a removed claim's source never inherited, fabricated page/filename/table references in text dropped before any attribution, mixed supported+contradicted+unavailable answers; partial `verify_all` results (a missing sibling result counts as unavailable for exactly that claim — never a pass — and missing results for all claims remove them all), the real parser's `contradicted` verdict treated as a successful negative (no unavailable marker), and the rendered per-claim line pinned to the `verifier-supported` qualifier; refusal / unavailable / malformed / timeout outcomes; `BOOKS_VERIFIER=off` byte-compatibility (default unchanged); local-path privacy (absolute paths never in LLM context or answer); stored source metadata + persisted Chroma content untouched | no | 0 | pytest: 35 passed |
| `python scripts\benchmark_verification.py [--check]` | M4B-H/M4C deterministic verification benchmark (38 cases, offline): every case carries an explicit expected verdict, evidence, rationale, and three outcomes — recorded deterministic baseline (BOOKS_VERIFIER=off), scripted verifier verdict (mock layer; NOT model quality), and integrated design target. Verdicts may be scripted per claim (`mock_verifier_per_claim` + `mock_evidence_ids`, id-validated exactly like the production parser: a mis-attached or unknown id degrades to unavailable(malformed), never rendered as a pass), and 6 of the 7 M4C `claim_level_*` cases additionally declare an `expected_mapping` (claim→refs) that the runner parses from the verified layer's actual per-claim footer lines and requires to match EXACTLY — a valid answer-level footer can never satisfy it. Through the REAL books_agent with a faked engine + scripted `verify_all`: all 38 baseline MATCH and all 38 integrated MATCH; 17/38 rows are annotated `baseline-gap*` (the deterministic layer alone misses the target — the known D1/D2-style surface that only enabling `local` closes). Includes the §5 quality report (raw counts + denominators, off vs verified: e.g. unsupported accepted 5/9 -> 0/9, supported falsely rejected 2/15 -> 3/15, contradictions 3/9 -> 9/9 with numeric 2/4 -> 4/4 and relational 0/3 -> 3/3, malformed output -> fail-closed 0/3 -> 3/3, timeout/unavailable -> fail-closed 0/4 -> 4/4, unavailable never shown as successful 0/4 -> 4/4), the M4C row `claim-level mapping exact (verified layer): 6/6`, and deterministic budget probes (cap 6 / zero-budget deadline). `--check` exits 1 on baseline drift (CHANGED) or an integrated-layer GAP — never on the informational baseline-gap annotation, and never treats scripted verdicts as model accuracy | no | 0 | table + exit 0 |
| `python scripts\eval_verifier_local.py` | OPTIONAL M4B-H local-model experiment (installed `qwen3:0.6b` via Ollama; no downloads, no paid quota; never in the regression suite): all benchmark cases in the JSON (38 since M4C) through the real bounded verifier path at the PRODUCTION 10 s claim timeout (deliberately not loosened), recording expected vs actual verdict, parse failures, false accepts/rejects, per-claim latency (cold vs warm), timeout count, total elapsed, live verifier-thread accounting after the run, and a post-run probe. Results vary run to run: recorded runs had 17 and 26 of 31 calls exceed the 10 s bound with 0 parse failures and 0 false accepts among completed calls in both — small samples, not statistically conclusive; latency (not verdict quality) is the binding constraint. Exits 2 with a clear message when the local runtime is down — never substitutes a paid model | local Ollama | 0 | per-case table + summary |
| `python scripts\check_dependencies.py` | every required package imports | no | 0 | `ALL_DEPENDENCIES_OK` |
| `python scripts\check_chroma.py` | persisted collection: non-empty, prose sample, and manifest/store integrity (per-file chunk counts via `file_path` metadata, indexed-with-zero-chunks, failed/partial records, legacy entries, entries whose file is gone, stored vectors with no manifest entry). Read-only; labels a check UNAVAILABLE when the index lacks the metadata for it | no | 0 | counts by status + per-file ok/partial/failed/legacy lines; exit 0 clean, 1 on integrity error |
| `env\Scripts\python.exe -m pytest scripts/test_ingest_integrity.py -q` | M2 ingestion integrity (27): unsupported files reported as ignored, healthy PDF/DOCX indexed with store-verified chunk counts, corrupt PDF and "reader returned no documents" failed (never indexed), zero-text PDF reported as no-extractable-text (not "blank"), mixed PDF partial with the unextractable page named and the readable page still usable, oversized and page-limit rejects before the reader is called, encrypted PDF explicit failure, idempotent re-run (no duplicate chunks, no manifest drift), failed files retried, failed replacement keeps the previous vectors, changed file replaces its own stale chunks, legacy v1 manifest migrated and store-verified (or repaired when its vectors are missing), corrupt manifest tolerated, files missing from disk reported not pruned, and the check_chroma diagnostic's own exit codes. Temp dirs + `MockEmbedding`: no Ollama, no Gemini, no network, no quota, real knowledge base untouched | no | 0 | pytest: 27 passed |
| `python scripts\verify_graph.py` | **LIVE** smoke test: real graph, 3 scenarios | SearXNG + web | **~3 calls** | three scenario blocks; refusal on out-of-corpus |
| `python scripts\test_phase1_e2e.py` | **LIVE** spec steps 3 & 5 end to end | SearXNG + web | **~2 calls** | `STEPS_3_5_REAL_OK` |
| `python scripts\check_books_agent.py` | one real books answer with provenance | Ollama embeds | **1 call** | answer + `SOURCES: ['books']` |
| `python scripts\probe_books_prompt.py` | captures the exact prompt sent to the LLM by the query engine | Ollama embeds | **1 call** | answer + captured prompts |
| `python scripts\demo_rewrite_quality.py` | before/after search-quality demo of the rewrite step | SearXNG + web | **1 fast call** | raw vs rewritten result lists |

### M4C — what the tests prove (deterministic vs scripted vs live)

Claim-level evidence contract (verification-enabled mode only,
`BOOKS_VERIFIER=local`; `off` stays the byte-identical M3 default):

1. Every retained claim keeps its own evidence set — the deterministic
   covering set for that sentence, narrowed to the verifier's per-claim
   `evidence_ids` when it confirms fewer passages.
2. Evidence IDs resolve only to nodes actually supplied FOR THAT CLAIM.
   `_parse_output`'s per-claim whitelist rejects unknown ids, and an id
   that is valid but belongs to another claim is rejected the same way
   (malformed → fail-closed removal + fixed disclosure). Whitelisting
   and strict JSON validation are unchanged.
3. The trusted footer is built only from each retained claim's evidence:
   `References:` followed by one numbered line per claim,
   `[n] <ref>[; <ref>...] (verifier-supported: <claim>)`. The
   parenthetical is deliberately ATTRIBUTED wording: it reports the
   bounded verifier's per-claim judgment on the shown passages — a
   heuristic/small-model verdict — not independently established
   semantic entailment. Refs are the same
   `_source_ref` strings as the M3 answer-level footer (validated
   file/page metadata only — no model-authored filenames, pages, paths,
   or quotes; missing metadata → explicit "attribution unavailable").
   Multiple passages genuinely supporting one claim all appear on that
   claim's line; a passage supporting claim A never appears on claim B's
   line.
4. Claims the verifier contradicts, cannot confirm, or cannot process
   (timeout/malformed/busy/unknown id) are REMOVED with the existing
   fixed disclosures; their citations leave with them; verifier reasons
   and late results never reach users or mutate a returned answer.
5. Contradiction handling stays claim-scoped and separate from
   attribution: a valid citation can still contradict its own claim.

Covered by tests: reversed subject/object relations, conflicting values
for the same quantity under matching conditions, differing
units/periods/populations/conditions deliberately NOT treated as
contradictions, negation and conditional statements, compound sentences
with mixed support (sentence removed whole — conservative), multiple
passages vs claims (per-claim, including a similar-number/entity passage
that does not support the claim), missing metadata, fabricated model
citation blocks, refusals, and unavailable/timeout/malformed outcomes.
Unresolved by design (documented, not hidden): no clause-level splitting,
no general NLI or arithmetic engine, and the deterministic gate remains
token/metadata coverage — not semantic entailment; real entailment
quality is only observable in the optional live-model experiment.

Three layers must not be conflated: (a) the DETERMINISTIC grounding
gate — token/metadata coverage, always on; (b) the benchmark's
SCRIPTED verdicts — they test how the pipeline CONSUMES verdicts and
never measure model accuracy; (c) ACTUAL semantic entailment, which no
offline test establishes and which the small-sample optional live
experiment (`scripts/eval_verifier_local.py`) measures only crudely
and has not been re-run as part of the regression.

Commands (all zero-quota, offline):

```bat
env\Scripts\python.exe -m pytest scripts/test_m4c_attribution.py -q   :: 35 passed
env\Scripts\python.exe -m pytest scripts/test_verifier.py -q          :: 49 passed, 12 subtests
python scripts\benchmark_verification.py --check                      :: 38/38 baseline + integrated MATCH, mapping 6/6, exit 0
```

`test_m4c_attribution.py` asserts the ACTUAL claim→evidence mapping and
user-visible footer lines (not merely that a filename appears); the
benchmark's scripted verdicts measure how verdicts are consumed by the
pipeline (never model quality); `scripts/eval_verifier_local.py` remains
the only real-model evaluation and is NOT part of the regression.
Timeout, concurrency (6-permit), zero-retry, 6-claim/2000-char/30 s
budgets, path-privacy, and model-latency limitations are unchanged from
M4B-H (see the M4B rows above).

### Regression order after any code change

```bat
python -m py_compile main.py ingest_books.py app\*.py app\agents\*.py app\tools\*.py app\erp\*.py scripts\*.py
python scripts\test_phase1_routing.py
python scripts\test_phase2.py
python scripts\test_feasibility_degradation.py
python scripts\test_rewrite_step.py
python scripts\test_phase3_erp.py
env\Scripts\python.exe -m pytest scripts/test_phase4_workflow_qa.py scripts/test_phase5_qa_node.py scripts/test_phase6_routing_llm.py scripts/test_agent_architecture.py scripts/test_gui1_events.py scripts/test_api_preview.py scripts/test_gui16_settings.py scripts/test_gui5_operations.py scripts/test_phase7_planner.py scripts/test_phase8_execution.py scripts/test_books_grounding.py scripts/test_ingest_integrity.py scripts/test_citation_integrity.py scripts/test_verifier.py scripts/test_m4c_attribution.py -q
```

Full combined regression at the Phase 8 dispatcher commit:
**184 passed, 1 skipped, 14 subtests** (the skip is the opt-in
`OLLAMA_LIVE=1` live provider test). Phase 8 adds 21 of its own tests
(see the suite row above); all other suites are green unchanged.

Current combined regression (M4C, re-run 2026-10-16):
**383 passed, 1 skipped, 34 subtests** for the suite set in the command above
plus `scripts/test_gui12_plan_render.py` (GUI-12) — pre-M4B baseline 299,
M4B-H baseline 348 (both historical); the M4B/M4B-H suite is the 49 and the
M4C suite the 35 of those tests listed in section C. Verification is
opt-in (`BOOKS_VERIFIER=local`; the default `off` keeps byte-identical M3
behavior); the deterministic benchmark (38 cases, three layers per case)
is green on both drift gates (all 38 baseline MATCH + all 38 integrated
MATCH), including the M4C claim-level mapping check (6/6 exact), with
17 annotated baseline-gaps that only the verifier path closes, and the
optional qwen3:0.6b experiment at the production 10 s bound (0 parse
failures and 0 false accepts among completed calls; 17–26 of 31 calls
across recorded runs exceed the bound — model latency, not verdict
quality, is the binding constraint on this hardware) is recorded in
`scripts/eval_verifier_local.py`. That experiment is the repository's
only live-model evidence: a small-sample latency run, NOT a measurement
of semantic accuracy — no offline result in this file establishes
real-model entailment quality.

Known environment-dependent flake in `test_phase4_workflow_qa.py::
TestEndToEndFeasibility::test_full_feasibility_workflow`
(predates Phase 8; NOT a dispatcher bug): with a LIVE local Ollama server
running, the qwen3:0.6b planner produces a valid but minimal LLM plan
(often research+synthesis without a books task) for the e2e question, so
the dispatched run differs from the old always-research-books-feasibility
chain the test asserts. With Ollama stopped (the deterministic zero-quota
mode this suite is designed for — planner LLM fails instantly →
deterministic fallback plan), the suite is 100% green. Long-term fixes
track under MASTER_PLAN Phase 12 (planner quality/verification).

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
