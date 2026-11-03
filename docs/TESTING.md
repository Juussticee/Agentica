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
| `env\Scripts\python.exe -m pytest scripts/test_phase4_workflow_qa.py scripts/test_phase5_qa_node.py scripts/test_phase6_routing_llm.py -q` | Phase 4 workflow/QA suite (30), Phase 5 production QA node (26) — objective-level checks (2026-10-29): completed-but-prerequisite-did-not-complete, a task recorded blocked although every prerequisite completed (a genuine dependency failure is not double-flagged), task left running, an approval request recorded as a completed result, and indistinguishable results across DIFFERENT capabilities (warning only: same-capability and short results are never compared); equivalent entity naming and insufficient evidence produce no finding; single-step/non-planned QA unchanged; plus Phase 6 LLM routing fallback (14) — all deterministic | no | 0 | pytest: 70 passed, 18 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_phase7_planner.py -q` | Phase 7 conditional planner (35): complex detection + PLANNER_MODE gating, validate_plan branches (whitelist, duplicate/malformed ids, unknown/forward deps, unsatisfiable inputs, task cap, garbage never raises), defensive JSON parsing, planner_node fallback + only-`plan` contract, supervisor complex marking, adversarial capability | no | 0 | pytest: 35 passed |
| `env\Scripts\python.exe -m pytest scripts/test_phase8_execution.py -q` | Phase 8 dynamic plan execution (38): task-scoped execution (2026-10-29) — a planned sub-request runs as its own objective while the original request travels beside it, template objectives keep the request, research always searches the request and an objective that looks like a web request does NOT authorize research, synthesis answers the original request, single-step routes gain no scoping key; 2026-10-29 task-output honesty (a node returning no/blank answer is `failed` with "the task produced no output", its dependent is blocked, QA flags the incomplete plan; a real answer is still completed with its result) + continuation progress reporting (`continue_plan(on_progress=)` reports the released plan and after every task; a raising callback never aborts the plan) + `CAPABILITY_AGENT_KEYS["synthesis"]` matches the channel `synthesis_node` writes; then dispatcher dependency ordering + partial-order flexibility, failed/blocked deps never execute dependents, unsupported capability fails safe to no node, completed tasks never re-executed on re-entry, hard iteration cap (`MAX_PLAN_TASKS*(1+MAX_TASK_RETRIES)`) fail-safe, simple/multi routes bypass planner+dispatcher byte-identically, provenance/agent_answers preserved through dispatch, QA runs after planned execution; 2026-10-28 synthesis node — the terminal task answers from completed results only (failed tasks excluded), provider failure returns those results verbatim, no result is reported honestly, and the feasibility route still produces the feasibility study | no | 0 | pytest: 26 passed |
| `env\Scripts\python.exe -m pytest scripts/test_gui1_events.py scripts/test_api_preview.py -q` | GUI-1 execution events + runner lifecycle (18); GUI-2 API endpoints, validation, event cursor, honest 404s (14) | no | 0 | pytest: 32 passed |
| `env\Scripts\python.exe -m pytest scripts/test_gui16_settings.py -q` | AI settings precedence/masking/persistence (7), readiness diagnostics vs a fake Ollama server (8), API endpoints (5) + one optional LIVE provider test (runs only with `OLLAMA_LIVE=1`) | no (fake server) | 0 | pytest: 21 passed, 1 skipped |
| `env\Scripts\python.exe -m pytest scripts/test_diagnostics_core.py -q` | Increment 2 shared diagnostic core (21 + 13 subtests, 2026-10-28): `provider_readiness()` is config-only (network tripwire fails the test on any socket attempt) and is the ONE rule behind both `/api/v1/health`'s `ai` block and the `/services` `ai_provider` rollup — they agree on every settings combination (incl. the historical unset-provider disagreement) while keeping their exact response schemas (`{provider,state,what_will_be_tested,state_hint}` and `{provider,state,settings_url,note}`); `fetch_ollama_tags` contract (HTTP status kept for 4xx/5xx, `AI_*` category, timeout passthrough, never raises); the `/services` Ollama row makes exactly ONE `GET /api/tags` per poll (the old `_http_status` probe path is patched to raise if reactivated) with running/degraded/unknown/unreachable states; the same `model_present` rule in both surfaces; `/ai/test` pass-through schema unchanged | no (pure functions + local fake HTTP server / closed loopback port) | 0 | pytest: 21 passed, 13 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_gui5_operations.py -q` | Operations dashboard: service snapshot states (running/unreachable/misconfigured/degraded/not_configured JSON, fail-soft probes, ERP bound-method serialization), routing.decided + qa.completed event record, agents-activity endpoint (live + failed runs), sequence endpoint (stage order, routing/qa stage inclusion, honest 404) | no | 0 | pytest: 14 passed |
| `env\Scripts\python.exe -m pytest scripts/test_gui12_plan_render.py -q` | GUI-12 planner-visualization data contract (20): plan serialization projection (task fields, bookkeeping pruned, long results trimmed + flagged), store plan column roundtrip + legacy-DB migration + corrupt-blob degradation, list rows omit plan, runner persists the executed plan + emits `execution.planned` (payload carries renderable tasks; precedes `execution.completed`), simple routes get no plan, failed runs still load with plan null, `/plan` endpoint (real plan / null for legacy+simple / honest 404), detail response backward-compatible with and without plan | no | 0 | pytest: 20 passed |
| `env\Scripts\python.exe -m pytest scripts/test_execution_instrumentation.py -q` | W2 structured execution events + provenance (35, offline: `requests`/`_crawl` faked, scripted LLM, faked query engine — no network, no Ollama, no Chroma): event-model constants + `EVENT_SCHEMA_VERSION`; the context-isolated in-process sink (call order, payload version stamp, restored on exception and for nested installs, no-op outside a run, **execution isolation**: barrier-synchronized overlapping sinks and two overlapping `runner.run()` calls each receive only their own events, leave no stale sink, and an emit after the runs reaches nothing — verified to FAIL against the pre-W2.1 module-global sink); a raising sink and a failing event write both leave the execution COMPLETED with its answer intact (bookkeeping never corrupts the result) and, since W2.1, count on `ExecutionStore.event_write_failures` which degrades the EXISTING `storage` row of `GET /api/v1/services` (`check_storage`) without failing the run; `executions.sources` persisted and returned (agent names + research URLs), additive `sources` column migration with a legacy pre-migration database loading as `sources=None`, corrupt JSON column degrading to `None`; operation events recorded as they happen — books `retrieval.completed` (file/page/score identifiers, never passage text), `recovery.outcome` for all four outcomes (denied / succeeded / rejected+reason / absent on non-refusal runs), `fallback.decided` for BOTH the allowed and the local-only-denied branch with its diagnostic code, `tool.completed`/`tool.failed` for search and crawl (count/duration/host/size); privacy assertions — no query, no passage or answer text, no secret-shaped strings, no raw exception message (category only). **Version stamp scope:** the payloads SUNK through the instrumented sink (`retrieval.completed`, `tool.*`, `recovery.outcome`, `fallback.decided`) carry `schema_version`; the RUNNER's own lifecycle events (`execution.*`, `agent.*`, `routing.decided`, `qa.completed`) are written directly to the store and carry NO `schema_version` — a consumer must not assume the field is present on every event | no (fakes only) | 0 | pytest: 35 passed, 2 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_routing_policy.py -q` | Routing policy (24, 2026-10-28 Priority A; 2026-10-29 acknowledgements, code authoring): table-driven routes for conversation / general-knowledge / local knowledge / requested research / unsupported (incl. "Do it", "Create a report", "What did I tell you earlier?"); code/draft authoring that is answered IN the reply ("Write a Python function that parses a CSV file", "Write a SQL query that joins orders and customers") routes to general — the model answers it, no agent runs, no plan is built, provenance stays empty — while operations on the user's machine ("Write a Python script to a file on disk", "Create a report", "Send an email", "Delete the temporary files") stay unsupported and never consult a model; short acknowledgements ("thanks, that helps", "great, thank you so much", "cheers, got it") are answered as conversation ("You're welcome", no agent, no sources) while one substantive word ("Thanks, how do I sort a list in Python?") keeps its real route; the old silent research fall-through is gone; conversation and unsupported touch zero tool seams (boom stubs), general answers from a faked model with EMPTY provenance and fails soft when the provider is down; every earlier routing pin (deictic, identity, current-info, local-only, greeting) preserved; ROUTING_MODE=llm label contract intact | no (deterministic rules + faked model seam) | 0 | pytest: 24 passed, 88 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_research_approval.py -q` | Outbound research approval (19, 2026-10-28, Priority B): module-wide search/crawl blocking seams prove zero external calls; nothing executes before approval (direct route, books->research fallback, plan-dispatched research task), the gated reply names the proposed action + why + how to authorize exactly this operation; authorization is per-request and never carried between runs; a declining/local-only message denies; local RAG and conversation need no prompt; general questions never research; absent/unknown state fails closed; `fallback_decision`, graph topology and QA contracts unchanged | no (seams raise on any call) | 0 | pytest: 19 passed |
| `env\Scripts\python.exe -m pytest scripts/test_approval_lifecycle.py -q` | GUI-10 approval lifecycle (58 + 10 subtests, 2026-10-29; plan continuation + interrupted-resume recovery + durable resume lease): store state machine — fingerprint-idempotent create/reuse, approve/reject, unknown id, repeat-of-same-decision idempotence vs conflicting-decision 409, expiry → 410, FK-bound execution rows, persistence across store reopen, competing decisions under the store lock, unsupported operation/malformed payload rejected; gate integration — pending request created with the approval-pending answer carrying its id, approval authorizes ONLY the bound payload, missing/mismatched/malformed approvals fail closed, rejection never executes, a later unrelated message cannot approve, books fallback creates the request while local RAG needs none, approval store failure = denial not execution; runner resume — approved runs the fake search/crawl seams exactly once with the bound question, rejected finalizes with the honest local-only answer, pending/expired/conflict refused BEFORE any seam; API contract — list/detail/approve/reject/resume with 404/409/410/422 shapes and idempotent repeats; one full-graph books-fallback test; plan continuation — approving the research task of a paused plan runs the remaining tasks through the dispatcher (completed tasks not re-run, dependent synthesis consumes the approved answer, a later failure is reported as failed/incomplete_plan, a second research task pauses for its own approval, a retried resume adds no work; interrupted continuation — a crash after the search and before the continuation is recovered by the retry without repeating the search, resuming preserves the task objective as plan data while the approved search stays the original request; a crash AFTER one continuation task finished resumes from that task (it is never re-run, its result is replayed to the dependent, the pending task runs) while a task persisted as `running` closes the run `failed/resume_interrupted` instead of replaying it, an already-claimed approval with no persisted result closes the run `failed/resume_interrupted`, a held `resume_claim` makes a concurrent resume a denied no-op, and the DURABLE claim is one atomic conditional UPDATE on the execution row — a second store connection on the SAME SQLite file is excluded while the first holds it, the claim is released on exit so a later retry (from any connection) can run, an EXPIRED claim is taken over (bounding a crashed holder) while the stale owner's release never clears the new holder's claim, and a legacy pre-lease database gains both columns on open without touching its rows; which is why the lease never authorizes repeating an external operation: that guarantee stays `approvals.executed_at` plus the persisted released plan; 2026-10-29 multi-gate recovery — a crash after the SECOND gate's research is recovered through the persisted released plan without repeating either search or re-running the first task, a spent second approval whose release was never persisted (plan still records the task approval-blocked) closes `failed/resume_interrupted` instead of trusting the first gate's result, and a crash AFTER the continuation finished finalizes `completed` from the persisted plan/answer without dispatching anything again). Module-wide search/crawl boom stubs + fake HTTP loopback only → zero external requests | no (seams raise on any call) | 0 | pytest: 54 passed, 10 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_multi_approval_api.py -q` | Multi-approval plan through the REAL API (4, 2026-10-29): one execution with TWO research tasks, each needing its own decision — initial pause at the first gate (status `waiting_for_approval`, no search, task statuses failed/blocked/blocked), resume before the decision is 409, approve+resume runs task 1 exactly once and pauses AGAIN for a DISTINCT second request (task 2 still failed, nothing searched), replayed approve is idempotent and a replayed resume is a denied no-op that appends only one `approval.finalized` note, then approve+resume runs task 2 and the synthesis, ending `completed` with both results on their own tasks, empty `qa_flags`, no pending request and a timeline of 2 created / 2 executed / 1 completed; rejecting the second request closes the run `failed/approval_rejected` with no second search; an unknown approval is 404; a crash between the gates (500 from the resume endpoint after the second approved search ran) is then recovered by the same endpoint from the persisted plan — both searches stay at two calls, synthesis runs once and the run ends `completed` with `recovered: true`. Only two seams are faked — `get_llm` (answers the REAL planner and synthesis prompts, raises on any other prompt) and the post-gate research node; SearXNG/crawl are stubbed to raise | no | 0 | pytest: 4 passed |
| `env\Scripts\python.exe -m pytest scripts/test_coding_workspace.py scripts/test_coding_patch.py -q` | Workspace boundary + patch engine (53, 2026-10-30, offline): one configured root (`CODING_WORKSPACE_ROOT`, absolute, read at call time; unset = `WorkspaceUnavailable`, relative/missing root refused); traversal (`..`, absolute, NUL, empty), a REAL reparse point (POSIX symlink or Windows junction, no elevation) resolved and refused outside the root, denied credential files (`.env*`, `*secret*`, `id_rsa*`, `*.pem/key/db`) and dependency/VCS directories, text-extension allowlist, size caps, `redact_secrets()` (quoted values with spaces, `sk-`/`AIza`/`ghp_`/PEM blocks) and a token-named source file that must stay readable; patch building leaves the file untouched, refuses an unchanged file, over-cap and duplicate-target patches, computes before/after hashes + unified diff, static-validates the PROPOSED content (Python compiled in memory, JSON parsed, other types `not_validated`) and records a syntax failure instead of offering it; `validate_payload` fails closed on a malformed/injected/hash-mismatched payload; `apply_patch` refuses the WHOLE patch when any target changed, verifies the digest + workspace identity, and `replace_text` is hash-guarded and atomic per file | no | 0 | pytest: 53 passed |
| `env\Scripts\python.exe -m pytest scripts/test_coding_agent.py -q` | The `coding` graph node (14, 2026-10-30): inspection names the files and path:line evidence it used, is READ-ONLY (no write, no approval request) and is never counted as a patch; patch mode prepares a diff + approval payload bound to the workspace, the question and the pre-change hashes WITHOUT touching the file, preparation is idempotent (one pending request), `find` text that is absent blocks the patch, the model cannot reach a file it was not shown, an unusable model reply blocks it, a failed offline validation is reported and NOT offered for approval, and with no approval scope nothing is registered or written; a denied path is reported not read, a disabled workspace answers honestly, and a request naming no source file asks for a target instead of guessing | no (scripted `get_llm`) | 0 | pytest: 14 passed |
| `env\Scripts\python.exe -m pytest scripts/test_coding_execution.py -q` | Workspace patch through the REAL runner (14, 2026-10-30): create+run routes to `coding`, prepares the patch and pauses `waiting_for_approval` with the diff in the answer and no write; resume before the decision is refused (file untouched); approve+resume applies it, the execution ends `completed` with the validation caveat and the `execution.approval_completed` event; a replay is `denied` and never rewrites; a file changed after review → `failed/stale_patch` with the owner's edit preserved; a workspace disabled after approval → `failed/workspace_unavailable`; rejection closes `failed` without writing; a second execution prepares no new pending request once the file is fixed; INTERRUPTED APPLY RECOVERY — when the claim was consumed and the terminal write was lost, a patch already on disk is confirmed `completed` from the content hashes, nothing-written closes `failed/patch_apply_unverified`, and an unprovable state closes `failed/patch_apply_conflict` naming the files — recovery never writes; routing separation (workspace enabled gates the coding route, web/manual cues and plain code questions keep their own routes, disabled = never coding) | no (scripted `get_llm`, temp workspace) | 0 | pytest: 14 passed |
| `env\Scripts\python.exe -m pytest scripts/test_coding_api.py -q` | Workspace patch through the REAL API (4, 2026-10-30): `POST /executions` pauses at the approval gate with the diff and no file change; `GET /approvals` shows the pending `workspace_patch` bound to that execution and question; resume before a decision is 409 and writes nothing; approve+resume applies the patch (`outcome=executed`), the execution reads back `completed` with the real answer and no pending request remains; a replayed resume is `denied` and does not rewrite; reject+resume closes `failed` with nothing written; a file changed after review is refused at resume (409-free `failed` + `failed/stale_patch` on the execution, owner's edit preserved); with `CODING_WORKSPACE_ROOT` unset the request is never routed to coding and no approval exists. Real router, graph, approval lifecycle, store, runner and write path; only `get_llm` (raises on any unexpected prompt) and the workspace root are faked | no | 0 | pytest: 4 passed |
| `env\Scripts\python.exe -m pytest scripts/test_data_analysis.py -q` | Dataset reader + deterministic analysis (29, 2026-10-30, offline): reading goes THROUGH the workspace boundary (`..` and absolute escape refused, non-dataset extension refused, size cap hit before reading, row cap recorded in the coverage note), typing by inference (quoted `1,000` is an integer, a text column stays text, a blank header becomes `column_N`, duplicate headers get unique names), ragged rows preserved (overflow joined into the last cell, short rows padded), blank/all-empty rows skipped, header-only file has columns and zero rows, TSV delimiter; `profile` exact figures (non-null/missing/distinct/min/max/mean/sum, and NO numeric aggregate for text columns), `value_counts` sorted with counts and missing values excluded, `group_agg` sum/mean/count with a group whose measure is entirely missing OMITTED and disclosed (never shown as 0), text measure refused for sum, group cap reported, `filter_count` numeric and text comparisons with share over non-missing values; every bad parameter (unsupported op, unknown column, bad operator/limit, comparison placed in `op`) refused with a reason, and `run_steps` records refusals instead of raising. Temp workspaces only — no model, no network, no Chroma | no | 0 | pytest: 29 passed |
| `env\Scripts\python.exe -m pytest scripts/test_data_agent.py -q` | The `data` specialist (19, 2026-10-30): computes the profile AND the model-chosen operation (`| north | 1300 | 2 |` is computed, not paraphrased), the provider never receives a data row (schema only — asserted against the actual prompt), interpretation is labelled "not a new calculation", answer states read-only; provider down still answers with the computed profile and says no further analysis was planned; unusable JSON, an invented op, a text column asked for a sum and an unknown column are each REPORTED as not run; absent/escaped/missing dataset and a disabled workspace fail honestly; two named datasets analyse the first and say so; routing sends a named `.csv` to `data` only with a workspace configured and leaves general/research/coding untouched; a planned `data` task executes through the real dispatcher into its own channel, its dependent `synthesis` task waits, then composes an answer carrying the COMPUTED figure (1300) — real information exchange, not just node calls; two full `GraphRunner` runs (success with empty QA flags, failed read reported honestly) with search/crawl stubbed to raise | no (scripted `get_llm`, temp workspace) | 0 | pytest: 19 passed |
| `env\Scripts\python.exe -m pytest scripts/test_agent_registry.py -q` | The agent registry cannot drift (15, 2026-10-30): every entry declares its full contract (kind, node, entry, routes, capability, tools, four permission dimensions, reads/writes, what it REJECTS, model use, tests); every `entry` resolves to a real callable; the registry's node set is EXACTLY the graph's node set; the planner whitelist, `CAPABILITY_NODES`, `_TASK_NODE_FNS` (whose only extra key is the direct-route `feasibility`) and `CAPABILITY_AGENT_KEYS` all agree with it, with each capability's result channel equal to the agent's own `agent_answers` key; declared test files and skill documents exist; the permission audit holds (network = research only, writes = coding + feasibility, approval = coding + research); no node is registered twice and no two agents share an entry point | no | 0 | pytest: 15 passed |
| `env\Scripts\python.exe -m pytest scripts/test_execution_lifecycle.py -q` | Lifecycle invariant (9, 2026-10-28, adapted from the Mohamad AI OS reference): `ExecutionStore.reconcile_interrupted()` drives orphaned `queued`/`running` rows to a terminal `failed` status with a structured `category=interrupted` error and exactly one closing `execution.failed` event; append-only guarantees (answers/routes/plans and the existing timeline never rewritten, closing event id = `MAX(event_id)+1` for that execution so it cannot collide with the runner's numbering), terminal rows untouched, idempotence; the `app.api.server.get_store()` startup hook runs it on first open against a patched `DEFAULT_DB_PATH` (the real `storage/` database is never touched) and stays fail-soft when reconciliation itself raises. Temp SQLite files only — no graph, no network, no LLM | no | 0 | pytest: 9 passed |
| `cd frontend && npm test` | GUI-12 PlanSection rendering (11, vitest+jsdom, mock data): multi-task order + objectives, backend statuses only, dependencies only where the plan has them (no fabricated edges), completed result summaries vs failed/blocked reasons, honest truncation, fallback-plan source note with recorded reasons, null/empty/legacy plan renders nothing, distinct "plan vs event log" card, llm plans unlabeled / default plans labeled | no | 0 | vitest: 11 passed |
| `env\Scripts\python.exe -m pytest scripts/test_agent_architecture.py -q` | Phase 4 agent-architecture contract (no BaseAgent, single factory, isolation) | no | 0 | 30 passed |
| `env\Scripts\python.exe -m pytest scripts/test_books_grounding.py -q` | Spec 2.1 RAG grounding (24 + 4 subtests): shape-based refusal detection (exact/paraphrased refusals, no references footer on refusal-only answers, high similarity never equals evidence, low-score floor note), the deterministic support gate (unsupported claims dropped + disclosed, mandate refusal when nothing is substantiated, supported answers and references intact), and the ADVERSARIAL polarity gates: negation reversal refused in both directions (agreeing negation preserved), number lifted from negated evidence refused, conflicting-passage claim kept only when its own evidence affirms it, partial with unsupported figure, no-evidence refusal — plus one explicitly labeled limitation test (cross-passage token splicing is still not detected; fixing it deterministically would over-refuse legitimate combined summaries; real entailment needs a verifier model, deliberately not added). Drives the real books_agent with faked query-engine responses and FIXED passages: no embeddings, no LLM, no network | no | 0 | pytest: 24 passed, 4 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_books_grounding_paraphrase.py -q` | Grounding paraphrase regression (11): the INVESTIGATED delta-v case — real indexed chunk + real raw answer — keeps BOTH supported sentences (the regression of the fixed false refusal); supported paraphrase wording over the source (orbital mechanics ← orbital kinetics, propulsion systems ← propulsion engine, required/needed morphology) admitted at gate and agent level with its validated References footer; directly supported wording unchanged; mixed supported/unsupported keeps exactly the supported sentence and discloses the rest; fluent fabrication and empty evidence still refuse with no footer. Deterministic only — fixed text through the real `_grounded_repair` and the real `books_agent` with a faked query engine: no embeddings, no LLM, no network, no Chroma | no | 0 | pytest: 11 passed |
| `env\Scripts\python.exe -m pytest scripts/test_books_grounding_adversarial.py -q` | Adversarial battery for the paraphrase-family + stemming gate (33 + 18 subtests), sections A–G: verified delta-v positive control; each synonym family admits an evidenced wording swap but NOT an unsupported claim riding on a family sibling (incl. estimate→determine precision, the change family not extending to motion verbs, modify inflections); mixed supported/unsupported; polarity gates across a family swap in both directions (plus same-polarity not over-refused); numeric integrity (changed value/decimal, wrong unit, sign word, a family word never carries a changed number, no numeric family members); metadata/citation/stemming (fabricated page dropped vs retrieved page kept, metadata-only tokens cannot ground content, `speed` untouched by the -ed rule, the plural `pages` stem still gated by the page number); evidence-boundary PINS — cross-passage and single-node cross-sentence splices admitted as documented limitations, and the family-3 subject swap pinned as a known limitation awaiting the directional-alias design note (`docs/GROUNDING_DIRECTIONAL_ALIASES_DESIGN.md`). Fixed strings + the real production grounding functions: no model, no network, no Chroma | no | 0 | pytest: 33 passed, 18 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_citation_integrity.py -q` | M3 citation/evidence integrity (22 + 4 subtests): supported answer cites its valid retrieved source; unsupported claim removed + disclosed; partial answer discloses the missing detail; refusals (exact / paraphrased / score-floor / empty) never carry a References footer; a model-generated filename not among retrieved sources is dropped — inline (even when every token of the sentence is otherwise supported) and as a fabricated References/Sources/Citations block (stripped, or degraded to refusal when it is the whole output); a fabricated page number is dropped by the metadata gate even though the number exists in the passage text, while the genuinely retrieved page survives; missing / malformed / junk source metadata yields an explicit "attribution unavailable" marker and never an invented filename (docx without page metadata cites file only, junk page labels are not rendered); two contradicting sources make the agent abstain while agreeing evidence is still answered; multi-claim answers cite each supporting passage exactly once; prompt-injection text inside a retrieved passage is dropped with the shape contract intact (no state escalation) and cannot hijack the footer; a retrieved-but-unused passage is never cited. Drives the real books_agent with faked responses: no embeddings, no LLM, no network | no | 0 | pytest: 22 passed, 4 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_verifier.py -q` | M4B/M4B-H bounded semantic verification + privacy (49 + 12 subtests): strict verifier JSON contract (four verdicts; malformed JSON, extra/missing fields, invalid enums, unknown evidence ids and non-string reasons all rejected as `unavailable`), enforceable per-claim timeout via a daemon-thread wrapper (langchain_ollama 1.1.0 exposes no timeout parameter — verified in installed source), 6-claim cap + zero retries + 30 s total deadline + evidence-length bounding, D1 relation reversal and D2 numeric conflict removed with conflict disclosure when enabled, paraphrases kept only when verified (still subject to the deterministic gate), per-claim evidence isolation, malformed/provider/timeout fail-closed (all-unavailable refusals are marked "semantic verification unavailable"; verified negatives are not), footer rebuilt from RETAINED claims only, verifier reasons never reach the user, injection-framed untrusted-evidence prompts, model-emitted References/Sources/Citations blocks proven unable to override the validated footer (stripped before handling; removal path never cites), the G1 regression inspects the ACTUAL LLM context (absolute file_path absent, page_label citation metadata intact, stored node metadata untouched), and `BOOKS_VERIFIER=off` proven byte-identical to M3 (verifier provably never called). M4B-H hardening: process-wide 6-permit in-flight budget with fail-fast `busy` (no thread spawned when exhausted), timeout-lifecycle/late-result-isolation/concurrent-permit-race tests, exactly-once permit release across timeout/exception/malformed/success outcomes, and request-semantics pins — non-streaming invoke on ollama (the installed ChatOllama streams by default, under which the httpx read timeout bounds only gaps between chunks) and `max_retries=0` on gemini (installed langchain_google_genai defaults to six SDK retries); both verified against installed source + a fake-model recording test, not a paid API call. Mocked verifier boundary: no model, no network, no quota | no | 0 | pytest: 49 passed, 12 subtests |
| `env\Scripts\python.exe -m pytest scripts/test_m4c_attribution.py -q` | M4C claim-level evidence attribution (35, offline: fake nodes + scripted verdicts, no Ollama/Gemini/SearXNG): two claims tied to DIFFERENT sources asserted as an exact claim→evidence mapping on the per-claim footer (`[n] <ref> (verifier-supported: <claim>)` lines — the parenthetical reports the bounded verifier's judgment, never an answer-level footer implying shared support); one claim genuinely supported by two passages keeps BOTH refs on its own line; the verifier's per-claim `evidence_ids` narrow a claim's citations to the confirmed passage; a VALID id attached to the wrong claim is rejected exactly like a fabricated one (that claim removed + disclosed, the sibling keeps its own source); unknown evidence id fail-closed; per-claim contradiction scoping (reversed relationship removed with the conflict note while the sibling claim keeps its own citation); supported+unsupported claims in one answer; compound mixed support; model-authored citation block stripped; missing/junk filename+page metadata → explicit "attribution unavailable", never an invented ref; a claim removed after verification never leaves its citation behind; the adversarial battery — keyword overlap asserting an OPPOSITE relationship removed, correct entity with the wrong number removed (siblings keep their sources), evidence ids swapped between two claims rejected on the misattached claim only, a valid id carrying an insufficient verdict removed, partially overlapping evidence kept per-claim, a multi-passage claim narrowed to its one relevant passage, a removed claim's source never inherited, fabricated page/filename/table references in text dropped before any attribution, mixed supported+contradicted+unavailable answers; partial `verify_all` results (a missing sibling result counts as unavailable for exactly that claim — never a pass — and missing results for all claims remove them all), the real parser's `contradicted` verdict treated as a successful negative (no unavailable marker), and the rendered per-claim line pinned to the `verifier-supported` qualifier; refusal / unavailable / malformed / timeout outcomes; `BOOKS_VERIFIER=off` byte-compatibility (default unchanged); local-path privacy (absolute paths never in LLM context or answer); stored source metadata + persisted Chroma content untouched | no | 0 | pytest: 35 passed |
| `python scripts\benchmark_verification.py [--check]` | M4B-H/M4C deterministic verification benchmark (38 cases, offline): every case carries an explicit expected verdict, evidence, rationale, and three outcomes — recorded deterministic baseline (BOOKS_VERIFIER=off), scripted verifier verdict (mock layer; NOT model quality), and integrated design target. Verdicts may be scripted per claim (`mock_verifier_per_claim` + `mock_evidence_ids`, id-validated exactly like the production parser: a mis-attached or unknown id degrades to unavailable(malformed), never rendered as a pass), and 6 of the 7 M4C `claim_level_*` cases additionally declare an `expected_mapping` (claim→refs) that the runner parses from the verified layer's actual per-claim footer lines and requires to match EXACTLY — a valid answer-level footer can never satisfy it. Through the REAL books_agent with a faked engine + scripted `verify_all`: all 38 baseline MATCH and all 38 integrated MATCH; 17/38 rows are annotated `baseline-gap*` (the deterministic layer alone misses the target — the known D1/D2-style surface that only enabling `local` closes). Includes the §5 quality report (raw counts + denominators, off vs verified: e.g. unsupported accepted 5/9 -> 0/9, supported falsely rejected 2/15 -> 3/15, contradictions 3/9 -> 9/9 with numeric 2/4 -> 4/4 and relational 0/3 -> 3/3, malformed output -> fail-closed 0/3 -> 3/3, timeout/unavailable -> fail-closed 0/4 -> 4/4, unavailable never shown as successful 0/4 -> 4/4), the M4C row `claim-level mapping exact (verified layer): 6/6`, and deterministic budget probes (cap 6 / zero-budget deadline). `--check` exits 1 on baseline drift (CHANGED) or an integrated-layer GAP — never on the informational baseline-gap annotation, and never treats scripted verdicts as model accuracy | no | 0 | table + exit 0 |
| `python scripts\eval_verifier_local.py` | OPTIONAL M4B-H local-model experiment (installed `qwen3:0.6b` via Ollama; no downloads, no paid quota; never in the regression suite): all benchmark cases in the JSON (38 since M4C) through the real bounded verifier path at the PRODUCTION 10 s claim timeout (deliberately not loosened), recording expected vs actual verdict, parse failures, false accepts/rejects, per-claim latency (cold vs warm), timeout count, total elapsed, live verifier-thread accounting after the run, and a post-run probe. Results vary run to run: recorded runs had 17 and 26 of 31 calls exceed the 10 s bound with 0 parse failures and 0 false accepts among completed calls in both — small samples, not statistically conclusive; latency (not verdict quality) is the binding constraint. Exits 2 with a clear message when the local runtime is down — never substitutes a paid model | local Ollama | 0 | per-case table + summary |
| `python scripts\check_dependencies.py` | every required package imports | no | 0 | `ALL_DEPENDENCIES_OK` |
| `python scripts\check_chroma.py` | persisted collection: non-empty, prose sample, and manifest/store integrity (per-file chunk counts via `file_path` metadata, indexed-with-zero-chunks, failed/partial records, legacy entries, entries whose file is gone, stored vectors with no manifest entry). Read-only; labels a check UNAVAILABLE when the index lacks the metadata for it | no | 0 | counts by status + per-file ok/partial/failed/legacy lines; exit 0 clean, 1 on integrity error |
| `env\Scripts\python.exe -m pytest scripts/test_rag_eval_harness.py -q` | RAG-1 retrieval-evaluation harness (38, offline: `MockEmbedding`, no Ollama/LLM/network): the deterministic fixture generator rebuilds every fixture byte-identically from `scripts/make_rag_fixtures.py`, guard unit tests (size+mtime+sha256 change detected / unchanged passes), gold-set schema + provenance validation against the fixture and corpus sources with a read-only hash guard, metrics semantics (hit@k, precision@k, MRR, nDCG@5 binary relevance, abstention-proxy polarity), the ISOLATION PROOF (every Chroma client constructed during evaluation lands on the temp path and never on `storage/`; the production books_agent seam restores `STORAGE_DIR`/`CHROMA_COLLECTION`/`_query_engine`/`Settings._llm`/`Settings._embed_model` by identity — including on failure — and the temp directory is provably removable after client release, so no SQLite/hnsw handle leaked), artifact-schema-3 evidence validation with an independent full recalculation of every metric from the in-artifact `retrieved_top5` + `gold_ref`, overwrite-safety tests (an existing `--out` report ⇒ refusal exit 3 **before** any guard/gold/Ollama work and no write; `--force` is the only opt-in; the documented bare run can no longer clobber the pinned v1), SHA-256 pins for `rag_gold_v1.json` and baselines v1/v2/v3 asserted unchanged, plus the real-store stat+sha256 guard asserted unchanged across the test. The frozen baseline is produced by `python scripts\eval_retrieval.py` (local Ollama `nomic-embed-text` embeds via localhost:11434, retrieval-only, no chat-model calls; **refuses with exit 3 and writes nothing when the `--out` file already exists** unless `--force` is passed, and stops without writing if `storage/` or the gold set changes). Frozen reference artifact: `benchmarks/rag_baseline_v3.json`, sha256 `aa89c37bc23b0dc748255df9410a28ba43a6f8d8043070ad54cafd6e175de1cd`, `artifact_schema.version=3` (`ordered-top5-per-case`), `metric_definitions.version=v2-page-metadata-aware` — recorded metrics: see "RAG-1 frozen baseline (v3)" below | no (suite); local Ollama embeds for a baseline run | 0 | pytest: 38 passed |
| `env\Scripts\python.exe -m pytest scripts/test_ingest_integrity.py -q` | M2 ingestion integrity (27): unsupported files reported as ignored, healthy PDF/DOCX indexed with store-verified chunk counts, corrupt PDF and "reader returned no documents" failed (never indexed), zero-text PDF reported as no-extractable-text (not "blank"), mixed PDF partial with the unextractable page named and the readable page still usable, oversized and page-limit rejects before the reader is called, encrypted PDF explicit failure, idempotent re-run (no duplicate chunks, no manifest drift), failed files retried, failed replacement keeps the previous vectors, changed file replaces its own stale chunks, legacy v1 manifest migrated and store-verified (or repaired when its vectors are missing), corrupt manifest tolerated, files missing from disk reported not pruned, and the check_chroma diagnostic's own exit codes. Temp dirs + `MockEmbedding`: no Ollama, no Gemini, no network, no quota, real knowledge base untouched | no | 0 | pytest: 27 passed |
| `python scripts\verify_graph.py` | **LIVE** smoke test: real graph, 3 scenarios | SearXNG + web | **~3 calls** | three scenario blocks; refusal on out-of-corpus |
| `python scripts\test_phase1_e2e.py` | **LIVE** spec steps 3 & 5 end to end | SearXNG + web | **~2 calls** | `STEPS_3_5_REAL_OK` |
| `python scripts\check_books_agent.py` | one real books answer with provenance | Ollama embeds | **1 call** | answer + `SOURCES: ['books']` |
| `python scripts\probe_books_prompt.py` | captures the exact prompt sent to the LLM by the query engine | Ollama embeds | **1 call** | answer + captured prompts |
| `python scripts\demo_rewrite_quality.py` | before/after search-quality demo of the rewrite step | SearXNG + web | **1 fast call** | raw vs rewritten result lists |

### RAG-1 frozen baseline (v3) — recorded metrics

**Frozen reference artifact** (retrieval-only; the comparison point for
every later retrieval change):

| Field | Value |
|---|---|
| File | `benchmarks/rag_baseline_v3.json` (95,700 bytes) |
| SHA-256 | `aa89c37bc23b0dc748255df9410a28ba43a6f8d8043070ad54cafd6e175de1cd` |
| Run | `books_eval_20261017-095829` (2026-10-17T06:58:29Z), git `7a44a9a7f747`, `git_dirty: true` (evaluated code was uncommitted — recorded deliberately) |
| Schema | `artifact_schema.version = 3`, evidence `ordered-top5-per-case` — every metric recomputable from the in-artifact `retrieved_top5` + `gold_ref` |
| Metric definitions | `v2-page-metadata-aware` (page filter skipped for nodes without page metadata; `docs/RAG_BASELINE_AUDIT.md` §1.5 and §7) |
| Guards in artifact | real store unchanged · gold unchanged · all Chroma clients on temp paths · production query-engine cache unused · fidelity check true · 8 eval chunks · 32 cases (25 answerable, 7 unanswerable) |

**Retrieval metrics — macro averages over answerable cases only:**

| Section | Denominator | hit@1 | hit@3 | hit@5 | precision@1 | precision@3 | coverage | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|---|---|
| real corpus | 11 answerable (4 unanswerable excluded) | 0.9091 (10/11) | 1.0 (11/11) | 1.0 (11/11) | 1.0 (11/11) | 0.3636 (12 covering nodes / 33 slots) | 1.0 (11/11) | 1.0 | 1.0 |
| synthetic fixtures | 14 answerable (3 unanswerable excluded) | 0.8571 (12/14) | 1.0 (14/14) | 1.0 (14/14) | 0.9286 (13/14) | 0.3571 (15 covering nodes / 42 slots) | 1.0 (14/14) | 0.9643 ((13×1 + 0.5)/14) | 0.9736 ((13×1 + 0.6309)/14) |

Caveats that must travel with these numbers:

1. **Definitional split, not an arithmetic error:** `hit@k` / `coverage`
   count *distinct gold documents* against `min_docs_covered`, while MRR /
   nDCG@5 use the *first covering node*. On the two multi-doc cases
   (R011, S016) `hit@1` is structurally false (one slot cannot hold two
   documents) while MRR/nDCG stay 1.0 — that is why hit@1 = 0.9091 next to
   MRR = 1.0.
2. **Not comparable to v1:** v3 uses the revised page-metadata-aware
   relevance rule; v1 (`2c1d4ae9…`) used the older rule. Only R011 and
   S016 differ (real hit@3/coverage 0.9091 → 1.0, precision@3 0.3333 →
   0.3636; synthetic hit@3/coverage 0.9286 → 1.0, precision@3 0.3333 →
   0.3571); every other value is identical across v1/v2/v3. Cross-version
   comparison is valid only for hit@1, MRR, nDCG@5 and precision@1 — full
   lineage in `docs/RAG_BASELINE_AUDIT.md` §7.
3. **The abstention proxy does NOT measure answer-level refusal.** The
   recorded `0/4` and `0/3` "correct on unanswerable" (and `0/11` / `0/14`
   false-abstains) only prove the 0.35 `BOOKS_MIN_SCORE` floor is
   unreachable — all top-1 scores lie in [0.65, 0.83]. Production refusal
   also depends on the synthesis prompt and the deterministic grounding
   gate (chat-model behavior); that path is measured by **RAG-AE
   (answer-quality evaluation — not started)**, never by this baseline.
4. **The real corpus cannot discriminate:** both real documents fit in a
   single chunk each (8-chunk eval collection), so real-section numbers
   are a ceiling, not a quality signal; synthetic-fixture metrics do not
   establish real-world retrieval accuracy.
5. Latency (retrieval p50 72.9 ms / p90 82.8 ms / max 90.3 ms; embed p50
   62.0 ms; ingest 13.1 s) is a machine-local measurement, not a guarantee.

Regeneration: `python scripts\eval_retrieval.py --out <new path>` — an
existing file is refused with exit 3 unless `--force` is passed. The gold
set and all three baselines are SHA-256-pinned by the harness suite and
must never be rewritten.

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
env\Scripts\python.exe -m pytest scripts/test_phase4_workflow_qa.py scripts/test_phase5_qa_node.py scripts/test_phase6_routing_llm.py scripts/test_agent_architecture.py scripts/test_gui1_events.py scripts/test_api_preview.py scripts/test_gui16_settings.py scripts/test_gui5_operations.py scripts/test_phase7_planner.py scripts/test_phase8_execution.py scripts/test_books_grounding.py scripts/test_ingest_integrity.py scripts/test_citation_integrity.py scripts/test_verifier.py scripts/test_m4c_attribution.py scripts/test_books_grounding_paraphrase.py scripts/test_books_grounding_adversarial.py scripts/test_execution_instrumentation.py -q
```

Full combined regression at the Phase 8 dispatcher commit:
**184 passed, 1 skipped, 14 subtests** (the skip is the opt-in
`OLLAMA_LIVE=1` live provider test). Phase 8 adds 21 of its own tests
(see the suite row above); all other suites are green unchanged.

Current combined regression (verified 2026-10-19, run under the fail-closed
guard): **427 passed, 1 skipped, 52 subtests** (exit 0) for the suite set in
the command above plus `scripts/test_gui12_plan_render.py` (GUI-12).
Composition of this run: 383 previously documented tests + 11 tests from
`scripts/test_books_grounding_paraphrase.py` + 33 tests from
`scripts/test_books_grounding_adversarial.py`; the subtest count reconciles
to 34 existing subtests + 18 adversarial subtests. The single skip is the
documented `OLLAMA_LIVE=1` opt-in. These are the verified results FOR THAT
RUN, not a guarantee that every future run produces the same count.

Canonical full suite (every file in `scripts/`, verified 2026-10-28 after
the authorized **GUI-10 approval-lifecycle** pass):
**821 passed, 2 skipped, 254 subtests (104s), exit 0**, run under the
fail-closed socket guard (`.diag/w2_network_audit.py`) with **external
network attempts = 0**. Composition: the 785 of the routing-policy +
research-approval run + 36 new GUI-10 approval-lifecycle tests
(`scripts/test_approval_lifecycle.py`, 7 subtests) + the AgentState key-set
pin updated for the deliberate `research_approval` channel (+1 counted
above as part of the suite, no other new tests). The two skips are the
documented `OLLAMA_LIVE=1` live provider test and
`test_chroma_isolation`'s symlink-privilege skip on Windows. The known
`test_phase4_workflow_qa::test_full_feasibility_workflow` Ollama-ordering
flake did NOT recur in the guarded full run, but it DID recur once in a
later standalone whole-file run of `test_phase4_workflow_qa` (the tail
truncation dropped the assertion detail — recorded rather than hidden);
Ollama was confirmed live (HTTP 200 on the loopback tags endpoint), which
is exactly the documented environment-dependent cause in section C below,
and identical reruns passed (file-level 30 passed; the single test twice
more). One first-run
failure of the AgentState pin (`research_approval` not yet expected) was
fixed by updating the pin with its justification — not by weakening the
assertion.

Previous canonical run (verified 2026-10-28 after the authorized
+ 18 routing-policy tests + 19 research-approval tests + 1 knowledge-fallback
test proving an AUTHORIZED request still executes the research node; the
two skips are the documented `OLLAMA_LIVE=1` live provider test and
`test_chroma_isolation`'s symlink-privilege skip on Windows. Two deliberate
behavior changes travel with this run: (a) Priority A removed the research
DEFAULT, so cue-less questions route to `general`/`unsupported` — the pins
in test_phase1_routing, test_phase3_erp, test_phase4_workflow_qa,
test_phase6_routing_llm and test_conversational_routing were updated where
they asserted the old default, each with its justification in place; (b)
Priority B gates the books->research fallback's EXTERNAL step, so
test_knowledge_fallback now expects an approval request instead of a search
(eligibility, topology, failure-class logging, bounded recovery order and
privacy rules unchanged, and case 6 proves the authorized path still runs).
The known `test_phase4_workflow_qa::test_full_feasibility_workflow`
Ollama-ordering flake did NOT recur in either the targeted or the full run.

Previous canonical run (verified 2026-10-28 after the authorized
**Increment 2** pass — shared diagnostics core + books-agent decomposition):
**747 passed, 2 skipped, 177 subtests (124s), exit 0**, run under the
fail-closed socket guard (`.diag/w2_network_audit.py`) with **external
network attempts = 0**. Composition: the 705 of the 2026-10-27 W2.1 run
+ 12 conversational-routing tests (greeting-misrouting fix) + 9 execution-
lifecycle tests + 21 new shared-diagnostic-core tests
(`scripts/test_diagnostics_core.py`). The books split
(`books_agent.py` → facade + `books_retrieval` / `books_grounding` /
`books_recovery`) is a behavior-identical relocation: the grounding,
recovery, citation, verifier, chroma-isolation and arch-contract suites
above are the evidence, not imports alone.

Previous canonical run (verified 2026-10-27 after
the authorized **W2.1 corrective pass**):
**705 passed, 2 skipped, 144 subtests (184s), exit 0**, run under the
fail-closed socket guard (`.diag/w2_network_audit.py`) with
**external network attempts = 0** — i.e. the 2026-10-25 fallback baseline of
661 + 6 new W1 grounding/eligibility tests + 27 new W2 instrumentation tests
+ 8 new W2.1 sink-isolation / event-observability tests + 3 new W2.1
`essentially` grounding tests, with 4 new subtests (2 + 2). The immediately
preceding run (same day, after W1 + W2) recorded 694 passed, 2 skipped, 140
subtests (166s), exit 0. RAG-AE offline evaluations were re-run after the
last source edit and are metrics-identical to the recorded baseline (unit
17/17/0, abstention P/R 1.0/1.0; integration 10/10/0, hit@1 0.8, hit@3 1.0,
coverage 1.0, `network_attempts: 0`, `guard_unchanged=True`; artifacts
`.diag/w21_eval/unit_w21.json` and `.diag/w21_eval/integration_w21.json`).
Scope caveats that travel with the number: the batch ran against the WORKING
TREE, which includes uncommitted guard/isolation and M5 self-refusal
recovery work that is NOT part of commit `55e599d`; a separate clean-checkout
run OF that commit verified its four committed grounding suites (90 passed,
26 subtests, exit 0) — the full 427-test batch was not run against a clean
checkout of `55e599d`. Prior documented baseline (M4C, re-run 2026-10-16):
383 passed, 1 skipped, 34 subtests — pre-M4B baseline 299,
M4B-H baseline 348 (both historical); the M4B/M4B-H suite is the 49 and the
M4C suite the 35 of those tests listed in section C.

After ANY change to the books grounding gate — since the Increment 2
decomposition these live in `app/agents/books_grounding.py` (re-exported
from `app/agents/books_agent.py`; `books_agent.py` now holds only the
orchestrator and the compatibility seams) — i.e. `_support_tokens`
normalization, `_PARAPHRASE_GROUPS`, `_sentence_covered`, the polarity /
numeric / citation gates — run the three
grounding suites FIRST — `scripts/test_books_grounding.py`,
`scripts/test_books_grounding_paraphrase.py`,
`scripts/test_books_grounding_adversarial.py` — then the full combined
regression above. The adversarial suite deliberately PINS current
limitations (cross-passage / cross-sentence splicing, the family-3 subject
swap): a change that flips one of those pins must be a deliberate, reviewed
redesign (see `docs/GROUNDING_DIRECTIONAL_ALIASES_DESIGN.md`), never an
accident. All of these runs stay under the fail-closed Chroma guard of
section F — production-store access is rejected by construction, and
`scripts/check_chroma.py` is not part of this regression (manual,
unarmed, never run automatically). Verification is
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

## F. Production Chroma store isolation (fail-closed guard)

After the RAG-1 closeout regression (`df3e683`) passed all suites while
`storage/chroma.sqlite3` changed bytes unexpectedly, test/eval access to the
production store now **fails closed**. Root-cause status of that mutation:
**likely, NOT confirmed** (no content-level store verification has ever been
permitted; see the investigation report).

### How isolation is enforced

`app/chroma_guard.py` is the single shared guard. Every check runs *before*
any client is constructed; paths are normalized with `realpath` + `normcase`,
so relative, absolute, `..`-mixed, symlinked, and case-different spellings
all compare equal. The production root `storage/` (and anything inside it)
raises `ProductionStoreAccessError`; any other path that is neither the
system temp dir nor a configured disposable root raises `ChromaGuardError`.

Enforcement points (every known persistent-client construction path):

1. `app/agents/books_agent.py::_get_query_engine` — explicit check as the
   first statement of the lazy singleton, before any Settings work or client
   construction.
2. `ingest_books.py::_build_index` — the ingestion entry point (also used by
   the eval's per-file `process_file` path).
3. `scripts/check_chroma.py::main` — guarded, but manual runs are unarmed
   and behave exactly as before (the script itself is still never run
   automatically).
4. `scripts/eval_retrieval.py` — `main()` arms the guard first, and checks
   sit in the instrumented `PersistentClient` recording wrapper plus both
   temp-store constructors.
5. `scripts/conftest.py` — pytest imports it before any test module, arms
   the guard process-wide and wraps `chromadb.PersistentClient`,
   `chromadb.Client`, and `chromadb.AdminClient`, so *direct* construction
   anywhere in a test — including a module-level import-time initializer —
   is rejected pre-construction.
6. The documented standalone scripts (`test_phase1_routing`, `test_phase2`,
   `test_feasibility_degradation`, `test_rewrite_step`, `test_phase3_erp`,
   `test_agent_architecture`) arm + wrap at startup so
   `env\Scripts\python.exe scripts\<name>.py` is protected like pytest.

Activation: pytest (`conftest` hook / `PYTEST_CURRENT_TEST`), an explicit
`chroma_guard.arm()`, or `AGENTICA_CHROMA_GUARD` set to **any** value
(including `""` — ambiguous configuration arms, never bypasses). There is no
disable switch. Outside test execution every check is a no-op: normal app,
ingest-CLI, and manual-diagnostic behavior is unchanged.

### Configuring disposable test stores

- System temp directory (`tempfile.gettempdir()` / `mkdtemp` /
  `TemporaryDirectory`) is implicitly allowed.
- `AGENTICA_TEST_STORAGE_DIRS` lists additional roots, `os.pathsep`-separated
  (e.g. `C:\Temp;D:\scratch`). Set-but-empty/blank raises
  `ChromaGuardConfigError` before *any* path is allowed (fail closed).
- No configuration at all means: production always blocked, temp allowed,
  everything else blocked.

### Running the isolation tests

```bat
env\Scripts\python.exe -m pytest scripts/test_chroma_isolation.py -q
:: 15 passed, 1 skipped (Windows symlink-privilege variant), 7 subtests, exit 0
```

Offline by construction: no Ollama, no network, no production Chroma. The
suite proves rejection precedes construction (against a recording
constructor double), equivalent-path rejection, temp/configured-path
permission, lazy-engine and import-time protection, mocked-engine
continuity, config fail-closed behavior, and that rejected attempts leave
the protected files byte- and mtime-identical (filesystem hashing only).

### What the guard does and does not guarantee

Guarantees: in an armed run no known persistent-client path can pass a
production or unknown path to chromadb — tests fail fast with an actionable
error instead of mutating the store; unarmed production runs are
behaviorally unchanged.

Does NOT guarantee: the earlier mutation's root cause (status: **likely,
unconfirmed**); protection of invocations that are neither pytest nor a
bootstrapped script nor `AGENTICA_CHROMA_GUARD=1`; anything about raw
sqlite3 writers (the guard covers chromadb constructors only); or logical
integrity of the production store (never content-verified).

### Suites unsafe/unverified after this change

- `test_agent_architecture.py::TestAgentContract::
  test_books_agent_returns_only_changed_keys` originally called the REAL
  `books_agent` without mocking `_get_query_engine` and was the single most
  likely writer of the earlier store change. Under the guard it failed fast
  with `ProductionStoreAccessError` (fail-closed, as designed); it has since
  been repaired to inject a deterministic fake at the lookup site and to
  assert the Chroma constructor is never reached — the changed-keys contract
  assertion is unchanged, and the suite is 30 passed.
- The full documented pytest batch HAS been re-run under the guard:
  **383 passed, 1 skipped, 34 subtests** (exit 0), matching this file's
  documented baseline at that time (superseded by the 427-test current
  baseline recorded in section C above).
- Live suites (`test_phase1_e2e`, `test_phase1_proxy`, `check_books_agent`,
  `probe_books_prompt`, `demo_rewrite_quality`) remain opt-in/network-bound;
  under pytest they would fail closed if they reach the production engine.
- Environment note: run documented scripts with `env\Scripts\python.exe`;
  the bare `python` on PATH in this environment lacks `langgraph`/`chromadb`,
  and the guard's wrapper self-skips when chromadb is not installed.

## RAG-AE — Answer-quality evaluation harness (2026-10-23)

Answer-quality evaluation (MASTER_PLAN RAG-1 §7 item 5) is now implemented:
harness `scripts/eval_answers.py`, dataset `benchmarks/rag_gold_v2.json`
(sha256-pinned in each artifact), offline test suite
`scripts/test_eval_answers.py` (63 tests), baselines
`benchmarks/rag_ae_unit_v1.json` + `benchmarks/rag_ae_integration_v1.json`.
Full metrics, command syntax, and limitations: `docs/RAG_AE_EVALUATION.md`.

Zero-quota and fully isolated — no production Chroma, no network, no model
API; a disposable temp Chroma is created per integration run
(`CHROMA_DISPOSABLE_ROOTS`-independent, temp-root allowed by the guard).

### RAG-AE v3 — C101–C103 resolution pass (2026-10-23)

The three pending records were re-investigated and carried into
`benchmarks/rag_gold_v3.json` (sha256-pinned by five new tests inside
`scripts/test_eval_answers.py`, which is now 68 tests):

- **C102 promoted executable** against a new deterministic fixture
  `benchmarks/rag_fixtures/fixture_multipage_table.pdf` (added to
  `scripts/make_rag_fixtures.py`; byte-identical rebuild enforced by the
  generator's self-check; the pre-existing 5 fixtures regenerate
  unchanged). `scripts/test_rag_gold_v2_candidate.py` now verifies the
  draft's blocked-on staleness in BOTH directions
  (`provenance.blocker_removed` must not point at a missing file).
- **C101 and C103 stay unresolved** with VERIFIED-FINAL findings
  (source+test citations in the file): C101's two-affirmative-value
  adjudication is deterministically unmeasurable
  (`test_case3_conflict_claim_with_affirmative_evidence_kept`,
  `scripts/test_books_grounding.py:331` — the mirror shape that IS
  measurable is scored by C009); C103's query subject exists in no
  document and its expected wording is ingest-manifest vocabulary, while
  the behavior class is scored by C008.

Commands and observed results (both modes exit 0, deterministic under
`strip_volatile`, zero network, guard unchanged):

```bat
env\Scripts\python.exe -m pytest scripts\test_eval_answers.py -q
    :: 68 passed
env\Scripts\python.exe scripts\eval_answers.py --mode unit ^
    --gold benchmarks\rag_gold_v3.json --out benchmarks\rag_ae_unit_v3.json
    :: 17/17 executable passed; C101/C103 reported unresolved
env\Scripts\python.exe scripts\eval_answers.py --mode integration ^
    --gold benchmarks\rag_gold_v3.json ^
    --out benchmarks\rag_ae_integration_v3.json
    :: 10/10 scored; hit@1 0.8, hit@3 1.0, fact_coverage 1.0
```

Related regression runs: `scripts/test_rag_gold_v2_candidate.py` +
`scripts/test_chroma_isolation.py` + grounding/citation/recovery suites:
198 passed, 1 skipped, 70 subtests (exit 0). The gold-v2 dataset file,
its sha pin, and the v1/v2 baselines are untouched.
(`CHROMA_DISPOSABLE_ROOTS`-independent, temp-root allowed by the guard).

## Continuous learning (2026-10-30)

Offline, deterministic, no model and no network (the model call is replaced by
`llm_call=` / the test LLM double):

```bat
env\Scripts\python.exe -m pytest scripts\test_learning.py -q
    :: signals, activation policy, dedupe/supersede, retrieval scoping,
    :: skill gate + rollback, QA honesty checks, "learning writes no files"
env\Scripts\python.exe -m pytest scripts\test_learning_e2e.py -q
    :: through GraphRunner: correction -> feedback route -> lesson ->
    :: later prompt carries it -> lesson hits/success + event recorded;
    :: unrelated capability untouched; disabled lesson stops applying;
    :: edited skill version passes/fails its gate and rolls back
env\Scripts\python.exe -m pytest scripts\test_learning_api.py -q
    :: GET /api/v1/learning, review (activate/disable/reject/correct),
    :: skill rollback, and the Memory/Skills projections
```

What the tests actually prove: the requirement is absent from the baseline
prompt and present in the later comparable task's prompt (`test_the_requirement
_reaches_a_later_comparable_task`), the execution records
`learning.lessons_applied` only for lessons it was given, and a lesson whose
status is `disabled` produces neither a prompt block nor an applied event.
`test_learning.py::TestQAHonesty` proves a reply that claims a save without a
stored lesson is flagged, and that an invented `lessons_applied` id is flagged.
`test_learning.py::TestRepeatedFailureCapture` proves the first failure of a
signature records nothing, the second becomes a candidate, a third merges into
the same lesson, and plan/task counters (`t1` vs `t12`) do not split the
signature. `test_learning_e2e.py::TestFailureObservation` proves the runner
observes a raised graph failure, learns only on the repeat, and never applies
the candidate to the next run.

### Live verification against the local provider (2026-10-31)

Run through the real API (`POST /api/v1/executions`, the same path the UI uses)
with Ollama (`qwen3:0.6b`), two runs per step, no test doubles:

```bat
:: baseline -> correction -> comparable task -> unrelated task
curl -X POST http://127.0.0.1:8000/api/v1/executions -H "Content-Type: application/json" ^
     -d "{\"question\": \"What are the trade-offs of using Redis instead of PostgreSQL for a cache layer?\"}"
curl -s http://127.0.0.1:8000/api/v1/executions/<id>/events
```

What it proved: the correction routes to `feedback`, `learning.lesson_recorded`
stores an ACTIVE lesson scoped `general`, a later comparable question emits
`learning.lessons_applied` with that id and its answer carries the comparison,
and an unrelated question emits no `learning.lessons_applied`.

Six real defects were found this way and fixed (each has a regression test):

* the model answered the analysis prompt with its own TEMPLATE, and
  `<the concrete behaviour to follow next time>` was stored as an active rule
  (`test_learning.py::…schema_echo_never_becomes_the_learned_behaviour`);
* the content lesson was injected into "what does HTTP 429 mean?" because
  "next time" was read as a global preference (`_GENERAL_RE`) and content
  lessons had no relevance gate
  (`test_learning_e2e.py::…unrelated_question_in_the_same_capability_is_not_injected`);
* two different corrections ("trade-offs and numbers" / "caching and payload
  size") merged into one lesson, because dedupe compared the boilerplate
  problem sentence instead of the items the user named
  (`test_learning.py::…two_different_requirements_do_not_merge_into_one_lesson`);
* a pointing directive was stored verbatim ("Include those next time"), which
  says nothing in a later prompt, and the same pointer appeared in the
  "Not covered here" footer
  (`test_learning.py::…a_pointing_directive_names_what_the_user_asked_for`);
* the model itself supplied a pointing BEHAVIOUR ("include those next time")
  and it was kept verbatim
  (`test_learning.py::…a_pointing_behaviour_from_the_model_is_replaced`);
* a lesson's QA wording took part in relevance matching, so "what is a
  checksum?" pulled in an unrelated lesson through the word "check"
  (`test_learning.py::…the_validation_wording_is_not_part_of_the_subject`).

One operational defect was also found: a local provider call stalled a
feedback execution indefinitely (no transport timeout). Both learning calls are
now bounded (`lessons.ANALYSIS_TIMEOUT` 60 s, `feedback.COMPOSE_TIMEOUT` 90 s)
and fall back to the deterministic reading
(`test_learning.py::…the_analysis_call_is_bounded_by_a_transport_timeout`).

The live rows written before the fix were repaired in place (fields re-derived
from the stored user text, repair noted in the lesson's evidence); the
Control Center then showed the legible lesson with its real use count.

Related regressions run in this increment (all exit 0):
`test_control_center_api.py` (33), `test_agent_architecture.py`,
`test_gui1_events.py`, `test_phase5_qa_node.py`, `test_phase4_workflow_qa.py`,
`test_phase7_planner.py`, `test_phase8_execution.py`, `test_routing_policy.py`,
`test_e2e_assistant.py`, `test_coding_*`, `test_approval_lifecycle.py`,
`test_execution_lifecycle.py`. Frontend: `npx tsc -b`, `npx vitest run`
(75 tests, including `Learning.test.tsx`), `npm run build`.

## Learning evaluation harness (2026-10-30)

```bat
env\Scripts\python.exe -m pytest scripts\test_learning_eval.py -q
    :: 37 tests: dataset validation, requirement scoring, condition and
    :: comparison rules, regressions, per-lesson stats, reports, and the real
    :: GraphRunner A/B run on the shipped dataset
env\Scripts\python.exe scripts\eval_learning.py --print
    :: writes storage\learning_eval\<version>_<stamp>.json + .md
```

The end-to-end assertions are the point: both conditions use the identical
request, the baseline applies nothing, learning applies exactly its declared
fixtures, and the measured outcomes equal each case's `expected_outcome`
(3 improved, 1 regression, 2 unchanged on `learning_eval_v1`). A repeated run
must produce an identical `summary` (reproducibility).

Observed run of the shipped dataset (scripted offline provider):

```
completeness: baseline 53.8% -> learning 76.9% (+23.1 pp)
qa findings: 0 -> 0
improved 3, regressed 1, unchanged 2
  regression: gen-brevity-regression (requirement_lost) explains-latency
  regression: gen-brevity-regression (requirement_lost) names-tradeoff
  regression: gen-brevity-regression (requirement_lost) not-too-short
```

That regression is intentional: `fx-too-brief` is a conflicting preference that
drops requirements the baseline satisfied, so it is reported as evidence
against the lesson, not as an improvement.

## Full-system live audit harness (2026-10-31)

`scripts/audit_live_matrix.py` is an **audit** tool, not application code: it
POSTs to the same endpoint the Control Center uses and records route, status,
latency, answer, QA flags, event types and applied/recorded lesson ids.

```bat
env\Scripts\python.exe scripts\audit_live_matrix.py            :: both batches, 23 cases
env\Scripts\python.exe scripts\audit_live_matrix.py --batch 2  :: calibration/QA/tools/lifecycle only
    :: exit 1 if any case takes an unexpected route (the audit's findings)
    :: evidence: storage\audit\live_matrix*.json
```

Cases are ordered on purpose: `feedback` and `followup` read the most recent
completed execution, so a correction follows the answer it criticises.

Two deterministic checks added by that audit live in the product and are pinned
by unit tests: numeric grounding for `general`/`feedback` answers
(`qa_node._unsourced_figures`) and prompt-echo removal
(`llm_client.strip_instruction_echo`). Their tests are in
`test_phase5_qa_node.py` (`TestNumericGrounding`, `TestInstructionEcho`); the
routing fixes are in `test_routing_policy.py`, the relevance fix in
`test_learning.py`.

Results, defect list and limitations: `docs/FULL_SYSTEM_RELIABILITY_AUDIT.md`.

## Evidence-grounded general answering (2026-11-01)

Offline suite (scripted provider double, fixed passages, no network, no vector
store, no Ollama):

```bat
env\Scripts\python.exe -m pytest scripts\test_general_evidence.py -q    :: 35 tests
```

Evaluation harness — the same `general` node and QA gate run twice, retrieval
off (baseline) vs on (grounded), 15 cases with authoritative fixtures:

```bat
env\Scripts\python.exe scripts\eval_evidence.py --print
env\Scripts\python.exe scripts\eval_evidence.py --fail-on-regression
    :: evidence: storage\evidence_eval\evidence_general_report.{json,md}
    :: last run: baseline 7/12 -> grounded 12/12, no regressions
```

The harness measures the PIPELINE (what survives the grounding gate, what is
cited, what QA reports), never the model's knowledge: the provider double
repeats each case's prepared reply.

Design, evidence flow, claim classes, live execution ids and limitations:
`docs/EVIDENCE_GROUNDED_ANSWERING.md`.

## Retrieval reranking, claim verification, evidence snapshots (2026-11-01)

```bat
env\Scripts\python.exe -m pytest scripts\test_claim_verification.py -q   :: 32 tests
env\Scripts\python.exe scripts\eval_claims.py --print
    :: retrieval: hit@1 0.333 -> 1.000, MRR 0.625 -> 1.000, no regressions
    :: claims:    11/11 exact verdicts, false verification rate 0.0
    :: evidence: storage\evidence_eval\evidence_claims_report.{json,md}
```

Snapshot checks are in the same suite: roundtrip, no local paths, unsafe ids,
the runner hook (writes for an evidence run, writes nothing without one), and
the read-only API (404, `evidence: null` with a reason, and a stored snapshot).
Live: `GET /api/v1/executions/<id>/evidence` returns the snapshot after a
backend restart.

Design and limitations: `docs/RETRIEVAL_AND_CLAIM_VERIFICATION.md`.

## Snapshot completeness and the books route (2026-11-02)

```bat
env\Scripts\python.exe -m pytest scripts/test_evidence_snapshot.py -q    :: 14 tests
```

Deterministic, no model and no vector store: four retrieved passages persist as
four (the shared-budget truncation that dropped the fourth), a prompt budget
that supplies fewer passages than were retrieved still persists every one and
marks `supplied` per row, a per-passage cap marks a long passage `truncated`
(never omitted) and the snapshot reports `complete: false` with a reason, plus
the books route's answered / refused / no-retrieval paths (the last writes no
snapshot at all), the runner's record selection, and reading a snapshot back in
a separate process (= a real restart). Live: general `a49f58678a01` recorded
4/4 passages (2 supplied, `complete: true`); books `18bff98789c9` recorded its
3 consulted passages on a refusal; both were re-read unchanged after a backend
restart.
