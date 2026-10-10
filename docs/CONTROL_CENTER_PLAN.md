# Control Center Plan — GUI for the Non-Technical Operator

Status: planning document only. **Nothing in this document is implemented.**
This is the implementation roadmap for a graphical Control Center layered ON
TOP of the Master Agent System. It does not change the agent architecture.

Relationship to the backend roadmap: `docs/MASTER_PLAN.md` is authoritative
for agent/execution architecture (Phases 5–16). This document references it
and never duplicates or contradicts it (see Section 34).

Rules for this task (already applied): no production code modified, no
`MASTER_PLAN.md` changes, no dependencies added, no frontend/API files
created, no commits.

---

## 1. Inspection of the existing system (verified against code at `c527ff3` + `docs/MASTER_PLAN.md`)

| Area | Current reality (code-verified) | Consequence for the GUI |
|---|---|---|
| Entry point | `main.py` CLI only (`--once`, `--demo`, interactive `input()` loop) | No API server, no HTTP surface exists today |
| Execution model | None — `graph.invoke({"question": ...})` runs synchronously to completion in-process | The GUI's core object (an Execution that can be watched/controlled) does not exist yet; it must be built backend-first |
| State | `AgentState` (question, route, sub_routes, sources, answer, qa_flags, agent_answers) — per-invocation, not persisted | GUI needs a persisted `Execution` model wrapping runs; `AgentState` stays internal |
| Supervisor | Keyword routing (feasibility → erp → books → research) | Route = one displayable step; becomes "mode" with MASTER_PLAN Phase 6 |
| Graph | Nodes: supervisor, books, research, erp, feasibility; feasibility chain research→books→feasibility | Step boundaries for the timeline come from graph node execution |
| Agents | books (Chroma/LlamaIndex RAG), research (SearXNG+Crawl4AI), feasibility (synthesizer + `.docx`), erp (mock adapter, no LLM) | Each agent needs a human-friendly display name + technical name mapping |
| QA | `run_qa()` exists only in `scripts/test_phase4_workflow_qa.py`; `qa_flags` never written in production | GUI must not display QA findings until MASTER_PLAN Phase 5 puts QA in the graph |
| LLM | `get_llm()` factory, providers gemini/ollama via `.env` | Health indicator + Settings page read `LLM_PROVIDER`; provider stays backend-owned |
| RAG | Vector-only Chroma index built by `ingest_books.py`; manifest at `storage/.index_manifest.json` | Knowledge page can be built from manifest + collection, no embedding internals exposed |
| Logging | Only `app/erp/log.py` → `logs/erp_actions.log` (append line per ERP op) | No general event stream; the event model (Section 10) is net-new |
| Reports | `reports/feasibility_YYYYMMDD_HHMMSS.docx` written by feasibility agent | Reports page = index of these files linked to their executions (needs provenance link added) |
| Approvals | None anywhere (ERP boundary is read-only by construction) | Approval Center is net-new backend capability (Section 12), not a UI overlay |
| Controls | None — no pause/resume/cancel exists | Control endpoints are net-new backend capability (Section 14) |
| Tests | 101 deterministic tests + 10 subtests (7 suites), zero quota; Phase 4 workflow suite proves graph orchestration | GUI tests must reuse this discipline; backend event/API tests join the zero-quota set |
| Frontend | **None** — no `package.json`, no framework, no web deps in `requirements.txt` | Frontend technology is a free choice (Section 27) |
| Web stack | No FastAPI/Flask/uvicorn/SSE/WebSocket anywhere | API server is net-new; dependency addition must be justified at GUI-1/GUI-2 |

Key conclusion from inspection: **the GUI's real prerequisite is a backend
execution/event/API layer that does not exist yet.** The GUI roadmap is
therefore ordered foundation-first (Sections 29–34); the frontend never
fabricates the data it needs (Section 33).

---

## 2. Product vision

Audience: a completely non-technical operator. They never need to know
Python, LangGraph, AgentState, nodes, embeddings, Chroma, LlamaIndex,
SearXNG, ERP adapters, prompts, model providers, exceptions, or APIs. The UI
translates technical operations into plain concepts (rules in Section 32).

The operator must always be able to answer the fifteen questions:

| # | Question | Where the UI answers it |
|---|---|---|
| 1 | What is the system doing? | Execution detail: current step with human name |
| 2 | Why is it doing it? | Step expansion: WHY field; plan preview before start |
| 3 | What has already happened? | Execution timeline (event history) |
| 4 | What is happening now? | Live event stream; dashboard "running" tiles |
| 5 | What happens next? | Plan steps list (`○` upcoming) + NEXT STEP field |
| 6 | Is it waiting for me? | Approval Center badge + "Waiting for approval" status |
| 7 | Did something fail? | Status `Failed` / `Requires attention` + error card |
| 8 | Why did it fail? | Error card: human explanation + WHY |
| 9 | What can I do about it? | Error card actions (Retry / Skip / Edit / Cancel) |
| 10 | Can I retry? | Recovery actions, bounded by backend budgets |
| 11 | Can I stop it? | Pause / Stop / Cancel controls (backend-enforced) |
| 12 | Can I approve it? | Approval Center approve/reject with impact summary |
| 13 | Can I edit it? | Edit & Rerun (creates a child execution; history immutable) |
| 14 | Can I rerun it? | Rerun / Clone on Execution History |
| 15 | Can I inspect the result? | Execution detail RESULT panel + Reports page |

Product principle: **every screen answers real state; nothing is simulated**
(Section 33).

---

## 3. Product structure (evaluated against the repository)

Proposed structure — evaluated and accepted with two additions:

```text
Dashboard            health + what needs me + what is running
Chat                 natural-language entry; creates/inspects Executions
New Execution        structured task entry + plan preview + approval gate
Executions           history list (all executions, filters, statuses)
Execution Detail     live progress + timeline + result + controls
Approval Center      pending approvals queue (actionable)
Knowledge            documents/index status (non-technical RAG view)
Reports / Results    generated outputs, linked to executions
Settings             approval mode, agent, knowledge, integrations, advanced
Notifications (tray) recent alerts — panel, not a page
Audit (in Settings)  user/agent action log — view, not a top-level page
```

Why this structure fits the code: "Executions" maps to persisted
`graph.invoke()` runs; "Knowledge" maps to `ingest_books.py` + the manifest;
"Reports" maps to `reports/*.docx`; "Approval Center" and "Chat" are new
capabilities with new backend models (they do not exist today).
Notifications and Audit were demoted to tray/Settings because a non-technical
operator needs 9 primary destinations, not 11. Navigation is left rail,
persistent; Execution Detail is reached from Executions, Dashboard, Chat, or
Notifications, never from the rail.

---

## 4. Dashboard

Sections (in order, all human-readable):

1. **SYSTEM HEALTH** — six tiles, each `OK / Degraded / Down` + one plain
   line:
   - Agent system (graph builds & executes)
   - LLM / provider (which provider configured; a lightweight ping)
   - Research (SearXNG reachable)
   - Knowledge / RAG (index non-empty; manifest healthy)
   - ERP (mock adapter responds; future: real adapter)
   - Backend (API + event stream up)
2. **RUNNING EXECUTIONS** — tiles: id, title, current step, elapsed, Watch.
3. **WAITING FOR APPROVAL** — count + oldest-waiting + Review buttons.
4. **FAILED EXECUTIONS** (recent unhandled) — card with human cause + action.
5. **RECENT COMPLETED** — id, title, duration, Open result.
6. **RECENT ACTIVITY** — merged feed of latest events + audit entries
   ("20:31 Web Research completed — 14 sources").

Language rule example (Section 32 enforces this):

```text
BAD:   ConnectionError: HTTPConnectionPool(host='localhost', port=8888)...
GOOD:  "Web research is currently unavailable."
       [View technical details]  ▸  (collapsed, on demand)
```

Health tiles poll the `health` API (Section 26) every 30 s — polling is fine
here; live SSE (Section 9) is reserved for execution detail.

---

## 5. Chat

Chat is an ENTRY AND INSPECTION surface, **never a second orchestrator**
(Section 28). It only creates or queries Executions through the same API as
New Execution.

Flow:

```text
USER:  Find active dentists in Lebanon.
CHAT:  I created Execution #1044.
       [Watch Execution]
(user leaves; execution continues server-side)
USER:  How did execution 1044 go?
CHAT:  It completed in 3m 12s and produced 1 report.
       [Open result]  [View execution]
```

Rules:

- Sending a message either (a) creates an Execution (reply links it) or
  (b) is answered as a *read-only question* about existing Executions,
  detected by backend intent lookup — never by the frontend deciding.
- Chat answers about executions come from the real Execution/event store
  ("Execution #1044 failed during Web Research: the research service was
  unavailable"), not from model memory.
- Chat history is persisted per session, but execution truth always wins.
- A chat reply NEVER blocks on execution completion; it replies immediately
  with the created/read reference.
- Duplicate-submit protection: identical pending request within a short
  window returns the existing execution (Section 31).

---

## 6. New Execution

Fields:

```text
Required:  Natural-language task (large textarea)
Optional:  Task type (auto-detected suggestion, editable)
           Files (upload → become Knowledge sources for this run)
           Knowledge sources (pick specific documents)
           ERP (which system: mock for now)
           Approval mode (this run: Approval Required | Autonomous)
```

Complex-task flow (plan preview gate — mirrors MASTER_PLAN Phase 7/8):

```text
User request → Planner (backend) → Plan preview (rendered) → User approval → Execution
```

Preview renders the REAL backend plan (schema in Section 19), e.g.:

```text
Find active dentists in Lebanon.

PLAN
✓ Search sources
○ Extract candidates
○ Verify activity
○ Deduplicate
○ Review evidence
○ Generate result
```

Buttons: **Start** (submit), **Edit** (modify task/plan steps the user is
allowed to edit — plan edits are re-validated by the backend), **Cancel**
(discard, nothing persisted beyond a draft).

Simple tasks (no plan required) skip preview and go straight to Start — the
backend decides when a plan is needed; the UI never guesses.

---

## 7. Executions (history)

Columns: ID · Title · Original request (truncated) · Status · Created ·
Started · Completed · Duration · Current step · Approval state · Result
indicator · Error indicator.

Filters: status, date range, task type, text search. Sort: newest first.

**Human-readable statuses** (canonical, backend-owned enum):

```text
Queued                 accepted, not started
Planning               planner running / plan not yet approved
Waiting for approval   blocked on a human decision
Running                a step is executing
Paused                 user-paused, resumable
Completed              finished, result available
Failed                 finished with error, no usable result
Cancelled              user cancelled before completion
Stopped                user stopped (hard stop) — terminal
Requires attention     finished with degraded/flagged result OR expired
                       approval — needs a human look
```

**Lifecycle transitions** (enforced backend-side; UI only requests them):

```text
Queued        → Planning | Cancelled
Planning      → Waiting for approval | Running | Failed | Cancelled
Waiting …     → Running (approved) | Failed (rejected) | Cancelled | Expired→Requires attention
Running       → Paused | Completed | Failed | Cancelled | Stopped | Waiting for approval (mid-run gate)
Paused        → Running | Cancelled | Stopped
Requires attn → Running (retry/resume) | Cancelled
Completed/Failed/Cancelled/Stopped  → terminal (rerun creates a NEW execution)
```

Row click → Execution Detail. Status shown as icon + text (never color
alone). Duration for open runs shows live elapsed.

---

## 8. Execution Detail (most important screen)

Layout: header (id, title, status, controls) → progress list → expandable
timeline → RESULT panel.

**Progress list** (mirrors plan steps / graph nodes):

```text
Find active dentists in Lebanon

✓ Understanding request
✓ Creating plan
✓ Searching sources
● Verifying dentists        ← current (spinner)
○ Removing duplicates
○ Reviewing evidence
○ Generating result
```

Every step expands to a fixed field set:

```text
WHAT      "Searching the web for dentists in Lebanon"
WHY       "This step gathers candidate sources (step 1 of the plan)"
STATUS    Started 20:31:05 · Completed 20:31:12 (7s) · Attempts 1/3
INPUT     "dentists in Lebanon active"   (humanized)
OUTPUT    "14 sources found · 9 kept after filtering"
AGENT     Web Research  (technical: research_agent — toggle)
TOOLS     Web Search, Page Reader  (technical: searxng_client, crawl_extractor)
SOURCES   clickable URLs / document names
RETRIES   attempt list, if any
ERRORS    none | human card (Section 15)
NEXT STEP "Extract candidates"
```

The panel separates **human-readable explanation** (default, always visible)
from **technical details** (collapsed `▸`; raw event payload, provider names,
stack traces — behind "View technical details").

**RESULT panel**: final answer (rendered markdown), provenance list
(sources), qa_flags as plain-language notices("The library had no matching material, so this used web sources only"),
generated files with Open/Download.

---

## 9. Live execution monitoring

Options evaluated:

| Option | Pros | Cons for this app |
|---|---|---|
| Polling | simplest, works everywhere | latency up to poll interval; wasteful for many open tabs; "activity" is inferred, not pushed |
| WebSockets | bidirectional, low latency | bidirectionality unneeded (commands already fit REST); harder reconnect/auth story; more server complexity |
| **SSE (Server-Sent Events)** | one-way server→client fits events; automatic browser reconnect with `Last-Event-ID`; plain HTTP (auth/proxies friendly); trivial client |

**Recommendation: SSE for the event stream + plain REST POSTs for commands**
(pause/approve/retry). Polling remains the *degradation fallback* when SSE
disconnects (reconnect → backfill via `/events?after=` → resume stream).
WebSockets are rejected until a proven need for client→server streaming.

Event feed shows the timeline exactly as structured data — never parsed from
log text (logs are for machines; the frontend consumes the event API):

```text
20:31:02  Planner started
20:31:04  Plan created
20:31:05  Research Agent started
20:31:12  14 sources found
20:31:14  Research completed
```

Delivery contract: SSE stream = ordered, at-least-once; each event carries
`event_id` + `correlation_id`; the client drops duplicates, applies events
monotonically by `event_id`, and on reconnect requests everything after the
last seen id (backfill). Heartbeat comment every 15 s keeps proxies alive.

---

## 10. Execution event model (canonical)

The backend emits structured events; the frontend only renders them.

**Event envelope** (every event, all fields required unless noted):

```text
event_id          monotonically increasing per execution (or ULID)
execution_id      owning execution
timestamp         ISO-8601 UTC
type              from the catalog below
step_id           optional — current plan/graph step
agent_id          optional — technical agent id (research, books, ...)
message           human-readable, one sentence, Section 32 rules
payload           structured, type-specific (JSON), for technical details
severity          info | warning | error
correlation_id    ties events of one logical action together
```

**Catalog:**

```text
execution.created            execution.started        execution.planned
execution.waiting_for_approval  execution.approved     execution.rejected
execution.paused             execution.resumed        execution.cancel_requested
execution.cancelled          execution.completed      execution.failed
execution.timeout            execution.requires_attention

step.started   step.completed   step.failed   step.retrying   step.skipped

agent.started  agent.completed  agent.failed

tool.started   tool.completed   tool.failed

rag.retrieval_started  rag.retrieval_completed

review.started  review.completed
```

Rules: event types are append-only (new types added, never repurposed);
`message` is UI-ready (no jargon); `payload` carries everything the
"technical details" drawer shows; ordering within an execution is guaranteed
by `event_id`; failed steps emit `step.failed` AND `execution.*` only when
the execution itself terminates.

---

## 11. Current state vs historical events

Two distinct artifacts, both persisted server-side:

| | Current execution state | Historical execution events |
|---|---|---|
| Content | status enum, current step, approval state, result/error refs | full ordered timeline (Section 10 catalog) |
| Example | `Running · Current step: Web Research` | `20:31 started · 20:32 plan · 20:33 research …` |
| Storage | `Execution` row (single source of truth; authoritative) | `ExecutionEvent` rows (append-only, immutable) |
| Read path | `GET /executions/{id}` on load | `GET /executions/{id}/events?after=` for backfill |
| Live path | initial state, then derived from events | SSE stream of new events |

Persistence rules: state transitions are transactional (state row + event
row written together, event first in the same transaction);
events are never updated or deleted (audit-grade); state can always be
reconstructed from events + latest snapshot (events are the ledger,
snapshot is the cache). UI treats the state endpoint as truth for "what is
now" and the event log as truth for "what happened".

---

## 12. Approval Center

Approval is a **real backend capability** (a step blocked on an
`ApprovalRequest` row) — never a frontend confirm dialog.

Card layout:

```text
ACTION REQUIRES APPROVAL
The agent wants to update 24 CRM records.

Why:     These records matched today's invoices.
Impact:  24 records · CRM · write action

[Approve]  [Reject]  [View Details]
```

**ApprovalRequest model fields:** approval_id, execution_id, step_id,
action (structured: kind + targets + payload), reason (human), affected_data
(count/summary), status, created_at, resolved_at, resolver (user id).

Statuses: `Pending → Approved | Rejected | Expired | Cancelled`.

Behavior: the owning step stays `Waiting for approval` until resolution or
expiry (expiry policy configurable; default: expire → execution becomes
`Requires attention`, never auto-approve). Rejecting sets execution
`Failed` with a "rejected by user" cause (recoverable via Edit & Rerun).
Approve/Reject are idempotent POSTs; the UI reflects resolution via SSE
(`execution.approved` / `execution.rejected`).

Queue page: pending (sorted by age), resolved (history), filters by
execution/impact. Dashboard badge shows pending count.

---

## 13. Approval modes and policies

**MODE A — Approval Required (default):** every action whose policy class is
not `auto` pauses for human approval.

**MODE B — Autonomous:** the system proceeds without waiting for *normal*
approvals — but this is **not an unrestricted bypass**. Approval policy is a
backend table, independent of mode:

```text
Safe read actions (search, RAG reads, ERP reads)      → auto
External writes (ERP/CRM updates, emails, exports)     → approval
Delete operations                                     → approval
High-impact actions (bulk > N records, irreversible)  → approval (cannot be
                                                         downgraded to auto)
```

**What Autonomous Mode changes exactly:** policy class `approval` acts like
`auto` for actions NOT marked high-impact; high-impact actions STILL require
approval in Autonomous Mode; every auto-approved action is logged to the
audit trail with `mode: autonomous` so it is reviewable later.

UI must be unmistakable:

```text
AUTONOMOUS MODE
The agent can execute permitted actions without waiting for your approval.
High-impact actions still ask for confirmation.
[Enable]  (confirm dialog + audit entry)
```

Mode is settable globally (Settings) and per-execution (New Execution); the
effective mode is frozen on the execution at creation and shown on its
detail page (toggling global mode does not retroactively change running
executions — see Section 31).

---

## 14. Pause / Resume / Stop / Cancel

Meaning (backend-enforced, UI only requests):

```text
Pause    finish current step, do not start the next one. Resumable.
Resume   continue from where paused.
Cancel   graceful: abort before completion; partial results kept; status
         Cancelled (terminal, rerun creates a new execution).
Stop     hard stop: terminate now even mid-step; best-effort cleanup;
         status Stopped (terminal).
```

Difference Cancel vs Stop: Cancel = "I no longer want this outcome" (clean,
waits for step boundary preferred); Stop = "halt immediately" (may discard
in-flight step work).

Implementation requirement: controls map to backend control endpoints
(`/pause`, `/resume`, `/cancel`) honored by the execution runtime between
steps AND inside long steps via a cancellation checkpoint; the UI never
locally fakes a stopped status — it optimistically shows `Pause requested…`
until the real state event arrives.

---

## 15. Error UX

Every error the user sees is a structured error card with five fields
(populated by the backend error mapper, not invented in the frontend):

```text
WHAT happened   "Web research could not be completed."
WHY             "The research service could not be reached."
Affected step   "Searching sources (step 3 of 6)"
Recoverable     yes / no
You can         [Retry] [Skip] [Edit Task] [Cancel Execution]

Technical details ▼     (collapsed by default: exception type, message,
                         stack trace, event payload)
```

Rules:

- Raw stack traces are NEVER shown by default (Section 32).
- If no action is possible, the card says why ("This execution is already
  finished") instead of hiding buttons.
- The action buttons are enabled only for actions the backend currently
  accepts (disabled state explains why — e.g. "Retry used: 2 of 2").
- One execution can accumulate multiple error cards (per step) — all remain
  visible on the detail timeline.
- Error text comes from `ExecutionError` (Section 25): `human_title`,
  `human_why`, `technical` payload, `recoverable`, `suggested_actions[]`.

---

## 16. Retry / recovery

Supported recovery actions (all backend endpoints; UI reflects backend
budgets — **the GUI never invents unlimited retries**):

```text
Retry step        re-run the failed step (counts against step budget)
Retry execution   restart failed execution from the failed step (same budget)
Resume            from Paused / Requires attention
Skip step         mark step skipped, continue plan (only if plan allows)
Edit Task         open the task/plan for edit → creates a NEW execution
Cancel            abandon
```

Budget display comes directly from MASTER_PLAN §5.6: `attempt X of Y`
(`MAX_TASK_RETRIES=2`), `replan N of M` (`MAX_REPLANS=1`). When the budget
is exhausted, Retry is disabled with the text: "Automatic retries are used
up. Edit the task or cancel." — mirroring the backend's bounded
retry/replanning exactly.

`step.retrying` events drive the UI's attempt indicator (attempt number,
next retry time if delayed).

---

## 17. Edit / Clone / Rerun / Archive (immutable history)

Executions are **immutable records**. Actions never mutate the past:

```text
View        open detail (read-only for terminal executions)
Edit & Rerun  pre-fills New Execution from #1042 → creates #1043
              (parent_execution_id = #1042, audit entry)
Rerun       identical request → new execution, parent link, "attempt 2"
Clone       copy task + options as an editable draft (not auto-started)
Archive     hide from default list; still reachable via filter; reversible
Delete      permanent; ONLY allowed for executions with no external side
            effects AND not an approval/audit source; double-confirm;
            preferred default is Archive
```

```text
Execution #1042   [Edit & Rerun]  →  Execution #1043 (Parent: #1042)
```

Safety rule: **Archive is the default destructive-looking action; Delete is
buried, double-confirmed, audited, and refused for executions that produced
ERP writes or approvals** (their history is legally/operationally relevant).
Deleting an execution never deletes its produced files implicitly — the
confirmation lists what would remain orphaned (outputs) and offers archive
instead.

---

## 18. Agent activity visualization

Backend view (technical) vs UI view (human), mapped dynamically from an
event-driven agent registry — **never hardcoded screen strings:**

```text
Backend:  Planner → Research Agent → Web Search → Books Agent →
          Synthesizer → Reviewer

UI:       Creating plan → Researching the web → Checking internal
          knowledge → Combining findings → Reviewing result
```

Rules:

- Display name is a field on the backend agent registry (event
  `agent_id` → `display_name`); unknown ids degrade to a prettified id
  ("Agent research" → "Research Agent"), never to a blank.
- Every step shows the human name by default; a toggle reveals the
  technical name (`research_agent`) in the technical details drawer.
- Flow rendering (vertical chain in the detail view, live) is derived from
  actual `step.*`/`agent.*` events of THIS execution — same code renders
  single-agent and plan-driven executions; no hardcoded flow diagrams.

---

## 19. Planner visualization

The GUI renders the backend's structured plan (MASTER_PLAN §5.2) — the
frontend never generates or interprets plan semantics itself.

Conceptual schema exposed to the UI:

```text
Plan:  plan_id · execution_id · description · steps[] · status
PlanStep:  step_id · what (human) · why (human) · depends_on[]
           · expected_output · approval_requirement · status
           · technical {task_id, capability}   ← drawer only
```

Mapping: MASTER_PLAN plan tasks (`task_id`, `capability`, `objective`,
`depends_on`, `expected_output`, `done_when`) are translated server-side
into the display fields (`what` ← objective, `why` ← objective/goal context,
status from step events). The plan preview (Section 6) and the detail-page
progress list (Section 8) render the SAME plan object at different moments
(`planned` vs executing).

Statuses per step: `pending → active → done | failed | skipped | waiting`
(icons `○ ● ✓ ✗ ⊘` + approval variant `⏸`). Dependencies render as a
simple ordered list with inline "(after 1, 2)" references; only when a plan
actually branches does the UI draw connectors — no decorative graphs.

---

## 20. Knowledge (non-technical RAG view)

Sources table:

```text
Document · Type · Added · Status · Last indexed · Errors
```

Actions: **Add document** (upload → ingestion job) · **Remove document** ·
**Re-index** · **View source** (open original file) · **Search knowledge**
(natural-language probe that shows which documents would be retrieved —
"looks under the hood" on demand).

Rules: the normal view NEVER exposes Chroma, embeddings, vector distances,
collection names, or scores. An expandable **Advanced** section may show
chunk counts, index size, model name — collapsed by default, aimed at a
curious expert, not the operator.

Data source: extend `ingest_books.py`/manifest knowledge with a per-document
status (indexed / indexing / failed) written during ingest; Execution Detail
eventually shows per-run retrieval context as human lines: "Knowledge
sources searched: 2 documents · Documents retrieved: 3 · Relevant excerpts
used: 2" (fed by `rag.retrieval_*` events, not by frontend inference).

---

## 21. Reports / Results

Every output the system produces (`.docx` studies, final answers, exports)
is an `ExecutionOutput` row: type, title, created, execution link, file ref.

List: thumbnail/title · type · created · execution id · status.
Actions: **Open** (in-app viewer/download) · **Download** · **Rename** ·
**Archive/Delete** (same safety rules as Section 17) · **Regenerate**
(creates a new execution with the same inputs, parent-linked) ·
**View execution** (jump to the producing execution).

Hard rule: **every result stays linked to the execution that created it**
and shows provenance (sources) on its detail — unlinkable orphans are a bug.
Current files in `reports/` are adopted by backfilling an
`ExecutionOutput` with `execution_id` when the feasibility agent registers
its report (small backend hook in Phase GUI-1/2).

---

## 22. Settings

```text
GENERAL      Approval mode (Required | Autonomous, Section 13) ·
             Notifications (Section 23) · language/time format
AGENT        Model/provider (read-only display + switch for expert mode:
              gemini | ollama + model name) · Execution limits (max parallel,
              timeout) · Retry limits (display of MASTER_PLAN budgets;
              editing is expert-only and bounded)
KNOWLEDGE    RAG settings (top-k, min score — expert-only, with plain
              explanations) · default sources
INTEGRATIONS ERP (adapter: mock now / future real — status only until
              Phase 15) · SearXNG URL (health only)
ADVANCED     Technical configuration (raw .env-adjacent values),
              explicitly gated (Section 17-style double confirm + audit),
              hidden from ordinary users by role (Section GUI-18)
```

Principle: anything an ordinary user can break lives behind an explanation
("Higher values make answers more thorough but slower") + confirm; anything
cryptographic/credential-related is view-status-only, never editable in v1
stays in `.env`.

---

## 23. Notifications

Trigger → notification (bell tray + optional desktop/email later):

```text
execution completed   → "Execution #1044 finished — report ready"
execution failed      → "Execution #1044 stopped during Web Research"
approval required     → "Waiting for your approval: update 24 CRM records"
execution paused      → "Execution #1043 is paused"
requires attention    → "Execution #1045 needs a look (result flagged)"
```

Rules: generated from execution state transitions server-side (the user
never watches the detail page to learn an outcome); tray marks read/unread;
clicking navigates to the exact execution/approval; duplicate suppression
(one notification per state transition); quiet hours optional. Delivery
channels beyond the in-app tray are future work — the event/audit layer
makes them cheap later.

---

## 24. Audit trail

Append-only `AuditEvent` log (distinct from execution events — these record
HUMAN and privileged actions):

```text
user created execution · user approved action · user rejected action ·
user paused execution · user cancelled execution · user changed settings ·
user enabled/disabled autonomous mode · agent executed action (esp. ERP
writes) · execution failed · execution retried · execution rerun
(cloned) · approval expired · document added/removed
```

Fields: audit_id, timestamp, actor (user id or "system:<agent>"), action,
target (execution/approval/setting id), mode (approval/autonomous), summary
human line, payload (technical, drawer). View lives in Settings → Audit
(filters: actor, action, date, target). Audit entries are never editable or
deletable through the UI. ERP/external writes ALWAYS produce an audit entry
— this is the primary accountability mechanism for Autonomous Mode.

---

## 25. Data model (conceptual — NOT implemented)

```text
Execution          1 — * ExecutionStep
                   1 — * ExecutionEvent
                   1 — * ApprovalRequest
                   1 — * ExecutionError
                   1 — * ExecutionOutput
                   0..1 Plan        (complex tasks)
                   0..1 parent Execution  (rerun/clone lineage)

ExecutionStep      step_id, order, what/why (display), status, attempts,
                   started_at, finished_at, agent_id, error → ExecutionError

ExecutionEvent     append-only envelope (Section 10) — FK: execution, step

Plan               plan_id, execution_id, description, status
PlanStep           step_id, plan_id, what, why, depends_on[],
                   expected_output, approval_requirement, status

ApprovalRequest    approval_id, execution_id, step_id, action, reason,
                   affected_data, status, created_at, resolved_at, resolver

ExecutionError     error_id, execution_id, step_id?, human_title,
                   human_why, technical (payload), recoverable,
                   suggested_actions[]

ExecutionOutput    output_id, execution_id, kind (report|answer|export),
                   title, file_ref, created_at, provenance[]

Notification       notification_id, type, execution_id?, message,
                   created_at, read_at

AuditEvent         audit_id, timestamp, actor, action, target, mode,
                   summary, payload   (append-only)

KnowledgeSource    doc_id, filename, type, added_at, status, last_indexed,
                   error?
```

Relationships: an Execution OWNS steps, events, approvals, errors, outputs;
Plan belongs 1:1 to an Execution; events reference step/agent but nothing
references the frontend; AuditEvent references targets loosely (id strings)
so audit survives deletions. `AgentState` is deliberately NOT modeled — it
stays an in-process runtime structure (one canonical state rule; persistence
here is execution-level, not agent-level).

---

## 26. Backend / API boundary (conceptual resources — NOT implemented)

The frontend NEVER imports agent internals; it speaks only this API:

```text
/executions                 POST (create) · GET (list, filters)
/executions/{id}            GET (state snapshot) · DELETE (archive/delete)
/executions/{id}/steps      GET
/executions/{id}/events     GET (?after= backfill) · SSE (live stream)
/executions/{id}/pause      POST   /executions/{id}/resume   POST
/executions/{id}/cancel     POST   /executions/{id}/stop     POST
/executions/{id}/retry      POST (step_id?; budgets enforced server-side)
/executions/{id}/rerun      POST (child) · /clone POST (draft)

/plans/{id}                 GET (preview + live status)
/approvals                  GET (queue) · /approvals/{id}/approve|reject POST
/reports                    GET · /reports/{id} GET (download)
/knowledge                  GET · POST (upload) ·
                            /knowledge/{id}/reindex | DELETE
/health                     GET (six tiles)
/settings                   GET · PATCH (role-gated, audited)
/audit                      GET (filters)
/notifications              GET · POST /notifications/{id}/read
/chat                       POST (creates execution or read-only answer)
```

Contract rules: responses carry display-ready human message fields
(Section 32); errors use the `ExecutionError` shape; control endpoints are
idempotent; SSE shares auth with REST; implementation will version paths
(`/api/v1/...`).

---

## 27. Frontend technology

Inspection result: **no frontend framework exists** (no `package.json`, no
web deps in `requirements.txt`) — free choice.

| Criterion | Raw React (CDN/no build) | Next.js | **Vite + React (TS)** |
|---|---|---|---|
| Simplicity | low tooling but manual everything | high complexity (SSR/routing server) | low: dev server + build only |
| Real-time SSE | fine | fine | fine |
| Execution visualization | manual DOM/state | fine | fine (component model + TS types shared with API) |
| Forms | manual | fine | fine |
| Maintainability | poor at scale | good | good |
| Future scalability | poor | best (but pays SSR/mode complexity now) | good (can migrate later) |
| Dev effort | highest long-term | highest upfront | **lowest** |
| Fit for an internal operator tool served by a Python API | no SSR need | SSR/auth/server routes redundant with our API | SPA served as static files — ideal |

**Recommendation: Vite + React + TypeScript**, a static SPA served beside
the future API. Next.js is rejected: SSR, server routes, and its rendering
model buy nothing for an authenticated internal dashboard whose data comes
from a Python API + SSE. React over Vue/Svelte: largest ecosystem, easy
hiring, first-class SSE/state libs — not hype, just lowest risk. Typescript
is required (not optional) because the event/data model (Sections 10/25) is
cross-contract with the backend.

New dependencies are frontend-only (Node devDeps), introduced at GUI-3
with justification; **backend Python dependencies unchanged until GUI-1/2**
(API framework decision, there: recommend FastAPI + uvicorn, the standard
SSE-friendly Python pairing — final call at implementation time).

---

## 28. Responsibility boundary

```text
BACKEND (agent system + execution/API layer)      FRONTEND (Control Center)
──────────────────────────────────────────────    ──────────────────────────
planning                                          display
execution & dependency order                      user input
AgentState & step state                           approval decisions
event emission & ordering                         execution control requests
approvals (policy enforcement)                    visualization (real data)
bounded retries & replanning                      human explanations of
cancellation enforcement (checkpoints)              backend-provided texts
agents, RAG, ERP                                  file download/viewing
persistence (executions, events, audit)            notifications display
error mapping (human_title/why/suggested)
```

**The frontend must NOT become a second orchestration engine**: it never
decides step order, never retries on its own initiative, never fabricates
status transitions (optimistic labels like "Pausing…" are clearly request
state, replaced by real events), never parses logs, never talks to
LLM/RAG/ERP directly. Every button maps to exactly one backend endpoint.

---

## 29. Phased GUI implementation roadmap

Ordering rationale (from inspection): the backend has **no execution model,
no events, no API** — so phases 1–2 are backend-first; then the read path
(history → detail → live) validates the contract before any write path
(create → control → approvals); chat comes only after creation exists;
planner visualization waits on MASTER_PLAN Phases 7/8; settings/notifications
/auth/hardening close. 19 phases (the suggested 18 merged error-UX into
control, split foundation into execution+API, and added hardening last).

Dependency graph:

```text
GUI-1 → GUI-2 → GUI-3 → GUI-4 → GUI-5 → GUI-6 → GUI-7
                              └→ GUI-8 → GUI-9 → GUI-10
                                        └→ GUI-11
GUI-5 ─→ GUI-12 (needs MASTER_PLAN 7/8)
GUI-2 → GUI-13, GUI-14 (read-only pages, parallel after GUI-3)
GUI-9/10 → GUI-15 → GUI-16 → GUI-17 → GUI-18 → GUI-19
```

### Phase GUI-1 — Execution & event foundation (backend)

### Goal
Introduce the Execution lifecycle and structured event emission inside the
backend so every future run can be observed as data.

### Why
Nothing observable exists: `graph.invoke()` is synchronous and silent
except prints. Every GUI feature depends on this (Sections 10–11).

### Dependencies
MASTER_PLAN Phase 13 pattern (privacy-safe logging) is the design precedent;
no other backend phase required.

### Backend changes
Execution runner wrapping `build_graph().invoke()` in a background job;
state machine (Section 7 transitions); event emitter emitting the Section 10
catalog at graph node boundaries (instrumented node wrapper, fail-soft);
human-name registry (Section 18); error mapper → `ExecutionError` shape.

### Frontend changes
None.

### Data model
`Execution`, `ExecutionStep`, `ExecutionEvent`, `ExecutionError` (Section 25)
in SQLite (default) — local, zero-infra, migration-friendly.

### API/events
None exposed yet (internal emitter only); event ordering/backfill semantics
defined and unit-testable (event_id monotonic per execution).

### Files likely affected
NEW `app/execution/` (runner, events, errors), `app/graph.py` (node wrapper
hook only), NEW `scripts/test_gui1_events.py`, `docs/TESTING.md`.

### Deterministic tests
State-machine legal/illegal transitions; event ordering & monotonic ids;
fail-soft emitter (logging failure never breaks a run); error mapper human
strings for every known failure class; wrapper does not alter node outputs
(regression: all 101 backend tests green unchanged).

### Integration tests
One real execution (books route, fake LLM) produces the expected event
sequence `execution.created → started → step.* → completed`.

### Failure cases
Node raises → `step.failed` + `execution.failed` events, run terminates
cleanly; emitter exception swallowed; duplicate completion events deduped.

### Security considerations
Events carry no secrets/question text beyond what answer already contains;
follows ERP-log privacy rule; file paths internal only.

### Acceptance criteria
Every execution (all 4 routes) yields a complete legal event sequence;
zero quota for deterministic suite; backend regression 101+10 green.

### Definition of done
Event log for a test run can be asserted end-to-end from tests alone.

### Explicit exclusions
No API server, no persistence UI, no SSE, no pause/cancel logic yet
(terminal states only), no MASTER_PLAN code changes.

### Phase GUI-2 — Persistence + API boundary (backend)

### Goal
Expose executions, events, and controls through a versioned HTTP API with
SSE — the contract the frontend will consume.

### Why
The GUI must never import agent internals (Section 26); the API is the
single seam between the two systems.

### Dependencies
GUI-1 (execution model + events must exist to serve).

### Backend changes
API server (recommended FastAPI + uvicorn, added as backend deps with
justification) serving Section 26 resources; SQLite persistence wired to
GUI-1 tables; SSE endpoint with `Last-Event-ID` backfill; control endpoints
(pause/resume/cancel implemented at step boundaries — full mid-step
checkpoints land with GUI-9); archive/delete rules (Section 17) enforced
server-side; `health` aggregating the six checks.

### Frontend changes
None.

### Data model
Adds `Plan`/`PlanStep` tables (created, unused until GUI-8/12),
`ExecutionOutput` backfill hook for `reports/*.docx` (Section 21).

### API/events
All Section 26 paths (subset: read + create + cancel initially); every list
response includes human message fields; errors → `ExecutionError` JSON.

### Files likely affected
NEW `app/api/` (server, routes, sse), `app/config.py` (ports/paths),
`requirements.txt` (+fastapi, uvicorn), NEW `scripts/test_api2_*.py`,
`docs/TESTING.md`, `.env.example` (placeholder only).

### Deterministic tests
Route contract tests with in-process TestClient (fake LLM): list/get/steps/
events backfill correctness; SSE emits ordered events then heartbeats;
control endpoint idempotency; archive/delete safety rules (refuse delete of
approval/ERP-write executions); 404/409/422 shapes; auth stub accepts/rejects
(GUI-18 replaces stub).

### Integration tests
Full run: create → poll events → terminal state; SSE reconnect with
`Last-Event-ID` receives no gaps/duplicates.

### Failure cases
Backend restart mid-run → execution resurfaces as `Requires attention`
(never silently lost); SSE client disconnect → server closes stream;
concurrent create with same task → dedupe key returns existing execution.

### Security considerations
Bind localhost by default; no secrets in any response; payload size limits;
CORS locked to the GUI origin; technical details gated behind the same auth
as the rest.

### Acceptance criteria
Section 26 contract fully exercisable by tests; regression green; SSE
guarantees (ordered, backfillable, at-least-once) proven by tests.

### Definition of done
A curl-only client can create, watch, and cancel an execution.

### Explicit exclusions
No frontend, no approvals endpoints (GUI-10), no chat (GUI-11), no auth
roles (GUI-18).

### Phase GUI-3 — Control Center shell (frontend scaffold)

### Goal
Vite + React + TS app with routing, layout, API client, design tokens, and
the nine destinations as stubs.

### Why
Every later phase drops into a stable shell; the typed API/event client is
built once here (Section 26 contract → generated TS types).

### Dependencies
GUI-2 (API exists to generate types against).

### Backend changes
Static file serving of the built SPA (same origin as API — avoids CORS in
production); nothing else.

### Frontend changes
NEW frontend app: router, left rail (Section 3), theme, error boundary,
API client layer (fetch + typed models), notification bell placeholder,
`docker`-free dev command (`npm run dev` proxies to API).

### Data model
None (client types only, mirrored from Section 25).

### API/events
Client library implements Section 26 paths + SSE consumer util
(reconnect/backfill logic lives HERE, once).

### Files likely affected
NEW `frontend/` (vite config, router, api client, layout), `package.json`,
`.gitignore` (+ node_modules, dist), `docs/TESTING.md`.

### Deterministic tests
Component/unit tests (vitest): rail navigation, API client typed against
mocked responses, SSE util reconnect math (event_id dedupe/backfill) with
fake EventSource.

### Integration tests
Shell loads against the real API (health tile shows real data), all nine
routes render their placeholder without console errors.

### Failure cases
API down → global error banner with Retry (Section 15 style), app still
navigable; SSE unavailable → client falls back to polling (Section 9).

### Security considerations
No tokens in localStorage beyond session handling decided in GUI-18;
secrets never enter the frontend bundle; dev proxy localhost-only.

### Acceptance criteria
Shell renders with real health data; typed client compiles against the
API contract; zero backend behavior changes.

### Definition of done
`npm run build` produces a static bundle the backend can serve.

### Explicit exclusions
No real pages yet (stubs only), no state management beyond fetch cache, no
auth UI.

### Phase GUI-4 — Execution history page

### Goal
Full list/filters/statuses of executions (Section 7) — the first read path
that proves the contract end-to-end.

### Why
History is the cheapest real screen: pure GET, no live updates, no writes;
it validates statuses, human text, and navigation before anything complex.

### Dependencies
GUI-3 (shell + API client).

### Backend changes
List endpoint filters (status, date, text) + pagination if missing; status
labels served as data (not frontend-computed).

### Frontend changes
Executions page: table, filters, status badges (icon+text), live elapsed
for open runs (client ticker over server state), row → detail link
(detail still stub), empty/loading/error states (Section 15 patterns).

### Data model
None new.

### API/events
`GET /executions` (filters); no SSE yet.

### Files likely affected
`frontend/src/pages/Executions.tsx`, API client additions,
`app/api/` (filters), NEW `scripts/test_api4_list.py`, frontend unit tests.

### Deterministic tests
Filter/pagination contract tests; frontend table renders each status with
the exact human label; unknown future status degrades to prettified string
never blank; error state renders ExecutionError card.

### Integration tests
Seed 3 executions (via API), verify list shows all with correct fields and
ordering; archive one → hidden from default view, visible with filter.

### Failure cases
API down → page-level retry card; huge list → paginated not frozen;
execution created while page open → user can refresh (live is GUI-6's job,
explicitly not faked here).

### Security considerations
List responses never include technical payloads (they live on detail);
delete/archive gated client-side but enforced server-side.

### Acceptance criteria
All Section 7 columns present with real data; statuses exactly match the
canonical enum; zero fake data anywhere.

### Definition of done
A user can find any execution and reach its detail page.

### Explicit exclusions
No live updates, no row actions beyond navigation (GUI-9/15), no bulk ops.

### Phase GUI-5 — Execution detail + timeline

> **Recovery status (2026-10): PARTIAL.** A truthful execution-sequence view
> shipped early during the workspace recovery: the backend
> `/api/v1/executions/{id}/sequence` endpoint (stage list rebuilt from the
> append-only event log, including `routing.decided` and `qa.completed`)
> and a **Tasks & Sequence** page render the created → supervisor → routed
> → agents → qa → completed ordering for real executions, plus honest 404
> handling (`scripts/test_gui5_operations.py`, 14 tests). The full GUI-5
> detail page — step cards with WHAT/WHY/INPUT/OUTPUT/RETRIES/ERRORS, the
> expandable timeline, technical drawer, and RESULT panel — is still
> **NOT IMPLEMENTED** and remains this phase's definition of done.

### Goal
The most important screen (Section 8): progress list, expandable step
timeline, human/technical separation, result panel.

### Why
Every later feature (live, controls, approvals, planner) lands ON this
page; its field set (WHAT/WHY/STATUS/INPUT/OUTPUT/AGENT/TOOLS/SOURCES/
RETRIES/ERRORS/NEXT STEP) defines the display contract.

### Dependencies
GUI-4 (navigation), GUI-2 (`/events?after=` backfill for the timeline).

### Backend changes
Steps endpoint returns display fields (what/why/duration/attempts);
step events must carry INPUT/OUTPUT human summaries (tighten GUI-1 emitter
where gaps found); agent registry lookup (display + technical names).

### Frontend changes
Detail page: header, progress list (icons per Section 19 statuses),
timeline grouped by step with expandable cards, technical-drawer toggle,
RESULT panel (markdown answer, provenance, qa_flags-as-notices, outputs),
error cards (Section 15), agent activity flow (Section 18, static render
from stored events — live comes in GUI-6).

### Data model
None new (consumes ExecutionStep/Event/Error/Output).

### API/events
`GET /executions/{id}`, `/steps`, `/events?after=`; event → display mapper
shared with timeline component.

### Files likely affected
`frontend/src/pages/ExecutionDetail.tsx` + step/timeline/result components,
API client, `app/api/` steps shaping, `app/execution/events.py` (payload
completeness), NEW frontend unit tests.

### Deterministic tests
Event→display mapping table (every catalog type renders a defined card);
unknown event type renders generic fallback (never crash); technical drawer
toggle shows raw payload; human/technical name swap; error card field
completeness (WHAT/WHY/actions) from canned ExecutionError fixtures;
markdown renderer sanitized (no script injection).

### Integration tests
Real execution → open detail → timeline matches actual event log exactly
(same count/order); result shows real sources + report links.

### Failure cases
Events missing (older execution) → timeline shows what exists + "details
unavailable" note, no fake steps; oversized payload → drawer paginates;
report file missing → download button explains (recoverable).

### Security considerations
Markdown sanitized; file downloads served through API with auth (not raw
filesystem paths); technical drawer shows payload but never secrets
(backend already strips — asserted).

### Acceptance criteria
Field set from Section 8 complete on every step card; human vs technical
distinction visible; timeline fidelity proven by test against raw events.

### Definition of done
A finished execution is fully explainable on this page without logs.

### Explicit exclusions
No controls (GUI-9), no SSE (GUI-6), no plan preview edit (GUI-12),
no qa_flags until MASTER_PLAN Phase 5 lands them in production.

### Phase GUI-6 — Live execution updates (SSE)

### Goal
Watch executions while running: the event feed from Section 9 streams into
progress list, timeline, and header — the "watch it work" experience.

### Why
The user's explicit priority: see what is happening NOW without refreshing.

### Dependencies
GUI-5 (renders the events), GUI-2 (SSE endpoint).

### Backend changes
SSE hardening: heartbeats, per-connection backfill (`Last-Event-ID` /
`?after=`), event size caps, connection limits; status-snapshot message on
connect so the header is correct immediately.

### Frontend changes
SSE consumer (from GUI-3 util) wired to detail page; monotonic event_id
dedupe; "requesting" labels → real events replace them; browser-tab title
shows running status; polling fallback loop when SSE drops (Section 9);
live elapsed timers; dashboard tiles NOT in scope (polling there, GUI-7).

### Data model
None new (event log already append-only).

### API/events
SSE stream subscribed to full catalog (Section 10); client-side reducer
applies events to (state + timeline) without refetch.

### Files likely affected
`frontend/src/hooks/useExecutionStream.ts`, detail page integration,
`app/api/sse.py` (hardening), NEW `scripts/test_gui6_sse.py`, frontend tests.

### Deterministic tests
Reducer tests: ordered apply, duplicate drop, out-of-order backfill merge,
state-machine labels derived only from events; reconnect logic with fake
EventSource (disconnect → backfill after last id → no gaps/dupes);
polling fallback engages and recovers; SSE error mid-run does not blank
the page.

### Integration tests
Run real execution while subscribed: UI shows step transitions within 2 s;
kill SSE mid-run → automatic reconnect → no missed final `completed`;
browser refresh mid-run → backfilled timeline complete.

### Failure cases
Backend restart → stream closes, fallback polls, page shows "reconnecting…"
(honest request state, Section 28); event flood → client batches renders
(coalesce) without losing final state; stale stream (id regression) → full
refetch instead of corrupt apply.

### Security considerations
SSE requires same auth as GET detail; events respect per-user visibility
(GUI-18); no secrets in stream; connection idle timeouts.

### Acceptance criteria
Live watch works for all four routes; refresh/disconnect proven lossless by
tests; zero fabricated activity (every visible change traces to an event id).

### Definition of done
The Section 9 example timeline renders from a real run, live.

### Explicit exclusions
No controls, no dashboard live tiles (polling), no chat streaming (GUI-11
reuses this consumer but is built later).

### Phase GUI-7 — Dashboard

> **Recovery status (2026-10): PARTIAL.** Two of this phase's capabilities
> shipped early during the workspace recovery: real bounded service-health
> probes (`app/services_health.py`, `GET /api/v1/services`: backend, SQLite
> storage, Ollama, SearXNG, ERP adapter — with state/latency/reason/next
> action) and event-derived agent activity (`GET /api/v1/agents/activity`),
> rendered on dedicated **Services** and **Agents** pages
> (`scripts/test_gui5_operations.py`). NOT yet implemented: the aggregated
> `/dashboard` endpoint (status counts, recent panels, merged activity
> feed), six-tile home screen, 30 s polling, and stale-data handling —
> the current pages poll their endpoints directly.

### Goal
Section 4 overview: six health tiles, running/approval/failed/recent panels,
activity feed — the operator's home screen.

### Why
By now history/detail/live exist; the dashboard is aggregation over proven
endpoints, so it cannot fake what it summarizes.

### Dependencies
GUI-4 (list API) + GUI-6 (event feed for activity); `health` from GUI-2.

### Backend changes
Aggregations endpoint (counts by status, recent N, activity feed merge of
events + audit); health checks with human status lines + degraded reasons.

### Frontend changes
Dashboard page per Section 4; 30 s polling for tiles + counts; degraded
health tiles show human line + "View technical details"; activity feed
renders event messages directly (they are already human — Section 10).

### Data model
None new.

### API/events
`GET /health`, `GET /dashboard` (aggregates), tail of events for feed.

### Files likely affected
`frontend/src/pages/Dashboard.tsx`, API client, `app/api/health.py`,
`app/api/dashboard.py`, NEW `scripts/test_api7_dashboard.py`, frontend tests.

### Deterministic tests
Aggregation math from seeded fixtures (counts per status);
health → human-line mapping for every degraded cause (SearXNG down →
"Web research is currently unavailable", never ConnectionError);
activity feed ordering; empty-state panels (no executions yet).

### Integration tests
Seed mixed executions + kill a dependency (e.g. point SearXNG at a dead
port) → tile flips to Degraded with correct human line; restore → back to OK.

### Failure cases
Dashboard endpoint down → cached last values + honest "stale since HH:MM";
all six systems down → still renders (per-tile errors, not page crash).

### Security considerations
Health detail omits URLs/credentials (hostnames behind technical drawer);
aggregates respect user visibility scope.

### Acceptance criteria
All Section 4 panels present with real data; language rule from Section 4
demonstrated by tests; no panel ever shows a raw exception by default.

### Definition of done
Opening the app answers "what is happening / what needs me" in 3 seconds.

### Explicit exclusions
No widgets customization, no historical charts/metrics (future), no
notifications tray (GUI-17).

### Phase GUI-8 — New Execution (create + plan preview)

### Goal
Section 6: task entry with optional fields, plan preview gate for complex
tasks, Start/Edit/Cancel.

### Why
First WRITE path: creation is the entry point for every subsequent feature
(control, approval, chat) and the first consumer of the plan contract.

### Dependencies
GUI-2 (create + `/plans/{id}`), GUI-6 (redirect to live detail after
create); plan rendering data needs MASTER_PLAN Phase 7 **only for real
plans** — until then the backend simply never returns a plan (simple path),
so this phase ships complete with plans absent.

### Backend changes
Create endpoint accepts task/options; optional plan step (trigger planner
when task classified complex — planner itself from MASTER_PLAN Phases 7/8;
API returns `plan: null` meanwhile); file upload → per-run knowledge sources;
duplicate-request dedupe key (Section 5/31).

### Frontend changes
New Execution page: textarea, optional sections, plan preview renderer
(checklist per Section 6, reusing GUI-5 components), Start → navigate to
detail; Edit → returns to form with plan-step editing only where backend
marks steps editable; Cancel → discard draft.

### Data model
Consumes `Plan/PlanStep` (created in GUI-2); draft state is client-side
only.

### API/events
`POST /executions`, `GET /plans/{id}`, upload endpoint; first event
`execution.created` (or `execution.planned` when planned) drives the
post-create UX.

### Files likely affected
`frontend/src/pages/NewExecution.tsx` + plan preview component, API client,
`app/api/executions.py`, NEW `scripts/test_api8_create.py`, frontend tests.

### Deterministic tests
Form validation (empty task, oversized task, bad file type); plan preview
renders canned plan fixtures (statuses, dependencies "(after 1)"); plan
`null` → preview skipped without error; duplicate submit → same execution
id returned; per-execution approval mode field round-trips.

### Integration tests
Create simple → lands on live detail with events streaming; create with
file → file listed under that execution's knowledge scope.

### Failure cases
Create fails (backend down) → form keeps input, retry possible (Section 15);
planner unavailable → task still starts simple with notice (backend default
plan behavior from MASTER_PLAN §5.2); upload rejected mid-form → other fields
preserved.

### Security considerations
Upload type/size limits + malware-minded validation; options never accept
raw node names/capabilities (whitelist enforced server-side, Section 19).

### Acceptance criteria
Simple creation works end-to-end today; plan preview path covered by
fixture tests so it activates untouched when MASTER_PLAN Phase 7 lands.

### Definition of done
A non-technical user creates and starts any task from one screen.

### Explicit exclusions
No chat (GUI-11), no batch/queue scheduling (future), no plan semantics
interpreted in frontend beyond rendering display fields.

### Phase GUI-9 — Controls & recovery (pause/resume/cancel/stop + error/retry UX)

### Goal
Sections 14–16 made real: backend-enforced controls, error cards with
actions, budget-aware retry/skip/edit.

### Why
Control without backend enforcement is exactly the "fake monitor" anti-
pattern (Section 33); this phase adds the missing backend enforcement and
the full error UX together (buttons and error cards are one interaction).

### Dependencies
GUI-2 (control endpoints exist, mid-run enforcement incomplete), GUI-5/6
(errors displayed; live state replaces optimistic labels).

### Backend changes
Cancellation checkpoints inside long steps (research crawl, LLM invoke,
feasibility write); pause gate between steps (state machine from GUI-1 gets
`Paused`); retry/skip endpoints enforcing MASTER_PLAN §5.6 budgets
(`attempt X of Y`, `replan N of M` served as data); error mapper completes
`suggested_actions[]` per failure class (availability-driven enablement).

### Frontend changes
Header controls (Pause/Resume/Stop/Cancel with confirm dialogs + plain-English
consequences from Section 14); optimistic "Pause requested…" labels replaced
by events; error cards with enabled/disabled action buttons + reasons;
budget text display; Edit Task → New Execution prefilled (GUI-8 route).

### Data model
`ExecutionError.suggested_actions` consumed; state machine gains
Paused/stopped transitions (already in Section 7 enum).

### API/events
`pause|resume|cancel|stop|retry|skip` endpoints; events `execution.paused/
resumed/cancel_requested/cancelled/timeout`, `step.retrying/skipped`.

### Files likely affected
`app/execution/runner.py` (checkpoints), `app/api/` (control routes),
`frontend/src/pages/ExecutionDetail.tsx` (controls + error cards), NEW
`scripts/test_gui9_controls.py`, frontend tests.

### Deterministic tests
State machine: every control illegal in every illegal state (409 + human
reason); cancellation observed between steps AND inside a fake long step
(checkpoint fake); budget exhaustion → retry disabled + correct text;
idempotent double-click; skip only when plan marks step skippable; error
card actions derived from server data, never frontend logic.

### Integration tests
Pause mid-run → status Paused + no new step starts; resume → completes;
cancel mid-run → Cancelled with partial results kept; forced failing fake →
retry appears, exhausts at budget, `Requires attention` reached honestly.

### Failure cases
Cancel during LLM call → finishes current call then stops (documented
latency); control during approval wait → cancel wins; double-approve race
→ first wins, second 409; pause signal lost → execution still coherent.

### Security considerations
Controls require execution-scoped permission (stub until GUI-18);
confirm dialogs name exact impact; every control → audit entry (Section 24
starts here).

### Acceptance criteria
All Section 14/16 behaviors proven by deterministic tests; UI never shows
a state the backend has not confirmed (except clearly-marked request
labels); budgets impossible to exceed from UI.

### Definition of done
User can stop or recover any running execution; errors fully actionable.

### Explicit exclusions
No approvals yet (GUI-10), no auto-retry policy editing (Settings/GUI-16),
no mid-step rollback semantics (out of scope of plan).

### Phase GUI-10 — Approval Center + autonomous mode

### Goal
Sections 12–13: real ApprovalRequest backend, queue UI, approve/reject with
impact, two modes with policy classes, unmistakable autonomous banner.

### Why
Approval is the core trust mechanism for a non-technical operator — it must
be backend state, never a frontend confirm (Section 33).

### Dependencies
GUI-9 (blocked-step state `Waiting for approval` + controls), GUI-2 (events
`execution.waiting_for_approval/approved/rejected`).

### Backend changes
`ApprovalRequest` model + policy engine (class table from Section 13:
auto/approval/high-impact, mode-independent high-impact);
step-blocking wait with expiry policy (default → `Requires attention`,
NEVER auto-approve); approve/reject endpoints (idempotent, resolver-stamped);
autonomous mode stored per-execution at creation (frozen — Section 13);
audit entries for every decision + every autonomous auto-approval.

### Frontend changes
Approval Center page (pending by age, resolved history, filters); approval
card with WHAT/WHY/impact + View Details drawer (Section 12); detail-page
waiting banner; global + per-run autonomous toggle with confirm dialog and
the exact Section 13 warning copy; badge counts (bell + rail).

### Data model
`ApprovalRequest` (Section 25); `Execution.mode_at_creation`.

### API/events
`GET /approvals`, `approve|reject` POSTs; events
`execution.waiting_for_approval/approved/rejected` + `execution.expired`
maps to `requires_attention`.

### Files likely affected
NEW `app/approvals/` (model, policy, routes), `app/api/approvals.py`,
`frontend/src/pages/ApprovalCenter.tsx`, detail integration, NEW
`scripts/test_gui10_approvals.py`, frontend tests.

### Deterministic tests
Policy matrix: read→auto, write→approval, delete→approval, high-impact→
approval EVEN in autonomous; approve resumes step (fake runner observes
signal); reject → Failed with "rejected by user" cause; expiry → attention,
never approved; idempotent resolve; double-resolve race; autonomous toggle
frozen for running executions (Section 31 case) asserted; audit rows
generated for auto-approvals.

### Integration tests
Scripted run hits approval gate → UI shows card with real impact data →
approve → execution completes; reject → honest failure; queue dashboard
badge counts correct.

### Failure cases
Approval never resolved → expiry path (Section 31); resolver crashes →
request stays pending (idempotent retry); two users resolve → first wins;
SSE missed resolution → snapshot reconcile on refresh.

### Security considerations
Approve/reject are privileged actions (role stub now, real in GUI-18);
payload cannot be edited from frontend; impact summary computed backend;
all decisions audited immutably.

### Acceptance criteria
Policy matrix fully green; autonomous mode visibly unmistakable; high-
impact ALWAYS asks even autonomous (proven); no execution can proceed past
a pending approval without resolution.

### Definition of done
The Section 12 card flow works with real blocking behavior.

### Explicit exclusions
No batch approvals, no delegation/escalation, no external-ERP writes
themselves (MASTER_PLAN Phase 15 — approvals framework is provider-agnostic
and will gate those when real).

### Phase GUI-11 — Chat

### Goal
Section 5: natural-language entry that creates/queries Executions through
the same API as New Execution — chat as an entry and inspection surface.

### Why
Operators ask "what happened to 1044?" in prose; chat must answer from
real execution data, never from model memory of it.

### Dependencies
GUI-8 (execution creation API), GUI-10 (approval answers reference real
approval state), GUI-6 (chat links to live detail view).

### Backend changes
Chat session persistence; a constrained intent parser mapping utterances →
API operations (`create_execution`, `get_execution`, `list_executions`,
`explain_failure`, `compare_runs`) — deterministic routing first, LLM
fallback through `get_llm()`; every answer grounded in fetched execution
rows/events with cited execution IDs; refusal for out-of-scope asks.

### Frontend changes
Chat page with thread history, inline execution cards, `[Watch Execution]`
CTA, deep links to detail/approval pages; explicit "Chat created Execution
#1044" confirmation block after any creation.

### Data model
New `ChatMessage` (session, role, text, referenced_execution_id,
created_at) — additive; no AgentState change.

### API/events
`POST /chat/messages`, `GET /chat/sessions`; reuses `/executions` verbatim
(no second orchestration path — Section 28).

### Files likely affected
NEW `app/chat/` (session, intent parser, grounding), `app/api/chat.py`,
`frontend/src/pages/Chat.tsx`, NEW `scripts/test_gui11_chat.py`.

### Deterministic tests
Intent table maps correctly; "why did 1042 fail" fetches the REAL error
row (fake LLM cannot invent it); chat-created execution equals
form-created execution (same endpoint, same fields); out-of-scope refusal;
chat never writes execution state directly (static import check).

### Integration tests
Chat: "Find active dentists in Lebanon" → execution card + link → detail
page shows the run; follow-up "how did 1044 go?" answers from events.

### Failure cases
Ambiguous intent → asks a clarifying question instead of guessing;
referenced execution missing → "I couldn't find execution #9999";
backend down → honest offline message (Section 15).

### Security considerations
Chat cannot bypass approvals or permissions; grounded answers only expose
data the current user may read (GUI-18); prompt-injection from retrieved
web/content cannot trigger API operations (capability allow-list).

### Acceptance criteria
Every chat-created execution appears identically in the Executions page;
every factual answer cites a real execution ID; zero hidden orchestration
(static proof).

### Definition of done
Section 5 dialogue flows work end-to-end against real data.

### Explicit exclusions
No free-form agent autonomy, no tool-calling from chat, no memory beyond
session history + execution records.

### Phase GUI-12 — Planner visualization

### Goal
Section 19: render the structured plan from MASTER_PLAN Phases 5–6 as
live step cards (what/why/dependency/expected/approval/status).

### Why
The plan preview is the user's first trust checkpoint before execution;
late-binds the GUI to the planner contract.

### Dependencies
MASTER_PLAN Phase 5 (plan schema + validator) and Phase 6 (dispatcher
emits step status) — gated exactly as Section 19 states; GUI-8 for preview.

### Backend changes
Expose `Plan`/`PlanStep` as an API resource; emit `execution.planned` with
renderable steps; map plan-step status → events; default-plan fallback
flag surfaced so the UI can say "built-in plan (no planner call)".

### Frontend changes
Plan preview block on New Execution (Start/Edit/Cancel); per-step cards
with dependency arrows and approval markers; live plan progress on detail
page (ties into Section 18 activity graph).

### Data model
`Plan`, `PlanStep` per Section 25 — read-only projections of the
MASTER_PLAN plan structure (no parallel schema).

### API/events
`GET /executions/{id}/plan`; events `execution.planned`,
`step.status_changed`.

### Files likely affected
`app/api/plans.py`, plan serializer in execution API, `frontend/src/` plan
components, NEW `scripts/test_gui12_plan_render.py`.

### Deterministic tests
Schema→UI-field mapping complete (every PlanStep field has a renderer or
an explicit hidden rule); invalid plan rejected before preview (validator
errors listed readably); default-plan flagged; status transitions match
dispatcher semantics.

### Integration tests
Planned run shows preview → approval gate → live checkmarks progressing
through the exact plan steps.

### Failure cases
Plan generation fails → default plan path with explanation; plan updated
mid-run (replan) → UI re-renders with a "plan updated" audit entry, never
silently.

### Security considerations
Plan JSON rendered as data (no HTML/JS injection from objective text);
steps never expose executable payloads (Section 9, MASTER_PLAN §5.3).

### Acceptance criteria
Every plan field from Section 19 is visible or explicitly collapsed;
preview gates Start; live progress matches event truth.

### Definition of done
Section 19 flows work against the real planner contract.

### Explicit exclusions
No plan editing beyond reorder/annotate until MASTER_PLAN Phase 6 exposes
an edit contract; no planner of the GUI's own.

### Phase GUI-13 — Knowledge (non-technical RAG view)

### Goal
Section 20: manage document sources and search knowledge without ever
showing Chroma, embeddings, or vector distances by default.

### Why
The operator's mental model is "documents I gave the system", not
"collections and indexes".

### Dependencies
GUI-3 (shell), GUI-2 (health events for index status).

### Backend changes
`KnowledgeSource` CRUD + reindex job with status events; human-readable
index health (ready/indexing/failed + plain-language cause); content-
sensitive search endpoint returning ranked excerpts (scores hidden behind
Advanced).

### Frontend changes
Knowledge page (table per Section 20 + Add/Remove/Re-index/View/Search);
collapsed "Advanced (technical)" disclosure; detail-page panel listing
"Knowledge sources searched / Documents retrieved / Sources used".

### Data model
`KnowledgeSource` per Section 25 (type, added_at, status, last_indexed_at,
error_summary).

### API/events
`GET|POST|DELETE /knowledge`, `POST /knowledge/{id}/reindex`,
`GET /knowledge/search`; events `knowledge.index_started/completed/failed`.

### Files likely affected
NEW `app/api/knowledge.py` wrapping the existing LlamaIndex/Chroma path,
frontend Knowledge page, NEW `scripts/test_gui13_knowledge.py`.

### Deterministic tests
Status machine (ready/indexing/failed) with plain-language error mapping;
search returns excerpts with source names, no score fields in default
projection; remove/reindex idempotency; ingestion errors summarized not
leaked.

### Integration tests
Add PDF → index → status Ready → question retrieves it → detail page shows
the document under "Sources used"; corrupt file → Failed + readable cause.

### Failure cases
Indexing fails → Failed status + Retry reindex; backend restart mid-index
→ status reconciled to Failed/queued, never stuck "indexing" forever
(Section 31).

### Security considerations
Uploads size/type-limited; document access follows user permissions
(GUI-18); no raw filesystem paths exposed by default.

### Acceptance criteria
A non-technical user can add, verify, and search a document with zero
jargon on screen; the execution detail cites knowledge sources by name.

### Definition of done
Section 20 flows work against the real RAG pipeline.

### Explicit exclusions
No embedding/tuning knobs, no collection browsing, no RAG-quality
benchmarks (those belong to MASTER_PLAN RAG-1…RAG-6).

### Phase GUI-14 — Reports / Results

### Goal
Section 21: a Results page listing every artifact (reports, documents,
datasets, answers, exports) linked to the execution that created it.

### Why
Operators care about outputs first; provenance to the producing execution
is what makes an artifact trustworthy and re-creatable.

### Dependencies
GUI-2 (execution persistence), GUI-5 (detail page back-links), the
feasibility `.docx` output path that already exists in the codebase.

### Backend changes
`ExecutionOutput` registry: every run registers its artifacts (path,
kind, title, size, created_at, execution_id); regenerate = clone-and-rerun
(GUI-15); rename/archive metadata; download streaming with permission
checks.

### Frontend changes
Results page (table + filters by type/date/status), artifact detail with
Open/Download/Rename/Archive/Regenerate/View execution, back-link from
detail page result panel.

### Data model
`ExecutionOutput` per Section 25 (execution FK is mandatory).

### API/events
`GET /outputs`, `GET /outputs/{id}` (download), `PATCH /outputs/{id}`
(rename/archive), `POST /outputs/{id}/regenerate` → clone execution.

### Files likely affected
NEW `app/api/outputs.py`, output registration hook in execution runner,
frontend Results page, NEW `scripts/test_gui14_outputs.py`.

### Deterministic tests
Artifact registration happens for every completed execution; no artifact
exists without an execution FK; archive is soft (history intact);
regenerate produces a NEW execution with parent link (immutability,
Section 17); download permission enforced.

### Integration tests
Feasibility run → `.docx` appears in Results → View execution jumps back
to the run that made it → Regenerate creates #N+1 parented to #N.

### Failure cases
Artifact file missing on disk → row shows "File unavailable" + human
cause, not a crash; regenerate run fails → old artifact untouched.

### Security considerations
Download authorization per execution ownership (GUI-18); no arbitrary
filesystem paths from client input; archived artifacts retained per
policy, not silently wiped.

### Acceptance criteria
Every result is reachable from its execution and vice versa; no orphan
artifacts; destructive actions are archive-by-default.

### Definition of done
Section 21 flows work against real artifacts produced by real runs.

### Explicit exclusions
No document editor, no version diffing, no permanent delete (Section 17).

### Phase GUI-15 — Edit / Clone / Rerun / Archive

### Goal
Section 17: manage historical executions immutably — view, edit-and-rerun,
clone, archive — where every re-run is a NEW execution with a parent link.

### Why
Operators iterate on tasks; mutating history would destroy auditability
(Section 24) and make approvals/audit meaningless.

### Dependencies
GUI-4/5 (history + detail), GUI-14 (regenerate reuses this), GUI-9 (retry
semantics for "retry failed step" vs "rerun whole execution").

### Backend changes
Clone endpoint (copies question, options, mode, knowledge selection —
never status/events/results); archive flag (soft-delete, excluded from
default lists, retained for audit); rerun = clone + start; per-step retry
delegates to existing bounded retry state (MASTER_PLAN §5.6 caps stay
authoritative — GUI cannot exceed them).

### Frontend changes
Row/detail actions: View / Edit & Rerun / Clone / Archive; edit form
pre-filled from the parent's request; "Parent execution: #1042" badges;
permanent delete hidden behind advanced confirmation (or omitted).

### Data model
`Execution.parent_execution_id`, `archived_at` (additive columns).

### API/events
`POST /executions/{id}/clone`, `POST /executions/{id}/rerun`,
`POST /executions/{id}/archive`; events `execution.cloned` on the child.

### Files likely affected
Execution API, frontend history/detail action menus, NEW
`scripts/test_gui15_clone_rerun.py`.

### Deterministic tests
Clone copies request-fields only (assert status/events/outputs NOT copied);
parent link set; original row byte-identical after clone/rerun/archive;
retry offers only steps actually retryable per remaining budget; archived
rows hidden from default list but fetchable by id; rerun of failed run
does not resurrect old events.

### Integration tests
Failed run → Edit & Rerun with fixed request → new run completes; audit
shows both executions linked; archive hides it, `?include_archived`
restores it.

### Failure cases
Clone while parent still running → allowed (copies request only) or
blocked by policy — pick blocked, say why; archive of running execution
→ rejected with plain message.

### Security considerations
Clone inherits parent permissions (GUI-18); archive/restore audited;
no hard-delete endpoint at all in v1.

### Acceptance criteria
Immutable-history invariant proven by test; every re-run traceable to its
original; retry UI strictly respects backend budgets.

### Definition of done
Section 17 flows work with `parent_execution_id` proven end-to-end.

### Explicit exclusions
Permanent deletion, branching/diffing, bulk archive.

### Phase GUI-16 — Settings + autonomous mode surface

### Goal
Section 22: grouped Settings (General/Agent/Knowledge/Integrations/
Advanced) with technical knobs hidden from ordinary users.

### Why
Configuration must be changeable without editing `.env`, but advanced
settings exposed raw would violate the product vision (Section 2).

### Dependencies
GUI-10 (approval mode lives here + per-run toggle), GUI-13 (knowledge
settings), existing `app/config.py` as the single source of truth.

### Backend changes
Settings API over validated config (read/write subset per role); settings
change audit events (Section 24); validation errors mapped to plain
language; effective-config view showing defaults vs overrides.

### Frontend changes
Settings page with the five groups; approval-mode card with the exact
Section 13 autonomous copy + confirm dialog; model/provider selector
reading available providers from health (never hardcoded); Advanced group
behind an explicit disclosure.

### Data model
No new tables — settings persist in existing config storage; `Setting`
projection conceptually (key, value, scope, updated_by, updated_at).

### API/events
`GET|PATCH /settings`, `GET /settings/schema` (drives UI grouping);
event `settings.changed` in audit stream.

### Files likely affected
NEW `app/api/settings.py`, frontend Settings page, NEW
`scripts/test_gui16_settings.py`.

### Deterministic tests
Schema-driven render (every schema key has a widget or explicit hidden
rule); validation rejects bad values with human messages; unknown/secret
keys never returned; role gating (advanced = admin);
changing default approval mode does NOT alter already-running executions
(mode is frozen at creation — GUI-10 invariant re-asserted here).

### Integration tests
Toggle autonomous default → new execution inherits it; running execution
keeps its original mode banner.

### Failure cases
Invalid value → inline human error, nothing applied; concurrent edits →
last-write-wins with audit of both.

### Security considerations
Secrets (API keys) write-only or excluded entirely from the API; settings
changes fully audited; CSRF/auth on PATCH.

### Acceptance criteria
A non-technical user changes approval mode and retry limits without seeing
a single environment variable; advanced config stays hidden by default.

### Definition of done
Section 22 groups exist, schema-driven, tested, audited.

### Explicit exclusions
No per-agent prompt editing, no runtime provider installation, no `.env`
file editing from the UI.

### Phase GUI-17 — Notifications

### Goal
Section 23: the operator learns about completed/failed/paused/attention/
approval-needed executions without watching the page.

### Why
Long-running executions make polling by the human unreasonable;
notifications are how the system "asks for the user".

### Dependencies
GUI-9 (control events), GUI-10 (approval events), GUI-6 (event stream is
the source).

### Backend changes
Notification fan-out subscribing to execution/approval events;
in-app notification persistence + read/unread; per-type and per-run
digest preferences; dedupe rules (one notification per state change).

### Frontend changes
Bell with unread badge, notification list with deep links to the relevant
detail/approval screen, per-type toggles in Settings → General; browser
desktop notifications opt-in (with permission prompt handled UX-wise).

### Data model
`Notification` per Section 25 (user, type, execution_id, message,
read_at, created_at).

### API/events
`GET /notifications`, `POST /notifications/{id}/read`,
`POST /notifications/read-all`; produced from existing execution events —
no new execution semantics.

### Files likely affected
NEW `app/notifications/` service, API routes, frontend bell component,
NEW `scripts/test_gui17_notifications.py`.

### Deterministic tests
Event→notification mapping table (which event types notify which user);
dedupe (duplicate events don't duplicate notifications); read/unread
state machine; preferences suppress correctly; approvals notify the
resolution-requested user, not everyone.

### Integration tests
Scripted failed run → unread badge appears → click deep-links to the
failed step → marked read.

### Failure cases
Notification delivery fails → event sourcing allows replay from events;
user has notifications disabled → still sees approval state in UI badges
(never fully silent for pending approvals).

### Security considerations
Notifications never contain secrets or full payload bodies; scoped to
users who can see the execution.

### Acceptance criteria
Every Section 23 type fires exactly once per occurrence, deep-links work,
and pending approvals can't be silently missed.

### Definition of done
Section 23 flows tested end-to-end with persisted, deep-linked notices.

### Explicit exclusions
Email/SMS/push providers (configurable hooks only), notification rules
engine, marketing-style digests.

### Phase GUI-18 — Authentication, permissions, audit

### Goal
Roles (viewer/operator/admin), authenticated API+UI, and the immutable
audit trail from Section 24 queryable from the product.

### Why
Approvals, settings, ERP-facing actions, and audit are meaningless
without identity; this is the prerequisite for any multi-user or
production use.

### Dependencies
GUI-10 (approve/reject privilege stub), GUI-16 (settings role gating),
GUI-15 (audit of clone/archive) — audit events already emitted since
GUI-1/2, this phase authenticates and exposes them.

### Backend changes
Session/token auth (local-first: single-operator login + role table);
role checks at API boundary; `AuditEvent` query API; audit write path
hardened (append-only, no update/delete routes); login/logout events.

### Frontend changes
Login screen; role-aware UI (buttons hidden AND enforced backend);
Audit Trail page (who/what/when/execution links, filters, export);
session expiry UX (save work, re-login, resume view).

### Data model
`User`, `Role`, `AuditEvent` per Section 25 (append-only).

### API/events
Auth endpoints; `GET /audit` (admin); every mutating route gains
`X-Actor` provenance; 403 responses mapped to human messages.

### Files likely affected
Auth middleware, API routes, frontend router guards, Audit page, NEW
`scripts/test_gui18_auth_audit.py`.

### Deterministic tests
Role matrix (viewer cannot create/approve/settings; operator cannot edit
roles; admin can) exhaustively; audit rows for all Section 24 action
kinds; audit has no update/delete endpoint (static check); session
expiry returns 401 → UI redirect; approval decisions record resolver
identity; unauthenticated access denied to every non-public route.

### Integration tests
Two users: operator creates + admin approves high-impact action; audit
shows both identities in order; viewer sees read-only UI and receives 403
on direct API calls.

### Failure cases
Auth backend unavailable → read-only degraded mode with clear banner,
never a silent bypass; expired session mid-execution → execution continues
(server-side), user re-authenticates to the same live view (Section 31).

### Security considerations
Password hashing, no secrets in tokens, rate-limited login, audit
immutability, privilege checks server-side ONLY (hidden buttons are UX,
not security).

### Acceptance criteria
No mutating route without auth+role checks; full Section 24 audit coverage
green; expired-session behavior proven.

### Definition of done
Role matrix + audit tests green; login/logout/refresh flows work.

### Explicit exclusions
SSO/OAuth/MFA, multi-tenant isolation, external user directories.

### Phase GUI-19 — Production hardening

### Goal
Make the whole Control Center reliable under real-world faults: reconnect
logic, load, packaging, backup, and a complete edge-case matrix pass.

### Why
Sections 30–31 define behaviors that only become provable when all pieces
integrate; hardening is the release gate, not a feature.

### Dependencies
All previous GUI phases + MASTER_PLAN phases present at that time.

### Backend changes
Event retention/compaction policy; SSE reconnect with `Last-Event-ID`;
execution-timeout enforcement server-side; health aggregation hardening;
backup/restore of executions+events+knowledge metadata; graceful shutdown
(draining SSE with final snapshot).

### Frontend changes
Offline/degraded banners, global error boundary with human messages
(Section 15), reconnect backoff, stale-data indicators, performance pass
(large histories virtualized), cross-browser check.

### Data model
No new entities; retention fields/policies only.

### API/events
`GET /health` hardened + `GET /events?since=` replay endpoint; rate limits.

### Files likely affected
SSE endpoint, health API, frontend app shell, NEW
`scripts/test_gui19_hardening.py` + E2E suite.

### Deterministic tests
Full Section 31 edge-case matrix as automated tests (browser close,
refresh, disconnect, backend restart, agent crash, model/RAG/ERP down,
approval expiry/reject/delay, simultaneous executions ×2, stuck-execution
timeout, archive, autonomous on/off mid-run); duplicate submission →
idempotent single execution; event replay gap-free after reconnect.

### Integration tests
Kill backend mid-run → restart → browser reconnects → history consistent;
long run under page refresh ×N; two concurrent runs don't cross-contaminate
state/events (isolation assert).

### Failure cases
Each Section 31 case documented with observed behavior + screenshot-level
UI state; any case without a defined behavior is a release blocker.

### Security considerations
Auth/TLS termination guidance, rate limiting, secret scanning in CI,
dependency audit, backup encryption, no stack traces in any default view
(Section 15 enforced by test).

### Acceptance criteria
All Section 31 cases automated or explicitly signed off; zero jargon-only
error screens in the UX audit; performance budget met on 10k-event
executions.

### Definition of done
Hardening checklist complete; E2E suite green; runbook written.

### Explicit exclusions
Horizontal scaling/microservices, HA clusters, mobile-native apps,
internationalization.

---

## 30. Testing strategy

Deterministic first, integration second — mirroring the Phase 4 split the
backend already proved (101+10 green).

**Deterministic (no browser, no network, fake runner):**

```text
execution creation (form == chat path, idempotent duplicate)
execution state machine (all legal + all illegal transitions)
event ordering (monotonic ids/timestamps, no gaps, replay from 0)
live update protocol (SSE framing, Last-Event-ID resume, snapshot+delta)
approve / reject / expiry / idempotent double-resolve
autonomous mode policy matrix (incl. high-impact never auto)
pause / resume / cancel / stop semantics (server-enforced)
retry bounded by backend budgets (GUI cannot exceed MAX_TASK_RETRIES)
failure mapping (every error kind → human message + actions)
edit/clone/rerun (immutability, parent links, archive)
chat intent routing + grounding in real rows
role matrix + audit coverage (Section 24 action kinds)
knowledge/output/archive soft-delete invariants
UX-language lint (no raw exception text reaches default projections)
```

**Integration (browser + backend, fake LLM/tools):**

```text
browser refresh / disconnect / reconnect during run
backend restart recovery (event replay, state reconcile)
approval flow end-to-end (gate → decision → resume)
two simultaneous executions (isolation)
model unavailable / RAG unavailable / ERP unavailable degradation
Chat → Execution → Detail → Chat round-trip
notification deep links
```

**Live-model smoke (manual, quota-conscious, never in CI):** real Ollama or
Gemini run observed through the UI — same rule as the backend's existing
integration suites.

**Test doubles:** the fake plan/step runner from Phase 4 (`fakes`)
extrapolated to emit the canonical events — the GUI must be provable
without any LLM, exactly like `test_phase4_workflow_qa.py`.

---

## 31. Edge-case behavior matrix (all must be defined; all tested in GUI-19)

```text
Browser closes during execution  → execution continues server-side; reopen
                                   shows correct live state (snapshot).
Browser refresh                  → reattach SSE at last event id; no
                                   duplicate/dropped steps.
Network disconnect               → banner "reconnecting"; auto-retry with
                                   backoff; reconcile on success.
Backend restart                  → executions resume or fail per backend
                                   policy; events replay; UI never shows a
                                   step un-started after it completed.
Agent crash                      → step Failed with human cause; retry /
                                   skip / cancel per Section 16.
Model unavailable                → execution pauses at next LLM step with
                                   "AI model unavailable" + Retry later.
RAG unavailable                  → feasibility degrades web-only (existing
                                   MASTER_PLAN behavior); visible note.
ERP unavailable                  → ERPError categories already mapped to
                                   safe messages; execution continues
                                   without ERP data or fails cleanly.
Approval expires                 → Requires attention + notify; NEVER
                                   auto-approved.
Approval rejected                → step Failed with "rejected by user";
                                   no automatic re-ask of the same action.
Approval delayed                 → execution remains Paused; no timeout-
                                   driven progress; user notified on
                                   deadline approach.
Two simultaneous executions      → independent state/event streams,
                                   distinct ids, no cross-writes (tested).
Execution stuck                  → server-side timeout → Failed "took too
                                   long" + retry option (no silent hang).
Execution timeout                → configurable limit surfaced in Settings
                                   (bounded, validated).
Delete old execution             → Archive only (soft); history + audit
                                   retained; restore possible.
Edit old execution               → creates new execution with parent link;
                                   original immutable.
Autonomous enabled               → applies to NEW executions only.
Autonomous disabled mid-run      → running execution keeps creation-time
                                   mode; high-impact steps still ask.
Duplicate submission             → idempotency key → same execution, not a
                                   second run.
```

---

## 32. UX language rules (normative — enforced by a lint test)

1. **Name capabilities, not code.** `Web Research`, `Internal Knowledge`,
   `Feasibility Review` — never a bare `research_agent`, node name, or class
   name as the primary label. Technical ids live behind
   "View technical details".
2. **Every error answers WHAT / WHY / WHAT NOW.** Default view shows no
   stack traces, no exception class names, no HTTP codes. Example:

   ```text
   Web research could not be completed.
   Why: The research service could not be reached.
   You can: [Retry] [Skip] [Edit Task] [Cancel Execution]
   Technical details ▼   (collapsed; ConnectionError...)
   ```

3. **Every approval answers WHAT WILL HAPPEN / WHY / WHAT DATA CHANGES /
   WHAT THE IMPACT IS** (Section 12 card format).
4. **Every running step answers WHAT IS IT DOING / WHY / WHAT IT HAS FOUND
   SO FAR** — human message first, progress second.
5. **Status vocabulary is the fixed ten-word set** from Section 7 — never
   invented synonyms (`error!`, `done`, `working...` are forbidden).
6. **Time is local, absolute, human-formatted** in UI; raw ISO only in
   technical details.
7. **Numbers and ids are stable and clickable** (execution `#1044` is a
   link everywhere it appears).
8. **Never blame the user; never lie.** If state is unknown, say
   "Checking status…" rather than guessing a status.
9. **Jargon stays one disclosure away**, never more than one click deep.
10. **Language is translatable** — all strings in a resource file from day
    one (no hardcoded text in components).

A deterministic lint test scans rendered default projections for forbidden
raw tokens (`Traceback`, `HTTPConnectionPool`, `_agent`, `LangGraph`,
`Chroma`, `Exception:`) and fails on hits.

---

## 33. Do not build a fake monitor (binding rule)

Every visible element must be derived from real backend state or events:

```text
FORBIDDEN                          REQUIRED
--------------------------------   --------------------------------
count-up animations of fake %      real step events
hardcoded agent activity           activity from event stream
pre-baked timelines                timeline = persisted events
"Approve" that only greys out      approval API that gates execution
Cancel that only hides the row     cancel signal enforced by runner
Retry that re-renders locally      bounded backend retry
progress inferred from logs        structured emitted events
```

The frontend is a **projection of the execution event log** — if the event
does not exist in the backend, the UI must not show it. Reviewers should
reject any PR where a UI state variable can advance without a corresponding
event or API response.

---

## 34. Relationship to `MASTER_PLAN.md`

```text
docs/MASTER_PLAN.md
    ↓  agent architecture (Phases 5–16): plan schema, dispatcher,
    ↓  bounded retries, synthesis, reviewer, RAG evolution, ERP gate
Execution & event contract (GUI-1 → GUI-2 of this plan)
    ↓
docs/CONTROL_CENTER_PLAN.md
    ↓  presentation, approvals surface, controls, UX language
GUI (frontend) + thin API boundary
```

**Direct dependencies (gated, not duplicated):**

```text
MASTER_PLAN Phase 5 (Planner)          → GUI-12 plan preview/visualization
MASTER_PLAN Phase 6 (Dispatcher)       → GUI-12 live plan progress,
                                          GUI-9 step-level retry/cancel
MASTER_PLAN §5.6 (Bounded retries)    → GUI-9/GUI-15 retry budgets (UI
                                          mirrors backend caps, never adds)
MASTER_PLAN Phase 7/8 (Synthesis)      → GUI-5 result panel semantics,
                                          GUI-14 artifacts
MASTER_PLAN Phase 9/10 (Reviewer/
  correction loop)                     → GUI-5 review events, GUI-9
                                          correction UX
MASTER_PLAN RAG-1…RAG-6               → GUI-13 hides all of it; only
                                          "sources used" surfaces
MASTER_PLAN Phase 15 (Real ERP)        → GUI-10 approval classes are the
                                          gate for external writes
```

**What this plan deliberately does NOT duplicate:** agent internals, RAG
evolution, planner LLM design, ERP integration, model/provider strategy —
those remain exclusively in `MASTER_PLAN.md`. This document adds only the
**execution/event/approval/API/presentation contracts** the GUI needs, and
references MASTER_PLAN phases by name wherever behavior originates there.

If a future task says "implement GUI-X", the agent reads this file plus the
referenced MASTER_PLAN phase(s) — never one without the other.

