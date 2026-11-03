# Agent Registry — Agentica, 2026-10-30

The authoritative machine-readable version of this document is
[`app/agents/registry.py`](../app/agents/registry.py). Nothing at runtime reads
it; `scripts/test_agent_registry.py` proves it cannot drift from the graph, the
planner whitelist and the dispatcher map. **If this page and the code disagree,
the code is right** — and the test is failing.

This page exists because "which specialist did what, with which permissions?"
must be answerable without reading nine modules.

---

## 1. What counts as an agent here

An **agent** is a specialist that performs work no other agent can perform,
with its own tools, permissions and output contract. It is registered only if
all three exist:

1. a real implementation (a module, not a prompt variant);
2. at least one real tool or deterministic computation;
3. an explicit, tested statement of what it **refuses**.

The `general` / `conversational` / `unsupported` entries are registered as
agents because they own a distinct *answer contract* (model knowledge with no
sources, a fixed local reply, an honest capability refusal) — not because they
wrap a different prompt around the same model. They have no tools and no
permissions, which is exactly the point.

## 2. Agents

| Agent | Route | Tools / work it really does | Tools it does NOT have | Approves |
|---|---|---|---|---|
| `conversational` | greeting/ack only | fixed local reply | model, retrieval, tools | — |
| `general` | general | one model completion | retrieval, files, network | — |
| `unsupported` | unsupported | fixed guidance for missing capabilities | any execution | — |
| `books` | books | local RAG retrieval + grounding/refusal gate | web, files, writes | — |
| `research` | research | SearXNG search, page extraction | files, writes | **yes** (external search) |
| `erp` | erp | read-only ERP adapter | writes, network beyond the adapter | — |
| `feasibility` | feasibility, multi | composes research+books into a study + one `.docx` | network of its own | — |
| `coding` | coding | workspace read/search, diff builder, static validation, hash-checked patch | arbitrary shell, files outside the root | **yes** (patch application) |
| `data` | data | bounded dataset reader + deterministic analysis ops | row-level export, writes, network | — |
| `synthesis` | plan terminal step | composes completed task results into the answer | tools, writes | — |

Planner-named capabilities (`app/planner.py: CAPABILITIES`) are exactly:
`research`, `books`, `erp`, `data`, `synthesis`. Anything else is rejected
before a plan is executed.

## 3. Orchestration (decides, never produces)

| Node | Responsibility |
|---|---|
| `supervisor` | classifies the request, sets `route` / `complex`, never answers |
| `planner` | decides whether planning is needed, emits validated plan DATA |
| `dispatcher` | one whitelisted capability→node hop per visit, dependency order, per-task progress, iteration cap |
| `qa` | deterministic verification (empty answer, plan completion, status/result contradictions, prepared-vs-applied patches) |

Warnings never convert a successful execution into a failure; they are
reported, and the run's status reflects what actually happened.

## 4. Skills vs tools (the distinction that keeps permissions honest)

* **Tool** — one bounded operation an agent may call. Enumerated per agent in
  the registry (`tools`), implemented in `app/tools/`, `app/coding/`,
  `app/data/`, `app/erp/`. A tool is a *permission*.
* **Skill** — a reusable procedure or set of instructions: documentation a
  human or a prompt reads. No agent can "run a skill"; skills carry no
  permissions. Current skills: `docs/AGENT_DEVELOPMENT.md`,
  `docs/TESTING.md`, `docs/ARCHITECTURE.md` (workspace boundary section).

## 5. Memory

There is no agent scratchpad and no vector store of model conclusions.

| Store | What persists |
|---|---|
| executions | question, route, answer, plan, sources, error, status (SQLite) |
| approvals | operation, payload, fingerprint, decision, `executed_at` claim |
| events | append-only per-execution timeline |
| plans | on the execution row; task progress written as each task completes |
| knowledge | the RAG index over `knowledge/` documents only |

## 6. Permission audit

Enforced by `scripts/test_agent_registry.py`:

* **network** → `research` only (through its two tools, after approval).
* **filesystem writes** → `feasibility` (one study `.docx`) and `coding`
  (hash-checked patch, after approval).
* **approval required** → `research`, `coding`.
* every agent declares all four dimensions (`network`, `filesystem`, `writes`,
  `approval_required`), so "it only reads" is a checked claim, not a comment.

## 7. What is *not* implemented (honest gaps)

| Role requested | Status |
|---|---|
| Personal Assistant (task context, outstanding work) | **not implemented** — no conversation memory exists; `unsupported` says so |
| Automation agent (workflows, triggers, integrations) | **not implemented** — no integrations are connected |
| Quality/Verification as an agent | partially: deterministic checks in `qa`; no model-based evaluation |
| Document agent beyond RAG | partially: retrieval + citations exist; no extraction/summarisation deliverable pipeline |
| Memory beyond executions/RAG | **not implemented** |

A role is listed here instead of being registered as an agent with a prompt and
no capability, because that would be exactly the "chatbot with many names" this
project rejects.
