# Agent Development Guide

This document defines the contract every future agent must satisfy, how agents
register with the graph, and the rules that keep adding an agent a small,
predictable change. It is the realization of the **Phase 4 standardization
task**: no real ERP, no new dependency, no framework, no heavy abstraction —
only a consistent, documented pattern.

---

## 1. Agent contract

Every agent is a single module in `app/agents/` with one public entry point:

```python
def <agent_name>(state: AgentState) -> dict:
    ...
```

The contract, in full:

1. **Receives `AgentState`** — the single TypedDict defined in `app/state.py`.
   The agent reads only the fields it needs.
2. **Returns only the keys it changes** — a dict containing exactly the keys it
   should add or overwrite (usually `answer`, `sources`, and/or
   `agent_answers`). LangGraph merges node updates, so a node that returns
   `{"answer": ...}` cannot touch `state["sources"]` or
   `state["agent_answers"]`.
3. **Records its contribution in `sources`** — the agent appends its own name
   (`"books"`, `"research"`, `"erp"`, `"feasibility"`, ...) to
   `state["sources"]` via the shared `with_agent_output(state, agent, ...)`
   helper. This is the *only* provenance mechanism. Never build a second one.
4. **Does not call another agent.** No agent node imports or invokes another
   agent node. Data flows between agents exclusively through
   `AgentState` (the feasibility route already relies on this: the Super-
   visor sets `sub_routes = ["research", "books"]`, so both prior answers sit
   in `state["agent_answers"]` when the feasibility agent runs).
5. **Does not mutate unrelated state.** A node may only write keys it returns.
   Never reach into `state` and modify a field it does not own.
6. **Does not instantiate models directly.** If the agent needs an LLM, it
   calls `get_llm()` from `app/llm_client.py` — the shared factory. Switching
   providers is a one-line change in `app/config.py`.
7. **Never raises unexpectedly.** Tool callers, adapters, and the graph treat
   the agent as the last safe boundary. Map failures to fixed, safe messages;
   never leak exception text (credentials, endpoints, internal data) into the
   answer.
8. **Document any new tool/dependency.** If the agent adds a library or tool
   call, update `docs/ARCHITECTURE.md` and `docs/TESTING.md` so the graph and
   the test matrix stay accurate.

---

## 2. AgentState rules

- `app/state.py` is the **only** definition of `AgentState`. Never create a
  second state type, never redefine per node.
- Nodes return **only** `answer`, `sources`, and `agent_answers` via
  `with_agent_output()` (and `sub_routes`/`route` are the Supervisor's keys).
  A new agent must not require a new `AgentState` key — if it needs to carry
  a value across the graph, put it in one of those existing buckets
  (`answer` + `agent_answers`, or `sources`), or extend `with_agent_output()`
  in `app/state.py` with a documented helper.
- `agent_answers` accumulates one entry per agent (`agent_answers["books"]`,
  `agent_answers["research"]`, ...). Downstream agents read prior answers
  from it — this is how the feasibility chain receives two inputs in one
  state.

---

## 3. Registration: how a new agent plugs in

Adding an agent means **five explicit steps** — there is no plugin system,
no registry file, no dynamic discovery:

1. **Create the module**
   `app/agents/example_agent.py` with `def example_agent(state: AgentState) -> dict`.
2. **Import it in `app/graph.py`** and add `graph.add_node("example", example_agent)`.
3. **Add the route in `app/supervisor.py`** — extend `_ROUTES` (the
   keyword→node map) with e.g. `"example": "example"`, and add the keyword list
   before the supervisor function returns the route. Keep routing order
   intentional: specific concepts before generic fallbacks.
4. **Connect it to the graph** — a `conditional_edges` or a plain
   `add_edge` from the Supervisor to the new node, plus an `END` edge once a
   direct route. For chains, reuse the `_after_*` pattern already used for
   feasibility (`_after_research` / `_after_books`).
5. **Add deterministic tests** in `scripts/` (zero-quota). Run the regression
   suite.

The graph stays **explicit and readable** — every node and every route is a
literal line in `app/graph.py`. Do not centralize metadata into a registry
that the graph reads dynamically; the hand-written map is the documentation.

---

## 4. Graph rules

- `app/graph.py` builds the **single** `StateGraph(AgentState)`, exposes
  `build_graph()`, and compiles once.
- **START → supervisor → agent nodes → END** is the skeleton. Feasibility is a
  chain (research → books → feasibility) wired with `_after_*` conditional
  edges; single-agent routes fall through to `END`.
- The **Supervisor decides *where* the question goes** — it never calls an
  agent node directly. It only returns `{"route": ...}` (and `sub_routes` for
  chains). Routing logic lives here, and is the only place that changes when
  the intent classifier changes.

---

## 5. Tool usage

- Tool clients live in `app/tools/` (`search`, `extract`, ...). They are
  **fail-soft**: they never raise. A tool boundary returns `[]` / `""` /
  `None` on failure and lets the agent decide.
- Adapters (current example: `app/erp/`) sit between an agent and an external
  system. The agent is the only caller of an adapter. Failures are mapped to
  the adapter's error taxonomy and never reach the user; logs carry only
  coarse categories, never secrets or query text.
- External capabilities always go through the agent. The agent never opens a
  socket, never reads a file it did not create, never touches a vendor SDK
  directly.

---

## 6. LLM factory usage

- `app/llm_client.py` owns `get_llm()` (configured via `LLM_PROVIDER`) and
  `llm_text()` (normalizes Gemini 3 block output / Ollama string).
- Agents import `get_llm` **from the module**, never at module import time
  (`from app.llm_client import get_llm`), so tests can rebind it. The
  feasibility agent already uses this two-step patch pattern; follow it when
  you bind the factory inside a test.
- Prefer `get_llm(fast=True)` for short classification/rewrite calls; use the
  plain factory for synthesis.

---

## 7. Provenance

- One canonical `state["sources"]` list. An agent appends its name
  (`["books"]`, `["research", "https://..."]`, `["erp"]`).
- `with_agent_output(state, agent, answer, sources)` merges: it records the
  agent's answer in `agent_answers[agent]` and appends fresh provenance to
  the existing `sources` list instead of replacing it.
- **Never** rebuild provenance by hand (`state["sources"] = [...]`) inside a
  node — always go through the shared helper.

---

## 8. Error handling

- An agent **never raises**. Tool failures, adapter failures, and model
  failures are caught inside the agent and converted to one of:

  * a fixed, safe user-facing message;
  * a graceful `[]` / `""` / `None` value;
  * a fail-soft fallback that lets the rest of the pipeline continue.

- Exception details go to the log (if the agent provides logging), never to
  the user answer. The log carries only metadata: timestamp, operation, adapter
  name, status, duration, and a coarse error category.
- The graph treats the agent as the last safe boundary: any unhandled error
  inside an agent should be caught at the node wrapper level so one agent can
  never crash the whole run.

---

## 9. Testing requirements

- **Zero-quota suites** in `scripts/` use fake LLMs and real code paths where
  possible. Only the explicitly-marked live scripts consume Gemini quota.
- Every new agent needs a **deterministic test** covering at least: correct
  routing, normal path, empty/edge input, failure path, and provenance.
- Regression order after a code change:

  ```bat
  python -m py_compile main.py ingest_books.py app\*.py app\agents\*.py app\tools\*.py app\erp\*.py scripts\*.py
  python scripts\test_phase1_routing.py
  python scripts\test_phase2.py
  python scripts\test_feasibility_degradation.py
  python scripts\test_rewrite_step.py
  python scripts\test_phase3_erp.py
  ```

- Run `scripts/check_dependencies.py` and `scripts/check_chroma.py` as
  environment checks; run the phase suites in the order above.

---

## 10. What an agent must NOT do

* Do not call another agent directly — the graph controls orchestration.
* Do not mutate unrelated `AgentState` keys.
* Do not instantiate a model; use `get_llm()` from the shared factory.
* Do not build a second provenance system; use `state["sources"]` + the helper.
* Do not raise; translate failures into safe messages or graceful fallbacks.
* Do not create a BaseAgent, a DI container, a plugin marketplace, or a
  dynamic registry. Consistency comes from the documented contract, not from
  an inheritance hierarchy.

---

## 11. The standardized pattern, end to end

```text
START
  │
  ▼
Supervisor  ── sets state["route"] / state["sub_routes"] only
  │
  ▼
Agent node  ── reads AgentState, performs its domain work, returns ONLY
  │              changed keys + sources + agent_answers
  ▼
Graph merges node result into AgentState
  │
  ▼
END (or next node / chain)
```

- **Agent** = domain work.
- **Supervisor** = routing decisions only.
- **Graph** = execution/order.
- **State** = shared contract.
- **Tools** = external capabilities.
- **LLM factory** = model/provider creation.

Everything else is out of scope. Adding a future agent follows steps 1–5 in
Section 3, takes minutes, and never touches `AgentState`, the Supervisor's
architecture, or the existing agents.
