# Phase 3 — ERP-Agnostic Integration Boundary

Status: **boundary COMPLETE; real ERP integration intentionally NOT started.**
The actual ERP has not been specified — no vendor is assumed or named anywhere
in code or configuration. This document is the contract the future ERP
integration will plug into without redesigning Books, Research, or Feasibility.

## Objective

Prepare the architecture so a future ERP can be connected by *adding* an
adapter, not by *changing* existing agents or graph wiring:

```text
Supervisor
   ↓  (keyword routing, sets route="erp")
ERP Agent                app/agents/erp_agent.py
   ↓  (the ONLY caller of the boundary)
ERP Adapter Interface    app/erp/adapter.py  (vendor-neutral ABC)
   ↓
Mock ERP Adapter         app/erp/mock_adapter.py  (deterministic fake data)
   → later replaced by:  Real ERP Adapter (new module, same interface)
```

## Architecture

* The supervisor routes ERP-shaped questions to the `erp` node (see
  `app/supervisor.py`, `_ERP_RE`). The route order is feasibility → ERP →
  books → research fallback. ERP keywords are matched before books because
  ERP terms are specific, while book questions usually self-identify.
* `app/graph.py` needs no new wiring: `_ROUTES["erp"] = "erp"` and
  `add_edge("erp", END)` already existed since Phase 1 (the stub).
* The ERP agent is the only module that imports the adapter interface.
  No agent calls another agent; data flows only through `AgentState`.
* Adapter selection is a single config switch: `ERP_ADAPTER` in
  `app/config.py` (default `"mock"`), consumed by `get_erp_adapter()` in
  `app/erp/__init__.py` — the ONLY way any code obtains an adapter.

## ERP adapter contract

Defined in `app/erp/adapter.py` — **read-only by construction**. The ABC
declares exactly three members and nothing else:

| Member | Signature | Notes |
|---|---|---|
| `name` | `() -> str` | short adapter id for logging |
| `search` | `(query: str, limit: int = 5) -> List[ERPRecord]` | free-text lookup; `[]` when nothing matches, never `None` |
| `get_record` | `(entity: str, record_id: str) -> ERPRecord` | one record by (entity, id) |

Deliberately deferred: a structured `query(...)` operation — only
`search`/`get_record` are justified by the current architecture (a
natural-language question flowing into `erp_agent`). The interface will grow
only when the real ERP defines a query model.

`ERPRecord` is a frozen dataclass with plain string fields only
(`entity`, `record_id`, `title`, `fields: Dict[str, str]`) — no vendor types
may leak across the boundary.

Error taxonomy (all subclasses of `ERPError`, each with a `category` string):
`ERPInvalidRequestError` (`invalid_request`),
`ERPAdapterUnavailableError` (`adapter_unavailable`),
`ERPPermissionDeniedError` (`permission_denied`),
`ERPRecordNotFoundError` (`record_not_found`),
`ERPTimeoutError` (`timeout`). Adapters MUST translate any vendor/transport
exception into one of these; exception messages are for logs only.

## ERP agent contract

`app/agents/erp_agent.py`:

* accepts `AgentState`, returns a dict containing **only** the keys it
  changes (`answer`, `sources`, `agent_answers`) — built by the shared
  `with_agent_output()` helper, exactly like every other agent;
* talks exclusively to the `ERPAdapter` interface obtained via
  `get_erp_adapter()` — never to a vendor SDK, never to another agent;
* makes **no LLM call** in Phase 3 (deterministic formatting, zero Gemini
  quota). If the real integration later needs synthesis, it must go through
  `get_llm()` in `app/llm_client.py` — never instantiate a model directly;
* never raises: every adapter exception is mapped to a fixed, safe
  user-facing message (see Error handling);
* answers built from adapter data carry an explicit note that the data comes
  from the mock adapter, so Phase 3 output cannot be mistaken for real ERP
  data. Error answers carry no such note.

## Mock adapter

`app/erp/mock_adapter.py`: five fictional records (2 customers, 2 invoices,
1 product — "ACME Manufacturing", "Globex Corporation", etc.), no network, no
credentials, no real customer information, deterministic ordering and results.
Search semantics: the query is split into tokens; a record matches if any
token of 3+ characters occurs in entity/id/title/field values. Invalid input
(empty query, bad limit, missing ids) raises `ERPInvalidRequestError`; an
unknown (entity, id) raises `ERPRecordNotFoundError`.

## Read-only boundary

Not merely documented — enforced by the interface itself:

* `ERPAdapter` declares no `create`/`update`/`delete`/`upsert`/`save`/...
  methods. A write operation would require changing the interface first,
  which is an explicit, reviewable code change.
* `ERPRecord` is a frozen dataclass — records cannot be mutated in place.
* `ERPAdapter` is abstract and cannot be instantiated without implementing
  the read methods.

**Future writes require an explicit approval boundary:** a write-capable
interface extension (e.g. a separate `WriteCapableERPAdapter` or new ABC
methods) may only be added in a later phase, with the user's explicit
approval, plus permission checks in the real adapter. Phase 3 ships no write
capability anywhere.

## Error handling

The agent maps error categories to fixed user-facing messages
(`_SAFE_MESSAGES` in `app/agents/erp_agent.py`):

| Category | User-facing message |
|---|---|
| `invalid_request` | "This request could not be processed as a valid ERP query." |
| `adapter_unavailable` | "The ERP system is currently unavailable. Please try again later." |
| `permission_denied` | "Access to the requested ERP data was denied (insufficient permissions)." |
| `record_not_found` | "No matching record was found in the ERP." |
| `timeout` | "The ERP request timed out. Please try again later." |
| any other `ERPError` | "An ERP error occurred while processing this request." |
| non-ERP unexpected failure | "An unexpected error occurred while accessing the ERP." |

Exception details (which could contain credentials, tokens, endpoints, or
internal data) never reach the user-facing answer; only the coarse category
reaches the log. Tests assert that injected secrets in exception text do not
leak into answers.

## Provenance

The existing `state["sources"]` mechanism is reused — no second provenance
system. ERP contributions are tagged `"erp"` in `sources` and recorded as
`agent_answers["erp"]` via `with_agent_output(state, "erp", answer, ["erp"])`.

## Logging

`app/erp/log.py` appends one minimal line per adapter operation to
`logs/erp_actions.log` (git-ignored; path defined once as `ERP_LOG_PATH` in
`app/config.py`):

```text
<ISO timestamp> | op=search | adapter=mock | status=success | duration_ms=3 | records=2
<ISO timestamp> | op=search | adapter=mock | status=failure | duration_ms=1 | error_category=permission_denied
```

Never logged: passwords, API keys, access tokens, authorization headers,
query text, customer data, or record contents/ids. Every write is fail-soft
(observability never breaks an ERP answer).

## State impact

**No changes to `AgentState`.** The existing keys are sufficient: the ERP
agent reads `question` (and prior `sources`) and writes `answer`,
`sources`, `agent_answers` through the existing helper. `route="erp"` was
already a documented value in `AgentState` since Phase 1. `qa_flags` remains
reserved Phase 4 space. Explicitly documented as required.

## Graph impact

**No changes.** `app/graph.py` is untouched: the supervisor → `erp` → END
route existed since Phase 1 and is verified by tests. The only supervisor
change is the `_ERP_RE` keyword list (a data change, not a structural one);
Books/Research/Feasibility routing is unchanged and regression-tested.

## Tests

`scripts/test_phase3_erp.py` — deterministic, zero Gemini quota, no network:
35 tests covering (1) ERP routing (keywords + real graph invoke), (2) mock
adapter semantics, (3) successful read, (4) record-not-found, (5) adapter
failure (unavailable/timeout/unexpected/base-ERP), (6) permission denied
with leak assertions, (7) provenance via existing sources mechanism,
(8) write protection (interface surface, frozen records, abstractness), plus
factory selection and logging-field tests.

## Future integration process

When the actual ERP is specified, plugging it in requires:

1. Implement a new adapter module (e.g. `app/erp/<real>_adapter.py`)
   subclassing `ERPAdapter` — translate vendor/transport exceptions into the
   `ERPError` taxonomy; no vendor types across the boundary.
2. Add its selection branch in `get_erp_adapter()` (and, if genuinely
   needed, minimal ERP settings in `app/config.py` — never credentials in
   code; keys go in the git-ignored `.env`).
3. Set `ERP_ADAPTER=<real>` in `.env`.
4. Extend tests with an adapter-level suite (fakes for the vendor client;
   optional live smoke test explicitly marked quota/network-consuming).
5. Replace or gate the mock note in the agent answer once real data flows.

No changes to the graph, supervisor (beyond possible keyword tuning), other
agents, or `AgentState` should be necessary.

## ERP-Specific Details Deferred

These are **intentionally unknown** and will be decided only after the actual
ERP details are provided. Nothing in Phase 3 assumes or encodes any of them:

* **ERP vendor** — not specified; no vendor is named in code, config, or docs.
* **API availability** — unknown whether the ERP exposes REST/GraphQL/OData
  or only UI/database-level access.
* **MCP availability** — whether an MCP server exists for the ERP is unknown;
  MCP is intentionally not introduced in Phase 3.
* **Authentication** — unknown (OAuth2 / API key / basic / SSO / cert); no
  credentials were created and none are referenced.
* **Endpoints** — unknown; the mock hardcodes nothing resembling a URL.
* **Data model** — unknown; `ERPRecord`'s string fields are the only
  neutral representation, deliberately vendor-shaped.
* **Entities** — unknown; the mock's five fictional records (customer,
  invoice, product) exist purely to exercise code paths, NOT as a data-model
  commitment.
* **Permissions** — unknown; the `permission_denied` category exists so the
  boundary already speaks the concept, not any real permission model.
* **SDK/dependencies** — none installed; no ERP-specific dependency was added.
* **Deployment model** — unknown (cloud SaaS / on-prem / hybrid).
* **Sandbox** — none requested or configured.
* **Integration protocol** — unknown (direct API / MCP / middleware / file
  exchange); deferred entirely.

These decisions gate Phase 3.5 ("Real ERP integration"), which has NOT
started.
