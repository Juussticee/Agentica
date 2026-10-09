"""ERP agent (Phase 3): the ONLY caller of the ERP adapter boundary.

Contract (full details: docs/PHASE3_ERP_PLAN.md):
  * accepts AgentState, returns a dict with ONLY the keys it changes
    (answer, sources, agent_answers) via the shared with_agent_output() helper;
  * talks exclusively to the vendor-neutral ERPAdapter interface in app/erp/
    — never to a specific ERP vendor, never to another agent;
  * deterministic output formatting: NO LLM call is made in Phase 3 (zero
    Gemini quota by design). If the real ERP integration later needs
    natural-language synthesis, it must go through get_llm() in
    app/llm_client.py — never instantiate a model here;
  * every adapter failure is mapped to a fixed, safe user-facing message;
    exception details (which could contain credentials/URLs/internal data)
    never reach the answer — only a coarse error category reaches the log;
  * provenance flows through the existing state["sources"] mechanism, tagged
    "erp" (plus agent_answers["erp"]) — no second provenance system.
"""

import time
from typing import Dict, List

from app.erp import get_erp_adapter
from app.erp.log import log_operation
from app.state import AgentState, with_agent_output

_SEARCH_LIMIT = 5

# Explicit transparency: answers built from adapter data state that the data
# is mock data, so nobody mistakes Phase 3 output for real ERP records.
_MOCK_NOTE = "(Data source: mock ERP adapter — Phase 3 placeholder, not real ERP data.)"

# error category (from ERPError.category) -> fixed, safe user-facing message.
_SAFE_MESSAGES: Dict[str, str] = {
    "invalid_request": "This request could not be processed as a valid ERP query.",
    "adapter_unavailable": "The ERP system is currently unavailable. Please try again later.",
    "permission_denied": "Access to the requested ERP data was denied (insufficient permissions).",
    "record_not_found": "No matching record was found in the ERP.",
    "timeout": "The ERP request timed out. Please try again later.",
}
_GENERIC_ERP_MESSAGE = "An ERP error occurred while processing this request."
_UNEXPECTED_MESSAGE = "An unexpected error occurred while accessing the ERP."
_EMPTY_MESSAGE = "The ERP returned no records matching this request."


def _format_records(records: List, adapter_name: str) -> str:
    lines = [f"ERP lookup results (adapter: {adapter_name}):"]
    for position, record in enumerate(records, start=1):
        fields = "; ".join(f"{key}: {value}" for key, value in record.fields.items())
        line = f"{position}. [{record.entity} {record.record_id}] {record.title}"
        if fields:
            line += f" | {fields}"
        lines.append(line)
    lines.append(_MOCK_NOTE)
    return "\n".join(lines)


def erp_agent(state: AgentState) -> dict:
    """Answer ERP questions from the adapter boundary; never raises."""
    question = (state.get("question") or "").strip()
    adapter = get_erp_adapter()

    started = time.perf_counter()
    try:
        records = adapter.search(question, limit=_SEARCH_LIMIT)
    except Exception as exc:  # noqa: BLE001 — the agent is the safety boundary
        duration_ms = int((time.perf_counter() - started) * 1000)
        category = getattr(exc, "category", None)
        if category is not None:
            log_operation(
                "search", adapter.name(), duration_ms, "failure", error_category=category
            )
            message = _SAFE_MESSAGES.get(category, _GENERIC_ERP_MESSAGE)
        else:
            log_operation(
                "search",
                adapter.name(),
                duration_ms,
                "failure",
                error_category="unexpected_failure",
            )
            message = _UNEXPECTED_MESSAGE
        return with_agent_output(state, "erp", message, ["erp"])

    duration_ms = int((time.perf_counter() - started) * 1000)
    log_operation("search", adapter.name(), duration_ms, "success", records=len(records))

    if not records:
        answer = f"{_EMPTY_MESSAGE}\n{_MOCK_NOTE}"
    else:
        answer = _format_records(records, adapter.name())
    return with_agent_output(state, "erp", answer, ["erp"])
