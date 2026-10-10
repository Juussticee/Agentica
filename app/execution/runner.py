"""Execution runner (GUI-1): drive the real graph, record real events.

Principles (Sections 28/33 of the CONTROL_CENTER_PLAN):
  * The UI is a projection of the event log it produces — fake progress is
    forbidden and no progress inference from logs.
  * A AI-unavailable / graph failure MUST be recorded honestly
    (status=failed + structured ExecutionError), never as success.
  * The event store survives server restarts (SQLite).
"""

import json
import uuid
from itertools import count
from typing import Any, Dict, Optional

from app.execution.errors import (
    ExecutionError,
    ai_error_for_provider_unavailable,
    classify_ai_exception,
)
from app.execution.events import (
    AGENT_COMPLETED,
    AGENT_FAILED,
    AGENT_STARTED,
    EXECUTION_COMPLETED,
    EXECUTION_CREATED,
    EXECUTION_FAILED,
    EXECUTION_STARTED,
    SEV_ERROR,
    SEV_INFO,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_STARTED,
)
from app.execution.store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    ExecutionStore,
)

_HUMAN_STEP_NAMES = {
    "supervisor": "Understanding the request",
    "research": "Researching the web",
    "books": "Checking internal knowledge",
    "erp": "Looking up business data",
    "feasibility": "Assessing feasibility",
    "qa": "Checking the result",
}

_AI_PROVIDER_CRASH_MARKERS = (
    "GOOGLE_API_KEY",   # raised by get_llm when gemini key missing
    "API key",          # generic provider auth message
    "LLM_PROVIDER",
    "langchain_google_genai",
    "ChatOllama",
    "google.genai.errors",
    "ConnectionError",
    "api_key",
)


def human_step_name(node: str) -> str:
    return _HUMAN_STEP_NAMES.get(node, node.replace("_", " ").title())


def execution_id_new() -> str:
    return uuid.uuid4().hex[:12]


class GraphRunner:
    """Owns executions: create -> run (graph) -> persist events + result."""

    def __init__(self, store: ExecutionStore):
        self.store = store
        self._seq = count(1)

    # -- helpers ----------------------------------------------------------

    def _event(self, eid, type, message, *, severity=SEV_INFO, step_id=None,
               agent_id=None, payload=None):
        self.store.append_event(
            eid,
            next(self._seq),
            type,
            message,
            step_id=step_id,
            agent_id=agent_id,
            severity=severity,
            payload=payload or {},
        )

    def _node_wrapper(self, eid):
        def wrap(name, fn):
            def wrapped(state):
                self._event(
                    eid, AGENT_STARTED, f"{human_step_name(name)} started",
                    agent_id=name, step_id=name,
                )
                try:
                    update = fn(state)
                except Exception as exc:
                    is_ai = any(m.lower() in str(exc).lower()
                                for m in _AI_PROVIDER_CRASH_MARKERS) or \
                            classify_ai_exception(exc) is not None
                    human = (f"{human_step_name(name)} could not run because the "
                             "AI provider is unavailable."
                             ) if is_ai else f"{human_step_name(name)} failed."
                    self._event(
                        eid, AGENT_FAILED, human,
                        severity=SEV_ERROR, agent_id=name, step_id=name,
                        payload={
                            "category": classify_ai_exception(exc) if is_ai
                            else "execution_error",
                            "details": str(exc)[:2000],
                        },
                    )
                    raise
                self._event(
                    eid, AGENT_COMPLETED, f"{human_step_name(name)} completed",
                    agent_id=name, step_id=name,
                )
                return update

            return wrapped

        return wrap

    # -- API --------------------------------------------------------------

    def create(self, question: str) -> Dict[str, Any]:
        eid = execution_id_new()
        self.store.create_execution(eid, question, STATUS_QUEUED)
        self._event(eid, EXECUTION_CREATED, "Execution created")
        return self.store.get_execution(eid)

    def run(self, eid: str) -> Dict[str, Any]:
        """Synchronously run an execution. Returns the final stored row."""
        stored = self.store.get_execution(eid)
        if stored is None:
            raise KeyError(eid)

        self.store.update_execution(eid, status=STATUS_RUNNING, started_at=True)
        self._event(eid, EXECUTION_STARTED, "Execution started")

        # Import here: keep agent startup cost out of the API import path and
        # let tests patch the graph builder module if needed.
        from app.graph import build_graph

        try:
            graph = build_graph(node_wrapper=self._node_wrapper(eid))
            final_state = graph.invoke({"question": stored["question"]})
        except Exception as exc:
            return self._fail(eid, exc)

        answer = final_state.get("answer", "")
        route = final_state.get("route", "")
        qa_flags = final_state.get("qa_flags", [])
        self.store.update_execution(
            eid,
            status=STATUS_COMPLETED,
            route=route,
            answer=answer,
            qa_flags=qa_flags,
            result=answer,
            completed_at=True,
        )
        self._event(
            eid, EXECUTION_COMPLETED, "Execution completed", payload={
                "route": route,
                "qa_flags": qa_flags,
            }
        )
        return self.store.get_execution(eid)

    def _fail(self, eid: str, exc: Exception) -> Dict[str, Any]:
        err = _execution_error(exc)
        self.store.update_execution(
            eid,
            status=STATUS_FAILED,
            error=err.to_dict(),
            completed_at=True,
        )
        self._event(
            eid, EXECUTION_FAILED, err.human_message, severity=SEV_ERROR,
            payload=err.to_dict(),
        )
        return self.store.get_execution(eid)


def _execution_error(exc: Exception) -> ExecutionError:
    """Map a runner exception to the structured, honest user-facing form."""
    raw = str(exc)
    category = classify_ai_exception(exc)
    is_ai = category in {
        "auth_rejected", "connection_refused", "dns_unreachable", "timeout",
        "wrong_service", "model_missing", "bad_response", "malformed_endpoint",
    }
    if is_ai or any(marker.lower() in raw.lower()
                    for marker in _AI_PROVIDER_CRASH_MARKERS):
        return ai_error_for_provider_unavailable("LLM", category)

    if isinstance(exc, json.JSONDecodeError):
        return ExecutionError(
            category="execution_error",
            human_message="The execution could not be recorded.",
            why="Internal data could not be read or written.",
            recovery="Retry the execution; if it persists, report this as a bug.",
            details={"exception": type(exc).__name__},
        )

    return ExecutionError(
        category="execution_error",
        human_message="The execution failed unexpectedly.",
        why="An internal error occurred while running the workflow.",
        recovery="You can retry the execution or check Settings → AI connection "
                 "if the problem looks provider-related.",
        details={"exception": type(exc).__name__, "message": raw[:2000]},
    )
