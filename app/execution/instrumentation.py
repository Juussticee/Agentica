"""In-process event sink: operations record themselves AS they happen.

Why this exists (W2, 2026-10-27): retrieval, web search, crawling, bounded
recovery and the books->research fallback decision all happen INSIDE a graph
node, where the runner's node-level wrapper cannot see them — until now the
only record was a ``print()`` line in a backend log. The runner installs a
sink for the duration of one execution (``app/execution/runner.py``); every
``emit()`` appends to that execution's event stream in call order, so the
timeline is observed, never reconstructed afterwards.

Rules (all enforced by tests in scripts/test_execution_instrumentation.py):

* **fail-soft**: ``emit()`` never raises and never touches the answer. An
  event-bookkeeping failure must not corrupt the main execution result — the
  same "agent/tool is the last safe boundary" convention used everywhere
  else in this project.
* **no-op outside a run** (CLI, unit tests, imports): nothing is buffered,
  queued or deferred, so agents behave exactly as they did before.
* **privacy**: payloads carry identifiers, counts, durations and coarse
  error *categories* only — never passage text, never model output, never
  credentials, environment values or raw exception messages (an exception
  string can echo a URL that contains the search query).
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Dict, Optional

from app.execution.events import EVENT_SCHEMA_VERSION

# (type, message, *, agent_id, step_id, severity, payload) -> None
Sink = Callable[..., None]

# Execution-context isolation (W2.1, 2026-10-27).
#
# The sink used to be a module-global, which is unsafe for the actual
# execution model: POST /api/v1/executions runs synchronously on a request
# thread, so two overlapping executions A and B fought over one global —
# A's operations were recorded under B's id (misattribution), A's `finally`
# wiped the sink while B was still running (lost events), and B's `finally`
# restored A's sink, leaving a STALE sink that later recorded unrelated
# emits into a finished execution.
#
# A ContextVar fixes all three because it is per-execution-context:
#   * each request/thread has its own context, so A and B never share;
#   * langgraph's sync BackgroundExecutor submits node work with
#     `copy_context().run(...)`, so operations inside a graph node still see
#     the sink the runner installed around `graph.invoke(...)`;
#   * `reset(token)` restores the previous value even on exceptions and for
#     nested installs, and leaves `None` (no recording) afterwards.
# Every emit() call site in this project runs inside the executing context
# (books/research/crawl/search nodes and the graph's fallback decision); the
# one detached thread in the codebase — the books verifier worker — emits
# nothing, so no cross-thread hand-off is needed (and none is attempted:
# inheriting across threads is exactly how events get attributed to the
# wrong run).
_sink_var: ContextVar[Optional[Sink]] = ContextVar(
    "agentica_execution_sink", default=None
)


@contextmanager
def install(sink: Sink):
    """Route emit() to `sink` for the duration of the block.

    Nested-safe and exception-safe: the token restores whatever was active
    before, in THIS execution context only.
    """
    token = _sink_var.set(sink)
    try:
        yield
    finally:
        _sink_var.reset(token)


def is_recording() -> bool:
    return _sink_var.get() is not None


def emit(type: str, message: str, *, agent_id: Optional[str] = None,
         step_id: Optional[str] = None, severity: str = "info",
         payload: Optional[Dict] = None) -> None:
    """Record one operation fact. Never raises; no-op when no sink is set."""
    sink = _sink_var.get()
    if sink is None:
        return
    body = dict(payload or {})
    body["schema_version"] = EVENT_SCHEMA_VERSION
    try:
        sink(type, message, agent_id=agent_id, step_id=step_id,
             severity=severity, payload=body)
    except Exception:  # noqa: BLE001 — bookkeeping must never break a run
        pass
