"""Execution event model (GUI-1, CONTROL_CENTER_PLAN Section 10).

Append-only structured events emitted by the execution runner — the single
source of truth for what the Control Center shows. The frontend must be a
projection of THIS log; it may never invent progress.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from typing import Any, Dict, Optional

# Execution lifecycle (matching the ten human-readable statuses in the plan):
#   queued -> planning -> running -> completed
#                                 -> failed
#                                 -> cancelled
# Worker-side failures raise events with severity "error" but never fake success.

EXECUTION_CREATED = "execution.created"
EXECUTION_STARTED = "execution.started"
STEP_STARTED = "step.started"
STEP_COMPLETED = "step.completed"
STEP_FAILED = "step.failed"
AGENT_STARTED = "agent.started"
AGENT_COMPLETED = "agent.completed"
AGENT_FAILED = "agent.failed"
EXECUTION_COMPLETED = "execution.completed"
EXECUTION_FAILED = "execution.failed"
EXECUTION_CANCELLED = "execution.cancelled"

SEV_INFO = "info"
SEV_WARNING = "warning"
SEV_ERROR = "error"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionEvent:
    """One observable fact about an execution, in order."""

    event_id: int                      # monotonic per execution (1-based seq)
    execution_id: str
    timestamp: str                     # ISO-8601 UTC
    type: str                          # one of the constants above
    message: str                       # human-readable headline (UX language rule)
    step_id: Optional[str] = None      # plan/graph step when applicable
    agent_id: Optional[str] = None     # agent/node name when applicable
    severity: str = SEV_INFO
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "message": self.message,
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "severity": self.severity,
            "payload": dict(self.payload),
        }


_id_seq = count(1)


def nextexecution_event_id() -> int:
    """Process-local tiebreaker; the store assigns per-execution seq."""
    return next(_id_seq)
