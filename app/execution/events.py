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
ROUTING_DECIDED = "routing.decided"          # supervisor's route choice
QA_COMPLETED = "qa.completed"                # deterministic QA gate result
EXECUTION_PLANNED = "execution.planned"          # real plan persisted (GUI-12)
EXECUTION_COMPLETED = "execution.completed"
EXECUTION_FAILED = "execution.failed"
EXECUTION_CANCELLED = "execution.cancelled"
EXECUTION_WAITING_APPROVAL = "execution.waiting_for_approval"
EXECUTION_APPROVAL_COMPLETED = "execution.approval_resolved"

# --- W2 structured instrumentation events (2026-10-27) -------------------
# Recorded by the operation itself at the moment it happens (see
# app/execution/instrumentation.py) — the runner's node wrapper cannot see
# inside a node, and stdout print lines are not a durable record.
#
# EVENT_SCHEMA_VERSION identifies the event model for consumers/exporters
# (the proposed docs/execution_record.schema.json is labelled a proposal;
# this constant is the implemented half). Events written by older rows
# simply predate it — nothing is rewritten.
EVENT_SCHEMA_VERSION = 1

RETRIEVAL_COMPLETED = "retrieval.completed"   # books: retrieved chunk ids
TOOL_COMPLETED = "tool.completed"             # search/crawl outcomes
TOOL_FAILED = "tool.failed"                   # tool error (category only)
RECOVERY_OUTCOME = "recovery.outcome"         # bounded M5 retry result
FALLBACK_DECIDED = "fallback.decided"         # books -> research decision

# --- GUI-10 approval lifecycle (2026-10-28) --------------------------------
# Written by the approval gate (creation), the approve/reject endpoints
# (decision), and the resume path (execution / rejection finalization).
# Additive type strings on the existing append-only timeline — no payload or
# schema change for older rows.
APPROVAL_CREATED = "approval.created"        # gate asked for permission
APPROVAL_DECIDED = "approval.decided"        # approve/reject persisted
APPROVAL_EXECUTED = "approval.executed"      # bound operation ran
APPROVAL_FINALIZED = "approval.finalized"    # rejection closed w/ local answer

# --- Continuous learning (2026-10-30) -----------------------------------
# Written by the `feedback` node and the runner: what was actually learned
# and which lessons were actually applied. Additive types on the existing
# append-only timeline; a run with no learning produces none of them.
LESSON_RECORDED = "learning.lesson_recorded"
LESSONS_APPLIED = "learning.lessons_applied"
SKILL_VERSION = "learning.skill_version"

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
