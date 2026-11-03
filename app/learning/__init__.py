"""Continuous learning (2026-10-30): feedback -> lesson -> retrieval -> skills.

The system has four separated layers, deliberately not one blob of "memory":

  signals.py   deterministic detection: is this message a correction, and
               what kind? (no model call)
  lessons.py   structured lessons + the activation policy. One model call is
               used to UNDERSTAND the feedback, and its output is validated;
               the deterministic analysis is used when the provider is down.
  retrieval.py relevance: which ACTIVE lessons a given task must respect
  skills.py    reusable procedure versions, whose activation requires a
               passed deterministic validation and is reversible

Storage lives in the existing SQLite store (`lessons`, `skill_versions`), so
provenance points at a real execution row and the Control Center reads the
same records. Nothing here can change a tool permission, a security boundary,
an approval rule, credentials or provider configuration: those proposals are
stored for review and never applied.

`writes_disabled()` (2026-10-30, evaluation harness) is an OBSERVATION
switch: inside it, the feedback and failure paths record nothing, so a
benchmark run can measure the learning system without teaching it. It never
changes what is read or applied, only whether new lessons are stored.
"""

import contextlib
import contextvars
from typing import Iterator

_WRITES_DISABLED: contextvars.ContextVar = contextvars.ContextVar(
    "learning_writes_disabled", default=False)


def writes_allowed() -> bool:
    """False while a caller is measuring the system (evaluation runs)."""
    return not _WRITES_DISABLED.get()


@contextlib.contextmanager
def writes_disabled(reason: str = "evaluation run") -> Iterator[None]:
    """Suspend lesson/skill writes for the duration of the block.

    Context-local (contextvars), so two overlapping runs cannot see each
    other's switch. Reads and retrieval are unaffected: a benchmark may still
    apply the active lessons it was given.
    """
    token = _WRITES_DISABLED.set(True)
    try:
        yield
    finally:
        _WRITES_DISABLED.reset(token)
