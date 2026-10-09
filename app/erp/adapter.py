"""Vendor-neutral ERP adapter interface — the Phase 3 integration boundary.

READ-ONLY BY CONSTRUCTION: the ABC defines ``name``/``search``/``get_record``
and NOTHING else. There are no create/update/delete methods to misuse, so a
write operation is impossible without changing this interface — which is an
explicit, reviewable decision (see docs/PHASE3_ERP_PLAN.md, "Future
integration process"). Phase 3 deliberately ships no write capability.

Adapter implementations MUST:
  * translate any vendor/transport exception into an ``ERPError`` subclass
    below — the ERP agent maps only these categories to user-facing messages;
  * keep exception messages free of credentials, tokens, endpoints and other
    internal details (exception text is for logs, never for users);
  * return plain ``ERPRecord`` values — no vendor types may leak through.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ERPRecord:
    """A minimal, vendor-neutral ERP record.

    Values are plain strings on purpose: the boundary must not leak vendor
    types. Frozen so consumers cannot mutate records in place.
    """

    entity: str      # conceptual entity, e.g. "customer", "invoice", "product"
    record_id: str
    title: str       # human-readable one-liner
    fields: Dict[str, str] = field(default_factory=dict)


class ERPError(Exception):
    """Base class for all ERP adapter failures.

    ``category`` is a coarse, safe label; the ERP agent maps it to a fixed
    user-facing message and writes it to the action log. Never put secrets,
    endpoints or connection details in these messages.
    """

    category = "erp_error"


class ERPInvalidRequestError(ERPError):
    """The request is malformed (empty query, bad limit, missing id, ...)."""

    category = "invalid_request"


class ERPAdapterUnavailableError(ERPError):
    """The ERP backend cannot be reached."""

    category = "adapter_unavailable"


class ERPPermissionDeniedError(ERPError):
    """Authentication/authorization failed or the account lacks permission."""

    category = "permission_denied"


class ERPRecordNotFoundError(ERPError):
    """No record exists for the requested (entity, record_id)."""

    category = "record_not_found"


class ERPTimeoutError(ERPError):
    """The operation exceeded its time budget."""

    category = "timeout"


class ERPAdapter(ABC):
    """The ONLY surface the ERP agent may program against. Read-only.

    Conceptual operations kept deliberately small: only what the current
    architecture (a natural-language question flowing into erp_agent) can
    justify. A structured ``query(...)`` operation is intentionally deferred
    until the real ERP defines its query model.
    """

    @abstractmethod
    def name(self) -> str:
        """Short adapter identifier for logging (e.g. ``"mock"``)."""

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[ERPRecord]:
        """Free-text lookup. Returns ``[]`` when nothing matches, never None."""

    @abstractmethod
    def get_record(self, entity: str, record_id: str) -> ERPRecord:
        """Fetch one record by (entity, record_id); ERPRecordNotFoundError if absent."""
