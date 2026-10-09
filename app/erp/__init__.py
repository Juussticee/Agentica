"""Vendor-neutral ERP integration boundary (Phase 3).

Public surface:
    ERPAdapter, ERPRecord, ERPError (+ subclasses) — the read-only contract
    MockERPAdapter                                 — deterministic fake data
    get_erp_adapter()                              — the ONLY way to obtain an adapter

The real ERP adapter will arrive later as a new module implementing
``ERPAdapter`` and will be selected via ``ERP_ADAPTER`` in app/config.py —
no changes to agents or the graph. Until the actual ERP is specified,
everything here is intentionally vendor-agnostic (see
docs/PHASE3_ERP_PLAN.md, "ERP-Specific Details Deferred").
"""

from app.erp.adapter import (
    ERPAdapter,
    ERPAdapterUnavailableError,
    ERPError,
    ERPInvalidRequestError,
    ERPPermissionDeniedError,
    ERPRecord,
    ERPRecordNotFoundError,
    ERPTimeoutError,
)
from app.erp.mock_adapter import MockERPAdapter

__all__ = [
    "ERPAdapter",
    "ERPAdapterUnavailableError",
    "ERPError",
    "ERPInvalidRequestError",
    "ERPPermissionDeniedError",
    "ERPRecord",
    "ERPRecordNotFoundError",
    "ERPTimeoutError",
    "MockERPAdapter",
    "get_erp_adapter",
]


def get_erp_adapter() -> ERPAdapter:
    """Return the configured ERP adapter. Phase 3 ships exactly one: mock."""
    # Function-level import so the switch stays patchable/testable and so
    # importing app.erp never requires config evaluation order assumptions.
    from app.config import ERP_ADAPTER

    if ERP_ADAPTER == "mock":
        return MockERPAdapter()
    raise ValueError(
        f"Unknown ERP_ADAPTER={ERP_ADAPTER!r}. Phase 3 ships only the deterministic "
        "'mock' adapter; real adapters arrive in a later phase, after the actual "
        "ERP is specified."
    )
