"""Deterministic mock ERP adapter — Phase 3 test double, NOT a real ERP.

* fake data only (obviously fictional companies/records; no real customer
  information), no network, no credentials, no vendor SDKs;
* read-only by construction: it implements the ERPAdapter ABC, which has no
  write methods at all;
* deterministic: fixed records, fixed ordering, no timestamps/randomness —
  the same call always returns the same result;
* search semantics: the query is split into whitespace-separated tokens and a
  record matches if ANY token of 3+ characters occurs (case-insensitively) in
  the entity name, record id, title, or field values. Shorter tokens are
  ignored as noise. This keeps natural-language agent questions usable
  without any LLM-based query understanding.

This adapter exists only so the ERP agent, routing and tests can run today.
The real ERP adapter will replace it (selected via ERP_ADAPTER in
app/config.py) once the actual ERP is specified.
"""

from typing import Dict, List, Tuple

from app.erp.adapter import (
    ERPAdapter,
    ERPInvalidRequestError,
    ERPRecord,
    ERPRecordNotFoundError,
)

# Keyed by (entity, record_id); dict order == deterministic result order.
_RECORDS: Dict[Tuple[str, str], ERPRecord] = {
    ("customer", "C-001"): ERPRecord(
        entity="customer",
        record_id="C-001",
        title="ACME Manufacturing",
        fields={"status": "Active", "region": "North", "payment_terms": "Net 30"},
    ),
    ("customer", "C-002"): ERPRecord(
        entity="customer",
        record_id="C-002",
        title="Globex Corporation",
        fields={"status": "Active", "region": "South", "payment_terms": "Net 45"},
    ),
    ("invoice", "INV-1001"): ERPRecord(
        entity="invoice",
        record_id="INV-1001",
        title="Invoice INV-1001 for ACME Manufacturing",
        fields={"customer": "C-001", "status": "Open", "amount_due": "1200.00", "currency": "EUR"},
    ),
    ("invoice", "INV-1002"): ERPRecord(
        entity="invoice",
        record_id="INV-1002",
        title="Invoice INV-1002 for Globex Corporation",
        fields={"customer": "C-002", "status": "Paid", "amount_due": "340.50", "currency": "EUR"},
    ),
    ("product", "P-9001"): ERPRecord(
        entity="product",
        record_id="P-9001",
        title="Standard Widget",
        fields={"in_stock": "42", "unit_price": "19.90", "currency": "EUR"},
    ),
}

_KNOWN_ENTITIES = sorted({entity for entity, _ in _RECORDS})


class MockERPAdapter(ERPAdapter):
    """Fake, deterministic, read-only ERP with a handful of records."""

    def name(self) -> str:
        return "mock"

    def search(self, query: str, limit: int = 5) -> List[ERPRecord]:
        text = (query or "").strip().lower()
        if not text:
            raise ERPInvalidRequestError("search requires a non-empty query")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ERPInvalidRequestError("limit must be a positive integer")
        tokens = [token for token in text.split() if len(token) >= 3]
        matches = [
            record
            for record in _RECORDS.values()
            if any(
                token in record.entity
                or token in record.record_id.lower()
                or token in record.title.lower()
                or any(token in str(value).lower() for value in record.fields.values())
                for token in tokens
            )
        ]
        return matches[:limit]

    def get_record(self, entity: str, record_id: str) -> ERPRecord:
        entity = (entity or "").strip().lower()
        record_id = (record_id or "").strip()
        if not entity or not record_id:
            raise ERPInvalidRequestError("get_record requires entity and record_id")
        record = _RECORDS.get((entity, record_id))
        if record is None:
            raise ERPRecordNotFoundError(
                f"no {entity!r} record {record_id!r} (known entities: {_KNOWN_ENTITIES})"
            )
        return record
