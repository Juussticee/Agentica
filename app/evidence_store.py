"""Durable evidence snapshots: what a run actually consulted, kept as used.

The execution record already preserves the answer, provenance and QA findings.
It does not preserve the PASSAGES, so a historical answer could not be audited
afterwards — and re-ingesting a document would silently change what a past
answer appears to have been based on.

This module writes one immutable JSON snapshot per execution:

    storage/evidence_snapshots/<execution_id>.json

WHY A SIDECAR FILE AND NOT A TABLE
The executions/events schema is shared with the Control Center and the runner,
and adding a column or a table would be a migration of live data for a
read-only audit trail. A file keyed by execution id needs no schema change, is
atomic (`tempfile` + `os.replace`), survives restarts, and is trivially
inspectable. Nothing in a run depends on it: a failed write is reported and
ignored.

WHAT IS (AND IS NOT) STORED
  * the passage text AS RETRIEVED during the run, with its retrieval and
    rerank scores, chunk id, file name, page and the document's own
    creation/modification stamp — that stamp is the *ingestion identity*:
    a later re-ingest changes it, so an old snapshot is distinguishable from
    the current index instead of being silently reinterpreted;
  * the query, the coverage verdict, the ranking, the claim-to-passage
    verification and the answer's provenance.
  * NEVER `file_path` or any other absolute filesystem path, and never
    credentials: the snapshot is meant to be readable by the UI and by a
    person auditing a past answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import CHROMA_COLLECTION, INDEX_MANIFEST, STORAGE_DIR

SNAPSHOT_DIR = STORAGE_DIR / "evidence_snapshots"
SNAPSHOT_VERSION = 1

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _path(execution_id: str) -> Optional[Path]:
    if not _SAFE_ID.match(str(execution_id or "")):
        return None
    return SNAPSHOT_DIR / f"{execution_id}.json"


def index_identity() -> Dict[str, Any]:
    """Which index a snapshot came from, without reading the store.

    The ingest manifest's digest changes whenever the corpus is rebuilt, so a
    snapshot and the live index can be compared instead of assumed equal.
    """
    identity: Dict[str, Any] = {"collection": CHROMA_COLLECTION}
    try:
        raw = INDEX_MANIFEST.read_bytes()
        identity["manifest_sha256"] = hashlib.sha256(raw).hexdigest()[:32]
        identity["manifest_entries"] = len(json.loads(raw.decode("utf-8")))
    except Exception:  # noqa: BLE001 — identity is optional provenance
        identity["manifest_sha256"] = None
        identity["manifest_entries"] = None
    return identity


def _completeness(passages: list, retrieved: Optional[int] = None) -> tuple:
    """(storage_complete, content_complete, reasons) for the snapshot.

    Two DIFFERENT questions, kept apart (2026-11-02):

      * storage completeness — is there a row for every passage the retriever
        returned? A missing row would mean evidence was lost from the record.
      * content completeness — is each row's text the full passage, or was it
        cut by `EVIDENCE_SNAPSHOT_CHARS` / empty?

    Collapsing them into one flag would let `complete: true` imply the text is
    whole when it is only present. A passage absent from the PROMPT is neither:
    that is recorded per row as `supplied: false`, not a loss.
    """
    reasons: list = []
    missing_rows = 0
    if isinstance(retrieved, int) and retrieved > len(passages):
        missing_rows = retrieved - len(passages)
        reasons.append(
            f"{missing_rows} retrieved passage(s) have no stored record "
            f"({len(passages)} of {retrieved} rows)")
    for row in passages:
        rank = row.get("rank")
        if row.get("truncated"):
            reasons.append(
                f"passage {rank} ({row.get('ref') or 'unknown source'}) stored "
                f"truncated at {len(row.get('text') or '')} of "
                f"{row.get('chars')} chars")
        if not str(row.get("text") or "").strip():
            reasons.append(f"passage {rank} had no text to store")
    storage_complete = missing_rows == 0 and all(
        str(row.get("text") or "").strip() or row.get("chars") == 0
        for row in passages)
    content_complete = not reasons
    return storage_complete, content_complete, reasons


def build(execution_id: str, *, question: str, route: str, answer: str,
          sources: list, qa_flags: list, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the snapshot payload from the run's final state."""
    passages = list(evidence.get("passages_detail") or [])
    retrieved = evidence.get("passages")
    storage_complete, content_complete, reasons = _completeness(
        passages, retrieved if isinstance(retrieved, int) else None)
    complete = storage_complete and content_complete
    supplied = [row for row in passages if row.get("supplied") is True]
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "execution_id": execution_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "route": route,
        "answer": answer,
        "sources": list(sources or []),
        "qa_flags": list(qa_flags or []),
        "index": index_identity(),
        "evidence": {
            "status": evidence.get("status"),
            "answered_from": evidence.get("answered_from"),
            "coverage": evidence.get("coverage"),
            "best_score": evidence.get("best_score"),
            "missing": evidence.get("missing") or [],
            "conflict": evidence.get("conflict"),
            "reason": evidence.get("reason"),
            "grounding": evidence.get("grounding"),
            "dropped_duplicates": evidence.get("dropped_duplicates"),
            "ranking": evidence.get("ranking") or [],
            "passages": passages,
            # Retrieved vs actually handed to the model: the snapshot must not
            # imply the model saw a passage it never did.
            "passages_retrieved": len(passages),
            "passages_supplied": len(supplied),
            # `complete` is the conjunction and can never be true when text
            # was cut; the two halves stay inspectable on their own.
            "storage_complete": storage_complete,
            "content_complete": content_complete,
            "complete": complete,
            "partial_reason": None if complete else "; ".join(reasons),
            "claims": (evidence.get("claim_checks") or {}).get("claims") or [],
            "claim_summary": (evidence.get("claim_checks") or {}).get("summary")
                             or {},
        },
    }


def write(execution_id: str, payload: Dict[str, Any]) -> Optional[Path]:
    """Atomically persist one snapshot. Returns the path, or None on failure."""
    path = _path(execution_id)
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path
    except Exception:  # noqa: BLE001 — audit trail must never fail a run
        return None


def read(execution_id: str) -> Optional[Dict[str, Any]]:
    """The stored snapshot for an execution, or None when there is none."""
    path = _path(execution_id)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — corrupt file means "no snapshot"
        return None
