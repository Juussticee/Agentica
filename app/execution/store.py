"""Execution store (GUI-2): SQLite persistence for executions and events.

Local-first: one SQLite file (storage/control_center.sqlite3), binary data
never committed (storage/ is git-ignored). Both the snapshot (current state
per execution) and the append-only event history live here so that:
  * the UI can render "current state" cheaply from the executions table
  * the UI can render the full timeline and replay after `after=<event_id>`.
"""

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.execution.errors import ExecutionError
from app.execution.events import (
    APPROVAL_DECIDED,
    EXECUTION_FAILED,
    SEV_ERROR,
    SEV_INFO,
    utc_now_iso,
)

DEFAULT_DB_PATH = Path("storage") / "control_center.sqlite3"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_WAITING_APPROVAL = "waiting_for_approval"

# --- approvals (GUI-10, CONTROL_CENTER_PLAN §12, 2026-10-28) ---------------
# The smallest ApprovalRequest the plan's lifecycle needs: ONE operation type
# (outbound research), a server-side description, the BOUND operation as a
# small JSON payload (question + route + local diagnosis — never a URL,
# command or executable anything), and a constrained state machine:
#     pending -> approved | rejected | expired
# A decision on a terminal request never changes the previous outcome.
APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_REJECTED = "rejected"
APPROVAL_EXPIRED = "expired"
APPROVAL_STATUSES = (APPROVAL_PENDING, APPROVAL_APPROVED,
                     APPROVAL_REJECTED, APPROVAL_EXPIRED)
APPROVAL_TTL_SECONDS = 24 * 60 * 60  # a pending ask expires after a day

# Human label per approvable operation, so one decision event reads correctly
# for every operation type (2026-10-30: workspace patches joined research).
_APPROVAL_LABELS = {
    "research": "Web research",
    "workspace_patch": "Workspace patch",
}

# GUI-10 recovery (2026-10-29): how long ONE resolver owns an execution's
# continuation. Bounds a crashed holder so a later retry can take over; it is
# a LEASE, never an authorization to repeat an external operation (that stays
# the persisted `approvals.executed_at` claim).
RESUME_LEASE_SECONDS = 5 * 60


class ApprovalNotFound(LookupError):
    """No approval with that id exists."""


class ApprovalConflict(Exception):
    """A decision was attempted on an already-decided approval."""


class ApprovalExpired(Exception):
    """The approval's decision window has passed."""


class ApprovalDecisionRequired(Exception):
    """Execution was requested while the approval is still pending."""


def _is_past(iso: Optional[str]) -> bool:
    """True when an ISO timestamp is in the past (or unreadable — an
    approval with an unreadable expiry counts as expired: fail closed)."""
    try:
        return datetime.fromisoformat(str(iso)) <= datetime.now(timezone.utc)
    except Exception:  # noqa: BLE001 — malformed expiry fails closed
        return True


def _decode_approval(d: Dict[str, Any]) -> Dict[str, Any]:
    raw = d.get("payload")
    try:
        payload = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
    except Exception:  # noqa: BLE001 — a corrupt payload degrades to None
        payload = None
    d["payload"] = payload if isinstance(payload, dict) else None
    return d


def serialize_approval(row: Dict[str, Any], *,
                        reused: Optional[bool] = None,
                        already_decided: Optional[bool] = None) -> Dict[str, Any]:
    """API/store projection of one approval request (never internal ids or
    connection state). `reused` / `already_decided` describe THIS call's
    outcome, not persisted state."""
    out = {
        "approval_id": row.get("approval_id"),
        "execution_id": row.get("execution_id"),
        "operation": row.get("operation"),
        "description": row.get("description"),
        "status": row.get("status"),
        "decision": row.get("decision"),
        "created_at": row.get("created_at"),
        "expires_at": row.get("expires_at"),
        "decided_at": row.get("decided_at"),
        "executed_at": row.get("executed_at"),
        "payload": row.get("payload"),
    }
    if reused is not None:
        out["reused"] = reused
    if already_decided is not None:
        out["already_decided"] = already_decided
    return out


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI/uvicorn and TestClient run handlers on
    # different threads than the one that created the store; all access is
    # serialized through self._lock, which is the actual safety mechanism.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _decode_json_list(raw: Any) -> Optional[List[Any]]:
    """Decode a persisted JSON list column (sources, qa_flags).

    NULL stays NULL so legacy rows (written before the column existed, or
    before a run completed) keep loading unchanged — never coerced to a
    fabricated value.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, list) else None


def _decode_lesson(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a lesson row's JSON columns (evidence list, calibration).

    A corrupt evidence blob decodes to [] rather than failing the row: the
    lesson's own text is the durable part, and the Control Center must still
    be able to show and disable it. The same applies to the calibration blob:
    an unreadable one means "unknown classification", never a crashed page.
    """
    evidence = row.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        try:
            decoded = json.loads(evidence)
            row["evidence"] = decoded if isinstance(decoded, list) else []
        except Exception:
            row["evidence"] = []
    elif evidence is None:
        row["evidence"] = []
    calibration = row.get("calibration")
    if isinstance(calibration, str) and calibration.strip():
        try:
            decoded = json.loads(calibration)
            row["calibration"] = decoded if isinstance(decoded, dict) else {}
        except Exception:
            row["calibration"] = {}
    elif calibration is None:
        row["calibration"] = {}
    return row


def _decode_skill_version(row: Dict[str, Any]) -> Dict[str, Any]:
    for col in ("source_lesson_ids", "evaluation"):
        raw = row.get(col)
        if isinstance(raw, str) and raw.strip():
            try:
                row[col] = json.loads(raw)
            except Exception:
                row[col] = [] if col == "source_lesson_ids" else None
        elif raw is None:
            row[col] = [] if col == "source_lesson_ids" else None
    return row


def _decode_plan(raw: Any) -> Optional[Dict[str, Any]]:
    """Decode a persisted plan JSON blob; None when absent or corrupt.

    A corrupt plan never breaks the detail page: it degrades to None exactly
    like a legacy execution without a plan.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        plan = json.loads(raw)
    except Exception:
        return None
    return plan if isinstance(plan, dict) else None


_RESULT_SNIPPET_MAX = 400  # keep result summaries bounded for the UI

capabilities_human = {
    "research": "Web research",
    "books": "Internal knowledge (library)",
    "erp": "Business data (ERP)",
    # 2026-10-30: specialists added since this map was written.
    "data": "Dataset analysis",
    "feasibility": "Feasibility study",
    "synthesis": "Combine results into the answer",
}


def serialize_plan(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Projection of the real executed plan for the API/UI (GUI-12,
    CONTROL_CENTER_PLAN Section 19).

    Renders plan DATA as recorded by the planner + Phase 8 dispatcher:
    identity, capability, objective, dependencies, status, and result/error
    summaries. Internal bookkeeping (dispatcher visit counters) is pruned,
    nothing is invented, and a falsy/malformed plan becomes None so legacy
    executions degrade gracefully.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        return None
    tasks = []
    for task in plan["tasks"]:
        if not isinstance(task, dict):
            continue
        result = task.get("result")
        truncated = False
        if isinstance(result, str) and len(result) > _RESULT_SNIPPET_MAX:
            result = result[:_RESULT_SNIPPET_MAX]
            truncated = True
        tasks.append({
            "task_id": task.get("task_id"),
            "capability": task.get("capability"),
            "capability_label": capabilities_human.get(task.get("capability"),
                                                       task.get("capability")),
            "objective": task.get("objective"),
            "depends_on": list(task.get("depends_on") or []),
            "required_inputs": list(task.get("required_inputs") or []),
            "expected_output": task.get("expected_output"),
            "done_when": task.get("done_when"),
            "status": task.get("status"),
            "result": result,
            "result_truncated": truncated,
            "error": task.get("error"),
            "block_reason": task.get("block_reason"),
        })
    source = plan.get("source")
    return {
        "goal": plan.get("goal"),
        "source": source,
        "fallback_reasons": list(plan.get("fallback_reasons") or [])
        if source in ("fallback",) else [],
        "tasks": tasks,
    }


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the GUI-2 schema, without touching data.

    * GUI-12: `plan`
    * W2 (2026-10-27): `sources` — the provenance list the graph returns in
      `state["sources"]`, which was previously dropped before storage and
      therefore never exportable.
    * GUI-10 recovery (2026-10-29): `resume_lease_owner` /
      `resume_lease_expires_at` — the DURABLE one-resolver claim (see
      `ExecutionStore.resume_claim`).

    Existing databases keep working unchanged: legacy rows simply have
    plan/sources = NULL and continue to load (backwards-compatible, additive
    only — no column is ever dropped, renamed or rewritten).
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(executions)")}
    if "plan" not in cols:
        conn.execute("ALTER TABLE executions ADD COLUMN plan TEXT")
    if "sources" not in cols:
        conn.execute("ALTER TABLE executions ADD COLUMN sources TEXT")
    if "resume_lease_owner" not in cols:
        conn.execute("ALTER TABLE executions ADD COLUMN resume_lease_owner TEXT")
    if "resume_lease_expires_at" not in cols:
        conn.execute(
            "ALTER TABLE executions ADD COLUMN resume_lease_expires_at TEXT")

    # 2026-10-30 (learning calibration): how the correction was classified and
    # how wide it may apply. Additive with defaults, so an existing learning
    # database keeps every lesson and simply reports the classification as
    # unknown (`NULL` / `{}`) for rows stored before it existed.
    lesson_cols = {r["name"] for r in conn.execute("PRAGMA table_info(lessons)")}
    if lesson_cols:
        if "feedback_type" not in lesson_cols:
            conn.execute("ALTER TABLE lessons ADD COLUMN feedback_type TEXT")
        if "impact" not in lesson_cols:
            conn.execute("ALTER TABLE lessons ADD COLUMN impact TEXT")
        if "temporary" not in lesson_cols:
            conn.execute("ALTER TABLE lessons ADD COLUMN temporary INTEGER "
                         "NOT NULL DEFAULT 0")
        if "calibration" not in lesson_cols:
            conn.execute("ALTER TABLE lessons ADD COLUMN calibration TEXT")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    id            TEXT PRIMARY KEY,
    question      TEXT NOT NULL,
    status        TEXT NOT NULL,
    route         TEXT,
    answer        TEXT,
    qa_flags      TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    completed_at  TEXT,
    result        TEXT,
    plan          TEXT,
    sources       TEXT,
    resume_lease_owner      TEXT,
    resume_lease_expires_at TEXT
);

CREATE TABLE IF NOT EXISTS events (
    execution_id  TEXT NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    event_id      INTEGER NOT NULL,
    timestamp     TEXT NOT NULL,
    type          TEXT NOT NULL,
    message       TEXT NOT NULL,
    step_id       TEXT,
    agent_id      TEXT,
    severity      TEXT NOT NULL DEFAULT 'info',
    payload       TEXT,
    PRIMARY KEY (execution_id, event_id)
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id   TEXT PRIMARY KEY,
    execution_id  TEXT REFERENCES executions(id) ON DELETE CASCADE,
    operation     TEXT NOT NULL,
    description   TEXT NOT NULL,
    payload       TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    decided_at    TEXT,
    decision      TEXT,
    executed_at   TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS approvals_pending_identity
    ON approvals(execution_id, fingerprint) WHERE status = 'pending';

-- 2026-10-30 (learning milestone): learned lessons and skill versions.
-- Two NEW tables only: `CREATE TABLE IF NOT EXISTS` on every open, no
-- column change to an existing table, no backfill, nothing to lose. Lessons
-- cannot live in a JSON column of `executions`: they outlive the run that
-- produced them, are scoped to a CAPABILITY rather than a run, and must be
-- queried by relevance (status + capability) on every request.
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id      TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    status         TEXT NOT NULL,
    category       TEXT NOT NULL,
    capability     TEXT NOT NULL,
    scope          TEXT NOT NULL,
    problem        TEXT NOT NULL,
    root_cause     TEXT,
    corrective_behavior TEXT NOT NULL,
    validation     TEXT,
    confidence     REAL NOT NULL,
    evidence       TEXT,
    source_execution_id TEXT,
    source_text    TEXT,
    origin         TEXT NOT NULL,
    feedback_type  TEXT,
    impact         TEXT,
    temporary      INTEGER NOT NULL DEFAULT 0,
    calibration    TEXT,
    hits           INTEGER NOT NULL DEFAULT 0,
    successes      INTEGER NOT NULL DEFAULT 0,
    failures       INTEGER NOT NULL DEFAULT 0,
    activated_at   TEXT,
    disabled_at    TEXT,
    superseded_by  TEXT
);

CREATE INDEX IF NOT EXISTS lessons_status_scope
    ON lessons(status, capability);

CREATE TABLE IF NOT EXISTS skill_versions (
    version_id      TEXT PRIMARY KEY,
    skill_id        TEXT NOT NULL,
    version         INTEGER NOT NULL,
    status          TEXT NOT NULL,
    instructions    TEXT NOT NULL,
    validation      TEXT,
    source_lesson_ids TEXT,
    evaluation      TEXT,
    created_at      TEXT NOT NULL,
    activated_at    TEXT,
    retired_at      TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS skill_versions_identity
    ON skill_versions(skill_id, version);
"""


class ExecutionStore:
    """Thread-safe persistence for the Control Center preview."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._lock = threading.Lock()
        # W2.1 observability: how many event writes failed and were then
        # failed-soft by the runner (it prints and the run continues). A
        # count only — never the payload, the error text or any row data —
        # surfaced through the EXISTING storage health check so a silently
        # degrading timeline is visible without a new monitoring subsystem.
        self.event_write_failures = 0
        # GUI-10: pending-approval TTL (patchable in tests).
        self.approval_ttl_seconds = APPROVAL_TTL_SECONDS
        # GUI-10 recovery: how long one resolver's durable claim lasts (see
        # resume_claim; patchable in tests).
        self.resume_lease_seconds = RESUME_LEASE_SECONDS
        self._conn = _connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        _migrate_schema(self._conn)
        self._conn.commit()

    # -- executions -------------------------------------------------------

    def create_execution(
        self, execution_id: str, question: str, status: str = STATUS_QUEUED
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO executions (id, question, status, created_at) "
                "VALUES (?, ?, ?, ?)",
                (execution_id, question, status, utc_now_iso()),
            )
            self._conn.commit()

    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM executions WHERE id = ?", (execution_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        # JSON columns come back as TEXT; decode for the API/UI layer.
        d["plan"] = _decode_plan(d.get("plan"))
        for col in ("qa_flags", "error"):
            val = d.get(col)
            if isinstance(val, str) and val.strip():
                try:
                    d[col] = json.loads(val)
                except Exception:
                    pass
        d["sources"] = _decode_json_list(d.get("sources"))
        return d

    def list_executions(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM executions ORDER BY created_at DESC, id DESC"
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            # List stays lightweight: the plan is only served by the detail
            # and /plan endpoints, never embedded in history rows.
            d.pop("plan", None)
            for col in ("qa_flags", "error"):
                val = d.get(col)
                if isinstance(val, str) and val.strip():
                    try:
                        d[col] = json.loads(val)
                    except Exception:
                        pass
            d["sources"] = _decode_json_list(d.get("sources"))
            out.append(d)
        return out

    def counts(self) -> Dict[str, Any]:
        """Read-only operational counts for the Control Center (2026-10-30).

        Aggregate SELECTs only — no row payloads, no new table, no migration.
        Used by the dashboard/memory projections so those pages never load
        the whole history to count it.
        """
        with self._lock:
            by_status_rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM executions GROUP BY status"
            ).fetchall()
            approvals_rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM approvals GROUP BY status"
            ).fetchall()
            events_total = self._conn.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()
            span = self._conn.execute(
                "SELECT MIN(created_at) AS oldest, MAX(created_at) AS newest "
                "FROM executions"
            ).fetchone()
        by_status = {str(r["status"]): int(r["n"]) for r in by_status_rows}
        by_approval = {str(r["status"]): int(r["n"]) for r in approvals_rows}
        return {
            "executions": {
                "total": sum(by_status.values()),
                "by_status": by_status,
            },
            "approvals": {
                "total": sum(by_approval.values()),
                "by_status": by_approval,
            },
            "events": int(events_total["n"] if events_total else 0),
            "oldest": span["oldest"] if span else None,
            "newest": span["newest"] if span else None,
        }

    # -- learned lessons --------------------------------------------------

    LESSON_COLUMNS = (
        "status", "category", "capability", "scope", "problem",
        "root_cause", "corrective_behavior", "validation", "confidence",
        "evidence", "source_execution_id", "source_text", "origin",
        "feedback_type", "impact", "temporary", "calibration",
        "hits", "successes", "failures", "activated_at", "disabled_at",
        "superseded_by",
    )

    # JSON columns written as text; every other column is stored verbatim.
    LESSON_JSON_COLUMNS = ("evidence", "calibration")

    def create_lesson(self, lesson: Dict[str, Any]) -> Dict[str, Any]:
        """Insert one lesson row (the caller validates the shape).

        Evidence/source lists are stored as JSON text; the row is returned
        decoded so callers never handle the storage format.
        """
        now = utc_now_iso()
        row = {
            "lesson_id": lesson["lesson_id"],
            "created_at": lesson.get("created_at") or now,
            "updated_at": now,
            **{col: lesson.get(col) for col in self.LESSON_COLUMNS},
        }
        row["confidence"] = float(row.get("confidence") or 0.0)
        for col in ("hits", "successes", "failures", "temporary"):
            row[col] = int(row.get(col) or 0)
        for col in self.LESSON_JSON_COLUMNS:
            value = row.get(col)
            if value is not None and not isinstance(value, str):
                row[col] = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO lessons ("
                + ", ".join(row.keys())
                + ") VALUES (" + ", ".join("?" * len(row)) + ")",
                tuple(row.values()),
            )
            self._conn.commit()
        return self.get_lesson(row["lesson_id"])

    def get_lesson(self, lesson_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM lessons WHERE lesson_id = ?", (lesson_id,)
            ).fetchone()
        return _decode_lesson(dict(row)) if row else None

    def list_lessons(self, status: Optional[str] = None,
                     capability: Optional[str] = None,
                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM lessons"
        where, params = [], []
        if status:
            where.append("status = ?"); params.append(status)
        if capability:
            where.append(
                "(capability = ? OR capability = 'any')"); params.append(capability)
        if where:
            sql += " WHERE " + " AND ".join(where)
        # Newest first; an activated lesson stays visible even after edits.
        sql += " ORDER BY created_at DESC, lesson_id DESC"
        if limit is not None:
            sql += " LIMIT ?"; params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [_decode_lesson(dict(r)) for r in rows]

    def update_lesson(self, lesson_id: str, **fields: Any) -> Dict[str, Any]:
        """Whitelisted partial update. Unknown fields raise ValueError."""
        unknown = set(fields) - set(self.LESSON_COLUMNS)
        if unknown:
            raise ValueError(f"unknown lesson field(s): {sorted(unknown)}")
        if not fields:
            return self.get_lesson(lesson_id)
        if "temporary" in fields:
            fields = {**fields, "temporary": int(bool(fields["temporary"]))}
        sets = [f"{name} = ?" for name in fields]
        params = [json.dumps(v, ensure_ascii=False)
                  if name in self.LESSON_JSON_COLUMNS and v is not None and
                  not isinstance(v, str) else v
                  for name, v in fields.items()]
        with self._lock:
            cur = self._conn.execute(
                "UPDATE lessons SET " + ", ".join(sets)
                + ", updated_at = ? WHERE lesson_id = ?",
                (*params, utc_now_iso(), lesson_id),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                raise KeyError(lesson_id)
        return self.get_lesson(lesson_id)

    def bump_lesson_stats(self, lesson_ids: List[str], *,
                          success: Optional[bool] = None) -> None:
        """Count a lesson's use (and its outcome when known).

        Only ever increments — a lesson's history is evidence, so it is not
        rewritten by a later run.
        """
        if not lesson_ids:
            return
        with self._lock:
            for lesson_id in lesson_ids:
                self._conn.execute(
                    "UPDATE lessons SET hits = hits + 1, updated_at = ? "
                    "WHERE lesson_id = ?", (utc_now_iso(), lesson_id))
                if success is True:
                    self._conn.execute(
                        "UPDATE lessons SET successes = successes + 1 "
                        "WHERE lesson_id = ?", (lesson_id,))
                elif success is False:
                    self._conn.execute(
                        "UPDATE lessons SET failures = failures + 1 "
                        "WHERE lesson_id = ?", (lesson_id,))
            self._conn.commit()

    def lesson_counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM lessons GROUP BY status"
            ).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}

    # -- skill versions ---------------------------------------------------

    def create_skill_version(self, version: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "version_id": version["version_id"],
            "skill_id": version["skill_id"],
            "version": int(version["version"]),
            "status": version.get("status") or "candidate",
            "instructions": version["instructions"],
            "validation": version.get("validation"),
            "source_lesson_ids": json.dumps(
                version.get("source_lesson_ids") or [], ensure_ascii=False),
            "evaluation": (json.dumps(version["evaluation"], ensure_ascii=False)
                           if version.get("evaluation") is not None else None),
            "created_at": version.get("created_at") or utc_now_iso(),
            "activated_at": version.get("activated_at"),
            "retired_at": None,
        }
        with self._lock:
            self._conn.execute(
                "INSERT INTO skill_versions ("
                + ", ".join(row.keys())
                + ") VALUES (" + ", ".join("?" * len(row)) + ")",
                tuple(row.values()),
            )
            self._conn.commit()
        return self.get_skill_version(row["version_id"])

    def get_skill_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM skill_versions WHERE version_id = ?",
                (version_id,)).fetchone()
        return _decode_skill_version(dict(row)) if row else None

    def list_skill_versions(self, skill_id: Optional[str] = None
                            ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM skill_versions"
        params: tuple = ()
        if skill_id:
            sql += " WHERE skill_id = ?"; params = (skill_id,)
        sql += " ORDER BY skill_id ASC, version DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_decode_skill_version(dict(r)) for r in rows]

    def set_skill_version_status(self, version_id: str, status: str,
                                 *, evaluation: Any = None) -> Dict[str, Any]:
        now = utc_now_iso()
        sets = ["status = ?"]
        params: List[Any] = [status]
        if status == "active":
            sets.append("activated_at = ?"); params.append(now)
        if status in ("retired", "rejected"):
            sets.append("retired_at = ?"); params.append(now)
        if evaluation is not None:
            sets.append("evaluation = ?")
            params.append(json.dumps(evaluation, ensure_ascii=False))
        with self._lock:
            cur = self._conn.execute(
                "UPDATE skill_versions SET " + ", ".join(sets)
                + " WHERE version_id = ?", (*params, version_id))
            self._conn.commit()
            if cur.rowcount == 0:
                raise KeyError(version_id)
        return self.get_skill_version(version_id)

    def update_execution(
        self,
        execution_id: str,
        *,
        status: Optional[str] = None,
        route: Optional[str] = None,
        answer: Optional[str] = None,
        qa_flags: Optional[List[str]] = None,
        error: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
        sources: Optional[List[str]] = None,
        started_at: bool = False,
        completed_at: bool = False,
    ) -> None:
        """Partial update. `error=None` explicitly clears an error.

        `sources` is the provenance list returned by the graph (agent names
        and research URLs); persisted as JSON so an execution record can
        later explain where its result came from (W2).
        """
        with self._lock:
            sets, params = [], []
            if status is not None:
                sets.append("status = ?"); params.append(status)
            if route is not None:
                sets.append("route = ?"); params.append(route)
            if answer is not None:
                sets.append("answer = ?"); params.append(answer)
            if qa_flags is not None:
                sets.append("qa_flags = ?"); params.append(json.dumps(qa_flags))
            if error is not None or error is None and error == []:
                sets.append("error = ?"); params.append(json.dumps(error) if isinstance(error, dict) else None)
            if result is not None:
                sets.append("result = ?"); params.append(result)
            if plan is not None:
                sets.append("plan = ?"); params.append(json.dumps(plan))
            if sources is not None:
                sets.append("sources = ?"); params.append(json.dumps(list(sources)))
            if started_at:
                sets.append("started_at = ?"); params.append(utc_now_iso())
            if completed_at:
                sets.append("completed_at = ?"); params.append(utc_now_iso())
            if not sets:
                return
            params.append(execution_id)
            self._conn.execute(
                f"UPDATE executions SET {', '.join(sets)} WHERE id = ?", params
            )
            self._conn.commit()

    # -- lifecycle invariant ------------------------------------------------

    def reconcile_interrupted(self) -> int:
        """Drive every non-terminal execution to a terminal status.

        Returns the number of executions fixed. Idempotent: fixed rows leave
        the non-terminal set, so a second call returns 0.

        WHY THIS EXISTS (2026-10-28, adapted from the reference
        implementation Mohamad AI OS — `unified_agent_runtime.execute`
        always persists a terminal status, and its mission control treats
        COMPLETED/ESCALATED/BLOCKED as the only legal end states): the
        runner can only write `failed` for exceptions it OBSERVES, so a
        process that dies mid-run leaves `queued`/`running` rows that the UI
        then presents as active forever (observed in the real database:
        execution a7bd96990f0b, still `running` from 2026-10-25). Every
        persisted execution must reach a terminal status, even across a
        restart.

        Contract:
          * call once when the API process opens the store, BEFORE it
            accepts requests — no live run exists in this process yet;
          * append-only: answers, routes, plans and existing events are
            never rewritten; exactly one honest closing event is appended
            per fixed execution;
          * fail-soft per row: one unreadable row must not stop the rest
            from reaching a terminal status, and a failure here must never
            prevent the store from being used (callers wrap it).
        """
        err = ExecutionError(
            category="interrupted",
            human_message="The run was interrupted before it finished.",
            why=("Agentica stopped while this execution was still in "
                 "progress — for example the application was closed or "
                 "restarted."),
            recovery=("Run the request again; only this attempt was lost, "
                      "nothing else."),
            recoverable=True,
            severity=SEV_ERROR,
            details={},
        )
        error_json = json.dumps(err.to_dict())
        fixed = 0
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, status FROM executions "
                "WHERE status IN (?, ?)",
                (STATUS_QUEUED, STATUS_RUNNING),
            ).fetchall()
            for row in rows:
                eid = row["id"]
                try:
                    self._conn.execute(
                        "UPDATE executions SET status = ?, error = ?, "
                        "completed_at = ? WHERE id = ?",
                        (STATUS_FAILED, error_json, utc_now_iso(), eid),
                    )
                    # Tail of this execution's own timeline, so the id can
                    # never collide with however the runner numbered events.
                    next_id = self._conn.execute(
                        "SELECT COALESCE(MAX(event_id), 0) + 1 "
                        "FROM events WHERE execution_id = ?",
                        (eid,),
                    ).fetchone()[0]
                    self._conn.execute(
                        "INSERT INTO events (execution_id, event_id, "
                        "timestamp, type, message, step_id, agent_id, "
                        "severity, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            eid, next_id, utc_now_iso(), EXECUTION_FAILED,
                            "The run was interrupted before it finished.",
                            None, None, SEV_ERROR,
                            json.dumps({"previous_status": row["status"],
                                        "category": "interrupted"}),
                        ),
                    )
                    fixed += 1
                except Exception:  # noqa: BLE001 — one bad row must not
                    # block the remaining rows from reaching a terminal
                    # status; the row stays as-is and is retried on the
                    # next startup.
                    continue
            self._conn.commit()
        return fixed

    # -- approvals (GUI-10) --------------------------------------------------

    def _approval_row(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Lock-free read (call WITH self._lock held)."""
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE approval_id = ?",
            (approval_id,),
        ).fetchone()
        return _decode_approval(dict(row)) if row else None

    def create_or_reuse_approval(
        self,
        *,
        operation: str,
        description: str,
        payload: Dict[str, Any],
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist (or re-deliver) the ONE pending request bound to this
        exact operation.

        Idempotent by fingerprint: repeated gate deliveries for the same
        execution + bound payload return the SAME pending approval_id
        (duplicate delivery can never mint uncontrolled requests). The
        payload is built server-side by the gate — a client can never
        submit an operation to approve.
        """
        # Only deliberately supported operations are approvable; a new
        # operation type must be added here explicitly.
        if operation == "research":
            if not isinstance(payload, dict) or not str(
                    payload.get("question") or "").strip():
                raise ValueError(
                    "approval payload must carry the bound question")
        elif operation == "workspace_patch":
            # Validated with the SAME schema the resume path enforces, so a
            # request can never be persisted with an unusable/unsafe patch.
            from app.coding.patch import PatchError, validate_payload

            if not isinstance(payload, dict) or not str(
                    payload.get("question") or "").strip():
                raise ValueError(
                    "approval payload must carry the bound request")
            try:
                validate_payload(payload)
            except PatchError as exc:
                raise ValueError(f"invalid workspace patch: {exc}") from exc
        else:
            raise ValueError(f"unsupported approval operation: {operation!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("approval description is required")
        fingerprint = hashlib.sha256(
            json.dumps({"operation": operation, "payload": payload},
                       sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        now = utc_now_iso()
        expires = (datetime.now(timezone.utc)
                   + timedelta(seconds=self.approval_ttl_seconds)).isoformat()
        approval_id = uuid.uuid4().hex[:12]
        with self._lock:
            if execution_id is None:
                existing = self._conn.execute(
                    "SELECT approval_id FROM approvals "
                    "WHERE execution_id IS NULL AND fingerprint = ? "
                    "AND status = ?",
                    (fingerprint, APPROVAL_PENDING),
                ).fetchone()
            else:
                existing = self._conn.execute(
                    "SELECT approval_id FROM approvals "
                    "WHERE execution_id = ? AND fingerprint = ? "
                    "AND status = ?",
                    (execution_id, fingerprint, APPROVAL_PENDING),
                ).fetchone()
            if existing:
                row = self._approval_row(existing["approval_id"])
                return serialize_approval(row, reused=True)
            self._conn.execute(
                "INSERT INTO approvals (approval_id, execution_id, operation, "
                "description, payload, fingerprint, status, created_at, "
                "expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (approval_id, execution_id, operation,
                 description.strip()[:400],
                 json.dumps(payload, ensure_ascii=False), fingerprint,
                 APPROVAL_PENDING, now, expires),
            )
            self._conn.commit()
            row = self._approval_row(approval_id)
        return serialize_approval(row, reused=False)

    def get_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._approval_row(approval_id)

    def list_approvals(self, status: str = APPROVAL_PENDING) -> List[Dict[str, Any]]:
        if status not in APPROVAL_STATUSES:
            raise ValueError(f"unknown approval status: {status!r}")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM approvals WHERE status = ? "
                "ORDER BY created_at ASC, approval_id ASC",
                (status,),
            ).fetchall()
            return [serialize_approval(self._approval_row(r["approval_id"]))
                    for r in rows]

    def decide_approval(self, approval_id: str, outcome: str) -> Dict[str, Any]:
        """Atomic pending -> approved | rejected transition.

        * unknown id -> ApprovalNotFound;
        * already decided: same outcome returns idempotently
          (already_decided=True, original decided_at untouched), a DIFFERENT
          outcome raises ApprovalConflict and never overwrites the first
          decision;
        * pending past its expiry -> the row is driven to `expired` and
          ApprovalExpired is raised (never silently approved);
        * the decision is appended to the bound execution's timeline.
        """
        if outcome not in (APPROVAL_APPROVED, APPROVAL_REJECTED):
            raise ValueError(f"invalid approval outcome: {outcome!r}")
        now = utc_now_iso()
        with self._lock:
            row = self._approval_row(approval_id)
            if row is None:
                raise ApprovalNotFound(approval_id)
            if row["status"] != APPROVAL_PENDING:
                if row.get("decision") == outcome:
                    return serialize_approval(row, already_decided=True)
                raise ApprovalConflict(approval_id)
            if _is_past(row.get("expires_at")):
                self._conn.execute(
                    "UPDATE approvals SET status = ? "
                    "WHERE approval_id = ? AND status = ?",
                    (APPROVAL_EXPIRED, approval_id, APPROVAL_PENDING),
                )
                self._conn.commit()
                raise ApprovalExpired(approval_id)
            self._conn.execute(
                "UPDATE approvals SET status = ?, decision = ?, decided_at = ? "
                "WHERE approval_id = ? AND status = ?",
                (outcome, outcome, now, approval_id, APPROVAL_PENDING),
            )
            self._conn.commit()
            row = self._approval_row(approval_id)
            if row.get("execution_id"):
                # Append-only timeline record, inserted inline (append_event
                # would deadlock on the same lock).
                next_id = self._conn.execute(
                    "SELECT COALESCE(MAX(event_id), 0) + 1 FROM events "
                    "WHERE execution_id = ?",
                    (row["execution_id"],),
                ).fetchone()[0]
                verb = "granted" if outcome == APPROVAL_APPROVED else "rejected"
                label = _APPROVAL_LABELS.get(
                    row.get("operation"), str(row.get("operation") or ""))
                self._conn.execute(
                    "INSERT INTO events (execution_id, event_id, timestamp, "
                    "type, message, step_id, agent_id, severity, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["execution_id"], next_id, now, APPROVAL_DECIDED,
                     f"{label} approval {verb}",
                     "approval", None, SEV_INFO,
                     json.dumps({"approval_id": approval_id,
                                 "outcome": outcome})),
                )
                self._conn.commit()
            return serialize_approval(row, already_decided=False)

    def claim_approval_execution(self, approval_id: str) -> bool:
        """Atomically reserve THE single execution this approval permits.

        Returns True exactly once per approval: only an approved, unexpired,
        never-yet-executed request can be claimed, so duplicate resume
        requests can never run the same authorized operation twice.
        """
        now = utc_now_iso()
        with self._lock:
            row = self._approval_row(approval_id)
            if (row is None or row["status"] != APPROVAL_APPROVED
                    or row.get("executed_at") or _is_past(row.get("expires_at"))):
                return False
            cur = self._conn.execute(
                "UPDATE approvals SET executed_at = ? "
                "WHERE approval_id = ? AND status = ? AND executed_at IS NULL",
                (now, approval_id, APPROVAL_APPROVED),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def release_approval_execution(self, approval_id: str) -> None:
        """Give the claim back when the approved operation could NOT run, so
        the approval is not silently consumed by a failure."""
        with self._lock:
            self._conn.execute(
                "UPDATE approvals SET executed_at = NULL "
                "WHERE approval_id = ?",
                (approval_id,),
            )
            self._conn.commit()

    def mark_expired_if_due(self, approval_id: str) -> bool:
        """Drive a PENDING request whose decision window passed to `expired`
        (idempotent). Returns True when this call performed the transition."""
        with self._lock:
            row = self._approval_row(approval_id)
            if (row is None or row["status"] != APPROVAL_PENDING
                    or not _is_past(row.get("expires_at"))):
                return False
            cur = self._conn.execute(
                "UPDATE approvals SET status = ? "
                "WHERE approval_id = ? AND status = ?",
                (APPROVAL_EXPIRED, approval_id, APPROVAL_PENDING),
            )
            self._conn.commit()
            return cur.rowcount == 1    # -- resume claim (GUI-10 recovery) --------------------------------------

    @contextmanager
    def resume_claim(self, execution_id: str):
        """Try-acquire the DURABLE exclusive claim to resolve an execution.

        Yields True for exactly one resolver — across threads AND across
        processes sharing this SQLite file — and False immediately for
        everyone else (no blocking, no queued duplicate work). The claim is
        one atomic conditional UPDATE on the execution row, so the winner is
        decided by SQLite, not by a process-local lock.

        The claim is released on exit (only by its owner), so a later retry
        can acquire it again. Expiry bounds a crashed holder; taking over an
        EXPIRED lease is safe because task progress is persisted task by task
        and a task left `running` closes the run `failed/resume_interrupted`
        instead of being replayed. The lease never authorizes repeating an
        external operation: that guarantee remains `approvals.executed_at`.
        """
        owner = uuid.uuid4().hex
        expires = (datetime.now(timezone.utc)
                   + timedelta(seconds=self.resume_lease_seconds)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE executions SET resume_lease_owner = ?, "
                "resume_lease_expires_at = ? "
                "WHERE id = ? AND (resume_lease_expires_at IS NULL "
                "OR resume_lease_expires_at <= ?)",
                (owner, expires, execution_id, utc_now_iso()),
            )
            self._conn.commit()
            acquired = cur.rowcount == 1
        try:
            yield acquired
        finally:
            if acquired:
                # Owner-checked release: a resolver whose lease already
                # expired (and was taken over) must never clear the NEW
                # holder's claim.
                with self._lock:
                    self._conn.execute(
                        "UPDATE executions SET resume_lease_owner = NULL, "
                        "resume_lease_expires_at = NULL "
                        "WHERE id = ? AND resume_lease_owner = ?",
                        (execution_id, owner),
                    )
                    self._conn.commit()

    # -- events -----------------------------------------------------------
    def append_event(
        self,
        execution_id: str,
        event_id: Optional[int],
        type: str,
        message: str,
        *,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        severity: str = "info",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one event. `event_id=None` assigns the next id FOR THIS
        EXECUTION inside the lock (GUI-10): resume/decision events share a
        timeline with events written by earlier runners, so a process-local
        counter would collide with the existing rows. An explicit id keeps
        working for existing callers/tests."""
        with self._lock:
            if event_id is None:
                event_id = self._conn.execute(
                    "SELECT COALESCE(MAX(event_id), 0) + 1 FROM events "
                    "WHERE execution_id = ?",
                    (execution_id,),
                ).fetchone()[0]
            try:
                self._conn.execute(
                    "INSERT INTO events (execution_id, event_id, timestamp, "
                    "type, message, step_id, agent_id, severity, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        execution_id,
                        event_id,
                        utc_now_iso(),
                        type,
                        message,
                        step_id,
                        agent_id,
                        severity,
                        json.dumps(payload or {}),
                    ),
                )
                self._conn.commit()
            except Exception:
                # W2.1: count the degradation, then RE-RAISE so the runner's
                # existing fail-soft path (print + keep the run alive) is
                # unchanged — the counter is what makes it observable.
                self.event_write_failures += 1
                raise

    def get_events(self, execution_id: str, after: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE execution_id = ? AND event_id > ? "
                "ORDER BY event_id ASC",
                (execution_id, after),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.get("payload") or "{}")
            out.append(d)
        return out

    def close(self) -> None:
        self._conn.close()
