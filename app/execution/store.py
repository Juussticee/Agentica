"""Execution store (GUI-2): SQLite persistence for executions and events.

Local-first: one SQLite file (storage/control_center.sqlite3), binary data
never committed (storage/ is git-ignored). Both the snapshot (current state
per execution) and the append-only event history live here so that:
  * the UI can render "current state" cheaply from the executions table
  * the UI can render the full timeline and replay after `after=<event_id>`.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.execution.events import utc_now_iso

DEFAULT_DB_PATH = Path("storage") / "control_center.sqlite3"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


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
    """Add columns introduced after the GUI-2 schema (GUI-12: `plan`).

    Existing databases keep working unchanged: legacy rows simply have
    plan = NULL and continue to load (backwards-compatible).
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(executions)")}
    if "plan" not in cols:
        conn.execute("ALTER TABLE executions ADD COLUMN plan TEXT")


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
    plan          TEXT
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
"""


class ExecutionStore:
    """Thread-safe persistence for the Control Center preview."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._lock = threading.Lock()
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
            out.append(d)
        return out

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
        started_at: bool = False,
        completed_at: bool = False,
    ) -> None:
        """Partial update. `error=None` explicitly clears an error."""
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

    # -- events -----------------------------------------------------------

    def append_event(
        self,
        execution_id: str,
        event_id: int,
        type: str,
        message: str,
        *,
        step_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        severity: str = "info",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (execution_id, event_id, timestamp, type, message, "
                "step_id, agent_id, severity, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
