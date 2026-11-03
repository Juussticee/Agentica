"""Control Center read-only projections (2026-10-30).

Every payload the management UI shows is built HERE from real state — the
agent registry, the execution store, config-only provider readiness, the
documented skill files and the knowledge corpus directory. No sample data, no
capability that is not implemented, no secret or absolute path.

  * `agent_catalog`      registered specialists + orchestration: tools,
                         permissions, model use, availability (config only),
                         and recent activity from real executions.
  * `skills_catalog`     the documented procedures, with their file's real
                         size/mtime and the explicit fact that they are
                         DOCUMENTATION — there is no skill runtime, so nothing
                         here can be "installed" or "enabled" from the UI.
  * `memory_summary`     what actually persists (execution history, approval
                         decisions, the document corpus) and what does NOT
                         (user preferences, learned procedures) — the gaps are
                         named, never mocked.
  * `dashboard_summary`  operational counts, the attention queue and a real
                         activity timeline merged from recorded events.

Availability is a CONFIG verdict, never a probe: it answers "could this agent
run right now if asked?" without touching the network, a model or the RAG
index. Anything that needs a live check stays in /api/v1/services.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents import registry

MAX_RECENT_EXECUTIONS = 12
MAX_ACTIVITY = 25
MAX_ATTENTION = 5
MAX_CORPUS_ITEMS = 20
MAX_LESSON_ITEMS = 25

# Presentation only: the node id stays the key everywhere else.
AGENT_TITLES = {
    "conversational": "Conversation",
    "general": "General Knowledge",
    "unsupported": "Capability Guidance",
    "books": "Internal Knowledge (RAG)",
    "research": "Web Research",
    "erp": "Business Data",
    "feasibility": "Feasibility Review",
    "coding": "Workspace Coding",
    "data": "Data Analyst",
    "followup": "Contextual Follow-up",
    "feedback": "Feedback & Learning",
    "synthesis": "Synthesis",
}
ORCHESTRATION_TITLES = {
    "supervisor": "Supervisor",
    "planner": "Planner",
    "dispatcher": "Dispatcher",
    "qa": "Quality Gate",
}

# Which agent requests an approval, keyed by the PERSISTED operation type.
APPROVAL_AGENTS = {
    "research": "research",
    "workspace_patch": "coding",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path, root: Path) -> str:
    """Workspace-relative POSIX path — never an absolute one."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _provider() -> Dict[str, Any]:
    """Config-only provider facts (no key, no endpoint credentials)."""
    try:
        from app import diagnostics_core

        readiness = diagnostics_core.provider_readiness()
    except Exception:  # noqa: BLE001 — availability must never 500 the page
        return {"provider": "unknown", "state": "unknown",
                "model": None, "note": "Provider settings could not be read."}
    what = str(readiness.get("what_will_be_tested") or "")
    return {
        "provider": readiness.get("provider"),
        "state": readiness.get("state"),
        "model": what or None,
        "note": readiness.get("state_hint"),
    }


def _workspace_state() -> tuple:
    """(configured_root: Path | None, problem: str | None)."""
    from app.config import CODING_WORKSPACE_ROOT

    raw = str(CODING_WORKSPACE_ROOT or "").strip()
    if not raw:
        return None, "No workspace is configured (CODING_WORKSPACE_ROOT is unset)."
    path = Path(raw)
    if not path.is_dir():
        return path, "The configured workspace path does not exist."
    return path, None


def _knowledge_files() -> List[Dict[str, Any]]:
    from app.config import BOOKS_DIR, PROJECT_ROOT

    items: List[Dict[str, Any]] = []
    if not BOOKS_DIR.is_dir():
        return items
    for path in sorted(BOOKS_DIR.iterdir())[:MAX_CORPUS_ITEMS]:
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append({
            "name": path.name,
            "path": _rel(path, PROJECT_ROOT),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        })
    return items


def _index_present() -> Optional[bool]:
    from app.config import INDEX_MANIFEST

    return INDEX_MANIFEST.is_file()


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

def _agent_availability(agent_id: str, entry: Dict[str, Any],
                        provider: Dict[str, Any]) -> tuple:
    """(state, reason) from configuration only.

    state: "available" | "unavailable" | "disabled".
    A "disabled" agent is intentionally switched off (an unset opt-in); an
    "unavailable" agent is configured but cannot run as things stand.
    """
    provider_ready = provider.get("state") == "ready"
    if agent_id in ("conversational", "unsupported", "qa"):
        return "available", "No external dependency."
    if agent_id == "data" or agent_id == "coding":
        _root, problem = _workspace_state()
        if problem:
            return "disabled", problem
        return ("available",
                "Read-only inside the configured workspace."
                if agent_id == "data"
                else "Read-only until a patch is approved.")
    if agent_id == "synthesis":
        return "available", ("Composes completed results; without a provider "
                            "it returns them verbatim instead of the model.")
    if agent_id == "research":
        from app.config import SEARXNG_URL

        if not str(SEARXNG_URL or "").strip():
            return "unavailable", "SEARXNG_URL is not configured."
        return "available", "Search runs only after an approval bound to the run."
    if agent_id == "books":
        from app.config import BOOKS_DIR

        if not BOOKS_DIR.is_dir() or not any(BOOKS_DIR.iterdir()):
            return "unavailable", "No documents in knowledge/books to retrieve."
        return "available", "Answers only from the indexed corpus."
    if agent_id == "erp":
        from app.config import ERP_ADAPTER

        if ERP_ADAPTER != "mock":
            return "unavailable", f"ERP adapter {ERP_ADAPTER!r} is not implemented."
        return "available", "Read-only mock adapter (no real ERP connected)."
    if agent_id == "feasibility":
        if not provider_ready:
            return "unavailable", "Needs an AI provider to compose the study."
        return "available", "Also needs research + books to have produced output."
    if agent_id == "general":
        if not provider_ready:
            return "unavailable", "Needs an AI provider; none is configured."
        return "available", "One model completion, no tools."
    return "available", "No external dependency."


def _agent_activity(store, agent_id: str, entry: Dict[str, Any],
                    executions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recent real executions this agent took part in (bounded)."""
    from app.execution.store import serialize_plan

    found: List[Dict[str, Any]] = []
    capability = entry.get("planner_capability")
    for row in executions:
        eid = row.get("id")
        at = row.get("completed_at") or row.get("created_at")
        if capability and row.get("plan"):
            plan = serialize_plan(row.get("plan")) or {}
            for task in plan.get("tasks") or []:
                if str(task.get("capability")) != str(capability):
                    continue
                found.append({
                    "execution_id": eid,
                    "question": row.get("question"),
                    "execution_status": row.get("status"),
                    "at": at,
                    "task_id": task.get("task_id"),
                    "task_status": task.get("status"),
                })
                break
        elif row.get("route") == agent_id:
            found.append({
                "execution_id": eid,
                "question": row.get("question"),
                "execution_status": row.get("status"),
                "at": at,
                "task_id": None,
                "task_status": None,
            })
        if len(found) >= 3:
            break
    return found


def _recent_rows(store, limit: int = MAX_RECENT_EXECUTIONS
                 ) -> List[Dict[str, Any]]:
    """Newest executions WITH their plans (bounded, one query each)."""
    rows: List[Dict[str, Any]] = []
    for listed in store.list_executions()[:limit]:
        row = store.get_execution(listed["id"])
        if row:
            rows.append(row)
    return rows


def agent_catalog(store) -> Dict[str, Any]:
    provider = _provider()
    executions = _recent_rows(store)
    agents: List[Dict[str, Any]] = []
    for agent_id, entry in registry.AGENTS.items():
        state, reason = _agent_availability(agent_id, entry, provider)
        agents.append({
            "id": agent_id,
            "title": AGENT_TITLES.get(agent_id, agent_id.replace("_", " ").title()),
            "kind": entry.get("kind"),
            "responsibility": entry.get("route_reason"),
            "availability": state,
            "availability_reason": reason,
            "tools": list(entry.get("tools") or []),
            "permissions": dict(entry.get("permissions") or {}),
            "model": entry.get("model"),
            "routes": list(entry.get("routes") or []),
            "planner_capability": entry.get("planner_capability"),
            "node": entry.get("node"),
            "reads": list(entry.get("reads") or []),
            "writes": list(entry.get("writes") or []),
            "rejects": entry.get("rejects"),
            "tests": entry.get("tests"),
            "implemented": True,
            "recent_activity": _agent_activity(store, agent_id, entry, executions),
        })

    orchestration = [
        {
            "id": node_id,
            "title": ORCHESTRATION_TITLES.get(node_id, node_id.title()),
            "does": entry.get("does"),
            "availability": "available",
            "availability_reason": "Always part of every run.",
            "tests": entry.get("tests"),
        }
        for node_id, entry in registry.ORCHESTRATION.items()
    ]

    counts = {"registered": len(agents), "available": 0, "disabled": 0,
              "unavailable": 0}
    for agent in agents:
        counts[agent["availability"]] = counts.get(agent["availability"], 0) + 1
    return {
        "checked_at": _now(),
        "provider": provider,
        "agents": agents,
        "orchestration": orchestration,
        "counts": counts,
        "planned": _planned_capabilities(),
        "note": ("Availability is a configuration verdict: it answers whether "
                 "an agent could run now, without contacting any service."),
    }


def _planned_capabilities() -> List[Dict[str, str]]:
    """Roles that are NOT implemented, kept out of the agent list on purpose.

    An unimplemented role is never registered as an agent: a prompt without a
    capability is exactly what this project refuses to label a specialist.
    """
    return [
        {"id": "personal_assistant",
         "title": "Personal Assistant",
         "reason": "No conversation or task memory exists yet, so there is "
                   "nothing to coordinate across requests."},
        {"id": "automation",
         "title": "Automation",
         "reason": "No integration is connected; the unsupported route says so."},
        {"id": "verification_model",
         "title": "Model-based verification",
         "reason": "QA is deterministic only; it never treats model "
                   "confidence as proof."},
    ]


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #

def skills_catalog(store=None) -> Dict[str, Any]:
    """Documented procedures + any LEARNED versions of them.

    `store=None` (the pre-learning callers) reports no learned versions
    instead of inventing them.
    """
    from app.config import PROJECT_ROOT

    skills: List[Dict[str, Any]] = []
    for skill_id, description in registry.SKILLS.items():
        path_text = description.split(" — ")[0].split(" § ")[0].strip()
        path = PROJECT_ROOT / path_text
        installed = path.is_file()
        stat = path.stat() if installed else None
        skills.append({
            "id": skill_id,
            "title": skill_id.replace("-", " ").title(),
            "purpose": description,
            "path": path_text,
            "installed": installed,
            "size_bytes": stat.st_size if stat else None,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc).isoformat(timespec="seconds"
                                                      ) if stat else None,
            "kind": "documented procedure",
            "executable": False,
            "owner_agent": None,
            "required_tools": [],
            "permissions": "None — a skill is documentation, not a permission.",
        })
    learned: List[Dict[str, Any]] = []
    if store is not None:
        try:
            learned = [_skill_view(v) for v in store.list_skill_versions()]
        except Exception:  # noqa: BLE001 — older database without the table
            learned = []
    return {
        "checked_at": _now(),
        "skills": skills,
        "learned_versions": learned,
        "count": len(skills),
        "installed": sum(1 for s in skills if s["installed"]),
        "can_install": False,
        "note": ("Skills are reusable PROCEDURES and instructions stored as "
                 "documents. Agentica has no skill runtime: a skill cannot be "
                 "installed, enabled or executed, and it grants no tool "
                 "access. Tools are what an agent may call; agents are who "
                 "does the work."),
    }


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def memory_summary(store) -> Dict[str, Any]:
    """What persists, what does not — with provenance for each section."""
    from app.config import BOOKS_DIR, COMPANY_DIR, PROJECT_ROOT, STORAGE_DIR

    counts = store.counts()
    executions = store.list_executions()[:10]
    approvals = {}
    for status in ("pending", "approved", "rejected", "expired"):
        try:
            approvals[status] = len(store.list_approvals(status))
        except ValueError:
            approvals[status] = 0

    # 2026-10-30 (learning): real lesson rows, not a placeholder gap. Read
    # fail-soft: an older database without the table reports an empty set.
    lessons: List[Dict[str, Any]] = []
    try:
        lessons = store.list_lessons(limit=MAX_LESSON_ITEMS)
        learn_counts = dict(store.lesson_counts())
    except Exception:  # noqa: BLE001
        learn_counts = {}
    learn_total = sum(
        value for key, value in learn_counts.items() if not key.startswith("_"))
    preferences = [
        {
            "id": lesson.get("lesson_id"),
            "behavior": lesson.get("corrective_behavior"),
            "capability": lesson.get("capability"),
            "created_at": lesson.get("created_at"),
        }
        for lesson in lessons
        if lesson.get("status") == "active"
        and lesson.get("category") == "preference"
    ][:MAX_LESSON_ITEMS]

    corpus = _knowledge_files()
    company = []
    if COMPANY_DIR.is_dir():
        for path in sorted(COMPANY_DIR.iterdir())[:MAX_CORPUS_ITEMS]:
            if path.is_file():
                company.append({"name": path.name,
                                "path": _rel(path, PROJECT_ROOT)})

    sections: List[Dict[str, Any]] = [
        {
            "id": "preferences",
            "title": "User preferences & working context",
            "status": "available" if preferences else "empty",
            "summary": (f"{len(preferences)} active preference lesson(s), "
                        "learned from corrections and applied to matching "
                        "tasks." if preferences else
                        "No preference has been learned yet. Preferences are "
                        "only created from an explicit user correction — "
                        "they are never inferred from a conversation."),
            "items": preferences,
            "provenance": ("same `lessons` table, category=preference "
                           "(see the Learning page)" if preferences else None),
        },
        {
            "id": "executions",
            "title": "Execution history",
            "status": "available" if counts["executions"]["total"] else "empty",
            "summary": f"{counts['executions']['total']} execution(s) "
                       "persisted locally, newest first.",
            "counts": counts["executions"],
            "items": [
                {
                    "id": row["id"],
                    "question": row.get("question"),
                    "status": row.get("status"),
                    "created_at": row.get("created_at"),
                    "completed_at": row.get("completed_at"),
                }
                for row in executions
            ],
            "provenance": "storage/control_center.sqlite3 — written by the "
                          "runner for every execution",
        },
        {
            "id": "approvals",
            "title": "Approval decisions",
            "status": "available" if sum(approvals.values()) else "empty",
            "summary": f"{approvals['pending']} pending · "
                       f"{approvals['approved']} approved · "
                       f"{approvals['rejected']} rejected · "
                       f"{approvals['expired']} expired.",
            "counts": approvals,
            "items": [],
            "provenance": "same database, `approvals` table (decisions are "
                          "never overwritten)",
        },
        {
            "id": "knowledge",
            "title": "Indexed knowledge (RAG)",
            "status": "available" if corpus else "empty",
            "summary": (f"{len(corpus)} document(s) under knowledge/books."
                        if corpus else
                        "No documents found under knowledge/books."),
            "index_present": _index_present(),
            "index_path": "storage/.index_manifest.json",
            "storage_dir_present": STORAGE_DIR.is_dir(),
            "books_dir": _rel(BOOKS_DIR, PROJECT_ROOT),
            "items": corpus,
            "company_items": company,
            "provenance": "Documents on disk; retrieval is read-only and "
                          "answers cite the passages they used.",
        },
        {
            "id": "procedures",
            "title": "Learned procedures",
            "status": "available" if learn_total else "empty",
            "summary": (f"{learn_counts.get('active', 0)} active lesson(s) "
                        f"from {learn_total} recorded, "
                        f"{learn_counts.get('candidate', 0)} awaiting review."
                        if learn_total else
                        "No lesson has been learned yet: corrections and "
                        "execution failures are not recorded as lessons."),
            "counts": learn_counts,
            "items": [
                {
                    "id": lesson.get("lesson_id"),
                    "status": lesson.get("status"),
                    "category": lesson.get("category"),
                    "capability": lesson.get("capability"),
                    "behavior": lesson.get("corrective_behavior"),
                    "confidence": lesson.get("confidence"),
                    "origin": lesson.get("origin"),
                    "created_at": lesson.get("created_at"),
                }
                for lesson in (lessons or [])[:MAX_LESSON_ITEMS]
            ],
            "provenance": "storage/control_center.sqlite3 — `lessons` table, "
                          "written only by the feedback node and review "
                          "actions (a lesson keeps its source execution id)",
        },
    ]
    return {
        "checked_at": _now(),
        "sections": sections,
        "counts": counts,
        "note": ("Model answers are never treated as trusted memory. "
                 "Persisted here: executions, approval decisions, the "
                 "document corpus, and lessons learned from explicit user "
                 "corrections or recorded failures."),
    }


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

def _activity(store, executions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for row in executions[:10]:
        for event in store.get_events(row["id"]):
            events.append({
                "at": event.get("timestamp"),
                "type": event.get("type"),
                "message": event.get("message"),
                "severity": event.get("severity"),
                "agent_id": event.get("agent_id"),
                "execution_id": event.get("execution_id") or row["id"],
                # Per-execution sequence, kept only for tie-breaking below: a
                # timestamp with second resolution ties for events written in
                # the same second, and the later event must still sort first.
                "_seq": int(event.get("event_id") or 0),
            })
    events.sort(key=lambda e: (str(e.get("at") or ""), e["_seq"]),
                reverse=True)
    for event in events:
        event.pop("_seq", None)
    return events[:MAX_ACTIVITY]


def dashboard_summary(store) -> Dict[str, Any]:
    provider = _provider()
    counts = store.counts()
    rows = _recent_rows(store)

    agents_available = 0
    disabled = 0
    unavailable = 0
    for agent_id, entry in registry.AGENTS.items():
        state, _reason = _agent_availability(agent_id, entry, provider)
        if state == "available":
            agents_available += 1
        elif state == "disabled":
            disabled += 1
        else:
            unavailable += 1

    pending = len(store.list_approvals("pending"))
    status_counts = counts["executions"]["by_status"]

    attention = []
    for row in rows:
        if row.get("status") not in ("failed", "waiting_for_approval"):
            continue
        error = row.get("error") or {}
        attention.append({
            "id": row["id"],
            "question": row.get("question"),
            "status": row.get("status"),
            "category": error.get("category"),
            "human_message": error.get("human_message"),
            "created_at": row.get("created_at"),
        })
        if len(attention) >= MAX_ATTENTION:
            break

    since = (datetime.now(timezone.utc) - timedelta(hours=24)
             ).isoformat(timespec="seconds")
    last_24h = sum(1 for row in rows if str(row.get("created_at") or "") >= since)

    return {
        "checked_at": _now(),
        "provider": provider,
        "agents": {
            "registered": len(registry.AGENTS),
            "available": agents_available,
            "disabled": disabled,
            "unavailable": unavailable,
            "orchestration": len(registry.ORCHESTRATION),
        },
        "skills": {"installed": sum(
            1 for path in _skill_paths() if (path).is_file()),
            "total": len(registry.SKILLS)},
        "executions": {
            "total": counts["executions"]["total"],
            "by_status": status_counts,
            "active": status_counts.get("running", 0) + status_counts.get("queued", 0),
            "waiting_approval": status_counts.get("waiting_for_approval", 0),
            "failed": status_counts.get("failed", 0),
            "last_24h": last_24h,
        },
        "approvals": {"pending": pending},
        "attention": attention,
        "recent_executions": [
            {
                "id": row["id"],
                "question": row.get("question"),
                "status": row.get("status"),
                "route": row.get("route"),
                "created_at": row.get("created_at"),
                "completed_at": row.get("completed_at"),
                "has_plan": bool(row.get("plan")),
            }
            for row in rows[:6]
        ],
        "activity": _activity(store, rows),
        "health": {
            "backend": "ok",
            "ai": {"state": provider.get("state"),
                   "provider": provider.get("provider"),
                   "model": provider.get("model")},
            "storage": "ok",
        },
        "note": ("Every figure is read from the local execution store and "
                 "the agent registry; nothing is estimated."),
    }


def _skill_paths() -> List[Path]:
    from app.config import PROJECT_ROOT

    paths: List[Path] = []
    for description in registry.SKILLS.values():
        text = description.split(" — ")[0].split(" § ")[0].strip()
        paths.append(PROJECT_ROOT / text)
    return paths


# --------------------------------------------------------------------------- #
# Helpers used by the API layer
# --------------------------------------------------------------------------- #

def _epoch(value: object) -> Optional[float]:
    """Parse a stored/queried timestamp to epoch seconds, or None.

    The store writes aware ISO strings with microseconds; the UI sends
    `Date#toISOString()`, i.e. milliseconds and a trailing `Z`. Comparing the
    two as STRINGS silently drops rows whose instant sits exactly on the
    boundary (`'+00:00'` sorts before `.000Z`), so both sides are parsed. A
    naive timestamp is read as UTC — that is how the store writes them.
    """
    text = str(value or "").strip()
    if not text:
        return None
    # A '+' offset arrives as a space when the caller does not percent-encode
    # it (query-string rules). Restore only that exact position — the sign of
    # a trailing HH:MM offset — instead of guessing at the whole string.
    if len(text) == 25 and text[19] == " " and text[22] == ":":
        text = text[:19] + "+" + text[20:]
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def filter_executions(rows: List[Dict[str, Any]], status: Optional[str] = None,
                      query: Optional[str] = None,
                      since: Optional[str] = None,
                      until: Optional[str] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Apply UI filters to the real rows (no new query per filter).

    A date bound that cannot be parsed is an error (the caller reports it);
    a row whose own timestamp cannot be parsed is EXCLUDED from a bounded
    range, because its position in that range cannot be proven.
    """
    out = rows
    if status:
        out = [r for r in out if r.get("status") == status]
    if query:
        needle = str(query).strip().lower()
        if needle:
            out = [r for r in out if needle in str(r.get("question") or "").lower()]
    if since:
        low = _epoch(since)
        if low is None:
            raise ValueError(f"since={since!r} is not an ISO-8601 timestamp")
        out = [r for r in out
               if _epoch(r.get("created_at")) is not None
               and _epoch(r.get("created_at")) >= low]
    if until:
        high = _epoch(until)
        if high is None:
            raise ValueError(f"until={until!r} is not an ISO-8601 timestamp")
        out = [r for r in out
               if _epoch(r.get("created_at")) is not None
               and _epoch(r.get("created_at")) <= high]
    if limit is not None:
        out = out[:max(0, int(limit))]
    return out


def approval_agent(operation: object) -> Optional[str]:
    return APPROVAL_AGENTS.get(str(operation or ""))


# --------------------------------------------------------------------------- #
# Learning (2026-10-30)
# --------------------------------------------------------------------------- #

# Review actions the Control Center may apply. Each one is a status change on
# ONE stored lesson; none of them can edit a tool permission, an approval
# rule or a credential, and none can make a failed evaluation pass.
LESSON_ACTIONS = {
    "activate": "active",
    "disable": "disabled",
    "reject": "rejected",
    "candidate": "candidate",
}


def _lesson_view(lesson: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": lesson.get("lesson_id"),
        "status": lesson.get("status"),
        "category": lesson.get("category"),
        "capability": lesson.get("capability"),
        "scope": lesson.get("scope"),
        "problem": lesson.get("problem"),
        "root_cause": lesson.get("root_cause"),
        "corrective_behavior": lesson.get("corrective_behavior"),
        "validation": lesson.get("validation"),
        "confidence": lesson.get("confidence"),
        "origin": lesson.get("origin"),
        "evidence": lesson.get("evidence") or [],
        "source_execution_id": lesson.get("source_execution_id"),
        "source_text": lesson.get("source_text"),
        # Calibration (2026-10-30): why this lesson is as wide as it is.
        # `None` means the row predates the classification — the UI says
        # "not classified" instead of guessing.
        "feedback_type": lesson.get("feedback_type"),
        "impact": lesson.get("impact"),
        "temporary": bool(lesson.get("temporary")),
        "calibration": lesson.get("calibration") or {},
        "restricts": bool((lesson.get("calibration") or {}).get("restricts")),
        "hits": lesson.get("hits"),
        "successes": lesson.get("successes"),
        "failures": lesson.get("failures"),
        "created_at": lesson.get("created_at"),
        "updated_at": lesson.get("updated_at"),
        "activated_at": lesson.get("activated_at"),
        "disabled_at": lesson.get("disabled_at"),
        "superseded_by": lesson.get("superseded_by"),
    }


def _skill_view(version: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = version.get("evaluation") or {}
    return {
        "version_id": version.get("version_id"),
        "skill_id": version.get("skill_id"),
        "version": version.get("version"),
        "status": version.get("status"),
        "instructions": version.get("instructions"),
        "validation": version.get("validation"),
        "source_lesson_ids": version.get("source_lesson_ids") or [],
        "evaluation": evaluation,
        "passed": evaluation.get("passed"),
        "created_at": version.get("created_at"),
        "activated_at": version.get("activated_at"),
        "retired_at": version.get("retired_at"),
    }


def skill_version_view(version: Dict[str, Any]) -> Dict[str, Any]:
    """Public projection of one skill version row (used by the API)."""
    return _skill_view(version)


def learning_summary(store) -> Dict[str, Any]:
    """Everything the Learning page may show, from persisted records only.

    Counts come from the `lessons` table, skill versions from
    `skill_versions`, and applied-lesson evidence from the execution events the
    runner writes. Nothing here is inferred: an empty table is reported as
    "nothing learned yet", and a lesson's status is whatever is stored.
    """
    try:
        lessons = store.list_lessons()
    except Exception:  # noqa: BLE001 — old database without the table
        lessons = []
    try:
        versions = store.list_skill_versions()
    except Exception:  # noqa: BLE001
        versions = []

    counts: Dict[str, int] = {}
    for lesson in lessons:
        status = str(lesson.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1

    # Repeated-failure patterns: the same capability correcting the same way
    # more than once. This is the signal for "fix the behaviour", and it is
    # computed from stored lessons rather than guessed.
    patterns: Dict[str, Dict[str, Any]] = {}
    for lesson in lessons:
        if lesson.get("status") in ("rejected", "disabled", "superseded"):
            continue
        key = f"{lesson.get('capability')}::{lesson.get('category')}"
        entry = patterns.setdefault(key, {
            "capability": lesson.get("capability"),
            "category": lesson.get("category"),
            "lessons": 0,
            "occurrences": 0,
            "example": lesson.get("corrective_behavior"),
        })
        entry["lessons"] += 1
        entry["occurrences"] += max(1, int(lesson.get("hits") or 0))
    repeated = sorted(
        [p for p in patterns.values() if p["lessons"] > 1],
        key=lambda p: (-p["lessons"], str(p["capability"])))

    return {
        "checked_at": _now(),
        "counts": counts,
        "total": len(lessons),
        "active": [_lesson_view(l) for l in lessons
                   if l.get("status") == "active"],
        "candidates": [_lesson_view(l) for l in lessons
                       if l.get("status") == "candidate"],
        "rejected": [_lesson_view(l) for l in lessons
                     if l.get("status") == "rejected"],
        "disabled": [_lesson_view(l) for l in lessons
                     if l.get("status") == "disabled"],
        "superseded": [_lesson_view(l) for l in lessons
                       if l.get("status") == "superseded"],
        "repeated_patterns": repeated,
        "skills": [_skill_view(v) for v in versions],
        "awaiting_approval": [_lesson_view(l) for l in lessons
                              if l.get("status") == "candidate"],
        "note": ("A lesson is applied only when it is ACTIVE and its capability "
                 "matches the task. Candidate lessons wait for review. "
                 "Recording a lesson does not prove behaviour changed — the "
                 "execution timeline shows which lessons were actually "
                 "inserted into a prompt."),
    }


def review_lesson(store, lesson_id: str, action: str,
                  corrective_behavior: Optional[str] = None,
                  reason: Optional[str] = None) -> Dict[str, Any]:
    """Apply a review decision to one stored lesson (audited).

    `correct` edits the behaviour text AND appends a review entry to the
    evidence list, so the original correction stays inspectable. Every action
    is a status/text change on a lesson row — never a permission, a rule or a
    configuration value.
    """
    from app.execution.store import utc_now_iso

    lesson = store.get_lesson(lesson_id)
    if lesson is None:
        raise KeyError(lesson_id)
    entry = {"kind": "review", "action": action, "at": utc_now_iso(),
             "reason": (reason or "").strip()[:300] or None}
    fields: Dict[str, Any] = {}
    if action == "correct":
        text = (corrective_behavior or "").strip()
        if not text:
            raise ValueError("correct requires corrective_behavior")
        fields["corrective_behavior"] = text[:400]
        fields["status"] = "active" if lesson.get("status") != "active" else \
            lesson.get("status")
    elif action in LESSON_ACTIONS:
        new_status = LESSON_ACTIONS[action]
        fields["status"] = new_status
        if new_status == "active":
            fields["activated_at"] = utc_now_iso()
            fields["disabled_at"] = None
        elif new_status in ("disabled", "rejected"):
            fields["disabled_at"] = utc_now_iso()
    else:
        raise ValueError(
            f"unknown action {action!r}; expected one of "
            f"{sorted(set(LESSON_ACTIONS) | {'correct'})}")
    fields["evidence"] = list(lesson.get("evidence") or []) + [entry]
    updated = store.update_lesson(lesson_id, **fields)
    return _lesson_view(updated)


# --- Learning impact evaluation (2026-10-30) ------------------------------- #

MAX_EVAL_HISTORY = 10


def _eval_dir():
    from app.config import STORAGE_DIR

    return STORAGE_DIR / "learning_eval"


def _eval_summary(report: Dict[str, Any], path: Path,
                  modified: float) -> Dict[str, Any]:
    """The small row the history table shows (never re-derived numbers)."""
    summary = report.get("summary") or {}
    base = summary.get("baseline") or {}
    learn = summary.get("learning") or {}
    delta = summary.get("delta") or {}
    return {
        "file": path.name,
        "run_at": report.get("finished_at") or report.get("started_at"),
        "modified_at": datetime.fromtimestamp(
            modified, timezone.utc).isoformat(timespec="seconds"),
        "benchmark_version": report.get("benchmark_version"),
        "mode": report.get("mode"),
        "live_model": bool(report.get("live_model")),
        "task_count": report.get("task_count"),
        "completeness_baseline": base.get("completeness"),
        "completeness_learning": learn.get("completeness"),
        "delta_pp": delta.get("completeness_pp"),
        "qa_findings_baseline": base.get("qa_findings"),
        "qa_findings_learning": learn.get("qa_findings"),
        "improved_tasks": delta.get("improved_tasks"),
        "regressed_tasks": delta.get("regressed_tasks"),
        "regressions": delta.get("regressions"),
    }


def _eval_latest(report: Dict[str, Any]) -> Dict[str, Any]:
    """The report, trimmed to what the page renders (all from the file)."""
    tasks = []
    for task in report.get("tasks") or []:
        comparison = task.get("comparison") or {}
        tasks.append({
            "task_id": task.get("task_id"),
            "capability": task.get("capability"),
            "request": task.get("request"),
            "baseline_execution_id": task.get("baseline_execution_id"),
            "learning_execution_id": task.get("learning_execution_id"),
            "lesson_fixtures": task.get("lesson_fixtures") or [],
            "baseline_completeness": comparison.get("completeness_baseline"),
            "learning_completeness": comparison.get("completeness_learning"),
            "delta_pp": comparison.get("completeness_delta_pp"),
            "outcome": comparison.get("outcome"),
            "applied_lessons": comparison.get("applied_lessons") or [],
            "restricted_lessons": comparison.get("restricted_lessons") or [],
            "retrieved_not_applied": comparison.get("retrieved_not_applied") or [],
            "regressions": comparison.get("regressions") or [],
        })
    # The classification audit is a property of the GATE, not of a task: it is
    # passed through as recorded (count, accuracy, mismatches), never merged
    # into the completeness number.
    audit = report.get("calibration") or {}
    calibration = None
    if isinstance(audit, dict) and audit:
        calibration = {
            "count": audit.get("count"),
            "accuracy": audit.get("accuracy") or {},
            "mismatches": audit.get("mismatches") or [],
        }
    return {
        "benchmark_version": report.get("benchmark_version"),
        "harness_version": report.get("harness_version"),
        "mode": report.get("mode"),
        "live_model": bool(report.get("live_model")),
        "provider": report.get("provider") or {},
        "started_at": report.get("started_at"),
        "finished_at": report.get("finished_at"),
        "task_count": report.get("task_count"),
        "summary": report.get("summary") or {},
        "calibration": calibration,
        "tasks": tasks,
        "notes": report.get("notes") or [],
    }


def learning_evaluation() -> Dict[str, Any]:
    """Latest benchmark run + history, read from the report files ONLY.

    The harness writes `storage/learning_eval/*.json`; nothing is recomputed
    or invented here, so a stale or hand-edited file shows exactly what it
    says. A malformed file is skipped and named, never silently replaced.
    """
    from app.config import PROJECT_ROOT

    directory = _eval_dir()
    payload: Dict[str, Any] = {
        "checked_at": _now(),
        "available": False,
        "latest": None,
        "history": [],
        "evaluation_dir": _rel(directory, PROJECT_ROOT),
        "skipped": [],
        "note": ("Evaluation is observational: a benchmark run executes each "
                 "task twice (baseline without a lesson set, learning-enabled "
                 "with the applicable active lessons) and records what "
                 "happened. It never activates, rejects or edits a lesson."),
        "run_command": ("env\\Scripts\\python.exe scripts\\eval_learning.py "
                        "--print"),
    }
    if not directory.is_dir():
        payload["reason"] = ("No benchmark has been run yet: the report "
                             "directory does not exist.")
        return payload
    try:
        files = sorted(directory.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError as exc:
        payload["reason"] = f"The report directory could not be read: {exc.__class__.__name__}"
        return payload

    history: List[Dict[str, Any]] = []
    latest: Optional[Dict[str, Any]] = None
    for path in files[:MAX_EVAL_HISTORY]:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict) or "summary" not in report:
                raise ValueError("not a benchmark report")
            history.append(_eval_summary(report, path, path.stat().st_mtime))
            if latest is None:
                latest = _eval_latest(report)
        except (OSError, ValueError, TypeError) as exc:
            payload["skipped"].append({"file": path.name,
                                       "reason": exc.__class__.__name__})
    payload["available"] = latest is not None
    payload["latest"] = latest
    payload["history"] = history
    if latest is None:
        payload["reason"] = ("Report files exist but none is a readable "
                             "benchmark report.")
    return payload
