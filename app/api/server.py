"""Control Center API (GUI-2, local-first preview; CONTROL_CENTER_PLAN §26).

Endpoints (v1):
  GET  /api/v1/health                     — backend + AI readiness block
  POST /api/v1/executions                 — create + run synchronously
  GET  /api/v1/executions                 — history (newest first)
  GET  /api/v1/executions/{id}            — detail incl. error/result
  GET  /api/v1/executions/{id}/events     — events after a cursor (`after=`)

Boundaries:
  * localhost bind by default (CORS restricted to the Vite dev origin)
  * never returns secrets; AI config lives behind GUI-16 settings endpoints
  * backend failures map to honest structured errors, never fake success
"""

import sqlite3
import sys
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import ai_settings
from app.execution.runner import GraphRunner
from app.execution.store import (
    ApprovalConflict,
    ApprovalDecisionRequired,
    ApprovalExpired,
    ApprovalNotFound,
    ExecutionStore,
    serialize_approval,
    serialize_plan,
)
from app.execution.store import DEFAULT_DB_PATH

app = FastAPI(title="Agentica Control Center", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


_store: Optional[ExecutionStore] = None


def get_store() -> ExecutionStore:
    global _store
    if _store is None:
        _store = ExecutionStore(DEFAULT_DB_PATH)
        # Lifecycle invariant (2026-10-28): executions orphaned by a killed
        # process would otherwise stay `running`/`queued` forever and the UI
        # would show them as active. Runs once, before this process accepts
        # any request; fail-soft by convention — bookkeeping must never
        # prevent the store from being used.
        try:
            fixed = _store.reconcile_interrupted()
            if fixed:
                print(f"[server] {fixed} interrupted execution(s) "
                      "closed as failed on startup")
        except Exception as exc:  # noqa: BLE001
            print(f"[server] reconcile_interrupted failed "
                  f"({exc.__class__.__name__}) — startup continues")
    return _store


def get_runner() -> GraphRunner:
    return GraphRunner(get_store())


# Execution statuses the store can record (kept next to the API so the
# filter can reject a typo instead of silently returning nothing).
EXECUTION_STATUSES = (
    "queued", "running", "completed", "failed", "cancelled",
    "waiting_for_approval",
)


def _registry_agents() -> Dict[str, Dict[str, Any]]:
    from app.agents import registry

    return registry.AGENTS


def _with_requesting_agent(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add WHICH registered agent asks for each approval (additive field)."""
    from app.api.control_center import approval_agent

    enriched = []
    for item in items:
        out = dict(item)
        out["requesting_agent"] = approval_agent(item.get("operation"))
        enriched.append(out)
    return enriched


# --- AI readiness (GUI-16 partial) -----------------------------------------
# Implemented here so the dashboard gets the readiness block without pulling
# the full GUI-16 settings machinery; the settings endpoints may supersede.


def ai_ready() -> Dict[str, Any]:
    """Cheap local AI-readiness assessment from EFFECTIVE settings (no network).

    Uses GUI-16 precedence: session > saved file > env/config. A full bounded
    connectivity/model/generation check is POST /api/v1/ai/test.

    The underlying check lives in app.diagnostics_core.provider_readiness()
    (Increment 2): /health and the /services `ai_provider` rollup must not
    disagree about what "configured" means. It is config-only by contract —
    no network request, no model call. Response keys are unchanged.
    """
    from app import diagnostics_core

    readiness = diagnostics_core.provider_readiness()
    return {
        "provider": readiness["provider"],
        "state": readiness["state"],
        "what_will_be_tested": readiness["what_will_be_tested"],
        "state_hint": readiness["state_hint"],
    }


# --- health -----------------------------------------------------------------


@app.get("/api/v1/health")
def health() -> Dict[str, Any]:
    backend = "ok"
    ai = ai_ready()
    try:
        get_store().list_executions()
    except sqlite3.Error as exc:
        backend = "error: storage unavailable"
        return {"status": "degraded", "backend": backend, "ai": None,
                "storage_error": {"human_message": "Could not open the local database.",
                                  "why": str(exc)[:200],
                                  "recovery": "Check the storage/ folder permissions."}}
    return {"status": "ok" if ai["state"] != "not_configured" else "ok_ai_not_configured",
            "backend": backend,
            "ai": ai,
            "version": app.version}


# --- executions --------------------------------------------------------------


class CreateExecutionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


@app.post("/api/v1/executions", status_code=201)
def create_execution(req: CreateExecutionRequest) -> Dict[str, Any]:
    """Create and run synchronously (v1 preview behavior)."""
    runner = get_runner()
    created = runner.create(req.question)
    try:
        finished = runner.run(created["id"])
    except HTTPException:
        raise
    except Exception as exc:  # fail-soft honest error
        raise HTTPException(status_code=500, detail={
            "human_message": "The execution could not be started.",
            "why": str(exc)[:300],
            "recovery": "Retry; if it persists, report a bug.",
        })
    return finished


@app.get("/api/v1/executions")
def list_executions(
    status: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None, max_length=200),
    since: Optional[str] = Query(default=None, max_length=40),
    until: Optional[str] = Query(default=None, max_length=40),
    limit: Optional[int] = Query(default=None, ge=1, le=500),
) -> Dict[str, Any]:
    """History, newest first. Optional filters are additive and read-only;
    called with no parameters the response is unchanged."""
    execs = get_store().list_executions()
    total = len(execs)
    if status or q or since or until or limit is not None:
        if status and status not in EXECUTION_STATUSES:
            raise HTTPException(status_code=422, detail={
                "human_message": "Unknown execution status filter.",
                "why": f"{status!r} is not a status this store records.",
                "recovery": "Use one of: " + ", ".join(EXECUTION_STATUSES),
            })
        from app.api.control_center import filter_executions

        try:
            execs = filter_executions(execs, status=status, query=q,
                                      since=since, until=until, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={
                "human_message": "Invalid date filter.",
                "why": str(exc)[:200],
                "recovery": "Send an ISO-8601 timestamp, e.g. "
                            "2026-10-30T00:00:00Z.",
            })
    return {"executions": execs, "count": len(execs), "total": total}


@app.get("/api/v1/executions/{execution_id}")
def get_execution(execution_id: str) -> Dict[str, Any]:
    stored = get_store().get_execution(execution_id)
    if stored is None:
        raise HTTPException(status_code=404, detail={
            "human_message": "Execution not found.",
            "why": "No execution with that ID exists locally.",
            "recovery": "Check the execution list.",
        })
    response = dict(stored)
    # GUI-12: the detail response carries the serialized plan projection
    # (nullable, backward-compatible: legacy rows get plan=null).
    response["plan"] = serialize_plan(stored.get("plan"))
    return response


@app.get("/api/v1/executions/{execution_id}/evidence")
def get_execution_evidence(execution_id: str) -> Dict[str, Any]:
    """The evidence snapshot this execution actually consulted (2026-11-01).

    Read-only and additive: the passage text, retrieval/rerank scores, chunk
    identities, coverage verdict, claim-to-passage checks and the index
    identity AS USED. Returns evidence=null with a reason when the run
    consulted nothing (e.g. a research or conversational route) — a missing
    snapshot is not an error and is never fabricated.
    """
    stored = get_store().get_execution(execution_id)
    if stored is None:
        raise HTTPException(status_code=404, detail={
            "human_message": "Execution not found.",
            "why": "No execution with that ID exists locally.",
            "recovery": "Check the execution list.",
        })
    from app import evidence_store

    snapshot = evidence_store.read(execution_id)
    if snapshot is None:
        return {
            "execution_id": execution_id,
            "route": stored.get("route"),
            "evidence": None,
            "reason": "no evidence snapshot was recorded for this execution "
                      "(the route consulted no local passages, or the run "
                      "predates evidence snapshots)",
        }
    response = dict(snapshot)
    response["execution_id"] = execution_id
    return response


@app.get("/api/v1/executions/{execution_id}/events")
def get_events(execution_id: str, after: int = Query(default=0, ge=0)) -> Dict[str, Any]:
    if get_store().get_execution(execution_id) is None:
        raise HTTPException(status_code=404, detail={
            "human_message": "Execution not found.",
            "why": "No execution with that ID exists locally.",
            "recovery": "Check the execution list.",
        })
    events = get_store().get_events(execution_id, after=after)
    return {"execution_id": execution_id, "after": after,
            "events": events, "count": len(events)}


@app.get("/api/v1/executions/{execution_id}/plan")
def get_execution_plan(execution_id: str) -> Dict[str, Any]:
    """The REAL persisted plan for one execution (GUI-12, CONTROL_CENTER_PLAN
    §19). Returns plan=null for executions without a plan — simple routes,
    legacy rows, or corrupt blobs — never a fabricated sequence.
    """
    stored = get_store().get_execution(execution_id)
    if stored is None:
        raise HTTPException(status_code=404, detail={
            "human_message": "Execution not found.",
            "why": "No execution with that ID exists locally.",
            "recovery": "Check the execution list.",
        })
    return {
        "execution_id": execution_id,
        "status": stored["status"],
        "plan": serialize_plan(stored.get("plan")),
    }


# --- error guard for malformed requests --------------------------------------


@app.exception_handler(sqlite3.Error)
def sqlite_error_handler(request, exc):
    return HTTPException(status_code=500, detail={
        "human_message": "A storage problem occurred.",
        "why": "The local database returned an error.",
        "recovery": "Check the storage/ folder; retry after fixing permissions.",
        "details": str(exc)[:300],
    })


# --- AI settings (GUI-16 recovery) ------------------------------------------


class AiSettingsUpdate(BaseModel):
    llm_provider: Optional[Literal["gemini", "ollama"]] = None
    gemini_api_key: Optional[str] = Field(default=None, max_length=512)
    gemini_model: Optional[str] = Field(default=None, max_length=128)
    main_model: Optional[str] = Field(default=None, max_length=128)
    ollama_base_url: Optional[str] = Field(default=None, max_length=256)


class LessonReview(BaseModel):
    """A review decision on ONE learned lesson (no privileged field)."""

    action: Literal["activate", "disable", "reject", "candidate", "correct"]
    corrective_behavior: Optional[str] = Field(default=None, max_length=400)
    reason: Optional[str] = Field(default=None, max_length=300)


class AiTestRequest(BaseModel):
    llm_provider: Optional[Literal["gemini", "ollama"]] = None
    gemini_api_key: Optional[str] = Field(default=None, max_length=512)
    gemini_model: Optional[str] = Field(default=None, max_length=128)
    main_model: Optional[str] = Field(default=None, max_length=128)
    ollama_base_url: Optional[str] = Field(default=None, max_length=256)
    with_generation: bool = True


@app.get("/api/v1/ai/settings")
def ai_settings_get() -> Dict[str, Any]:
    """Masked view; never returns a full API key."""
    return ai_settings.masked_settings()


@app.put("/api/v1/ai/settings")
def ai_settings_set(req: AiSettingsUpdate) -> Dict[str, Any]:
    """Persist settings for this server (explicitly requested by user)."""
    payload = req.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=422, detail={
            "human_message": "Nothing to save.",
            "why": "The request contained no recognized settings.",
            "recovery": "Fill in the fields you want to persist and save again.",
        })
    provider = payload.get("llm_provider")
    if provider == "gemini" and payload.get("ollama_base_url") is None:
        pass  # provider switch may also carry provider-specific fields only
    ai_settings.persist(payload)
    _invalidate_llm_cache()
    return ai_settings.masked_settings()


@app.delete("/api/v1/ai/settings")
def ai_settings_clear() -> Dict[str, Any]:
    """Remove persisted settings (back to env/config defaults)."""
    ai_settings.clear_persisted()
    ai_settings.set_session_overrides({})  # clear session too (explicit reset)
    _invalidate_llm_cache()
    return {"cleared": True, "settings": ai_settings.masked_settings()}


@app.post("/api/v1/ai/test")
def ai_test(req: AiTestRequest) -> Dict[str, Any]:
    """Bounded provider connectivity/readiness test.

    Overridden fields are TESTED but NOT saved — only PUT persists config.
    """
    eff = ai_settings.effective_settings()
    body = req.model_dump(exclude_none=True)
    provider = body.get("llm_provider") or eff.get("llm_provider") or "gemini"

    # Session overrides for what the test actually exercises next time.
    session_patch = {k: v for k, v in body.items()
                     if k not in ("with_generation",) and v}

    from app import ai_diagnostics as diag

    if provider == "ollama":
        base = body.get("ollama_base_url") or eff.get("ollama_base_url")
        model = body.get("main_model") or eff.get("main_model")
        result = diag.check_ollama(base, model,
                                   with_generation=body.get("with_generation", True))
        if result.get("state") == "ready" and session_patch:
            ai_settings.set_session_overrides(session_patch)
            _invalidate_llm_cache()
        return result

    if provider == "gemini":
        key = body.get("gemini_api_key") or eff.get("gemini_api_key")
        model = body.get("gemini_model") or eff.get("gemini_model")
        result = diag.check_gemini(key, model)
        if result.get("state") in ("ready", "partial") and session_patch:
            ai_settings.set_session_overrides(session_patch)
            _invalidate_llm_cache()
        return result

    raise HTTPException(status_code=400, detail={
        "human_message": "Unknown AI provider.",
        "why": f"Provider {provider!r} is not supported.",
        "recovery": "Choose Gemini or Ollama.",
    })


def _invalidate_llm_cache() -> None:
    """Session-level cache exists in some future paths; no-op for now but
    documented as the invalidation hook (GUI-16 requirement)."""
    for mod_name in list(sys.modules):
        if mod_name.startswith("app."):
            mod = sys.modules[mod_name]
            for attr in ("_llm_cache", "_cached_llm"):
                if hasattr(mod, attr):
                    try:
                        delattr(mod, attr)
                    except Exception:
                        pass


# --- health_patch: use effective AI settings for the readiness block ---------


@app.patch("/api/v1/ai/settings/session")
def ai_settings_session(req: AiSettingsUpdate) -> Dict[str, Any]:
    """Set THIS-session-only overrides (never persisted)."""
    payload = {k: v for k, v in req.model_dump(exclude_none=True).items()}
    ai_settings.set_session_overrides(payload)
    _invalidate_llm_cache()
    return ai_settings.masked_settings()


# --- Operations dashboard endpoints (GUI-7 extension) -----------------------


@app.get("/api/v1/agents")
def agents_catalog() -> Dict[str, Any]:
    """Registered specialists (registry-backed) + config-only availability.

    Read-only and secret-free: no key, token or absolute path is returned,
    and nothing here probes a service or calls a model.
    """
    from app.api import control_center

    return control_center.agent_catalog(get_store())


@app.get("/api/v1/skills")
def skills_catalog() -> Dict[str, Any]:
    """Documented procedures on disk, plus learned versions of them."""
    from app.api import control_center

    return control_center.skills_catalog(get_store())


# --- Continuous learning (2026-10-30) --------------------------------------


@app.get("/api/v1/learning")
def learning_summary() -> Dict[str, Any]:
    """Persisted lessons, repeated-failure patterns and skill versions.

    Read-only projection of the `lessons` / `skill_versions` tables. An empty
    result means nothing has been learned yet — it is never filled in.
    """
    from app.api import control_center

    return control_center.learning_summary(get_store())


@app.get("/api/v1/learning/evaluation")
def learning_evaluation() -> Dict[str, Any]:
    """Latest A/B benchmark report + recent history (read-only files).

    The harness writes reports under `storage/learning_eval/`; this endpoint
    only reads them, so the numbers are exactly what the run produced. An
    empty result means no benchmark has been run — it is never filled in.
    """
    from app.api import control_center

    return control_center.learning_evaluation()


@app.post("/api/v1/learning/lessons/{lesson_id}/review")
def review_lesson(lesson_id: str, req: LessonReview) -> Dict[str, Any]:
    """Activate, disable, reject or correct ONE stored lesson.

    This is the only way a lesson's status changes after it is recorded. It
    cannot grant permissions or alter an approval rule, and it does not touch
    the model/provider configuration.
    """
    from app.api import control_center

    try:
        lesson = control_center.review_lesson(
            get_store(), lesson_id, req.action,
            corrective_behavior=req.corrective_behavior,
            reason=req.reason,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown lesson")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"lesson": lesson}


@app.post("/api/v1/learning/skills/{skill_id}/rollback")
def rollback_skill(skill_id: str) -> Dict[str, Any]:
    """Restore the previous version of a learned skill (kept, not overwritten)."""
    from app.learning import skills as learning_skills

    outcome = learning_skills.rollback(get_store(), skill_id)
    if not outcome.get("rolled_back"):
        return {"rolled_back": False, "reason": outcome.get("reason")}
    from app.api import control_center

    return {"rolled_back": True,
            "version": control_center.skill_version_view(outcome["version"])}


@app.get("/api/v1/memory")
def memory_summary() -> Dict[str, Any]:
    """What actually persists — and the gaps, named explicitly."""
    from app.api import control_center

    return control_center.memory_summary(get_store())


@app.get("/api/v1/dashboard")
def dashboard_summary() -> Dict[str, Any]:
    """Operational counts, the attention queue and a real activity timeline."""
    from app.api import control_center

    return control_center.dashboard_summary(get_store())


@app.get("/api/v1/services")
def services_health() -> Dict[str, Any]:
    """Real bounded probes for each configured dependency.

    Availability only; the functional generation test stays behind
    POST /api/v1/ai/test (never run on a polling interval).
    """
    try:
        from app import services_health
        # W2.1: hand the LIVE store over so the storage row can report
        # failed event writes; never let that plumbing break the probe.
        try:
            store = get_store()
        except Exception:  # noqa: BLE001 — storage itself is probed below
            store = None
        return services_health.services_snapshot(store)
    except Exception as exc:
        return {
            "checked_at": None,
            "services": [],
            "error": {
                "human_message": "Service checks could not run.",
                "why": type(exc).__name__,
                "recovery": "Retry; if it persists, report a bug.",
            },
        }


@app.get("/api/v1/agents/activity")
def agents_activity() -> Dict[str, Any]:
    """Agent/node activity derived ONLY from real execution events.

    A node is 'working' when its started event is the latest lifecycle
    record without a completion; never inferred from graph topology. The
    inventory comes from the registered agents (2026-10-30) so a specialist
    cannot exist in the graph but be invisible here.
    """
    inventory: List[Dict[str, str]] = [
        {"node": node, "role": str(entry.get("route_reason") or "")}
        for node, entry in _registry_agents().items()
        if entry.get("node")
    ] + [
        {"node": "supervisor", "role": "Decides which capability handles the request"},
        {"node": "qa", "role": "Validates the completed workflow output"},
    ]
    execs = get_store().list_executions()
    agents: Dict[str, Dict[str, Any]] = {a["node"]: dict(a) for a in inventory}
    for node in agents:
        agents[node].update({"status": "idle", "execution_id": None,
                             "task": None, "last_event_at": None,
                             "last_event": None, "last_error": None})

    for exec_row in execs:
        eid = exec_row["id"]
        events = get_store().get_events(eid)
        if exec_row["status"] in ("running", "queued"):
            # active run: show last lifecycle record per agent (working/failed)
            last_by_agent: Dict[str, Dict[str, Any]] = {}
            for ev in events:
                a = ev.get("agent_id")
                if not a:
                    continue
                prev = last_by_agent.get(a)
                if prev is None or ev["event_id"] > prev["event_id"]:
                    last_by_agent[a] = ev
            for a, ev in last_by_agent.items():
                if a not in agents:
                    continue
                agents[a]["status"] = (
                    "failed" if ev["type"] == "agent.failed"
                    else "completed" if ev["type"] == "agent.completed"
                    else "working" if ev["type"] == "agent.started" else "idle"
                )
                agents[a]["execution_id"] = eid
                agents[a]["task"] = exec_row["question"]
                agents[a]["last_event_at"] = ev["timestamp"]
                agents[a]["last_event"] = ev["message"]
                if ev["type"] == "agent.failed" and ev.get("payload"):
                    agents[a]["last_error"] = ev["payload"].get("category")
        elif exec_row["status"] == "failed":
            # failed run: failed agents stay visible, but NOT as "working"
            for ev in events:
                a = ev.get("agent_id")
                if a == ev.get("agent_id") and ev["type"] == "agent.failed" \
                        and a in agents:
                    agents[a]["status"] = "failed"
                    agents[a]["execution_id"] = eid
                    agents[a]["task"] = exec_row["question"]
                    agents[a]["last_event_at"] = ev["timestamp"]
                    agents[a]["last_event"] = ev["message"]
                    agents[a]["last_error"] = (ev.get("payload") or {}).get(
                        "category")

    active = [a for a in agents.values()
              if a["status"] in ("working", "failed")]
    return {
        "agents": list(agents.values()),
        "active_count": len([a for a in active if a["status"] == "working"]),
        "failed_count": len([a for a in active if a["status"] == "failed"]),
        "note": "Activity reflects recorded execution events only.",
    }


@app.get("/api/v1/agents/{agent_id}")
def agent_detail(agent_id: str) -> Dict[str, Any]:
    """One registered agent, from the same projection as the list.

    Declared AFTER `/agents/activity` on purpose: FastAPI matches routes in
    declaration order, so a dynamic segment first would swallow the static
    one (it did — the activity endpoint answered 404 "agent 'activity'").
    """
    from app.api import control_center

    catalog = control_center.agent_catalog(get_store())
    for agent in catalog["agents"]:
        if agent["id"] == agent_id:
            return {"agent": agent,
                    "provider": catalog["provider"],
                    "checked_at": catalog["checked_at"]}
    raise HTTPException(status_code=404, detail={
        "human_message": "Agent not found.",
        "why": f"No registered agent has the id {agent_id[:40]!r}.",
        "recovery": "Open the Agents page for the registered list.",
    })


@app.get("/api/v1/executions/{execution_id}/sequence")
def execution_sequence(execution_id: str) -> Dict[str, Any]:
    """The REAL execution sequence for one run, derived from its events.

    This system executes graph routes sequentially (no separate queue
    service); the response presents the observed order plus the QA gate and
    never invents dependency edges or queue entries.
    """
    stored = get_store().get_execution(execution_id)
    if stored is None:
        raise HTTPException(status_code=404, detail={
            "human_message": "Execution not found.",
            "why": "No execution with that ID exists locally.",
            "recovery": "Check the execution list.",
        })
    events = get_store().get_events(execution_id)
    order: List[Dict[str, Any]] = []
    for ev in events:
        agent = ev.get("agent_id")
        t = ev["type"]
        # routing.decided / qa.completed carry agent ids but are stages of
        # their own — handle them before the generic agent-event branches.
        if t == "routing.decided":
            order.append({"stage": "routed", "node": agent or "supervisor",
                          "status": "completed", "at": ev["timestamp"],
                          "event_id": ev["event_id"], "message": ev["message"],
                          "route": (ev.get("payload") or {}).get("route")})
            continue
        if t == "qa.completed":
            order.append({"stage": "qa", "node": agent or "qa",
                          "status": "completed", "at": ev["timestamp"],
                          "event_id": ev["event_id"], "message": ev["message"],
                          "qa_flags": (ev.get("payload") or {}).get("qa_flags", [])})
            continue
        if not agent:
            if t == "execution.created":
                order.append({"stage": "created", "node": None,
                              "status": "completed", "at": ev["timestamp"],
                              "event_id": ev["event_id"],
                              "message": ev["message"]})
            elif t == "routing.decided":
                order.append({"stage": "routed", "node": "supervisor",
                              "status": "completed", "at": ev["timestamp"],
                              "event_id": ev["event_id"], "message": ev["message"],
                              "route": (ev.get("payload") or {}).get("route")})
            elif t == "qa.completed":
                order.append({"stage": "qa", "node": "qa",
                              "status": "completed", "at": ev["timestamp"],
                              "event_id": ev["event_id"], "message": ev["message"],
                              "qa_flags": (ev.get("payload") or {}).get("qa_flags", [])})
            elif t == "execution.completed":
                order.append({"stage": "completed", "node": None,
                              "status": "completed", "at": ev["timestamp"],
                              "event_id": ev["event_id"], "message": ev["message"]})
            elif t == "execution.failed":
                payload = ev.get("payload") or {}
                order.append({"stage": "failed", "node": None,
                              "status": "failed", "at": ev["timestamp"],
                              "event_id": ev["event_id"], "message": ev["message"],
                              "error": payload,
                              "next_action": payload.get("recovery") or
                                             "Retry the execution; see Settings "
                                             "→ AI connection if the problem "
                                             "looks provider-related."})
            continue
        if t == "agent.started":
            order.append({"stage": "agent", "node": agent, "status": "running",
                          "at": ev["timestamp"], "event_id": ev["event_id"],
                          "message": ev["message"]})
        elif t == "agent.completed":
            # update the matching open entry to completed
            for entry in reversed(order):
                if (entry["stage"] == "agent" and entry.get("node") == agent
                        and entry["status"] == "running"):
                    entry["status"] = "completed"
                    entry["completed_at"] = ev["timestamp"]
                    break
        elif t == "agent.failed":
            for entry in reversed(order):
                if (entry["stage"] == "agent" and entry.get("node") == agent
                        and entry["status"] == "running"):
                    entry["status"] = "failed"
                    entry["completed_at"] = ev["timestamp"]
                    entry["failure_category"] = (ev.get("payload") or {}).get(
                        "category")
                    entry["next_action"] = "Retry the execution; see Settings " \
                        "→ AI connection if the problem looks provider-related."
                    break
    return {
        "execution_id": execution_id,
        "status": stored["status"],
        "route": stored.get("route"),
        "question": stored["question"],
        "stages": order,
        "model": "sequential graph execution (no separate queue in the "
                 "current architecture)",
    }


def build_server() -> FastAPI:
    """For tests that import the app without starting uvicorn."""
    return app


# --- approvals (GUI-10, CONTROL_CENTER_PLAN §12, 2026-10-28) ---------------
#
# AUTHENTICATION LIMITATION (documented deliberately, not fixed here):
# like the rest of this local-first API these endpoints carry NO
# authentication or per-user authorization — they rely on localhost binding;
# approval mutations additionally reject untrusted browser Origin headers
# (CORS alone cannot prevent cross-site form POSTs). They must never be
# exposed beyond this machine. The Approval Center UI, its SSE decision
# events and the approval modes of §13 are still PENDING frontend work; what
# is implemented is the backend lifecycle: persist -> decide -> resume the
# ONE bound operation.


_APPROVAL_BROWSER_ORIGINS = frozenset({
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3333",
    "http://127.0.0.1:3333",
})


def _require_local_approval_origin(request: Request) -> None:
    """Reject browser cross-site POSTs to approval decision/execution routes.

    The local API has no authentication, so CORS alone is not a CSRF defense:
    a hostile page can submit a simple form POST and the browser will send it
    even if the response is unreadable. Browser-originated mutations must come
    from the local Control Center origins. Origin-less local CLI/test clients
    remain compatible; this is not a substitute for authentication if the API
    is ever exposed beyond localhost.
    """
    origin = request.headers.get("origin")
    if origin is not None and origin not in _APPROVAL_BROWSER_ORIGINS:
        raise HTTPException(status_code=403, detail={
            "human_message": "Approval actions are only accepted from the local Control Center.",
            "why": "The browser request Origin is not an approved local UI origin.",
            "recovery": "Open Agentica from the local Control Center and retry.",
        })


def _approval_404(approval_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail={
        "human_message": "Approval request not found.",
        "why": f"No approval with id {approval_id[:32]!r} exists locally.",
        "recovery": "List pending approvals at GET /api/v1/approvals.",
    })


def _decide_approval(approval_id: str, outcome: str) -> Dict[str, Any]:
    """Shared approve/reject path with the project's error conventions."""
    try:
        return get_store().decide_approval(approval_id, outcome)
    except ApprovalNotFound:
        raise _approval_404(approval_id)
    except ApprovalExpired:
        raise HTTPException(status_code=410, detail={
            "human_message": "This approval request has expired.",
            "why": "The decision window passed before a decision was made.",
            "recovery": "Nothing was executed. Ask the research question "
                        "again to create a fresh request.",
        })
    except ApprovalConflict:
        raise HTTPException(status_code=409, detail={
            "human_message": "This approval request was already decided.",
            "why": "A different decision is recorded; it is never "
                   "overwritten.",
            "recovery": "Create a new research request if the decision "
                        "needs to change.",
        })


@app.get("/api/v1/approvals")
def approvals_list(
    status: str = Query(default="pending"),
    execution_id: Optional[str] = Query(default=None, max_length=64),
) -> Dict[str, Any]:
    """List approval requests (default: pending queue for GUI-10).

    `execution_id` narrows the list to one run (used by the execution detail
    page). `status=all` returns every decision state — a read-only union, no
    new storage and no change to the approve/reject/resume paths.
    """
    store = get_store()
    try:
        if status == "all":
            items = []
            from app.execution.store import APPROVAL_STATUSES

            for each in APPROVAL_STATUSES:
                items.extend(store.list_approvals(each))
        else:
            items = store.list_approvals(status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "human_message": "Unknown approval status filter.",
            "why": str(exc)[:200],
            "recovery": "Use one of: pending, approved, rejected, expired, "
                        "or all.",
        })
    if execution_id:
        items = [i for i in items if i.get("execution_id") == execution_id]
    items = _with_requesting_agent(items)
    return {"approvals": items, "count": len(items), "status": status}


@app.get("/api/v1/approvals/{approval_id}")
def approval_detail(approval_id: str) -> Dict[str, Any]:
    row = get_store().get_approval(approval_id)
    if row is None:
        raise _approval_404(approval_id)
    return {"approval": _with_requesting_agent(
        [serialize_approval(row)])[0]}


@app.post("/api/v1/approvals/{approval_id}/approve")
def approval_approve(approval_id: str, request: Request) -> Dict[str, Any]:
    """Persist approval for EXACTLY the operation bound to this request.
    Idempotent for a repeat of the same decision; a conflicting decision on
    an already-decided request is rejected with 409 and never overwrites."""
    _require_local_approval_origin(request)
    approval = _decide_approval(approval_id, "approved")
    return {
        "approval": approval,
        "decision": {
            "approval_id": approval_id,
            "outcome": "approved",
            "decided_at": approval.get("decided_at"),
            "already_decided": bool(approval.get("already_decided")),
        },
        "resume": f"/api/v1/approvals/{approval_id}/resume",
    }


@app.post("/api/v1/approvals/{approval_id}/reject")
def approval_reject(approval_id: str, request: Request) -> Dict[str, Any]:
    """Persist rejection; the bound external operation stays blocked."""
    _require_local_approval_origin(request)
    approval = _decide_approval(approval_id, "rejected")
    return {
        "approval": approval,
        "decision": {
            "approval_id": approval_id,
            "outcome": "rejected",
            "decided_at": approval.get("decided_at"),
            "already_decided": bool(approval.get("already_decided")),
        },
        "resume": f"/api/v1/approvals/{approval_id}/resume",
    }


@app.post("/api/v1/approvals/{approval_id}/resume")
def approval_resume(approval_id: str, request: Request) -> Dict[str, Any]:
    """Resolve a decided approval's bound operation.

    approved -> the research step runs through the real gate (persisted
    decision + bound payload + single-execution claim revalidated) and the
    execution is updated with the REAL answer; rejected -> the execution is
    closed with the honest local-only answer (idempotent); pending/expired
    fail closed with 409/410. The operation is always read back from the
    stored request — this endpoint accepts no payload.
    """
    _require_local_approval_origin(request)
    try:
        return get_runner().resume_approval(approval_id)
    except ApprovalDecisionRequired:
        raise HTTPException(status_code=409, detail={
            "human_message": "This request is still waiting for a decision.",
            "why": "Approve or reject it first; nothing executes while it "
                   "is pending.",
            "recovery": "POST .../approve or .../reject, then retry.",
        })
    except ApprovalExpired:
        raise HTTPException(status_code=410, detail={
            "human_message": "This approval request has expired.",
            "why": "The decision or execution window passed.",
            "recovery": "Nothing was executed. Ask the research question "
                        "again to create a fresh request.",
        })
    except ApprovalConflict:
        raise HTTPException(status_code=409, detail={
            "human_message": "This approval request cannot be executed.",
            "why": "It is not bound to a resumable execution.",
            "recovery": "Create a new research request.",
        })
    except LookupError:
        raise _approval_404(approval_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — honest structured failure
        raise HTTPException(status_code=500, detail={
            "human_message": "The approved operation could not be executed.",
            "why": str(exc)[:300],
            "recovery": "Retry the resume; if it persists, report a bug. "
                        "When the approved operation already ran, the retry "
                        "continues the remaining work without repeating it; "
                        "when it never ran, the decision is still unused.",
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
