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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import ai_settings
from app.execution.runner import GraphRunner
from app.execution.store import ExecutionStore
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
    return _store


def get_runner() -> GraphRunner:
    return GraphRunner(get_store())


# --- AI readiness (GUI-16 partial) -----------------------------------------
# Implemented here so the dashboard gets the readiness block without pulling
# the full GUI-16 settings machinery; the settings endpoints may supersede.


def ai_ready() -> Dict[str, Any]:
    """Cheap local AI-readiness assessment from EFFECTIVE settings (no network).

    Uses GUI-16 precedence: session > saved file > env/config. A full bounded
    connectivity/model/generation check is POST /api/v1/ai/test.
    """
    eff = ai_settings.effective_settings()
    provider = eff.get("llm_provider") or "gemini"
    if provider == "gemini":
        configured = bool(eff.get("gemini_api_key"))
        what = "Gemini"
        model = eff.get("gemini_model")
    elif provider == "ollama":
        configured = bool(eff.get("ollama_base_url"))
        what = f"Ollama ({eff.get('ollama_base_url')})"
        model = eff.get("main_model")
    else:
        configured = False
        what = f"unknown provider {provider!r}"
        model = ""

    state = "ready" if configured else "not_configured"
    return {
        "provider": provider,
        "state": state,
        "what_will_be_tested": f"{what}, model {model}",
        "state_hint": (
            "Configured. Agentica will use it for AI-backed steps."
            if configured else
            "No API key or endpoint set. AI-backed steps will fail until you "
            "configure a provider in Settings."
        ),
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
def list_executions() -> Dict[str, Any]:
    execs = get_store().list_executions()
    return {"executions": execs, "count": len(execs)}


@app.get("/api/v1/executions/{execution_id}")
def get_execution(execution_id: str) -> Dict[str, Any]:
    stored = get_store().get_execution(execution_id)
    if stored is None:
        raise HTTPException(status_code=404, detail={
            "human_message": "Execution not found.",
            "why": "No execution with that ID exists locally.",
            "recovery": "Check the execution list.",
        })
    return stored


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


@app.get("/api/v1/services")
def services_health() -> Dict[str, Any]:
    """Real bounded probes for each configured dependency.

    Availability only; the functional generation test stays behind
    POST /api/v1/ai/test (never run on a polling interval).
    """
    try:
        from app import services_health
        return services_health.services_snapshot()
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
    record without a completion; never inferred from graph topology.
    """
    inventory: List[Dict[str, str]] = [
        {"node": "supervisor", "role": "Decides which capability handles the request"},
        {"node": "research", "role": "Web research (SearXNG + Crawl4AI + model synthesis)"},
        {"node": "books", "role": "Answers from the internal book library (RAG)"},
        {"node": "erp", "role": "Business-data lookups via the adapter"},
        {"node": "feasibility", "role": "Synthesizes the feasibility study (web + books)"},
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
