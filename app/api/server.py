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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
    """Cheap local AI-readiness assessment: configured? fail-soft True/False.

    Makes no network call: a full connectivity/model check is a GUI-16
    "test connection" action. In 'ready' state we still assume the provider
    works only if configured and usable at request time (honest limitation;
    failures surface as step errors via structured execution failure).
    """
    from app.config import LLM_PROVIDER

    if LLM_PROVIDER == "gemini":
        from app.config import GOOGLE_API_KEY, GEMINI_MODEL
        configured = bool(GOOGLE_API_KEY)
        what = "Gemini"
        model = GEMINI_MODEL
    elif LLM_PROVIDER == "ollama":
        from app.config import MAIN_MODEL, OLLAMA_BASE_URL
        configured = bool(OLLAMA_BASE_URL)
        what = f"Ollama ({OLLAMA_BASE_URL})"
        model = MAIN_MODEL
    else:
        configured = False
        what = f"unknown provider {LLM_PROVIDER!r}"
        model = ""

    state = "ready" if configured else "not_configured"
    return {
        "provider": LLM_PROVIDER,
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


def build_server() -> FastAPI:
    """For tests that import the app without starting uvicorn."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
