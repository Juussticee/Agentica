"""Execution runner (GUI-1): drive the real graph, record real events.

Principles (Sections 28/33 of the CONTROL_CENTER_PLAN):
  * The UI is a projection of the event log it produces — fake progress is
    forbidden and no progress inference from logs.
  * A AI-unavailable / graph failure MUST be recorded honestly
    (status=failed + structured ExecutionError), never as success.
  * The event store survives server restarts (SQLite).
"""

import json
import copy
import uuid
from typing import Any, Dict, List, Optional

from app.execution import instrumentation as instr
from app.execution.errors import (
    ExecutionError,
    ai_error_for_provider_unavailable,
    classify_ai_exception,
)
from app.execution.events import (
    AGENT_COMPLETED,
    AGENT_FAILED,
    AGENT_STARTED,
    APPROVAL_FINALIZED,
    EXECUTION_COMPLETED,
    EXECUTION_CREATED,
    EXECUTION_FAILED,
    EXECUTION_PLANNED,
    EXECUTION_WAITING_APPROVAL,
    EXECUTION_APPROVAL_COMPLETED,
    EXECUTION_STARTED,
    LESSON_RECORDED,
    LESSONS_APPLIED,
    QA_COMPLETED,
    ROUTING_DECIDED,
    SEV_ERROR,
    SEV_INFO,
    SEV_WARNING,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_STARTED,
)
from app.execution.store import (
    ApprovalConflict,
    ApprovalDecisionRequired,
    ApprovalExpired,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_WAITING_APPROVAL,
    ExecutionStore,
    serialize_approval,
    serialize_plan,
)

_HUMAN_STEP_NAMES = {
    "supervisor": "Understanding the request",
    "planner": "Planning the work",
    "research": "Researching the web",
    "books": "Checking internal knowledge",
    "erp": "Looking up business data",
    "feasibility": "Assessing feasibility",
    "qa": "Checking the result",
    # 2026-10-28 (Priority A routes): jargon-free labels for the two new
    # answer steps, so ROUTING_DECIDED and the step events read as English.
    "general": "General knowledge",
    "unsupported": "Capability guidance",
    # 2026-10-30: workspace coding (inspection or patch preparation).
    "coding": "Working in the workspace",
    # 2026-10-30: dataset analysis (deterministic computation).
    "data": "Analysing the dataset",
}

_AI_PROVIDER_CRASH_MARKERS = (
    "GOOGLE_API_KEY",   # raised by get_llm when gemini key missing
    "API key",          # generic provider auth message
    "LLM_PROVIDER",
    "langchain_google_genai",
    "ChatOllama",
    "google.genai.errors",
    "ConnectionError",
    "api_key",
)


def human_step_name(node: str) -> str:
    return _HUMAN_STEP_NAMES.get(node, node.replace("_", " ").title())


def execution_id_new() -> str:
    return uuid.uuid4().hex[:12]


def snapshot_evidence(state: Any) -> Any:
    """The evidence record this run should persist, from either route.

    `general` writes `state["evidence"]`, `books` writes
    `state["books_evidence"]` (same shape, different channel so QA's general-
    route claim checks keep one answer path). A record without passages is
    not worth a snapshot: there is nothing to audit. Returns None when
    neither route consulted a passage (2026-11-02).
    """
    if not isinstance(state, dict):
        return None
    for key in ("evidence", "books_evidence"):
        record = state.get(key)
        if isinstance(record, dict) and record.get("passages_detail"):
            return record
    return None


class GraphRunner:
    """Owns executions: create -> run (graph) -> persist events + result."""

    def __init__(self, store: ExecutionStore, *, lessons_enabled: bool = True):
        self.store = store
        # 2026-10-30 (learning evaluation): a BASELINE condition must run the
        # same tasks with no learned lessons without touching the stored rows.
        # False means "this runner was not given a lesson set" — it does not
        # disable, delete or rewrite anything in the store, and it is the only
        # difference between the two benchmark conditions.
        self.lessons_enabled = bool(lessons_enabled)

    # -- helpers ----------------------------------------------------------

    def _event(self, eid, type, message, *, severity=SEV_INFO, step_id=None,
               agent_id=None, payload=None):
        try:
            self.store.append_event(
                eid,
                # GUI-10: the STORE assigns the next per-execution id — a
                # resume or an approval decision appends to a timeline that
                # already has rows, so a process-local counter would collide.
                None,
                type,
                message,
                step_id=step_id,
                agent_id=agent_id,
                severity=severity,
                payload=payload or {},
            )
        except Exception as exc:  # noqa: BLE001 — W2: bookkeeping never
            # corrupts the execution itself. The run, its answer and its
            # status stay authoritative; the lost event is reported loudly
            # instead of silently (project fail-soft convention).
            print(f"[runner] event persistence failed for {type} "
                  f"({exc.__class__.__name__}) — execution continues")

    def _node_wrapper(self, eid, seen):
        # `seen` is PER RUN (W2.1): it used to live on the runner, so a
        # second execution on the same runner silently lost its own
        # "Plan ready" event to the first one's bookkeeping.
        def wrap(name, fn):
            def wrapped(state):
                # Phase 7: record the plan as a real event when present
                # (plan DATA only — not execution progress, per §5.2).
                plan = state.get("plan")
                if plan and name != "planner" and "planner" not in seen:
                    seen.add("planner")
                    self._event(
                        eid, AGENT_COMPLETED,
                        "Plan ready: "
                        + str(len(plan.get("tasks", []))) + " task(s) ("
                        + str(plan.get("source", "unknown")) + ")",
                        agent_id="planner", step_id="planner",
                        payload={"plan_source": plan.get("source")},
                    )
                self._event(
                    eid, AGENT_STARTED, f"{human_step_name(name)} started",
                    agent_id=name, step_id=name,
                )
                try:
                    update = fn(state)
                    if name == "supervisor":
                        route = (state.get("route") or
                                 (update or {}).get("route") or "research")
                        self._event(
                            eid, ROUTING_DECIDED,
                            f"Routed the request to {human_step_name(route)}",
                            agent_id=name, step_id=name,
                            payload={"route": route},
                        )
                except Exception as exc:
                    is_ai = any(m.lower() in str(exc).lower()
                                for m in _AI_PROVIDER_CRASH_MARKERS) or \
                            classify_ai_exception(exc) is not None
                    human = (f"{human_step_name(name)} could not run because the "
                             "AI provider is unavailable."
                             ) if is_ai else f"{human_step_name(name)} failed."
                    self._event(
                        eid, AGENT_FAILED, human,
                        severity=SEV_ERROR, agent_id=name, step_id=name,
                        payload={
                            "category": classify_ai_exception(exc) if is_ai
                            else "execution_error",
                            "details": str(exc)[:2000],
                        },
                    )
                    raise
                self._event(
                    eid, AGENT_COMPLETED, f"{human_step_name(name)} completed",
                    agent_id=name, step_id=name,
                )
                return update

            return wrapped

        return wrap

    # -- API --------------------------------------------------------------

    def create(self, question: str) -> Dict[str, Any]:
        eid = execution_id_new()
        self.store.create_execution(eid, question, STATUS_QUEUED)
        self._event(eid, EXECUTION_CREATED, "Execution created")
        return self.store.get_execution(eid)

    def _prior_execution(self, eid: str) -> Optional[Dict[str, Any]]:
        """The most recent COMPLETED execution (before this one) with an answer.

        Used ONLY by the follow-up route: a request like "what about the one
        you recommended?" has no subject of its own, so the assistant must
        read it against something the user can actually inspect. That record
        is this row — the same one the Executions page shows — not a hidden
        conversation buffer and not a model's recollection. Read-only and
        fail-soft: an unreadable store means "no prior result", which the
        follow-up node reports honestly instead of inventing one.
        """
        try:
            rows = self.store.list_executions()
        except Exception:
            return None
        for row in rows:  # newest first (store contract)
            if str(row.get("id")) == str(eid):
                continue
            if row.get("status") != STATUS_COMPLETED:
                continue
            answer = str(row.get("answer") or "").strip()
            if not answer:
                continue
            return {
                "id": row.get("id"),
                "question": row.get("question"),
                "answer": answer,
                "completed_at": row.get("completed_at"),
            }
        return None

    def _active_lessons(self) -> List[Dict[str, Any]]:
        """ACTIVE lessons for this run (fail-soft: [] when unreadable).

        Read once, before the graph starts, so every node sees the same set
        and no node needs a store handle of its own. Relevance filtering is
        per capability inside the nodes (app/learning/retrieval.py), which is
        what keeps an unrelated task from receiving irrelevant lessons.
        """
        if not self.lessons_enabled:
            return []
        try:
            return self.store.list_lessons(status="active")
        except Exception:
            return []

    def run(self, eid: str) -> Dict[str, Any]:
        """Synchronously run an execution. Returns the final stored row."""
        stored = self.store.get_execution(eid)
        if stored is None:
            raise KeyError(eid)

        self.store.update_execution(eid, status=STATUS_RUNNING, started_at=True)
        self._event(eid, EXECUTION_STARTED, "Execution started")

        # Import here: keep agent startup cost out of the API import path and
        # let tests patch the graph builder module if needed.
        from app.graph import approval_scope, build_graph

        try:
            graph = build_graph(node_wrapper=self._node_wrapper(eid, set()))
            # W2: operations inside a node (retrieval, search, crawl,
            # recovery, the fallback decision) record themselves into THIS
            # execution's event stream while they happen. install() is a
            # no-op outside a run, is isolated per execution context (two
            # overlapping runs can never see each other's events), and
            # restores the previous sink in finally. GUI-10: the approval
            # scope binds THIS execution's approval store around the run so
            # the research gate can create/reuse a pending request bound to
            # this execution only.
            prior = self._prior_execution(eid)
            lessons = self._active_lessons()
            with instr.install(
                lambda t, msg, **kw: self._event(eid, t, msg, **kw)
            ), approval_scope(self.store, eid):
                final_state = graph.invoke({
                    "question": stored["question"],
                    # 2026-10-30: the ONLY conversational context a run gets.
                    # It is the recorded result the user can see in the
                    # Control Center, or None — never a model's memory.
                    "prior_execution": prior,
                    # 2026-10-30: ACTIVE learned lessons, read once so a node
                    # applies exactly what this run was given. Selecting the
                    # RELEVANT subset happens in the nodes (per capability).
                    "lessons": lessons,
                })
        except Exception as exc:
            return self._fail(eid, exc)

        answer = final_state.get("answer", "")
        route = final_state.get("route", "")
        qa_flags = final_state.get("qa_flags", [])
        # Phase 8 QA already flags failed/blocked plan tasks, but the runner
        # used to persist every non-raising graph invocation as completed.
        # Preserve the answer and QA flags while making the terminal status
        # honest for API/UI consumers.
        from app.qa_node import has_incomplete_plan

        incomplete_plan = has_incomplete_plan(final_state)
        pending_approval = self._execution_has_pending_approval(eid)
        # GUI-10: a pending approval BOUND to this execution means the run is
        # paused for a user decision, NOT failed — even when the plan still
        # has unfinished tasks (they are unfinished precisely because the
        # pause happened mid-plan). The decision closes the run truthfully:
        # approve -> resume the research step, reject -> failed
        # (approval_rejected). An incomplete plan alone keeps its old
        # failed/incomplete_plan semantics.
        waiting_for_approval = pending_approval
        show_incomplete = incomplete_plan and not waiting_for_approval
        terminal_status = (
            STATUS_WAITING_APPROVAL if waiting_for_approval else
            STATUS_FAILED if show_incomplete else STATUS_COMPLETED
        )
        terminal_error = (
            _incomplete_plan_error(final_state.get("plan")).to_dict()
            if show_incomplete else None
        )
        # QA gate evidence (the graph's qa node just ran its update above).
        self._event(
            eid, QA_COMPLETED,
            ("QA check passed" if not qa_flags
             else f"QA noted {len(qa_flags)} issue(s)"),
            agent_id="qa", step_id="qa",
            payload={"qa_flags": list(qa_flags)},
        )
        # GUI-12: persist the REAL executed plan (planner data + Phase 8
        # dispatcher task statuses) so the Control Center can visualize what
        # the agent intended and what each task actually did. A failed run
        # has no final state to read; the plan is simply not persisted there.
        plan = final_state.get("plan")
        if isinstance(plan, dict) and isinstance(plan.get("tasks"), list):
            self.store.update_execution(eid, plan=plan)
            n_tasks = len(plan.get("tasks") or [])
            self._event(
                eid, EXECUTION_PLANNED,
                f"Execution plan recorded: {n_tasks} task(s)",
                agent_id="planner", step_id="planner",
                payload={"plan": serialize_plan(plan)},
            )
        self.store.update_execution(
            eid,
            status=terminal_status,
            route=route,
            answer=answer,
            qa_flags=qa_flags,
            error=terminal_error,
            result=answer,
            # W2: provenance used to be dropped before storage — the graph's
            # final `sources` (agent names + research URLs) is now part of
            # the execution record, so a result can explain where it came
            # from. Never contains credentials; URLs are already shown to
            # the user in the answer itself.
            sources=list(final_state.get("sources") or []),
            completed_at=True,
        )
        # 2026-11-01: preserve the evidence THIS run consulted, as used.
        # A filesystem sidecar keyed by execution id (app/evidence_store) —
        # no schema migration, immutable, and never required for the run to
        # finish. Re-ingesting a document later changes the index manifest,
        # not this snapshot, so a historical answer stays auditable.
        # Both routes record the passages they consulted: `evidence` for the
        # general route, `books_evidence` for the books route (2026-11-02).
        # One writer, one format — the snapshot is keyed by execution id and
        # its `route` field says which route produced it.
        evidence_state = snapshot_evidence(final_state)
        if isinstance(evidence_state, dict):
            try:
                from app import evidence_store

                evidence_store.write(eid, evidence_store.build(
                    eid,
                    question=str(stored.get("question") or ""),
                    route=route,
                    answer=answer,
                    sources=final_state.get("sources") or [],
                    qa_flags=qa_flags,
                    evidence=evidence_state,
                ))
            except Exception:  # noqa: BLE001 — the audit trail is best effort
                pass

        # 2026-10-30: count what was actually applied. `success` is the run's
        # own honest verdict (completed AND no QA flags), so a lesson that was
        # in the prompt of a failed answer is recorded as a failure.
        self._record_lesson_use(
            eid, final_state,
            success=(terminal_status == STATUS_COMPLETED and not qa_flags),
        )
        self._observe_failure(
            eid, question=str(stored.get("question") or ""),
            route=route, error=terminal_error, qa_flags=qa_flags,
        )
        if waiting_for_approval:
            event_type = EXECUTION_WAITING_APPROVAL
            event_message = "Execution paused while waiting for approval"
            event_severity = SEV_WARNING
        else:
            event_type = EXECUTION_FAILED if show_incomplete else EXECUTION_COMPLETED
            event_message = ("Execution finished with incomplete planned tasks"
                             if show_incomplete else "Execution completed")
            event_severity = SEV_ERROR if show_incomplete else SEV_INFO
        self._event(
            eid,
            event_type,
            event_message,
            severity=event_severity,
            payload={
                "route": route,
                "qa_flags": qa_flags,
                "sources": list(final_state.get("sources") or []),
                **({"category": "incomplete_plan"} if show_incomplete else {}),
            },
        )
        return self.store.get_execution(eid)

    def _observe_failure(self, eid: str, *, question: str, route: str,
                         error: Optional[Dict[str, Any]],
                         qa_flags: List[str]) -> None:
        """Record a candidate lesson when the SAME failure repeats.

        A single failure is not a lesson (see app.learning.failures). This is
        deterministic, never auto-applied, and never allowed to affect the
        run: any error here is swallowed after being reported as an event.
        """
        from app.learning import failures as learning_failures

        try:
            outcome = learning_failures.observe(
                self.store, execution_id=eid, question=question, route=route,
                error=error,
                # QA findings are evidence too, but a COMPLETED run with no
                # flags is not a failure and must not be recorded as one.
                qa_flags=(list(qa_flags)
                          if (error or qa_flags) else []),
            )
        except Exception:  # noqa: BLE001 — learning never breaks an execution
            return
        if not outcome.get("recorded"):
            return
        lesson = outcome.get("lesson") or {}
        self._event(
            eid, LESSON_RECORDED,
            f"Repeated failure recorded as candidate lesson "
            f"{lesson.get('lesson_id')}",
            agent_id="qa", step_id="learning", severity=SEV_WARNING,
            payload={"lesson_id": lesson.get("lesson_id"),
                     "status": lesson.get("status"),
                     "occurrences": outcome.get("occurrences"),
                     "action": outcome.get("action")},
        )

    def _record_lesson_use(self, eid: str, final_state: Dict[str, Any],
                           *, success: bool) -> None:
        """Count applied lessons on this execution (best effort, additive).

        Only lessons a node actually inserted into a prompt are counted — an
        ACTIVE lesson that did not apply to this task is not credited.
        """
        applied = [str(x) for x in (final_state.get("lessons_applied") or [])]
        if not applied:
            return
        try:
            self.store.bump_lesson_stats(applied, success=success)
        except Exception:
            return
        self._event(
            eid, LESSONS_APPLIED,
            f"Applied {len(applied)} learned requirement(s)",
            agent_id="qa", step_id="learning",
            payload={"lesson_ids": applied, "success": success},
        )

    # -- GUI-10 approval lifecycle ------------------------------------------

    def resume_approval(self, approval_id: str) -> Dict[str, Any]:
        """Resolve a decided approval's bound operation (2026-10-28).

        Statuses, all fail-closed and idempotent:
          * pending   -> ApprovalDecisionRequired (decide first; nothing runs)
          * expired   -> ApprovalExpired (nothing runs)
          * rejected  -> finalize the honest LOCAL-ONLY answer on the bound
                         execution (only when run 1 answered with the
                         approval request); repeats are no-ops
          * approved  -> re-run ONLY the research step through the real gate
                         (research_entry revalidates the persisted decision,
                         bound payload and single-execution claim); a second
                         call finds the claim taken and runs nothing.

        Raises LookupError when the approval or its execution is unknown,
        ApprovalConflict when the approval is not bound to an execution.
        Nothing here trusts client input: the operation is always read back
        from the persisted request.
        """
        store = self.store
        row = store.get_approval(approval_id)
        if row is None:
            raise LookupError(approval_id)
        status = row.get("status")
        if status == "pending":
            if store.mark_expired_if_due(approval_id):
                raise ApprovalExpired(approval_id)
            raise ApprovalDecisionRequired(approval_id)
        if status == "expired":
            raise ApprovalExpired(approval_id)
        eid = row.get("execution_id")
        if not eid:
            raise ApprovalConflict(approval_id)
        stored = store.get_execution(eid)
        if stored is None:
            raise LookupError(eid)

        if status == "rejected":
            return self._finalize_rejected(eid, row, stored)

        # --- approved: resolve the BOUND operation for this operation type
        # 2026-10-30: approvals now cover TWO operations. The dispatch is by
        # the PERSISTED operation (never by client input), and each handler
        # revalidates its own bindings against the stored execution.
        operation = str(row.get("operation") or "research")

        def _resolve() -> Dict[str, Any]:
            if operation == "workspace_patch":
                return self._resume_workspace_patch(
                    approval_id, row, eid, stored)
            return self._resume_approved(approval_id, row, eid, stored)

        # GUI-10 recovery (2026-10-28): resolve an execution ONE at a time. The
        # persisted `executed_at` claim remains the durable guarantee that an
        # external operation runs once; this claim guarantees two overlapping
        # resume requests cannot continue the same plan concurrently. It lives
        # on the shared store because the API builds a runner per request.
        claim = getattr(store, "resume_claim", None)
        if claim is None:
            return _resolve()
        with claim(eid) as claimed:
            if not claimed:
                return {
                    "outcome": "denied",
                    "reason": "another resume is already resolving this "
                              "execution",
                    "approval": serialize_approval(row),
                    "execution": stored,
                }
            return _resolve()

    def _resume_approved(self, approval_id: str, row: Dict[str, Any],
                         eid: str, stored: Dict[str, Any]) -> Dict[str, Any]:
        """Run the bound approved operation — or recover its interrupted tail.

        Called only while this execution's exclusive resume claim is held.
        `executed_at` set means the external operation ALREADY ran once: it is
        never run again. If the persisted plan carries its result the
        remaining tasks are recovered (see `_recover_interrupted_continuation`);
        otherwise the run is closed honestly instead of guessing or searching
        a second time.
        """
        store = self.store
        payload = row.get("payload") or {}
        if payload.get("question") != stored.get("question"):
            return {
                "outcome": "denied",
                "reason": "bound question does not match this execution",
                "approval": serialize_approval(row),
                "execution": stored,
            }

        if (store.get_approval(approval_id) or {}).get("executed_at"):
            recovered = self._recover_interrupted_continuation(
                approval_id, row, eid, stored)
            if recovered is not None:
                return recovered
            self._event(
                eid, APPROVAL_FINALIZED,
                "Approved research was not executed again (already executed)",
                agent_id="research", step_id="approval", severity=SEV_WARNING,
                payload={"approval_id": approval_id, "outcome": "denied",
                         "reason": "already executed (single-execution "
                                   "claim)"},
            )
            return {
                "outcome": "denied",
                "reason": "already executed (single-execution claim)",
                "approval": serialize_approval(
                    store.get_approval(approval_id)),
                "execution": store.get_execution(eid),
            }

        state: Dict[str, Any] = {
            "question": payload.get("question") or stored.get("question") or "",
            "route": stored.get("route") or payload.get("route") or "",
            "research_approval": {"approval_id": approval_id},
        }
        if payload.get("books_refusal"):
            state["books_refusal"] = payload["books_refusal"]
        if stored.get("sources"):
            state["sources"] = list(stored["sources"])
        if payload.get("local_answer"):
            state["agent_answers"] = {"books": payload["local_answer"]}

        from app.graph import approval_scope, research_entry

        # `executed_at` going set here means THIS call ran the operation (the
        # durable single-execution claim inside the real gate).
        with instr.install(
            lambda t, msg, **kw: self._event(eid, t, msg, **kw)
        ), approval_scope(store, eid):
            update = research_entry(state) or {}
        executed_after = (store.get_approval(approval_id) or {}).get(
            "executed_at")

        sources = list(update.get("sources") or [])
        answer = str(update.get("answer") or "")
        if executed_after is not None:
            from app.graph import released_plan

            # A plan-bound execution still has the tasks that were blocked
            # when the run paused: release them and CONTINUE through the
            # existing dispatcher (GUI-10 follow-up).
            plan, released = released_plan(
                copy.deepcopy(stored.get("plan")), answer)
            if released is not None:
                # Durable proof BEFORE any continuation work: the released
                # plan (research task completed with its real result) and the
                # research answer are persisted first, so an interrupted
                # continuation can be recovered without searching again.
                self.store.update_execution(
                    eid, plan=plan, answer=answer, result=answer,
                    sources=sources)
                continued = self._continue_plan(
                    eid, store, stored, plan, answer, sources)
                summary = self._persist_continued(eid, approval_id, continued)
                return {
                    "outcome": "executed",
                    "approval_id": approval_id,
                    "answer": summary["answer"],
                    "approval": serialize_approval(
                        store.get_approval(approval_id)),
                    "execution": self.store.get_execution(eid),
                }

            # No releasable plan task (e.g. a direct research route): claim
            # `completed` only when the WHOLE plan finished — the partial
            # research result is kept either way.
            from app.qa_node import has_incomplete_plan

            plan_complete = not has_incomplete_plan(
                {"plan": stored.get("plan")})
            if plan_complete:
                self.store.update_execution(eid, status=STATUS_COMPLETED,
                                            answer=answer, result=answer,
                                            sources=sources, error=[],
                                            completed_at=True)
            else:
                self.store.update_execution(
                    eid, status=STATUS_FAILED, answer=answer, result=answer,
                    sources=sources,
                    error=_incomplete_plan_error(stored.get("plan")).to_dict(),
                    completed_at=True,
                )
            final_status = STATUS_COMPLETED if plan_complete else STATUS_FAILED
            self._event(
                eid, EXECUTION_APPROVAL_COMPLETED,
                ("Approved research finished" if plan_complete else
                 "Approved research finished — remaining planned tasks "
                 "did not complete"),
                agent_id="research", step_id="approval",
                payload={"approval_id": approval_id, "status": final_status},
            )
            return {
                "outcome": "executed",
                "approval_id": approval_id,
                "answer": answer,
                "approval": serialize_approval(store.get_approval(approval_id)),
                "execution": self.store.get_execution(eid),
            }

        # The gate denied (it revalidated against the persisted row and its
        # decision is authoritative). The execution's honest state is
        # untouched; the denial text rides in the response.
        reason = "the approval gate refused the stored approval"
        self._event(
            eid, APPROVAL_FINALIZED,
            "Approved research was not executed (approval unusable)",
            agent_id="research", step_id="approval", severity=SEV_WARNING,
            payload={"approval_id": approval_id, "outcome": "denied",
                     "reason": reason},
        )
        return {
            "outcome": "denied",
            "reason": reason,
            "answer": answer,
            "approval": serialize_approval(store.get_approval(approval_id)),
            "execution": self.store.get_execution(eid),
        }

    def _execution_has_pending_approval(self, eid: str) -> bool:
        """Whether THIS execution is still waiting for a user decision."""
        if not hasattr(self.store, "list_approvals"):
            return False
        return any(
            approval.get("execution_id") == eid
            for approval in self.store.list_approvals("pending")
        )

    def _recover_interrupted_continuation(self, approval_id: str,
                                          row: Dict[str, Any], eid: str,
                                          stored: Dict[str, Any]):
        """Continue an execution whose approved operation ran before a crash.

        Safe ONLY when persisted state PROVES the external operation already
        produced a result: the execution is still `waiting_for_approval` and
        its released plan carries a COMPLETED research task with that result
        (the approved path persists both before it continues). The operation is
        never repeated. Returns None when recovery does not apply, so the
        caller keeps the ordinary "already executed" denial.
        """
        from app.graph import is_approval_blocked_task_error
        from app.qa_node import has_incomplete_plan

        if stored.get("status") != STATUS_WAITING_APPROVAL:
            return None
        # A continuation that legitimately paused for ANOTHER decision is not
        # an interrupted one (2026-10-29): a pending request already bound to
        # this execution proves the plan got as far as the next gate, so
        # recovering here would re-run that gate and append duplicate events
        # for a replayed request. The user's decision on that request resumes
        # the plan instead.
        lister = getattr(self.store, "list_approvals", None)
        if lister is not None:
            for other in lister("pending") or []:
                if (other.get("execution_id") == eid
                        and other.get("approval_id") != approval_id):
                    return None
        plan = copy.deepcopy(stored.get("plan"))
        if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
            # A paused run with no plan cannot be finished either: its answer
            # WAS the research result the interruption lost.
            return self._close_unrecoverable_resume(
                approval_id, eid, stored, plan=None)
        if not has_incomplete_plan({"plan": plan}):
            # 2026-10-29: the continuation RAN TO THE END but the process died
            # before its terminal write, so the plan is complete while the
            # execution still says `waiting_for_approval`. Nothing is re-run:
            # the persisted plan and answer ARE the result, so finalize from
            # them. Refused honestly when the plan holds no persisted research
            # result (the approved operation's output is then unproven).
            if not any(
                isinstance(task, dict)
                and task.get("capability") == "research"
                and task.get("status") == "completed"
                and isinstance(task.get("result"), str)
                and task["result"].strip()
                for task in plan["tasks"]
            ):
                return self._close_unrecoverable_resume(
                    approval_id, eid, stored, plan=plan)
            return self._finalize_recovered_completion(
                approval_id, eid, stored, plan)
        # A task persisted as `running` means the plan was captured while that
        # task was mid-flight: nothing proves whether its side effects landed,
        # so continuing could repeat non-idempotent work. Refuse honestly.
        if any(isinstance(task, dict) and task.get("status") == "running"
               for task in plan["tasks"]):
            return self._close_unrecoverable_resume(
                approval_id, eid, stored, plan=plan,
                why=("The execution stopped while a planned task was still "
                     "running, so it is unknown whether that task finished."))
        # A task still marked `failed` with the approval-blocked error means
        # the plan was NEVER released for this decision: the operation may have
        # been claimed, but its result was not persisted, and an EARLIER
        # completed research task (from a previous gate) must not be mistaken
        # for proof of this one. Fail honestly instead of continuing.
        if any(
            isinstance(task, dict)
            and task.get("capability") == "research"
            and task.get("status") == "failed"
            and is_approval_blocked_task_error(task.get("error"))
            for task in plan["tasks"]
        ):
            return self._close_unrecoverable_resume(
                approval_id, eid, stored, plan=plan,
                why=("The plan still records a research task as blocked by "
                     "approval, so the result of the approved operation was "
                     "never recorded."))
        for task in plan["tasks"]:
            if (isinstance(task, dict)
                    and task.get("capability") == "research"
                    and task.get("status") == "completed"
                    and isinstance(task.get("result"), str)
                    and task["result"].strip()):
                break
        else:
            return self._close_unrecoverable_resume(
                approval_id, eid, stored, plan=plan)

        self._event(
            eid, STEP_STARTED,
            "Recovering the interrupted plan continuation",
            agent_id="dispatcher", step_id="dispatcher",
            payload={"approval_id": approval_id, "recovered": True},
        )
        continued = self._continue_plan(
            eid, self.store, stored, plan,
            str(stored.get("answer") or ""),
            list(stored.get("sources") or []))
        summary = self._persist_continued(eid, approval_id, continued)
        return {
            "outcome": "executed",
            "recovered": True,
            "approval_id": approval_id,
            "answer": summary["answer"],
            "approval": serialize_approval(
                self.store.get_approval(approval_id)),
            "execution": self.store.get_execution(eid),
        }

    def _finalize_recovered_completion(self, approval_id: str, eid: str,
                                       stored: Dict[str, Any],
                                       plan: Dict[str, Any]) -> Dict[str, Any]:
        """Close a paused run whose continuation actually FINISHED (2026-10-29).

        The process died after the last task but before `_persist_continued`,
        so the plan is complete while the execution still says
        `waiting_for_approval`. No task is dispatched again: the persisted plan
        is the result, so the terminal state is written from it (QA recomputed
        deterministically — the stored flags still describe the paused run).
        """
        from app.graph import CAPABILITY_AGENT_KEYS
        from app.qa_node import run_qa

        tasks = [t for t in plan.get("tasks") or [] if isinstance(t, dict)]
        answer = str(stored.get("answer") or "")
        if not answer.strip():
            # Nothing was recorded for the answer channel: use the last real
            # task result rather than inventing a summary.
            for task in tasks:
                if (task.get("status") == "completed"
                        and isinstance(task.get("result"), str)
                        and task["result"].strip()):
                    answer = task["result"]
        sources = list(stored.get("sources") or [])
        answers: Dict[str, str] = {}
        for task in tasks:
            if task.get("status") != "completed":
                continue
            key = CAPABILITY_AGENT_KEYS.get(task.get("capability"))
            result = task.get("result")
            if key and isinstance(result, str) and result.strip():
                answers[key] = result
        try:
            qa_flags = run_qa({
                "question": stored.get("question") or "",
                "route": stored.get("route") or "",
                "answer": answer,
                "sources": sources,
                "agent_answers": answers,
                "plan": plan,
            })
        except Exception:  # noqa: BLE001 — QA is fail-soft by design
            qa_flags = list(stored.get("qa_flags") or [])
        self.store.update_execution(
            eid, status=STATUS_COMPLETED, plan=plan, answer=answer,
            result=answer, sources=sources, qa_flags=qa_flags, error=[],
            completed_at=True,
        )
        self._event(
            eid, EXECUTION_APPROVAL_COMPLETED,
            "The interrupted continuation had already finished — plan "
            "completed",
            agent_id="research", step_id="approval",
            payload={"approval_id": approval_id, "status": STATUS_COMPLETED,
                     "recovered": True},
        )
        self._event(
            eid, EXECUTION_COMPLETED, "Execution completed",
            payload={"approval_id": approval_id,
                     "resumed_after_approval": True, "recovered": True},
        )
        return {
            "outcome": "executed",
            "recovered": True,
            "approval_id": approval_id,
            "answer": answer,
            "approval": serialize_approval(
                self.store.get_approval(approval_id)),
            "execution": self.store.get_execution(eid),
        }

    def _close_unrecoverable_resume(self, approval_id: str, eid: str,
                                    stored: Dict[str, Any],
                                    plan: Optional[Dict[str, Any]],
                                    why: Optional[str] = None):
        """Close a stranded paused execution truthfully (no guessing).

        The approval is spent, yet nothing proves what the external operation
        returned, so continuing would mean repeating a search. The run ends
        `failed` with a structured explanation and a recovery hint.
        """
        self.store.update_execution(
            eid, status=STATUS_FAILED,
            error=ExecutionError(
                category="resume_interrupted",
                human_message=("The approved research finished, but the "
                               "execution stopped before the rest of the "
                               "work could be recorded."),
                why=(why or
                     "The approval is already used and no persisted result "
                     "exists, so continuing would repeat the external search."),
                recovery=("Send the request again to start a new run; the old "
                          "approval is never reused."),
                recoverable=True,
                details={"approval_id": approval_id,
                         "plan_persisted": bool(plan)},
            ).to_dict(),
            completed_at=True,
        )
        self._event(
            eid, EXECUTION_FAILED,
            ("The interrupted resume could not continue — the approved "
             "research result was not persisted"),
            agent_id="research", step_id="approval", severity=SEV_ERROR,
            payload={"approval_id": approval_id,
                     "category": "resume_interrupted"},
        )
        return {
            "outcome": "denied",
            "recovered": False,
            "reason": ("approved operation already executed but its result "
                       "is not recoverable"),
            "approval": serialize_approval(self.store.get_approval(approval_id)),
            "execution": self.store.get_execution(eid),
        }

    def _continue_plan(self, eid: str, store, stored: Dict[str, Any],
                       plan: Dict[str, Any], seed_answer: str,
                       seed_sources) -> Dict[str, Any]:
        """Run the REMAINING tasks of an already-released plan (GUI-10).

        Reuses the existing dispatcher: completed tasks are skipped, a later
        research task still passes through the approval gate (creating its own
        pending request instead of searching), and failures are recorded
        honestly. Bounded by the dispatcher's own hard visit cap.

        Progress is persisted AFTER EVERY task (2026-10-29): a crash during the
        continuation then leaves the completed tasks and their real results in
        the stored plan, so `_recover_interrupted_continuation` resumes where
        it stopped instead of replaying work that may not be idempotent. The
        mid-flight writes never touch status/completed_at — only the terminal
        persistence does that.
        """
        from app.graph import CAPABILITY_AGENT_KEYS, approval_scope, continue_plan

        # Rebuild the accumulated answer channel from the persisted plan: the
        # store keeps task results but not agent_answers, and dependent tasks
        # legitimately read earlier outputs (feasibility consumes research and
        # books). Only real completed results are replayed.
        answers: Dict[str, str] = {}
        for task in plan.get("tasks") or []:
            if not isinstance(task, dict) or task.get("status") != "completed":
                continue
            key = CAPABILITY_AGENT_KEYS.get(task.get("capability"))
            result = task.get("result")
            if key and isinstance(result, str) and result.strip():
                answers[key] = result

        state: Dict[str, Any] = {
            "question": stored.get("question") or "",
            "route": stored.get("route") or "",
            "plan": plan,
            "answer": str(seed_answer or ""),
            "sources": list(seed_sources or []),
            "agent_answers": answers,
        }
        self._event(
            eid, STEP_STARTED,
            "Continuing the remaining planned tasks after the approval",
            agent_id="dispatcher", step_id="dispatcher",
            payload={"completed": sorted(answers)},
        )

        def _persist_progress(current: Dict[str, Any]) -> None:
            """Write task progress as it happens (crash-safe continuation)."""
            progress_plan = current.get("plan")
            if not isinstance(progress_plan, dict):
                return
            self.store.update_execution(
                eid, plan=progress_plan,
                answer=str(current.get("answer") or ""),
                sources=list(current.get("sources") or []),
            )

        with instr.install(
            lambda t, msg, **kw: self._event(eid, t, msg, **kw)
        ), approval_scope(store, eid):
            continued = continue_plan(state, on_progress=_persist_progress)
        tasks = (continued.get("plan") or {}).get("tasks") or []
        self._event(
            eid, STEP_COMPLETED,
            "Remaining planned tasks finished",
            agent_id="dispatcher", step_id="dispatcher",
            payload={
                "tasks": [
                    {"task_id": t.get("task_id"), "status": t.get("status")}
                    for t in tasks if isinstance(t, dict)
                ],
            },
        )
        return continued

    def _persist_continued(self, eid: str, approval_id: str,
                           continued: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a continued plan with its truthful terminal status/events."""
        from app.qa_node import has_incomplete_plan

        plan = continued["plan"]
        answer = str(continued.get("answer") or "")
        sources = list(continued.get("sources") or [])
        qa_flags = list(continued.get("qa_flags") or [])
        incomplete = has_incomplete_plan({"plan": plan})
        # A LATER research task may itself need a decision: that is a pause
        # again, never a search and never a success.
        awaiting = self._execution_has_pending_approval(eid)
        if awaiting:
            status, error = STATUS_WAITING_APPROVAL, None
        elif incomplete:
            status = STATUS_FAILED
            error = _incomplete_plan_error(plan).to_dict()
        else:
            status, error = STATUS_COMPLETED, None
        self.store.update_execution(
            eid, status=status, plan=plan, answer=answer, result=answer,
            sources=sources, qa_flags=qa_flags, error=error, completed_at=True,
        )
        if awaiting:
            message = ("Approved research finished — the plan is paused "
                       "again for another decision")
        elif incomplete:
            message = ("Approved research finished — remaining planned "
                       "tasks did not complete")
        else:
            message = "Approved research finished — plan completed"
        self._event(
            eid, EXECUTION_APPROVAL_COMPLETED, message,
            agent_id="research", step_id="approval",
            severity=SEV_INFO if status == STATUS_COMPLETED else SEV_WARNING,
            payload={"approval_id": approval_id, "status": status},
        )
        # Same terminal event contract as a normal run: the timeline must say
        # how the CONTINUED execution actually ended.
        self._event(
            eid,
            (EXECUTION_WAITING_APPROVAL if awaiting else
             EXECUTION_FAILED if incomplete else EXECUTION_COMPLETED),
            ("Execution paused while waiting for approval" if awaiting
             else "Execution finished with incomplete planned tasks"
             if incomplete else "Execution completed"),
            severity=(SEV_WARNING if awaiting else
                      SEV_ERROR if incomplete else SEV_INFO),
            payload={
                "approval_id": approval_id,
                "resumed_after_approval": True,
                **({"category": "incomplete_plan"} if incomplete else {}),
            },
        )
        return {"answer": answer, "status": status}

    # -- workspace coding (2026-10-30) ---------------------------------------

    def _resume_workspace_patch(self, approval_id: str, row: Dict[str, Any],
                                eid: str, stored: Dict[str, Any]
                                ) -> Dict[str, Any]:
        """Apply an APPROVED workspace patch — or refuse it honestly.

        The reviewed patch is revalidated from the PERSISTED request: the
        bound request must match this execution, the payload must still match
        its own digest, and EVERY target file must still have the content
        hash the patch was prepared against. Any mismatch means nothing is
        written (`stale_patch`), because applying half a review is worse than
        not applying it. The write itself is atomic per file; a failure
        midway reports exactly what was written.
        """
        from app.coding.patch import (
            PatchError,
            apply_patch,
            validate_payload,
        )
        from app.coding.workspace import (
            WorkspaceError,
            configured_workspace,
            describe_workspace_problem,
            redact_secrets,
        )

        store = self.store
        payload = row.get("payload") or {}
        if payload.get("question") != stored.get("question"):
            return self._patch_refused(
                approval_id, eid, "bound request does not match this "
                "execution", category="invalid_patch", applied=False)

        try:
            workspace = configured_workspace()
        except WorkspaceError as exc:
            return self._patch_refused(
                approval_id, eid, describe_workspace_problem(exc),
                category="workspace_unavailable", applied=False)
        if workspace is None:
            return self._patch_refused(
                approval_id, eid,
                "Workspace coding is not enabled: no coding workspace root is "
                "configured, so the reviewed patch was not applied.",
                category="workspace_unavailable", applied=False)

        try:
            changes = validate_payload(payload)
        except PatchError as exc:
            return self._patch_refused(
                approval_id, eid, str(exc), category="invalid_patch",
                applied=False)

        # Exactly one application per approval (durable claim).
        if not store.claim_approval_execution(approval_id):
            # The claim is consumed, so this call MUST NOT write. If the
            # execution is still non-terminal the earlier attempt died before
            # its terminal write: reconcile from DISK instead of leaving a
            # paused execution that can never progress (and never re-apply).
            recovered = self._recover_executed_patch(
                approval_id, eid, workspace, changes, payload)
            if recovered is not None:
                return recovered
            self._event(
                eid, APPROVAL_FINALIZED,
                "Workspace patch was not applied again (already applied)",
                agent_id="coding", step_id="approval", severity=SEV_WARNING,
                payload={"approval_id": approval_id, "outcome": "denied",
                         "reason": "already executed (single-execution claim)"},
            )
            return {
                "outcome": "denied",
                "reason": "already executed (single-execution claim)",
                "approval": serialize_approval(
                    store.get_approval(approval_id)),
                "execution": store.get_execution(eid),
            }

        try:
            result = apply_patch(workspace, changes, payload)
        except PatchError as exc:
            return self._patch_refused(
                approval_id, eid, redact_secrets(str(exc)),
                category="stale_patch", applied=False)
        except WorkspaceError as exc:
            return self._patch_refused(
                approval_id, eid, describe_workspace_problem(exc),
                category="stale_patch", applied=False)

        validation = [dict(v) for v in (payload.get("validation") or [])
                      if isinstance(v, dict)]
        validation_note = (
            "Offline validation (does not prove the change is correct): "
            + "; ".join(
                f"{v.get('path')}: {v.get('status')}" for v in validation)
            if validation else
            "No offline validation was recorded for this patch.")

        if result.ok:
            answer = (
                "Applied the approved workspace patch to: "
                + ", ".join(result.applied)
                + ". Each file was re-checked against the reviewed content "
                "hash before and after writing.\n" + validation_note
            )
            self.store.update_execution(
                eid, status=STATUS_COMPLETED, answer=answer, result=answer,
                completed_at=True)
            self._event(
                eid, EXECUTION_APPROVAL_COMPLETED,
                "Workspace patch applied: " + ", ".join(result.applied),
                agent_id="coding", step_id="approval", severity=SEV_INFO,
                payload={"approval_id": approval_id,
                         "status": STATUS_COMPLETED,
                         "files": list(result.applied)},
            )
            return {
                "outcome": "executed",
                "approval_id": approval_id,
                "answer": answer,
                "applied": list(result.applied),
                "approval": serialize_approval(
                    store.get_approval(approval_id)),
                "execution": self.store.get_execution(eid),
            }

        # Partial/failed application: name what landed and what did not, and
        # never claim success.
        failed_paths = [f.get("path") for f in result.failed]
        answer = (
            "The approved patch was only PARTIALLY applied."
            + (" Written: " + ", ".join(result.applied) if result.applied
               else " No file was written.")
            + (" Failed: " + ", ".join(str(p) for p in failed_paths)
               if failed_paths else "")
            + ("\n" + validation_note if validation else "")
        )
        self.store.update_execution(
            eid, status=STATUS_FAILED, answer=answer, result=answer,
            error=ExecutionError(
                category="patch_apply_failed",
                human_message="The approved workspace patch could not be "
                              "fully applied.",
                why="; ".join(
                    f"{f.get('path')}: {f.get('reason')}"
                    for f in result.failed)[:2000] or "unknown write failure",
                recovery="Re-read the affected files and prepare a new patch.",
                recoverable=True,
            ).to_dict(),
            completed_at=True,
        )
        self._event(
            eid, EXECUTION_FAILED,
            "Workspace patch partially applied",
            agent_id="coding", step_id="approval", severity=SEV_ERROR,
            payload={"approval_id": approval_id, "status": STATUS_FAILED,
                     "applied": list(result.applied),
                     "failed": [str(p) for p in failed_paths]},
        )
        return {
            "outcome": "failed",
            "approval_id": approval_id,
            "answer": answer,
            "applied": list(result.applied),
            "approval": serialize_approval(
                store.get_approval(approval_id)),
            "execution": self.store.get_execution(eid),
        }

    def _patch_refused(self, approval_id: str, eid: str, reason: str, *,
                       category: str, applied: bool) -> Dict[str, Any]:
        """Close a workspace-patch resume that could not apply anything.

        Ordering matters: the file checks run BEFORE the first write, so a
        refusal here means no byte changed. The execution is closed `failed`
        with a structured error rather than left waiting forever.
        """
        answer = (
            "The reviewed patch was NOT applied. " + str(reason)
            + " No file was modified."
        )
        self.store.update_execution(
            eid, status=STATUS_FAILED, answer=answer, result=answer,
            error=ExecutionError(
                category=category,
                human_message="The approved workspace patch was not applied.",
                why=str(reason)[:2000],
                recovery=("Prepare a new patch from the current file "
                          "contents and approve that one."),
                recoverable=True,
            ).to_dict(),
            completed_at=True,
        )
        self._event(
            eid, EXECUTION_FAILED,
            "Workspace patch refused: " + str(reason)[:300],
            agent_id="coding", step_id="approval", severity=SEV_ERROR,
            payload={"approval_id": approval_id, "status": STATUS_FAILED,
                     "category": category, "applied": applied},
        )
        return {
            "outcome": "failed",
            "approval_id": approval_id,
            "reason": reason,
            "answer": answer,
            "applied": [],
            "approval": serialize_approval(
                self.store.get_approval(approval_id)),
            "execution": self.store.get_execution(eid),
        }

    def _recover_executed_patch(self, approval_id: str, eid: str,
                                workspace, changes, payload: Dict[str, Any]
                                ) -> Optional[Dict[str, Any]]:
        """Reconcile a patch whose single-execution claim was already consumed.

        Reached when an earlier resume claimed the approval and then died
        before writing the execution's terminal state. NOTHING is written
        here: every target file is compared against the reviewed BEFORE and
        AFTER content hashes, so the report is disk evidence, not a guess.

        * all AFTER     -> close `completed`; the patch is on disk
        * all BEFORE    -> close `failed`; nothing was modified
        * anything else -> close `failed`, naming exactly what is (un)proven

        Returns None when the execution is already terminal, so the ordinary
        "already executed" denial is kept.
        """
        stored = self.store.get_execution(eid) or {}
        if stored.get("status") not in (None, STATUS_WAITING_APPROVAL,
                                        STATUS_RUNNING):
            return None

        applied: List[str] = []
        untouched: List[str] = []
        unproven: List[str] = []
        for change in changes:
            try:
                _raw, current = workspace.read_raw(change.path)
            except Exception:  # noqa: BLE001 — unreadable is unproven
                unproven.append(change.path)
                continue
            if current == change.after_sha256:
                applied.append(change.path)
            elif current == change.before_sha256:
                untouched.append(change.path)
            else:
                unproven.append(change.path)

        validation = [dict(v) for v in (payload.get("validation") or [])
                      if isinstance(v, dict)]
        validation_note = (
            "Offline validation (does not prove the change is correct): "
            + "; ".join(
                f"{v.get('path')}: {v.get('status')}" for v in validation)
            if validation else
            "No offline validation was recorded for this patch.")

        if applied and not untouched and not unproven:
            answer = (
                "The approved workspace patch is present on disk: "
                + ", ".join(applied)
                + ". Each file matches the reviewed content hash; the "
                "earlier attempt lost its final status update, so nothing "
                "was re-applied.\n" + validation_note
            )
            self.store.update_execution(
                eid, status=STATUS_COMPLETED, answer=answer, result=answer,
                completed_at=True)
            self._event(
                eid, APPROVAL_FINALIZED,
                "Workspace patch confirmed on disk after an interrupted "
                "resume",
                agent_id="coding", step_id="approval", severity=SEV_WARNING,
                payload={"approval_id": approval_id,
                         "outcome": "recovered",
                         "files": list(applied)},
            )
            return {
                "outcome": "recovered",
                "approval_id": approval_id,
                "answer": answer,
                "applied": list(applied),
                "approval": serialize_approval(
                    self.store.get_approval(approval_id)),
                "execution": self.store.get_execution(eid),
            }

        if untouched and not applied and not unproven:
            reason = ("the earlier attempt consumed this approval but no file "
                      "was written")
            category = "patch_apply_unverified"
        else:
            parts = []
            if applied:
                parts.append(f"already applied: {', '.join(applied)}")
            if untouched:
                parts.append(f"unchanged: {', '.join(untouched)}")
            if unproven:
                parts.append(f"unproven: {', '.join(unproven)}")
            reason = ("recovery cannot prove one outcome: "
                      + "; ".join(parts)
                      + " — nothing was re-applied")
            category = "patch_apply_conflict"
        answer = (
            "The reviewed patch was NOT re-applied after the interrupted "
            "resume. " + reason + "\n" + validation_note
        )
        self.store.update_execution(
            eid, status=STATUS_FAILED, answer=answer, result=answer,
            error=ExecutionError(
                category=category,
                human_message="The approved workspace patch could not be "
                              "confirmed.",
                why=reason[:2000],
                recovery=("Re-read the affected files and prepare a new "
                          "patch from their current contents."),
                recoverable=True,
            ).to_dict(),
            completed_at=True,
        )
        self._event(
            eid, EXECUTION_FAILED,
            "Workspace patch recovery failed: " + reason[:300],
            agent_id="coding", step_id="approval", severity=SEV_ERROR,
            payload={"approval_id": approval_id,
                     "category": category,
                     "applied": list(applied),
                     "unchanged": list(untouched),
                     "unproven": list(unproven)},
        )
        return {
            "outcome": "failed",
            "approval_id": approval_id,
            "reason": reason,
            "answer": answer,
            "applied": list(applied),
            "approval": serialize_approval(
                self.store.get_approval(approval_id)),
            "execution": self.store.get_execution(eid),
        }

    def _finalize_rejected(self, eid: str, row: Dict[str, Any],
                           stored: Dict[str, Any]) -> Dict[str, Any]:
        """Close a rejected request without executing its external operation.

        A request that paused its execution must leave ``waiting_for_approval``
        for the truthful failed terminal state. A preserved local answer may
        replace the approval-pending text, but no unrelated answer is rewritten.
        Repeated resolution is idempotent and appends no duplicate event.
        """
        payload = row.get("payload") or {}
        local = payload.get("local_answer")
        answer = str(stored.get("answer") or "")
        # 2026-10-30: the same rejection path closes BOTH operations, with
        # text that names what was refused (a patch is never "research").
        is_patch = str(row.get("operation") or "research") == "workspace_patch"
        updates: Dict[str, Any] = {}
        if stored.get("status") == STATUS_WAITING_APPROVAL:
            updates.update(
                status=STATUS_FAILED,
                error=ExecutionError(
                    category="approval_rejected",
                    human_message=(
                        "The execution stopped because the workspace patch "
                        "was rejected." if is_patch else
                        "The execution stopped because research approval "
                        "was rejected."),
                    why=(
                        "The proposed file change was not authorized; no "
                        "file was modified." if is_patch else
                        "The proposed external research was not authorized."),
                    recovery=(
                        "Ask for a new patch if you want to reconsider."
                        if is_patch else
                        "Retry with a new research request if you want to "
                        "reconsider."),
                    recoverable=True,
                ).to_dict(),
                completed_at=True,
            )

        from app.supervisor import APPROVAL_PENDING_MARKER

        use_local = (not is_patch and isinstance(local, str)
                     and bool(local.strip())
                     and APPROVAL_PENDING_MARKER in answer)
        if use_local:
            updates.update(answer=local, result=local,
                           sources=list(stored.get("sources") or []))

        finalized = bool(updates)
        if finalized:
            self.store.update_execution(eid, **updates)
            self._event(
                eid, APPROVAL_FINALIZED,
                ("Workspace patch rejected — no file was modified"
                 if is_patch else
                 "Research rejected — execution closed with local-only results"
                 if use_local else
                 "Research rejected — execution closed without external research"),
                agent_id="research", step_id="approval",
                payload={
                    "approval_id": row.get("approval_id"),
                    "outcome": "rejected",
                    **({"status": STATUS_FAILED}
                       if stored.get("status") == STATUS_WAITING_APPROVAL else {}),
                },
            )
        return {
            "outcome": "rejected",
            "finalized": finalized,
            "approval": serialize_approval(self.store.get_approval(
                str(row.get("approval_id")))),
            "execution": self.store.get_execution(eid),
        }

    def _fail(self, eid: str, exc: Exception) -> Dict[str, Any]:
        err = _execution_error(exc)
        self.store.update_execution(
            eid,
            status=STATUS_FAILED,
            error=err.to_dict(),
            completed_at=True,
        )
        self._event(
            eid, EXECUTION_FAILED, err.human_message, severity=SEV_ERROR,
            payload=err.to_dict(),
        )
        # 2026-10-30 (learning): a raise is a failure too, so it participates
        # in repeated-failure detection exactly like an incomplete plan does.
        stored = self.store.get_execution(eid) or {}
        self._observe_failure(
            eid, question=str(stored.get("question") or ""),
            route=str(stored.get("route") or ""), error=err.to_dict(),
            qa_flags=[],
        )
        return self.store.get_execution(eid)


def _incomplete_plan_error(plan: Any) -> ExecutionError:
    """Structured, honest error for a plan that ended with unfinished tasks.

    Shared by `run()` (plan incomplete at run end) and
    `resume_approval()` (approved research ran, but the plan it belongs to
    still has failed/blocked tasks) so both report the identical contract.
    """
    tasks = (plan or {}).get("tasks") or []
    incomplete_tasks = [
        {"task_id": task.get("task_id"), "status": task.get("status"),
         "error": task.get("error") or task.get("block_reason")}
        for task in tasks if isinstance(task, dict)
        and task.get("status") != "completed"
    ]
    return ExecutionError(
        category="incomplete_plan",
        human_message="The request was only partially completed.",
        why="One or more planned tasks did not complete successfully.",
        recovery="Review the task results and retry the incomplete work.",
        recoverable=True,
        details={"incomplete_tasks": incomplete_tasks},
    )


def _execution_error(exc: Exception) -> ExecutionError:
    """Map a runner exception to the structured, honest user-facing form."""
    raw = str(exc)
    category = classify_ai_exception(exc)
    is_ai = category in {
        "auth_rejected", "connection_refused", "dns_unreachable", "timeout",
        "wrong_service", "model_missing", "bad_response", "malformed_endpoint",
    }
    if is_ai or any(marker.lower() in raw.lower()
                    for marker in _AI_PROVIDER_CRASH_MARKERS):
        return ai_error_for_provider_unavailable("LLM", category)

    if isinstance(exc, json.JSONDecodeError):
        return ExecutionError(
            category="execution_error",
            human_message="The execution could not be recorded.",
            why="Internal data could not be read or written.",
            recovery="Retry the execution; if it persists, report this as a bug.",
            details={"exception": type(exc).__name__},
        )

    return ExecutionError(
        category="execution_error",
        human_message="The execution failed unexpectedly.",
        why="An internal error occurred while running the workflow.",
        recovery="You can retry the execution or check Settings → AI connection "
                 "if the problem looks provider-related.",
        details={"exception": type(exc).__name__, "message": raw[:2000]},
    )
