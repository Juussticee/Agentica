"""Graph wiring (spec Section 3): START -> supervisor -> agents -> qa -> END.

Phase 2: the feasibility route runs the spec's sub_routes chain first
(research -> books -> feasibility), so both prior answers are in state when
the feasibility agent synthesizes the study. Single-agent routes end directly.
Phase 5: every route's terminal hop passes through the deterministic QA
gate, which only appends state["qa_flags"] (never blocks, never routes).
Phase 7: when the supervisor marked the request "complex", one conditional
hop routes supervisor -> planner (structured plan, data only) -> the SAME
route table below. Simple/multi requests bypass the planner, byte-identical
to pre-Phase-7 wiring.

Phase 8 (MASTER_PLAN §5.3): a validated plan is EXECUTED through a
dispatcher node, a literal capability→node mapping, and a hard iteration
cap. The dispatcher picks the next task whose depends_on have COMPLETED
SUCCESSFULLY, runs exactly one whitelisted graph node for it, records the
task status on the plan structure, and returns to itself until every task
has a terminal status — then hands the final state to the deterministic QA
gate (with plan-completion checks). Failed/blocked/malformed tasks are
marked honestly on the plan; execution can never loop forever, and
simple/multi requests never enter the dispatcher at all (their existing
edges are byte-identical to pre-Phase-8).
"""

from langgraph.graph import END, START, StateGraph

from app.agents.books_agent import books_agent
from app.agents.erp_agent import erp_agent
from app.agents.feasibility_agent import feasibility_agent
from app.agents.research_agent import research_agent
from app.config import MAX_PLAN_TASKS, MAX_TASK_RETRIES
from app.planner import planner_node
from app.qa_node import qa_node
from app.state import AgentState
from app.supervisor import supervisor

# supervisor -> first node per route; feasibility enters via its sub_routes
# chain (research first, per the feasibility_agent contract).
_ROUTES = {
    "books": "books",
    "research": "research",
    "erp": "erp",
    "feasibility": "research",
}

# ---------------------------------------------------------------------------
# Phase 8: dispatcher (MASTER_PLAN §5.3)
# ---------------------------------------------------------------------------

# Literal capability -> graph-node mapping, defined in CODE. Plan data
# (including LLM-planned data after Phase 7 validation) may only select
# capabilities from here — never arbitrary node names, never callables.
# "synthesis" dispatches to the existing feasibility agent as the terminal
# consumer (MASTER_PLAN Phase 8; the dedicated synthesizer is Phase 10).
CAPABILITY_NODES = {
    "research": "research",
    "books": "books",
    "erp": "erp",
    "synthesis": "feasibility",
}

# Task statuses recorded on the plan structure (Section 5.6: bookkeeping
# lives on the plan, NO new AgentState keys).
TASK_PENDING = "pending"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_BLOCKED = "blocked"       # a dependency failed/blocked ⇒ never executed
TASK_UNSUPPORTED = "unsupported"  # capability outside the code mapping

_TERMINAL_STATUSES = (TASK_COMPLETED, TASK_FAILED, TASK_BLOCKED, TASK_UNSUPPORTED)

# Plan key holding the dispatcher's visit counter (bookkeeping on the plan
# structure, Section 5.6 — never a new AgentState key).
_DISPATCH_VISITS = "dispatcher_visits"


def _hard_iteration_cap(plan: dict) -> int:
    """Absolute dispatch-loop bound (MASTER_PLAN §5.3):
    MAX_PLAN_TASKS * (1 + MAX_TASK_RETRIES) dispatch visits. Best case one
    task completes per visit; anything above the cap proves a pathological
    plan (e.g. tasks that can never become ready) and must fail safe
    instead of looping forever."""
    return MAX_PLAN_TASKS * (1 + MAX_TASK_RETRIES)


def _normalize_task(task: dict) -> dict:
    """Ensure a task dict carries execution bookkeeping fields (status)."""
    if not isinstance(task, dict):
        return task
    if "status" not in task:
        task["status"] = TASK_PENDING
    return task


def _plan_components(plan: dict):
    """Return (plan_dict, tasks_list) for a structurally usable plan, or
    (None, None) when the plan is not usable at all."""
    if not isinstance(plan, dict):
        return None, None
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        return None, None
    return plan, tasks


def _dependency_satisfied(task: dict, tasks_by_id: dict) -> str:
    """Check a task's depends_on against current statuses.

    Returns one of: "ready" (all deps completed), "blocked" (some dep
    failed/blocked/unsupported), "waiting" (some dep still pending).
    Unknown dep id / malformed dep list is treated as blocked (fail safe).
    """
    deps = task.get("depends_on")
    if not isinstance(deps, list):
        return TASK_BLOCKED
    for dep_id in deps:
        dep = tasks_by_id.get(dep_id)
        if dep is None or not isinstance(dep, dict):
            return TASK_BLOCKED
        status = dep.get("status")
        if status == TASK_COMPLETED:
            continue
        if status in _TERMINAL_STATUSES:
            return TASK_BLOCKED
        return "waiting"
    return "ready"


_TASK_NODE_FNS = {
    "research": "research_agent",
    "books": "books_agent",
    "erp": "erp_agent",
    "feasibility": "feasibility_agent",
}


def _dispatch_task(state: AgentState, task: dict, node_name: str) -> dict:
    """Run one task's graph node; record an HONEST terminal status.

    Task results flow through the EXISTING channels (agent_answers /
    sources / answer via with_agent_output inside the agent nodes) — no new
    state keys. Only per-task status/result bookkeeping lives on the plan.
    A task is marked completed ONLY when its node actually returned output;
    a raising node is marked failed (never crashes the run).

    Returns the state-update dict to hand to LangGraph: the node's own
    output already contains the FULL accumulated channels (the agent nodes
    merge prior agent_answers/sources via with_agent_output), so returning
    it on every dispatcher visit keeps state identical to the direct routes.
    """
    task["status"] = "running"
    module_fn_name = _TASK_NODE_FNS[node_name]
    # Node identity is resolved FROM THIS MODULE'S BINDINGS so test doubles
    # installed on app.graph (the established convention in every suite)
    # apply to dispatcher-executed nodes too.
    node_fn = globals()[module_fn_name]
    try:
        update = node_fn(state) or {}
    except Exception as exc:  # noqa: BLE001 — dispatcher is the safety boundary
        task["status"] = TASK_FAILED
        task["result"] = None
        task["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return {}

    task["status"] = TASK_COMPLETED
    task["result"] = update.get("answer") if isinstance(update, dict) else None
    return update if isinstance(update, dict) else {}


def _fail_safe_plan(plan: dict, reason: str) -> dict:
    """Hard-cap / unusable-plan fail safe (MASTER_PLAN §5.3): mark every
    non-terminal task failed, note the reason, never loop, never raise."""
    for task in plan.get("tasks") or []:
        if isinstance(task, dict) and task.get("status") not in _TERMINAL_STATUSES:
            task["status"] = TASK_FAILED
            task["result"] = None
            task["error"] = reason
    note = plan.get("fallback_note")
    plan["fallback_note"] = (note + "; " if note else "") + reason
    return plan


def _after_research(state: AgentState):
    """Chain hand-off inside feasibility; direct research route ends here."""
    return "books" if state.get("route") == "feasibility" else END


def _after_books(state: AgentState):
    """Chain hand-off inside feasibility; direct books route ends here."""
    return "feasibility" if state.get("route") == "feasibility" else END


def _accumulated_channels(state: AgentState, node_update: dict) -> dict:
    """Build the dispatcher's full state-update dict.

    with_agent_output() (used by every agent node) already returns the FULL
    prior sources/agent_answers merged with the new values, so a task node's
    update is the freshest accumulated state. LangGraph persists exactly the
    keys the dispatcher returns, so every visit must re-emit the channels —
    otherwise earlier task outputs would be forgotten before dependent
    tasks (or QA) read them. Values already present in the node update win;
    when this visit ran no task, the current state channels are re-emitted
    unchanged.
    """
    update = dict(node_update)
    for key in ("answer", "sources", "agent_answers"):
        if key not in update and key in state:
            update[key] = state[key]
    return update


def dispatcher_node(state: AgentState) -> dict:
    """Phase 8 dispatcher (MASTER_PLAN §5.3).

    Chooses ONE next runnable task per invocation and runs its whitelisted
    node. The graph loops dispatcher → dispatcher until every task reaches a
    terminal status, then routes to the QA gate (which now also verifies
    plan completion). Honesty rules: a task is only ever marked completed
    when its node actually produced output; failed/blocked/unsupported
    tasks are recorded on the plan and their dependents become blocked —
    never executed, never marked completed.

    The hard visit cap (MAX_PLAN_TASKS * (1 + MAX_TASK_RETRIES)) makes any
    pathological plan terminate deterministically: when exceeded, every
    non-terminal task is marked failed with the cap reason and the plan is
    shipped to QA fail-safe.

    Returns ONLY changed state keys: the plan (status bookkeeping) plus the
    currently-accumulated answer/sources/agent_answers channels, which the
    executed task's node produced via with_agent_output(). Returning the
    accumulated channels on EVERY visit is required: LangGraph applies only
    the returned keys, so dropping them between visits would forget earlier
    task outputs before dependencies read them.
    """
    plan, tasks = _plan_components(state.get("plan"))
    if tasks is None:
        # Defensive: an unusable plan arrived in state; nothing to execute.
        # QA will flag the empty answer; execution cannot loop here.
        return {}

    tasks = [_normalize_task(t) for t in tasks if isinstance(t, dict)]
    plan["tasks"] = tasks

    # --- hard iteration cap (fail safe, never loop forever) ---
    visits = int(plan.get(_DISPATCH_VISITS, 0) or 0) + 1
    plan[_DISPATCH_VISITS] = visits
    if visits > _hard_iteration_cap(plan):
        return _accumulated_channels(
            state,
            {
                "plan": _fail_safe_plan(
                    plan, "dispatcher iteration cap exhausted — plan failed safe"
                )
            },
        )

    usable_tasks = [t for t in tasks if isinstance(t, dict)]
    tasks_by_id = {t.get("task_id"): t for t in usable_tasks}

    next_task = None
    for task in usable_tasks:
        # Skip already-terminal tasks (re-entries never re-execute them).
        if task.get("status") in _TERMINAL_STATUSES:
            continue
        readiness = _dependency_satisfied(task, tasks_by_id)
        if readiness == "ready":
            next_task = task
            break
        if readiness == TASK_BLOCKED:
            # Deterministically blocked: record honestly, never execute.
            task["status"] = TASK_BLOCKED
            task["result"] = None
            task["block_reason"] = "a dependency did not complete successfully"
            continue
        # readiness == "waiting": try the next candidate.

    if next_task is None:
        # Every task has a terminal status here; waiting-only outcomes are
        # impossible because a waiting task's dependency is always earlier
        # in the (validated) plan and already handled on this or a previous
        # visit. The visit cap above guarantees termination regardless.
        return _accumulated_channels(state, {"plan": plan})

    capability = next_task.get("capability")
    node_name = CAPABILITY_NODES.get(capability)
    if node_name is None:
        # Unsupported capability (validated plans cannot produce this, so it
        # can only be corrupt state) — fail safe, never dispatch to an
        # arbitrary node.
        next_task["status"] = TASK_UNSUPPORTED
        next_task["result"] = None
        next_task["error"] = (
            f"capability {capability!r} has no whitelisted graph node"
        )
        return _accumulated_channels(state, {"plan": plan})

    # Run the one task and return BOTH the plan bookkeeping, the task
    # node's own state update (answer/sources/agent_answers) and the
    # accumulated channels — LangGraph merges them, exactly like the direct
    # route edges used to.
    task_update = _dispatch_task(state, next_task, node_name)
    return _accumulated_channels(state, {"plan": plan, **task_update})


def _after_dispatcher(state: AgentState) -> str:
    """Router after the dispatcher: loop back while unfinished (ready or
    waiting-but-not-terminal) tasks exist, else hand the final state to QA.

    The dispatcher itself enforces the hard visit cap, so this router can
    never spin forever on a pathological plan — once the cap fires, every
    task reaches a terminal (failed) status and the loop exits to QA."""
    plan = state.get("plan")
    if not isinstance(plan, dict):
        return "qa"
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        return "qa"
    for task in tasks:
        if isinstance(task, dict) and task.get("status") not in _TERMINAL_STATUSES:
            return "dispatcher"
    return "qa"


def build_graph(node_wrapper=None):
    """Compile the graph; optionally wrap every agent node for observation.

    node_wrapper (optional): callable (name, node_fn) -> node_fn used by the
    execution runner to emit events. Default path with no argument keeps the
    exact Phase 5 behavior (all 124 regression tests call build_graph()).
    The QA node is never wrapped (it is infrastructure, not an agent), and
    neither is the planner node (Phase 7: plan data production, not an
    agented step). The Phase 8 dispatcher is likewise infrastructure and is
    never wrapped.
    """
    def _maybe_wrap(name, fn):
        if node_wrapper is None or name not in (
            "supervisor", "books", "research", "erp", "feasibility"
        ):
            return fn
        return node_wrapper(name, fn)

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _maybe_wrap("supervisor", supervisor))
    graph.add_node("planner", planner_node)
    graph.add_node("books", _maybe_wrap("books", books_agent))
    graph.add_node("research", _maybe_wrap("research", research_agent))
    graph.add_node("erp", _maybe_wrap("erp", erp_agent))
    graph.add_node("feasibility", _maybe_wrap("feasibility", feasibility_agent))
    graph.add_node("dispatcher", _maybe_wrap("dispatcher", dispatcher_node))

    graph.add_edge(START, "supervisor")
    # Phase 7 conditional hop: complex requests visit the planner first;
    # Phase 8: the planner now chains into the EXECUTION dispatcher instead
    # of the old route table (plans are executed, not routed by capability).
    graph.add_conditional_edges(
        "supervisor",
        lambda state: "planner" if state.get("complex") else state.get("route", "research"),
        {"planner": "planner", **_ROUTES},
    )
    graph.add_edge("planner", "dispatcher")
    graph.add_conditional_edges(
        "dispatcher",
        _after_dispatcher,
        {"dispatcher": "dispatcher", "qa": "qa"},
    )
    graph.add_conditional_edges(
        "research",
        _after_research,
        # Phase 5: direct-research terminal passes through the QA gate.
        {"books": "books", END: "qa"},
    )
    graph.add_conditional_edges(
        "books",
        _after_books,
        # Phase 5: direct-books terminal passes through the QA gate.
        {"feasibility": "feasibility", END: "qa"},
    )
    graph.add_edge("feasibility", "qa")
    graph.add_edge("erp", "qa")
    graph.add_edge("qa", END)

    graph.add_node("qa", qa_node)

    return graph.compile()
