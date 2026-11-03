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

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, List, Optional

from langgraph.graph import END, START, StateGraph

from app.agents.books_agent import books_agent
from app.agents.erp_agent import erp_agent
from app.agents.feasibility_agent import feasibility_agent
from app.agents.research_agent import research_agent
from app.agents.data_agent import data_agent
from app import evidence
from app.coding.agent import coding_node
from app.config import MAX_PLAN_TASKS, MAX_TASK_RETRIES
from app.followup import followup_node
from app.learning import retrieval as learning_retrieval
from app.learning.feedback import feedback_node
from app.execution.events import (
    APPROVAL_CREATED,
    APPROVAL_EXECUTED,
    FALLBACK_DECIDED,
)
from app.execution.instrumentation import emit
from app.planner import GENERIC_PLAN_OBJECTIVES, planner_node
from app.qa_node import qa_node
from app.state import AgentState, with_agent_output
from app.supervisor import (
    approval_request_text,
    conversational_reply,
    fallback_decision,
    research_authorization,
    supervisor,
    unsupported_reply,
)

# supervisor -> first node per route; feasibility enters via its sub_routes
# chain (research first, per the feasibility_agent contract).
# "conversational" (2026-10-27 greeting fix): a bare opener is answered by
# the deterministic node below — never by Books or Web Research.
# "general" / "unsupported" (2026-10-28 Priority A): model-knowledge answer
# and deterministic guidance — both single-step, both source-free.
_ROUTES = {
    "books": "books",
    "research": "research",
    "erp": "erp",
    "feasibility": "research",
    "conversational": "conversational",
    "general": "general",
    "unsupported": "unsupported",
    # 2026-10-30: workspace coding — read-only inspection, and patch
    # PREPARATION that is persisted as an approval request (never applied by
    # this node). Only reachable when CODING_WORKSPACE_ROOT is configured.
    "coding": "coding",
    # 2026-10-30: dataset analysis — read-only, workspace-confined, every
    # figure computed in app/data/analysis.py. Only reachable when a
    # workspace root is configured.
    "data": "data",
    # 2026-10-30: contextual follow-up — answers from the RECORDED prior
    # execution the runner supplied, or says there is none. It never invents
    # what the user was told earlier.
    "followup": "followup",
    # 2026-10-30: a correction of earlier work — revises this answer from the
    # recorded result and records a structured lesson (app/learning/).
    "feedback": "feedback",
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
    # 2026-10-30: dataset analysis as a plan task (read-only, computed in
    # app/data/analysis.py).
    "data": "data",
    # 2026-10-28: the terminal synthesis of a NON-feasibility plan answers the
    # user from the completed tasks' results. It used to dispatch to the
    # feasibility agent, which wrote a Market/Financial/Risk business study
    # (and a .docx) for ordinary books/ERP/research requests. The feasibility
    # route is unchanged: its synthesis task IS the study, and synthesis_node
    # delegates there.
    "synthesis": "synthesis",
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

# Bookkeeping strings shared by the dispatcher and the approval-resume path
# (GUI-10 follow-up): the dispatcher records WHY an unapproved research task
# did not run, and resume_approval must recognise exactly those failures to
# continue the plan after the operation is approved. One definition each, so
# the two sides can never drift.
DEPENDENCY_BLOCK_REASON = "a dependency did not complete successfully"
APPROVAL_BLOCKED_TASK_PREFIX = (
    "external web research was not authorized for this request")


def is_approval_blocked_task_error(error: object) -> bool:
    """Whether a plan task failed only because research needed approval."""
    text = str(error or "")
    return (APPROVAL_BLOCKED_TASK_PREFIX in text
            and "approval request is pending" in text)


# capability -> the agent_answers key its node writes (with_agent_output).
# Used to rebuild the accumulated answer channel from a persisted plan when
# a paused plan is continued, and by synthesis_node to find the results of
# completed tasks. "synthesis" writes under its own key (2026-10-29): the
# terminal synthesis of an ordinary plan is answered by synthesis_node, not
# by the feasibility agent (which the feasibility ROUTE still uses).
CAPABILITY_AGENT_KEYS = {
    "research": "research",
    "books": "books",
    "erp": "erp",
    "data": "data",
    "synthesis": "synthesis",
}


def _hard_iteration_cap(plan: dict) -> int:
    """Absolute dispatch-loop bound (MASTER_PLAN §5.3):
    MAX_PLAN_TASKS * (1 + MAX_TASK_RETRIES) dispatch visits. Best case one
    task completes per visit; anything above the cap proves a pathological
    plan (e.g. tasks that can never become ready) and must fail safe
    instead of looping forever."""
    return MAX_PLAN_TASKS * (1 + MAX_TASK_RETRIES)


def request_text(state: AgentState) -> str:
    """The ORIGINAL user request behind this run.

    A dispatched task may run with `state["question"]` set to its OWN
    instruction (see `task_instruction`), so everything that must stay bound
    to what the user actually asked — research authorization, the approval
    payload, the synthesis prompt — reads this helper instead of the question.
    Outside a dispatched task it is simply the question, so single-step
    routes are unchanged and no extra state key is needed on the run.
    """
    return str(state.get("request") or state.get("question") or "")


def task_instruction(task: dict, request: str) -> Optional[str]:
    """The instruction one plan task's specialist should receive, or None.

    Returns None when the task must answer the original request:
      * a `research` task — the external search stays exactly the operation
        the user approved, never a plan-generated query (research_entry
        revalidates the approval against the request, so searching anything
        else would widen what was approved);
      * a TEMPLATE objective (app.planner.GENERIC_PLAN_OBJECTIVES) — those
        describe a step of a plan that already routes the original question
        to one source, so they ask nothing on their own;
      * an objective equal to the request, or no objective at all.

    Otherwise (an LLM-planned sub-request) the objective IS the task's
    question. The original request travels beside it as `state["request"]`.
    """
    if str(task.get("capability") or "") == "research":
        return None
    objective = " ".join(str(task.get("objective") or "").split())
    if not objective or objective == " ".join(request.split()):
        return None
    if objective in GENERIC_PLAN_OBJECTIVES:
        return None
    return objective


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
    # research goes through its approval gate (2026-10-28 Priority B):
    # a plan task may not become a back door around it. The gate resolves
    # the research_agent global at call time, so every existing test double
    # installed on app.graph keeps applying.
    "research": "research_entry",
    "books": "books_agent",
    "erp": "erp_agent",
    "feasibility": "feasibility_agent",
    # 2026-10-30: dataset analysis — read-only, workspace-confined.
    "data": "data_agent",
    "synthesis": "synthesis_node",
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
    # TASK-SCOPED EXECUTION (2026-10-29): a planned sub-request runs as its own
    # instruction; the original request stays available as state["request"] for
    # anything bound to what the user asked. Template objectives and research
    # tasks keep the original question (task_instruction).
    request = request_text(state)
    instruction = task_instruction(task, request)
    working: AgentState = (state if instruction is None else
                           {**state, "request": request,
                            "question": instruction})
    # Priority B (2026-10-28): outbound research is gated HERE as well, so a
    # plan task can never become a back door around the approval gate. The
    # gate node itself persists/reuses the pending ApprovalRequest and builds
    # the approval-pending answer (it never executes when unauthorized); this
    # layer exists so the plan records an HONEST outcome — a task that did not
    # run is `failed`, never `completed` with an approval request as its result.
    if module_fn_name == "research_entry":
        allowed, code = research_authorization(request, state.get("route", ""))
        if not allowed:
            update = research_entry(working) or {}
            task["status"] = TASK_FAILED
            task["result"] = None
            task["error"] = (
                f"{APPROVAL_BLOCKED_TASK_PREFIX} ({code}) — an approval "
                "request is pending"
            )[:300]
            return {"answer": update.get("answer", "")}
    # Node identity is resolved FROM THIS MODULE'S BINDINGS so test doubles
    # installed on app.graph (the established convention in every suite)
    # apply to dispatcher-executed nodes too.
    node_fn = globals()[module_fn_name]
    try:
        update = node_fn(working) or {}
    except Exception as exc:  # noqa: BLE001 — dispatcher is the safety boundary
        task["status"] = TASK_FAILED
        task["result"] = None
        task["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return {}

    update = update if isinstance(update, dict) else {}
    answer = update.get("answer")
    if not (isinstance(answer, str) and answer.strip()):
        # The documented contract is "completed ONLY when its node actually
        # produced output". A node that returned no answer did NOT complete
        # its task — recording it completed with a `None` result let a partial
        # run report success and gave dependent tasks nothing to consume.
        task["status"] = TASK_FAILED
        task["result"] = None
        task["error"] = "the task produced no output"
        return update
    task["status"] = TASK_COMPLETED
    task["result"] = answer
    return update


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
    """Chain hand-off inside feasibility; direct books route ends here —
    unless the orchestrated knowledge-fallback decision sends the ORIGINAL
    question to the existing research agent (2026-10-25 additional phase).

    Exactly ONE fallback transition is possible: research's own edge below
    is END for every non-feasibility route, so books <-> research loops are
    structurally impossible; fallback_decision additionally refuses when
    research already answered. Diagnostics: the fallback log line carries
    the classification code (failure class + refusal reason).
    """
    if state.get("route") == "feasibility":
        return "feasibility"
    allowed, code = fallback_decision(state)
    if state.get("books_refusal"):
        # W2: record the DECISION itself (both outcomes) while it is made —
        # previously only the allowed branch printed a log line. Payload is
        # the diagnostic code already proven scalar-only (no passage text).
        emit(
            FALLBACK_DECIDED,
            f"Knowledge fallback decision: {code}",
            agent_id="supervisor", step_id="books",
            payload={"allowed": bool(allowed), "code": code},
        )
    if allowed:
        print(f"[graph] knowledge fallback books->research ({code})")
        return "research"
    return END


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


def conversational_node(state: AgentState) -> dict:
    """Deterministic reply to a bare conversational opener (2026-10-27).

    Root-cause fix for execution 3fc3bc8460c9 ("hello" -> live web search ->
    irrelevant sourced answer). No LLM, no retrieval, no tools: the reply is
    a fixed string from app.supervisor.conversational_reply(), provenance
    stays empty because no external source was consulted, and only the keys
    this node changes are returned (existing state contract). Infrastructure
    node like planner/dispatcher/qa — never wrapped by the runner's agent
    observer, so no new event names or payload shapes appear.
    """
    reply = conversational_reply(state.get("question", ""))
    return with_agent_output(state, "conversational", reply, [])


def _memory_answer(question: str, learned: str, note: str) -> str:
    """Answer from model knowledge, explicitly labelled as unchecked.

    The label is unconditional: on this path no source was consulted, so the
    reply must never read like verified evidence (2026-11-01).
    """
    instructions = (
        "Answer the user's general question concisely and factually "
        "from your own knowledge. If you are not sure, say so plainly "
        "instead of guessing. If the request asks for code or a draft, "
        "put it in your reply: you cannot create files or run commands. "
    )
    prompt = instructions + "\n\nQuestion: " + str(question) + learned
    from app.llm_client import get_llm, llm_text, strip_instruction_echo

    answer = strip_instruction_echo(
        str(llm_text(get_llm().invoke(prompt)) or "").strip(), instructions)
    if not answer:
        return ""
    return answer.rstrip() + "\n\n" + note


def general_node(state: AgentState) -> dict:
    """Answer a general question, grounded in local evidence when it exists.

    2026-10-28 (A): the routing policy only sends a question here when it
    matches the question-shape rules and NO source cue, so a local document
    question or a current-information request can never be answered here.

    2026-11-01 (evidence grounding): before answering from model knowledge
    the node consults the SAME persisted index the books route uses and asks
    the deterministic coverage question — do these passages actually cover
    what was asked? Three outcomes, each reported honestly:

      grounded  every requested topic is in the passages -> the model composes
                from the passages only and the existing books grounding gate
                keeps only what they support; citations come from the
                passages that supported a retained sentence.
      partial   the topic is covered but a requested detail is missing ->
                supported content is kept and the missing detail is named.
      none      nothing relevant was retrieved, the passages do not survive
                the gate, or the evidence contradicts itself -> answer from
                memory with an explicit unchecked label and NO sources.

    Fail-soft throughout: no index, no embedding model or no provider yields
    an honest message, never research and never a fabricated citation.
    """
    question = str(state.get("question") or "")
    # Learned requirements for this kind of task (app/learning/retrieval.py):
    # only ACTIVE lessons whose capability matches, rendered compactly. Empty
    # when nothing applies, so a fresh installation is byte-identical.
    learned = learning_retrieval.notes(state, "general", question)

    raw_nodes = evidence.retrieve(question)
    # Rerank before assessment (2026-11-01): similarity alone ranks a passage
    # that merely shares the topic above the one that answers the question,
    # returns duplicates, and cannot cover a two-part question from two
    # documents. Deterministic, offline, and recorded in state["evidence"] so
    # the decision is auditable after the fact.
    picked = evidence.select(question, raw_nodes)
    nodes = picked["nodes"]
    coverage = evidence.assess(question, nodes)
    use_evidence = coverage["status"] in ("grounded", "partial")

    answer, sources, grounding = "", [], {"mode": "none", "claims": 0,
                                           "dropped": []}
    supplied: List[int] = []
    answered_from = "model"
    provider_error = None
    try:
        if use_evidence:
            from app.llm_client import get_llm, llm_text, strip_instruction_echo

            # Render once and keep the indices the prompt contains, so the
            # snapshot can distinguish "retrieved" from "supplied to the
            # model" instead of implying every passage was consulted
            # (2026-11-02).
            block, supplied = evidence.prompt_passages(nodes)
            raw = strip_instruction_echo(
                str(llm_text(get_llm().invoke(
                    evidence.compose_prompt(question, nodes, learned, block)
                )) or "").strip(),
                evidence.EVIDENCE_INSTRUCTIONS)
            answer, sources, grounding = evidence.postprocess(raw, nodes)
            if answer:
                extras = []
                if coverage["missing"]:
                    extras.append(
                        "Not stated by the passages: "
                        + ", ".join(coverage["missing"]) + ".")
                if coverage["conflict"]:
                    extras.append(
                        "The passages name different licences, so no single "
                        "licence can be confirmed from them.")
                if extras:
                    answer = answer.rstrip() + "\n\n" + " ".join(extras)
                answered_from = "evidence"
        if not answer:
            # Either no relevant evidence, or evidence that did not survive
            # the grounding gate. Say which, and label the reply unchecked.
            note = (evidence.LOCAL_ONLY_NOTE if nodes
                    else "(Answered from my own knowledge; not checked "
                         "against any source.)")
            answer = _memory_answer(question, learned, note)
    except Exception as exc:  # noqa: BLE001 — fail-soft, never routes
        provider_error = type(exc).__name__
        answered_from = "error"
        answer = (
            "I could not generate an answer right now "
            f"(the AI provider is unavailable: {provider_error}). "
            "Try again once the service is configured."
        )

    if not answer:
        answer = (
            "I could not produce an answer for this question. Rephrase it, "
            "or ask for research explicitly."
        )

    # Claim-level verification (2026-11-01): the grounding gate works over the
    # WHOLE retrieved set, so a sentence can survive on passage A's support
    # while citing passage B. Each claim is therefore re-checked against the
    # passage it actually cites; only evidence-built answers have claims to
    # check (a memory answer has no cited passage by definition).
    claims = {"claims": [], "summary": {}, "checkable": False}
    if answered_from == "evidence":
        from app import claim_check

        claims = claim_check.verify(answer, nodes)

    update = with_agent_output(state, "general", answer, sources)
    # QA reads what was actually consulted; passage text stays in memory only
    # (the runner never persists the full graph state).
    update["evidence"] = {
        "status": coverage["status"] if not provider_error else "error",
        "answered_from": answered_from,
        "coverage": coverage.get("coverage"),
        "passages": coverage["passages"],
        "best_score": coverage["best_score"],
        "missing": coverage["missing"],
        "conflict": coverage["conflict"],
        "reason": coverage["reason"],
        "grounding": grounding["mode"],
        # Number of sentences the grounding gate retained as substantive
        # claims (NOT the verification list below — that is `claim_checks`).
        "grounding_claims": grounding["claims"],
        "files": evidence.evidence_files(nodes),
        "used": list(sources),
        "text": evidence.evidence_text(nodes),
        # Reranking + provenance (2026-11-01): the decision trail, the passages
        # as used (file/page/chunk/timestamps, NEVER a local path) and the
        # claim-to-passage verdicts. The runner persists these as an immutable
        # snapshot; QA reports them.
        "ranking": picked["ranking"],
        "dropped_duplicates": picked["dropped_duplicates"],
        "passages_detail": evidence.passage_details(
            nodes, picked["ranking"], supplied=supplied),
        "claim_checks": claims,
    }
    applied = learning_retrieval.applied_ids(state, "general", question)
    if applied:
        update["lessons_applied"] = applied
    return update


def unsupported_node(state: AgentState) -> dict:
    """Deterministic guidance for requests we will not execute (2026-10-28).

    Covers requests for capabilities that do not exist, requests for
    conversation memory we do not keep, and underspecified fragments. No
    LLM, no tools, empty provenance — an honest "I did not do that".
    """
    reply = unsupported_reply(state.get("question", ""))
    return with_agent_output(state, "unsupported", reply, [])


def synthesis_node(state: AgentState) -> dict:
    """Terminal `synthesis` task: answer from the work the plan really did.

    The planner's plans end with a synthesis task whose expected output is
    "answer to the user's question". This node composes that answer from the
    completed tasks' results already in `agent_answers` — no new retrieval, no
    tools, no report file, and no invented findings: when the model is
    unavailable the results are returned verbatim with an explicit note.

    The feasibility route keeps its own terminal step: there the synthesis IS
    the feasibility study, so it delegates to the existing agent (resolved
    from this module's bindings so every established test double applies).
    """
    if state.get("route") == "feasibility":
        return globals()["feasibility_agent"](state)

    answers = state.get("agent_answers") or {}
    findings = []
    for task in (state.get("plan") or {}).get("tasks") or []:
        if not isinstance(task, dict) or task.get("status") != "completed":
            continue
        key = CAPABILITY_AGENT_KEYS.get(task.get("capability"))
        text = answers.get(key) if key else None
        if not (isinstance(text, str) and text.strip()):
            continue
        findings.append((str(task.get("objective") or key).strip(),
                         text.strip()))

    if not findings:
        return with_agent_output(
            state, "synthesis",
            "None of the planned steps produced a usable result, so there is "
            "nothing to combine into an answer. Nothing was invented to fill "
            "the gap.",
            [],
        )

    blocks = "\n\n".join(f"Step: {objective}\nResult:\n{text}"
                          for objective, text in findings)
    # The synthesis answers the ORIGINAL request, even when it runs as a
    # scoped task (its own objective is a step description, not the question).
    question = request_text(state)
    # Learned requirements for the capability this plan served (a research
    # correction must reach the answer that reports the research).
    learned = learning_retrieval.notes(state, str(state.get("route") or ""),
                                       question)
    answer = ""
    try:
        from app.llm_client import get_llm, llm_text

        answer = str(llm_text(get_llm().invoke(
            "Combine the completed work below into ONE answer to the user's "
            "request. Use only these results; where they are insufficient, "
            "say so plainly instead of inventing facts.\n\n"
            f"Request: {question}\n\n{blocks}" + learned
        )) or "").strip()
    except Exception as exc:  # noqa: BLE001 — fail-soft, never fabricates
        answer = (
            "I could not compose a combined answer right now (the AI "
            f"provider is unavailable: {type(exc).__name__}). Here is what "
            "the completed steps actually found:\n\n" + blocks
        )
    if not answer:
        answer = ("The completed steps found the following. I could not "
                  "compose them into a single answer:\n\n" + blocks)
    update = with_agent_output(state, "synthesis", answer, ["synthesis"])
    applied = learning_retrieval.applied_ids(
        state, str(state.get("route") or ""), question)
    if applied:
        update["lessons_applied"] = applied
    return update


def research_entry(state: AgentState) -> dict:
    """Approval gate in front of the web-research node (Priority B + GUI-10).

    Decision order — every branch fails closed:
      1. in-message authorization (research_authorization): an explicit web
         request, a current-information request, or a feasibility study ->
         the research node runs, exactly as before GUI-10.
      2. state["research_approval"] (set ONLY by GraphRunner.resume_approval):
         the persisted decision is revalidated HERE — status approved, not
         expired, bound question/route/diagnosis match this state, and the
         single-execution claim succeeds — before the node runs.
      3. otherwise: create or reuse a PENDING ApprovalRequest bound to this
         execution's operation and answer with its id. Nothing external
         happens; if persistence is unavailable the answer is a denial, not
         a permission.
    """
    question = request_text(state)
    route = state.get("route", "")
    allowed, code = research_authorization(question, route)
    if allowed:
        # Resolved from this module's bindings so test doubles installed on
        # app.graph (the established convention) apply unchanged. The search
        # query is the REQUEST even when the task ran scoped: a plan objective
        # may narrow the work, never widen the approved external operation.
        asked = state if state.get("question") == question else {
            **state, "question": question}
        return globals()["research_agent"](asked)
    diagnosis = state.get("books_refusal")
    diagnosis = diagnosis if isinstance(diagnosis, dict) else None
    if diagnosis:
        code = f"{code}:{fallback_decision(state)[1]}"
    ref = state.get("research_approval")
    if isinstance(ref, dict) and ref.get("approval_id"):
        return _resumed_approval_update(state, code, diagnosis,
                                        str(ref["approval_id"]))
    return _pending_approval_update(state, code, diagnosis)


# ---------------------------------------------------------------------------
# GUI-10 (2026-10-28): approval scope + request lifecycle around the gate
# ---------------------------------------------------------------------------

# Execution-context isolation, mirroring the W2.1 event-sink pattern: the
# runner binds (store, execution_id) around graph.invoke()/resume, restores
# the prior scope in `finally`, and two overlapping runs can never see each
# other's approvals. Outside a run it is None -> no persistence -> the gate
# stays fail-closed without minting an unverifiable request.
_approval_scope: ContextVar = ContextVar("agentica_approval_scope",
                                         default=None)


@contextmanager
def approval_scope(store, execution_id: str):
    """Bind the approval store + owning execution for the current context."""
    token = _approval_scope.set((store, execution_id))
    try:
        yield store, execution_id
    finally:
        _approval_scope.reset(token)


def current_approval_scope():
    return _approval_scope.get()


def _research_description(question: str) -> str:
    """Human description of the PROPOSED action (server-side built; the
    client never supplies it)."""
    q = " ".join(str(question or "").split())
    return f"Web search (up to 8 results, reading up to 3 pages) for: {q[:240]}"


def _research_payload(state: AgentState, diagnosis: dict | None) -> dict:
    """The CONSTRAINED operation bound to the request: question, route, the
    scalar-only local diagnosis and (for a fallback) the local-only answer
    a rejection can fall back to. No URLs, no commands, no code — there is
    nothing executable to smuggle through an approval."""
    payload = {
        # The ORIGINAL request, never a task instruction: the decision binds
        # to what the user asked, and the resume path revalidates it against
        # the execution's stored question.
        "question": request_text(state),
        "route": str(state.get("route") or ""),
        "books_refusal": diagnosis,
    }
    if diagnosis:
        answers = state.get("agent_answers")
        local = (answers or {}).get("books") if isinstance(answers, dict) else None
        if isinstance(local, str) and local.strip():
            payload["local_answer"] = local[:4000]
    return payload


def _pending_approval_update(state: AgentState, code: str,
                             diagnosis: dict | None) -> dict:
    """Create/reuse the PENDING request (when a store is in scope) and build
    the approval-pending answer. Never executes anything external."""
    # The ORIGINAL request: the pending request, its description and its
    # payload all describe what the user asked, not a task instruction.
    question = request_text(state)
    scope = current_approval_scope()
    if scope is None:
        # No execution context (CLI, direct calls): fail-closed answer with
        # no request id — an unverifiable id would be worse than none.
        return with_agent_output(
            state, "research", approval_request_text(question, code, diagnosis),
            [],
        )
    store, execution_id = scope
    try:
        request = store.create_or_reuse_approval(
            operation="research",
            description=_research_description(question),
            payload=_research_payload(state, diagnosis),
            execution_id=execution_id,
        )
    except Exception as exc:  # noqa: BLE001 — persistence failure = DENIAL
        return with_agent_output(
            state, "research",
            approval_request_text(question, code, diagnosis)
            + "\n\nI could not register the approval request "
            f"({type(exc).__name__}), so nothing will be searched. "
            "Retry, or answer again later.",
            [],
        )
    emit(
        APPROVAL_CREATED, "Approval requested: outbound web research",
        agent_id="research", step_id="approval",
        payload={"approval_id": request["approval_id"],
                 "operation": "research", "reused": bool(request.get("reused"))},
    )
    return with_agent_output(
        state, "research",
        approval_request_text(question, code, diagnosis,
                              request["approval_id"]),
        [],
    )


def _resumed_approval_update(state: AgentState, code: str,
                             diagnosis: dict | None,
                             approval_id: str) -> dict:
    """GUI-10 resume path: revalidate the PERSISTED decision against THIS
    state, claim the single permitted execution, then run the research node.

    Any mismatch — no scope, unknown id, foreign execution, operation not
    research, not approved, expired, bound question/route/diagnosis
    different, or already executed — denies with an honest answer that
    still discloses that nothing was searched.
    """
    question = request_text(state)
    base = approval_request_text(question, code, diagnosis)
    scope = current_approval_scope()

    def _denied(reason: str) -> dict:
        return with_agent_output(
            state, "research",
            base + f"\n\nThe approval could not be used ({reason}), so no "
                   "search was performed.",
            [],
        )

    if scope is None:
        return _denied("no approval store in this context")
    store, execution_id = scope
    row = store.get_approval(approval_id)
    if row is None:
        return _denied("unknown approval request")
    if row.get("operation") != "research":
        return _denied("operation mismatch")
    if row.get("execution_id") != execution_id:
        return _denied("approval belongs to another execution")
    if row.get("status") != "approved":
        return _denied(f"approval is {row.get('status')}")
    payload = row.get("payload") or {}
    if payload.get("question") != str(question):
        return _denied("bound question mismatch")
    if payload.get("route", "") != str(state.get("route") or ""):
        return _denied("bound route mismatch")
    if (payload.get("books_refusal") or None) != (diagnosis or None):
        return _denied("bound diagnosis mismatch")
    if not store.claim_approval_execution(approval_id):
        return _denied("already executed or expired")
    try:
        update = globals()["research_agent"](state)
    except Exception:
        # The approved operation failed: give the claim back so the decision
        # is not silently consumed by an infrastructure error.
        store.release_approval_execution(approval_id)
        raise
    emit(
        APPROVAL_EXECUTED, "Approved outbound web research executed",
        agent_id="research", step_id="approval",
        payload={"approval_id": approval_id, "operation": "research"},
    )
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
            task["block_reason"] = DEPENDENCY_BLOCK_REASON
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


def released_plan(plan: dict, research_answer: str) -> tuple:
    """Un-block a persisted plan after its research task was approved (GUI-10).

    Only bookkeeping changes here — no task is executed:
      * the FIRST research task that failed solely because it needed approval
        becomes `completed` with the real approved research answer (the task
        did complete, just later than the run);
      * tasks the dispatcher marked `blocked` are reset to `pending` so the
        SAME dispatcher re-evaluates them against the current statuses — a
        dependency that is still failed is re-blocked without running.
      * every other status is left exactly as recorded (completed tasks are
        never re-run, failed tasks stay failed, unsupported stays unsupported).

    Returns (plan, task_id_of_the_released_research_task_or_None). The plan is
    returned unchanged (with None) when no approval-blocked research task
    exists — the caller then keeps the honest incomplete-plan outcome.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        return plan, None
    released = None
    for task in plan["tasks"]:
        if not isinstance(task, dict):
            continue
        if (released is None
                and task.get("capability") == "research"
                and task.get("status") == TASK_FAILED
                and is_approval_blocked_task_error(task.get("error"))):
            task["status"] = TASK_COMPLETED
            task["result"] = research_answer
            task.pop("error", None)
            released = task.get("task_id")
            continue
        if (task.get("status") == TASK_BLOCKED
                and task.get("block_reason") == DEPENDENCY_BLOCK_REASON):
            task["status"] = TASK_PENDING
            task.pop("block_reason", None)
    return plan, released


def continue_plan(state: AgentState,
                  on_progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Run the REMAINING tasks of an already-released plan (GUI-10).

    Used by GraphRunner.resume_approval after an approved research step: the
    plan (with its recorded task statuses) is fed back through the EXISTING
    dispatcher exactly as the graph would, so completed tasks are skipped,
    dependencies are honoured, a later research task still has to pass the
    approval gate (creating a new pending request instead of searching), and
    failures are recorded honestly. Bounded by the same hard visit cap as a
    normal run; never loops, never raises.

    Returns the plan plus the accumulated answer/sources/agent_answers/QA
    channels; empty when there is no usable plan.

    `on_progress(current)` (2026-10-29) is called after EVERY dispatcher visit
    with the state as it stands at that moment, so a caller can persist task
    progress as it happens. A crash mid-continuation then leaves a plan whose
    completed tasks carry their real results, and a recovery continues from
    there instead of replaying non-idempotent work. The callback is strictly
    observational: raising from it would lose the plan, so failures are
    reported and swallowed here.
    """
    plan, tasks = _plan_components(state.get("plan"))
    if tasks is None:
        return {}
    current: dict = dict(state)
    current["plan"] = plan

    def _report() -> None:
        if on_progress is None:
            return
        try:
            on_progress(current)
        except Exception as exc:  # noqa: BLE001 — bookkeeping is fail-soft
            print(f"[graph] plan progress could not be recorded "
                  f"({exc.__class__.__name__}) — continuation continues")

    _report()  # the plan as released, BEFORE the first task runs
    for _ in range(_hard_iteration_cap(plan) + 1):
        update = dispatcher_node(current)
        if not isinstance(update, dict) or not update:
            break
        current.update(update)
        _report()
        if _after_dispatcher(current) != "dispatcher":
            break
    # The run-1 QA flags describe a state that no longer exists; recompute
    # them from the continued plan (qa_node merges, so start from an empty
    # list rather than appending to the stale findings).
    current["qa_flags"] = []
    current.update(qa_node(current) or {})
    return {
        "plan": current.get("plan"),
        "answer": current.get("answer", ""),
        "sources": list(current.get("sources") or []),
        "agent_answers": dict(current.get("agent_answers") or {}),
        "qa_flags": list(current.get("qa_flags") or []),
    }


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
            "supervisor", "books", "research", "erp", "feasibility",
            # 2026-10-28: these two carry a real agent answer too, so the
            # runner's observer records them like every other answer step.
            "general", "unsupported",
            # 2026-10-30: workspace coding and dataset analysis carry a real
            # answer as well.
            "coding", "data",
            # 2026-10-30: the follow-up node answers from a recorded run, so
            # its step is observable like every other answer step.
            "followup",
            # 2026-10-30: the feedback node revises an answer and records a
            # lesson; both belong in the observable timeline.
            "feedback"
        ):
            return fn
        return node_wrapper(name, fn)

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _maybe_wrap("supervisor", supervisor))
    graph.add_node("planner", planner_node)
    graph.add_node("books", _maybe_wrap("books", books_agent))
    # research_entry owns the approval gate; the observer must wrap the
    # OUTER function so an approval-blocked run still records its step.
    graph.add_node("research", _maybe_wrap("research", research_entry))
    graph.add_node("erp", _maybe_wrap("erp", erp_agent))
    graph.add_node("feasibility", _maybe_wrap("feasibility", feasibility_agent))
    graph.add_node("conversational", conversational_node)
    graph.add_node("general", _maybe_wrap("general", general_node))
    graph.add_node("unsupported", _maybe_wrap("unsupported", unsupported_node))
    graph.add_node("coding", _maybe_wrap("coding", coding_node))
    graph.add_node("data", _maybe_wrap("data", data_agent))
    graph.add_node("followup", _maybe_wrap("followup", followup_node))
    graph.add_node("feedback", _maybe_wrap("feedback", feedback_node))
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
        # 2026-10-25: the knowledge-fallback hop to the EXISTING research
        # node joins the map (research's own edge is END for this route, so
        # the transition is one-way).
        {"feasibility": "feasibility", "research": "research", END: "qa"},
    )
    graph.add_edge("feasibility", "qa")
    graph.add_edge("conversational", "qa")
    graph.add_edge("general", "qa")
    graph.add_edge("unsupported", "qa")
    graph.add_edge("coding", "qa")
    graph.add_edge("data", "qa")
    graph.add_edge("followup", "qa")
    graph.add_edge("feedback", "qa")
    graph.add_edge("erp", "qa")
    graph.add_edge("qa", END)

    graph.add_node("qa", qa_node)

    return graph.compile()
