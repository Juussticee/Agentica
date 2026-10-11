"""Planner (Phase 7, MASTER_PLAN §5.2): conditional structured decomposition.

Responsibilities (all enforced here):
  * Decide whether a request NEEDS a multi-step plan (complex detection) or
    can go through the existing direct execution path unchanged.
  * When planning is needed, ask the LLM (via get_llm(), never a second
    factory) for a structured plan and parse it DEFENSIVELY (JSON-in-text,
    provider neutral).
  * Validate the plan deterministically (pure function `validate_plan`) —
    task_ids unique/well-formed, depends_on refs exist, graph is acyclic,
    required_inputs satisfiable from the plan's own capabilities, task
    count capped by MAX_PLAN_TASKS, capabilities inside the code-defined
    whitelist.
  * On ANY planner failure (disabled, LLM error, malformed/adversarial
    plan) fall back to a deterministic default plan derived from the
    current keyword route — the request is still answered.

It NEVER executes tasks, never calls agents/tools, never calls other
agents, and never writes `answer`/`sources`/`agent_answers` (MASTER_PLAN
§5.2: the plan node returns only plan data). Plan-driven execution is
Phase 8's dispatcher — explicitly out of scope here.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

# Module-level so tests can patch per-test (mirrors the supervisor's style).
from app.config import MAX_PLAN_TASKS, PLANNER_MODE, ROUTING_MODE
from app.state import AgentState

# Capability whitelist — defined in CODE (MASTER_PLAN §5.2), mirroring the
# dispatch mapping the Phase 8 executor will legalize in app/graph.py.
# Anything outside it ⇒ plan rejected deterministically.
CAPABILITIES = frozenset({"research", "books", "erp", "synthesis"})

# Satisfiable required_inputs: the user's question (always available) or the
# output of any capability some task in the plan will produce.
_SATISFIABLE_INPUTS = frozenset(CAPABILITIES) | {"question"}

_TASK_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

_COMPLETENESS_SIGNALS = re.compile(
    r"\b(research|stud(?:y|ies)|assess|evaluat(?:e|ion)|compar(?:e|ison)|"
    r"analy(?:s|z)e|then|after(?:wards)?|followed by|and also|additionally|"
    r"combine|both)\b",
    re.IGNORECASE,
)

_ROUTING_HINTS = re.compile(
    r"\b(erp|invoices?|inventory|stock|purchase orders?|sales orders?|"
    r"chapter|pages?|manual|books?)\b",
    re.IGNORECASE,
)


def needs_planning(state: AgentState) -> bool:
    """Phase 7 complex detection (prerequisite delivered by Phase 6).

    Decision order:
      1. PLANNER_MODE gates: "off" → never plan (graph identical to
         Phase ≤6); "always" → always plan (debug/tests only); "auto" →
         levels 2–3 below.
      2. Deterministic heuristic: multi-objective phrasing ( coordinating
         signals in a long-enough request ) means the existing single
         keyword route would collapse the objectives (MASTER_PLAN Phase 7
         "Why this phase exists"). Plain questions — the regression floor —
         answer False here, so existing routes never reach the planner.
      3. When ROUTING_MODE="llm" is already opted into (config), ask the
         LLM to double-check borderline cases (some signals but short, or
         long but 0–1 signals). Any LLM failure ⇒ deterministic result
         from level 2.

    Default builds never depend on a live provider.
    """
    if PLANNER_MODE == "off":
        return False
    if PLANNER_MODE == "always":
        return True

    question = str(state.get("question") or "").strip()
    if not question:
        return False

    signals = set(_COMPLETENESS_SIGNALS.findall(question))
    word_count = len(question.split())
    clear_complex = word_count >= 12 and len(signals) >= 2
    clear_simple = word_count < 8 or not signals

    if ROUTING_MODE == "llm" and not clear_complex and not clear_simple:
        llm_verdict = _llm_complexity(question)
        if llm_verdict is not None:
            return llm_verdict
        # LLM missing/malformed → deterministic heuristic decides.

    return clear_complex


def _llm_complexity(question: str) -> Optional[bool]:
    """Ask the LLM whether planning is needed; None on ANY failure.

    Fail-soft like the Phase 6 supervisor classifier: a broken provider can
    never crash the routing decision.
    """
    try:
        from app.llm_client import get_llm, llm_text

        response = get_llm(fast=True).invoke(
            "Classify whether this request NEEDS a multi-step plan (several "
            "distinct capabilities: e.g. web research AND a book library "
            "check AND a written study).\n"
            "Answer with ONLY one word: complex or simple.\n\n"
            f"Request: {question}\n"
            "Label:"
        )
        verdict = llm_text(response).strip().lower()
    except Exception:
        return None
    if verdict == "complex":
        return True
    if verdict == "simple":
        return False
    return None  # malformed label → deterministic fallback


def _task(task_id: str, capability: str, objective: str,
          depends_on: List[str], required_inputs: List[str],
          expected_output: str) -> dict:
    return {
        "task_id": task_id,
        "capability": capability,
        "objective": objective,
        "depends_on": list(depends_on),
        "required_inputs": list(required_inputs),
        "expected_output": expected_output,
        "done_when": f"{expected_output} is available in the workflow state",
    }


def default_plan(state: AgentState) -> dict:
    """Deterministic fallback plan derived from the CURRENT route.

    Used when the LLM plan is malformed/adversarial/oversized or planning
    itself fails. Mirrors what the graph already does on keyword routes
    today (including the feasibility chain research→books→feasibility), so
    fallback behavior is never worse than pre-Phase-7 — it is now available
    as structured plan data for Phase 8's dispatcher.
    """
    route = state.get("route", "research")
    if route == "feasibility":
        tasks = [
            _task("t1", "research",
                  "Gather web research for the feasibility study",
                  [], ["question"], "Web research summary"),
            _task("t2", "books",
                  "Check the user's book library for relevant material",
                  ["t1"], ["research"], "Relevant book extracts"),
            _task("t3", "synthesis",
                  "Assess feasibility combining research and books findings",
                  ["t1", "t2"], ["research", "books"],
                  "Feasibility study with recommendation"),
        ]
    elif route == "erp":
        tasks = [
            _task("t1", "erp", "Look up the requested ERP business data",
                  [], ["question"], "ERP data extract"),
            _task("t2", "synthesis", "Answer the question from the ERP data",
                  ["t1"], ["erp"], "Answer to the user's question"),
        ]
    elif route == "books":
        tasks = [
            _task("t1", "books", "Answer from the user's book library",
                  [], ["question"], "Book-based answer"),
            _task("t2", "synthesis", "Answer the question from the book "
                  "answer", ["t1"], ["books"], "Answer to the user's question"),
        ]
    else:
        # research — also the safe default for anything unrecognized.
        tasks = [
            _task("t1", "research", "Research the request on the web",
                  [], ["question"], "Web research summary"),
            _task("t2", "synthesis", "Answer the question from the research "
                  "summary", ["t1"], ["research"],
                  "Answer to the user's question"),
        ]
    return {
        "goal": str(state.get("question") or "").strip(),
        "tasks": tasks,
        "replan_count": 0,
        "source": "default",
    }


def parse_plan(raw_text: object) -> Optional[dict]:
    """Defensively extract a JSON plan object from provider-neutral LLM text.

    Handles fenced JSON, leading/trailing prose, and "Plan: [...]" prefixes.
    Returns None on ANY structural problem — semantic checking belongs to
    validate_plan(); this only decodes.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    text = raw_text.strip()

    candidates: List[str] = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    for candidate in candidates:
        try:
            plan = json.loads(candidate)
        except (json.JSONDecodeError, ValueError, RecursionError):
            continue
        if isinstance(plan, dict):
            return plan
    return None


def validate_plan(plan: object) -> Tuple[bool, List[str]]:
    """Deterministic validator (pure function, MASTER_PLAN §5.2).

    Returns (ok, reasons). Enforces, in order:
      * plan is a dict with a non-empty string `goal`;
      * `tasks` is a non-empty list of dicts;
      * task count ≤ MAX_PLAN_TASKS;
      * per task: task_id present, well-formed, unique; capability inside
        the code whitelist; depends_on is a list of previously seen
        task_ids (references exist); required_inputs satisfiable from the
        plan itself (a capability some task produces, or "question");
        objective/expected_output/done_when non-empty strings.
    A plan with ANY violation is rejected in full — the caller falls back
    to the deterministic default plan. The validator NEVER raises.
    """
    reasons: List[str] = []
    if not isinstance(plan, dict):
        return False, ["plan is not a JSON object"]

    goal = plan.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        reasons.append("goal must be a non-empty string")

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        reasons.append("tasks must be a non-empty list")
        return False, reasons

    if len(tasks) > MAX_PLAN_TASKS:
        reasons.append(
            f"plan has {len(tasks)} tasks; MAX_PLAN_TASKS={MAX_PLAN_TASKS}"
        )

    # A task's required_inputs must be producible INSIDE this plan: either
    # "question" (always available) or the output of a capability some
    # task in the plan performs (§5.2 "satisfiable from capabilities in
    # the plan").
    plan_capabilities = {
        t.get("capability") for t in tasks if isinstance(t, dict)
    }
    satisfiable = plan_capabilities | {"question"}

    ids: List[str] = []
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            reasons.append(f"tasks[{i}] is not an object")
            continue

        tid = task.get("task_id")
        if not isinstance(tid, str) or not _TASK_ID_RE.fullmatch(tid):
            reasons.append(f"tasks[{i}].task_id is missing or malformed")
        elif tid in ids:
            reasons.append(f"duplicate task_id: {tid}")
        else:
            ids.append(tid)

        cap = task.get("capability")
        if cap not in CAPABILITIES:
            reasons.append(
                f"tasks[{i}].capability {cap!r} is not a whitelisted "
                "capability"
            )

        deps = task.get("depends_on")
        if not isinstance(deps, list) or any(
            not isinstance(d, str) for d in deps
        ):
            reasons.append(
                f"tasks[{i}].depends_on must be a list of task_ids"
            )
        else:
            unknown = [d for d in deps if d not in ids]
            if unknown:
                reasons.append(
                    f"tasks[{i}].depends_on references unknown or "
                    "later task_id(s): " + ", ".join(unknown)
                )

        inputs = task.get("required_inputs")
        if not isinstance(inputs, list) or any(
            not isinstance(x, str) for x in inputs
        ):
            reasons.append(
                f"tasks[{i}].required_inputs must be a list of strings"
            )
        else:
            unsat = [x for x in inputs if x not in satisfiable]
            if unsat:
                reasons.append(
                    f"tasks[{i}].required_inputs unsatisfiable from the "
                    "plan's own capabilities: " + ", ".join(unsat)
                )

        for key in ("objective", "expected_output", "done_when"):
            if not isinstance(task.get(key), str) \
                    or not task.get(key, "").strip():
                reasons.append(
                    f"tasks[{i}].{key} must be a non-empty string"
                )

    if reasons:
        return False, reasons

    # Acyclicity: because deps may only reference previously seen ids
    # (enforced above), a forward/back edge would already have been
    # rejected as "unknown or later task_id". A self-dependency would be
    # caught the same way. Explicitly prove acyclicity anyway (DFS) so the
    # validator stays self-contained if the ordering rule loosens.
    dep_map: Dict[str, List[str]] = {
        t["task_id"]: list(t.get("depends_on") or [])
        for t in tasks if isinstance(t, dict) and isinstance(t.get("task_id"), str)
    }
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {tid: WHITE for tid in dep_map}

    def _has_cycle(tid: str) -> bool:
        color[tid] = GRAY
        for dep in dep_map.get(tid, []):
            if color.get(dep) == GRAY:
                return True
            if color.get(dep) == WHITE and _has_cycle(dep):
                return True
        color[tid] = BLACK
        return False

    for tid in dep_map:
        if color[tid] == WHITE and _has_cycle(tid):
            return False, ["depends_on forms a cycle"]

    return True, []


_PLANNER_PROMPT = """You are a task planner. Decompose the user request into
SMALL, ordered tasks. Each task is executed by ONE capability:
- research: general web research
- books: answering from the user's own book library
- erp: ERP business data (invoices, inventory, orders, customers)
- synthesis: combining prior task outputs into a final answer

Reply with ONLY a JSON object, no code fences, no prose, shaped exactly like:
{{"goal": "...", "tasks": [{{"task_id": "t1", "capability": "research",
"objective": "...", "depends_on": [], "required_inputs": ["question"],
"expected_output": "...", "done_when": "..."}}, ...]}}

Rules: task_id values t1, t2, ...; depends_on lists only EARLIER task_ids;
required_inputs lists "question" or a capability whose output an earlier
task produces; 2 to 5 tasks; the last task must have capability "synthesis".

User request: {question}"""


def planner_node(state: AgentState) -> dict:
    """Graph node: returns {"plan": plan} (plus nothing else) — Phase 7.

    Contract (MASTER_PLAN §5.2):
      * Asks get_llm(fast=True) for a plan; parses defensively; validates
        deterministically.
      * On LLM error, malformed JSON, or failed validation → deterministic
        default_plan(state) derived from the current route. The request is
        still answerable (Phase 8 execution / Phase 10 synthesis consume
        the plan; this node never does).
      * Never mutates answer/sources/agent_answers; never executes tasks;
        never calls agents.
    """
    from app.llm_client import get_llm, llm_text

    question = str(state.get("question") or "").strip()

    parsed: Optional[dict] = None
    plan_source = "default"
    try:
        response = get_llm(fast=True).invoke(
            _PLANNER_PROMPT.format(question=question)
        )
        parsed = parse_plan(llm_text(response))
    except Exception:
        parsed = None  # any provider failure → deterministic fallback

    if parsed is not None:
        candidate_plan = dict(parsed)
        # Accept both bare plans and {"plan": {...}} wrappers.
        if isinstance(candidate_plan.get("plan"), dict):
            candidate_plan = candidate_plan["plan"]
        ok, reasons = validate_plan(candidate_plan)
        if ok:
            candidate_plan.setdefault("replan_count", 0)
            candidate_plan["source"] = "llm"
            plan_source = "llm"
            plan = candidate_plan
        else:
            plan = default_plan(state)
            plan["source"] = "fallback"
            plan["fallback_reasons"] = reasons[:5]
    else:
        # No parsable LLM output (provider failure OR unparseable text).
        plan = default_plan(state)
        plan["source"] = "fallback"
        plan["fallback_reasons"] = [
            "planner LLM unavailable or returned no parsable plan"
        ]
    return {"plan": plan}
