"""Production QA node (Phase 5): deterministic workflow validation.

Moved from the Phase 4 test-suite copy (scripts/test_phase4_workflow_qa.py)
into production per docs/MASTER_PLAN.md Phase 5: qa_flags was declared in
AgentState but never written by production code.

Responsibilities (frozen by the 30 Phase 4 QA tests):
  * evaluates whether the completed workflow produced the required output
  * checks route-aware expectations
  * never decides routing / never invokes agents / never replaces the
    supervisor or the graph / never uses an LLM
"""

import re
from typing import List

from app.state import AgentState
from app.supervisor import APPROVAL_PENDING_MARKER, is_conversational

# Minimum result length before two completed steps may be called
# "indistinguishable": a short "OK"/"Noted" says nothing either way.
_MIN_COMPARABLE_RESULT = 40

# A result shorter than this is never called an echo of its instruction:
# "OK" can legitimately equal a one-word objective.
_MIN_ECHO_LENGTH = 10

_WHITESPACE_RE = re.compile(r"\s+")


def _normalized_result(result: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(result or "")).strip().lower()


def _plan_objective_findings(tasks: List[dict],
                             question: str = "") -> List[str]:
    """Deterministic checks on what the plan claims each step ACCOMPLISHED.

    Objective-level, not string-level: these are structural contradictions
    between a task's objective/status/result and the plan around it. Deliberately
    NO entity or substring matching — a step that names an entity differently
    ("the AI RMF" vs "the Risk Management Framework"), or explains it instead
    of repeating the words, can never fail a task here. When the evidence is
    not enough to judge, nothing is reported: uncertainty is not a finding.
    """
    findings: List[str] = []
    by_id = {t.get("task_id"): t for t in tasks}

    # (a) A dependent step completed while a prerequisite did not: the
    # dispatcher never runs such a task, so the plan contradicts itself.
    for task in tasks:
        if task.get("status") != "completed":
            continue
        unmet = [d for d in (task.get("depends_on") or [])
                 if (by_id.get(d) or {}).get("status") != "completed"]
        if unmet:
            findings.append(
                "QA: task(s) recorded completed although a prerequisite did "
                f"not complete: {task.get('task_id', '?')} (depends on "
                + ", ".join(str(d) for d in unmet) + ")"
            )

    # (a2) A step marked BLOCKED although every prerequisite completed: the
    # dispatcher only blocks a task it cannot run, so this plan contradicts
    # the execution that produced it (work left undone for no reason).
    for task in tasks:
        if task.get("status") != "blocked":
            continue
        deps = task.get("depends_on") or []
        if not deps or any((by_id.get(d) or {}).get("status") != "completed"
                           for d in deps):
            continue
        findings.append(
            "QA: task(s) recorded blocked although every prerequisite "
            f"completed: {task.get('task_id', '?')} — the work was never "
            "attempted"
        )

    # (b) The execution ended while a task was still in flight: its side
    # effects are unknown, so nothing may be claimed about it.
    running = [t.get("task_id", "?") for t in tasks
               if t.get("status") == "running"]
    if running:
        findings.append(
            "QA: task(s) were still running when the execution ended: "
            + ", ".join(running)
        )

    # (c) An approval REQUEST recorded as the output of a completed step: the
    # gate answered instead of the work, so that objective was not attempted.
    asked = [t.get("task_id", "?") for t in tasks
             if t.get("status") == "completed"
             and APPROVAL_PENDING_MARKER in str(t.get("result") or "")]
    if asked:
        findings.append(
            "QA: task(s) recorded an approval request as their result — the "
            "work was not performed: " + ", ".join(asked)
        )

    # (d) Two DIFFERENT capabilities returned indistinguishable output: a
    # POSSIBLE sign a step was skipped or the wrong source answered. Reported
    # as a warning, never as failure — one shared source can legitimately
    # produce the same answer for two objectives, which is why steps of the
    # SAME capability are never compared and short results are ignored.
    seen: dict = {}
    for task in tasks:
        if task.get("status") != "completed" or task.get("capability") == "synthesis":
            continue
        text = _normalized_result(task.get("result"))
        if len(text) < _MIN_COMPARABLE_RESULT:
            continue
        seen.setdefault(text, []).append(task)
    for group in seen.values():
        if len({t.get("capability") for t in group}) < 2:
            continue
        ids = ", ".join(str(t.get("task_id", "?")) for t in group)
        findings.append(
            f"QA: task(s) {ids} returned indistinguishable results for "
            "different objectives — one step may not have done its own work "
            "(this is a possibility, not proof; a shared source can legitimately "
            "answer both)"
        )

    # (e) A completed step whose result IS its own instruction (or the user's
    # request) did not answer anything (2026-10-29): the specialist echoed the
    # prompt back. Grounded entirely in recorded task state — no entity or
    # keyword matching, and only an exact normalized equality counts.
    request = _normalized_result(question)
    echoed = []
    for task in tasks:
        if task.get("status") != "completed":
            continue
        text = _normalized_result(task.get("result"))
        if len(text) < _MIN_ECHO_LENGTH:
            continue
        if text == _normalized_result(task.get("objective")) or (
                request and text == request):
            echoed.append(str(task.get("task_id", "?")))
    if echoed:
        findings.append(
            "QA: task(s) returned their own instruction as the result — the "
            "step echoed the question instead of doing work: "
            + ", ".join(echoed)
        )

    return findings


# --- Numerical grounding (2026-10-31 live audit) ---------------------------
# The general/feedback routes answer from model knowledge and consult no
# source, so a benchmark figure in their output cannot be checked by this
# process. The live audit had "Redis is 10x faster than Memcached",
# "can reduce load by 30-50%" and "NVMe offers 300-400 MB/s" all pass QA
# with zero flags. This check does NOT verify facts (it cannot): it makes the
# unfounded ones visible instead of silently counted as clean.
#
# Deliberately narrow, to keep the signal trustworthy: only percentages,
# multipliers ("10x", "3 times") and data rates ("400 MB/s") are considered.
# Plain counts and computed durations are NOT, because a correct sum
# ("40 ms + 60 ms = 100 ms") legitimately contains a number the question does
# not state, and flagging arithmetic would train the user to ignore QA.
# Units seen in the audit's fabricated claims: "30-50%", "10x faster",
# "300-400 MB/s", "35-40 Gbps". Link rates and data rates are included
# because a model asked for "a number" about hardware invents them.
_FIGURE_RE = re.compile(
    r"(?<![\w.])(\d{1,4}(?:[.,]\d+)?)\s?"
    r"(%|percent\b|x\b|times\b|[MGKT]B/s\b|[MGK]bps\b|Gb/s\b|IOPS\b)",
    re.IGNORECASE,
)
_UNSOURCED_ROUTES = ("general", "feedback")


def _figures(text: str) -> set:
    """Normalized figures mentioned in a piece of text."""
    out = set()
    for number, unit in _FIGURE_RE.findall(str(text or "")):
        value = number.replace(",", ".")
        if "." in value:  # trim decimals only; 300 stays 300, not "3"
            value = value.rstrip("0").rstrip(".") or "0"
        out.add(value + str(unit).lower().replace("percent", "%"))
    return out


# A figure the answer itself presents as an ESTIMATE is not an unsupported
# claim: the honesty requirement is that the reader can tell, not that the
# number is absent. Kept to explicit hedge words (an audit regression: "10x
# faster" and "300-400 MB/s" carried none of them).
_ESTIMATE_RE = re.compile(
    r"\b(approximately|approx\.?|roughly|estimated?|estimate|ballpark|"
    r"order of magnitude)\b",
    re.IGNORECASE,
)


def _unsourced_figures(state: dict) -> List[str]:
    """Figures in the answer that no input to this run supplies."""
    answer = str(state.get("answer") or "")
    if not answer.strip():
        return []
    supported = _figures(state.get("question"))
    prior = state.get("prior_execution")
    if isinstance(prior, dict):  # a revision may reuse its own earlier figures
        supported |= _figures(prior.get("answer"))
    # NOTE: `agent_answers` is deliberately NOT used as support: on these
    # routes the answer under review IS that entry, so including it would make
    # every figure self-supporting. Provenance strings are included because a
    # cited source that carries a figure is real support.
    for value in (state.get("sources") or []):
        supported |= _figures(value)
    # Evidence grounding (2026-11-01): a figure the ANSWERED-FROM passages
    # literally state is supported — that is what retrieval is for. Only the
    # evidence route counts: when the reply fell back to model knowledge, the
    # passages were not its source and must not bless its numbers.
    ev = state.get("evidence")
    if isinstance(ev, dict) and ev.get("answered_from") == "evidence":
        supported |= _figures(ev.get("text"))
    unsupported = _figures(answer) - supported
    if not unsupported:
        return []
    # Drop figures the answer openly labels as estimates (still visible in
    # the text, never presented as sourced).
    estimated = set()
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if _ESTIMATE_RE.search(sentence):
            estimated |= _figures(sentence)
    return sorted(unsupported - estimated)


# --- Claim-level evidence checks (2026-11-01) -----------------------------
# The general route now names which local passages it consulted (state["evidence"]).
# That makes three claim classes CHECKABLE for the first time, and this section
# checks exactly those three — nothing more:
#
#   1. licence claims   — a closed vocabulary of licence names, compared with
#                         the passages (the audit's Redis/Memcached defect);
#   2. citations        — a filename the run did not retrieve is fabricated;
#   3. attribution      — an answer built on passages but citing none, or one
#                         that points at "your documents" when nothing relevant
#                         was found.
#
# Deliberately NOT attempted: general factual correctness, version compatibility,
# security assertions. No deterministic check here can establish those, and a
# check that guessed would be worse than the honest "unverifiable" label. A
# single weak claim never fails the answer either: these are findings, not a
# verdict.

# Dots are part of a filename: "NIST.AI.100-1.pdf" must match WHOLE, not as
# "100-1.pdf" — the loose form produced a fabricated-citation finding against
# a correct, properly attributed answer (2026-11-01 live run).
_CITATION_FILE_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.\-]*\.(?:pdf|docx|txt|md|csv)\b", re.I)
_CLAIMS_DOCS_RE = re.compile(
    r"\byour (?:documents?|books?|files?|library|corpus)\b"
    r"|\bthe (?:passages?|excerpts?)\b",
    re.I,
)
# Words a licence sentence and a licence clause may share WITHOUT meaning the
# same subject. Excluded so "licensed under X" in the passage cannot alone
# make an unrelated licence claim look like a contradiction.
_LICENCE_WORDS = frozenset({
    "licen", "licens", "licenc", "license", "licenc", "softwar", "open",
    "sourc", "releas", "distribut", "availabl", "under", "term", "code",
    "free", "paid", "commercial",
})


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or ""))
            if s.strip()]


def _subject_tokens(text: str) -> set:
    """Content tokens of a sentence, minus licence vocabulary and numbers."""
    return {t for t in _evidence_support_tokens(text)
            if not t.isdigit() and t not in _LICENCE_WORDS and len(t) > 3}


def _evidence_support_tokens(text: str) -> set:
    """The books tokenizer, imported lazily so QA stays dependency-light."""
    from app.agents.books_grounding import _support_tokens

    return _support_tokens(text)


def _licence_findings(answer: str, ev_text: str,
                      from_evidence: bool) -> List[str]:
    """Compare the answer's licence claims with the consulted passages."""
    from app.evidence import licence_values

    claimed = licence_values(answer)
    if not claimed:
        return []
    named = ", ".join(sorted(claimed))
    found = licence_values(ev_text)
    if not from_evidence or not str(ev_text or "").strip():
        return [
            "QA: the answer states licence(s) " + named + " but no source "
            "was consulted for them — unverified"]
    # A sentence is CONTRADICTED only when no passage clause states the same
    # licence about the same subject and at least one states a different one.
    # The supporting half of this rule matters: conflicting documents are
    # already disclosed by the node, so an answer corroborated by one passage
    # is not "contradicted" merely because another passage disagrees.
    contradicted = []
    for sentence in _split_sentences(answer):
        mine = licence_values(sentence)
        if not mine:
            continue
        subject = _subject_tokens(sentence)
        if not subject:
            continue
        support, against = False, []
        for clause in re.split(r"(?<=[.;])\s+", ev_text):
            theirs = licence_values(clause)
            if not theirs or not (subject & _subject_tokens(clause)):
                continue
            if theirs == mine:
                support = True
            else:
                against.append(sorted(theirs)[0])
        if against and not support:
            contradicted.append(sorted(mine)[0] + " vs "
                                + ", ".join(sorted(set(against))))
    if contradicted:
        return [
            "QA: the answer's licence claim conflicts with the consulted "
            "passages (" + "; ".join(sorted(set(contradicted))) + ") — "
            "treat it as contradicted, not confirmed"]
    if claimed.isdisjoint(found):
        return [
            "QA: the answer states licence(s) " + named + " but the "
            "consulted passages never name that licence — unsupported"]
    return []


def _claim_verification_findings(ev: dict) -> List[str]:
    """Report claim-to-passage verdicts recorded by the `general` node.

    The node verifies each claim against the passage it CITES (app.claim_
    check); this gate only reports what is not clean. A single weak claim
    never fails the execution — it is named so a reader knows which sentence
    to distrust, and the supported ones are left alone.
    """
    checks = ev.get("claim_checks")
    if not isinstance(checks, dict):
        return []
    claims = [c for c in (checks.get("claims") or []) if isinstance(c, dict)]
    if not claims:
        return []
    out: List[str] = []

    def _named(status: str, limit: int = 2) -> List[str]:
        rows = [c for c in claims if c.get("status") == status]
        return [f"\"{str(c.get('text'))[:110]}\" ({c.get('reason')})"
                for c in rows[:limit]]

    contradicted = _named("contradicted")
    if contradicted:
        out.append(
            "QA: claim(s) contradicted by the passage they cite: "
            + "; ".join(contradicted))
    unsupported = _named("unsupported")
    if unsupported:
        out.append(
            "QA: claim(s) not supported by the passage they cite: "
            + "; ".join(unsupported))
    partial = _named("partial")
    if partial:
        out.append(
            "QA: claim(s) only partly covered by the passage they cite: "
            + "; ".join(partial))
    return out


def _evidence_findings(state: dict) -> List[str]:
    """Claim-level findings that need the evidence record to judge."""
    ev = state.get("evidence")
    if not isinstance(ev, dict):
        return []
    answer = str(state.get("answer") or "")
    if not answer.strip():
        return []
    from_evidence = ev.get("answered_from") == "evidence"
    files = {str(f).lower() for f in (ev.get("files") or [])}
    flags: List[str] = []

    # (1) Citations: a source name this run did not retrieve was invented.
    cited = {m.group(0).lower() for m in _CITATION_FILE_RE.finditer(answer)}
    foreign = sorted(cited - files)
    if foreign:
        flags.append(
            "QA: the answer cites file(s) this run did not retrieve: "
            + ", ".join(foreign) + " — the attribution is fabricated")

    # (2) Attribution: an evidence-built answer with no evidence provenance,
    # or a memory answer that points at documents it never opened.
    if from_evidence and not any(str(s).startswith("evidence:")
                                 for s in (state.get("sources") or [])):
        flags.append(
            "QA: the answer was built from local passages but cites none — "
            "it cannot be attributed to a source")
    if (not from_evidence
            and _CLAIMS_DOCS_RE.search(answer)
            and ev.get("status") in ("none", "error", None)):
        flags.append(
            "QA: the answer refers to local documents as its source, but no "
            "relevant passage was found or used")

    # (3) Licence claims.
    flags.extend(_licence_findings(answer, str(ev.get("text") or ""),
                                   from_evidence))
    if from_evidence:
        flags.extend(_claim_verification_findings(ev))
    return flags


def run_qa(state: dict) -> List[str]:
    """Deterministic QA. Takes the completed workflow output and flags issues."""
    flags = []
    route = state.get("route", "research")
    sources = state.get("sources") or []
    answers = state.get("agent_answers") or {}

    if route in ("feasibility", "multi"):
        # Feasibility and multi-agent routes require research + books.
        has_research = "research" in sources or "research" in answers
        has_books = "books" in sources or "books" in answers
        if not has_research:
            flags.append("QA: feasibility route missing required research output")
        if not has_books:
            flags.append("QA: feasibility route missing required books output")

        missing = [name for name in ("research", "books") if name not in answers]
        if missing:
            flags.append(
                "QA: feasibility route incomplete — missing upstream agent output(s): "
                + ", ".join(missing)
            )
    elif route == "erp":
        # ERP route must NOT be forced to need research/books.
        if "erp" not in sources:
            flags.append("QA: ERP route missing required erp provenance")

    # 2026-10-27 (greeting misrouting, recorded in execution 3fc3bc8460c9):
    # a BARE conversational opener may only be answered by the
    # conversational route. Plausible prose plus a source list is not a
    # reply to "hello" — that is exactly the irrelevant web summary that
    # passed QA here. Deterministic full-string opener match (see
    # app.supervisor.is_conversational); substantive questions never match,
    # so no ordinary research/books answer is affected.
    if (route != "conversational"
            and is_conversational(str(state.get("question") or ""))):
        flags.append(
            "QA: conversational greeting answered by the "
            f"{route or 'unknown'} route — a sourced agent summary is not "
            "a reply to a greeting"
        )

    if not str(state.get("answer") or "").strip():
        flags.append("QA: final answer is empty — workflow produced no output")

    if route in _UNSOURCED_ROUTES:
        unsourced = _unsourced_figures(state)
        if unsourced:
            flags.append(
                "QA: the answer states figure(s) this run cannot support "
                f"(no consulted source states them): {', '.join(unsourced)} "
                "— treat them as unverified"
            )

    if route == "general":
        flags.extend(_evidence_findings(state))

    # Workspace coding (2026-10-30): report what the node ACTUALLY did, and
    # never let "patch prepared" be read as "patch applied". A prepared patch
    # must carry the approval request that binds it; a blocked preparation
    # must keep saying that nothing was written.
    coding = state.get("coding")
    if isinstance(coding, dict):
        status = str(coding.get("status") or "")
        if status == "patch_prepared" and not str(
                coding.get("approval_id") or "").strip():
            flags.append(
                "QA: a workspace patch was prepared but no approval request "
                "binds it — it must not be treated as applicable")
        if status == "patch_prepared":
            failed_checks = [
                v.get("path") for v in (coding.get("validation") or [])
                if isinstance(v, dict) and v.get("status") == "failed"
            ]
            if failed_checks:
                flags.append(
                    "QA: a patch was offered for approval although offline "
                    "validation failed for: " + ", ".join(
                        str(p) for p in failed_checks))
    elif route == "coding":
        flags.append(
            "QA: the workspace coding route produced no coding result")

    # Dataset analysis (2026-10-30): the analyst computes its figures in
    # app/data/analysis.py, so there is never a "number" to distrust here.
    # What QA CAN prove is that the route produced an analysis at all, and
    # that it did not answer a group/filter question with only the profile
    # (which would look like a complete answer while the requested
    # computation never ran). Both are warnings: they never turn a run that
    # finished its work into a failure.
    if route == "data":
        data_answer = str((answers or {}).get("data") or "")
        if not data_answer.strip():
            flags.append(
                "QA: the data route produced no analysis result")
        elif "Computed results" not in data_answer and not any(
                marker in data_answer for marker in
                # A refusal is not a missing analysis: the node says why it
                # could not compute (no workspace, no dataset named, no rows).
                ("is not enabled", "No analysis was performed",
                 "Name the dataset file")):
            flags.append(
                "QA: the data route answered without a computed-results "
                "section — no figure can be attributed to a calculation")
        elif "was NOT run" in data_answer:
            flags.append(
                "QA: part of the requested dataset analysis was refused — "
                "the answer reports it rather than approximating")

    # Contextual follow-up (2026-10-30): the route may ONLY answer from a
    # recorded execution. QA can prove two things deterministically: the
    # route produced an answer at all, and an answer that claims an earlier
    # result names the record it used (sources "execution:<id>"). An answer
    # with no record is the fabrication this capability exists to prevent.
    if route == "followup":
        followup_answer = str((answers or {}).get("followup") or "")
        cited = [s for s in sources if str(s).startswith("execution:")]
        if not followup_answer.strip():
            flags.append("QA: the follow-up route produced no answer")
        elif state.get("prior_execution") and not cited:
            flags.append(
                "QA: the follow-up answer names no recorded execution — it "
                "cannot be attributed to earlier work")
        elif not state.get("prior_execution") and cited:
            flags.append(
                "QA: a follow-up answer cites an execution record that was "
                "not supplied to the run")
        # No prior record AND no citation is the honest "nothing recorded"
        # answer: not a defect, so it is deliberately not flagged.

    # Continuous learning (2026-10-30): the feedback route may claim only what
    # it actually recorded. `state["learning"]["learned"]` is written by the
    # node from the store's response, so a claim made while nothing was stored
    # is a provable contradiction — the same class of check as a completed task
    # with no result. Applied-lesson evidence is checked the other way: a node
    # reports a lesson it inserted, and that lesson must be one the run was
    # given, never one it invented.
    if route == "feedback":
        learning = state.get("learning")
        learning = learning if isinstance(learning, dict) else {}
        feedback_answer = str((answers or {}).get("feedback") or "")
        claim_re = re.compile(
            r"\b(i'?ve (recorded|learned|remembered)|i'?ll remember|"
            r"recorded (that|this|the) correction|noted for (next|future))\b",
            re.I)
        if not feedback_answer.strip():
            flags.append("QA: the feedback route produced no answer")
        elif claim_re.search(feedback_answer) and not learning.get("learned"):
            flags.append(
                "QA: the reply claims the correction was recorded, but nothing "
                "was stored — it must not be relied on for future tasks"
            )
        if learning.get("lesson_id") and not any(
                str(s) == f"learning:{learning['lesson_id']}" for s in sources):
            flags.append(
                "QA: a lesson was stored without being attributed to its "
                "source execution"
            )

    applied = state.get("lessons_applied")
    if isinstance(applied, list) and applied:
        supplied = {
            str(l.get("lesson_id"))
            for l in (state.get("lessons") or [])
            if isinstance(l, dict)
        }
        invented = [str(i) for i in applied if str(i) not in supplied]
        if invented:
            flags.append(
                "QA: a node reported applying lesson(s) the run was never "
                "given: " + ", ".join(invented)
            )

    # Phase 8 (MASTER_PLAN §5.3 / §5.5): when a dispatched plan exists,
    # QA verifies plan completion honestly — any task that did NOT complete
    # successfully is flagged (failed / blocked / unsupported / cap-exhausted).
    plan = state.get("plan")
    if isinstance(plan, dict) and isinstance(plan.get("tasks"), list):
        # Planner fell back because the AI service is DOWN: surface that as
        # its own clear flag (user-requested server-down UX) — distinct from
        # task-level failures so "why is my answer degraded?" is answerable.
        reasons = " | ".join(str(r) for r in plan.get("fallback_reasons") or [])
        if plan.get("source") == "fallback" and "unavailable" in reasons:
            flags.append(
                "QA: AI planning service unavailable — answered with the "
                "built-in fallback plan"
            )
        failed = [
            t.get("task_id", "?")
            for t in plan["tasks"]
            if isinstance(t, dict) and t.get("status") not in (None, "completed")
        ]
        if failed:
            flags.append(
                "QA: plan incomplete — task(s) not completed: " + ", ".join(failed)
            )
        # 2026-10-29: a task marked completed must carry the output it
        # completed WITH. A completed task whose recorded result is missing or
        # blank is a contradiction between the plan and the work — the run
        # cannot claim that step succeeded, and its dependents consumed
        # nothing. Checked on the PLAN (bookkeeping), never inferred from the
        # answer text.
        empty = [
            t.get("task_id", "?")
            for t in plan["tasks"]
            if isinstance(t, dict) and t.get("status") == "completed"
            and not (isinstance(t.get("result"), str) and t["result"].strip())
        ]
        if empty:
            flags.append(
                "QA: plan records completed task(s) with no result: "
                + ", ".join(empty)
            )
        flags.extend(
            _plan_objective_findings(
                [t for t in plan["tasks"] if isinstance(t, dict)],
                str(state.get("question") or ""),
            )
        )

    # Phase 8 records incomplete plans in qa_flags, but the runner previously
    # marked every returned graph state "completed". That made a task marked
    # failed/blocked look successful to the API and UI. The diagnostic contract
    # stays unchanged; add a separate predicate so persistence can distinguish
    # an incomplete dispatched plan without treating ordinary QA notices as a
    # failed execution.
    return flags


def has_incomplete_plan(state: dict) -> bool:
    """Whether a dispatched plan has any task that did not complete."""
    plan = state.get("plan")
    tasks = plan.get("tasks") if isinstance(plan, dict) else None
    return isinstance(tasks, list) and any(
        isinstance(task, dict) and task.get("status") != "completed"
        for task in tasks
    )


def qa_node(state: AgentState) -> dict:
    """Graph node wrapper: append QA findings to state["qa_flags"].

    Merges with any existing flags (never clears accumulated ones), returns
    only the key it changes, and is fail-soft: a QA crash can never fail a
    run — it appends a QA error note instead (MASTER_PLAN Phase 5 failure
    case).
    """
    flags = state.get("qa_flags") or []
    try:
        findings = run_qa(state)
    except Exception as exc:  # fail-soft by design
        findings = [f"QA: internal error while validating output ({exc})"]
    return {"qa_flags": list(flags) + list(findings)}
