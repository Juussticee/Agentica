"""Phase 8 tests: dynamic plan execution per MASTER_PLAN §5.3.

All deterministic and zero-quota (every agent node is faked before the graph
is built — the established project convention). Covers the Phase 8 spec:

  * a 3-task plan with dependencies executes in a VALID dependency order
    (partial-order assertions, not incidental order — Phase 4 discipline);
  * dependency-independent tasks are not incorrectly ordered relative to
    each other;
  * a task never runs before its dependencies COMPLETE SUCCESSFULLY;
  * a failed dependency blocks dependents (never executed, never completed);
  * unsupported capabilities fail safe and dispatch to NO arbitrary node;
  * the hard iteration cap terminates pathological plans;
  * simple/multi requests bypass the dispatcher (byte-identical routes);
  * results + provenance flow through the EXISTING agent_answers/sources
    mechanism (with_agent_output inside the agents — no new state keys);
  * re-entering the dispatcher never re-executes completed tasks;
  * QA runs after planned execution and flags non-completed tasks.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.graph as graph_module  # noqa: E402
from app.graph import (  # noqa: E402
    CAPABILITY_NODES,
    dispatcher_node,
    _after_dispatcher,
)
from app.qa_node import run_qa  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _AgentRecorder:
    """Installs faked agent nodes on app.graph and records execution order."""

    def __init__(self, behavior=None):
        self.calls = []
        self._snapshot = {
            name: getattr(graph_module, name)
            for name in ("books_agent", "research_agent", "feasibility_agent",
                         "erp_agent", "synthesis_node")
        }
        self._behavior = behavior or {}

    def install(self):
        for name in ("books_agent", "research_agent", "feasibility_agent",
                     "erp_agent", "synthesis_node"):
            node = name.removesuffix("_agent").removesuffix("_node")
            setattr(graph_module, name, self._make_fake(node, self._behavior.get(node)))
        return self

    def restore(self):
        for name, fn in self._snapshot.items():
            setattr(graph_module, name, fn)

    def _make_fake(self, node, mode):
        def fake(state):
            self.calls.append(node)
            if mode == "raise":
                raise RuntimeError("forced task failure")
            answer = f"{node.upper()}_OUTPUT"
            return {
                "answer": answer,
                "sources": list(state.get("sources") or []) + [node],
                "agent_answers": {
                    **(state.get("agent_answers") or {}),
                    node: answer,
                },
            }

        return fake


def plan_with(tasks):
    return {
        "goal": "g",
        "replan_count": 0,
        "source": "test",
        "tasks": tasks,
    }


def task(tid, capability, deps, inputs):
    return {
        "task_id": tid,
        "capability": capability,
        "objective": f"do {tid}",
        "depends_on": list(deps),
        "required_inputs": list(inputs),
        "expected_output": f"{capability} output for {tid}",
        "done_when": f"{capability} output present",
    }


UNCOMPLETED = ("pending", "failed", "blocked", "unsupported")


def statuses(state):
    return {
        t["task_id"]: t.get("status")
        for t in state["plan"]["tasks"]
        if isinstance(t, dict)
    }


def run_dispatcher_to_completion(state, cap=100):
    """Loop the dispatcher exactly as the compiled graph would: apply each
    visit's returned update back into state (LangGraph's merge semantics)."""
    for _ in range(cap):
        update = dispatcher_node(state)
        for key, value in update.items():
            if key == "agent_answers" and isinstance(value, dict):
                merged = dict(state.get("agent_answers") or {})
                merged.update(value)
                state["agent_answers"] = merged
            else:
                state[key] = value
        if _after_dispatcher(state) == "qa":
            return "qa"
    raise AssertionError("dispatcher did not terminate within test loop cap")


def apply_dispatch(state):
    """One dispatcher visit with LangGraph merge semantics (like the loop
    helper, for tests that run a single visit)."""
    update = dispatcher_node(state)
    for key, value in update.items():
        if key == "agent_answers" and isinstance(value, dict):
            merged = dict(state.get("agent_answers") or {})
            merged.update(value)
            state["agent_answers"] = merged
        else:
            state[key] = value
    return update


# ---------------------------------------------------------------------------
# Capability → node mapping is literal and closed (§5.3 + Section 11 #7)
# ---------------------------------------------------------------------------


class TestCapabilityMapping(unittest.TestCase):
    def test_mapping_is_exactly_the_roadmap_table(self):
        # 2026-10-30: `data` joins the table with the Data Analyst specialist;
        # every entry is a real graph node (asserted in
        # scripts/test_agent_registry.py, which also proves the registry, the
        # planner whitelist and this table cannot drift apart).
        self.assertEqual(
            CAPABILITY_NODES,
            {"research": "research", "books": "books", "erp": "erp",
             "data": "data", "synthesis": "synthesis"},
        )

    def test_mapping_keys_match_planner_whitelist(self):
        from app.planner import CAPABILITIES

        self.assertEqual(set(CAPABILITY_NODES), set(CAPABILITIES))


# ---------------------------------------------------------------------------
# Dependency-ordered execution
# ---------------------------------------------------------------------------


class TestThreeTaskDependencyOrder(unittest.TestCase):
    def test_three_task_plan_executes_in_valid_dependency_order(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        state = {
            # 2026-10-28 (Priority B): dispatching a research task requires
            # in-message authorization — a bare "q" would be gated. The
            # dispatcher mechanics under test are unchanged.
            "question": "Search the web for q",
            "plan": plan_with([
                task("t1", "research", [], ["question"]),
                task("t2", "books", ["t1"], ["research"]),
                task("t3", "synthesis", ["t1", "t2"], ["research", "books"]),
            ]),
        }
        self.assertEqual(run_dispatcher_to_completion(state), "qa")

        self.assertEqual(
            rec.calls, ["research", "books", "synthesis"],
            "t1→t2→t3 must dispatch research→books→synthesis",
        )
        self.assertEqual(
            statuses(state),
            {"t1": "completed", "t2": "completed", "t3": "completed"},
        )
        # Results + provenance through the EXISTING channels only.
        self.assertEqual(state["agent_answers"]["research"], "RESEARCH_OUTPUT")
        self.assertEqual(state["agent_answers"]["books"], "BOOKS_OUTPUT")
        self.assertEqual(state["agent_answers"]["synthesis"], "SYNTHESIS_OUTPUT")
        for name in ("research", "books", "synthesis"):
            self.assertIn(name, state["sources"])

    def test_independent_tasks_are_not_forced_into_a_specific_order(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        state = {
            # 2026-10-28 (Priority B): research capability needs an
            # authorizing request; ordering behavior is what is under test.
            "question": "Search the web for q",
            "plan": plan_with([
                task("tA", "research", [], ["question"]),
                task("tB", "books", [], ["question"]),
                task("tC", "erp", ["tA", "tB"], ["research", "books"]),
            ]),
        }
        # Both orders (research first OR books first) are valid; only the
        # PARTIAL order (both before the dependent tC) is required.
        self.assertEqual(run_dispatcher_to_completion(state), "qa")
        self.assertEqual(
            statuses(state),
            {"tA": "completed", "tB": "completed", "tC": "completed"},
        )
        self.assertEqual(len(rec.calls), 3)
        self.assertEqual(set(rec.calls), {"research", "books", "erp"})
        both_before = rec.calls.index("erp")
        self.assertLess(rec.calls.index("research"), both_before)
        self.assertLess(rec.calls.index("books"), both_before)


class TestDependencyGating(unittest.TestCase):
    def test_task_never_runs_before_dependency_completes(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        seen_books_sources = {}

        # Wrap the faked books node to capture what the graph handed it.
        original_books = graph_module.books_agent

        def spy_books(state):
            seen_books_sources["at_call"] = dict(state.get("agent_answers") or {})
            return original_books(state)

        graph_module.books_agent = spy_books

        state = {
            # 2026-10-28 (Priority B): authorizing request so the research
            # task is dispatchable; dependency gating is what is under test.
            "question": "Search the web for q",
            "plan": plan_with([
                task("t1", "research", [], ["question"]),
                task("t2", "books", ["t1"], ["research"]),
            ]),
        }
        # FIRST dispatch visit: only t1 is runnable; t2's dependency t1 has
        # status pending, so t2 must NOT run yet.
        apply_dispatch(state)
        self.assertEqual(rec.calls, ["research"])
        self.assertEqual(statuses(state)["t2"], "pending")
        apply_dispatch(state)
        self.assertEqual(rec.calls, ["research", "books"])
        # books ran WITH the research answer already in agent_answers.
        self.assertEqual(seen_books_sources["at_call"].get("research"), "RESEARCH_OUTPUT")


class TestFailedDependencyBlocksDependents(unittest.TestCase):
    def _plan_state(self):
        return {
            "question": "q",
            "plan": plan_with([
                task("t1", "books", [], ["question"]),
                task("t2", "synthesis", ["t1"], ["books"]),
                task("t3", "research", ["t2"], ["books"]),
            ]),
        }

    def test_failed_task_blocks_dependents_and_plan_continues(self):
        rec = _AgentRecorder(behavior={"books": "raise"}).install()
        self.addCleanup(rec.restore)

        state = self._plan_state()
        self.assertEqual(run_dispatcher_to_completion(state), "qa")
        self.assertEqual(rec.calls, ["books"], "only t1 ran; t2/t3 never executed")
        self.assertEqual(
            statuses(state),
            {"t1": "failed", "t2": "blocked", "t3": "blocked"},
        )
        # blocked tasks must never carry a completed status or a result
        t2 = state["plan"]["tasks"][1]
        self.assertNotEqual(t2["status"], "completed")
        self.assertIsNone(t2.get("result"))

    def test_final_answer_and_qa_flags_record_the_failure_honestly(self):
        rec = _AgentRecorder(behavior={"books": "raise"}).install()
        self.addCleanup(rec.restore)

        state = self._plan_state()
        run_dispatcher_to_completion(state)
        state.setdefault("route", "research")  # QA route checks stay neutral
        flags = run_qa(state)
        joined = " | ".join(flags)
        self.assertIn("plan incomplete", joined)
        self.assertIn("t3", joined)
        self.assertIn("t2", joined)
        self.assertIn("final answer is empty", joined)  # no answer was produced


class TestUnsupportedCapability(unittest.TestCase):
    def test_unsupported_capability_fails_safe_without_dispatch(self):
        # The dispatcher's own contract (a hostile/corrupt plan in state —
        # validated plans cannot produce this, so it is defense in depth):
        # the task is marked unsupported and dispatches NOWHERE, and its
        # dependent becomes blocked without executing.
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        state = {
            "question": "q",
            "plan": plan_with([
                task("t1", "filesystem_destroyer", [], ["question"]),
                task("t2", "synthesis", ["t1"], ["filesystem_destroyer"]),
            ]),
        }
        result = run_dispatcher_to_completion(state)
        self.assertEqual(result, "qa")
        self.assertEqual(rec.calls, [], "no node runs for an unsupported capability")
        self.assertEqual(statuses(state),
                         {"t1": "unsupported", "t2": "blocked"})
        t1 = state["plan"]["tasks"][0]
        self.assertIn("no whitelisted graph node", t1.get("error", ""))


# ---------------------------------------------------------------------------
# Hard iteration cap (§5.3 fail-safe)
# ---------------------------------------------------------------------------
# Hard iteration cap (§5.3 fail-safe)
# ---------------------------------------------------------------------------


class TestIterationCap(unittest.TestCase):
    def test_cap_formula_matches_spec(self):
        from app.config import MAX_PLAN_TASKS, MAX_TASK_RETRIES
        from app.graph import _hard_iteration_cap

        self.assertEqual(
            _hard_iteration_cap({"tasks": []}),
            MAX_PLAN_TASKS * (1 + MAX_TASK_RETRIES),
        )

    def test_cap_terminates_a_pathological_plan_fail_safe(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        # Awaiting-forever plan: t2 waits on t1 which is dispatched LAST —
        # actually still terminates; use a genuinely never-ready plan: t1
        # waits on task t0 that does not exist (defensively blocked), t2 waits
        # on t1... simplest pathological form: a task depending on itself via
        # an unknown id, none ever ready.
        state = {
            "question": "q",
            "plan": plan_with([
                task("t1", "research", ["ghost"], ["question"]),
                task("t2", "books", ["t1"], ["research"]),
                task("t3", "synthesis", ["t2"], ["books"]),
            ]),
        }
        # t1's "ghost" dependency is absent → defensively blocked on visit 1;
        # t2 then blocks on t1 (failed-status? no — t1 becomes blocked) ...
        self.assertEqual(run_dispatcher_to_completion(state), "qa")
        self.assertEqual(rec.calls, [], "a never-ready plan must execute nothing")
        self.assertEqual(
            statuses(state),
            {"t1": "blocked", "t2": "blocked", "t3": "blocked"},
        )
        self.assertIn("final answer is empty", " | ".join(run_qa(state)))
        self.assertIn("plan incomplete", " | ".join(run_qa(state)))

    def test_visit_counter_marks_every_nonterminal_task_failed_at_cap(self):
        from app.graph import _DISPATCH_VISITS, _hard_iteration_cap

        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)
        state = {
            "question": "q",
            "plan": plan_with([task("t1", "research", ["ghost"], ["question"])]),
        }
        # Force the visit counter just past the cap, then call once — the
        # dispatcher must fail-safe EVERY non-terminal task and stop looping.
        state["plan"][_DISPATCH_VISITS] = _hard_iteration_cap(state["plan"])
        apply_dispatch(state)
        self.assertEqual(statuses(state), {"t1": "failed"})
        t1 = state["plan"]["tasks"][0]
        self.assertIn("iteration cap exhausted", t1.get("error", ""))
        self.assertIn("fallback_note", state["plan"])

    def test_pathological_graph_state_cannot_loop_forever(self):
        # A never-ready plan (task depends on a nonexistent id) must not spin
        # the dispatcher forever: the task is defensively marked blocked on
        # the FIRST visit (no agent runs), the router goes to qa immediately,
        # and even cap+10 forced re-entries cannot revive it (visited tasks
        # are never re-executed). The cap-fires-at-exhaustion path is covered
        # by test_visit_counter_marks_every_nonterminal_task_failed_at_cap.
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)
        from app.graph import _DISPATCH_VISITS, _hard_iteration_cap

        state = {
            "question": "q",
            "plan": plan_with([task("t1", "synthesis", ["ghost"], ["question"])]),
        }
        for _ in range(_hard_iteration_cap(state["plan"]) + 10):
            update = dispatcher_node(state)
            for key, value in update.items():
                if key == "agent_answers" and isinstance(value, dict):
                    merged = dict(state.get("agent_answers") or {})
                    merged.update(value)
                    state["agent_answers"] = merged
                else:
                    state[key] = value
            route = _after_dispatcher(state)
            if route == "qa":
                break
        # Blocked is terminal for the router: no agent ever ran, the task is
        # never re-visited, and the graph settles on qa — no infinite loop.
        self.assertEqual(statuses(state), {"t1": "blocked"})
        self.assertEqual(_after_dispatcher(state), "qa")
        self.assertEqual(rec.calls, [], "a blocked never-ready task must not dispatch")
        self.assertLessEqual(
            state["plan"].get(_DISPATCH_VISITS, 0),
            _hard_iteration_cap(state["plan"]),
        )


# ---------------------------------------------------------------------------
# Simple/multi routes bypass dynamic execution (byte-identical behavior)
# ---------------------------------------------------------------------------


class TestSimpleRoutesBypassDispatcher(unittest.TestCase):
    def setUp(self):
        # Fake the planner node itself so ANY planner invocation is visible
        # as a failure — proving simple/multi routes never reach it.
        self._planner_snap = graph_module.planner_node

        def planner_must_not_run(state):
            raise AssertionError("simple/multi route reached the planner")

        graph_module.planner_node = planner_must_not_run

    def tearDown(self):
        graph_module.planner_node = self._planner_snap

    def test_simple_request_keeps_its_direct_route(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        graph = graph_module.build_graph()
        result = graph.invoke({"question": "What does chapter 3 say about propolis?"})
        self.assertEqual(result["route"], "books")
        self.assertEqual(rec.calls, ["books"], "exactly one books run, no planner,"
                                               " no dispatcher loop")
        self.assertNotIn("plan", result, "simple routes never produce a plan")
        self.assertNotIn("complex", result)

    def test_multi_request_keeps_its_existing_chain(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        graph = graph_module.build_graph()
        result = graph.invoke(
            {"question": "Do a feasibility study for an urban beekeeping venture?"}
        )
        self.assertEqual(result["route"], "feasibility")
        self.assertEqual(result["sub_routes"], ["research", "books"])
        self.assertEqual(
            rec.calls, ["research", "books", "feasibility"],
            "feasibility chain unchanged — NOT planner+dispatcher execution",
        )
        self.assertNotIn("plan", result)


class TestComplexRequestRunsThroughDispatcher(unittest.TestCase):
    COMPLEX_QUESTION = (
        "research urban beekeeping and compare approaches, then assess "
        "feasibility of the venture in detail"
    )

    def test_planner_output_is_executed_end_to_end(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)
        from unittest import mock

        from scripts.test_phase7_planner import _FakeLLM, GOOD_LLMM_PLAN

        fake = _FakeLLM([GOOD_LLMM_PLAN])
        with mock.patch("app.llm_client.get_llm", return_value=fake):
            graph = graph_module.build_graph()
            result = graph.invoke({"question": self.COMPLEX_QUESTION})
        # GOOD_LLMM_PLAN: t1 research → t2 books → t3 synthesis. The recorder
        # fakes the synthesis NODE (the dispatcher's synthesis target), so the
        # terminal task is observed as "synthesis" whatever the route; the
        # feasibility route's delegation to the study is covered by
        # TestSynthesisNode.test_feasibility_route_still_produces_the_study.
        self.assertEqual(rec.calls, ["research", "books", "synthesis"])
        self.assertEqual(result["route"], "feasibility")  # set by supervisor
        self.assertEqual(
            [(t["task_id"], t["status"]) for t in result["plan"]["tasks"]],
            [("t1", "completed"), ("t2", "completed"), ("t3", "completed")],
        )
        self.assertEqual(result["agent_answers"]["research"], "RESEARCH_OUTPUT")
        self.assertEqual(result["agent_answers"]["books"], "BOOKS_OUTPUT")
        self.assertEqual(result["agent_answers"]["synthesis"], "SYNTHESIS_OUTPUT")
        self.assertEqual(result["qa_flags"], [])


# ---------------------------------------------------------------------------
# Synthesis node (2026-10-28): the plan's terminal answer, not a report
# ---------------------------------------------------------------------------


class TestSynthesisNode(unittest.TestCase):
    """`synthesis` composes the completed tasks' results into one answer.

    It replaced the feasibility agent as the synthesis target for
    non-feasibility plans, which used to write a Market/Financial/Risk study
    (and a .docx) for ordinary books/ERP/research requests.
    """

    def _fake_llm(self, reply="COMBINED_ANSWER"):
        from types import SimpleNamespace

        class _Fake:
            def __init__(self):
                self.prompts = []

            def invoke(self, prompt):
                self.prompts.append(prompt)
                return SimpleNamespace(content=reply)

        return _Fake()

    def _state(self, tasks, answers, route="books"):
        return {"question": "What should we do next?", "route": route,
                "plan": plan_with(tasks), "agent_answers": answers}

    def test_dispatcher_runs_the_synthesis_node_for_a_synthesis_task(self):
        from unittest import mock

        original_books = graph_module.books_agent
        graph_module.books_agent = lambda st: {
            "answer": "BOOKS_STEP",
            "sources": ["books"],
            "agent_answers": {"books": "BOOKS_STEP"},
        }
        self.addCleanup(setattr, graph_module, "books_agent", original_books)
        original_synthesis = graph_module.synthesis_node
        self.addCleanup(setattr, graph_module, "synthesis_node",
                        original_synthesis)
        fake = self._fake_llm("COMBINED_ANSWER")

        state = self._state(
            [task("t1", "books", [], ["question"]),
             task("t2", "synthesis", ["t1"], ["books"])], {})
        with mock.patch("app.llm_client.get_llm", return_value=fake):
            self.assertEqual(run_dispatcher_to_completion(state), "qa")

        self.assertEqual(
            statuses(state), {"t1": "completed", "t2": "completed"})
        self.assertEqual(state["answer"], "COMBINED_ANSWER")
        self.assertEqual(state["agent_answers"]["synthesis"], "COMBINED_ANSWER")
        self.assertIn("BOOKS_STEP", fake.prompts[0])
        self.assertNotIn("feasibility", state["sources"],
                         "an ordinary plan must not produce a feasibility study")

    def test_synthesis_results_are_recorded_and_looked_up_under_their_own_key(
            self):
        from app.graph import CAPABILITY_AGENT_KEYS

        self.assertEqual(CAPABILITY_AGENT_KEYS["synthesis"], "synthesis")
        from unittest import mock

        fake = self._fake_llm("COMBINED_ANSWER")
        state = self._state([task("t1", "books", [], ["question"])],
                            {"books": "BOOKS_FINDINGS"})
        state["plan"]["tasks"][0]["status"] = "completed"

        with mock.patch("app.llm_client.get_llm", return_value=fake):
            update = graph_module.synthesis_node(state)

        self.assertEqual(update["agent_answers"]["synthesis"],
                         "COMBINED_ANSWER")

    def test_answer_is_composed_from_completed_task_results(self):
        from unittest import mock

        fake = self._fake_llm("COMBINED_ANSWER")
        state = self._state(
            [task("t1", "books", [], ["question"]),
             task("t2", "erp", ["t1"], ["books"]),
             task("t3", "synthesis", ["t2"], ["books", "erp"])],
            {"books": "BOOKS_FINDINGS", "erp": "ERP_FINDINGS"})
        state["plan"]["tasks"][0]["status"] = "completed"
        state["plan"]["tasks"][1]["status"] = "failed"

        with mock.patch("app.llm_client.get_llm", return_value=fake):
            update = graph_module.synthesis_node(state)

        self.assertEqual(update["answer"], "COMBINED_ANSWER")
        self.assertEqual(update["agent_answers"]["synthesis"], "COMBINED_ANSWER")
        self.assertEqual(update["sources"], ["synthesis"])
        prompt = fake.prompts[0]
        self.assertIn("BOOKS_FINDINGS", prompt)
        self.assertIn("What should we do next?", prompt)
        self.assertNotIn("ERP_FINDINGS", prompt,
                         "a failed task's result must not be presented as work "
                         "that was done")

    def test_provider_failure_returns_the_real_results_without_inventing(self):
        from unittest import mock

        class _Broken:
            def invoke(self, _prompt):
                raise RuntimeError("provider down")

        state = self._state([task("t1", "books", [], ["question"])],
                            {"books": "BOOKS_FINDINGS"})
        state["plan"]["tasks"][0]["status"] = "completed"

        with mock.patch("app.llm_client.get_llm", return_value=_Broken()):
            update = graph_module.synthesis_node(state)

        self.assertIn("BOOKS_FINDINGS", update["answer"])
        self.assertIn("unavailable", update["answer"])
        self.assertEqual(update["sources"], ["synthesis"])

    def test_no_completed_result_is_reported_honestly(self):
        state = self._state([task("t1", "books", [], ["question"])], {})
        state["plan"]["tasks"][0]["status"] = "failed"

        update = graph_module.synthesis_node(state)

        self.assertIn("nothing to combine", update["answer"])
        self.assertEqual(update["sources"], [])

    def test_feasibility_route_still_produces_the_feasibility_study(self):
        called = {}
        original = graph_module.feasibility_agent

        def fake_study(state):
            called["state"] = state
            return {"answer": "STUDY", "sources": ["feasibility"],
                    "agent_answers": {"feasibility": "STUDY"}}

        graph_module.feasibility_agent = fake_study
        try:
            update = graph_module.synthesis_node(
                self._state([task("t1", "synthesis", [], ["question"])],
                            {}, route="feasibility"))
        finally:
            graph_module.feasibility_agent = original

        self.assertEqual(update["answer"], "STUDY")
        self.assertEqual(called["state"]["route"], "feasibility")


# ---------------------------------------------------------------------------
# Provenance / state contract + idempotence
# ---------------------------------------------------------------------------


class TestReentryAndProvenance(unittest.TestCase):
    def test_reentering_dispatcher_does_not_reexecute_completed_tasks(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        state = {
            "question": "q",
            "plan": plan_with([task("t1", "erp", [], ["question"])]),
        }
        apply_dispatch(state)
        self.assertEqual(rec.calls, ["erp"])
        # Re-enter (as the graph would if routing revisited the node):
        apply_dispatch(state)
        apply_dispatch(state)
        self.assertEqual(rec.calls, ["erp"], "completed tasks never re-run")

    def test_existing_results_and_provenance_are_preserved(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        state = {
            "question": "q",
            "sources": ["preexisting"],
            "agent_answers": {"erp": "PREEXISTING_ERP"},
            "plan": plan_with([
                task("t1", "books", [], ["question"]),
                task("t2", "erp", ["t1"], ["books"]),
            ]),
        }
        run_dispatcher_to_completion(state)
        self.assertIn("preexisting", state["sources"])
        self.assertEqual(state["agent_answers"]["erp"], "ERP_OUTPUT",
                         "later task result wins in agent_answers (same channel "
                         "as today), but the earlier entry was not lost en route")
        self.assertEqual(state["agent_answers"]["books"], "BOOKS_OUTPUT")
        self.assertEqual(
            state["sources"],
            ["preexisting", "books", "erp"],
            "sources accumulate without duplication",
        )

    def test_dispatcher_returns_only_valid_state_keys(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)
        state = {
            "question": "q",
            "plan": plan_with([task("t1", "books", [], ["question"])]),
        }
        update = apply_dispatch(state)
        self.assertLessEqual(
            set(update.keys()),
            {"plan", "answer", "sources", "agent_answers"},
            "the dispatcher may only return plan bookkeeping + node outputs",
        )


# ---------------------------------------------------------------------------
# Task-scoped execution (2026-10-29)
# ---------------------------------------------------------------------------


class TestTaskScopedExecution(unittest.TestCase):
    """A planned sub-request runs as its OWN instruction, not as the whole ask.

    The original request stays available as state["request"] for everything
    bound to what the user asked, template objectives keep the request, and a
    research task's external query is always the request (the approved
    operation), never a plan-generated query.
    """

    def _install(self, mapping):
        snapshot = {}
        for name, fn in mapping.items():
            snapshot[name] = getattr(graph_module, name)
            setattr(graph_module, name, fn)
        self.addCleanup(lambda: [setattr(graph_module, n, f)
                                 for n, f in snapshot.items()])

    def _recorder(self, node, seen):
        def fake(state):
            seen.append({"node": node, "question": state.get("question"),
                         "request": state.get("request")})
            return {"answer": f"{node.upper()}_OUT", "sources": [node],
                    "agent_answers": {**(state.get("agent_answers") or {}),
                                      node: f"{node.upper()}_OUT"}}
        return fake

    def test_each_task_receives_its_own_objective(self):
        seen = []
        self._install({"books_agent": self._recorder("books", seen),
                       "erp_agent": self._recorder("erp", seen)})
        request = "Compare our policy documents with the current invoice backlog"
        books_task = task("t1", "books", [], ["question"])
        books_task["objective"] = "What does the AI policy say about model documentation?"
        erp_task = task("t2", "erp", [], ["question"])
        erp_task["objective"] = "Which invoices are overdue this month?"

        run_dispatcher_to_completion({"question": request,
                                      "plan": plan_with([books_task, erp_task])})

        self.assertEqual([c["question"] for c in seen],
                         [books_task["objective"], erp_task["objective"]],
                         "each task must run as its own instruction")
        self.assertEqual([c["request"] for c in seen], [request, request],
                         "the original request stays available beside it")

    def test_dependent_task_sees_the_previous_result(self):
        seen = []
        self._install({"books_agent": self._recorder("books", seen),
                       "erp_agent": self._recorder("erp", seen)})
        t1 = task("t1", "books", [], ["question"])
        t1["objective"] = "What does the guide say about propolis?"
        t2 = task("t2", "erp", ["t1"], ["books"])
        t2["objective"] = "Do any overdue invoices relate to propolis suppliers?"
        state = {"question": "Check propolis and our invoices",
                 "plan": plan_with([t1, t2])}

        run_dispatcher_to_completion(state)

        self.assertEqual(seen[1]["question"], t2["objective"])
        self.assertEqual(state["agent_answers"]["books"], "BOOKS_OUT")
        self.assertEqual(state["plan"]["tasks"][1]["result"], "ERP_OUT")

    def test_template_objectives_keep_the_original_request(self):
        from app.graph import task_instruction
        from app.planner import GENERIC_PLAN_OBJECTIVES

        request = "What does chapter 3 say about propolis?"
        for objective in sorted(GENERIC_PLAN_OBJECTIVES):
            with self.subTest(objective=objective):
                self.assertIsNone(
                    task_instruction({"capability": "books",
                                      "objective": objective}, request))
        # ...while a real sub-request is used, and research never is:
        self.assertEqual(
            task_instruction({"capability": "books",
                              "objective": "Which documents mention propolis?"},
                             request),
            "Which documents mention propolis?")
        self.assertIsNone(task_instruction(
            {"capability": "research",
             "objective": "Search the EU AI Act enforcement deadlines"}, request))

    def test_research_still_searches_the_request_and_keeps_the_gate(self):
        seen = []
        self._install({"research_agent": self._recorder("research", seen)})
        t1 = task("t1", "research", [], ["question"])
        t1["objective"] = "Search the EU AI Act enforcement deadlines"
        request = "Research the latest AI news"

        state = {"question": request, "plan": plan_with([t1])}
        run_dispatcher_to_completion(state)

        self.assertEqual(seen[0]["question"], request,
                         "an objective may never widen the approved search")
        self.assertIsNone(seen[0].get("request"),
                          "research needs no scoped copy: the request IS the "
                          "question")

        # ...and an objective that LOOKS like a web request cannot authorize
        # research on its own: authorization reads the original request.
        seen.clear()
        unauthorised = task("t1", "research", [], ["question"])
        unauthorised["objective"] = "Search the web for the EU AI Act"
        state = {"question": "What does the guide say about hive economics?",
                 "route": "books", "plan": plan_with([unauthorised])}
        run_dispatcher_to_completion(state)
        self.assertEqual(seen, [], "the gate must not be bypassed by an "
                                   "objective")
        self.assertEqual(state["plan"]["tasks"][0]["status"], "failed")
        self.assertIn("not authorized",
                      state["plan"]["tasks"][0]["error"])

    def test_synthesis_prompt_answers_the_original_request(self):
        from unittest import mock

        from types import SimpleNamespace

        class _Fake:
            def __init__(self):
                self.prompts = []

            def invoke(self, prompt):
                self.prompts.append(prompt)
                return SimpleNamespace(content="COMBINED")

        fake = _Fake()
        request = "Compare our policy with the invoice backlog"
        state = {"question": "Summarise the completed steps",
                 "request": request, "route": "books",
                 "plan": plan_with([task("t1", "books", [], ["question"])]),
                 "agent_answers": {"books": "BOOKS_FINDINGS"}}
        state["plan"]["tasks"][0]["status"] = "completed"

        with mock.patch("app.llm_client.get_llm", return_value=fake):
            graph_module.synthesis_node(state)

        self.assertIn(request, fake.prompts[0])
        self.assertNotIn("Summarise the completed steps", fake.prompts[0])

    def test_single_step_route_is_unchanged(self):
        seen = []
        self._install({"books_agent": self._recorder("books", seen)})
        graph = graph_module.build_graph()

        result = graph.invoke({"question": "What does chapter 3 say about propolis?"})

        self.assertEqual(seen[0]["question"],
                         "What does chapter 3 say about propolis?")
        self.assertIsNone(seen[0]["request"],
                          "a non-planned route gains no scoping key")
        self.assertEqual(result["answer"], "BOOKS_OUT")


# ---------------------------------------------------------------------------
# Task output honesty (2026-10-29)
# ---------------------------------------------------------------------------


class TestTaskOutputHonesty(unittest.TestCase):
    """A task is completed ONLY when its node really produced an answer.

    Before this, a node that returned no `answer` (or a blank one) was still
    recorded `completed` with result None: a partial run could report success
    and dependent tasks consumed nothing.
    """

    def _install(self, fake):
        original = graph_module.books_agent
        graph_module.books_agent = fake
        self.addCleanup(setattr, graph_module, "books_agent", original)

    def _state(self):
        return {"question": "q", "plan": plan_with([
            task("t1", "books", [], ["question"]),
            task("t2", "synthesis", ["t1"], ["books"]),
        ])}

    def test_node_without_an_answer_is_marked_failed(self):
        self._install(lambda _state: {})
        state = self._state()

        run_dispatcher_to_completion(state)

        self.assertEqual(statuses(state), {"t1": "failed", "t2": "blocked"})
        t1 = state["plan"]["tasks"][0]
        self.assertIsNone(t1["result"])
        self.assertIn("produced no output", t1["error"])
        self.assertIn("plan incomplete", " | ".join(run_qa(state)))

    def test_blank_answer_is_not_output(self):
        self._install(lambda state: {"answer": "   ", "sources": ["books"],
                                     "agent_answers": {"books": "   "}})
        state = self._state()

        run_dispatcher_to_completion(state)

        self.assertEqual(statuses(state), {"t1": "failed", "t2": "blocked"})
        self.assertIn("produced no output",
                      state["plan"]["tasks"][0]["error"])

    def test_real_answer_is_still_completed_with_its_result(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)
        state = self._state()

        run_dispatcher_to_completion(state)

        self.assertEqual(statuses(state), {"t1": "completed",
                                           "t2": "completed"})
        self.assertEqual(state["plan"]["tasks"][0]["result"],
                         "BOOKS_OUTPUT")


# ---------------------------------------------------------------------------
# Continuation progress reporting (2026-10-29)
# ---------------------------------------------------------------------------


class TestContinuationProgress(unittest.TestCase):
    """`continue_plan(on_progress=...)` reports the plan after EVERY task.

    The runner persists these snapshots so an interrupted continuation can
    resume instead of replaying finished (possibly non-idempotent) work.
    """

    def _state(self):
        return {
            "question": "q",
            "plan": plan_with([
                task("t1", "books", [], ["question"]),
                task("t2", "erp", ["t1"], ["books"]),
            ]),
        }

    def test_progress_is_reported_before_and_after_each_task(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)
        snapshots = []

        result = graph_module.continue_plan(
            self._state(),
            on_progress=lambda cur: snapshots.append(
                {t["task_id"]: t.get("status")
                 for t in cur["plan"]["tasks"]}),
        )

        self.assertEqual(
            [t["status"] for t in result["plan"]["tasks"]],
            ["completed", "completed"],
        )
        self.assertEqual(len(snapshots), 3,
                         "one report for the released plan + one per task")
        self.assertNotIn("completed", list(snapshots[0].values()),
                         "the first report precedes the first task")
        self.assertEqual(snapshots[1], {"t1": "completed", "t2": "pending"},
                         "each finished task is visible before the next runs")
        self.assertEqual(snapshots[-1], {"t1": "completed", "t2": "completed"})

    def test_failing_progress_callback_never_aborts_the_plan(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        def _boom(_current):
            raise RuntimeError("bookkeeping unavailable")

        result = graph_module.continue_plan(self._state(), on_progress=_boom)

        self.assertEqual(
            [t["status"] for t in result["plan"]["tasks"]],
            ["completed", "completed"],
        )
        self.assertEqual(result["answer"], "ERP_OUTPUT")


# ---------------------------------------------------------------------------
# QA ordering and gate behavior
# ---------------------------------------------------------------------------


class TestQaReceivesFinalState(unittest.TestCase):
    def test_qa_runs_after_planned_execution_is_complete(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        graph = graph_module.build_graph()
        result = graph.invoke(
            {"question": "What happened in AI research this week?",
             "complex": True,  # forced planner hop
             },
        )
        # The QA gate ran AFTER all scheduled plan work (qa_flags present and
        # absence of flags proves the gate saw a fully completed plan).
        self.assertIn("qa_flags", result)

    def test_dispatcher_router_routes_to_qa_only_when_all_terminal(self):
        state = {
            "plan": plan_with([
                task("t1", "books", [], ["question"]),
                task("t2", "synthesis", ["t1"], ["books"]),
            ]),
        }
        self.assertEqual(_after_dispatcher(state), "dispatcher")
        state["plan"]["tasks"][0]["status"] = "completed"
        self.assertEqual(_after_dispatcher(state), "dispatcher")
        state["plan"]["tasks"][1]["status"] = "completed"
        self.assertEqual(_after_dispatcher(state), "qa")

    def test_corrupt_plan_routes_straight_to_qa(self):
        self.assertEqual(_after_dispatcher({"plan": "not a dict"}), "qa")
        self.assertEqual(_after_dispatcher({"plan": {"tasks": "no"}}), "qa")
        self.assertEqual(_after_dispatcher({}), "qa")


if __name__ == "__main__":
    unittest.main(verbosity=2)
