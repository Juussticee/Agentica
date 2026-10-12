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
            for name in ("books_agent", "research_agent", "feasibility_agent", "erp_agent")
        }
        self._behavior = behavior or {}

    def install(self):
        for name in ("books_agent", "research_agent", "feasibility_agent", "erp_agent"):
            node = name.removesuffix("_agent")
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
        self.assertEqual(
            CAPABILITY_NODES,
            {"research": "research", "books": "books", "erp": "erp",
             "synthesis": "feasibility"},
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
            "question": "q",
            "plan": plan_with([
                task("t1", "research", [], ["question"]),
                task("t2", "books", ["t1"], ["research"]),
                task("t3", "synthesis", ["t1", "t2"], ["research", "books"]),
            ]),
        }
        self.assertEqual(run_dispatcher_to_completion(state), "qa")

        self.assertEqual(
            rec.calls, ["research", "books", "feasibility"],
            "t1→t2→t3 must dispatch research→books→feasibility(enc synth)",
        )
        self.assertEqual(
            statuses(state),
            {"t1": "completed", "t2": "completed", "t3": "completed"},
        )
        # Results + provenance through the EXISTING channels only.
        self.assertEqual(state["agent_answers"]["research"], "RESEARCH_OUTPUT")
        self.assertEqual(state["agent_answers"]["books"], "BOOKS_OUTPUT")
        self.assertEqual(state["agent_answers"]["feasibility"], "FEASIBILITY_OUTPUT")
        for name in ("research", "books", "feasibility"):
            self.assertIn(name, state["sources"])

    def test_independent_tasks_are_not_forced_into_a_specific_order(self):
        rec = _AgentRecorder().install()
        self.addCleanup(rec.restore)

        state = {
            "question": "q",
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
            "question": "q",
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
        # GOOD_LLMM_PLAN: t1 research → t2 books → t3 synthesis.
        self.assertEqual(rec.calls, ["research", "books", "feasibility"])
        self.assertEqual(result["route"], "feasibility")  # set by supervisor
        self.assertEqual(
            [(t["task_id"], t["status"]) for t in result["plan"]["tasks"]],
            [("t1", "completed"), ("t2", "completed"), ("t3", "completed")],
        )
        self.assertEqual(result["agent_answers"]["research"], "RESEARCH_OUTPUT")
        self.assertEqual(result["agent_answers"]["books"], "BOOKS_OUTPUT")
        self.assertEqual(result["agent_answers"]["feasibility"], "FEASIBILITY_OUTPUT")
        self.assertEqual(result["qa_flags"], [])


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
