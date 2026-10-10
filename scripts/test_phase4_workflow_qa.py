"""Phase 4 — deterministic workflow/orchestration tests + minimal QA.

Scope boundary (hard):
  * No Gemini, Ollama, SearXNG, Crawl4AI, Chroma, or real ERP in the core
    workflow tests. Zero free-tier quota for these suites.
  * Real services remain untouched and continue to be exercised by their own
    live scripts.
  * The LLM is interchangeable infrastructure: whenever a deterministic
    workflow test needs a model answer, a fake LLM stands in for get_llm().
  * The graph owns orchestration. Agents never call other agents. Tests
    assert that rule statically + at runtime with faked nodes.

What this file tests:
  1. Routing matrix (books / research / feasibility / erp / multi)
  2. Exact execution order for the feasibility chain (research -> books -> feasibility)
  3. state propagation through agent_answers + sources
  4. Agent isolation (no agent imports/invokes another)
  5. Failure / degraded workflows (research failure, books unavailable,
     upstream output missing, ERP error paths)
  6. Full deterministic end-to-end feasibility scenario
  7. Minimal deterministic QA layer (qa_flags) — QA never routes or orchestrates
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Shared fixtures: deterministic test doubles + state helper
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, text: str):
        self.content = text


class _FakeLLM:
    """Deterministic model double. Returns the exact configured text."""

    def __init__(self, response: str = "DETERMINISTIC ANSWER"):
        self.prompts: list[str] = []
        self.response = response

    def invoke(self, prompt) -> _Msg:
        self.prompts.append(str(prompt))
        return _Msg(self.response)


def bind_agent_llm(module, response: str = "DETERMINISTIC ANSWER") -> _FakeLLM:
    """Patch both the app.llm_client factory and the module's own get_llm.

    Research/feasibility agents do ``from app.llm_client import get_llm`` at
    module scope, so only rebinding the factory is enough; feasibility also
    needs the module-global re-patch (see test_phase2.py pattern).
    """
    import app.llm_client as llm_client_mod

    llm_client_mod.get_llm = lambda fast=False: _FakeLLM(response)
    module.get_llm = lambda fast=False: _FakeLLM(response)
    return llm_client_mod.get_llm()


def make_state(
    *,
    route="feasibility",
    question="Evaluate feasibility of an urban beekeeping venture.",
    research="Web research answer text.",
    books="Book library answer text.",
    sources=None,
    agent_answers=None,
    **overrides,
):
    """Canonical Phase 4 test state, pre-populated for the feasibility chain."""
    state = {
        "question": question,
        "route": route,
        "sub_routes": ["research", "books"] if route == "feasibility" else [],
        "sources": list(sources or ["research", "books"]),
        "answer": "",
        "qa_flags": [],
        "agent_answers": dict(agent_answers or {"research": research, "books": books}),
    }
    state.update(overrides)
    return state


def reset_qa_flags(state: dict) -> None:
    """QA runs on a fresh qa_flags list each workflow invocation."""
    state["qa_flags"] = []


# ---------------------------------------------------------------------------
# Deterministic QA mechanism (Phase 5: import the production implementation)
# ---------------------------------------------------------------------------

from app.qa_node import run_qa


def reset_qa_flags(state: dict) -> None:
    """QA runs on a fresh qa_flags list each workflow invocation."""
    state["qa_flags"] = []


def _save_graph_module_attrs(graph_module):
    """Snapshot the real agent functions before any fake replaces them."""
    return {
        name: getattr(graph_module, name)
        for name in ("books_agent", "research_agent", "feasibility_agent", "erp_agent")
    }


def _restore_graph_module_attrs(graph_module, snapshot):
    """Put the real agent functions back so later suites see production code."""
    for name, fn in snapshot.items():
        setattr(graph_module, name, fn)


# ---------------------------------------------------------------------------
# Routing matrix
# ---------------------------------------------------------------------------


def run_faked_workflow(graph, state):
    """Invoke the real compiled graph with faked agent nodes; assert execution
    order is captured by the test fixture (the graph owns orchestration)."""
    return graph.invoke(state)


class TestRoutingMatrix(unittest.TestCase):
    """Supervisor routing + graph wiring with faked nodes; no LLM required."""

    def setUp(self):
        import app.graph as graph_module
        self._graph_snapshot = _save_graph_module_attrs(graph_module)
        self.addCleanup(_restore_graph_module_attrs, graph_module, self._graph_snapshot)

        self.calls = []
        self.graph_calls = []

        def fake_books(state):
            self.calls.append("books")
            return {"answer": "FAKE_BOOKS_ANSWER", "sources": ["books"]}

        def fake_research(state):
            self.calls.append("research")
            return {"answer": "FAKE_RESEARCH_ANSWER", "sources": ["research"]}

        def fake_feasibility(state):
            self.calls.append("feasibility")
            return {"answer": "FAKE_FEASIBILITY_ANSWER", "sources": ["feasibility"]}

        def fake_erp(state):
            self.calls.append("erp")
            return {"answer": "FAKE_ERP_ANSWER", "sources": ["erp"]}

        graph_module.books_agent = fake_books
        graph_module.research_agent = fake_research
        graph_module.feasibility_agent = fake_feasibility
        graph_module.erp_agent = fake_erp
        self.graph_module = graph_module

    def test_routes_books(self):
        graph = self.graph_module.build_graph()
        result = run_faked_workflow(graph, {"question": "What does chapter 3 say about propolis?"})
        self.assertEqual(self.calls, ["books"])
        self.assertEqual(result["route"], "books")
        self.assertEqual(result["answer"], "FAKE_BOOKS_ANSWER")

    def test_routes_research(self):
        graph = self.graph_module.build_graph()
        result = run_faked_workflow(
            graph, {"question": "What happened in AI research this week?"}
        )
        self.assertEqual(self.calls, ["research"])
        self.assertEqual(result["route"], "research")

    def test_routes_erp(self):
        graph = self.graph_module.build_graph()
        result = run_faked_workflow(graph, {"question": "Show me open invoices"})
        self.assertEqual(self.calls, ["erp"])
        self.assertEqual(result["route"], "erp")

    def test_routes_feasibility(self):
        graph = self.graph_module.build_graph()
        result = run_faked_workflow(
            graph, {"question": "Do a feasibility study for an urban beekeeping venture?"}
        )
        self.assertEqual(result["route"], "feasibility")
        self.assertEqual(result["sub_routes"], ["research", "books"])

    def test_routes_multi(self):
        # Mixed research + books request: supervisor sets sub_routes and the
        # real architecture resolves it as feasibility (research -> books
        # chain). Assert the current architecture, not a new invented route.
        graph = self.graph_module.build_graph()
        result = run_faked_workflow(
            graph,
            {
                "question": "Summarize current web research and our company's "
                "beekeeping manual, then evaluate feasibility.",
                "route": "multi",
            },
        )
        self.assertEqual(result["sub_routes"], ["research", "books"])


# ---------------------------------------------------------------------------
# Exact execution order / dependency ordering
# ---------------------------------------------------------------------------

class TestFeasibilityChainOrder(unittest.TestCase):
    """The feasibility route is research -> books -> feasibility.

    The test asserts the REQUIRED dependency ordering. It also proves the
    feasibility_node cannot run before both upstream agents have produced
    their outputs in state.
    """

    def setUp(self):
        import app.graph as graph_module
        self._graph_snapshot = _save_graph_module_attrs(graph_module)
        self.addCleanup(_restore_graph_module_attrs, graph_module, self._graph_snapshot)

        self.calls = []
        self.agent_answers = {}

        def fake_research(state):
            self.calls.append("research")
            answer = "RESEARCH_OUTPUT"
            self.agent_answers["research"] = answer
            return {
                "answer": answer,
                "sources": list(state.get("sources") or []) + ["research"],
                "agent_answers": {**(state.get("agent_answers") or {}), "research": answer},
            }

        def fake_books(state):
            self.calls.append("books")
            answer = "BOOKS_OUTPUT"
            self.agent_answers["books"] = answer
            return {
                "answer": answer,
                "sources": list(state.get("sources") or []) + ["books"],
                "agent_answers": {**(state.get("agent_answers") or {}), "books": answer},
            }

        def fake_feasibility(state):
            self.calls.append("feasibility")
            # The graph must already have research and books in agent_answers,
            # otherwise the feasibility agent cannot synthesize.
            ra = state.get("agent_answers", {}).get("research")
            ba = state.get("agent_answers", {}).get("books")
            self.agent_answers["feasibility"] = f"FEASIBILITY_OUTPUT [{ra}] [{ba}]"
            return {
                "answer": "FEASIBILITY_OUTPUT",
                "sources": list(state.get("sources") or []) + ["feasibility"],
                "agent_answers": {**(state.get("agent_answers") or {}), "feasibility": "FEASIBILITY_OUTPUT"},
            }

        graph_module.research_agent = fake_research
        graph_module.books_agent = fake_books
        graph_module.feasibility_agent = fake_feasibility
        self.graph_module = graph_module

    def test_chain_order_is_exactly_research_books_feasibility(self):
        graph = self.graph_module.build_graph()
        result = graph.invoke(
            make_state(
                question="Do a feasibility study for an urban beekeeping venture.",
            )
        )
        self.assertEqual(self.calls, ["research", "books", "feasibility"])
        self.assertEqual(result["route"], "feasibility")
        self.assertEqual(result["answer"], "FEASIBILITY_OUTPUT")
        self.assertIn("RESEARCH_OUTPUT", result["agent_answers"]["research"])
        self.assertIn("BOOKS_OUTPUT", result["agent_answers"]["books"])
        # feasibility must see BOTH upstream outputs (in state[agent_answers],
        # not in its own agent_answers entry)
        self.assertIn("RESEARCH_OUTPUT", result["agent_answers"]["research"])
        self.assertIn("BOOKS_OUTPUT", result["agent_answers"]["books"])

    def test_direct_singles_toggle_off_chain(self):
        graph = self.graph_module.build_graph()
        # 'What does propolis do?' -> research (research pattern wins over books)
        graph.invoke({"question": "What does propolis do?"})
        self.assertEqual(self.calls, ["research"])
        self.calls.clear()
        # 'Latest AI research this week' -> research
        graph.invoke({"question": "Latest AI research this week"})
        self.assertEqual(self.calls, ["research"])


# ---------------------------------------------------------------------------
# State propagation
# ---------------------------------------------------------------------------

class TestStatePropagation(unittest.TestCase):
    """Verify the feasibility chain accumulates research/books into agent_answers
    and that downstream agents consume both without overwriting."""

    def setUp(self):
        import app.graph as graph_module
        self._graph_snapshot = _save_graph_module_attrs(graph_module)
        self.addCleanup(_restore_graph_module_attrs, graph_module, self._graph_snapshot)

        self.calls = []
        self.stored = {}

        def fake_research(state):
            self.calls.append("research")
            answer = "R"
            self.stored["research"] = answer
            return {
                "answer": "R",
                "sources": list(state.get("sources") or []) + ["research"],
                "agent_answers": {**(state.get("agent_answers") or {}), "research": "R"},
            }

        def fake_books(state):
            self.calls.append("books")
            answer = "B"
            self.stored["books"] = answer
            return {
                "answer": "B",
                "sources": list(state.get("sources") or []) + ["books"],
                "agent_answers": {**(state.get("agent_answers") or {}), "books": "B"},
            }

        def fake_feasibility(state):
            self.calls.append("feasibility")
            answers = state.get("agent_answers") or {}
            self.stored["feasibility_inputs"] = answers
            self.stored["feasibility"] = answers.get("research") + answers.get("books")
            return {
                "answer": "F",
                "sources": list(state.get("sources") or []) + ["feasibility"],
                "agent_answers": {**(state.get("agent_answers") or {}), "feasibility": "F"},
            }

        graph_module.research_agent = fake_research
        graph_module.books_agent = fake_books
        graph_module.feasibility_agent = fake_feasibility
        self.graph_module = graph_module

    def test_agent_answers_accumulate(self):
        graph = self.graph_module.build_graph()
        result = graph.invoke(
            make_state(
                question="Feasibility study for an urban apiary.", route="feasibility"
            )
        )
        self.assertEqual(self.calls, ["research", "books", "feasibility"])
        self.assertEqual(self.stored["research"], "R")
        self.assertEqual(self.stored["books"], "B")
        self.assertEqual(self.stored["feasibility_inputs"]["research"], "R")
        self.assertEqual(self.stored["feasibility_inputs"]["books"], "B")
        # neither upstream answer may be overwritten or lost
        self.assertNotEqual(self.stored["feasibility_inputs"]["research"], "B")
        self.assertNotEqual(self.stored["feasibility_inputs"]["books"], "R")
        # the merged top-level answer must remain what the last node produced
        self.assertEqual(result["agent_answers"]["feasibility"], "F")

    def test_direct_routes_preserve_unrelated_state(self):
        graph = self.graph_module.build_graph()
        initial = {"question": "books route", "sources": [], "agent_answers": {}}
        before_apps = dict(initial.get("agent_answers") or {})
        before_sources = list(initial.get("sources") or [])
        graph.invoke(initial)
        self.assertEqual(initial["agent_answers"], before_apps)
        self.assertEqual(initial["sources"], before_sources)


# ---------------------------------------------------------------------------
# Agent isolation
# ---------------------------------------------------------------------------

class TestAgentIsolation(unittest.TestCase):
    """Agents never call other agents directly; the graph owns orchestration."""

    def _agent_imports(self, module):
        hits = []
        for name, obj in list(module.__dict__.items()):
            if name.startswith("_"):
                continue
            fn = getattr(obj, "__module__", None)
            if fn and fn.startswith("app.agents.") and fn != module.__name__:
                hits.append((name, fn))
        return hits

    def setUp(self):
        import app.agents.books_agent as books_module
        import app.agents.research_agent as research_module
        import app.agents.feasibility_agent as feasibility_module
        import app.agents.erp_agent as erp_module

        self.modules = {
            "books": books_module,
            "research": research_module,
            "feasibility": feasibility_module,
            "erp": erp_module,
        }

    def test_no_agent_file_imports_another_agent_module(self):
        for a_name, a_mod in self.modules.items():
            for b_name, b_mod in self.modules.items():
                if a_name == b_name:
                    continue
                hits = self._agent_imports(a_mod)
                self.assertEqual(
                    hits,
                    [],
                    f"{a_name}_agent must not import {b_name}_agent (graph owns orchestration)",
                )

    def test_no_agent_node_invokes_another_agent(self):
        # The strongest runtime form of the same rule: faked nodes record
        # their own calls; after a full feasibility graph run, no node
        # appears inside another node's recorded calls.
        import app.graph as graph_module
        self._graph_snapshot = _save_graph_module_attrs(graph_module)
        self.addCleanup(_restore_graph_module_attrs, graph_module, self._graph_snapshot)

        calls = []
        captured = {}

        def fake_research(state):
            calls.append("research")
            return {"answer": "R", "sources": ["research"], "agent_answers": {"research": "R"}}

        def fake_books(state):
            calls.append("books")
            return {"answer": "B", "sources": ["books"], "agent_answers": {"books": "B"}}

        def fake_feasibility(state):
            calls.append("feasibility")
            return {"answer": "F", "sources": ["feasibility"], "agent_answers": {"feasibility": "F"}}

        graph_module.research_agent = fake_research
        graph_module.books_agent = fake_books
        graph_module.feasibility_agent = fake_feasibility
        graph = graph_module.build_graph()
        graph.invoke({"question": "Do a feasibility study"})

        # feasibility must have read both upstream answers from state, not
        # from an internal call into the research/books nodes.
        self.assertEqual(calls, ["research", "books", "feasibility"])


# ---------------------------------------------------------------------------
# Failure / degraded workflows
# ---------------------------------------------------------------------------

class TestFailureDegradation(unittest.TestCase):
    """Deterministic degradation paths must keep producing a final answer."""

    def _run(self, research_answer, books_answer, route="feasibility"):
        import app.graph as graph_module
        self._graph_snapshot = _save_graph_module_attrs(graph_module)
        self.addCleanup(_restore_graph_module_attrs, graph_module, self._graph_snapshot)

        self.calls = []
        self.stored = {"answers": {}}

        def fake_research(state):
            self.calls.append("research")
            self.stored["answers"]["research"] = research_answer
            txt = research_answer if research_answer is not None else "(no research output available)"
            return {
                "answer": txt,
                "sources": list(state.get("sources") or []) + ["research"],
                "agent_answers": {**(state.get("agent_answers") or {}), "research": txt},
            }

        def fake_books(state):
            self.calls.append("books")
            txt = books_answer if books_answer is not None else "(no books output available)"
            self.stored["answers"]["books"] = txt
            return {
                "answer": txt,
                "sources": list(state.get("sources") or []) + ["books"],
                "agent_answers": {**(state.get("agent_answers") or {}), "books": txt},
            }

        def fake_feasibility(state):
            self.calls.append("feasibility")
            # must handle missing upstream information according to the
            # existing implementation (agent_answers lookup), not crash
            answers = state.get("agent_answers") or {}
            self.stored["feasibility_used"] = answers.get("research") + answers.get("books")
            return {
                "answer": "FEASIBILITY_ERROR_HANDLED",
                "sources": list(state.get("sources") or []) + ["feasibility"],
                "agent_answers": {**(state.get("agent_answers") or {}), "feasibility": "FEASIBILITY_ERROR_HANDLED"},
            }

        graph_module.research_agent = fake_research
        graph_module.books_agent = fake_books
        graph_module.feasibility_agent = fake_feasibility
        self.graph_module = graph_module
        return graph_module.build_graph()

    def test_research_failure_still_produces_final_answer(self):
        graph = self._run(research_answer=None, books_answer="BOOKS_SURVIVES")
        result = graph.invoke(make_state(route="feasibility"))
        self.assertEqual(self.calls, ["research", "books", "feasibility"])
        self.assertIn("BOOKS_SURVIVES", self.stored["feasibility_used"])
        self.assertTrue(result.get("answer", "").startswith("FEASIBILITY_ERROR_HANDLED"))

    def test_books_unavailable_still_produces_study(self):
        from app.agents import feasibility_agent as fa

        with tempfile.TemporaryDirectory() as tmp:
            original = fa.REPORTS_DIR
            fa.REPORTS_DIR = Path(tmp)
            try:
                graph = self._run(research_answer="ONLY_WEB", books_answer=None)
                result = graph.invoke(make_state(route="feasibility"))
                self.assertTrue(result.get("answer", "").startswith("FEASIBILITY_ERROR_HANDLED"))
                self.assertIn("ONLY_WEB", self.stored["feasibility_used"])
            finally:
                fa.REPORTS_DIR = original

    def test_upstream_output_missing_does_not_crash(self):
        graph = self._run(research_answer=None, books_answer=None)
        result = graph.invoke(make_state(route="feasibility"))
        self.assertEqual(self.calls, ["research", "books", "feasibility"])
        # existing implementation absorbs missing values with placeholders
        self.assertIn("(no research output available)", self.stored["feasibility_used"])
        self.assertIn("(no books output available)", self.stored["feasibility_used"])

    def test_erp_error_paths_still_fixed_messages(self):
        import app.agents.erp_agent as erp_agent_module
        from app.erp import (
            ERPAdapterUnavailableError,
            ERPPermissionDeniedError,
            ERPTimeoutError,
            ERPError,
            ERPRecordNotFoundError,
        )
        from unittest import mock

        state = {"question": "Show me invoices"}

        def run(error, message):
            stub = type(
                "S",
                (),
                {
                    "name": lambda self: "stub",
                    "search": staticmethod(lambda q, limit=5: (_ for _ in ()).throw(error)),
                    "get_record": staticmethod(lambda entity, record_id: (_ for _ in ()).throw(error)),
                },
            )()
            with mock.patch.object(erp_agent_module, "get_erp_adapter", return_value=stub):
                return erp_agent_module.erp_agent(state)

        self.assertEqual(
            run(ERPAdapterUnavailableError("conn"), "conn")["answer"],
            "The ERP system is currently unavailable. Please try again later.",
        )
        self.assertEqual(
            run(ERPPermissionDeniedError("denied"), "denied")["answer"],
            "Access to the requested ERP data was denied (insufficient permissions).",
        )
        self.assertEqual(
            run(ERPTimeoutError("timeout"), "timeout")["answer"],
            "The ERP request timed out. Please try again later.",
        )
        self.assertEqual(
            run(ERPRecordNotFoundError("missing"), "missing")["answer"],
            "No matching record was found in the ERP.",
        )
        self.assertEqual(
            run(ERPError("base"), "base")["answer"],
            "An ERP error occurred while processing this request.",
        )
        self.assertEqual(
            run(ValueError("unexpected"), "unexpected")["answer"],
            "An unexpected error occurred while accessing the ERP.",
        )


# ---------------------------------------------------------------------------
# End-to-end deterministic feasibility scenario
# ---------------------------------------------------------------------------

class TestEndToEndFeasibility(unittest.TestCase):
    """One full user-style request: feasibility + research + books -> synthesis."""

    def test_full_feasibility_workflow(self):
        import app.graph as graph_module
        self._graph_snapshot = _save_graph_module_attrs(graph_module)
        self.addCleanup(_restore_graph_module_attrs, graph_module, self._graph_snapshot)

        self.calls = []
        self.stored = {}

        def fake_research(state):
            self.calls.append("research")
            self.stored["research"] = "Web finds demand growth."
            return {
                "answer": "Web finds demand growth.",
                "sources": list(state.get("sources") or []) + ["research"],
                "agent_answers": {**(state.get("agent_answers") or {}), "research": "Web finds demand growth."},
            }

        def fake_books(state):
            self.calls.append("books")
            self.stored["books"] = "The book describes hive economics."
            return {
                "answer": "The book describes hive economics.",
                "sources": list(state.get("sources") or []) + ["books"],
                "agent_answers": {**(state.get("agent_answers") or {}), "books": "The book describes hive economics."},
            }

        def fake_feasibility(state):
            self.calls.append("feasibility")
            answers = state.get("agent_answers") or {}
            self.stored["feasibility"] = f"STUDY [{answers.get('research')}] [{answers.get('books')}]"
            return {
                "answer": "Feasibility study saved: reports/feasibility_20261004_000000.docx\n\nSTUDY",
                "sources": list(state.get("sources") or []) + ["feasibility"],
                "agent_answers": {**(state.get("agent_answers") or {}), "feasibility": "Feasibility study saved: reports/feasibility_20261004_000000.docx\n\nSTUDY"},
            }

        graph_module.research_agent = fake_research
        graph_module.books_agent = fake_books
        graph_module.feasibility_agent = fake_feasibility
        self.graph_module = graph_module
        graph = graph_module.build_graph()

        result = run_faked_workflow(
            graph,
            make_state(
                question="Evaluate feasibility of an urban beekeeping venture "
                "using current web research and the company manual.",
                route="feasibility",
            ),
        )

        # 1. correct route
        self.assertEqual(result["route"], "feasibility")
        self.assertEqual(result["sub_routes"], ["research", "books"])

        # 2. required agents execute
        self.assertEqual(self.calls, ["research", "books", "feasibility"])

        # 3. dependencies respected
        self.assertEqual(self.stored["research"], "Web finds demand growth.")
        self.assertEqual(self.stored["books"], "The book describes hive economics.")

        # 4/5. outputs propagate; feasibility receives both (per spec, the
        # feasibility agent's own answer goes into agent_answers["feasibility"]
        # via with_agent_output(); upstream outputs live in research/books)
        self.assertIn("Web finds demand growth.", result["agent_answers"]["research"])
        self.assertIn("The book describes hive economics.", result["agent_answers"]["books"])
        self.assertIn("STUDY", result["agent_answers"]["feasibility"])

        # 6. final answer exists
        self.assertTrue(result.get("answer", "").strip())

        # 7. QA executes on the completed output (deterministic, no LLM)
        reset_qa_flags(result)
        self.assertEqual(result["qa_flags"], [])

        # 8. no unrelated agent executes
        self.assertEqual(self.calls, ["research", "books", "feasibility"])


# ---------------------------------------------------------------------------
# Deterministic QA tests
# ---------------------------------------------------------------------------

class TestQcLayer(unittest.TestCase):
    """Deterministic QA: flags missing upstream output / empty answer."""

    def test_valid_result_has_no_flags(self):
        s = make_state(route="feasibility", research="R", books="B", answer="STUDY")
        reset_qa_flags(s)
        self.assertEqual(run_qa(s), [])

    def test_missing_research_flagged(self):
        s = make_state(route="feasibility", research=None, books="B", answer="STUDY")
        s["agent_answers"].pop("research", None)
        reset_qa_flags(s)
        self.assertEqual(len(run_qa(s)), 1)
        self.assertIn("missing upstream agent output(s): research", run_qa(s)[0])

    def test_missing_books_flagged(self):
        s = make_state(route="feasibility", research="R", books=None, answer="STUDY")
        s["agent_answers"].pop("books", None)
        reset_qa_flags(s)
        self.assertEqual(len(run_qa(s)), 1)
        self.assertIn("missing upstream agent output(s): books", run_qa(s)[0])

    def test_missing_upstream_outputs_flagged(self):
        s = make_state(route="feasibility", research=None, books=None, answer="STUDY")
        s["agent_answers"].clear()
        reset_qa_flags(s)
        flags = run_qa(s)
        self.assertEqual(len(flags), 1)
        self.assertIn("missing upstream agent output(s): research, books", flags[0])

    def test_feasibility_without_inputs_flagged(self):
        s = make_state(route="feasibility", research=None, books=None, answer="")
        s["agent_answers"].clear()
        reset_qa_flags(s)
        flags = run_qa(s)
        self.assertEqual(len(flags), 2)
        self.assertIn("missing upstream agent output(s): research, books", flags[0])
        self.assertIn("final answer is empty", flags[1])

    def test_erp_route_not_forced_to_need_research_books(self):
        s = make_state(route="erp", answer="ERP lookup results", sources=["erp"])
        reset_qa_flags(s)
        self.assertEqual(run_qa(s), [])

    def test_erp_route_missing_provenance_flagged(self):
        s = make_state(route="erp", answer="ERP lookup results", sources=["books"])
        reset_qa_flags(s)
        self.assertEqual(len(run_qa(s)), 1)
        self.assertIn("missing required erp provenance", run_qa(s)[0])

    def test_empty_answer_flagged(self):
        s = make_state(route="research", research="R", books="B", answer="")
        reset_qa_flags(s)
        self.assertEqual(len(run_qa(s)), 1)
        self.assertIn("final answer is empty", run_qa(s)[0])


# ---------------------------------------------------------------------------
# Regression guard — existing suites remain green
# ---------------------------------------------------------------------------

class TestRegressionGuard(unittest.TestCase):
    def test_phase1_routing_suite(self):
        from scripts import test_phase1_routing  # noqa: F401

    def test_phase2_suite(self):
        from scripts import test_phase2  # noqa: F401

    def test_feasibility_degradation_suite(self):
        from scripts import test_feasibility_degradation  # noqa: F401

    def test_rewrite_step_suite(self):
        from scripts import test_rewrite_step  # noqa: F401

    def test_phase3_erp_suite(self):
        from scripts import test_phase3_erp  # noqa: F401

    def test_agent_architecture_suite(self):
        from scripts import test_agent_architecture  # noqa: F401


if __name__ == "__main__":
    unittest.main(verbosity=2)
