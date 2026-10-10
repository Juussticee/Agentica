"""Phase 5 tests: production deterministic QA node (zero quota).

Covers MASTER_PLAN Phase 5 acceptance criteria:
  * qa_node appends to qa_flags, merging (never clearing) existing flags
  * returns ONLY qa_flags
  * fail-soft: an internal QA error becomes a flag, never a crash
  * end-to-end: every route's terminal hop passes through the gate
"""

import unittest
from unittest.mock import patch

from app.qa_node import qa_node, run_qa


def make_state(**overrides):
    state = {
        "question": "q",
        "route": "research",
        "sources": ["research"],
        "answer": "ANSWER",
        "qa_flags": [],
        "agent_answers": {"research": "R"},
    }
    state.update(overrides)
    return state


class TestQaNodeUnit(unittest.TestCase):
    def test_returns_only_qa_flags_key(self):
        update = qa_node(make_state())
        self.assertEqual(set(update.keys()), {"qa_flags"})

    def test_valid_single_route_has_no_flags(self):
        self.assertEqual(qa_node(make_state())["qa_flags"], [])

    def test_appends_to_existing_flags_never_clears(self):
        state = make_state(route="erp", sources=["books"], qa_flags=["QA: pre-existing"])
        result = qa_node(state)["qa_flags"]
        self.assertIn("QA: pre-existing", result)
        self.assertIn("QA: ERP route missing required erp provenance", result)

    def test_feasibility_missing_books_flagged(self):
        state = make_state(route="feasibility", agent_answers={"research": "R"}, sources=["research"])
        flags = qa_node(state)["qa_flags"]
        joined = " | ".join(flags)
        self.assertIn("missing required books output", joined)
        self.assertIn("missing upstream agent output(s): books", joined)

    def test_empty_answer_flagged(self):
        flags = qa_node(make_state(answer="   "))["qa_flags"]
        self.assertEqual(len(flags), 1)
        self.assertIn("final answer is empty", flags[0])

    def test_fail_soft_on_internal_error(self):
        with patch("app.qa_node.run_qa", side_effect=RuntimeError("boom")):
            flags = qa_node(make_state())["qa_flags"]
        self.assertEqual(len(flags), 1)
        self.assertIn("QA: internal error", flags[0])
        self.assertIn("boom", flags[0])

    def test_flag_order_deterministic(self):
        state = make_state(
            route="feasibility", agent_answers={}, sources=[], answer=""
        )
        flags = qa_node(state)["qa_flags"]
        self.assertEqual(
            flags,
            [
                "QA: feasibility route missing required research output",
                "QA: feasibility route missing required books output",
                "QA: feasibility route incomplete — missing upstream agent output(s): research, books",
                "QA: final answer is empty — workflow produced no output",
            ],
        )

    def test_run_qa_contract_unchanged_from_phase4(self):
        # Same expectations the 30 Phase 4 tests froze.
        self.assertEqual(run_qa(make_state(route="erp", sources=["erp"])), [])
        self.assertEqual(run_qa(make_state()), [])
        s = make_state(route="erp", sources=["books"])
        self.assertIn("missing required erp provenance", run_qa(s)[0])


class TestQaNodeInGraph(unittest.TestCase):
    """End-to-end: every route ends through the qa gate (faked agents)."""

    @classmethod
    def setUpClass(cls):
        import app.graph as graph_module
        cls.graph_module = graph_module
        # Save the real agents once; restore in tearDownClass so later suites
        # (test_phase3_erp with the REAL erp agent) never see these fakes.
        cls._snapshot = {
            name: getattr(graph_module, name)
            for name in ("books_agent", "research_agent", "feasibility_agent", "erp_agent")
        }
        cls._install_fakes()

    @classmethod
    def tearDownClass(cls):
        for name, fn in cls._snapshot.items():
            setattr(cls.graph_module, name, fn)

    @classmethod
    def _install_fakes(cls):
        graph_module = cls.graph_module

        def fake_books(state):
            return {
                "answer": "BOOKS",
                "sources": list(state.get("sources") or []) + ["books"],
                "agent_answers": {**(state.get("agent_answers") or {}), "books": "BOOKS"},
            }

        def fake_research(state):
            return {
                "answer": "RESEARCH",
                "sources": list(state.get("sources") or []) + ["research"],
                "agent_answers": {**(state.get("agent_answers") or {}), "research": "RESEARCH"},
            }

        def fake_feasibility(state):
            return {
                "answer": "STUDY",
                "sources": list(state.get("sources") or []) + ["feasibility"],
                "agent_answers": {**(state.get("agent_answers") or {}), "feasibility": "STUDY"},
            }

        def fake_erp(state):
            return {
                "answer": "ERP",
                "sources": list(state.get("sources") or []) + ["erp"],
                "agent_answers": {**(state.get("agent_answers") or {}), "erp": "ERP"},
            }

        cls.graph_module.books_agent = fake_books
        cls.graph_module.research_agent = fake_research
        cls.graph_module.feasibility_agent = fake_feasibility
        cls.graph_module.erp_agent = fake_erp

    def test_books_route_has_qa_flags_key(self):
        g = self.graph_module.build_graph()
        r = g.invoke({"question": "What does chapter 3 say about propolis?"})
        self.assertEqual(r["route"], "books")
        self.assertEqual(r["qa_flags"], [])

    def test_research_route_has_qa_flags_key(self):
        g = self.graph_module.build_graph()
        r = g.invoke({"question": "What happened in AI research this week?"})
        self.assertEqual(r["route"], "research")
        self.assertEqual(r["qa_flags"], [])

    def test_erp_route_qa_does_not_demand_research_books(self):
        g = self.graph_module.build_graph()
        r = g.invoke({"question": "Show me open invoices"})
        self.assertEqual(r["route"], "erp")
        self.assertEqual(r["qa_flags"], [])

    def test_feasibility_route_qa_flags_present(self):
        g = self.graph_module.build_graph()
        r = g.invoke(
            {"question": "Do a feasibility study for an urban beekeeping venture?"}
        )
        self.assertEqual(r["route"], "feasibility")
        self.assertEqual(r["qa_flags"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
