"""Phase 6 tests: LLM-assisted supervisor routing with deterministic fallback.

All tests use fakes (zero quota). ROUTING_MODE is patched per-test; keyword
routing must stay the always-available path.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import supervisor as sup  # noqa: E402
from app.config import ROUTING_MODE  # noqa: E402


class _FakeLLMResponse:
    def __init__(self, text):
        self.content = text


class _FakeLLM:
    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self._exc is not None:
            raise self._exc
        return _FakeLLMResponse(self._text)


BASE_STATE = {"question": "What does chapter 3 say about propolis?"}


class TestDefaultKeywordMode(unittest.TestCase):
    def test_default_mode_is_keyword(self):
        self.assertEqual(ROUTING_MODE, "keyword")

    def test_default_mode_never_calls_llm(self):
        fake = _FakeLLM(text="books")
        with mock.patch.object(sup, "ROUTING_MODE", "keyword"), \
             mock.patch.dict(sys.modules, {"app.llm_client": mock.MagicMock()}):
            result = sup.supervisor(BASE_STATE)
        self.assertEqual(result, {"route": "books"})
        self.assertEqual(fake.calls, [])


class TestLlmClassification(unittest.TestCase):
    def _run_llm(self, fake):
        with mock.patch.object(sup, "ROUTING_MODE", "llm"), \
             mock.patch("app.llm_client.get_llm", return_value=fake):
            return sup.supervisor(BASE_STATE)

    def test_each_label_routes_correctly(self):
        for label, expected in [
            ("books", "books"),
            ("research", "research"),
            ("feasibility", "feasibility"),
            ("erp", "erp"),
            ("BOOKS", "books"),   # case-insensitive
            (" feasibility ", "feasibility"),
        ]:
            with self.subTest(label=label):
                result = self._run_llm(_FakeLLM(text=label))
                self.assertEqual(result["route"], expected)
                if expected == "feasibility":
                    self.assertEqual(result["sub_routes"], ["research", "books"])

    def test_malformed_output_falls_back_to_keywords(self):
        result = self._run_llm(_FakeLLM(text="I am not sure, maybe books?"))
        self.assertEqual(result["route"], "books")  # keyword hit for this question

    def test_unknown_label_falls_back(self):
        result = self._run_llm(_FakeLLM(text="shopping"))
        self.assertEqual(result["route"], "books")

    def test_llm_exception_falls_back(self):
        result = self._run_llm(_FakeLLM(exc=RuntimeError("provider down")))
        self.assertEqual(result["route"], "books")

    def test_llm_never_stored_in_state(self):
        result = self._run_llm(_FakeLLM(text="books"))
        self.assertNotIn("routing_decision", result)
        self.assertEqual(set(result.keys()), {"route"})

    def test_feasibility_llm_contract_keeps_sub_routes(self):
        result = self._run_llm(_FakeLLM(text="feasibility"))
        self.assertEqual(
            result, {"route": "feasibility", "sub_routes": ["research", "books"]}
        )


class TestKeywordFallbackPreserved(unittest.TestCase):
    """Priority order (feasibility > erp > books > research) in BOTH modes."""

    CASES = [
        ("Do a feasibility study for the beekeeping venture?", "feasibility"),
        ("Show me open invoices", "erp"),
        ("What does chapter 3 of the manual say?", "books"),
        ("What happened in AI research this week?", "research"),
    ]

    def test_keyword_mode_priority(self):
        for question, expected in self.CASES:
            with self.subTest(q=question, mode="keyword"):
                result = sup.keyword_route({"question": question})
                self.assertEqual(result["route"], expected)

    def test_llm_mode_fallback_matches_keyword_priority(self):
        for question, expected in self.CASES:
            with self.subTest(q=question, mode="llm-fallback"):
                fake = _FakeLLM(text="garbage-label")
                with mock.patch.object(sup, "ROUTING_MODE", "llm"), \
                     mock.patch("app.llm_client.get_llm", return_value=fake):
                    result = sup.supervisor({"question": question})
                self.assertEqual(result["route"], expected)


class TestGraphContractUnchanged(unittest.TestCase):
    def test_keyword_mode_graph_routes_unchanged(self):
        import app.graph as gm
        from app.agents.books_agent import books_agent as books_agent_real
        from app.agents.research_agent import research_agent as research_agent_real
        from app.agents.erp_agent import erp_agent as erp_agent_real

        def fake_books(state):
            return {"answer": "B", "sources": ["books"]}

        def fake_research(state):
            return {"answer": "R", "sources": ["research"]}

        def fake_erp(state):
            return {"answer": "E", "sources": ["erp"]}

        gm.books_agent = fake_books
        gm.research_agent = fake_research
        gm.erp_agent = fake_erp
        self.addCleanup(
            lambda: (
                setattr(gm, "books_agent", books_agent_real),
                setattr(gm, "research_agent", research_agent_real),
                setattr(gm, "erp_agent", erp_agent_real),
            )
        )
        g = gm.build_graph()
        r = g.invoke({"question": "What does chapter 3 say about propolis?"})
        self.assertEqual(r["route"], "books")
        r2 = g.invoke({"question": "Latest AI news"})
        self.assertEqual(r2["route"], "research")
        r3 = g.invoke({"question": "Show me open invoices"})
        self.assertEqual(r3["route"], "erp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
