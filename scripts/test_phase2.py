"""Phase 2 tests (spec steps 7-8 wiring) — fake LLM, zero Gemini quota.

  step 7: feasibility_agent standalone — consumes agent_answers from state,
          writes a .docx with Market/Financial/Risk sections, returns the path
  step 8 (wiring): supervisor -> research -> books -> feasibility chain order,
          provenance accumulation, direct routes still end after one agent
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from docx import Document as DocxReader  # noqa: E402


class _Msg:
    def __init__(self, text):
        self.content = text


class FakeFeasibilityLLM:
    """Deterministic stand-in returning a clearly sectioned markdown study."""

    def invoke(self, prompt):
        low = str(prompt).lower()
        assert "findings from web research" in low, "prompt must include research answer"
        assert "findings from the book" in low, "prompt must include books answer"
        study = (
            "## Market\nDemand is growing per web research; the books support the trend.\n\n"
            "## Financial\nUpfront cost is unknown from the findings; revenue gap noted.\n\n"
            "## Risk\nKey risk: missing cost data; second risk: adoption uncertainty.\n"
        )
        return _Msg(study)


class TestFeasibilityAgent(unittest.TestCase):
    def test_step7_docx_and_contract(self):
        import app.llm_client as llm_client_module

        # Patch FIRST, then import the agent, so its module-level
        # "from app.llm_client import get_llm" binds the fake (zero quota).
        llm_client_module.get_llm = lambda: FakeFeasibilityLLM()
        from app.agents import feasibility_agent as fa

        # Belt and braces: also rebind inside the agent module itself.
        fa.get_llm = lambda: FakeFeasibilityLLM()

        with tempfile.TemporaryDirectory() as tmp:
            import app.agents.feasibility_agent as fa

            original_reports_dir = fa.REPORTS_DIR
            fa.REPORTS_DIR = Path(tmp)  # redirect output for the test
            try:
                state = {
                    "question": "Should we launch an urban beekeeping product line?",
                    "route": "feasibility",
                    "sub_routes": ["research", "books"],
                    "agent_answers": {
                        "research": "Web says honey demand up 20% YoY.",
                        "books": "The field guide covers hive economics basics.",
                    },
                    "sources": ["research", "https://example.com", "books"],
                }
                result = fa.feasibility_agent(state)

                answer = result["answer"]
                self.assertIn("Feasibility study saved to:", answer)
                docx_path = Path(answer.split("saved to: ")[1].split("\n")[0].strip())
                self.assertTrue(docx_path.exists(), "the .docx must exist on disk")

                doc = DocxReader(str(docx_path))
                headings = [
                    p.text for p in doc.paragraphs if p.style.name.startswith("Heading")
                ]
                for section in ("Market", "Financial", "Risk"):
                    self.assertIn(section, headings, f"missing {section} heading")

                # contract: provenance accumulates, answer recorded per-agent
                self.assertIn("feasibility", result["sources"])
                self.assertEqual(result["agent_answers"]["feasibility"], answer)
                self.assertIn("books", result["agent_answers"])
            finally:
                fa.REPORTS_DIR = original_reports_dir


class TestPhase2Routing(unittest.TestCase):
    def test_step8_chain_order(self):
        import app.graph as graph_module

        calls = []

        def fake_research(state):
            calls.append("research")
            return {
                "answer": "R",
                "sources": list(state.get("sources") or []) + ["research"],
                "agent_answers": {**(state.get("agent_answers") or {}), "research": "R"},
            }

        def fake_books(state):
            calls.append("books")
            return {
                "answer": "B",
                "sources": list(state.get("sources") or []) + ["books"],
                "agent_answers": {**(state.get("agent_answers") or {}), "books": "B"},
            }

        def fake_feasibility(state):
            calls.append("feasibility")
            assert state.get("agent_answers", {}).get("research") == "R"
            assert state.get("agent_answers", {}).get("books") == "B"
            return {
                "answer": "F",
                "sources": list(state.get("sources") or []) + ["feasibility"],
                "agent_answers": {**(state.get("agent_answers") or {}), "feasibility": "F"},
            }

        graph_module.research_agent = fake_research
        graph_module.books_agent = fake_books
        graph_module.feasibility_agent = fake_feasibility
        graph = graph_module.build_graph()

        r = graph.invoke(
            {"question": "Do a feasibility study for an urban beekeeping venture"}
        )
        self.assertEqual(calls, ["research", "books", "feasibility"])
        self.assertEqual(r["route"], "feasibility")
        self.assertEqual(r["sub_routes"], ["research", "books"])
        self.assertEqual(r["answer"], "F")
        for s in ("research", "books", "feasibility"):
            self.assertIn(s, r["sources"])

        # direct routes must still terminate after one agent
        calls.clear()
        graph.invoke({"question": "What is propolis in my guide?"})
        self.assertEqual(calls, ["books"])

        calls.clear()
        graph.invoke({"question": "Latest AI news this week"})
        self.assertEqual(calls, ["research"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
