"""Feasibility graceful-degradation tests — fake LLM, zero Gemini quota.

Covers:
  1. books not-found refusal  -> web-only study, flagged, Limitations in .docx
  2. books answer missing key -> web-only study, flagged
  3. books placeholder string -> web-only study, flagged
  4. books real content       -> normal study, NO flag, NO Limitations section
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


STUDY = (
    "## Market\nWeb findings show growing demand.\n\n"
    "## Financial\nCost data gap noted.\n\n"
    "## Risk\nAdoption uncertainty.\n"
)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        return _Msg(STUDY)


def run_case(books_answer, question="Feasibility of an urban apiary?"):
    import app.agents.feasibility_agent as fa

    fake = FakeLLM()
    fa.get_llm = lambda: fake
    with tempfile.TemporaryDirectory() as tmp:
        original = fa.REPORTS_DIR
        fa.REPORTS_DIR = Path(tmp)
        try:
            state = {
                "question": question,
                "route": "feasibility",
                "sub_routes": ["research", "books"],
                "agent_answers": {"research": "Web says demand is up."},
                "sources": ["research"],
            }
            if books_answer is not None:
                state["agent_answers"]["books"] = books_answer
            result = fa.feasibility_agent(state)

            # parse the .docx while the temp dir still exists
            docx_path = Path(result["answer"].split("saved to: ")[1].split("\n")[0].strip())
            doc = DocxReader(str(docx_path))
            headings = [
                p.text for p in doc.paragraphs if p.style.name.startswith("Heading")
            ]
        finally:
            fa.REPORTS_DIR = original

    return fake, result, headings


class TestDegradation(unittest.TestCase):
    def test_web_only_on_refusal(self):
        from app.config import NOT_FOUND_PHRASE

        fake, result, headings = run_case(
            f"{NOT_FOUND_PHRASE} (best similarity score 0.61)"
        )
        prompt = fake.prompts[0]
        self.assertIn("NO relevant material", prompt)          # web-only template
        self.assertNotIn("Findings from the book/company library:", prompt)
        self.assertIn("Limitations", headings)
        for s in ("Market", "Financial", "Risk"):
            self.assertIn(s, headings)
        self.assertTrue(result["answer"].startswith("⚠"))
        self.assertIn("Web-only study", result["answer"])

    def test_web_only_on_missing_books(self):
        fake, result, headings = run_case(None)
        self.assertIn("NO relevant material", fake.prompts[0])
        self.assertTrue(result["answer"].startswith("⚠"))

    def test_web_only_on_placeholder(self):
        fake, result, headings = run_case("(no books output available)")
        self.assertIn("NO relevant material", fake.prompts[0])
        self.assertTrue(result["answer"].startswith("⚠"))

    def test_normal_path_unchanged(self):
        fake, result, headings = run_case(
            "The field guide explains hive economics and startup costs in detail."
        )
        prompt = fake.prompts[0]
        self.assertIn("Findings from the book/company library:", prompt)
        self.assertIn("hive economics", prompt)                # books text included
        self.assertNotIn("NO relevant material", prompt)
        self.assertNotIn("Limitations", headings)
        self.assertFalse(result["answer"].startswith("⚠"))
        for s in ("Market", "Financial", "Risk"):
            self.assertIn(s, headings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
