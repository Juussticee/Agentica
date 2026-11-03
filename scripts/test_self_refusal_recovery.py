"""M5 — bounded recovery from a pure books-model self-refusal (deterministic).

Drives the REAL books_agent with a faked query engine AND a SCRIPTED fake
LLM patched at `app.agents.books_agent.get_llm`. No Ollama, no network, no
Chroma (the fail-closed conftest guard stays armed).

Pins the recovery contract from the implementation:
  * eligibility = pure `_refusal_only` response + usable source nodes with
    text + score >= BOOKS_MIN_SCORE + question-to-evidence topical coverage;
  * at most ONE regeneration, from the ORIGINAL retrieved passages, and its
    result re-enters the UNCHANGED grounding/polarity/numeric/citation gates;
  * every failure mode (second refusal, unsupported, fabricated numbers,
    polarity reversal, invoke error) abstains with the existing not-found
    behavior — never a third attempt, never a weakened gate;
  * non-refusal answers generate exactly once (no extra LLM call);
  * the branch markers are asserted so eligibility, retry, grounding
    rejection and final abstention are distinguishable — while raw model
    responses and passage text are never part of any marker.
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.books_agent import (  # noqa: E402
    _question_covered_by_evidence,
    books_agent,
)
from app.config import BOOKS_MIN_SCORE, NOT_FOUND_PHRASE  # noqa: E402

ORBITAL = "orbital_kinetics_principles.docx"
ORBITAL_PASSAGE = (
    "Orbital Kinetics Principles. A Hohmann transfer between circular orbits "
    "requires two engine burns and depends on the delta-v budget of the "
    "spacecraft."
)
DELTA_QUESTION = (
    "According to orbital_kinetics_principles.docx, what does delta-v mean, "
    "and how is it used in orbital mechanics?"
)

# Grounded in ORBITAL_PASSAGE (every content token evidenced; no numbers).
RECOVERED = (
    "A Hohmann transfer between circular orbits requires two engine burns "
    "and depends on the delta-v budget of the spacecraft."
)
REFUSAL = NOT_FOUND_PHRASE


def _node(score, text=ORBITAL_PASSAGE, file_name=ORBITAL, page="3"):
    return SimpleNamespace(score=score,
                           metadata={"file_name": file_name, "page_label": page},
                           text=text)


class _FakeResponse:
    def __init__(self, text, nodes):
        self._text = text
        self.source_nodes = nodes

    def __str__(self):
        return self._text


class _FakeEngine:
    """One scripted retrieval per agent run; `Exception` entries raise."""

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def query(self, question):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _ChatMsg:
    """Minimal LangChain-shaped message (llm_text reads `.content`)."""

    def __init__(self, text):
        self.content = text


class _ScriptedLLM:
    """Counts every regeneration attempt and records prompts (never printed)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("recovery attempt beyond the scripted bound")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _ChatMsg(reply)


class RecoveryTestBase(unittest.TestCase):
    def setUp(self):
        import app.agents.books_agent as ba

        self.ba = ba
        self._orig_engine = ba._get_query_engine
        self._orig_get_llm = ba.get_llm

    def tearDown(self):
        self.ba._get_query_engine = self._orig_engine
        self.ba.get_llm = self._orig_get_llm

    def _run_agent(self, response, question=DELTA_QUESTION, replies=()):
        """Returns (out, stdout_text); exposes self.llm / self.engine."""
        self.llm = _ScriptedLLM(list(replies))
        self.engine = _FakeEngine(response)
        self.ba._get_query_engine = lambda: self.engine
        self.ba.get_llm = lambda: self.llm
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self.ba.books_agent({"question": question})
        return out, buf.getvalue()

    def _assert_refusal(self, out):
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE),
                        out["answer"])
        self.assertNotIn("References:", out["answer"])
        self.assertEqual(out["sources"], ["books"])


class TestEligibleRecovery(RecoveryTestBase):
    """Cases 1 + 2: eligible self-refusal -> exactly one grounded retry."""

    def test_eligible_self_refusal_triggers_exactly_one_recovery_attempt(self):
        out, log = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                   replies=[RECOVERED])
        self.assertEqual(self.engine.calls, 1)      # no second retrieval
        self.assertEqual(self.llm.calls, 1)         # exactly one regeneration
        self.assertIn("self_refusal_recovery_attempted", log)
        self.assertIn("self_refusal_recovery_succeeded", log)
        self.assertIn("two engine burns", out["answer"])

    def test_successful_recovery_returns_grounded_answer_with_valid_citations(self):
        out, _ = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                 replies=[RECOVERED])
        self.assertEqual(
            out["answer"],
            RECOVERED + f"\n\nReferences: {ORBITAL}, page 3")
        self.assertEqual(out["sources"], ["books"])

    def test_retry_prompt_carries_the_original_retrieved_evidence(self):
        # The regeneration must reuse the REAL retrieved passage text and
        # the original question — never fabricated or reconstructed context.
        _, log = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                 replies=[RECOVERED])
        self.assertEqual(len(self.llm.prompts), 1)
        prompt = self.llm.prompts[0]
        self.assertIn(ORBITAL_PASSAGE, prompt)
        self.assertIn(DELTA_QUESTION, prompt)
        self.assertNotIn("file_path", prompt)
        # Diagnostics never leak the model response or passage text.
        self.assertNotIn(RECOVERED, log)
        self.assertNotIn(ORBITAL_PASSAGE, log)


class TestRetryBoundAndAbstention(RecoveryTestBase):
    """Cases 3 + 13: second refusal -> abstain; never a third attempt."""

    def test_second_refusal_final_output_is_not_found_phrase(self):
        out, log = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                   replies=[REFUSAL])
        self._assert_refusal(out)
        self.assertIn("self_refusal_recovery_rejected (reason=second_refusal",
                      log)

    def test_retry_bound_never_exceeds_one(self):
        # Third scripted reply would be used only by a second retry; its
        # presence proves the bound independently of the call counter.
        out, _ = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                 replies=[REFUSAL, REFUSAL, RECOVERED])
        self.assertEqual(self.llm.calls, 1)
        self.assertEqual(self.engine.calls, 1)
        self._assert_refusal(out)


class TestEligibilityGates(RecoveryTestBase):
    """Cases 4 + 5 + 6 + 7: ineligible situations get ZERO recovery attempts."""

    def test_unsupported_question_gets_zero_recovery_attempts(self):
        out, log = self._run_agent(
            _FakeResponse(REFUSAL, [_node(0.78)]),
            question="What is quantum tunneling in particle physics?")
        self.assertEqual(self.llm.calls, 0)
        self.assertNotIn("self_refusal_recovery_attempted", log)
        self._assert_refusal(out)

    def test_low_score_gets_zero_recovery_attempts(self):
        out, log = self._run_agent(
            _FakeResponse(REFUSAL, [_node(BOOKS_MIN_SCORE - 0.05)]),
            replies=[RECOVERED])
        self.assertEqual(self.llm.calls, 0)
        self.assertNotIn("self_refusal_recovery_attempted", log)
        self._assert_refusal(out)

    def test_empty_retrieval_gets_zero_recovery_attempts(self):
        for nodes in ([], [_node(0.78, text="")]):
            with self.subTest(nodes=nodes):
                out, log = self._run_agent(_FakeResponse(REFUSAL, nodes),
                                           replies=[RECOVERED])
                self.assertEqual(self.llm.calls, 0)
                self.assertNotIn("self_refusal_recovery_attempted", log)
                self._assert_refusal(out)

    def test_retrieval_exception_behavior_is_preserved(self):
        # Recovery lives AFTER a successful retrieval; an engine exception
        # must propagate exactly as before, with zero regeneration attempts.
        with self.assertRaises(RuntimeError):
            self._run_agent(RuntimeError("index offline"),
                            replies=[RECOVERED])
        self.assertEqual(self.llm.calls, 0)
        self.assertEqual(self.engine.calls, 1)


class TestRecoveredAnswerIsFullyGated(RecoveryTestBase):
    """Cases 8 + 9 + 10 + 11: the recovered text is untrusted model output."""

    def test_unsupported_recovered_answer_abstains(self):
        fabricated = ("The delta-v budget of a Hohmann transfer is 9.4 "
                      "million dollars per mission.")
        out, log = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                   replies=[fabricated])
        self.assertEqual(self.llm.calls, 1)
        self.assertIn("self_refusal_recovery_succeeded", log)
        self.assertNotIn("9.4", out["answer"])
        self._assert_refusal(out)          # grounding gate -> abstain

    def test_recovered_numeric_fabrication_remains_rejected(self):
        # Changed value (5 vs the evidenced "two") and a fabricated unit
        # ("kilometers per second" is not in the passages) both fail the
        # numeric/token gates -> nothing substantiable -> abstain.
        for bad in (
            "A Hohmann transfer requires 5 engine burns and depends on the "
            "delta-v budget of the spacecraft.",
            "A Hohmann transfer needs about 2.4 kilometers per second of "
            "delta-v between circular orbits.",
        ):
            with self.subTest(bad=bad):
                out, _ = self._run_agent(
                    _FakeResponse(REFUSAL, [_node(0.78)]), replies=[bad])
                self.assertNotIn("5 engine burns", out["answer"])
                self.assertNotIn("kilometers", out["answer"])
                self._assert_refusal(out)

    def test_recovered_polarity_reversal_is_rejected(self):
        reversed_claim = ("A Hohmann transfer does not require the delta-v "
                          "budget of the spacecraft.")
        out, _ = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                 replies=[reversed_claim])
        self.assertNotIn("does not require", out["answer"])
        self._assert_refusal(out)

    def test_recovered_citations_stay_tied_to_actual_retrieved_nodes(self):
        smuggled = (RECOVERED
                    + " The transfer is described on page 7 of "
                    + "other_book.pdf.")
        out, _ = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                 replies=[smuggled])
        # The fabricated page/filename sentences are dropped; the agent's
        # own footer cites ONLY the real retrieved node.
        self.assertNotIn("page 7", out["answer"])
        self.assertNotIn("other_book.pdf", out["answer"])
        self.assertIn("two engine burns", out["answer"])
        self.assertIn(f"References: {ORBITAL}, page 3", out["answer"])

    def test_recovery_invoke_error_abstains_silently(self):
        out, log = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                   replies=[RuntimeError("llm down")])
        self.assertEqual(self.llm.calls, 1)
        self.assertIn("self_refusal_recovery_rejected (reason=empty_or_error",
                      log)
        self._assert_refusal(out)


class TestNormalPathUnchanged(RecoveryTestBase):
    """Cases 12 + 14: non-refusal answers never pay for the recovery path."""

    def test_normal_supported_answer_generates_no_extra_call(self):
        out, log = self._run_agent(_FakeResponse(RECOVERED, [_node(0.78)]))
        self.assertEqual(self.llm.calls, 0)          # no extra generation
        self.assertEqual(self.engine.calls, 1)
        self.assertNotIn("self_refusal_recovery", log)
        self.assertEqual(
            out["answer"],
            RECOVERED + f"\n\nReferences: {ORBITAL}, page 3")

    def test_mixed_response_keeps_supported_content_without_recovery(self):
        mixed = (RECOVERED
                 + " The requested information was not found in the books.")
        out, log = self._run_agent(_FakeResponse(mixed, [_node(0.78)]))
        # Not a PURE refusal -> no recovery; the existing gate keeps the
        # supported sentence and discloses the dropped marker sentence.
        self.assertEqual(self.llm.calls, 0)
        self.assertNotIn("self_refusal_recovery_attempted", log)
        self.assertIn("two engine burns", out["answer"])
        self.assertIn("Note: the answer above is limited", out["answer"])
        self.assertIn(f"References: {ORBITAL}, page 3", out["answer"])
        self.assertNotEqual(out["answer"], NOT_FOUND_PHRASE)


class TestCoverageGateUnit(RecoveryTestBase):
    """Branch-level assertions on the eligibility coverage check itself."""

    def test_coverage_discriminates_topical_and_off_topic_questions(self):
        node = _node(0.78)
        self.assertTrue(_question_covered_by_evidence(DELTA_QUESTION, [node]))
        self.assertFalse(_question_covered_by_evidence(
            "What is quantum tunneling in particle physics?", [node]))
        self.assertFalse(_question_covered_by_evidence("", [node]))
        self.assertFalse(_question_covered_by_evidence(DELTA_QUESTION, []))

    def test_question_structure_words_no_longer_block_eligibility(self):
        # A6 GUI shape (2026-10-25): coverage was 0.45 solely because the
        # evidence never echoes interrogatives/task verbs ("how", "what",
        # summarize) — question-structure words are no longer counted.
        node = _node(0.81, text=(
            "For AI systems to be trustworthy, they often need to be "
            "responsive to a multiplicity of criteria. Appendix D: "
            "Attributes of the AI RMF. The Framework provides four "
            "functions: GOVERN, MAP, MEASURE, and MANAGE, with "
            "characteristics of trustworthy AI."))
        question = ("Using this document, summarize how the framework "
                    "connects its overall purpose, its core functions, and "
                    "the characteristics of trustworthy AI.")
        self.assertTrue(_question_covered_by_evidence(question, [node]))

    def test_missing_detail_question_still_blocked_from_recovery(self):
        # B1 GUI shape: the bee chunk is topically close but lacks
        # "supers"/"inspecting"/"recommend" — eligibility must NOT be
        # over-granted (the abstention stays correct).
        node = _node(0.783, text=(
            "Field Guide to Urban Beekeeping. A healthy hive in midsummer "
            "can contain up to 60,000 worker bees, plus one queen. Honey "
            "is harvested only when its moisture content drops below 18 "
            "percent."))
        question = ("What does the field guide recommend for inspecting a "
                    "hive before adding honey supers?")
        self.assertFalse(_question_covered_by_evidence(question, [node]))

    def test_recovery_prompt_carries_the_partial_coverage_clause(self):
        # 2026-10-25: the retry prompt lacked the main prompt's
        # partial-coverage clause, so a weak model could only choose
        # between full answer and NOT_FOUND (observed: A1/A4 retries
        # re-refused). The clause must be present verbatim.
        out, _ = self._run_agent(_FakeResponse(REFUSAL, [_node(0.78)]),
                                 replies=[RECOVERED])
        self.assertEqual(len(self.llm.prompts), 1)
        prompt = self.llm.prompts[0]
        self.assertIn("state ONLY the covered facts from the passages",
                      prompt)
        self.assertIn("do not claim the whole topic is absent", prompt)
        # NOT_FOUND stays reserved for passages that support nothing.
        self.assertIn(NOT_FOUND_PHRASE, prompt)


class TestUnrelatedRoutesUnchanged(RecoveryTestBase):
    """Recovery is books-only: the deterministic ERP route must be exactly
    its documented behavior with zero books-agent activity."""

    def test_erp_route_is_unaffected_by_recovery(self):
        import app.agents.erp_agent as ea

        def _boom(*args, **kwargs):
            raise AssertionError("books-agent recovery path reached a "
                                 "non-books route")

        orig_engine = self.ba._get_query_engine
        orig_get_llm = self.ba.get_llm
        self.ba._get_query_engine = _boom
        self.ba.get_llm = _boom
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                out = ea.erp_agent({"question": "Show me open invoices"})
        finally:
            self.ba._get_query_engine = orig_engine
            self.ba.get_llm = orig_get_llm
        log = buf.getvalue()
        self.assertEqual(out["sources"], ["erp"])
        self.assertIn("ERP lookup results", out["answer"])
        self.assertIn("mock ERP adapter", out["answer"])
        self.assertNotIn("self_refusal_recovery", log)
        self.assertNotIn("books_agent", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
