"""Spec section 2.1 — mandatory RAG grounding behavior (deterministic).

Drives the REAL books_agent with a faked query-engine response (fixed
retrieved passages + fixed answer text). No embeddings, no LLM, no network,
no live Ollama. Verifies, for each evidence condition, both the answer
content and the reference handling, PLUS the adversarial conditions from the
review brief:

  1. fully unsupported question            -> mandated refusal, no footer
  2. relevant doc that does not answer      -> no topical cross-over tokens
  3. partially supported, one bogus claim   -> claim dropped + disclosure
  4. fabricated claim + refusal sentence    -> refusal only, no footer
  5. fully supported answer                 -> untouched, references kept
  6. refusal paraphrase                     -> refusal normalized
  7. empty / malformed output               -> mandated refusal with note
  8. reference handling per condition       -> only where claims are grounded

Grounding rule implemented in the agent: a substantive answer sentence is
kept only when every content token (numbers exact, content words after a
coarse plural stem) is present in the retrieved passages. That is a
deterministic support GATE over fixed evidence, not semantic entailment —
negation, synonyms, and token collisions inside large passages are known
limits, stated in the agent code.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.books_agent import books_agent  # noqa: E402
from app.config import NOT_FOUND_PHRASE  # noqa: E402

GUIDE = "urban_beekeeping_field_guide.pdf"
ORBITAL = "orbital_kinetics_principles.docx"

# FIXED retrieved passages (stand-ins for what the real retriever returns).
BEE_PASSAGE = (
    "Field Guide to Urban Beekeeping. Chapter 1: Colony Structure. "
    "A healthy hive in midsummer can contain up to 60,000 worker bees, plus "
    "one queen and a few hundred drones depending on the season. The queen "
    "lays up to 2,000 eggs per day at the peak of spring buildup."
)
PROPOLIS_PASSAGE = (
    "Chapter 3: Hive Products. Propolis is a resinous substance bees collect "
    "from plant buds and use to seal small gaps and sterilize the interior "
    "of the hive. Honey is harvested only when its moisture content drops "
    "below 18 percent, which prevents fermentation during storage."
)
ORBITAL_PASSAGE = (
    "Orbital Kinetics Principles. A Hohmann transfer between circular orbits "
    "requires two engine burns and depends on the delta-v budget of the "
    "spacecraft."
)

REFUSAL = "That requested info is not in the collection of books."
# (one paragraph: refusal-condition passages are not topical, but retrieval
#  is faked here, so a bee passage is still realistic for a BAD answer)


def _node(score, text=BEE_PASSAGE, file_name=GUIDE, page="1"):
    return SimpleNamespace(score=score,
                           metadata={"file_name": file_name, "page_label": page},
                           text=text)


class _FakeResponse:
    def __init__(self, text, nodes):
        self._text = text
        self.source_nodes = nodes

    def __str__(self):
        return self._text


class GroundingTestBase(unittest.TestCase):
    def setUp(self):
        import app.agents.books_agent as ba

        self._orig_get_query_engine = ba._get_query_engine

    def _run_agent(self, response, question="What does chapter 3 say about propolis?"):
        import app.agents.books_agent as ba

        ba._get_query_engine = lambda: SimpleNamespace(query=lambda q: response)
        try:
            return books_agent({"question": question})
        finally:
            ba._get_query_engine = self._orig_get_query_engine

    def _assert_refusal_exact(self, out):
        # The mandated phrase, optionally with the low-similarity score note
        # (that note is appended on the score-floor path only); NEVER a
        # references footer, never invented content.
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE),
                        out["answer"])
        self.assertNotIn("References:", out["answer"])
        self.assertEqual(out["sources"], ["books"])


class TestNoSupport(GroundingTestBase):
    """Condition 1: fully unsupported question."""

    def test_fully_unsupported_answer_degrades_to_refusal(self):
        # Model invents a confident business figure; the retrieved guide has
        # nothing about costs. Grounding gate must refuse, no footer.
        made_up = ("Urban beekeeping is highly profitable: startup costs "
                   "average $5,000 and annual profit is $12,000.")
        response = _FakeResponse(made_up, [_node(0.776)])
        self._assert_refusal_exact(self._run_agent(
            response,
            "What does it cost to start an urban beekeeping venture?"))

    def test_exact_refusal_without_footer(self):
        response = _FakeResponse(NOT_FOUND_PHRASE, [_node(0.35)])
        self._assert_refusal_exact(self._run_agent(response))

    def test_paraphrased_refusal_is_normalized_and_never_cites(self):
        response = _FakeResponse(
            "The information about economic feasibility was not found in "
            "your books.", [_node(0.78)])
        self._assert_refusal_exact(self._run_agent(response))

    def test_high_score_alone_is_never_evidence_of_an_answer(self):
        # A topical hit at high similarity without supporting passages is
        # still a refusal: similarity is not evidence.
        response = _FakeResponse(
            "The startup costs for a small apiary are surprisingly low "
            "compared with other ventures.", [_node(0.776)])
        self._assert_refusal_exact(self._run_agent(response))

    def test_low_score_floor_kept_with_mandated_phrase(self):
        response = _FakeResponse(
            "A midsummer hive may hold up to 60,000 worker bees.",
            [_node(0.20, text=BEE_PASSAGE)])
        out = self._run_agent(response)
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE))
        self.assertIn("score", out["answer"])  # honest low-similarity marker
        self.assertNotIn("References:", out["answer"])

    def test_no_support_invents_nothing(self):
        response = _FakeResponse(
            "The requested information was not found in the books.",
            [_node(0.776)])
        out = self._run_agent(response)
        self.assertNotIn("References:", out["answer"])
        self.assertNotIn("pdf", out["answer"])
        self.assertNotIn("60,000", out["answer"])

    def test_empty_or_blank_response_never_leaks_to_the_user(self):
        # Condition 7: empty or malformed model output.
        for text in ("", "   "):
            response = _FakeResponse(text, [_node(0.71, text=PROPOLIS_PASSAGE)])
            out = self._run_agent(response)
            self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE),
                            out["answer"])
            self.assertIn("score", out["answer"])
            self.assertNotIn("References:", out["answer"])


class TestParaphraseFamilies(GroundingTestBase):
    """Condition 6: refusal paraphrases must be caught."""

    def test_paraphrase_families_are_caught(self):
        for text in (
            "There is no information about quantum tunneling in the books.",
            "The passages do not contain the answer to this question.",
            "The field guide does not contain the answer.",
            "That requested info is not in the collection of books.",
        ):
            response = _FakeResponse(text, [_node(0.776)])
            with self.subTest(text=text):
                self._assert_refusal_exact(self._run_agent(response))


class TestConditionTwoRelevantNotAnswering(GroundingTestBase):
    """Condition 2: a relevant document whose content does not answer."""

    def test_topical_document_does_not_lend_its_content_to_an_off_question(self):
        # The guide is retrieved (topical), but the question asks about
        # spacecraft — the answer invented from general knowledge must not
        # pass the gate, even though a different, irrelevant passage was
        # ALSO retrieved (cross-document token borrowing is impossible: the
        # gate matches each sentence's tokens against the actual corpus).
        response = _FakeResponse(
            "A Hohmann transfer needs about 2.4 km/s of delta-v.",
            [_node(0.60, text=BEE_PASSAGE), _node(0.55, text=ORBITAL_PASSAGE,
                                                  file_name=ORBITAL, page="2")])
        out = self._run_agent(
            response, "How much delta-v does a Hohmann transfer need?")
        # Falls into the gate: the fabricated figure 2.4 is NOT in the
        # retrieved orbital passage (it says only "two engine burns"), so
        # nothing substantive survives -> refusal, no footer.
        self._assert_refusal_exact(out)


class TestPartialSupport(GroundingTestBase):
    """Conditions 3 + 8: partial support with one unsupported claim."""

    PARTIAL_WITH_BOGUS = (
        "Propolis is a resinous substance bees collect from plant buds and "
        "use to seal small gaps and sterilize the interior of the hive. "
        "Urban beekeeping generates about $15,000 in yearly profit for a "
        "small apiary. Honey is harvested only when its moisture content "
        "drops below 18 percent, which prevents fermentation during storage."
    )

    def test_unsupported_claim_dropped_supported_kept_disclosure_added(self):
        response = _FakeResponse(
            self.PARTIAL_WITH_BOGUS,
            [_node(0.74, text=PROPOLIS_PASSAGE),
             _node(0.66, text=BEE_PASSAGE)])
        out = self._run_agent(
            response, "What does the book say about propolis and honey?")
        answer = out["answer"]
        # Supported claims retained verbatim.
        self.assertIn("resinous substance bees collect", answer)
        self.assertIn("below 18 percent", answer)
        # Fabricated claim dropped (no general-knowledge gap-filling).
        self.assertNotIn("15,000", answer)
        # Missing part explicitly disclosed.
        self.assertIn("not found in the books", answer)
        # References still attached (they support the retained claims).
        self.assertIn(f"References: {GUIDE}, page 1", answer)
        self.assertEqual(out["sources"], ["books"])

    def test_fabricated_claim_followed_by_refusal_sentence(self):
        # Condition 4: a fabricated claim followed by a refusal sentence.
        # The gate sees the refusal sentence as shape, the fabricated
        # sentence as unsupported, and nothing else as supported content
        # -> mandates the refusal, no footer.
        mixed = ("Urban beekeeping yields a net margin of 35 percent. "
                 "The requested figures were not found in your books.")
        response = _FakeResponse(mixed, [_node(0.776)])
        self._assert_refusal_exact(self._run_agent(response))

    def test_partial_never_collapses_into_refusal(self):
        response = _FakeResponse(
            "The books cover colony structure only: a healthy midsummer hive "
            "may contain up to 60,000 worker bees. The books do not address "
            "the financial feasibility of an urban beekeeping venture.",
            [_node(0.74, text=BEE_PASSAGE)])
        out = self._run_agent(response)
        self.assertNotEqual(out["answer"], NOT_FOUND_PHRASE)
        self.assertIn("60,000 worker bees", out["answer"])
        self.assertIn("References:", out["answer"])


class TestSufficientSupport(GroundingTestBase):
    """Conditions 5 + 8: fully supported answer, references retained."""

    FULL = ("Propolis is a resinous substance bees collect from plant buds "
            "and use to seal small gaps and sterilize the hive interior; "
            "honey is harvested below 18 percent moisture.")

    def test_full_answer_retains_supporting_references(self):
        response = _FakeResponse(self.FULL,
                                 [_node(0.71, text=PROPOLIS_PASSAGE)])
        out = self._run_agent(response,
                              "What does the book say about propolis?")
        self.assertIn("Propolis", out["answer"])
        self.assertIn(f"References: {GUIDE}, page 1", out["answer"])
        self.assertEqual(out["sources"], ["books"])

    def test_above_threshold_is_never_refused(self):
        response = _FakeResponse(self.FULL,
                                 [_node(0.35, text=PROPOLIS_PASSAGE)])
        out = self._run_agent(response)
        self.assertIn("18 percent", out["answer"])
        self.assertNotIn(NOT_FOUND_PHRASE, out["answer"])


class TestAdversarialPolarityGates(GroundingTestBase):
    """Adversarial: token overlap must never be treated as proof of support.

    The polarity gates are monotone — they can only mark content
    UNSUPPORTED, never invent support — and fire only on unambiguous
    evidence (a single clause covering every content token, or a number
    whose own clause disagrees in polarity).
    """

    NEGATED = "Propolis is not used to seal small gaps in the hive."
    AFFIRMED = "Propolis is used to seal small gaps in the hive."
    CONFLICT_A = ("A healthy hive in midsummer can contain up to 60,000 "
                  "worker bees.")
    CONFLICT_B = "A healthy hive cannot exceed 20,000 worker bees."

    # --- case 1: negation reversal -------------------------------------
    def test_case1_evidence_negates_answer_asserts(self):
        response = _FakeResponse(self.AFFIRMED,
                                 [_node(0.77, text=self.NEGATED)])
        self._assert_refusal_exact(self._run_agent(response))

    def test_case1b_answer_negates_evidence_asserts(self):
        response = _FakeResponse(self.NEGATED,
                                 [_node(0.77, text=self.AFFIRMED)])
        self._assert_refusal_exact(self._run_agent(response))

    def test_case1c_agreeing_negation_is_not_over_refused(self):
        response = _FakeResponse(self.NEGATED,
                                 [_node(0.77, text=self.NEGATED)])
        out = self._run_agent(response)
        self.assertNotEqual(out["answer"], NOT_FOUND_PHRASE)
        self.assertIn("not used", out["answer"])
        self.assertIn("References:", out["answer"])

    # --- case 2: number collision / provenance --------------------------
    def test_case2_number_collision_financial_claim(self):
        # 2,000 exists in the passage (eggs/day) but the answer assigns it a
        # financial meaning; the fabricated surrounding vocabulary fails the
        # token gate -> refusal, no footer.
        response = _FakeResponse(
            "A single hive can generate 2,000 dollars per year in honey sales.",
            [_node(0.74, text=BEE_PASSAGE)])
        self._assert_refusal_exact(self._run_agent(response))

    def test_case2b_number_whose_own_clause_is_negated(self):
        # The figure's own evidence clause is negated ("cannot exceed") while
        # the answer asserts the figure as fact -> refused.
        response = _FakeResponse(
            "A healthy hive can contain up to 20,000 worker bees.",
            [_node(0.72, text=self.CONFLICT_A),
             _node(0.70, text=self.CONFLICT_B, file_name=ORBITAL, page="4")])
        self._assert_refusal_exact(self._run_agent(response))

    # --- case 3: conflicting passages -----------------------------------
    def test_case3_conflict_claim_with_affirmative_evidence_kept(self):
        """The safeguard does not itself pick a side of a corpus conflict.

        It refuses only claims whose OWN covering evidence negates them
        (case 2b); a claim its evidence affirms is kept. When two passages
        disagree with each other, the deterministic layer cannot adjudicate
        — that residual is reported, not hidden.
        """
        response = _FakeResponse(
            self.CONFLICT_A,
            [_node(0.72, text=self.CONFLICT_A),
             _node(0.70, text=self.CONFLICT_B, file_name=ORBITAL, page="4")])
        out = self._run_agent(response)
        self.assertIn("60,000", out["answer"])
        self.assertIn("References:", out["answer"])

    # --- case 4: partial support ----------------------------------------
    def test_case4_partial_supported_sentence_plus_unsupported_figure(self):
        response = _FakeResponse(
            "The queen lays up to 2,000 eggs per day. Urban beekeeping "
            "yields about $15,000 in yearly profit.",
            [_node(0.74, text=BEE_PASSAGE)])
        out = self._run_agent(response)
        answer = out["answer"]
        self.assertIn("2,000 eggs", answer)          # supported kept
        self.assertNotIn("15,000", answer)           # figure dropped
        self.assertIn("Note: the answer above is limited", answer)
        self.assertIn("References:", answer)

    # --- case 5: fully supported ----------------------------------------
    def test_case5_fully_supported_answer_and_references_intact(self):
        sentence = ("The queen lays up to 2,000 eggs per day at the peak of "
                    "spring buildup.")
        response = _FakeResponse(sentence, [_node(0.74, text=BEE_PASSAGE)])
        out = self._run_agent(response)
        self.assertEqual(
            out["answer"],
            sentence + f"\n\nReferences: {GUIDE}, page 1")

    # --- case 6: no evidence --------------------------------------------
    def test_case6_no_evidence_mandated_refusal(self):
        response = _FakeResponse(
            "Urban beekeeping is highly profitable with a 35 percent annual "
            "margin.",
            [_node(0.71, text=BEE_PASSAGE)])
        self._assert_refusal_exact(self._run_agent(response))

    # --- documented limitation (NOT a guarantee) -------------------------
    def test_documented_limitation_cross_passage_splice_not_detected(self):
        """KNOWN GAP, adversarial: a claim splicing tokens from UNRELATED
        passages still passes.

        Closing it deterministically would require single-passage-sentence
        coverage, which over-refuses legitimate combined summaries ("a hive
        holds 60,000 bees and the queen lays 2,000 eggs daily"); real
        entailment needs a separate verifier model/dependency — deliberately
        not added. This test pins the CURRENT behavior so the gap stays
        visible and any future strengthening breaks loudly here.
        """
        response = _FakeResponse(
            "Propolis is harvested below 18 percent moisture.",
            [_node(0.70, text="Propolis is a resinous substance bees collect "
                              "from plant buds."),
             _node(0.68, text="Honey is harvested only when its moisture "
                              "content drops below 18 percent.",
                   file_name=GUIDE, page="1")])
        out = self._run_agent(response)
        self.assertIn("Propolis is harvested", out["answer"])  # current gap


if __name__ == "__main__":
    unittest.main(verbosity=2)
