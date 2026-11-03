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


class TestOutputFormattingIntegrity(GroundingTestBase):
    """Phase 4 of the approved RAG fixes: output-handling integrity.

    The 2026-10-24 verification established that the degenerate GUI
    formatting (collapsed list numbering, mid-sentence starts) originates
    with the 0.6B model, NOT in the app's handling. These tests pin that the
    handling provably never introduces the defects itself:

      * whole sentences are kept in order, never truncated or re-split;
      * the References footer is separated from the answer by a blank line;
      * disclosures appear only when something was dropped, and only once;
      * formatting adds no content beyond input sentences + fixed notes.
    """

    MULTI_PART = (
        "The queen lays up to 2,000 eggs per day. A healthy hive can contain "
        "60,000 worker bees."
    )

    def test_whole_sentences_kept_in_order_never_truncated(self):
        response = _FakeResponse(
            self.MULTI_PART,
            [_node(0.74, text=BEE_PASSAGE),
             _node(0.66, text="A healthy hive in midsummer can contain up "
                              "to 60,000 worker bees.",
                   file_name=ORBITAL, page="3")])
        state = self._run_agent(response)
        body = state["answer"].split("\n\nReferences:")[0]
        # Both parts survive as whole sentences, in input order, no drops.
        self.assertNotIn("Note: the answer above is limited", body)
        self.assertEqual(body, self.MULTI_PART, body)
        # No mid-sentence start: the first character is uppercase.
        self.assertTrue(body[0].isupper(), body)

    def test_footer_separated_by_blank_line(self):
        response = _FakeResponse(
            self.MULTI_PART,
            [_node(0.74, text=BEE_PASSAGE),
             _node(0.66, text="A healthy hive in midsummer can contain up "
                              "to 60,000 worker bees.",
                   file_name=ORBITAL, page="3")])
        out = self._run_agent(response)
        self.assertIn("\n\nReferences: ", out["answer"])
        # The footer is exactly one blank-line-separated trailing block.
        parts = out["answer"].split("\n\nReferences: ")
        self.assertEqual(len(parts), 2, out["answer"])
        self.assertNotIn("References:", parts[0])
        self.assertNotIn("References:", parts[1])

    def test_disclosure_appended_once_and_only_on_drops(self):
        # Drop case: exactly one disclosure, appended after the kept text.
        response = _FakeResponse(
            self.MULTI_PART.split(". ")[0] + ". Urban beekeeping yields "
            "about $15,000 in yearly profit.",
            [_node(0.74, text=BEE_PASSAGE)])
        out = self._run_agent(response)
        answer = out["answer"]
        self.assertEqual(
            answer.count("Note: the answer above is limited"), 1,
            out["answer"])
        self.assertIn("the rest was not found in the books.", answer)
        # The validated footer is still the final block after the disclosure.
        self.assertTrue(
            answer.endswith(f"\n\nReferences: {GUIDE}, page 1"),
            out["answer"])
        self.assertNotIn("15,000", answer)
        # Clean case: no disclosure at all.
        clean = _FakeResponse(
            self.MULTI_PART.split(". ")[0] + ".",
            [_node(0.74, text=BEE_PASSAGE)])
        self.assertNotIn("Note: the answer",
                         self._run_agent(clean)["answer"])

    def test_formatting_never_adds_content(self):
        content_words = ("queen", "eggs", "profit", "15,000", "drones")
        response = _FakeResponse(
            self.MULTI_PART.split(". ")[0] + ". Urban beekeeping yields "
            "about $15,000 in yearly profit.",
            [_node(0.74, text=BEE_PASSAGE)])
        answer = self._run_agent(response)["answer"]
        on_topic = [w for w in content_words if w in answer]
        # Every content word that reached the output was in the input or the
        # fixed disclosure; nothing new was introduced by formatting.
        self.assertEqual(on_topic, ["queen", "eggs"], on_topic)
        self.assertNotIn("drones", answer)


# --- 2026-10-25 fix cycle fixtures: VERBATIM source text from
# knowledge/books/NIST.AI.100-1.pdf (pypdf extraction; whitespace-collapsed;
# source hyphenation preserved). Each string is a verbatim prefix of the page.
GOVERN_PAGE22 = (
    "NIST AI 100-1 AI RMF 1.0 GOVERN is a cross-cutting function that is "
    "infused throughout AI risk management and enables the other functions "
    "of the process. Aspects of GOVERN , especially those related to "
    "compliance or evaluation, should be integrated into each of the other "
    "functions. Attention to governance is a continual and intrinsic "
    "requirement for effective AI risk management over an AI system's "
    "lifespan and the organization's hierarchy. Strong governance can drive "
    "and enhance internal practices and norms to facilitate orga- nizational "
    "risk culture. Governing authorities can determine the overarching "
    "policies that direct an organization's mission, goals, values, culture, "
    "and risk tolerance. Senior leader- ship sets the tone for risk "
    "management within an organization, and with it, organizational culture."
)
MAP_PAGE25 = (
    "NIST AI 100-1 AI RMF 1.0 This complexity and varying levels of "
    "visibility can introduce uncertainty into risk man- agement practices. "
    "Anticipating, assessing, and otherwise addressing potential sources of "
    "negative risk can mitigate this uncertainty and enhance the integrity "
    "of the decision process. The information gathered while carrying out "
    "the MAP function enables negative risk pre- vention and informs "
    "decisions for processes such as model management, as well as an initial "
    "decision about appropriateness or the need for an AI solution. Outcomes "
    "in the MAP function are the basis for the MEASURE and MANAGE functions. "
    "The MAP function is intended to enhance an organization's ability to "
    "identify risks and broader contributing factors."
)
MEASURE_PAGE29 = (
    "NIST AI 100-1 AI RMF 1.0 Practices related to measuring AI risks are "
    "described in the NIST AI RMF Playbook. Table 3 lists the MEASURE "
    "function's categories and subcategories. Table 3: Categories and "
    "subcategories for the MEASURE function. MEASURE 1: Appropriate methods "
    "and metrics are identified and applied."
)
DISCLAIMER_PAGE_I = (
    "Certain commercial entities, equipment, or materials may be identified "
    "in this document in order to describe an experimental procedure or "
    "concept adequately. Such identification is not intended to imply "
    "recommenda- tion or endorsement by the National Institute of Standards "
    "and Technology, nor is it intended to imply that the entities, "
    "materials, or equipment are necessarily the best available for the "
    "purpose. This publication is available free of charge from: "
    "https://doi.org/10.6028/NIST.AI.100-1"
)

# The RAW A5 model output captured offline on 2026-10-25 (local qwen3:0.6b,
# real retrieval, score 0.736) — kept verbatim as the regression input.
A5_RAW = (
    "The document states that the Govern function involves mapping, "
    "measuring, and managing AI risks, with specific subcategories like "
    "legal regulations, trustworthy AI characteristics, and processes for "
    "risk tolerance. "
    "It emphasizes that GOVERN is a cross-cutting function that integrates "
    "compliance and evaluation into AI risk management, with governance "
    "driving organizational culture. "
    "The function's content relates to other functions such as MAP "
    "(processes for operator and practitioner proficiency) and "
    "interdisciplinarity. "
    "The answer is derived from the context provided."
)
A5_CROSS_CUTTING = (
    "It emphasizes that GOVERN is a cross-cutting function that integrates "
    "compliance and evaluation into AI risk management, with governance "
    "driving organizational culture."
)

# --- 2026-10-27 (authorized W1): the three REMAINING diagnosed false
# refusals, pinned against the real retriever output. Fixtures are verbatim
# prefixes of the actual top-1 chunks (whitespace normalized only, exactly
# how the other fixtures in this file are kept), dumped read-only from the
# production index by .diag/w1_diagnosis/dump_nodes.py.
A1_QUESTION = (
    "I want to understand the NIST AI Risk Management Framework. What is "
    "this document about, and what are its main parts?"
)
A1_MAIN_PARTS = (
    "The NIST AI Risk Management Framework is about managing AI risks, with "
    "its main parts including the Core and functions GOVERN, MAP, MEASURE, "
    "and MANAGE."
)
A1_PAGE3 = (
    'NIST AI 100-1 AI RMF 1.0 valid and reliable, safe, secure and resilient,'
    + ' accountable and transparent, explainable and interpretable, privacy enh'
    + 'anced, and fair with their harmful biases managed. Part 2 comprises the '
    + '\u201cCore\u201d of the Framework. It describes four specific functions to help or'
    + 'ganizations address the risks of AI systems in practice. These functions'
    + ' \u2013 GOVERN , MAP, MEASURE , and MANAGE \u2013 are broken down further into cat'
    + 'egories and subcate- gories. While GOVERN applies to all stages of organ'
    + 'izations\u2019 AI risk management pro- cesses and procedures, the MAP, MEASUR'
    + 'E , and MANAGE functions can be applied in AI system-specific contexts a'
    + 'nd at specific stages of the AI lifecycle. Additional resources related '
    + 'to the Framework are included'
)
A2_PAGE12 = (
    'NIST AI 100-1 AI RMF 1.0 3. AI Risks and Trustworthiness For AI systems '
    + 'to be trustworthy, they often need to be responsive to a multiplicity of'
    + ' cri- teria that are of value to interested parties. Approaches which en'
    + 'hance AI trustworthiness can reduce negative AI risks. This Framework ar'
    + 'ticulates the following characteristics of trustworthy AI and offers gui'
    + 'dance for addressing them. Characteristics of trustworthy AI systems inc'
    + 'lude: valid and reliable, safe, secure and resilient, accountable and tr'
    + 'ans- parent, explainable and interpretable, privacy-enhanced, and fair w'
    + 'ith harmful bias managed.'
)
# The two A2 sentences the GUI `gate_drop:` lines dropped, on the single
# tokens "essential" and "highlight" respectively (the report truncates them
# with "..."; both are written here from the recorded prefix plus the
# surrounding CLAIM, and every content token is checked against A2_PAGE12
# by test_a2_essential_and_highlight_sentences_survive_the_gate).
A2_ESSENTIAL = (
    "The framework explains that trustworthy AI characteristics such as "
    "valid and reliable are essential for trustworthiness."
)
A2_HIGHLIGHT = (
    "The document highlights how approaches which enhance AI trustworthiness "
    "can reduce negative AI risks."
)
T3_QUESTION = (
    "Using only this document, explain the difference between the Map and "
    "Measure functions. Give one concrete activity associated with each and "
    "cite the supporting evidence."
)
T3_TABLE2_PAGE28 = (
    'NIST AI 100-1 AI RMF 1.0 Table 2: Categories and subcategories for the M'
    + 'AP function. (Continued) MAP 5.2: Practices and personnel for supporting'
    + ' regular en- gagement with relevant AI actors and integrating feedback a'
    + 'bout positive, negative, and unanticipated impacts are in place and docu'
    + 'mented. Categories Subcategories 5.3 Measure The MEASURE function employ'
    + 's quantitative, qualitative, or mixed-method tools, tech- niques, and me'
    + 'thodologies to analyze, assess, benchmark, and monitor AI risk and relat'
    + 'ed impacts. It uses knowledge relevant to AI risks identified in the MAP'
    + ' function and informs the MANAGE function. AI systems should be tested b'
    + 'efore their deployment and regu- larly while in operation. AI risk measu'
    + 'rements include documenting aspects of systems\u2019 functionality and trustw'
    + 'orthiness. Measuring AI risks includes tracking metrics for trustworthy '
    + 'characteristics, social impact, and human-AI configurations. Processes d'
    + 'eveloped or adopted in the MEASURE function should include rigorous soft'
    + 'ware testing and performance assessment methodologies with associated'
)


class TestRealSourceFalseRefusalFixes(GroundingTestBase):
    """2026-10-25 fix cycle, Phase 1: the verified GUI false-refusal shapes
    against VERBATIM NIST source text (no LLM, no Chroma, fixed passages)."""

    def _gov_node(self, score=0.736):
        return _node(score, text=GOVERN_PAGE22,
                     file_name="NIST.AI.100-1.pdf", page="22")

    def test_a5_cross_cutting_sentence_survives_the_gate(self):
        # Offline repro of the live A5 refusal: with "emphasiz*" treated as
        # a non-assertive reporting word, the evidenced cross-cutting
        # sentence is KEPT, while the unevidenced embellishment ("legal
        # regulations" — absent from this page) is still dropped.
        import app.agents.books_agent as ba

        mode, kept, claims = ba._grounded_repair(A5_RAW, [self._gov_node()])
        self.assertEqual(mode, "partial", mode)
        self.assertIn(A5_CROSS_CUTTING, kept)
        self.assertFalse(any("legal regulations" in s for s in kept), kept)

    def test_a5_agent_answers_with_govern_evidence_instead_of_refusing(self):
        # Same shape end-to-end through the REAL books_agent: the GUI run
        # returned the mandated refusal here; now the supported content
        # must reach the user with its disclosure and validated footer.
        response = _FakeResponse(A5_RAW, [self._gov_node()])
        out = self._run_agent(
            response,
            question="What does the document say about the Govern function, "
                     "and how does it relate to the other functions?")
        # NOT a refusal: the partial disclosure note (which itself says
        # "not found") is expected and correct here.
        self.assertFalse(out["answer"].startswith(NOT_FOUND_PHRASE),
                         out["answer"])
        self.assertIn("cross-cutting function that integrates compliance",
                      out["answer"])
        self.assertIn("Note: the answer above is limited", out["answer"])
        self.assertIn("References: NIST.AI.100-1.pdf, page 22", out["answer"])

    def test_a1_main_parts_sentence_survives_the_gate(self):
        # GUI A1 (2026-10-25/27): the bounded retry SUCCEEDED, then the gate
        # dropped the one fully evidenced sentence on the single token
        # "main" (recorded: `missing tokens: main`). Every other content
        # token of that sentence is verbatim in the retrieved page-3 chunk.
        import app.agents.books_agent as ba

        node = _node(0.825, text=A1_PAGE3,
                     file_name="NIST.AI.100-1.pdf", page="3")
        mode, kept, claims = ba._grounded_repair(A1_MAIN_PARTS, [node])
        self.assertEqual(mode, "ok", (mode, kept))
        self.assertIn(A1_MAIN_PARTS, kept)
        self.assertEqual(len(claims), 1, claims)

    def test_a1_agent_answers_with_retrieved_evidence_instead_of_refusing(self):
        # Same shape end-to-end through the REAL books_agent: the GUI run
        # ended in the mandated refusal; the evidenced content must now
        # reach the user with its validated footer.
        node = _node(0.825, text=A1_PAGE3,
                     file_name="NIST.AI.100-1.pdf", page="3")
        out = self._run_agent(_FakeResponse(A1_MAIN_PARTS, [node]),
                              question=A1_QUESTION)
        self.assertFalse(out["answer"].startswith(NOT_FOUND_PHRASE),
                         out["answer"])
        self.assertIn("main parts including the Core", out["answer"])
        self.assertIn("References: NIST.AI.100-1.pdf, page 3", out["answer"])

    def test_a2_essential_and_highlight_sentences_survive_the_gate(self):
        # GUI A2 (2026-10-27): `gate_drop` named exactly these two tokens on
        # two otherwise evidenced sentences ("missing tokens: essential" /
        # "highlight"). Both are reporting/meta scaffolding of the same
        # class as the already-fixed "emphasiz*".
        import app.agents.books_agent as ba

        node = _node(0.858, text=A2_PAGE12,
                     file_name="NIST.AI.100-1.pdf", page="12")
        for sentence in (A2_ESSENTIAL, A2_HIGHLIGHT):
            with self.subTest(sentence=sentence):
                mode, kept, claims = ba._grounded_repair(sentence, [node])
                self.assertEqual(mode, "ok", (mode, kept))
                self.assertIn(sentence, kept)
                self.assertEqual(len(claims), 1, claims)

    def test_essentially_adverb_does_not_refuse_an_evidenced_sentence(self):
        # W2.1 review case: "essentially" (the adverbial form of the already
        # non-assertive "essential") was the ONLY token standing between a
        # fully evidenced sentence and the mandated refusal — the same
        # sentence against the same verbatim page-12 chunk returned
        # mode="ok" without it and mode="refusal" with it (offline repro:
        # .diag/w1_essentially_check.py).
        import app.agents.books_agent as ba

        node = _node(0.858, text=A2_PAGE12,
                     file_name="NIST.AI.100-1.pdf", page="12")
        control = ("The framework articulates the following characteristics "
                   "of trustworthy AI and offers guidance for addressing "
                   "them.")
        with_adverb = ("The framework essentially articulates the following "
                       "characteristics of trustworthy AI and offers guidance "
                       "for addressing them.")
        for sentence in (control, with_adverb):
            with self.subTest(sentence=sentence):
                mode, kept, claims = ba._grounded_repair(sentence, [node])
                self.assertEqual(mode, "ok", (mode, kept))
                self.assertIn(sentence, kept)
                self.assertEqual(len(claims), 1, claims)

    def test_essentially_does_not_ground_an_unsupported_claim(self):
        # The other side of the boundary: dropping the requirement on one
        # non-assertive filler word must never let an unsupported claim
        # through — its content tokens still have no evidence.
        import app.agents.books_agent as ba

        node = _node(0.858, text=A2_PAGE12,
                     file_name="NIST.AI.100-1.pdf", page="12")
        unsupported = ("The framework essentially proves that AI systems are "
                       "guaranteed to be free of legal regulations.")
        mode, kept, claims = ba._grounded_repair(unsupported, [node])
        self.assertEqual(mode, "refusal", (mode, kept))
        self.assertEqual(kept, [])
        self.assertEqual(claims, [])

    def test_essentially_adverb_agent_answers_with_evidenced_footer(self):
        # Same shape end-to-end through the REAL books_agent: the answer is
        # fully evidenced, so the user must get the content and the
        # validated References footer — not the mandated refusal.
        node = _node(0.858, text=A2_PAGE12,
                     file_name="NIST.AI.100-1.pdf", page="12")
        answer = ("The framework essentially articulates the following "
                  "characteristics of trustworthy AI and offers guidance "
                  "for addressing them.")
        out = self._run_agent(
            _FakeResponse(answer, [node]),
            question="What does the document say about the characteristics "
                     "of trustworthy AI?")
        self.assertFalse(out["answer"].startswith(NOT_FOUND_PHRASE),
                         out["answer"])
        self.assertIn("essentially articulates the following characteristics",
                      out["answer"])
        self.assertIn("References: NIST.AI.100-1.pdf, page 12", out["answer"])

    def test_t3_comparison_question_is_recovery_eligible_against_real_chunk(self):
        # T3 (the original Map/Measure question): the real retrieved Table 2
        # chunk never echoes the INSTRUCTION vocabulary (difference / give
        # / cite), so coverage measured 5/12 = 0.42 and the ONE bounded
        # retry was denied. Those words are now subtracted from the
        # ELIGIBILITY screen only -> 5/9 = 0.56.
        import app.agents.books_agent as ba

        node = _node(0.719, text=T3_TABLE2_PAGE28,
                     file_name="NIST.AI.100-1.pdf", page="28")
        self.assertTrue(ba._question_covered_by_evidence(T3_QUESTION, [node]))

    def test_instruction_words_alone_never_make_a_question_eligible(self):
        # Opposite side of the same screen: dropping the instruction words
        # must not make a question whose SUBJECT matter is absent eligible.
        import app.agents.books_agent as ba

        node = _node(0.719, text=T3_TABLE2_PAGE28,
                     file_name="NIST.AI.100-1.pdf", page="28")
        off_subject = ("Explain the difference between photosynthesis and "
                       "cellular respiration, and cite the supporting evidence.")
        self.assertFalse(ba._question_covered_by_evidence(off_subject, [node]))

    def test_t3_invented_activity_still_refuses_despite_eligibility(self):
        # Eligibility only buys ONE retry; what may be SAID is still gated.
        # An invented activity and a fabricated metric must not survive even
        # though the question is now eligible for recovery.
        node = _node(0.719, text=T3_TABLE2_PAGE28,
                     file_name="NIST.AI.100-1.pdf", page="28")
        out = self._run_agent(
            _FakeResponse(
                "One concrete activity for the MAP function is holding a "
                "brainstorming workshop, and the MEASURE function activity "
                "score is 0.95.",
                [node]),
            question=T3_QUESTION)
        self._assert_refusal_exact(out)
        self.assertNotIn("0.95", out["answer"])
        self.assertNotIn("brainstorm", out["answer"])

    def test_reporting_and_meta_words_do_not_kill_evidenced_sentences(self):
        # "emphasizes" / "like" / "content" are non-assertive: a sentence
        # whose FACTS are verbatim evidence must not die on the reporting
        # scaffolding around them.
        import app.agents.books_agent as ba

        smoker = _node(0.78, text=(
            "Chapter 8: Maintenance. Smokers calm bees by masking alarm "
            "pheromones during inspections."))
        for sentence in (
            "It emphasizes smokers calm bees by masking alarm pheromones "
            "during inspections.",
            "The guide content says smokers calm bees by masking alarm "
            "pheromones during inspections.",
        ):
            with self.subTest(sentence=sentence):
                mode, kept, _ = ba._grounded_repair(sentence, [smoker])
                self.assertIn(mode, ("ok", "partial"), (mode, kept))
                self.assertIn(sentence, kept)

    def test_map_and_measure_claims_are_kept_against_their_sections(self):
        # The Map/Measure GUI question (T3/T3v) was refused although the
        # retrieved sections state exactly this material.
        import app.agents.books_agent as ba

        nodes = [
            _node(0.78, text=MAP_PAGE25,
                  file_name="NIST.AI.100-1.pdf", page="25"),
            _node(0.77, text=MEASURE_PAGE29,
                  file_name="NIST.AI.100-1.pdf", page="29"),
        ]
        answer = (
            "Outcomes in the MAP function are the basis for the MEASURE and "
            "MANAGE functions. "
            "The MAP function is intended to enhance an organization's "
            "ability to identify risks and broader contributing factors. "
            "MEASURE 1: Appropriate methods and metrics are identified and "
            "applied."
        )
        mode, kept, claims = ba._grounded_repair(answer, nodes)
        self.assertEqual(mode, "ok", (mode, kept))
        self.assertEqual(len(kept), 3, kept)
        self.assertEqual(len(claims), 3, claims)

    def test_multi_part_answer_keeps_claims_from_multiple_passages(self):
        import app.agents.books_agent as ba

        nodes = [_node(0.78, text=PROPOLIS_PASSAGE),
                 _node(0.77, text=BEE_PASSAGE)]
        answer = ("Propolis is a resinous substance bees collect from plant "
                  "buds and use to seal small gaps and sterilize the interior "
                  "of the hive. Honey is harvested only when its moisture "
                  "content drops below 18 percent, which prevents "
                  "fermentation during storage.")
        mode, kept, claims = ba._grounded_repair(answer, nodes)
        self.assertEqual(mode, "ok", (mode, kept))
        self.assertEqual(len(claims), 2, claims)

    def test_unanswerable_ingestion_question_still_refuses(self):
        # Phase 6B-C shape: the source never describes an ingestion pipeline;
        # an invented figure must never survive.
        response = _FakeResponse(
            "The ingestion pipeline uses a chunk size of 512 characters.",
            [_node(0.662, text=DISCLAIMER_PAGE_I,
                   file_name="NIST.AI.100-1.pdf", page="i")])
        out = self._run_agent(
            response,
            question="What exact chunk size in characters does this "
                     "document's ingestion pipeline use?")
        self._assert_refusal_exact(out)
        self.assertNotIn("512", out["answer"])

    def test_relevant_looking_evidence_without_the_asked_detail_refuses(self):
        # Bee passages are topically right for the B1 hive question but say
        # nothing about "supers"/"inspecting" — relevant-looking yet
        # insufficient evidence must still produce an honest abstention.
        response = _FakeResponse(
            "Add honey supers after inspecting the hive for queen cells.",
            [_node(0.783, text=BEE_PASSAGE),
             _node(0.70, text=PROPOLIS_PASSAGE)])
        out = self._run_agent(
            response,
            question="What does the field guide recommend for inspecting a "
                     "hive before adding honey supers?")
        self._assert_refusal_exact(out)
        self.assertNotIn("supers", out["answer"])

    def test_gate_refusal_diagnostics_name_stage_and_missing_tokens(self):
        # 2026-10-25 requirement: refusal diagnostics must identify the
        # actual stage and reason (log-only; the user-facing phrase stays).
        import contextlib
        import io

        response = _FakeResponse(
            "The ingestion pipeline uses a chunk size of 512 characters.",
            [_node(0.662, text=DISCLAIMER_PAGE_I,
                   file_name="NIST.AI.100-1.pdf", page="i")])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self._run_agent(
                response,
                question="What exact chunk size in characters does this "
                         "document's ingestion pipeline use?")
        log = buf.getvalue()
        self._assert_refusal_exact(out)
        self.assertIn("[books_agent] grounding refusal", log)
        self.assertIn("[books_agent] gate_drop:", log)
        self.assertIn("missing tokens:", log)
        self.assertIn("512", log)      # the dropped sentence's own reason


if __name__ == "__main__":
    unittest.main(verbosity=2)
