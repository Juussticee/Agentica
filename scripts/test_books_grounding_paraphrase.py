"""Regression: `_grounded_repair` must not refuse a fully supported PARAPHRASE.

Isolated Chroma investigation established the defect: a correct delta-v
answer whose wording differs from the source (morphology + domain synonyms,
e.g. "orbital mechanics" over source "orbital kinetics", "propulsion
systems" over source "propulsion engine", "required" over source
"requires") had every sentence dropped, degrading to the mandated
not-found phrase despite retrieval at ~0.78 similarity.

Deterministic only: fixed source text, fixed answer text, the real
production grounding function (via the real `books_agent` with a faked
query engine and via `_grounded_repair` directly). No model calls, no
embeddings, no Chroma, no network.

Cases (per the fix brief):
  A. supported paraphrase           -> must NOT be refused (fails pre-fix)
  B. directly supported wording     -> keeps passing
  C. unsupported + supported mixed  -> supported survives, unsupported
                                       dropped; one good sentence never
                                       carries unrelated claims
  D. entirely unsupported, fluent   -> still refused, no footer
  E. empty/unusable evidence        -> safe refusal (direct gate check)

The gate stays sentence-level and token-coverage based: these tests pin
both the repaired acceptance (A) and the non-weakened rejections
(C/D/E).
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.books_agent import _grounded_repair  # noqa: E402
from app.config import NOT_FOUND_PHRASE  # noqa: E402
from scripts.test_books_grounding import (  # noqa: E402
    GroundingTestBase,
    _FakeResponse,
    _node,
    ORBITAL,
)

# Fixed evidence: describes delta-v as a change in velocity used in orbital
# maneuvers, with orbital-transfer details (delta-v budget, Hohmann,
# exhaust/mass ratio) — all verifiable statements in this text alone.
DELTA_SOURCE = (
    "Orbital kinetics describes delta-v as the change in velocity that a "
    "spacecraft requires to alter its orbit. Planners calculate the delta-v "
    "budget for each orbital maneuver: a Hohmann transfer to geostationary "
    "orbit needs about 3.9 kilometers per second, supplied by the propulsion "
    "engine through exhaust velocity multiplied by mass ratio."
)

# Same content in natural alternative wording: "orbital mechanics", the
# "propulsion systems" family, "required" over "requires", "needed by".
DELTA_PARAPHRASE = (
    "Delta-v is the change in velocity required to alter an orbit during an "
    "orbital maneuver in orbital mechanics. It is needed by propulsion "
    "systems during an orbital maneuver: a Hohmann transfer to geostationary "
    "orbit needs about 3.9 kilometers per second."
)

DELTA_QUESTION = ("According to orbital_kinetics_principles.docx, what does "
                  "delta-v mean, and how is it used in orbital mechanics?")


def _delta_node(score=0.78):
    return _node(score, text=DELTA_SOURCE, file_name=ORBITAL, page="1")


class TestASupportedParaphrase(GroundingTestBase):
    """A. Wording differs; evidence supports every claim -> no refusal."""

    def test_paraphrase_answer_is_not_refused(self):
        out = self._run_agent(_FakeResponse(DELTA_PARAPHRASE, [_delta_node()]),
                              question=DELTA_QUESTION)
        answer = out["answer"]
        self.assertNotIn(NOT_FOUND_PHRASE, answer, answer)
        # The alternative wording itself survives (the defect dropped it).
        self.assertIn("orbital mechanics", answer)
        self.assertIn("propulsion systems", answer)
        self.assertIn("3.9 kilometers per second", answer)
        # Grounded answers keep their validated references.
        self.assertIn("References: orbital_kinetics_principles.docx, page 1",
                      answer)
        self.assertEqual(out["sources"], ["books"])

    def test_gate_classifies_paraphrase_as_supported_not_refusal(self):
        mode, kept, claims = _grounded_repair(DELTA_PARAPHRASE, [_delta_node()])
        self.assertIn(mode, ("ok", "partial"), mode)
        self.assertEqual(len(kept), 2, kept)
        self.assertEqual(len(claims), 2, claims)


class TestFRiskHarmParaphrase(GroundingTestBase):
    """Approved 2026-10-24 RAG fix: the {risk, harm} paraphrase family.

    Verified GUI failures (NIST AI-RMF corpus) refused a correct in-corpus
    answer paraphrasing "possible AI harms" although the retrieved passages
    say "risks" — the AI RMF's own synonym pair. The family admits that
    supported relabel; its contract (relabel-only, one token at a time,
    numbers exact, group can never invent support, gates run first) is
    re-pinned here: unsupported and polarity-inverted "harm" claims still
    refuse exactly as before.
    """

    RISK_EVIDENCE = (
        "The MAP function identifies risks and provides methods to measure "
        "and manage them. The relevant context is documented in which "
        "possible risks may arise across the system lifecycle."
    )
    HARM_ANSWER = (
        "The MAP function identifies possible harms and documents the "
        "relevant context across the system lifecycle."
    )

    def _risk_node(self, score=0.78):
        return _node(score, text=self.RISK_EVIDENCE,
                     file_name="ai_framework.pdf", page="20")

    def test_harm_over_risk_paraphrase_is_admitted(self):
        out = self._run_agent(_FakeResponse(self.HARM_ANSWER,
                                            [self._risk_node()]))
        answer = out["answer"]
        self.assertNotIn(NOT_FOUND_PHRASE, answer, answer)
        self.assertIn("possible harms", answer)
        self.assertIn("References: ai_framework.pdf, page 20", answer)

    def test_group_never_invents_harm_support(self):
        # The relabel needs a sibling member in the EVIDENCE: an unrelated
        # corpus (delta-v orbital text, no risk/harm vocabulary) must still
        # refuse a harm claim — the family cannot manufacture support.
        mode, _, _ = _grounded_repair(self.HARM_ANSWER, [_delta_node()])
        self.assertEqual(mode, "refusal", mode)

    def test_polarity_gate_runs_before_the_family(self):
        negated = ("This framework does not catalog individual harms of AI "
                   "systems.")
        affirmed = "This framework catalogs individual harms of AI systems."
        mode, _, _ = _grounded_repair(
            affirmed,
            [_node(0.8, text=negated, file_name="ai_framework.pdf", page="3")])
        self.assertEqual(mode, "refusal", mode)

    def test_question_to_evidence_coverage_uses_the_family(self):
        from app.agents.books_agent import _question_covered_by_evidence
        question = ("If a team wants to identify and understand the context "
                    "of possible AI harms before measuring them, which "
                    "function is most relevant, and why?")
        # Evidence says "risks" where the question says "harms": eligibility
        # must hold via the family (this unblocked M5 on the verified run).
        self.assertTrue(_question_covered_by_evidence(
            question, [self._risk_node()]))
        self.assertFalse(_question_covered_by_evidence(
            question, [_delta_node()]))


class TestBDirectlySupportedWording(GroundingTestBase):
    """B. Source-overlapping answers keep passing unchanged."""

    def test_verbatim_source_wording_still_passes(self):
        answer = ("Planners calculate the delta-v budget for each orbital "
                  "maneuver.")
        out = self._run_agent(_FakeResponse(answer, [_delta_node()]),
                              question=DELTA_QUESTION)
        self.assertIn("delta-v budget", out["answer"])
        self.assertNotIn(NOT_FOUND_PHRASE, out["answer"])

    def test_median_supported_sentence_passes_gate(self):
        sentence = ("A Hohmann transfer to geostationary orbit needs about "
                    "3.9 kilometers per second.")
        mode, kept, claims = _grounded_repair(sentence, [_delta_node()])
        self.assertEqual(mode, "ok", mode)
        self.assertEqual(kept, [sentence])


class TestCMixedSupportedAndUnsupported(GroundingTestBase):
    """C. One supported sentence never carries an unrelated claim."""

    MIXED = (
        "A Hohmann transfer to geostationary orbit needs about 3.9 "
        "kilometers per second. Comet flybys occur every 5.2 years and cost "
        "8 million dollars to arrange."
    )

    def test_supported_survives_unsupported_dropped(self):
        out = self._run_agent(_FakeResponse(self.MIXED, [_delta_node()]),
                              question=DELTA_QUESTION)
        answer = out["answer"]
        self.assertIn("3.9 kilometers per second", answer)   # kept
        self.assertNotIn("5.2", answer)                       # dropped
        self.assertNotIn("8 million", answer)                 # dropped
        self.assertNotEqual(answer, NOT_FOUND_PHRASE)
        # Existing partial-support contract: explicit disclosure + footer.
        self.assertIn("not found in the books", answer)
        self.assertIn("References:", answer)
        # Direct gate view: partial mode with exactly the supported
        # sentence retained (the user-facing disclosure line is composed
        # by books_agent, not by the gate itself).
        mode, kept, claims = _grounded_repair(self.MIXED, [_delta_node()])
        self.assertEqual(mode, "partial", mode)
        self.assertEqual(
            kept, ["A Hohmann transfer to geostationary orbit needs about "
                   "3.9 kilometers per second."], kept)
        self.assertEqual(len(claims), 1, claims)


class TestDFullyUnsupportedAnswer(GroundingTestBase):
    """D. Fluent, generically-overlapping prose is still refused."""

    def test_fluent_fabrication_with_source_vocabulary_is_refused(self):
        fabricated = (
            "The books describe delta budgeting as a practice worth 9 "
            "billion dollars yearly, and every orbital maneuver obeys a "
            "licence fee scale reviewed each season."
        )
        out = self._run_agent(_FakeResponse(fabricated, [_delta_node()]),
                              question=DELTA_QUESTION)
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE),
                        out["answer"])
        self.assertNotIn("References:", out["answer"])
        self.assertNotIn("billion", out["answer"])

    def test_pure_fabrication_without_any_support_is_refused(self):
        fabricated = ("A Hohmann transfer to geostationary orbit takes 12 "
                      "days and happens only once per lunar cycle.")
        out = self._run_agent(_FakeResponse(fabricated, [_delta_node()]),
                              question=DELTA_QUESTION)
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE),
                        out["answer"])
        self.assertNotIn("References:", out["answer"])


class TestEEmptyEvidence(GroundingTestBase):
    """E. Empty or unusable evidence keeps the safe refusal contract."""

    def test_direct_gate_refuses_on_empty_source_nodes(self):
        mode, kept, claims = _grounded_repair(DELTA_PARAPHRASE, [])
        self.assertEqual(mode, "refusal", mode)
        self.assertEqual(kept, [])
        self.assertEqual(claims, [])

    def test_agent_answers_not_found_when_nothing_was_retrieved(self):
        out = self._run_agent(_FakeResponse(DELTA_PARAPHRASE, []),
                              question=DELTA_QUESTION)
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE),
                        out["answer"])
        self.assertNotIn("References:", out["answer"])
        self.assertEqual(out["sources"], ["books"])


class TestRealVerifiedEvidence(GroundingTestBase):
    """Pin the ACTUAL investigated case: the real indexed chunk and the real
    model answer from the isolated-Chroma run that reproduced the defect.
    Pre-fix this returned mode='refusal' (kept=0); both sentences must now
    be retained — this is the literal regression of the reported bug."""

    REAL_CHUNK = (
        "Orbital Kinetics: Principles of Spacecraft Motion\n\nSecond edition, "
        "prepared for training purposes.\n\nChapter 1: Basic Orbits\n\n"
        "A circular orbit is the simplest trajectory: the spacecraft moves at "
        "constant altitude with a speed determined solely by the central "
        "body's gravitational parameter and the orbital radius. For low Earth "
        "orbit at 400 km altitude, the orbital speed is approximately 7.67 "
        "kilometers per second and one revolution takes about 92.4 minutes."
        "\n\nChapter 2: Maneuvers\n\nThe Hohmann transfer is the minimum-energy "
        "two-impulse maneuver connecting two circular, coplanar orbits. "
        "Moving from low Earth orbit to geostationary transfer orbit requires "
        "roughly 2.4 kilometers per second of delta-v, and circularizing at "
        "geostationary altitude adds approximately 1.5 kilometers per second "
        "more.\n\nGeostationary orbit lies at an altitude of approximately "
        "35,786 kilometers above the equator, where the orbital period matches "
        "Earth's rotation, so the spacecraft appears fixed over one "
        "longitude.\n\nChapter 3: The Rocket Equation\n\nThe Tsiolkovsky rocket "
        "equation relates delta-v to exhaust velocity and mass ratio: delta-v "
        "equals exhaust velocity times the natural logarithm of the initial "
        "mass divided by final mass. It was first derived in 1903 and remains "
        "the foundation of mission design."
    )
    REAL_RAW_ANSWER = (
        "According to the context, delta-v refers to the change in velocity of "
        "a rocket, calculated using the rocket equation: delta-v equals "
        "exhaust velocity multiplied by the natural logarithm of the initial "
        "mass divided by final mass. It is used in orbital mechanics to "
        "determine the required delta-v for maneuvers or propulsion systems."
    )

    def test_real_investigated_evidence_is_no_longer_refused(self):
        node = _node(0.781241890515103, text=self.REAL_CHUNK,
                     file_name=ORBITAL)
        mode, kept, claims = _grounded_repair(self.REAL_RAW_ANSWER, [node])
        self.assertIn(mode, ("ok", "partial"), mode)
        # BOTH investigated sentences must survive (pre-fix: kept=0).
        self.assertEqual(len(kept), 2, kept)
        self.assertEqual(len(claims), 2, claims)

    def test_real_case_user_answer_keeps_definition_and_footer(self):
        node = _node(0.781241890515103, text=self.REAL_CHUNK,
                     file_name=ORBITAL)
        out = self._run_agent(_FakeResponse(self.REAL_RAW_ANSWER, [node]),
                              question=DELTA_QUESTION)
        answer = out["answer"]
        self.assertNotIn(NOT_FOUND_PHRASE, answer, answer)
        self.assertIn("orbital mechanics", answer)
        self.assertIn("propulsion systems", answer)
        self.assertIn("References:", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)