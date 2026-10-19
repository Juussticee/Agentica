"""Adversarial regression for the bounded paraphrase groups + stemming gate.

Deterministic only: fixed evidence/answer strings and the real production
functions (`_grounded_repair` directly and via `books_agent` with a faked
engine) — no model, no network, no Chroma, no production data.

Every case states why it must be admitted or refused; assertions target the
externally observable grounding outcome (mode, kept sentences, user answer).

A. verified positive control — the investigated delta-v chunk + raw answer
   must keep BOTH supported sentences (the defect the fix repaired).
B. synonym families — a legitimate wording swap of an EVIDENCED statement is
   admitted; an unsupported claim is NOT admitted merely because a family
   sibling occurs in the evidence (non-family tokens/numbers must fail).
C. mixed — the supported sentence survives; a group-adjacent unsupported
   sentence is dropped and disclosed (one good sentence never carries it).
D. negation/polarity — polarity protections survive the group path in both
   directions (negated answer vs affirmative evidence, and vice versa).
E. numeric integrity — changed values, wrong units, signs, and dropped
   numbers; no paraphrase group may cover a numeric token (structural
   invariant asserted too).
F. metadata/citations/stemming — fabricated pages are dropped, real pages
   kept; metadata page tokens alone cannot ground content; the `speed`
   token (ends in "ed" as a substring) is never mangled by stemming.
G. evidence boundaries — the documented cross-passage splice gap is PINNED
   as current behavior (group-covered tokens included), not redesigned.

Second adversarial pass (this review) adds: family-swap depth probes
(estimate-vs-determine precision, the change family NOT extending to
"moves", the modify inflection — admitted after the "modifi" family fix
that closed its demonstrated over-refusal, the family-3 subject-swap
limitation PINNED as current behavior), a multiply-family homonym probe
(duration "time" vs multiplication "times"), polarity gates tested
ACROSS a family swap, decimal-digit changes, plural "page" stemming vs
the page-number check, and a same-passage (single node) cross-sentence
splice pin. Pre-fix failures this review recorded were: time-homonym
and polarity-family FALSE POSITIVES (closed by the narrow grounding
fix) and the modify-inflection over-refusal (closed by the "modifi"
family entry); everything else passed pre-fix as regression pins.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import app.agents.books_agent as ba  # noqa: E402
from app.agents.books_agent import _grounded_repair  # noqa: E402
from app.config import NOT_FOUND_PHRASE  # noqa: E402
from scripts.test_books_grounding import (  # noqa: E402
    GroundingTestBase,
    _FakeResponse,
    _node,
    ORBITAL,
)
from scripts.test_books_grounding_paraphrase import (  # noqa: E402
    DELTA_QUESTION,
    TestRealVerifiedEvidence as _RealEvidenceFixtures,
)

# The verified investigated fixtures (real indexed chunk + real raw answer).
REAL_CHUNK = _RealEvidenceFixtures.REAL_CHUNK
REAL_RAW_ANSWER = _RealEvidenceFixtures.REAL_RAW_ANSWER


class TestAPositiveControl(GroundingTestBase):
    """A. The investigated case stays fixed (must be admitted)."""

    def test_real_case_keeps_both_sentences(self):
        node = _node(0.781241890515103, text=REAL_CHUNK, file_name=ORBITAL)
        mode, kept, claims = _grounded_repair(REAL_RAW_ANSWER, [node])
        self.assertEqual(mode, "ok", mode)
        self.assertEqual(len(kept), 2, kept)
        self.assertEqual(len(claims), 2, claims)


# (evidence, admitted paraphrase using a different family member,
#  unsupported claim that must be refused)
_FAMILY_CASES = (
    ("mechanics/kinetics",
     "Orbital kinetics requires precise planning for transfers.",
     "Orbital mechanics requires precise planning for transfers.",
     "Orbital kinetics requires precise planning for construction."),
    ("propulsion/engine/rocket",
     "The propulsion engine provides reliable thrust.",
     "The rocket engine provides reliable thrust.",
     "The reactor engine provides reliable thrust."),
    ("system/spacecraft/vehicle",
     "Each spacecraft system carries a backup radio.",
     "Each spacecraft vehicle carries a backup radio.",
     "Each airline vehicle carries a backup radio."),
    ("need/require",
     "The maneuver requires 2.4 kilometers per second.",
     "The maneuver needs 2.4 kilometers per second.",
     "The surgery needs 2.4 kilometers per second."),
    ("calculate/derive/determine",
     "Planners derive the budget from mass ratio.",
     "Planners calculate the budget from mass ratio.",
     "Planners estimate the hive budget from mass ratio."),
    ("multiply/times/equal",
     "Delta-v equals exhaust velocity times the mass ratio.",
     "Delta-v equals exhaust velocity multiplied by the mass ratio.",
     "Delta-v equals propolis times the mass ratio."),
    ("refer/relate/describe",
     "The chapter relates delta-v to exhaust velocity.",
     "The chapter refers to delta-v and exhaust velocity.",
     "The chapter relates quantum tunneling to exhaust velocity."),
    ("change/alter",
     "The maneuver alters the orbital altitude.",
     "The maneuver changes the orbital altitude.",
     "The maneuver alters the quantum altitude."),
)


class TestBFamilyParaphrases(GroundingTestBase):
    """B. Family swaps of an EVIDENCED statement pass; an unsupported claim
    that merely shares a family sibling with the evidence never does."""

    def test_legitimate_family_paraphrase_is_admitted(self):
        for name, evidence, paraphrase, _ in _FAMILY_CASES:
            with self.subTest(family=name):
                mode, kept, _ = _grounded_repair(paraphrase,
                                                  [_node(0.72, text=evidence)])
                self.assertIn(mode, ("ok", "partial"),
                              f"{name}: paraphrase wrongly refused")
                self.assertEqual(kept, [paraphrase])

    def test_unsupported_claim_is_not_admitted_by_family_presence(self):
        for name, evidence, _, unsupported in _FAMILY_CASES:
            with self.subTest(family=name):
                mode, kept, _ = _grounded_repair(unsupported,
                                                  [_node(0.72, text=evidence)])
                self.assertEqual(mode, "refusal",
                                 f"{name}: unsupported claim admitted")


class TestBMultiplyFamilyTimeHomonym(GroundingTestBase):
    """B/multiply — DEFECT DEMONSTRATION (fails pre-fix).

    Duration "time" and multiplication "times" are different words.
    The multiply family listed the SINGULAR form, so evidence that merely
    spends TIME satisfied the group for an answer claiming the budget is
    MULTIPLIED — a materially unsupported claim admitted purely through
    the group. Must be refused.
    """

    EVIDENCE = ("The mission planning leaves time for review of the "
                "transfer budget.")
    ANSWER = ("The mission planning multiplies review of the transfer "
              "budget.")

    def test_duration_time_does_not_cover_multiplication_claim(self):
        mode, kept, _ = _grounded_repair(self.ANSWER,
                                         [_node(0.75, text=self.EVIDENCE)])
        self.assertEqual(mode, "refusal", (mode, kept))


class TestBFamilySwapDepth(GroundingTestBase):
    """B — depth probes the tuple loop does not reach.

    estimate→determine (precision upgrade must be refused — it held
    pre-fix through literal coverage, since the group's "estim" entry is
    dead), the change family NOT extending to motion verbs in the answer
    direction, the modify inflection admitted after the "modifi" family
    fix (its over-refusal pin flipped to a positive regression), and the
    family-3 subject swap PINNED as a known limitation (closing it needs
    DIRECTIONAL aliases — a coverage redesign outside this task —
    reported rather than silently changed).
    """

    def test_estimate_word_does_not_upgrade_to_determined_claim(self):
        # Evidence only ESTIMATES; the answer asserts the figure is
        # DETERMINED. Precision is part of the claim, so the family swap
        # is unsupported — not a safe paraphrase (whether via literal
        # coverage or via the group: estimate forms must stay outside the
        # calculate/derive/determine family).
        evidence = "The manual estimates the transfer cost at launch."
        answer = "The manual determines the transfer cost at launch."
        mode, kept, _ = _grounded_repair(answer, [_node(0.75, text=evidence)])
        self.assertEqual(mode, "refusal", (mode, kept))

    def test_change_family_does_not_extend_to_moves(self):
        # The answer uses a motion verb; "moves" is deliberately NOT a
        # member (moving an orbit != changing it — regression D1), so the
        # change-family sibling in the evidence must not admit it.
        evidence = "The maneuver changes the orbital altitude."
        answer = "The maneuver moves the orbital altitude."
        mode, kept, _ = _grounded_repair(answer, [_node(0.75, text=evidence)])
        self.assertEqual(mode, "refusal", (mode, kept))

    def test_modify_member_unsupported_claim_refused(self):
        # Brief B for the modify member: the evidence holds a
        # change-family sibling, the answer uses "modify" inside an
        # UNSUPPORTED claim — family presence must not admit it.
        evidence = "The maneuver changes the orbital altitude."
        answer = "The maneuver modifies the quantum altitude."
        mode, kept, _ = _grounded_repair(answer, [_node(0.75, text=evidence)])
        self.assertEqual(mode, "refusal", (mode, kept))

    def test_modify_inflection_paraphrase_is_admitted(self):
        # Positive regression (flipped from the over-refusal pin): the
        # normalizer emits "modifi" for modifies/modified/modifying, and
        # the change family now lists it. The former "modif" entry was
        # DEAD (no input produces it), so every natural inflection used
        # to be REFUSED against sibling-member evidence — demonstrated by
        # this review's pre-fix run and closed by the "modifi" family
        # entry. These are genuinely supported paraphrases of the
        # evidenced claim and must be admitted; the family still never
        # carries unsupported content (the quantum-altitude and
        # motion-verb cases above stay refused).
        evidence = "The maneuver alters the orbital altitude."
        for answer in ("The maneuver modifies the orbital altitude.",
                       "The maneuver modified the orbital altitude."):
            with self.subTest(answer=answer):
                mode, kept, _ = _grounded_repair(
                    answer, [_node(0.75, text=evidence)])
                self.assertIn(mode, ("ok", "partial"), (mode, kept))
                self.assertEqual(kept, [answer])

    def test_family3_subject_swap_pinned_as_known_limitation(self):
        # KNOWN LIMITATION, pinned deliberately: symmetric relabeling
        # admits a subject swap (evidence "system" → answer "spacecraft")
        # although the sentences are not equivalent. NOT fixed here:
        # closing it requires asymmetric/directional coverage, which
        # would redesign the group contract and must break THIS test
        # loudly first. Reported in the closeout limitations.
        evidence = "Each system carries a backup radio."
        answer = "Each spacecraft carries a backup radio."
        mode, kept, _ = _grounded_repair(answer, [_node(0.75, text=evidence)])
        self.assertEqual(mode, "ok", (mode, kept))
        self.assertEqual(kept, [answer])


class TestCD1MotionBridgeNotAdmitted(GroundingTestBase):
    """C + defect D1 (pre-fix failure expected): the change-family must NOT
    bridge to motion verbs. Evidence describes MOVING; the answer asserts
    CHANGING — materially different, so the sentence must be dropped while
    the supported sibling sentence survives and is disclosed around."""

    EVIDENCE = ("A Hohmann transfer to geostationary orbit needs about 3.9 "
                "kilometers per second. Moving the spacecraft requires "
                "constant control.")
    ANSWER = ("A Hohmann transfer to geostationary orbit needs about 3.9 "
              "kilometers per second. Changing the spacecraft requires "
              "constant control.")

    def test_motion_verb_bridge_does_not_admit_changing_claim(self):
        mode, kept, claims = _grounded_repair(self.ANSWER,
                                              [_node(0.8, text=self.EVIDENCE)])
        self.assertEqual(mode, "partial", mode)
        self.assertEqual(kept, [self.ANSWER.split(". ")[0] + "."], kept)
        self.assertEqual(len(claims), 1, claims)

    def test_agent_discloses_dropped_motion_bridge_sentence(self):
        out = self._run_agent(_FakeResponse(self.ANSWER,
                                            [_node(0.8, text=self.EVIDENCE)]),
                              question="How much delta-v does a Hohmann "
                                       "transfer need?")
        answer = out["answer"]
        self.assertIn("3.9 kilometers per second", answer)
        self.assertNotIn("Changing the spacecraft", answer)
        self.assertNotEqual(answer, NOT_FOUND_PHRASE)
        self.assertIn("not found in the books", answer)


class TestCD2EqualityAliasNotAdmitted(GroundingTestBase):
    """C + defect D2 (pre-fix failure expected): the multiply family must
    NOT bridge to equality words alone. The evidence only DEFINES with
    \"equals\"; the answer rewrites the relation as \"multiplied by\" with
    the operands swapped — a materially wrong formula that must be refused
    (every other token is literal, so the group alias is the sole admitting
    path)."""

    EVIDENCE = ("The rocket equation states delta-v equals exhaust velocity "
                "and mass ratio.")
    INVERTED = ("Exhaust velocity multiplied by delta-v equals mass ratio.")

    def test_equality_alias_does_not_admit_inverted_relation(self):
        mode, kept, _ = _grounded_repair(self.INVERTED,
                                         [_node(0.75, text=self.EVIDENCE)])
        self.assertEqual(mode, "refusal", (mode, kept))

    def test_legitimate_multiplication_wording_still_admitted(self):
        # Same family word, but the relation matches the evidence (times).
        evidence = ("The rocket equation states delta-v equals exhaust "
                    "velocity times mass ratio.")
        supported = ("Exhaust velocity multiplied by mass ratio equals "
                     "delta-v in the rocket equation.")
        mode, kept, _ = _grounded_repair(supported,
                                         [_node(0.75, text=evidence)])
        self.assertIn(mode, ("ok", "partial"), mode)
        self.assertEqual(kept, [supported])


class TestDPolarityPreserved(GroundingTestBase):
    """D. Polarity protections survive the group path in both directions."""

    AFFIRM = "The maneuver requires a velocity change of 2.4 kilometers per second."
    NEGATE = ("The maneuver does not require a velocity change of 2.4 "
              "kilometers per second.")

    def test_negated_answer_vs_affirmative_evidence_refused(self):
        # "not" is not in the evidence: the coverage gate itself refuses.
        mode, _, _ = _grounded_repair(self.NEGATE,
                                      [_node(0.77, text=self.AFFIRM)])
        self.assertEqual(mode, "refusal", mode)

    def test_affirmative_answer_vs_negated_evidence_refused_by_polarity(self):
        # The negated evidence supplies every token (incl. "not"), so
        # coverage passes and the POLARITY gate must refuse the affirmative
        # claim whose own covering clause disagrees (number rule included).
        mode, _, _ = _grounded_repair(self.AFFIRM,
                                      [_node(0.77, text=self.NEGATE)])
        self.assertEqual(mode, "refusal", mode)


class TestDPolarityFamilyBridge(GroundingTestBase):
    """D — polarity gates must hold ACROSS a paraphrase-family swap.

    Coverage consults the family, so the polarity gate must too:
    evidence that explicitly NEGATES a need must not admit an affirmative
    "requires" claim just because rule A compared raw tokens (DEFECT
    DEMONSTRATION: fails pre-fix). Both remaining directions pin that
    the fix does not over-refuse.
    """

    NEG_EVIDENCE = ("The maneuver does not need a velocity change for the "
                    "orbit.")
    AFFIRM = "The maneuver requires a velocity change for the orbit."
    NEG_ANSWER = ("The maneuver does not require a velocity change for the "
                  "orbit.")

    def test_negated_evidence_blocks_affirmative_family_swap(self):
        # Every other token matches; "require" reaches the evidence only
        # through the need/require group. The covering clause NEGATES the
        # claim, so the affirmative answer must be refused.
        mode, kept, _ = _grounded_repair(self.AFFIRM,
                                         [_node(0.77, text=self.NEG_EVIDENCE)])
        self.assertEqual(mode, "refusal", (mode, kept))

    def test_affirmative_evidence_does_not_support_negated_answer(self):
        # Vice versa: "not" is absent from the evidence, so the coverage
        # gate itself refuses — no group can invent a negation.
        evidence = "The maneuver requires a velocity change for the orbit."
        mode, kept, _ = _grounded_repair(self.NEG_ANSWER,
                                         [_node(0.77, text=evidence)])
        self.assertEqual(mode, "refusal", (mode, kept))

    def test_same_polarity_family_swap_still_admitted(self):
        # Guard against over-refusal: both sides negated — the swap is
        # still the same claim and must survive.
        mode, kept, _ = _grounded_repair(self.NEG_ANSWER,
                                         [_node(0.77, text=self.NEG_EVIDENCE)])
        self.assertIn(mode, ("ok", "partial"), (mode, kept))
        self.assertEqual(kept, [self.NEG_ANSWER])


class TestENumericIntegrity(GroundingTestBase):
    """E. Numbers stay exact, signed, unit-bound; groups never cover them."""

    EVIDENCE = ("Planners calculate the delta-v budget for each orbital "
                "maneuver: a Hohmann transfer to geostationary orbit needs "
                "about 3.9 kilometers per second.")

    def test_changed_value_refused(self):
        answer = ("A Hohmann transfer to geostationary orbit needs about "
                  "4.1 kilometers per second.")
        mode, _, _ = _grounded_repair(answer, [_node(0.8, text=self.EVIDENCE)])
        self.assertEqual(mode, "refusal", mode)

    def test_wrong_unit_refused(self):
        answer = ("A Hohmann transfer to geostationary orbit needs about "
                  "3.9 meters per second.")
        mode, _, _ = _grounded_repair(answer, [_node(0.8, text=self.EVIDENCE)])
        self.assertEqual(mode, "refusal", mode)

    def test_sign_word_refused(self):
        answer = ("A Hohmann transfer to geostationary orbit needs about "
                  "negative 3.9 kilometers per second.")
        mode, _, _ = _grounded_repair(answer, [_node(0.8, text=self.EVIDENCE)])
        self.assertEqual(mode, "refusal", mode)

    def test_no_group_contains_a_numeric_member(self):
        for group in ba._PARAPHRASE_GROUPS:
            for member in group:
                self.assertFalse(any(ch.isdigit() for ch in member),
                                 f"numeric member {member!r} in a group")

    def test_numberless_paraphrase_admitted_but_fabricates_no_number(self):
        # Current contract: dropping the quantity is not fabrication; the
        # admitted sentence must contain NO number the evidence lacks.
        answer = ("A Hohmann transfer to geostationary orbit needs about "
                  "kilometers per second.")
        mode, kept, _ = _grounded_repair(answer,
                                         [_node(0.8, text=self.EVIDENCE)])
        self.assertIn(mode, ("ok", "partial"), mode)
        for sentence in kept:
            self.assertFalse(any(ch.isdigit() for ch in sentence), sentence)


class TestEDecimalAndGroupNumber(GroundingTestBase):
    """E — decimal-digit changes and family-adjacent numbers stay exact."""

    EVIDENCE = ("Planners calculate the delta-v budget for each orbital "
                "maneuver: a Hohmann transfer to geostationary orbit needs "
                "about 3.9 kilometers per second.")

    def test_changed_decimal_fraction_refused(self):
        # 3.9 → 3.95 differs only AFTER the decimal point; numbers are
        # never group-covered, so the sentence must still be refused.
        answer = ("A Hohmann transfer to geostationary orbit needs about "
                  "3.95 kilometers per second.")
        mode, kept, _ = _grounded_repair(answer,
                                         [_node(0.8, text=self.EVIDENCE)])
        self.assertEqual(mode, "refusal", (mode, kept))

    def test_family_word_cannot_carry_a_changed_number(self):
        # "derive" is family-covered by "calculate", but 4.4 is not in
        # the evidence: the group must not smuggle the number past it.
        answer = ("Planners derive the delta-v budget for each orbital "
                  "maneuver: about 4.4 kilometers per second.")
        mode, kept, _ = _grounded_repair(answer,
                                         [_node(0.8, text=self.EVIDENCE)])
        self.assertEqual(mode, "refusal", (mode, kept))


class TestFMetadataCitationsStemming(GroundingTestBase):
    """F. Citation/metadata/stemming integrity around the new normalizer."""

    def test_fabricated_page_dropped_real_page_kept(self):
        evidence = "The manual lists transfer records on page 1."
        node = _node(0.74, text=evidence, file_name=ORBITAL, page="1")
        good = "The manual lists transfer records on page 1."
        bad = "The manual lists transfer records on page 7."
        mode, kept, _ = _grounded_repair(good + " " + bad, [node])
        self.assertEqual(mode, "partial", mode)
        self.assertEqual(kept, [good], kept)

    def test_metadata_page_alone_cannot_ground_content(self):
        node = _node(0.74, text="The manual lists transfer records.",
                     file_name=ORBITAL, page="42")
        answer = ("On page 42, the manual promises unlimited delta-v for "
                  "every transfer.")
        mode, kept, _ = _grounded_repair(answer, [node])
        self.assertEqual(mode, "refusal", (mode, kept))

    def test_speed_token_survives_ed_suffix_rule(self):
        # "speed" ends in the letters "ed" but must stay literal.
        evidence = ("The orbital speed is approximately 7.67 kilometers per "
                    "second.")
        node = _node(0.8, text=evidence)
        ok = ("The orbital speed is approximately 7.67 kilometers per "
              "second.")
        bad = ("The orbital speed is approximately 9.99 kilometers per "
               "second.")
        mode, kept, _ = _grounded_repair(ok, [node])
        self.assertEqual(mode, "ok", mode)
        mode2, kept2, _ = _grounded_repair(bad, [node])
        self.assertEqual(mode2, "refusal", (mode2, kept2))


class TestFPageStemNumber(GroundingTestBase):
    """F — the plural-stem form of "page" still requires the page NUMBER."""

    def test_plural_page_stem_does_not_bypass_page_number_check(self):
        node = _node(0.74, text="The manual lists transfer records on page 1.",
                     file_name=ORBITAL, page="1")
        # "pages" normalizes to "page", which the metadata contributes
        # verbatim: the wording variant must be admitted...
        good = "The manual lists transfer records on pages 1."
        mode, kept, _ = _grounded_repair(good, [node])
        self.assertIn(mode, ("ok", "partial"), (mode, kept))
        self.assertEqual(kept, [good])
        # ...but the stemmed word can never stand in for a number the
        # retrieved metadata does not carry.
        bad = "The manual lists transfer records on pages 7."
        mode2, kept2, _ = _grounded_repair(bad, [node])
        self.assertEqual(mode2, "refusal", (mode2, kept2))


class TestGEvidenceBoundariesPinned(GroundingTestBase):
    """G. DOCUMENTED cross-passage splice gap — pinned as current behavior
    (group-covered tokens included). Any future strengthening must break
    this test loudly; this task does not redesign the gap."""

    def test_cross_passage_splice_with_group_token_still_admitted(self):
        node1 = _node(0.70, text="Orbital kinetics determines transfer "
                                 "budgets.")
        node2 = _node(0.68, text="Propolis is collected from plant buds.",
                      file_name=ORBITAL, page="2")
        answer = "Propolis determines transfer budgets."
        mode, kept, _ = _grounded_repair(answer, [node1, node2])
        self.assertEqual(mode, "ok", mode)      # current documented gap
        self.assertEqual(kept, [answer])


class TestGSamePassageSplicePinned(GroundingTestBase):
    """G — the same splice gap WITHOUT cross-passage: one node, two
    sentences, an answer combining them is admitted (coverage runs over
    the whole corpus, not the relevant sentence). PINNED, not redesigned.
    """

    def test_single_node_cross_sentence_splice_still_admitted(self):
        node = _node(0.72, text=("Orbital kinetics determines transfer "
                                 "budgets. Propolis is collected from "
                                 "plant buds."))
        answer = "Propolis determines transfer budgets."
        mode, kept, _ = _grounded_repair(answer, [node])
        self.assertEqual(mode, "ok", (mode, kept))   # current documented gap
        self.assertEqual(kept, [answer])


if __name__ == "__main__":
    unittest.main(verbosity=2)