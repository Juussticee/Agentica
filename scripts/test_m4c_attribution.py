"""M4C — claim-level evidence attribution and contradiction handling.

Deterministic, offline suite (fake retrieval nodes, scripted verifier
responses; no Ollama, Gemini, SearXNG, or network). Every test asserts the
ACTUAL claim-to-evidence mapping in the user-visible footer, not merely
that a filename appears somewhere.

The M4C contract under test (local mode only; off mode stays byte-M3):

* One numbered footer line per retained claim:
      [n] <ref>[; <ref>...] (verifier-supported: <claim text>)
  The parenthetical is an ATTRIBUTED verifier judgment (bounded
  heuristic/model verdict on the shown passages), never an
  independently-established entailment claim.
* A line cites ONLY the nodes attributed to that claim — the evidence the
  verifier confirmed for it (ids validated against that claim's OWN
  assigned evidence), or the deterministic covering set as fallback.
* A valid evidence id attached to the wrong claim is rejected by the
  strict parser exactly like an unknown id (fail-closed).
* Removed/removed-by-timeout/removed-by-malformed claims leave no
  citation behind and are never attributed to a surviving claim.
* Contradiction handling stays scoped to its claim: conflicting evidence
  never strips or re-attributes a sibling claim's sources.
* Values that differ only by unit/period/population are NOT treated as
  contradictions (kept, each attributed to its own source).
* Same validated reference strings as the answer-level footer: missing
  metadata yields the per-claim "attribution unavailable" marker, never
  an invented name; absolute local paths never appear.
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import app.agents.books_agent as ba  # noqa: E402
import app.verifier as verifier  # noqa: E402
from app.agents.books_agent import books_agent  # noqa: E402
from app.config import NOT_FOUND_PHRASE  # noqa: E402

GUIDE = "urban_beekeeping_field_guide.pdf"
ORBITAL = "orbital_kinetics_principles.docx"

PROP_CLAIM = (
    "Propolis is a resinous substance bees collect from plant buds and use "
    "to seal small gaps and sterilize the interior of the hive."
)
ORBITAL_CLAIM = (
    "A Hohmann transfer between circular orbits requires two engine burns "
    "and depends on the delta-v budget of the spacecraft."
)
WORKER_CLAIM = "Worker bees lay up to 2,000 eggs per day."
BEE_CLAIM = "A healthy hive in midsummer can contain up to 60,000 worker bees."
WINTER_CLAIM = "In winter the same hive may shrink to fewer than 20,000 worker bees."

PROPOLIS_PASSAGE = (
    "Chapter 3: Hive Products. Propolis is a resinous substance bees collect "
    "from plant buds and use to seal small gaps and sterilize the interior "
    "of the hive. Honey is harvested only when its moisture content drops "
    "below 18 percent, which prevents fermentation during storage."
)
PROPOLIS_PASSAGE_COPY = (
    "Orbital Kinetics Principles. Propolis is a resinous substance bees "
    "collect from plant buds and use to seal small gaps and sterilize the "
    "interior of the hive."
)
ORBITAL_PASSAGE = (
    "Orbital Kinetics Principles. A Hohmann transfer between circular orbits "
    "requires two engine burns and depends on the delta-v budget of the "
    "spacecraft."
)
BEE_PASSAGE = (
    "Field Guide to Urban Beekeeping. Chapter 1: Colony Structure. "
    "A healthy hive in midsummer can contain up to 60,000 worker bees, plus "
    "one queen and a few hundred drones depending on the season. The queen "
    "lays up to 2,000 eggs per day at the peak of spring buildup."
)
WINTER_PASSAGE = (
    "Field Guide to Urban Beekeeping. Chapter 7: Wintering. In winter the "
    "same hive may shrink to fewer than 20,000 worker bees."
)

PROP_REF = f"{GUIDE}, page 1"
ORBIT_REF = ORBITAL


def _node(score, text, file_name=GUIDE, page="1", extra_meta=None):
    meta = {}
    if file_name is not None:
        meta["file_name"] = file_name
    if page is not None:
        meta["page_label"] = page
    if extra_meta:
        meta.update(extra_meta)
    return SimpleNamespace(score=score, metadata=meta, text=text)


class _FakeResponse:
    def __init__(self, text, nodes):
        self._text = text
        self.source_nodes = nodes

    def __str__(self):
        return self._text


def _verified(verdict, ids=("e0",), reason="checked"):
    return {"status": verifier.STATUS_VERIFIED, "verdict": verdict,
            "evidence_ids": list(ids), "reason": reason, "error": None}


def _unavailable(error=verifier.ERR_MALFORMED):
    return {"status": verifier.STATUS_UNAVAILABLE, "verdict": None,
            "evidence_ids": [], "reason": "", "error": error}


class M4CBase(unittest.TestCase):
    """Fake query engine + scripted verify_all or scripted _call_llm."""

    def setUp(self):
        self._orig_engine = ba._get_query_engine
        self._orig_mode = ba.BOOKS_VERIFIER
        self._orig_verify_all = verifier.verify_all
        self._orig_call_llm = verifier._call_llm
        self.captured_items = None

    def tearDown(self):
        ba._get_query_engine = self._orig_engine
        ba.BOOKS_VERIFIER = self._orig_mode
        verifier.verify_all = self._orig_verify_all
        verifier._call_llm = self._orig_call_llm

    def _run_agent(self, response, mode="local", verdict_for=None,
                   call_llm=None, verify_all=None):
        """Drive the real books_agent. Exactly one of verdict_for (patches
        verify_all — scripted RESULTS, bypassing the parser), call_llm
        (patches the transport — the REAL verify_all/verify_claim/parse
        path runs, which is how id-whitelist behavior is tested), or
        verify_all (a raw results sequence — for PARTIAL result lists no
        per-claim scripting can express)."""
        ba.BOOKS_VERIFIER = mode
        ba._get_query_engine = lambda: SimpleNamespace(
            query=lambda q: response)
        if verify_all is not None:
            verifier.verify_all = verify_all
        elif call_llm is not None:
            verifier._call_llm = call_llm
        else:
            if verdict_for is None:
                verdict_for = lambda item: _verified(
                    verifier.VERDICT_SUPPORTED,
                    ids=tuple(e["id"] for e in item["evidence"]))

            def _capture(items):
                self.captured_items = items
                return [verdict_for(item) for item in items]

            verifier.verify_all = _capture
        return books_agent({"question": "What do the books say?"})

    def _claim_lines(self, answer):
        """Parse the claim-level footer into [(refs, supports_text)]."""
        self.assertIn("References:", answer)
        lines = answer.split("References:\n", 1)[1].strip().splitlines()
        out = []
        for line in lines:
            head, _, supports = line.partition(" (verifier-supported: ")
            self.assertTrue(supports.endswith(")"), line)
            num, _, refs = head.partition("] ")
            self.assertTrue(num.startswith("["), line)
            # §4 wording: the attribution is attributed to the verifier,
            # never asserted as plain "supports:" (which would overstate
            # what the bounded verifier establishes).
            self.assertNotIn("supports:", line)
            out.append((refs, supports[:-1]))
        return out


class TestTwoClaimsTwoSources(M4CBase):
    """Two factual claims supported by different source nodes."""

    def test_each_claim_is_attributed_to_its_own_source(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes))
        lines = self._claim_lines(out["answer"])
        self.assertEqual(lines, [(PROP_REF, PROP_CLAIM),
                                 (ORBIT_REF, ORBITAL_CLAIM)])
        # Evidence isolation on the wire too: each verification call saw
        # only its own claim's node.
        self.assertEqual(
            [[e["id"] for e in item["evidence"]]
             for item in self.captured_items], [["e0"], ["e1"]])


class TestMixedSupportInOneAnswer(M4CBase):
    """One supported claim and one unsupported claim in the same answer."""

    def test_unsupported_claim_removed_sibling_attribution_intact(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]

        def _verdicts(item):
            if item["claim"] == PROP_CLAIM:
                return _verified(verifier.VERDICT_SUPPORTED)
            return _verified(verifier.VERDICT_INSUFFICIENT)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes),
            verdict_for=_verdicts)
        answer = out["answer"]
        self.assertIn(PROP_CLAIM, answer)
        self.assertNotIn("Hohmann", answer)          # removed claim gone
        self.assertIn("could not confirm", answer)   # disclosed (removed note)
        self.assertEqual(self._claim_lines(answer),
                         [(PROP_REF, PROP_CLAIM)])
        self.assertNotIn(ORBIT_REF, answer)          # its source too


class TestMultiPassageSingleClaim(M4CBase):
    """One claim genuinely supported by multiple passages."""

    def test_both_passages_cited_on_the_claims_own_line(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, PROPOLIS_PASSAGE_COPY, file_name=ORBITAL,
                       page=None)]
        out = self._run_agent(_FakeResponse(PROP_CLAIM, nodes))
        self.assertEqual(self._claim_lines(out["answer"]),
                         [(f"{PROP_REF}; {ORBIT_REF}", PROP_CLAIM)])


class TestVerifierCitationNarrowsAttribution(M4CBase):
    """The verifier's own evidence choice narrows the claim's sources."""

    def test_narrowed_citation_cites_only_the_confirmed_passage(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, PROPOLIS_PASSAGE_COPY, file_name=ORBITAL,
                       page=None)]
        # Deterministic layer covers BOTH nodes; verifier confirms e0 only.
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM, nodes),
            verdict_for=lambda item: _verified(
                verifier.VERDICT_SUPPORTED, ids=("e0",)))
        self.assertEqual(self._claim_lines(out["answer"]),
                         [(PROP_REF, PROP_CLAIM)])
        self.assertNotIn(ORBIT_REF, out["answer"])

    def test_unresolvable_citation_falls_back_to_deterministic_nodes(self):
        # Scripted result cites a node outside the claim's set (the
        # production parser would have rejected it); attribution must
        # fall back to the deterministic covering set, never empty or
        # invented.
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, PROPOLIS_PASSAGE_COPY, file_name=ORBITAL,
                       page=None)]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM, nodes),
            verdict_for=lambda item: _verified(
                verifier.VERDICT_SUPPORTED, ids=("e7",)))
        self.assertEqual(self._claim_lines(out["answer"]),
                         [(f"{PROP_REF}; {ORBIT_REF}", PROP_CLAIM)])


class TestCrossClaimIdsRejectedByTheRealParser(M4CBase):
    """A valid source id attached to the WRONG claim, through the REAL
    verify_all -> verify_claim -> strict parser path (transport scripted)."""

    def test_wrong_claim_id_is_malformed_fail_closed(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]

        def _llm(prompt, timeout):
            # Claim 1 cites e1 — a REAL id, but one that belongs to the
            # other claim's evidence. Must be rejected like a fabrication.
            # Claim 2 legitimately cites its own e1.
            payload = {"verdict": "supported", "evidence_ids": ["e1"],
                       "reason": "ok"}
            return "ok", json.dumps(payload)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes),
            call_llm=_llm)
        answer = out["answer"]
        # The claim whose citation failed validation is removed
        # (fail-closed), while the legitimately cited claim survives with
        # exactly its own source attributed.
        self.assertNotIn(PROP_CLAIM, answer)
        self.assertIn(ORBITAL_CLAIM, answer)
        self.assertIn("could not be confirmed", answer)
        self.assertEqual(self._claim_lines(answer), [(ORBIT_REF, ORBITAL_CLAIM)])

    def test_unknown_evidence_id_fails_closed_at_claim_level(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1")]

        def _llm(prompt, timeout):
            return "ok", json.dumps({"verdict": "supported",
                                     "evidence_ids": ["e9"],
                                     "reason": "ok"})
        out = self._run_agent(_FakeResponse(PROP_CLAIM, nodes), call_llm=_llm)
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE))
        self.assertIn("semantic verification unavailable", out["answer"])
        self.assertNotIn(GUIDE, out["answer"])
        self.assertNotIn("verifier-supported:", out["answer"])


class TestContradictionScopedToItsClaim(M4CBase):
    """Contradiction handling: reversed relationship + conflicting number,
    each scoped to its own claim and never re-attributed elsewhere."""

    def test_reversed_relationship_removed_and_not_reattributed(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.70, BEE_PASSAGE, file_name=GUIDE, page="2")]

        def _verdicts(item):
            if item["claim"] == PROP_CLAIM:
                return _verified(verifier.VERDICT_SUPPORTED)
            return _verified(verifier.VERDICT_CONTRADICTED)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + WORKER_CLAIM, nodes),
            verdict_for=_verdicts)
        answer = out["answer"]
        self.assertNotIn("Worker bees", answer)
        self.assertIn("conflict", answer)
        # The contradicted claim's evidence is NOT lent to the survivor:
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])
        self.assertNotIn("page 2", answer)

    def test_conflicting_number_same_conditions_scoped(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.72, BEE_PASSAGE, file_name=GUIDE, page="2"),
                 _node(0.70, "A healthy hive cannot exceed 20,000 worker "
                             "bees.", file_name=GUIDE, page="4")]

        def _verdicts(item):
            if item["claim"].startswith("Propolis"):
                return _verified(verifier.VERDICT_SUPPORTED)
            return _verified(verifier.VERDICT_CONTRADICTED)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + BEE_CLAIM, nodes),
            verdict_for=_verdicts)
        answer = out["answer"]
        self.assertNotIn("60,000", answer)
        self.assertIn("no resolution was invented", answer)
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])


class TestLegitimateDifferencesAreNotContradictions(M4CBase):
    """Different numbers that are true of different periods stay, each
    attributed to its own source — no over-rejection, no cross-lending."""

    def test_period_scoped_values_keep_their_own_sources(self):
        nodes = [_node(0.74, BEE_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, WINTER_PASSAGE, file_name=GUIDE, page="7")]
        out = self._run_agent(
            _FakeResponse(BEE_CLAIM + " " + WINTER_CLAIM, nodes))
        answer = out["answer"]
        self.assertIn(BEE_CLAIM, answer)
        self.assertIn(WINTER_CLAIM, answer)
        self.assertNotIn("Note:", answer)
        self.assertEqual(self._claim_lines(answer),
                         [(f"{GUIDE}, page 1", BEE_CLAIM),
                          (f"{GUIDE}, page 7", WINTER_CLAIM)])


class TestNegationVariants(M4CBase):
    def test_agreeing_negation_is_kept_and_attributed(self):
        claim = "Propolis is not used to seal small gaps in the hive."
        nodes = [_node(0.74,
                       "Chapter 9: Propolis is not used to seal small gaps "
                       "in the hive.", file_name=GUIDE, page="9")]
        out = self._run_agent(_FakeResponse(claim, nodes))
        self.assertEqual(self._claim_lines(out["answer"]),
                         [(f"{GUIDE}, page 9", claim)])

    def test_negation_mismatch_refused_before_any_attribution(self):
        self.captured_items = "not-called"
        nodes = [_node(0.74,
                       "Chapter 9: Propolis is not used to seal small gaps "
                       "in the hive.", file_name=GUIDE, page="9")]
        out = self._run_agent(_FakeResponse(PROP_CLAIM, nodes))
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE))
        self.assertEqual(self.captured_items, "not-called")


class TestCompoundMixedSupport(M4CBase):
    def test_unsupported_clause_removes_its_sentence_sibling_kept(self):
        mixed = "Propolis prevents fermentation during storage."
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1")]

        def _verdicts(item):
            if item["claim"] == PROP_CLAIM:
                return _verified(verifier.VERDICT_SUPPORTED)
            return _verified(verifier.VERDICT_INSUFFICIENT)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + mixed, nodes),
            verdict_for=_verdicts)
        answer = out["answer"]
        self.assertIn(PROP_CLAIM, answer)
        self.assertNotIn("prevents fermentation", answer)
        self.assertIn("could not confirm", answer)   # removed-note disclosure
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])


class TestModelFabricatedBlocksCannotHijackMapping(M4CBase):
    def test_fabricated_block_stripped_claim_line_is_the_only_footer(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1")]
        text = PROP_CLAIM + "\n\nReferences: fabricated.pdf, page 9"
        out = self._run_agent(_FakeResponse(text, nodes))
        answer = out["answer"]
        self.assertNotIn("fabricated.pdf", answer)
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])
        self.assertEqual(answer.count("References:"), 1)


class TestMissingMetadataPerClaim(M4CBase):
    def test_missing_metadata_yields_per_claim_unavailable_marker(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=None, page=None)]
        out = self._run_agent(_FakeResponse(PROP_CLAIM, nodes))
        self.assertEqual(
            self._claim_lines(out["answer"]),
            [(f"attribution unavailable for 1 supporting passage(s)",
              PROP_CLAIM)])


class TestRefusalAndUnavailableOutcomes(M4CBase):
    def test_refusal_carries_no_footer_and_no_mapping(self):
        nodes = [_node(0.74, BEE_PASSAGE, file_name=GUIDE, page="2")]

        def _verdicts(item):
            return _verified(verifier.VERDICT_CONTRADICTED)
        out = self._run_agent(_FakeResponse(WORKER_CLAIM, nodes),
                              verdict_for=_verdicts)
        answer = out["answer"]
        self.assertTrue(answer.startswith(NOT_FOUND_PHRASE))
        self.assertNotIn("References:", answer)
        self.assertNotIn("verifier-supported:", answer)
        self.assertNotIn(GUIDE, answer)

    def test_timeout_never_produces_attribution(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1")]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM, nodes),
            call_llm=lambda prompt, timeout: ("timeout", ""))
        answer = out["answer"]
        self.assertTrue(answer.startswith(NOT_FOUND_PHRASE))
        self.assertIn("semantic verification unavailable", answer)
        self.assertNotIn(GUIDE, answer)
        self.assertNotIn("verifier-supported:", answer)

    def test_malformed_output_never_produces_attribution(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1")]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM, nodes),
            call_llm=lambda prompt, timeout: ("ok", "not json at all"))
        answer = out["answer"]
        self.assertTrue(answer.startswith(NOT_FOUND_PHRASE))
        self.assertIn("semantic verification unavailable", answer)
        self.assertNotIn(GUIDE, answer)
        self.assertNotIn("verifier-supported:", answer)

    def test_unavailable_claim_is_never_presented_as_confirmed(self):
        # Sibling context: supported claim kept with its line; the
        # unavailable claim is removed and disclosed — never "confirmed".
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]

        def _verdicts(item):
            if item["claim"] == PROP_CLAIM:
                return _verified(verifier.VERDICT_SUPPORTED)
            return _unavailable(verifier.ERR_TIMEOUT)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes),
            verdict_for=_verdicts)
        answer = out["answer"]
        self.assertIn("verification was unavailable", answer)
        self.assertNotIn("Hohmann", answer)
        self.assertNotIn(ORBIT_REF, answer)
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])


class TestOffModeCompatibility(M4CBase):
    def test_off_mode_answer_is_byte_identical_m3(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes),
            mode="off")
        self.assertEqual(
            out["answer"],
            PROP_CLAIM + " " + ORBITAL_CLAIM +
            f"\n\nReferences: {GUIDE}, page 1; {ORBITAL}")
        self.assertNotIn("verifier-supported:", out["answer"])


class TestPrivacyAndMetadataIntegrity(M4CBase):
    def test_local_paths_absent_and_stored_metadata_untouched(self):
        secret = r"C:\Users\secret_user\books\urban_beekeeping_field_guide.pdf"
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1",
                       extra_meta={"file_path": secret})]
        snapshot = [dict(getattr(n, "metadata", None) or {}) for n in nodes]
        out = self._run_agent(_FakeResponse(PROP_CLAIM, nodes))
        answer = out["answer"]
        self.assertIn(PROP_REF, answer)          # validated citation present
        self.assertNotIn("secret_user", answer)  # username/abs path absent
        self.assertNotIn("file_path", answer)
        self.assertNotIn("\\", answer)
        # The agent must not mutate stored node metadata (persistence is
        # covered by check_chroma in the documented regression).
        self.assertEqual(
            [dict(getattr(n, "metadata", None) or {}) for n in nodes],
            snapshot)



class TestAdversarialFalseAttribution(M4CBase):
    """§3 offline adversarial battery: false attribution, cross-claim
    inheritance, fabrication, and sibling isolation. Offline fake nodes
    + deterministic verdicts only (no model, no network)."""

    def test_keyword_match_with_opposite_relationship_is_removed(self):
        # Shared vocabulary asserting the OPPOSITE relationship (model
        # never lays eggs): every token exists in the passage, so only
        # polarity/semantics can catch it. Scripted verifier contradicts.
        nodes = [_node(0.74, BEE_PASSAGE, file_name=GUIDE, page="2")]

        def _verdicts(item):
            return _verified(verifier.VERDICT_CONTRADICTED)
        out = self._run_agent(_FakeResponse(WORKER_CLAIM, nodes),
                              verdict_for=_verdicts)
        answer = out["answer"]
        self.assertTrue(answer.startswith(NOT_FOUND_PHRASE))
        self.assertNotIn("References:", answer)
        self.assertNotIn(GUIDE, answer)

    def test_correct_entity_wrong_number_removed_sibling_intact(self):
        # Same entity (hive) and a plausible figure, wrong population
        # (workers vs eggs): contradicted -> removed; the sibling keeps
        # its own claim and source.
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, BEE_PASSAGE, file_name=GUIDE, page="2")]

        def _verdicts(item):
            if item["claim"].startswith("Propolis"):
                return _verified(verifier.VERDICT_SUPPORTED)
            return _verified(verifier.VERDICT_CONTRADICTED)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + WORKER_CLAIM, nodes),
            verdict_for=_verdicts)
        answer = out["answer"]
        self.assertNotIn("2,000", answer)
        self.assertIn("conflict", answer)
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])
        self.assertNotIn("page 2", answer)

    def test_swapped_evidence_ids_reject_only_the_misattached_claim(self):
        # Two claims whose cited ids are SWAPPED (each cites the other's
        # evidence) through the REAL parser: both fail their own
        # whitelist -> both removed, fail-closed, no fabricated mapping.
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]

        def _llm(prompt, timeout):
            # Both claims see prompts carrying both evidence blocks
            # (e0 + e1) and cite the id that belongs to the OTHER claim.
            if "Propolis is a resinous" in prompt:
                cited = "e1"
            else:
                cited = "e0"
            payload = {"verdict": "supported", "evidence_ids": [cited],
                       "reason": "ok"}
            return "ok", json.dumps(payload)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes),
            call_llm=_llm)
        answer = out["answer"]
        self.assertTrue(answer.startswith(NOT_FOUND_PHRASE))
        self.assertIn("semantic verification unavailable", answer)
        self.assertNotIn("verifier-supported:", answer)
        self.assertNotIn(GUIDE, answer)
        self.assertNotIn(ORBITAL, answer)

    def test_valid_id_insufficient_verdict_removed_sibling_intact(self):
        # The verdict cites a VALID id from its own evidence but judges it
        # insufficient — id validity never implies support. Through the
        # REAL parser the claim is removed + disclosed; the sibling keeps
        # exactly its own source (the removed claim's source never leaks).
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]

        def _llm(prompt, timeout):
            if "Propolis is a resinous" in prompt:
                payload = {"verdict": "insufficient_evidence",
                           "evidence_ids": ["e0"],
                           "reason": "evidence does not entail the claim"}
            else:
                payload = {"verdict": "supported",
                           "evidence_ids": ["e1"], "reason": "ok"}
            return "ok", json.dumps(payload)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes),
            call_llm=_llm)
        answer = out["answer"]
        self.assertNotIn(PROP_CLAIM, answer)
        self.assertIn("could not confirm", answer)
        self.assertEqual(self._claim_lines(answer),
                         [(ORBIT_REF, ORBITAL_CLAIM)])
        self.assertNotIn(GUIDE, answer)

    def test_partially_overlapping_evidence_stays_per_claim(self):
        # Both claims' token sets are covered by BOTH nodes (shared corpus
        # sentence), yet each keeps only its OWN source: coverage is
        # per-claim, never shared by adjacency.
        shared = "Honey is harvested only when its moisture content drops below 18 percent."
        nodes = [_node(0.74, PROPOLIS_PASSAGE + " " + shared,
                       file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE + " " + shared,
                       file_name=ORBITAL, page=None)]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes))
        self.assertEqual(self._claim_lines(out["answer"]),
                         [(PROP_REF, PROP_CLAIM),
                          (ORBIT_REF, ORBITAL_CLAIM)])
        self.assertNotIn("page 1;", out["answer"])  # no merged/dup refs

    def test_multi_passage_claim_only_one_passage_relevant_narrows(self):
        # Deterministic cover picks BOTH nodes; the verifier's citation
        # confirms only the relevant passage -> attribution narrows to it.
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM, nodes),
            verdict_for=lambda item: _verified(
                verifier.VERDICT_SUPPORTED, ids=("e0",)))
        self.assertEqual(self._claim_lines(out["answer"]),
                         [(PROP_REF, PROP_CLAIM)])
        self.assertNotIn(ORBIT_REF, out["answer"])

    def test_removed_claims_source_is_never_inherited(self):
        # Three claims, middle one removed: exactly two lines, refs stay
        # with their own claims, the removed claim's source vanishes.
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.72, ORBITAL_PASSAGE, file_name=ORBITAL, page=None),
                 _node(0.70, BEE_PASSAGE, file_name=GUIDE, page="2")]

        def _verdicts(item):
            if item["claim"] == ORBITAL_CLAIM:
                return _verified(verifier.VERDICT_CONTRADICTED)
            return _verified(verifier.VERDICT_SUPPORTED)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM + " " +
                          BEE_CLAIM, nodes),
            verdict_for=_verdicts)
        answer = out["answer"]
        self.assertNotIn("Hohmann", answer)
        self.assertEqual(self._claim_lines(answer),
                         [(PROP_REF, PROP_CLAIM),
                          (f"{GUIDE}, page 2", BEE_CLAIM)])
        self.assertNotIn(ORBIT_REF, answer)  # removed claim's source gone

    def test_mixed_supported_and_unsupported_stay_scoped(self):
        # Three claims: supported, contradicted, unavailable -> only the
        # supported one survives with its own line and disclosures cover
        # the other two.
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.72, ORBITAL_PASSAGE, file_name=ORBITAL, page=None),
                 _node(0.70, BEE_PASSAGE, file_name=GUIDE, page="2")]

        def _verdicts(item):
            if item["claim"] == PROP_CLAIM:
                return _verified(verifier.VERDICT_SUPPORTED)
            if item["claim"] == ORBITAL_CLAIM:
                return _verified(verifier.VERDICT_CONTRADICTED)
            return _unavailable(verifier.ERR_TIMEOUT)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM + " " +
                          WORKER_CLAIM, nodes),
            verdict_for=_verdicts)
        answer = out["answer"]
        self.assertIn(PROP_CLAIM, answer)
        self.assertNotIn("Hohmann", answer)
        self.assertNotIn("Worker bees", answer)
        self.assertIn("conflict", answer)
        self.assertIn("verification was unavailable", answer)
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])

    def test_fabricated_page_filename_table_in_text_removed(self):
        # The model invents a page number, a filename, and a table id
        # absent from the retrieved nodes: fabricated attribution, never
        # kept — and the fabricated sentence is dropped at the GATE, so it
        # is never even shown to the verifier (only the clean claim is).
        fabricated = (PROP_CLAIM + " This is documented on page 99 of "
                      "field_notes_2024.pdf in table 7.")
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1")]
        out = self._run_agent(_FakeResponse(fabricated, nodes))
        answer = out["answer"]
        self.assertEqual([i["claim"] for i in self.captured_items],
                         [PROP_CLAIM])   # fabricated sentence never verified
        self.assertNotIn("page 99", answer)
        self.assertNotIn("field_notes_2024.pdf", answer)
        self.assertNotIn("table 7", answer)
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])


class TestPartialMultiClaimResults(M4CBase):
    """§5 sibling isolation: a PARTIALLY valid multi-claim response —
    verify_all returned fewer results than claims — removes exactly the
    claims without results and never affects their siblings."""

    def test_missing_sibling_result_is_unavailable_not_a_pass(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]

        def _short_results(items):
            # Result for claim 1 only: claim 2 has NO result at all.
            self.captured_items = items
            return [_verified(verifier.VERDICT_SUPPORTED)]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes),
            verify_all=_short_results)
        answer = out["answer"]
        self.assertIn(PROP_CLAIM, answer)
        self.assertNotIn("Hohmann", answer)          # never a silent pass
        self.assertIn("verification was unavailable", answer)
        self.assertEqual(self._claim_lines(answer), [(PROP_REF, PROP_CLAIM)])
        self.assertNotIn(ORBIT_REF, answer)

    def test_missing_results_remove_all_claims_fail_closed(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1")]
        out = self._run_agent(_FakeResponse(PROP_CLAIM, nodes),
                              verify_all=lambda items: [])
        answer = out["answer"]
        self.assertTrue(answer.startswith(NOT_FOUND_PHRASE))
        self.assertIn("semantic verification unavailable", answer)
        self.assertNotIn("verifier-supported:", answer)
        self.assertNotIn(GUIDE, answer)


class TestFooterWordingIsTheVerifierJudgment(M4CBase):
    """§4: the parenthetical ATTRIBUTES the support judgment to the bounded
    verifier — it must not read as independently established entailment,
    and off mode must render no per-claim judgment at all."""

    def test_local_line_format_pins_the_verifier_qualifier(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.71, ORBITAL_PASSAGE, file_name=ORBITAL, page=None)]
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + ORBITAL_CLAIM, nodes))
        answer = out["answer"]
        self.assertIn(
            f"References:\n"
            f"[1] {PROP_REF} (verifier-supported: {PROP_CLAIM})\n"
            f"[2] {ORBIT_REF} (verifier-supported: {ORBITAL_CLAIM})",
            answer)
        # The unqualified assertion is gone everywhere:
        self.assertNotIn("supports:", answer)

    def test_off_mode_renders_no_per_claim_judgment(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1")]
        out = self._run_agent(_FakeResponse(PROP_CLAIM, nodes), mode="off")
        self.assertEqual(out["answer"],
                         PROP_CLAIM + f"\n\nReferences: {PROP_REF}")
        self.assertNotIn("verifier-supported:", out["answer"])
        self.assertNotIn("supports:", out["answer"])


class TestRealParserContradictedIsANegativeNotUnavailable(M4CBase):
    """§5: a completed contradicting verdict (real parser) is a
    SUCCESSFUL negative — refusal WITHOUT the unavailable marker, proving
    the marker distinguishes 'verification failed' from 'verified false'."""

    def test_contradicted_both_claims_refuses_without_marker(self):
        nodes = [_node(0.74, PROPOLIS_PASSAGE, file_name=GUIDE, page="1"),
                 _node(0.70, BEE_PASSAGE, file_name=GUIDE, page="2")]

        def _llm(prompt, timeout):
            ids = ["e0"] if "Propolis is a resinous" in prompt else ["e1"]
            payload = {"verdict": "contradicted", "evidence_ids": ids,
                       "reason": "evidence reverses the relation"}
            return "ok", json.dumps(payload)
        out = self._run_agent(
            _FakeResponse(PROP_CLAIM + " " + WORKER_CLAIM, nodes),
            call_llm=_llm)
        answer = out["answer"]
        self.assertTrue(answer.startswith(NOT_FOUND_PHRASE))
        self.assertNotIn("semantic verification unavailable", answer)
        self.assertNotIn("References:", answer)
        self.assertNotIn("verifier-supported:", answer)
        self.assertNotIn(GUIDE, answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
