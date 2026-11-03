"""RAG-AE harness test suite (offline, deterministic; no models, no network).

Covers, per the RAG-AE Phase 6 verification list:
  * gold-v2 dataset schema validation (every rejection rule + the real file);
  * metric/check math on hand-built examples;
  * answerable vs unanswerable vs empty-context vs controlled-failure cases;
  * retrieval ranking / evidence-coverage scoring (hand-built rows plus one
    small REAL isolated integration run through the production ingest path
    and the production seam, all inside a temporary store);
  * citation validation, refusal-recovery regression via the real agent;
  * production-path isolation guard tests (production Chroma refused);
  * JSON artifact schema, exit codes (0/1/2/3/4), determinism, no-network.

Drives the REAL books_agent with faked engines and scripted LLM replies —
no Ollama, no Gemini, no network, no production store.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_answers as ea  # noqa: E402
from app import chroma_guard  # noqa: E402
from app.config import NOT_FOUND_PHRASE  # noqa: E402

GOLD_SHA256 = "b34d68894c44a7249f7e01c5610e005e5b8ba4f3d35e61d70e9d8e43f650b5d1"


def load_gold():
    ds, sha = ea.load_gold_v2(ea.GOLD_PATH)
    return ds, sha


def case_by_id(ds, cid):
    return next(c for c in ds["cases"] if c["id"] == cid)


# --------------------------------------------------------------------------
# dataset validation
# --------------------------------------------------------------------------
class DatasetValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds, cls.sha = load_gold()
        cls.source_texts = ea.load_source_texts(
            {e["file_name"] for c in cls.ds["cases"]
             for e in (c.get("evidence") or [])
             if e.get("source") in ("corpus", "fixture") and e.get("file_name")})

    def test_real_dataset_is_valid(self):
        self.assertEqual(ea.validate_dataset(self.ds, self.source_texts), [])

    def test_dataset_sha_pinned(self):
        self.assertEqual(self.sha, GOLD_SHA256)

    def test_unresolved_cases_recorded_and_excluded(self):
        unresolved = [c for c in self.ds["cases"] if c["status"] == "unresolved"]
        self.assertEqual(sorted(c["id"] for c in unresolved),
                         ["C101", "C102", "C103"])
        for case in unresolved:
            self.assertTrue(case.get("unresolved_reason"))
            self.assertTrue(case.get("resolution_findings"))
            self.assertNotIn("expect", case)

    def _errors_for(self, mutate):
        ds = json.loads(json.dumps(self.ds))
        mutate(ds)
        return ea.validate_dataset(ds, self.source_texts)

    def test_duplicate_ids_rejected(self):
        def mutate(ds):
            ds["cases"].append(dict(ds["cases"][0]))
        errs = self._errors_for(mutate)
        self.assertTrue(any("duplicate case id" in e for e in errs), errs)

    def test_short_query_rejected(self):
        def mutate(ds):
            case_by_id(ds, "C001")["query"] = "huh?"
        errs = self._errors_for(mutate)
        self.assertTrue(any("field query" in e for e in errs), errs)

    def test_nonexistent_source_document_rejected(self):
        def mutate(ds):
            case_by_id(ds, "C001")["evidence"][0]["file_name"] = "ghost.docx"
        errs = self._errors_for(mutate)
        self.assertTrue(any("missing on disk" in e for e in errs), errs)

    def test_non_verbatim_evidence_rejected(self):
        def mutate(ds):
            case_by_id(ds, "C003")["evidence"][0]["text"] = \
                "Geostationary orbit lies at an altitude of approximately " \
                "36,000 kilometers above the equator."
        errs = self._errors_for(mutate)
        self.assertTrue(any("not a verbatim substring" in e for e in errs), errs)

    def test_answerable_without_supporting_evidence_rejected(self):
        def mutate(ds):
            case = case_by_id(ds, "C001")
            case["required_facts"].append("the bees orbit at kilometer nine")
        errs = self._errors_for(mutate)
        self.assertTrue(any("no identifiable support" in e for e in errs), errs)

    def test_unanswerable_must_expect_abstention(self):
        def mutate(ds):
            case_by_id(ds, "C006")["expect"]["abstain"] = False
        errs = self._errors_for(mutate)
        self.assertTrue(any("must expect abstention" in e for e in errs), errs)

    def test_citation_target_must_be_evidence_file(self):
        def mutate(ds):
            case_by_id(ds, "C001")["expect"]["citation_files"] = \
                ["urban_beekeeping_field_guide.pdf"]
        errs = self._errors_for(mutate)
        self.assertTrue(any("not among the case's evidence files" in e
                            for e in errs), errs)

    def test_unresolved_without_reason_rejected(self):
        def mutate(ds):
            case_by_id(ds, "C103").pop("unresolved_reason")
        errs = self._errors_for(mutate)
        self.assertTrue(any("must record a reason" in e for e in errs), errs)

    def test_empty_evidence_only_for_abstention_cases(self):
        def mutate(ds):
            case_by_id(ds, "C001")["evidence"] = []
        errs = self._errors_for(mutate)
        self.assertTrue(any("empty evidence" in e for e in errs), errs)

    def test_bad_score_rejected(self):
        def mutate(ds):
            case_by_id(ds, "C001")["evidence"][0]["score"] = 9.5
        errs = self._errors_for(mutate)
        self.assertTrue(any("score must be a number" in e for e in errs), errs)


# --------------------------------------------------------------------------
# pure utilities + embedding double
# --------------------------------------------------------------------------
class UtilityTests(unittest.TestCase):
    def test_content_tokens_numbers_and_hyphens(self):
        toks = ea.content_tokens("Delta-v of 2.4 km/s; 35,786 kilometers")
        self.assertIn("delta", toks)
        self.assertIn("v", toks)
        self.assertIn("2.4", toks)
        self.assertIn("35786", toks)       # commas stripped, digits kept
        self.assertIn("kilometers", toks)

    def test_number_tokens(self):
        nums = ea.number_tokens("about 60,000 bees and 92.4 minutes (see)")
        self.assertEqual(nums, {"60000", "92.4"})

    def test_split_footer_and_file_entries(self):
        answer = ("Text here.\n\nReferences: a.pdf, page 1; "
                  "attribution unavailable for 2 supporting passage(s)")
        body, entries = ea.split_footer(answer)
        self.assertEqual(body, "Text here.")
        self.assertEqual(len(entries), 2)
        self.assertEqual(ea.file_entries(entries), ["a.pdf"])
        body2, entries2 = ea.split_footer("No footer at all.")
        self.assertIsNone(entries2)

    def test_exempt_sentences(self):
        self.assertTrue(ea.is_exempt_sentence(NOT_FOUND_PHRASE))
        self.assertTrue(ea.is_exempt_sentence(ea._PARTIAL_DISCLOSURE))
        self.assertTrue(ea.is_exempt_sentence("(best similarity score 0.21)"))
        self.assertFalse(ea.is_exempt_sentence("Bees keep free of comb."))

    def test_distinctive_query_tokens_drop_stopwords(self):
        toks = ea.distinctive_query_tokens(
            "What is the moisture threshold in the apiary hive?")
        self.assertIn("moisture", toks)
        self.assertIn("threshold", toks)
        self.assertIn("apiary", toks)
        self.assertIn("hive", toks)
        # stopwords dropped by the PRODUCTION stop list (reused deliberately)
        self.assertNotIn("the", toks)
        self.assertNotIn("in", toks)
        self.assertNotIn("is", toks)
        # document-generic words are production stopwords too
        self.assertNotIn("guide", toks)

    def test_hashed_embedding_deterministic_and_normalized(self):
        emb = ea.HashedTfEmbedding()
        v1 = emb._get_query_embedding("the apiary inspection notes")
        v2 = emb._get_text_embedding("the apiary inspection notes")
        self.assertEqual(len(v1), 512)
        self.assertEqual(v1, v2)
        norm = sum(x * x for x in v1) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)
        other = emb._get_text_embedding("orbital delta maneuver")
        self.assertNotEqual(v1, other)
        self.assertEqual(emb.class_name(), "HashedTfEmbedding")

    def test_hashed_embedding_ignores_stopwords(self):
        emb = ea.HashedTfEmbedding()
        only_stops = emb._get_text_embedding("the of and is are was were")
        self.assertTrue(all(x == 0.0 for x in only_stops))


# --------------------------------------------------------------------------
# scoring checks on hand-built examples
# --------------------------------------------------------------------------
class ScoringCheckTests(unittest.TestCase):
    def test_abstention_both_directions(self):
        ok = ea.check_abstention(NOT_FOUND_PHRASE, True)
        self.assertEqual(ok["verdict"], "pass")
        false_refusal = ea.check_abstention(NOT_FOUND_PHRASE, False)
        self.assertEqual(false_refusal["verdict"], "fail")
        self.assertIn("FALSE REFUSAL", false_refusal["reason"])
        fabrication = ea.check_abstention("Bees live 40 days.", True)
        self.assertEqual(fabrication["verdict"], "fail")

    def test_facts_check(self):
        answer = "The burn costs roughly 2.4 kilometers per second of delta-v."
        passed = ea.check_facts(answer, ["roughly 2.4 kilometers per second"])
        self.assertEqual(passed["verdict"], "pass")
        failed = ea.check_facts(answer, ["approximately 1.5 kilometers"])
        self.assertEqual(failed["verdict"], "fail")
        self.assertIn("approximately 1.5 kilometers", failed["missing"])

    def test_forbidden_check(self):
        self.assertEqual(ea.check_forbidden("safe text", ["oxalic"])["verdict"],
                         "pass")
        hit = ea.check_forbidden("oxalic acid vaporization", ["oxalic acid"])
        self.assertEqual(hit["verdict"], "fail")

    def test_groundedness_pass_and_leak(self):
        evidence = ["Honey is harvested only when its moisture content drops "
                    "below 18 percent, which prevents fermentation."]
        good = ("Honey is harvested only when its moisture content drops below "
                "18 percent, which prevents fermentation.\n\n"
                "References: hive.pdf, page 1")
        self.assertEqual(ea.check_groundedness(good, evidence, "")["verdict"],
                         "pass")
        leaked = ea.check_groundedness(
            "Honey is harvested only when its moisture content drops below "
            "18 percent, which prevents fermentation. Oxalic acid works.",
            evidence, "")
        self.assertEqual(leaked["verdict"], "fail")
        self.assertTrue(leaked["problems"][0]["missing_tokens"])

    def test_groundedness_number_must_match_evidence(self):
        evidence = ["The hive holds 60,000 worker bees in midsummer."]
        bad = ea.check_groundedness("The hive holds 70,000 worker bees.",
                                    evidence, "")
        self.assertEqual(bad["verdict"], "fail")
        self.assertTrue(bad["problems"][0]["missing_numbers"])

    def test_groundedness_disclosure_is_exempt(self):
        evidence = ["Honey is harvested only when its moisture content drops "
                    "below 18 percent."]
        answer = ("Honey is harvested only when its moisture content drops "
                  "below 18 percent. " + ea._PARTIAL_DISCLOSURE)
        self.assertEqual(ea.check_groundedness(answer, evidence, "")["verdict"],
                         "pass")

    def test_citation_required_pass_and_wrong_file(self):
        expect = {"citation": "required", "citation_files": ["a.pdf"]}
        answer = "Claim.\n\nReferences: a.pdf, page 1"
        self.assertEqual(
            ea.check_citation(answer, expect, {"a.pdf"}, {"a.pdf": {"1"}})["verdict"],
            "pass")
        swapped = ea.check_citation(
            "Claim.\n\nReferences: b.pdf, page 1", expect,
            {"a.pdf", "b.pdf"}, {"b.pdf": {"1"}})
        self.assertEqual(swapped["verdict"], "fail")

    def test_citation_forbidden_and_unavailable(self):
        self.assertEqual(
            ea.check_citation("Claim.\n\nReferences: a.pdf",
                              {"citation": "forbidden"},
                              {"a.pdf"}, {})["verdict"], "fail")
        self.assertEqual(
            ea.check_citation("Claim.", {"citation": "forbidden"},
                              {"a.pdf"}, {})["verdict"], "pass")
        self.assertEqual(
            ea.check_citation(
                "Claim.\n\nReferences: attribution unavailable for 1 "
                "supporting passage(s)",
                {"citation": "unavailable"}, set(), {})["verdict"], "pass")

    def test_citation_page_mismatch(self):
        expect = {"citation": "required", "citation_files": ["a.pdf"]}
        bad = ea.check_citation("Claim.\n\nReferences: a.pdf, page 9", expect,
                                {"a.pdf"}, {"a.pdf": {"1"}})
        self.assertEqual(bad["verdict"], "fail")
        self.assertIn("not in retrieved pages", bad["reason"])

    def test_citation_forbidden_source_file(self):
        expect = {"citation": "required", "citation_files": ["a.pdf"],
                  "forbidden_citation_files": ["orchard_manual.pdf"]}
        bad = ea.check_citation("Claim.\n\nReferences: a.pdf; orchard_manual.pdf",
                                expect, {"a.pdf"}, {})
        self.assertEqual(bad["verdict"], "fail")

    def test_disclosure_check(self):
        answer = "fact. " + ea._PARTIAL_DISCLOSURE
        self.assertEqual(ea.check_disclosure(answer, "partial")["verdict"], "pass")
        self.assertEqual(ea.check_disclosure("fact.", "partial")["verdict"], "fail")
        self.assertEqual(ea.check_disclosure("fact.", "none")["verdict"], "pass")

    def test_recovery_check_markers(self):
        stdout = ("[books_agent] self_refusal_recovery_attempted (score=0.7)\n"
                  "[books_agent] self_refusal_recovery_succeeded (chars=10)\n")
        ok = ea.check_recovery(stdout, {"recovery_attempted": True,
                                        "recovery_outcome": "succeeded",
                                        "llm_calls": 1}, 1)
        self.assertEqual(ok["verdict"], "pass")
        bad = ea.check_recovery("", {"recovery_attempted": True,
                                     "recovery_outcome": "succeeded",
                                     "llm_calls": 1}, 0)
        self.assertEqual(bad["verdict"], "fail")


# --------------------------------------------------------------------------
# real-agent execution on dataset cases
# --------------------------------------------------------------------------
class RealAgentCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds, _ = load_gold()

    def _run(self, cid):
        case = case_by_id(self.ds, cid)
        outcome = ea.run_unit_case(case)
        return case, outcome, ea.score_unit_case(case, outcome)

    def test_c001_fully_grounded_pass(self):
        _, outcome, res = self._run("C001")
        self.assertEqual(res["status"], "pass")
        self.assertIn("orbital_kinetics_principles.docx", res["final_answer"])
        self.assertIn("References:", res["final_answer"])
        self.assertEqual(outcome["llm_calls"], 0)

    def test_c005_keeps_fact_drops_fabrication_discloses(self):
        _, _, res = self._run("C005")
        self.assertEqual(res["status"], "pass")
        answer = res["final_answer"]
        self.assertIn("below 18 percent", answer)
        self.assertIn(ea._PARTIAL_DISCLOSURE, answer)
        self.assertNotIn("oxalic", answer)
        self.assertIn("urban_beekeeping_field_guide.pdf, page 1", answer)

    def test_c006_abstains_with_zero_recovery(self):
        _, outcome, res = self._run("C006")
        self.assertEqual(res["status"], "pass")
        self.assertTrue(res["final_answer"].startswith(NOT_FOUND_PHRASE))
        self.assertNotIn("References:", res["final_answer"])
        self.assertEqual(outcome["llm_calls"], 0)

    def test_c009_conflicting_passages_abstain(self):
        _, _, res = self._run("C009")
        self.assertEqual(res["status"], "pass")
        self.assertTrue(res["final_answer"].startswith(NOT_FOUND_PHRASE))

    def test_c011_empty_retrieval_abstains(self):
        _, _, res = self._run("C011")
        self.assertEqual(res["status"], "pass")
        self.assertTrue(res["final_answer"].startswith(NOT_FOUND_PHRASE))

    def test_c012_controlled_failure_propagates(self):
        _, outcome, res = self._run("C012")
        self.assertEqual(res["status"], "pass")
        self.assertIn("RuntimeError", outcome["error"] or "")
        self.assertIsNone(res["final_answer"])

    def test_c013_malformed_metadata_never_invents_filename(self):
        _, _, res = self._run("C013")
        self.assertEqual(res["status"], "pass")
        self.assertIn("attribution unavailable", res["final_answer"])
        self.assertNotIn(".pdf", res["final_answer"])
        self.assertNotIn(".docx", res["final_answer"])

    def test_c014_recovery_succeeds_once_with_grounded_retry(self):
        _, outcome, res = self._run("C014")
        self.assertEqual(res["status"], "pass")
        self.assertEqual(outcome["llm_calls"], 1)
        self.assertIn("self_refusal_recovery_succeeded", outcome["stdout"])
        self.assertIn("References:", res["final_answer"])

    def test_c015_second_refusal_abstains_bounded(self):
        _, outcome, res = self._run("C015")
        self.assertEqual(res["status"], "pass")
        self.assertEqual(outcome["llm_calls"], 1)      # never a third attempt
        self.assertIn("self_refusal_recovery_rejected", outcome["stdout"])
        self.assertTrue(res["final_answer"].startswith(NOT_FOUND_PHRASE))

    def test_c016_planted_filename_never_cited(self):
        _, _, res = self._run("C016")
        self.assertEqual(res["status"], "pass")
        self.assertIn("oiled monthly", res["final_answer"])
        self.assertNotIn("orchard_manual", res["final_answer"])
        self.assertIn("References: fixture_adversarial.docx", res["final_answer"])

    def test_false_refusal_is_detected_and_diagnosed(self):
        # Same grounded C001 case, but the model double refuses AND the
        # recovery double refuses again: production must abstain — the
        # harness must flag this as a FALSE REFUSAL against the answerable
        # expectation (the C002 live-defect class).
        case = dict(case_by_id(self.ds, "C001"))
        case["raw_response"] = NOT_FOUND_PHRASE
        case["recovery_replies"] = [NOT_FOUND_PHRASE]
        case["expect"] = dict(case["expect"], recovery_attempted=True,
                              recovery_outcome="rejected", llm_calls=1)
        res = ea.score_unit_case(case, ea.run_unit_case(case))
        # dataset expectation is abstain=false -> the abstention is wrong
        self.assertEqual(res["status"], "fail")
        self.assertIn("abstention", res["failed_checks"])
        self.assertTrue(any("FALSE REFUSAL" in d for d in res["diagnostics"]))

    def test_fixture_defect_diagnostic_when_fact_not_in_evidence(self):
        case = dict(case_by_id(self.ds, "C003"))
        case["required_facts"] = ["approximately 35,786 kilometers",
                                  "a sentence absent from all evidence"]
        checks = {"facts_present": ea.check_facts(
            case["raw_response"], case["required_facts"])}
        diags = ea._classify_failure(case, {"answer": case["raw_response"]},
                                     {**checks,
                                      "facts_present": dict(
                                          checks["facts_present"])})
        self.assertTrue(any("fixture-or-evaluation-definition problem" in d
                            for d in diags), diags)

    def test_aggregate_and_category_math(self):
        results = [ea.score_unit_case(c, ea.run_unit_case(c))
                   for c in (case_by_id(self.ds, "C001"),
                             case_by_id(self.ds, "C006"))]
        dims = ea.aggregate_answers(results)
        self.assertEqual(dims["abstention"]["applicable"], 2)
        self.assertEqual(dims["abstention"]["passed"], 2)
        abst = ea.compute_abstention(
            [case_by_id(self.ds, "C001"), case_by_id(self.ds, "C006")], results)
        self.assertEqual(abst, {"true_abstain": 1, "false_abstain": 0,
                                "missed_abstain": 0, "true_answer": 1,
                                "precision": 1.0, "recall": 1.0})
        cats = ea.by_category(
            [case_by_id(self.ds, "C001"), case_by_id(self.ds, "C006")], results)
        self.assertEqual(cats["correct-abstention"], {"applicable": 1,
                                                      "passed": 1, "rate": 1.0})


# --------------------------------------------------------------------------
# integration-layer scoring (hand-built rows)
# --------------------------------------------------------------------------
class IntegrationScoringTests(unittest.TestCase):
    def _case(self):
        return {"id": "X", "answerable": True,
                "gold_documents": ["a.pdf"], "gold_pages": [1],
                "min_docs_covered": 1,
                "required_facts": ["hive holds 60,000 bees"]}

    def test_covered_case_passes_with_fact_coverage(self):
        rows = [{"file_name": "a.pdf", "page_label": "1", "score": 0.7,
                 "text": "the hive holds 60,000 bees in midsummer"}]
        res = ea.score_integration_case(self._case(), rows, 4)
        self.assertEqual(res["status"], "pass")
        self.assertTrue(res["fact_coverage"])
        self.assertTrue(res["metrics"]["coverage"])

    def test_absent_evidence_fails_with_diagnostic(self):
        rows = [{"file_name": "b.pdf", "page_label": "2", "score": 0.4,
                 "text": "unrelated text"}]
        res = ea.score_integration_case(self._case(), rows, 4)
        self.assertEqual(res["status"], "fail")
        self.assertIn("expected evidence absent from retrieval (top-4)",
                      res["diagnostics"])

    def test_context_assembly_gap_diagnostic(self):
        case = self._case()
        case["required_facts"] = ["a fact whose text is not retrieved"]
        rows = [{"file_name": "a.pdf", "page_label": "1", "score": 0.7,
                 "text": "some text from the gold document"}]
        res = ea.score_integration_case(case, rows, 4)
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("context-assembly gap" in d
                            for d in res["diagnostics"]), res["diagnostics"])

    def test_no_gold_case_is_skipped(self):
        case = {"id": "Y", "answerable": False, "gold_documents": [],
                "required_facts": []}
        res = ea.score_integration_case(case, [], 4)
        self.assertEqual(res["status"], "skipped")

    def test_aggregate_retrieval_math(self):
        per_case = [
            {"metrics": {"hit@1": True, "hit@3": True, "coverage": True,
                         "precision@1": 1.0, "precision@3": 0.5,
                         "reciprocal_rank": 1.0, "ndcg@5": 1.0},
             "fact_coverage": True},
            {"metrics": {"hit@1": False, "hit@3": True, "coverage": True,
                         "precision@1": 0.0, "precision@3": 0.25,
                         "reciprocal_rank": 0.5, "ndcg@5": 0.5},
             "fact_coverage": False},
        ]
        agg = ea.aggregate_retrieval(per_case)
        self.assertEqual(agg["cases_scored"], 2)
        self.assertEqual(agg["hit@1"], 0.5)
        self.assertEqual(agg["hit@3"], 1.0)
        self.assertEqual(agg["reciprocal_rank"], 0.75)
        self.assertEqual(agg["fact_coverage"], 0.5)


# --------------------------------------------------------------------------
# isolation guards
# --------------------------------------------------------------------------
class IsolationGuardTests(unittest.TestCase):
    def test_verify_isolation_accepts_temp_rejects_production(self):
        with tempfile.TemporaryDirectory() as td:
            info = ea.verify_isolation(Path(td))
            self.assertEqual(info["store"], str(Path(td).resolve()))
        with self.assertRaises(ea.ev.GuardFailure):
            ea.verify_isolation(Path(ea.ev.STORAGE_DIR))
        with self.assertRaises(ea.ev.GuardFailure):
            ea.verify_isolation(PROJECT_ROOT / "storage_new")

    def test_production_store_path_refused_by_chroma_guard(self):
        chroma_guard.arm()
        with self.assertRaises(chroma_guard.ChromaGuardError):
            chroma_guard.assert_store_path_allowed(ea.ev.STORAGE_DIR)
        import chromadb
        with self.assertRaises(chroma_guard.ChromaGuardError):
            chromadb.PersistentClient(path=str(ea.ev.STORAGE_DIR))

    def test_network_guard_blocks_and_records(self):
        import urllib.request
        guard = ea.NetworkGuard()
        guard.install()
        try:
            with self.assertRaises(RuntimeError):
                urllib.request.urlopen("http://localhost:11434/api/tags")
            import socket
            with self.assertRaises(RuntimeError):
                socket.create_connection(("127.0.0.1", 11434))
        finally:
            guard.uninstall()
        self.assertEqual(len(guard.attempts), 2)
        # restored behavior
        self.assertNotEqual(urllib.request.urlopen,
                            guard._blocked("x"))


# --------------------------------------------------------------------------
# small REAL isolated integration run (production ingest + seam, temp store)
# --------------------------------------------------------------------------
class RealIntegrationRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds, _ = load_gold()

    def test_mini_integration_run_is_isolated_and_passes(self):
        selected = [case_by_id(self.ds, cid)
                    for cid in ("C001", "C008", "C010", "C006")]
        guard_before = ea.ev.guard_snapshot(ea.ev.GUARD_FILES)
        integ = ea.run_integration(selected, 4, "pytest mini run")
        guard_after = ea.ev.guard_snapshot(ea.ev.GUARD_FILES)
        self.assertEqual(guard_before, guard_after)
        by_id = {c["id"]: c for c in integ["per_case"]}
        self.assertEqual(by_id["C001"]["status"], "pass")
        self.assertEqual(by_id["C008"]["status"], "pass")
        self.assertEqual(by_id["C010"]["status"], "pass")
        self.assertEqual(by_id["C006"]["status"], "skipped")
        iso = integ["isolation"]
        self.assertTrue(iso["all_clients_on_temp_store"])
        self.assertTrue(iso["guard_files_unchanged"])
        for path in iso["client_paths"]:
            self.assertTrue(path.replace("\\", "/").startswith(
                tempfile.gettempdir().replace("\\", "/")), path)
        for name, status in iso["files_ingested"].items():
            self.assertIn(status, ("indexed", "partial"), name)
        self.assertGreaterEqual(integ["aggregate"]["cases_scored"], 3)

    def test_retrieve_full_returns_full_text(self):
        # Same machinery, asserted directly: seam text must NOT be the 120-char
        # truncation used by eval_retrieval.query_production_seam.
        selected = [case_by_id(self.ds, "C001")]
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            ea.verify_isolation(temp_dir)
            embedder = ea.HashedTfEmbedding()
            ea.ev.instrument_client_paths()
            try:
                ea.ev.build_eval_index(
                    temp_dir, ea.EVAL_COLLECTION,
                    [ea._source_path("orbital_kinetics_principles.docx")],
                    embedder)
                rows = ea.retrieve_full(temp_dir, ea.EVAL_COLLECTION,
                                        ["delta-v Hohmann transfer"], embedder, 4)
            finally:
                ea.ev.release_chroma_clients()
        self.assertTrue(rows and rows[0])
        top_text = rows[0][0]["text"]
        self.assertGreater(len(top_text), 120)
        self.assertIn("delta-v", top_text)


# --------------------------------------------------------------------------
# CLI: exit codes, artifact schema, determinism
# --------------------------------------------------------------------------
class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "out.json"

    def _write_gold(self, name, mutate=None):
        ds, _ = load_gold()
        if mutate:
            mutate(ds)
        path = Path(self._tmp.name) / name
        path.write_text(json.dumps(ds), encoding="utf-8")
        return path

    def test_unit_mode_exit0_and_artifact_schema(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = ea.main(["--mode", "unit", "--out", str(self.out)])
        self.assertEqual(code, 0)
        rec = json.loads(self.out.read_text(encoding="utf-8"))
        for key in ("harness", "mode", "run_id", "timestamp_utc", "git",
                    "environment", "dataset", "config", "isolation",
                    "external_calls", "counts", "answer_quality", "retrieval",
                    "definitions", "cases"):
            self.assertIn(key, rec)
        self.assertEqual(rec["mode"], "unit")
        self.assertIsNone(rec["retrieval"])
        counts = rec["counts"]
        self.assertEqual(counts["discovered"],
                         counts["passed"] + counts["failed"]
                         + counts["skipped"] + counts["unresolved"])
        self.assertEqual(counts["failed"], 0)
        self.assertEqual(counts["unresolved"], 3)
        self.assertTrue(rec["external_calls"]["offline_deterministic"])
        self.assertEqual(rec["isolation"]["network_attempts"], [])
        self.assertTrue(rec["isolation"]["guard_files_unchanged"])
        self.assertEqual(rec["dataset"]["sha256"], GOLD_SHA256)
        c001 = next(c for c in rec["cases"] if c["id"] == "C001")
        self.assertEqual(c001["status"], "pass")
        self.assertIn("abstention", c001["checks"])
        self.assertIn("References:", c001["final_answer"])

    def test_overwrite_refusal_exit3_keeps_file(self):
        self.out.write_bytes(b"sentinel evidence")
        with contextlib.redirect_stdout(io.StringIO()):
            code = ea.main(["--mode", "unit", "--out", str(self.out)])
        self.assertEqual(code, 3)
        self.assertEqual(self.out.read_bytes(), b"sentinel evidence")

    def test_invalid_dataset_exit1(self):
        def mutate(ds):
            ds["cases"].append(dict(ds["cases"][0]))
        gold = self._write_gold("bad.json", mutate)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = ea.main(["--mode", "unit", "--gold", str(gold),
                            "--out", str(self.out)])
        self.assertEqual(code, 1)
        self.assertIn("DATASET INVALID", buf.getvalue())
        self.assertFalse(self.out.exists())

    def test_failed_case_exit4(self):
        def mutate(ds):
            case_by_id(ds, "C003")["expect"]["abstain"] = True
        gold = self._write_gold("flip.json", mutate)
        with contextlib.redirect_stdout(io.StringIO()):
            code = ea.main(["--mode", "unit", "--gold", str(gold),
                            "--out", str(self.out)])
        self.assertEqual(code, 4)
        rec = json.loads(self.out.read_text(encoding="utf-8"))
        c003 = next(c for c in rec["cases"] if c["id"] == "C003")
        self.assertEqual(c003["status"], "fail")
        self.assertIn("abstention", c003["failed_checks"])

    def test_local_model_mode_refused_without_opt_in(self):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = ea.main(["--mode", "local-model", "--out", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("opt-in", buf.getvalue())
        self.assertFalse(self.out.exists())

    def test_local_model_mode_refused_for_remote_provider(self):
        with mock.patch.object(ea, "LLM_PROVIDER", "gemini"):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                code = ea.main(["--mode", "local-model", "--enable-live",
                                "--out", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("never falls back", buf.getvalue())
        self.assertFalse(self.out.exists())

    def test_deterministic_rerun(self):
        out2 = Path(self._tmp.name) / "out2.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ea.main(["--mode", "unit", "--out",
                                      str(self.out)]), 0)
            self.assertEqual(ea.main(["--mode", "unit", "--out",
                                      str(out2)]), 0)
        rec1 = ea.strip_volatile(json.loads(self.out.read_text(encoding="utf-8")))
        rec2 = ea.strip_volatile(json.loads(out2.read_text(encoding="utf-8")))
        self.assertEqual(json.dumps(rec1, sort_keys=True),
                         json.dumps(rec2, sort_keys=True))

    def test_no_model_patch_leaks_after_run(self):
        import app.agents.books_agent as books
        engine_before = books._get_query_engine
        llm_before = books.get_llm
        with contextlib.redirect_stdout(io.StringIO()):
            ea.main(["--mode", "unit", "--out", str(self.out)])
        self.assertIs(books._get_query_engine, engine_before)
        self.assertIs(books.get_llm, llm_before)


# --------------------------------------------------------------------------
# gold-v3 dataset (C101-C103 resolution record; benchmarks/rag_gold_v3.json)
# --------------------------------------------------------------------------
GOLD3_SHA256 = "55aabf8f0a95eddf0527f74d687c0707c55adc28fc926702abdc34bcfa59d62a"


class GoldV3DatasetTests(unittest.TestCase):
    """Focused checks for the v3 resolution record.

    v2 keeps its own sha pin above; v3 is the single post-baseline
    resolution file: C102 promoted against the new deterministic
    fixture_multipage_table.pdf, C101/C103 unresolved with verified-final
    findings. The additive guarantee is asserted mechanically.
    """

    @classmethod
    def setUpClass(cls):
        cls.ds, cls.sha = ea.load_gold_v2(
            PROJECT_ROOT / "benchmarks" / "rag_gold_v3.json")
        files = {e["file_name"]
                 for c in cls.ds["cases"]
                 for e in (c.get("evidence") or [])
                 if e.get("source") in ("corpus", "fixture") and e.get("file_name")}
        cls.source_texts = ea.load_source_texts(files)

    def test_v3_sha_pinned(self):
        self.assertEqual(self.sha, GOLD3_SHA256)

    def test_v3_validates(self):
        self.assertEqual(ea.validate_dataset(self.ds, self.source_texts), [])

    def test_c102_executable_with_new_fixture(self):
        c = case_by_id(self.ds, "C102")
        self.assertEqual(c["status"], "executable")
        self.assertEqual(c["gold_documents"], ["fixture_multipage_table.pdf"])
        self.assertEqual(c["gold_pages"], [2])
        self.assertIn("2.84", c["required_facts"])
        self.assertIn("2.71 metres", c["expect"]["forbidden_facts"])
        self.assertEqual(c["modes"], ["unit", "integration"])
        # Evidence text must come from the ACTUAL page-2 extraction of the
        # fixture on disk (verbatim substring, whitespace-normalized).
        page2 = self.source_texts["fixture_multipage_table.pdf"]["pages"]["2"]
        self.assertIn(ea.norm_ws(c["evidence"][0]["text"]), page2)
        self.assertIn("The tide-log register records", page2)

    def test_unresolved_cases_carry_verified_final_findings(self):
        ids = sorted(c["id"] for c in self.ds["cases"]
                     if c["status"] == "unresolved")
        self.assertEqual(ids, ["C101", "C103"])
        for cid in ids:
            c = case_by_id(self.ds, cid)
            self.assertTrue(c.get("unresolved_reason"))
            findings = c.get("resolution_findings") or ""
            self.assertIn("VERIFIED-FINAL", findings)
            self.assertNotIn("expect", c)

    def test_additive_only_diff_against_v2(self):
        ds2, _ = load_gold()
        by2 = {c["id"]: c for c in ds2["cases"]}
        by3 = {c["id"]: c for c in self.ds["cases"]}
        self.assertEqual(set(by2), set(by3))
        changed = [k for k in by3 if by2[k] != by3[k]]
        self.assertEqual(sorted(changed), ["C101", "C102", "C103"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
