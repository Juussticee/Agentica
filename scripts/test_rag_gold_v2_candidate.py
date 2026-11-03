"""Candidate evaluation-set validator (offline, deterministic, zero quota).

Validates ``benchmarks/rag_gold_v2_candidate.json`` WITHOUT touching frozen
artifacts, production storage, Ollama, or the network:

  1. required top-level fields + explicit draft status (no execution claims);
  2. additive-relationship wording (never replaces/re-grades gold v1);
  3. per-case schema completeness (v1-compatible field set);
  4. category coverage: every label in categories_targeted appears on cases;
  5. answerable cases carry required_facts, unanswerable carry none;
  6. pending cases declare blocked_on and are section pending_synthetic;
  7. real_corpus cases derive from v1 provenance (derived_from present).

    env\\Scripts\\python.exe -m pytest scripts/test_rag_gold_v2_candidate.py -q
"""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CANDIDATE = PROJECT_ROOT / "benchmarks" / "rag_gold_v2_candidate.json"

REQUIRED_TOP = {"version", "status", "created", "owner_task",
                "relationship_to_v1", "immutability_rule", "sections",
                "categories_targeted", "cases"}
REQUIRED_CASE = {"id", "section", "synthetic", "query", "answerable",
                 "gold_documents", "gold_pages", "required_facts",
                 "citation_expectation", "labels", "provenance", "notes"}


class CandidateSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CANDIDATE, encoding="utf-8") as fh:
            cls.data = json.load(fh)
        cls.cases = cls.data["cases"]

    def test_top_level_fields_and_draft_status(self):
        self.assertTrue(REQUIRED_TOP <= set(self.data))
        status = self.data["status"].lower()
        self.assertIn("draft", status)
        # Honest status: explicitly NOT executed / NOT a baseline (the
        # proposal says so; a claim of execution would drop the negation).
        self.assertIn("not executed", status)
        self.assertIn("not a baseline", status)

    def test_additive_relationship_never_replaces_v1(self):
        rel = self.data["relationship_to_v1"].lower()
        self.assertIn("additive", rel)
        self.assertIn("does not replace", rel)

    def test_case_schema_completeness(self):
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)), "duplicate case ids")
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(REQUIRED_CASE <= set(case),
                                REQUIRED_CASE - set(case))
                self.assertIsInstance(case["labels"], list)
                self.assertTrue(case["labels"])

    def test_category_coverage(self):
        seen = {label for c in self.cases for label in c["labels"]}
        missing = set(self.data["categories_targeted"]) - seen
        self.assertFalse(missing, f"targeted categories without a case: {missing}")

    def test_answerability_contract(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                if case["section"] == "pending_synthetic":
                    # Blocked-on-fixture cases intentionally carry no facts
                    # and may name a not-yet-existing document.
                    continue
                if case["answerable"] is True:
                    self.assertTrue(case["required_facts"],
                                    "answerable case without required_facts")
                elif case["answerable"] is False:
                    self.assertEqual(case["required_facts"], [])
                    self.assertEqual(case["gold_documents"], [])

    def test_pending_cases_are_blocked_on_missing_fixtures(self):
        for case in self.cases:
            if case["section"] == "pending_synthetic":
                with self.subTest(case=case["id"]):
                    self.assertIn("blocked_on", case["provenance"])
                    self.assertTrue(case["synthetic"])
            else:
                with self.subTest(case=case["id"]):
                    self.assertIn("derived_from", case["provenance"],
                                  "real-corpus case without v1 provenance")

    def test_no_fixture_files_claimed_exist(self):
        # Pending cases must not pretend their fixture exists on disk — in
        # EITHER direction: a claimed-absent fixture must be absent, and a
        # documented blocker removal (provenance.blocker_removed, e.g. the
        # C102 resolution realized in rag_gold_v3.json) must NOT point at a
        # still-missing file. A stale note in either direction fails here.
        for case in self.cases:
            if case["section"] != "pending_synthetic":
                continue
            blocker_removed = case["provenance"].get("blocker_removed")
            for doc in case["gold_documents"]:
                with self.subTest(case=case["id"], doc=doc):
                    exists = (PROJECT_ROOT / "benchmarks" / "rag_fixtures"
                              / doc).exists()
                    if blocker_removed:
                        self.assertTrue(
                            exists,
                            f"{doc} claimed removed-blocker but is missing — "
                            f"stale blocker_removed note {blocker_removed!r}")
                    else:
                        self.assertFalse(
                            exists,
                            f"{doc} exists — case can be un-blocked and "
                            f"reclassified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
