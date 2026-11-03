"""Reranking, claim-level verification and evidence provenance (deterministic).

No model, no network, no vector store: fixed passages, a scripted provider
double, and a fake graph where the runner hook is exercised. Every assertion
is about what the pipeline can PROVE, not about whether a fact is true in the
world (docs/RETRIEVAL_AND_CLAIM_VERIFICATION.md).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import claim_check, evidence, evidence_store  # noqa: E402
from app.agents.books_grounding import (  # noqa: E402
    _grounded_repair,
    _reference_footer,
    _strip_model_citations,
)

REDIS_BSD = "Redis is distributed under the BSD-3 licence and is written in C."
REDIS_MEM = "Redis keeps its entire dataset in memory and is used as a cache."
MEMCACHED = "Memcached is distributed under the Apache 2.0 licence."
BENCH = "The cache reduces database load by 40% in the vendor's tests."


def passage(text, score=0.8, name="guide.pdf", page="3", node_id=None):
    return SimpleNamespace(
        text=text,
        metadata={"file_name": name, "page_label": page,
                  "last_modified_date": "2026-10-01", "file_size": 1024},
        score=score,
        node_id=node_id or f"chunk-{abs(hash(text)) % 10000}",
    )


class TestReranking(unittest.TestCase):
    def test_covering_passage_outranks_a_higher_scoring_topic_match(self):
        topical = passage(REDIS_MEM, score=0.92)
        answering = passage(REDIS_BSD, score=0.80)
        result = evidence.select(
            "What licence is Redis distributed under?", [topical, answering])
        self.assertEqual(result["nodes"][0].text, REDIS_BSD)
        self.assertGreater(result["ranking"][0]["coverage"], 0.9)

    def test_near_duplicate_passages_are_collapsed(self):
        first = passage(REDIS_MEM, score=0.9, page="3")
        duplicate = passage(REDIS_MEM, score=0.88, page="3")
        result = evidence.select("What does Redis keep in memory?",
                                 [first, duplicate])
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(result["dropped_duplicates"], 1)

    def test_multi_document_question_keeps_complementary_passages(self):
        licence = passage(REDIS_BSD, score=0.82, name="licensing.pdf")
        memory = passage(REDIS_MEM, score=0.81, name="architecture.pdf")
        result = evidence.select(
            "What licence is Redis distributed under and what does it keep "
            "in memory?", [licence, memory])
        self.assertEqual(len(result["nodes"]), 2)
        names = {row["ref"] for row in result["ranking"]}
        self.assertIn("licensing.pdf, page 3", names)
        self.assertIn("architecture.pdf, page 3", names)

    def test_exact_technical_term_breaks_a_near_tie(self):
        generic = passage("The policy describes licensing for released "
                          "software in general terms.", score=0.80)
        exact = passage("The 2024 policy section 7 states the redistribution "
                        "terms.", score=0.80)
        result = evidence.select(
            "What does the 2024 policy say about redistribution?", [generic, exact])
        self.assertEqual(result["nodes"][0].text, exact.text)
        self.assertGreater(result["ranking"][0]["exact"], 0.0)

    def test_selection_never_invents_or_drops_everything(self):
        only = passage(REDIS_BSD)
        result = evidence.select("What licence is Redis distributed under?",
                                 [only])
        self.assertEqual(result["nodes"], [only])
        empty = evidence.select("anything", [])
        self.assertEqual(empty["nodes"], [])
        self.assertEqual(empty["dropped_duplicates"], 0)

    def test_top_k_is_respected(self):
        nodes = [passage(f"Redis note {i} about memory usage and caching.",
                         score=0.5 + i / 100, name=f"n{i}.pdf")
                 for i in range(6)]
        result = evidence.select("What does Redis keep in memory?", nodes,
                                 top_k=2)
        self.assertEqual(len(result["nodes"]), 2)

    def test_ranking_is_an_audit_trail_not_a_support_claim(self):
        result = evidence.select("What licence is Redis distributed under?",
                                 [passage(REDIS_BSD), passage(REDIS_MEM, score=0.9)])
        for row in result["ranking"]:
            self.assertEqual(set(row), {"ref", "similarity", "rerank",
                                        "coverage", "exact", "position"})


class TestClaimVerification(unittest.TestCase):
    def test_claim_stated_by_the_cited_passage_is_supported(self):
        claim = claim_check.verify_sentence(
            "Redis is distributed under the BSD-3 licence.", [passage(REDIS_BSD)])
        self.assertEqual(claim["status"], claim_check.SUPPORTED)
        self.assertIn("guide.pdf", claim["evidence"][0])

    def test_claim_only_partly_covered_is_partial(self):
        claim = claim_check.verify_sentence(
            "Redis is distributed under the BSD-3 licence and needs 8 GB.",
            [passage(REDIS_BSD)])
        self.assertEqual(claim["status"], claim_check.UNSUPPORTED)
        self.assertIn("8", claim["missing"])

    def test_topically_related_but_unsupported_claim(self):
        claim = claim_check.verify_sentence(
            "Redis is 10x faster than Memcached.", [passage(REDIS_MEM)])
        self.assertEqual(claim["status"], claim_check.UNSUPPORTED)
        self.assertIn("10", claim["missing"])

    def test_contradictory_polarity_is_contradicted(self):
        claim = claim_check.verify_sentence(
            "Redis supports clustering.",
            [passage("Redis does not support clustering in this version.")])
        self.assertEqual(claim["status"], claim_check.CONTRADICTED)
        self.assertIn("opposite", claim["reason"])

    def test_licence_swapped_for_the_same_subject_is_contradicted(self):
        claim = claim_check.verify_sentence(
            "Redis is distributed under the MIT licence.", [passage(REDIS_BSD)])
        self.assertEqual(claim["status"], claim_check.CONTRADICTED)
        self.assertIn("BSD", claim["reason"])

    def test_one_supporting_passage_beats_a_conflicting_one(self):
        """Two documents disagreeing is a DISCLOSED conflict, not proof the
        answer is wrong — a claim corroborated by one is not contradicted
        (regression: E5 in the answer benchmark)."""
        nodes = [passage(REDIS_BSD, name="vendor_a.pdf", page="2"),
                 passage("Redis is licensed under the MIT licence.",
                         name="vendor_b.pdf", page="5")]
        claim = claim_check.verify_sentence(
            "Redis is distributed under the BSD-3 licence.", nodes)
        self.assertEqual(claim["status"], claim_check.SUPPORTED)

    def test_fabricated_citation_is_unsupported(self):
        claim = claim_check.verify_sentence(
            "Redis is distributed under the BSD-3 licence "
            "(internal_review.pdf, page 9).", [passage(REDIS_BSD)])
        self.assertEqual(claim["status"], claim_check.UNSUPPORTED)
        self.assertIn("not retrieved", claim["reason"])

    def test_citation_to_the_wrong_document_is_unsupported(self):
        nodes = [passage(REDIS_BSD, name="licensing.pdf", page="2"),
                 passage(REDIS_MEM, name="architecture.pdf", page="5")]
        claim = claim_check.verify_sentence(
            "Redis is distributed under the BSD-3 licence "
            "(architecture.pdf, page 5).", nodes)
        self.assertEqual(claim["status"], claim_check.UNSUPPORTED)
        self.assertEqual(claim["evidence"], ["architecture.pdf, page 5"])

    def test_unsupported_percentage(self):
        claim = claim_check.verify_sentence(
            "The cache reduces database load by 30-50%.", [passage(BENCH)])
        self.assertEqual(claim["status"], claim_check.UNSUPPORTED)
        self.assertIn("50", claim["missing"])

    def test_number_stated_by_the_passage_is_supported(self):
        claim = claim_check.verify_sentence(
            "The cache reduces database load by 40%.", [passage(BENCH)])
        self.assertEqual(claim["status"], claim_check.SUPPORTED)

    def test_two_supported_facts_with_an_unsupported_conclusion(self):
        claim = claim_check.verify_sentence(
            "Redis is in memory and therefore twice as fast as Memcached.",
            [passage(REDIS_MEM)])
        self.assertEqual(claim["status"], claim_check.UNSUPPORTED)

    def test_no_passages_is_unable_not_supported(self):
        result = claim_check.verify("Redis is BSD-3 licensed.", [])
        self.assertTrue(result["claims"])
        self.assertEqual(result["claims"][0]["status"], claim_check.UNABLE)
        self.assertFalse(result["checkable"])

    def test_disclosures_and_footers_are_not_claims(self):
        answer = ("Redis is distributed under the BSD-3 licence.\n\n"
                  "Not stated by the passages: vram.\n\n"
                  "References: guide.pdf, page 3")
        result = claim_check.verify(answer, [passage(REDIS_BSD)])
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["status"], claim_check.SUPPORTED)

    def test_summary_counts_every_status(self):
        answer = ("Redis is distributed under the BSD-3 licence. "
                  "Redis is 10x faster than Memcached.")
        result = claim_check.verify(answer, [passage(REDIS_BSD)])
        self.assertEqual(result["summary"][claim_check.SUPPORTED], 1)
        self.assertEqual(result["summary"][claim_check.UNSUPPORTED], 1)


class TestClaimVerificationInQa(unittest.TestCase):
    def state(self, ev, answer):
        return {"question": "What licence is Redis distributed under?",
                "route": "general", "answer": answer,
                "sources": ev.get("used") or [], "qa_flags": [],
                "agent_answers": {"general": answer}, "evidence": ev}

    def test_unsupported_claim_is_reported_with_its_reason(self):
        from app.qa_node import run_qa

        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 1, "files": ["guide.pdf"], "text": REDIS_BSD,
              "used": ["evidence:guide.pdf, page 3"], "grounding": "ok",
              "claim_checks": claim_check.verify(
                  "Redis is distributed under the BSD-3 licence. "
                  "Redis is 10x faster than Memcached.",
                  [passage(REDIS_BSD)])}
        flags = run_qa(self.state(ev, "Redis is distributed under the BSD-3 "
                                     "licence. Redis is 10x faster than Memcached."))
        joined = " | ".join(flags)
        self.assertIn("not supported by the passage they cite", joined)
        self.assertIn("10x", joined)

    def test_all_supported_claims_produce_no_claim_flag(self):
        from app.qa_node import run_qa

        answer = "Redis is distributed under the BSD-3 licence."
        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 1, "files": ["guide.pdf"], "text": REDIS_BSD,
              "used": ["evidence:guide.pdf, page 3"], "grounding": "ok",
              "claim_checks": claim_check.verify(answer, [passage(REDIS_BSD)])}
        flags = run_qa(self.state(ev, answer))
        self.assertFalse([f for f in flags if "passage they cite" in f], flags)


class TestEvidenceSnapshots(unittest.TestCase):
    def setUp(self):
        self._td = Path(tempfile.mkdtemp())
        self._patch = mock.patch.object(evidence_store, "SNAPSHOT_DIR",
                                        self._td / "snapshots")
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def payload(self):
        nodes = [passage(REDIS_BSD)]
        picked = evidence.select("What licence is Redis distributed under?", nodes)
        claims = claim_check.verify(REDIS_BSD, nodes)
        return evidence_store.build(
            "exec1", question="What licence is Redis distributed under?",
            route="general", answer=REDIS_BSD,
            sources=["evidence:guide.pdf, page 3"], qa_flags=[],
            evidence={"status": "grounded", "answered_from": "evidence",
                      "passages_detail": evidence.passage_details(
                          nodes, picked["ranking"]),
                      "ranking": picked["ranking"], "claim_checks": claims})

    def test_snapshot_roundtrip_and_contents(self):
        data = self.payload()
        path = evidence_store.write("exec1", data)
        self.assertIsNotNone(path)
        loaded = evidence_store.read("exec1")
        self.assertEqual(loaded["execution_id"], "exec1")
        self.assertEqual(loaded["evidence"]["status"], "grounded")
        self.assertTrue(loaded["evidence"]["passages"][0]["text"])
        self.assertEqual(loaded["evidence"]["claims"][0]["status"], "supported")
        self.assertIn("manifest_sha256", loaded["index"])

    def test_snapshot_never_contains_a_local_path(self):
        evidence_store.write("exec2", self.payload())
        raw = (self._td / "snapshots" / "exec2.json").read_text(encoding="utf-8")
        self.assertNotIn("file_path", raw)
        self.assertNotIn("C:\\\\", raw)
        self.assertNotIn("Users", raw)

    def test_unknown_or_unsafe_ids_return_none(self):
        self.assertIsNone(evidence_store.read("missing"))
        self.assertIsNone(evidence_store.write("../escape", self.payload()))
        self.assertIsNone(evidence_store.read("../escape"))

    def test_runner_writes_the_snapshot_for_an_evidence_run(self):
        from app.execution.runner import GraphRunner
        from app.execution.store import ExecutionStore

        store = ExecutionStore(self._td / "snap.sqlite3")
        runner = GraphRunner(store)
        store.create_execution("run1", "What licence is Redis distributed under?")
        fake_state = {
            "answer": REDIS_BSD, "route": "general", "qa_flags": [],
            "sources": ["evidence:guide.pdf, page 3"],
            "agent_answers": {"general": REDIS_BSD},
            "evidence": {
                "status": "grounded", "answered_from": "evidence",
                "passages_detail": [{"text": REDIS_BSD,
                                     "file_name": "guide.pdf", "page": "3"}],
                "claim_checks": {"claims": [], "summary": {}},
            },
        }
        graph = mock.Mock()
        graph.invoke.return_value = fake_state
        with mock.patch("app.graph.build_graph", return_value=graph):
            runner.run("run1")
        loaded = evidence_store.read("run1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["answer"], REDIS_BSD)
        store.close()

    def test_runner_records_no_snapshot_without_evidence(self):
        from app.execution.runner import GraphRunner
        from app.execution.store import ExecutionStore

        store = ExecutionStore(self._td / "snap2.sqlite3")
        runner = GraphRunner(store)
        store.create_execution("run2", "hi")
        graph = mock.Mock()
        graph.invoke.return_value = {"answer": "hello", "route": "conversational",
                                    "qa_flags": [], "sources": []}
        with mock.patch("app.graph.build_graph", return_value=graph):
            runner.run("run2")
        self.assertIsNone(evidence_store.read("run2"))
        store.close()


class TestEvidenceApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        import app.api.server as server
        from app.execution.store import ExecutionStore

        self._td = Path(tempfile.mkdtemp())
        self.server = server
        self.store = ExecutionStore(self._td / "api.sqlite3")
        server._store = self.store
        self.client = TestClient(server.build_server())
        self.store.create_execution("api1", "q")
        self._patch = mock.patch.object(evidence_store, "SNAPSHOT_DIR",
                                        self._td / "snapshots")
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.store.close()

    def test_missing_execution_is_404(self):
        self.assertEqual(
            self.client.get("/api/v1/executions/nope/evidence").status_code, 404)

    def test_execution_without_snapshot_reports_null_evidence(self):
        response = self.client.get("/api/v1/executions/api1/evidence")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body["evidence"])
        self.assertIn("no evidence snapshot", body["reason"])

    def test_snapshot_is_returned_read_only(self):
        evidence_store.write("api1", {
            "execution_id": "api1", "route": "general",
            "evidence": {"status": "grounded", "passages": []},
        })
        body = self.client.get("/api/v1/executions/api1/evidence").json()
        self.assertEqual(body["evidence"]["status"], "grounded")
        self.assertEqual(body["route"], "general")


class TestGroundedFlowUsesTheSeam(unittest.TestCase):
    def test_general_node_records_ranking_and_claim_verdicts(self):
        from app.graph import general_node

        class Model:
            def invoke(self, prompt):
                return SimpleNamespace(
                    content="Redis is distributed under the BSD-3 licence.")

        state = {"question": "What licence is Redis distributed under?",
                 "route": "general", "sources": [], "qa_flags": [],
                 "agent_answers": {}}
        nodes = [passage(REDIS_BSD)]
        with patch.object(evidence, "retrieve", return_value=nodes), \
                patch("app.llm_client.get_llm", return_value=Model()):
            update = general_node(state)
        ev = update["evidence"]
        self.assertEqual(ev["answered_from"], "evidence")
        self.assertTrue(ev["ranking"])
        self.assertEqual(ev["claim_checks"]["claims"][0]["status"],
                         "supported")
        self.assertTrue(ev["passages_detail"][0]["text"])


if __name__ == "__main__":
    unittest.main()
