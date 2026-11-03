"""Evidence snapshots: every consulted passage, from BOTH routes (deterministic).

Regression cover for the 2026-11-02 3-of-4 truncation: `passage_details` used
ONE cumulative character budget for all passages, so once the first three had
used it up the fourth was silently `continue`d — a snapshot that claimed a
passage was never consulted. The same writer now records the books route.

No embeddings, no LLM, no network, no vector store: fixed passages and a faked
query-engine response.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import evidence, evidence_store  # noqa: E402
from app.agents.books_agent import books_agent  # noqa: E402
from app.config import EVIDENCE_PASSAGE_CHARS, NOT_FOUND_PHRASE  # noqa: E402
from app.execution.runner import snapshot_evidence  # noqa: E402

GUIDE = "urban_beekeeping_field_guide.pdf"
PROPOLIS_PASSAGE = (
    "Chapter 3: Hive Products. Propolis is a resinous substance bees collect "
    "from plant buds and use to seal small gaps and sterilize the interior "
    "of the hive. Honey is harvested only when its moisture content drops "
    "below 18 percent, which prevents fermentation during storage."
)
SUPPORTED_ANSWER = (
    "Propolis is a resinous substance bees collect from plant buds and use to "
    "seal small gaps and sterilize the hive interior; honey is harvested "
    "below 18 percent moisture."
)


def passage(text, score=0.8, name=GUIDE, page="3", node_id=None):
    return SimpleNamespace(
        text=text,
        metadata={"file_name": name, "page_label": page,
                  "last_modified_date": "2026-10-01", "file_size": 1024},
        score=score,
        node_id=node_id or f"chunk-{abs(hash(text)) % 100000}",
    )


class _FakeResponse:
    def __init__(self, text, nodes):
        self._text = text
        self.source_nodes = nodes

    def __str__(self):
        return self._text


def long_passage(index, chars=2000):
    """A passage that pushes a shared 6000-char budget past its limit."""
    body = ("Redis keeps its dataset in memory and uses it as a cache. "
            f"Note {index}. ")
    return body + ("filler detail " * ((chars // len("filler detail ")) + 1))


class TestPassageDetailsCompleteness(unittest.TestCase):
    def test_four_retrieved_passages_persist_four(self):
        # The live defect: 4 passages of ~2 KB exhausted the shared 6000-char
        # budget during the 4th, which was then dropped entirely.
        nodes = [passage(long_passage(i), name=f"doc{i}.pdf", page=str(i))
                 for i in range(4)]
        details = evidence.passage_details(nodes)
        self.assertEqual(len(details), 4)
        self.assertEqual([row["rank"] for row in details], [1, 2, 3, 4])
        for row in details:
            self.assertTrue(row["text"].strip())
            self.assertFalse(row["truncated"])
        self.assertGreater(sum(row["chars"] for row in details),
                           EVIDENCE_PASSAGE_CHARS,
                           "fixture must exceed the old shared budget")

    def test_more_passages_than_the_prompt_budget_are_all_kept(self):
        nodes = [passage(long_passage(i), name=f"doc{i}.pdf") for i in range(6)]
        block, supplied = evidence.prompt_passages(nodes)
        self.assertLess(len(supplied), len(nodes),
                        "the prompt budget must still bound the prompt")
        details = evidence.passage_details(nodes, supplied=supplied)
        self.assertEqual(len(details), 6)
        self.assertEqual([row["supplied"] for row in details],
                         [index in supplied for index in range(6)])
        self.assertEqual(len(block), len(block))  # non-empty prompt block
        self.assertTrue(block)

    def test_long_passage_is_truncated_and_flagged_not_dropped(self):
        nodes = [passage("A" * 50, name="a.pdf"),
                 passage("B" * 50, name="b.pdf"),
                 passage("C" * 50, name="c.pdf")]
        details = evidence.passage_details(nodes, limit=20)
        self.assertEqual(len(details), 3)
        self.assertTrue(all(row["truncated"] for row in details))
        self.assertEqual([len(row["text"]) for row in details], [20, 20, 20])

    def test_every_row_carries_citation_safe_identity(self):
        details = evidence.passage_details([passage(PROPOLIS_PASSAGE)])
        row = details[0]
        self.assertEqual(row["file_name"], GUIDE)
        self.assertEqual(row["page"], "3")
        self.assertTrue(row["chunk_id"])
        self.assertEqual(row["document_modified"], "2026-10-01")
        self.assertEqual(row["score"], 0.8)
        self.assertNotIn("file_path", row)


class TestSnapshotPayload(unittest.TestCase):
    def payload(self, nodes, **kwargs):
        details = kwargs.pop("details", None)
        if details is None:
            details = evidence.passage_details(nodes)
        return evidence_store.build(
            "exec1", question="What does the guide say about propolis?",
            route=kwargs.pop("route", "general"), answer="answer",
            sources=["evidence:guide.pdf, page 3"], qa_flags=[],
            evidence={"status": "grounded", "answered_from": "evidence",
                      "passages_detail": details,
                      "claim_checks": {"claims": [{"status": "supported"}],
                                       "summary": {"supported": 1}},
                      **kwargs})

    def test_complete_snapshot_counts_retrieved_and_supplied(self):
        nodes = [passage(long_passage(i), name=f"doc{i}.pdf") for i in range(4)]
        block, supplied = evidence.prompt_passages(nodes[:2])
        details = evidence.passage_details(nodes, supplied=supplied)
        snapshot = self.payload(nodes, details=details, passages=4)
        ev = snapshot["evidence"]
        self.assertEqual(ev["passages_retrieved"], 4)
        self.assertEqual(ev["passages_supplied"], len(supplied))
        self.assertTrue(ev["complete"])
        self.assertTrue(ev["storage_complete"])
        self.assertTrue(ev["content_complete"])
        self.assertIsNone(ev["partial_reason"])
        # claim verdicts survive verbatim
        self.assertEqual(ev["claims"], [{"status": "supported"}])

    def test_incomplete_persistence_is_explicitly_partial(self):
        details = evidence.passage_details(
            [passage("A" * 50, name="a.pdf"), passage("B" * 50, name="b.pdf")],
            limit=10)
        snapshot = self.payload(None, details=details)
        ev = snapshot["evidence"]
        # A cut passage is CONTENT-incomplete but STORAGE-complete: every row
        # was written, so `complete: false` never hides a missing record.
        self.assertFalse(ev["complete"])
        self.assertTrue(ev["storage_complete"])
        self.assertFalse(ev["content_complete"])
        self.assertIn("passage 1", ev["partial_reason"])
        self.assertIn("passage 2", ev["partial_reason"])
        self.assertIn("truncated", ev["partial_reason"])
        # nothing was dropped: both rows are still there
        self.assertEqual(len(ev["passages"]), 2)

    def test_a_missing_row_is_storage_incomplete_and_named(self):
        # The retriever returned 4 passages but only 2 rows exist: that is a
        # lost record, not a truncated one — reported separately.
        details = evidence.passage_details(
            [passage("A" * 40, name="a.pdf"), passage("B" * 40, name="b.pdf")])
        snapshot = self.payload(None, details=details, passages=4)
        ev = snapshot["evidence"]
        self.assertFalse(ev["complete"])
        self.assertFalse(ev["storage_complete"])
        self.assertIn("no stored record", ev["partial_reason"])
        self.assertIn("2 of 4", ev["partial_reason"])

    def test_books_route_snapshot_uses_the_same_writer(self):
        nodes = [passage(PROPOLIS_PASSAGE, name=GUIDE)]
        record = {"status": "grounded", "answered_from": "books",
                  "best_score": 0.71, "passages_detail":
                  evidence.passage_details(nodes, supplied=[0])}
        snapshot = self.payload(None, details=record["passages_detail"],
                                route="books", **{})
        self.assertEqual(snapshot["route"], "books")
        self.assertEqual(len(snapshot["evidence"]["passages"]), 1)
        self.assertTrue(snapshot["evidence"]["passages"][0]["supplied"])

    def test_snapshot_survives_a_process_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(evidence_store, "SNAPSHOT_DIR", Path(tmp)):
                snapshot = self.payload([passage(PROPOLIS_PASSAGE)])
                path = evidence_store.write("restart1", snapshot)
                self.assertIsNotNone(path)
                read_back = evidence_store.read("restart1")
            self.assertEqual(read_back["evidence"]["passages_retrieved"], 1)
            # A separate process = a real restart: no in-memory state survives.
            code = (
                "import json,sys;"
                f"d=json.load(open(r'{path}', encoding='utf-8'));"
                "print(d['evidence']['passages'][0]['file_name']);"
                "print(d['evidence']['complete'])"
            )
            out = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn(GUIDE, out.stdout)
            self.assertIn("True", out.stdout)


class TestBooksRouteEvidence(unittest.TestCase):
    def setUp(self):
        import app.agents.books_agent as ba

        self._orig_get_query_engine = ba._get_query_engine

    def _run_agent(self, response, question="What does the book say about propolis?"):
        import app.agents.books_agent as ba

        ba._get_query_engine = lambda: SimpleNamespace(query=lambda q: response)
        try:
            return books_agent({"question": question})
        finally:
            ba._get_query_engine = self._orig_get_query_engine

    def test_answered_books_run_records_every_consulted_passage(self):
        nodes = [passage(PROPOLIS_PASSAGE, score=0.71, name=GUIDE, page="1"),
                 passage("Honey is harvested when moisture drops below 18 percent.",
                         score=0.66, name=GUIDE, page="2"),
                 passage("The queen lays up to 2,000 eggs per day in spring.",
                         score=0.51, name=GUIDE, page="4"),
                 passage("Drones are produced in summer and die after mating.",
                         score=0.40, name=GUIDE, page="5")]
        out = self._run_agent(_FakeResponse(SUPPORTED_ANSWER, nodes))

        self.assertIn("books_evidence", out)
        record = out["books_evidence"]
        self.assertEqual(record["answered_from"], "books")
        self.assertEqual(record["status"], "grounded")
        self.assertEqual(len(record["passages_detail"]), 4)
        self.assertTrue(all(row["supplied"] for row in record["passages_detail"]))
        self.assertEqual(
            [row["chunk_id"] for row in record["passages_detail"]],
            [node.node_id for node in nodes])
        # The user-facing answer and provenance are unchanged by the snapshot.
        self.assertEqual(out["sources"], ["books"])
        self.assertIn(f"References: {GUIDE}", out["answer"])

        snapshot = evidence_store.build(
            "books1", question="What does the book say about propolis?",
            route="books", answer=out["answer"], sources=out["sources"],
            qa_flags=[], evidence=record)
        self.assertEqual(snapshot["route"], "books")
        self.assertEqual(snapshot["evidence"]["passages_retrieved"], 4)
        self.assertTrue(snapshot["evidence"]["complete"])

    def test_refusal_after_retrieval_records_what_it_did_not_answer_from(self):
        out = self._run_agent(_FakeResponse(NOT_FOUND_PHRASE,
                                            [passage(PROPOLIS_PASSAGE, score=0.2)]))
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE))
        self.assertNotIn("References:", out["answer"])
        self.assertEqual(out["books_evidence"]["status"], "none")
        self.assertIn("refused", out["books_evidence"]["reason"])
        self.assertEqual(len(out["books_evidence"]["passages_detail"]), 1)

    def test_no_retrieved_passages_writes_no_snapshot(self):
        out = self._run_agent(_FakeResponse(NOT_FOUND_PHRASE, []))
        self.assertTrue(out["answer"].startswith(NOT_FOUND_PHRASE))
        self.assertNotIn("books_evidence", out)
        self.assertIsNone(snapshot_evidence(out))


class TestRunnerSnapshotSelection(unittest.TestCase):
    def test_general_record_is_preferred_when_both_exist(self):
        state = {"evidence": {"passages_detail": [{"rank": 1}]},
                 "books_evidence": {"passages_detail": [{"rank": 1}]}}
        self.assertIs(snapshot_evidence(state), state["evidence"])

    def test_books_record_is_used_when_there_is_no_general_record(self):
        state = {"books_evidence": {"passages_detail": [{"rank": 1}]}}
        self.assertIs(snapshot_evidence(state), state["books_evidence"])

    def test_no_record_or_empty_record_selects_nothing(self):
        self.assertIsNone(snapshot_evidence({}))
        self.assertIsNone(snapshot_evidence(None))
        self.assertIsNone(snapshot_evidence({"evidence": {"status": "none"}}))
        self.assertIsNone(snapshot_evidence(
            {"books_evidence": {"passages_detail": []}}))


if __name__ == "__main__":
    unittest.main()
