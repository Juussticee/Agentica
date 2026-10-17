"""RAG-1 evaluation-harness tests (deterministic, offline, zero quota).

Covers: metrics math, gold immutability/read-only enforcement, fixture
generator byte-determinism, guard behavior, and — mandatory safeguard #3 —
a deterministic ISOLATION PROOF that the evaluation constructs Chroma
clients only on temporary paths, never on the production storage/ path,
and that the real store files are untouched (stat + hash).

MockEmbedding returns a constant vector (verified in installed source), so
these tests need no Ollama, no network, and no chat model.

    env\\Scripts\\python.exe -m pytest scripts/test_rag_eval_harness.py -q
"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import eval_retrieval as ev  # noqa: E402
import make_rag_fixtures as mrf  # noqa: E402


class MetricsMathTest(unittest.TestCase):
    CASE = {
        "id": "T1", "answerable": True,
        "gold_documents": ["a.pdf"], "gold_pages": [2],
        "min_docs_covered": 1,
    }

    def _ranked(self, specs):
        return [{"file_name": f, "page_label": p} for f, p in specs]

    def test_hit_at_first_rank(self):
        ranked = self._ranked([("a.pdf", 2), ("b.pdf", 1), ("c.pdf", 3)])
        m = ev.compute_metrics(self.CASE, ranked)
        self.assertTrue(m["hit@1"])
        self.assertEqual(m["first_hit_rank"], 1)
        self.assertEqual(m["reciprocal_rank"], 1.0)
        self.assertEqual(m["ndcg@5"], 1.0)
        self.assertAlmostEqual(m["precision@1"], 1.0)
        self.assertEqual(m["precision@3"], round(1 / 3, 6))

    def test_hit_later_rank(self):
        ranked = self._ranked([("b.pdf", 1), ("a.pdf", 2), ("c.pdf", 3)])
        m = ev.compute_metrics(self.CASE, ranked)
        self.assertFalse(m["hit@1"])
        self.assertTrue(m["hit@3"])
        self.assertEqual(m["first_hit_rank"], 2)
        self.assertEqual(m["reciprocal_rank"], 0.5)
        # ndcg = 1/log2(1+2) = 1/log2(3)
        import math
        self.assertAlmostEqual(m["ndcg@5"], 1.0 / math.log2(3), places=5)

    def test_no_hit(self):
        ranked = self._ranked([("b.pdf", 1), ("c.pdf", 3)])
        m = ev.compute_metrics(self.CASE, ranked)
        self.assertFalse(m["hit@5"])
        self.assertEqual(m["reciprocal_rank"], 0.0)
        self.assertEqual(m["ndcg@5"], 0.0)

    def test_wrong_page_is_not_a_hit(self):
        ranked = self._ranked([("a.pdf", 9), ("a.pdf", 2)])
        m = ev.compute_metrics(self.CASE, ranked)
        self.assertEqual(m["first_hit_rank"], 2)

    def test_multi_doc_requires_both_documents(self):
        case = {"id": "T2", "answerable": True,
                "gold_documents": ["a.pdf", "b.docx"], "gold_pages": None,
                "min_docs_covered": 2}
        one = self._ranked([("a.pdf", None), ("x.pdf", 1)])
        both = self._ranked([("a.pdf", None), ("b.docx", None)])
        self.assertFalse(ev.compute_metrics(case, one)["hit@3"])
        self.assertTrue(ev.compute_metrics(case, both)["hit@3"])

    def test_aggregate_and_percentile(self):
        rows = [{"metrics": ev.compute_metrics(
            self.CASE, self._ranked([("a.pdf", 2)]))}]
        agg = ev.aggregate(rows)
        self.assertEqual(agg["cases"], 1)
        self.assertEqual(agg["hit@3"], 1.0)
        self.assertEqual(ev.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(ev.percentile([], 0.9), 0.0)

    def test_abstention_proxy(self):
        self.assertTrue(ev.abstention_proxy(None, True)["proxy_abstain"])
        self.assertTrue(ev.abstention_proxy(0.10, False)["correct"])
        self.assertFalse(ev.abstention_proxy(0.90, False)["correct"])
        self.assertFalse(ev.abstention_proxy(0.90, True)["proxy_abstain"])


class NodeCoversTest(unittest.TestCase):
    def test_file_only_case(self):
        case = {"gold_documents": ["a.docx"], "gold_pages": None}
        self.assertTrue(ev.node_covers({"file_name": "a.docx"}, case))
        self.assertFalse(ev.node_covers({"file_name": "b.docx"}, case))

    def test_page_match_and_junk(self):
        case = {"gold_documents": ["a.pdf"], "gold_pages": [2]}
        self.assertTrue(ev.node_covers({"file_name": "a.pdf",
                                        "page_label": "2"}, case))
        self.assertTrue(ev.node_covers({"file_name": "a.pdf",
                                        "page_label": 2}, case))
        self.assertFalse(ev.node_covers({"file_name": "a.pdf",
                                         "page_label": 9}, case))
        self.assertFalse(ev.node_covers({"file_name": "a.pdf",
                                         "page_label": "junk!!"}, case))
        self.assertFalse(ev.node_covers({}, case))
        self.assertFalse(ev.node_covers(None, case))

    # --- R011/S016 fix coverage (docs/RAG_BASELINE_AUDIT.md §1.5) ---------

    MULTI = {
        # gold_pages describes the PAGED member only; the docx member is
        # file-level evidence and must stay reachable (R011/S016 shape).
        "id": "T3", "answerable": True,
        "gold_documents": ["a.pdf", "b.docx"],
        "gold_pages": [1], "min_docs_covered": 2,
    }

    def test_pdf_node_on_allowed_page_covers(self):
        self.assertTrue(ev.node_covers({"file_name": "a.pdf",
                                        "page_label": "1"}, self.MULTI))
        self.assertTrue(ev.node_covers({"file_name": "a.pdf",
                                        "page_number": 1}, self.MULTI))

    def test_pdf_node_on_disallowed_page_rejected(self):
        self.assertFalse(ev.node_covers({"file_name": "a.pdf",
                                         "page_label": "2"}, self.MULTI))

    def test_pageless_docx_node_covers_in_multi_doc_gold_case(self):
        # key absent AND explicit None both mean "no page metadata"
        self.assertTrue(ev.node_covers({"file_name": "b.docx"}, self.MULTI))
        self.assertTrue(ev.node_covers({"file_name": "b.docx",
                                        "page_label": None}, self.MULTI))

    def test_missing_page_metadata_falls_back_to_file_match(self):
        case = {"gold_documents": ["a.pdf", "b.docx"], "gold_pages": [1]}
        self.assertTrue(ev.node_covers({"file_name": "b.docx"}, case))

    def test_malformed_page_metadata_is_still_a_miss(self):
        # present but unparseable: we cannot confirm the page, so it is
        # not evidence for a case that specifies pages
        self.assertFalse(ev.node_covers({"file_name": "a.pdf",
                                         "page_label": "junk!!"}, self.MULTI))

    def test_multi_doc_metric_counts_pageless_member(self):
        ranked = [{"file_name": "a.pdf", "page_label": "1"},
                  {"file_name": "b.docx"},
                  {"file_name": "z.pdf", "page_label": "7"}]
        m = ev.compute_metrics(self.MULTI, ranked)
        self.assertFalse(m["hit@1"])      # one slot cannot hold two docs
        self.assertTrue(m["hit@3"])       # docx at rank 2 now counts
        self.assertTrue(m["coverage"])
        self.assertEqual(m["first_hit_rank"], 1)


class GoldImmutabilityTest(unittest.TestCase):
    def test_gold_loads_and_hash_stable(self):
        data, digest = ev.load_gold(ev.GOLD_PATH)
        self.assertGreaterEqual(len(data["cases"]), 25)
        self.assertEqual(digest, ev.sha256_file(ev.GOLD_PATH))

    def test_synthetic_flag_consistent_with_section(self):
        data, _ = ev.load_gold(ev.GOLD_PATH)
        for case in data["cases"]:
            self.assertEqual(case["synthetic"],
                             case["section"] == "synthetic_fixtures",
                             case["id"])

    def test_gold_sources_exist_and_facts_present(self):
        data, _ = ev.load_gold(ev.GOLD_PATH)
        notes = ev.validate_gold_against_sources(data)
        self.assertEqual(notes, [], f"gold review notes: {notes}")

    def test_gold_rejects_inconsistent_case(self):
        bad = {"cases": [{"id": "X", "section": "real_corpus",
                          "synthetic": True, "query": "q", "answerable": True,
                          "gold_documents": [], "required_facts": [],
                          "citation_expectation": "refusal", "labels": [],
                          "provenance": "p"}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gold.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                ev.load_gold(p)


class FixtureGeneratorTest(unittest.TestCase):
    def test_builders_are_byte_deterministic(self):
        for name, builder in mrf.BUILDERS.items():
            self.assertEqual(builder(), builder(), name)

    def test_generated_files_on_disk_match_builders(self):
        for name, builder in mrf.BUILDERS.items():
            on_disk = (mrf.FIXTURE_DIR / name).read_bytes()
            self.assertEqual(on_disk, builder(),
                             f"{name} on disk differs from a fresh build — "
                             f"re-run scripts/make_rag_fixtures.py")

    def test_scan_fixture_page1_has_no_text_layer(self):
        from pypdf import PdfReader
        reader = PdfReader(str(mrf.FIXTURE_DIR / "fixture_scan.pdf"))
        self.assertEqual(len(reader.pages), 2)
        self.assertEqual((reader.pages[0].extract_text() or "").strip(), "")
        self.assertIn("eucalyptus", reader.pages[1].extract_text() or "")


class GuardTest(unittest.TestCase):
    def test_guard_detects_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "x.bin"
            f.write_bytes(b"hello")
            before = ev.guard_snapshot([f])
            f.write_bytes(b"changed!")
            after = ev.guard_snapshot([f])
            with self.assertRaises(ev.GuardFailure):
                ev.guard_assert_unchanged(before, after, "unit")

    def test_guard_passes_when_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "x.bin"
            f.write_bytes(b"stable")
            before = ev.guard_snapshot([f])
            after = ev.guard_snapshot([f])
            ev.guard_assert_unchanged(before, after, "unit")


class IsolationProofTest(unittest.TestCase):
    """Safeguard #3: evaluation touches ONLY temporary Chroma paths."""

    def test_clients_only_on_temp_paths_and_real_store_untouched(self):
        import gc

        from llama_index.core.embeddings import MockEmbedding

        real_store_files = [ev.STORAGE_DIR / "chroma.sqlite3",
                            ev.INDEX_MANIFEST]
        guard_before = ev.guard_snapshot(real_store_files)

        # Pristine production globals, captured BEFORE anything runs: other
        # suites in the same session may already have built the real query
        # engine, so restoration is asserted against these captured values
        # (not against hard-coded defaults), plus their production identity.
        import app.agents.books_agent as ba
        from llama_index.core import Settings

        pristine = (ba._query_engine, ba.CHROMA_COLLECTION, ba.STORAGE_DIR,
                    Settings._llm, Settings._embed_model)
        self.assertEqual(str(pristine[2]), str(ev.STORAGE_DIR))
        self.assertEqual(pristine[1], "books")

        tmp = tempfile.mkdtemp(prefix="rag_eval_test_")
        tmp_path = Path(tmp) / "chroma"
        try:
            docs_dir = Path(tmp) / "docs"
            docs_dir.mkdir()
            # tiny doc set built in-memory by the deterministic generator
            (docs_dir / "fixture_table.pdf").write_bytes(
                mrf.build_fixture_table_pdf())
            (docs_dir / "fixture_headings.docx").write_bytes(
                mrf.build_fixture_headings_docx())
            docs = sorted(docs_dir.iterdir())

            mock_embed = MockEmbedding(embed_dim=8)
            real_cls, paths = ev.instrument_client_paths()
            try:
                _index, collection, entries = ev.build_eval_index(
                    tmp_path, "books_eval_isolation", docs, mock_embed)
                rows = ev.query_production_seam(
                    tmp_path, "books_eval_isolation",
                    ["What is the harvest yield for quarter three?"],
                    mock_embed)
            finally:
                import chromadb
                chromadb.PersistentClient = real_cls

            # every constructed client was on the temp path
            self.assertTrue(paths, "no client paths recorded")
            for p in paths:
                self.assertIn(str(tmp_path), p,
                              f"client constructed off temp path: {p}")
                self.assertNotIn(str(ev.STORAGE_DIR), p,
                                 "a client was built on the production path")

            # the retrieval actually returned the seeded temp content
            self.assertEqual(collection.count(),
                             sum(e["chunks"] for e in entries.values()))
            self.assertGreater(len(rows[0]["nodes"]), 0)
            self.assertTrue(all(
                n["metadata"].get("file_name") in
                {"fixture_table.pdf", "fixture_headings.docx"}
                for n in rows[0]["nodes"]))

            # production caches restored to their pre-call state (identity,
            # not equality: a temp-built engine must never survive the seam)
            self.assertIs(ba._query_engine, pristine[0])
            self.assertEqual(ba.CHROMA_COLLECTION, pristine[1])
            self.assertEqual(str(ba.STORAGE_DIR), str(pristine[2]))
            self.assertIs(Settings._llm, pristine[3])
            # the seam restores the embed model to ITS entry state — which is
            # the mock build_eval_index installed, not the pristine value;
            # the finally below returns the pristine one.
            self.assertIs(Settings._embed_model, mock_embed)
        finally:
            # build_eval_index() writes Settings.embed_model globally; put it
            # back so this test leaves no trace even when an assertion fails
            (ba._query_engine, ba.CHROMA_COLLECTION, ba.STORAGE_DIR,
             Settings._llm, Settings._embed_model) = pristine
            # release file locks, then PROVE the temp dir is fully removable
            ev.release_chroma_clients()
            gc.collect()
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            self.assertFalse(Path(tmp).exists(),
                             "temp eval dir could not be removed — an eval "
                             "client is still holding files open")

        guard_after = ev.guard_snapshot(real_store_files)
        ev.guard_assert_unchanged(guard_before, guard_after,
                                  "real-store guard (isolation test)")

    def test_production_seam_restores_namespace_on_failure(self):
        from unittest import mock

        import app.agents.books_agent as ba

        saved = (ba.STORAGE_DIR, ba.CHROMA_COLLECTION, ba._query_engine)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(ba, "_get_query_engine",
                                   side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    ev.query_production_seam(Path(tmp), "coll", ["q"], None)
        self.assertEqual((ba.STORAGE_DIR, ba.CHROMA_COLLECTION,
                          ba._query_engine), saved)
        self.assertEqual(str(ba.STORAGE_DIR), str(ev.STORAGE_DIR))


class ArtifactEvidenceV3Test(unittest.TestCase):
    """artifact_schema 3: every reported metric is recomputable from the
    saved evidence — verified with an INDEPENDENT implementation that
    never calls ev.node_covers / ev.compute_metrics / ev.aggregate."""

    VAR_METADATA_KEYS = ("timestamp_utc", "run_id", "latency_ms", "ingest")

    @staticmethod
    def _rt(obj):
        """Round-trip through JSON so fixtures behave like stored files."""
        return json.loads(json.dumps(obj))

    @classmethod
    def _indep_metrics(cls, row):
        import math
        ref = row.get("gold_ref") or {}
        nodes = row.get("retrieved_top5") or []
        gold_docs = ref.get("gold_documents") or []
        gold_pages = ref.get("gold_pages")
        min_docs = int(ref.get("min_docs_covered", 1) or 1)
        rel = []
        for node in nodes:
            if node.get("file") not in gold_docs:
                rel.append(False)
                continue
            if not gold_pages:
                rel.append(True)
                continue
            label = node.get("page")
            if label is None:
                label = node.get("page_number")
            if label is None:
                rel.append(True)          # page-less node (docx), v2 rule
                continue
            try:
                rel.append(int(str(label).strip()) in {int(p)
                                                       for p in gold_pages})
            except (TypeError, ValueError):
                rel.append(False)

        def docs_upto(k):
            seen = set()
            for i in range(min(k, len(nodes))):
                if rel[i]:
                    seen.add(nodes[i].get("file"))
            return len(seen)

        first = next((n["rank"] for n, r in zip(nodes, rel) if r), None)
        out = {}
        for k in (1, 3, 5):
            hits = sum(rel[:k])
            out[f"hit@{k}"] = docs_upto(k) >= min_docs
            out[f"precision@{k}"] = round(hits / k, 6)
        out["first_hit_rank"] = first
        out["reciprocal_rank"] = round(1.0 / first, 6) if first else 0.0
        out["ndcg@5"] = round(1.0 / math.log2(1 + first), 6) \
            if first else 0.0
        out["coverage"] = docs_upto(5) >= min_docs
        return out

    @classmethod
    def _indep_aggregate(cls, rows):
        per = [cls._indep_metrics(r) for r in rows]
        n = len(per)
        out = {"cases": n}
        for key in ("hit@1", "hit@3", "hit@5"):
            out[key] = round(sum(1 for m in per if m[key]) / n, 4)
        for key in ("precision@1", "precision@3", "precision@5",
                    "reciprocal_rank", "ndcg@5"):
            out[key] = round(sum(m[key] for m in per) / n, 4)
        out["coverage"] = round(sum(1 for m in per
                                    if m["coverage"]) / n, 4)
        return out

    @staticmethod
    def _case(cid, section, answerable, docs=None, pages=None, min_docs=1):
        return {"id": cid, "section": section,
                "synthetic": section == "synthetic_fixtures",
                "answerable": answerable,
                "gold_documents": docs or [], "gold_pages": pages,
                "min_docs_covered": min_docs, "labels": ["fixture"]}

    def _row(self, case, entries):
        """Fixture row assembled from PRODUCTION writer components; its
        recorded metrics are what the harness would persist, and are then
        attacked/verified with the independent implementation only."""
        t5 = [{"node_id": nid, "score": sc,
               "metadata": {"file_name": f, "page_label": pl,
                            "page_number": pn}}
              for f, pl, pn, nid, sc in entries]
        prod = [{"node_id": nid, "score": sc}
                for *_, nid, sc in entries[:3]]
        metrics = ev.compute_metrics(case, [n["metadata"] for n in t5])
        top1 = prod[0]["score"] if prod else None
        row = {"id": case["id"], "section": case["section"],
               "synthetic": case["synthetic"],
               "answerable": case["answerable"],
               "labels": list(case["labels"]),
               "first_hit_rank": metrics["first_hit_rank"],
               "metrics": metrics,
               "abstention_proxy": ev.abstention_proxy(
                   top1, case["answerable"])}
        row.update(ev.case_evidence(case, t5, prod))
        return row

    def _fixture(self):
        cases = [
            self._case("F1", "real_corpus", True,
                       docs=["a.pdf"], pages=[1]),
            self._case("F2", "real_corpus", True, docs=["b.docx"]),
            self._case("F3", "synthetic_fixtures", True,
                       docs=["c.pdf", "d.docx"], pages=[1], min_docs=2),
            self._case("F4", "synthetic_fixtures", False),
        ]
        entries = {
            "F1": [("a.pdf", "1", None, "n1", 0.77),
                   ("z.pdf", "3", None, "n2", 0.61),
                   ("y.pdf", "9", None, "n3", 0.59)],
            "F2": [("z.pdf", "1", None, "n4", 0.70),
                   ("b.docx", None, None, "n5", 0.68),
                   ("x.pdf", "2", None, "n6", 0.60)],
            "F3": [("c.pdf", "1", None, "n7", 0.83),
                   ("d.docx", None, None, "n8", 0.76),
                   ("z.pdf", "7", None, "n9", 0.64)],
            "F4": [("z.pdf", "1", None, "n10", 0.72),
                    ("y.pdf", "2", None, "n11", 0.70),
                    ("x.pdf", "3", None, "n12", 0.65)],
        }
        rows = [self._row(c, entries[c["id"]]) for c in cases]
        return cases, rows

    def test_independent_full_recalculation_and_aggregates(self):
        cases, rows = self._fixture()
        rows = self._rt(rows)               # behave like a stored artifact
        # positive control: correct evidence passes the pre-write gate
        ev.validate_evidence(rows, cases)
        for row in rows:
            self.assertEqual(self._indep_metrics(row), row["metrics"],
                             row["id"])          # all 10 metric keys
            self.assertEqual(row["first_hit_rank"],
                             self._indep_metrics(row)["first_hit_rank"])
        for section in ("real_corpus", "synthetic_fixtures"):
            part = [r for r in rows
                    if r["section"] == section and r["answerable"]]
            self.assertEqual(self._indep_aggregate(part),
                             ev.aggregate(part), section)

    def test_sections_stay_separate(self):
        _, rows = self._fixture()
        rows = self._rt(rows)
        real = [r for r in rows
                if r["section"] == "real_corpus" and r["answerable"]]
        syn = [r for r in rows
               if r["section"] == "synthetic_fixtures" and r["answerable"]]
        self.assertEqual((len(real), len(syn)), (2, 1))
        self.assertTrue(all(r["section"] == "real_corpus" and
                            not r["synthetic"] for r in real))
        self.assertTrue(all(r["section"] == "synthetic_fixtures" and
                            r["synthetic"] for r in syn))
        real_block = self._indep_aggregate(real)
        self.assertEqual(real_block, ev.aggregate(real))
        # a mixed aggregation provably differs: no cross-section bleed
        mixed = self._indep_aggregate(real + syn)
        self.assertNotEqual(real_block, mixed)
        unans = [r for r in rows if not r["answerable"]]
        self.assertEqual([r["id"] for r in unans], ["F4"])

    def test_page_rules_against_v3_evidence(self):
        # PDF node on an allowed page -> hit
        c = self._case("P1", "real_corpus", True,
                       docs=["a.pdf"], pages=[1])
        row = self._row(c, [("a.pdf", "1", None, "n1", 0.8)])
        self.assertTrue(self._indep_metrics(self._rt(row))["hit@1"])
        # PDF node on a disallowed page -> no hit at any k
        row = self._row(c, [("a.pdf", "2", None, "n1", 0.8)])
        m = self._indep_metrics(self._rt(row))
        self.assertFalse(any(m[k] for k in ("hit@1", "hit@3", "hit@5",
                                            "coverage")))
        self.assertIsNone(m["first_hit_rank"])
        # page-less DOCX member of a multi-doc gold case stays reachable
        c = self._case("P2", "real_corpus", True,
                       docs=["c.pdf", "d.docx"], pages=[1], min_docs=2)
        row = self._row(c, [("c.pdf", "1", None, "n1", 0.9),
                            ("d.docx", None, None, "n2", 0.7),
                            ("z.pdf", "1", None, "n3", 0.5)])
        m = self._indep_metrics(self._rt(row))
        self.assertFalse(m["hit@1"])
        self.assertTrue(m["hit@3"])       # docx at rank 2 covers
        self.assertTrue(m["coverage"])
        # malformed page label on a gold file cannot confirm the page —
        # only the page-less docx member can still cover
        row = self._row(c, [("c.pdf", "junk!!", None, "n1", 0.9),
                            ("d.docx", None, None, "n2", 0.7)])
        m = self._indep_metrics(self._rt(row))
        self.assertEqual(m["first_hit_rank"], 2)
        self.assertFalse(m["hit@3"])      # 1 of min_docs=2 confirmed


class ArtifactValidationTest(unittest.TestCase):
    """Missing/malformed/incomplete/non-consecutive evidence must fail
    explicitly — never yield a purportedly fully auditable artifact."""

    def _case(self):
        return ArtifactEvidenceV3Test()._case(
            "V1", "real_corpus", True, docs=["a.pdf"], pages=[1])

    def _row(self):
        case = self._case()
        base = ArtifactEvidenceV3Test()
        return case, base._row(case, [("a.pdf", "1", None, "n1", 0.8),
                                      ("z.pdf", "2", None, "n2", 0.6)])

    def test_malformed_evidence_fails_explicitly(self):
        mutations = {}

        def mut(name, fn):
            mutations[name] = fn

        mut("missing_block",
            lambda r: r.pop("retrieved_top5"))
        mut("six_nodes",
            lambda r: r["retrieved_top5"].extend(
                dict(r["retrieved_top5"][0], rank=i, node_id=f"x{i}")
                for i in range(2, 8)))
        mut("rank_not_1_based",
            lambda r: r["retrieved_top5"][0].update(rank=2))
        mut("rank_non_consecutive",
            lambda r: r["retrieved_top5"][1].update(rank=4))
        mut("rank_not_int",
            lambda r: r["retrieved_top5"][0].update(rank="1"))
        mut("missing_identity_field",
            lambda r: r["retrieved_top5"][0].pop("file"))
        mut("node_id_not_str",
            lambda r: r["retrieved_top5"][0].update(node_id=7))
        mut("metrics_dangle_from_evidence",
            lambda r: r["metrics"].update(**{
                "hit@1": not r["metrics"]["hit@1"]}))
        mut("first_hit_rank_tampered",
            lambda r: r.update(first_hit_rank=2))
        mut("empty_evidence_with_hits",
            lambda r: r.update(retrieved_top5=[]))
        mut("metrics_block_missing",
            lambda r: r.pop("metrics"))

        for name, mutate in mutations.items():
            with self.subTest(case=name):
                case, row = self._row()
                row = ArtifactEvidenceV3Test._rt(row)
                mutate(row)
                with self.assertRaises(ev.GuardFailure):
                    ev.validate_evidence([row], [case])

    def test_gold_ref_mismatch_detected(self):
        case, row = self._row()
        row = ArtifactEvidenceV3Test._rt(row)
        row["gold_ref"]["min_docs_covered"] = 2
        with self.assertRaises(ev.GuardFailure):
            ev.validate_evidence([row], [case])
        case, row = self._row()
        row = ArtifactEvidenceV3Test._rt(row)
        row["gold_ref"]["gold_documents"] = ["ghost.pdf"]
        with self.assertRaises(ev.GuardFailure):
            ev.validate_evidence([row], [case])


class ArtifactDeterminismAndSafetyTest(unittest.TestCase):
    """Recalculation is deterministic outside proven variable metadata;
    the artifact-only reader path never builds a Chroma client or touches
    the production store; historical artifacts stay byte-identical."""

    PINNED = {
        "rag_gold_v1.json":
            "16e25677f2effbd65bba0475754f7b9f5c85344d468497dd322234b155bbb25d",
        "rag_baseline_v1.json":
            "2c1d4ae9de438a25c761680b06fe083022af4a8c7c27872e8f6a1e13027c659d",
        "rag_baseline_v2.json":
            "cfe43ee2f803eba0e28b1ce15eaf42a7a9a1b547e9acf7e0c0128a9e837f31ee",
        "rag_baseline_v3.json":
            "aa89c37bc23b0dc748255df9410a28ba43a6f8d8043070ad54cafd6e175de1cd",
    }
    VAR_METADATA_KEYS = ArtifactEvidenceV3Test.VAR_METADATA_KEYS

    @staticmethod
    def _normalize(art):
        out = {k: v for k, v in art.items()
               if k not in ArtifactDeterminismAndSafetyTest.VAR_METADATA_KEYS}
        if isinstance(out.get("guards"), dict):
            out["guards"] = {k: v for k, v in out["guards"].items()
                              if k != "chroma_client_paths_constructed"}
        return out

    def test_deterministic_recalc_and_timing_exclusion_is_exact(self):
        ev2 = ArtifactEvidenceV3Test()
        cases, rows = ev2._fixture()
        run_a = json.dumps([ev2._indep_metrics(r) for r in rows]) \
            + json.dumps({s: ev2._indep_aggregate(
                [r for r in rows if r["section"] == s])
                for s in ("real_corpus", "synthetic_fixtures")})
        run_b = json.dumps([ev2._indep_metrics(r) for r in rows]) \
            + json.dumps({s: ev2._indep_aggregate(
                [r for r in rows if r["section"] == s])
                for s in ("real_corpus", "synthetic_fixtures")})
        self.assertEqual(run_a, run_b)
        # normalization strips EXACTLY the proven-variable fields
        fixture_art = {"timestamp_utc": "t", "run_id": "r",
                       "latency_ms": {}, "ingest": {"seconds": 1},
                       "guards": {"chroma_client_paths_constructed": ["x"],
                                  "gold_unchanged": True},
                       "artifact_schema": {"version": 3}}
        norm = self._normalize(ArtifactEvidenceV3Test._rt(fixture_art))
        stripped = set(fixture_art) - set(norm)
        self.assertEqual(stripped, set(self.VAR_METADATA_KEYS))
        self.assertEqual(set(norm["guards"]), {"gold_unchanged"})
        self.assertEqual(norm,
                         self._normalize(ArtifactEvidenceV3Test._rt(
                             fixture_art)))
        for key in self.VAR_METADATA_KEYS:
            self.assertNotIn(key, norm)

    def test_artifact_only_reader_builds_no_chroma_client(self):
        guard_before = ev.guard_snapshot(ev.GUARD_FILES)
        real_cls, client_paths = ev.instrument_client_paths()
        try:
            # the complete artifact-only reader path: load gold + historical
            # artifacts, round-trip, validate, recompute metrics/aggregates
            _gold, _sha = ev.load_gold(ev.GOLD_PATH)
            mixing_probe = []
            for name in ("rag_baseline_v1.json", "rag_baseline_v2.json"):
                art = json.loads((ev.PROJECT_ROOT / "benchmarks" / name)
                                 .read_text(encoding="utf-8"))
                mixing_probe.append(sorted(art["sections"]))
            cases, rows = ArtifactEvidenceV3Test()._fixture()
            ev.validate_evidence(rows, cases)
            for row in rows:
                ev.compute_metrics(
                    row["gold_ref"],
                    [{"file_name": n["file"], "page_label": n["page"],
                      "page_number": n["page_number"]}
                     for n in row["retrieved_top5"]])
            ev.aggregate([r for r in rows if r["answerable"]])
            self.assertEqual(mixing_probe,
                             [["real_corpus", "synthetic_fixtures"]] * 2)
        finally:
            import chromadb
            chromadb.PersistentClient = real_cls
        self.assertEqual(client_paths, [],
                         "artifact-only recalculation constructed a client")
        ev.guard_assert_unchanged(
            guard_before, ev.guard_snapshot(ev.GUARD_FILES),
            "store guard during artifact-only recalculation")

    def test_pinned_gold_and_all_baseline_files_unchanged(self):
        for name, digest in self.PINNED.items():
            path = ev.PROJECT_ROOT / "benchmarks" / name
            self.assertEqual(ev.sha256_file(path), digest, name)


class _GateReached(Exception):
    """Raised inside a patched dependency to prove main() got that far."""


class OverwriteSafetyTest(unittest.TestCase):
    """Output safety: an existing report is NEVER overwritten silently.

    The refusal must fire before ANY work — no guard snapshot, no gold
    read, no Ollama probe, no write — so every test below patches
    guard_snapshot/urlopen with functions that raise: if main() ever
    reaches them, the test fails. Fully offline; the real baselines are
    only ever read (hash-checked) here, never written.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "report.json"

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _forbid_work():
        return (
            mock.patch.object(
                ev, "guard_snapshot",
                side_effect=AssertionError("reached the guard stage")),
            mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("reached the network stage")),
        )

    def test_existing_report_refused_before_any_work(self):
        self.out.write_text('{"pinned": true}', encoding="utf-8")
        before = self.out.read_bytes()
        guard_patch, net_patch = self._forbid_work()
        with guard_patch, net_patch, \
                contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = ev.main(["--out", str(self.out)])
        self.assertEqual(rc, 3)
        self.assertEqual(self.out.read_bytes(), before,
                         "refusal must not touch the existing report")
        msg = buf.getvalue()
        self.assertIn("REFUSING TO OVERWRITE", msg)
        self.assertIn("--force", msg)
        self.assertIn(str(self.out), msg)

    def test_documented_bare_command_never_clobbers_pinned_v1(self):
        pin = ArtifactDeterminismAndSafetyTest.PINNED[
            "rag_baseline_v1.json"]
        self.assertEqual(ev.DEFAULT_OUT.name, "rag_baseline_v1.json")
        self.assertTrue(ev.DEFAULT_OUT.exists())
        guard_patch, net_patch = self._forbid_work()
        with guard_patch, net_patch, \
                contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = ev.main([])
        self.assertEqual(rc, 3)
        self.assertEqual(ev.sha256_file(ev.DEFAULT_OUT), pin,
                         "bare run altered the pinned v1 baseline")
        self.assertIn("REFUSING TO OVERWRITE", buf.getvalue())

    def test_force_is_the_only_opt_in_and_still_writes_nothing_here(self):
        self.out.write_text('{"original": 1}', encoding="utf-8")
        before = self.out.read_bytes()
        with mock.patch.object(
                ev, "guard_snapshot",
                side_effect=_GateReached("past the overwrite gate")) as g, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(_GateReached):
                ev.main(["--out", str(self.out), "--force"])
            g.assert_called_once()
        self.assertEqual(self.out.read_bytes(), before,
                         "--force must not write before the run proceeds")

    def test_fresh_out_path_passes_the_gate_and_pre_creates_nothing(self):
        with mock.patch.object(
                ev, "guard_snapshot",
                side_effect=_GateReached("past the overwrite gate")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(_GateReached):
                ev.main(["--out", str(self.out)])
        self.assertFalse(self.out.exists(),
                         "a run stopped before writing must not create "
                         "the report file")


if __name__ == "__main__":
    unittest.main()
