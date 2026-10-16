"""M2 tests: ingestion integrity and resource limits (deterministic, offline).

Drives the REAL `ingest_books.run_ingest()` against a temporary books folder,
temporary Chroma storage and a temporary manifest. Embeddings are replaced by
LlamaIndex's `MockEmbedding`, so there is no Ollama, no Gemini, no network and
no quota. The production knowledge base, `storage/` and the real manifest are
never touched.

Covered (brief items 1-8 plus the manifest-compatibility invariants):
  0. unsupported files -> reported as ignored, never indexed or recorded
  1. healthy PDF -> indexed, chunks persisted and verified in the store
  2. healthy DOCX -> indexed
  3. corrupt PDF -> failed, never marked indexed
  4. reader returns no documents (the measured "Skipping..." defect) -> failed
  5. zero-text PDF -> reported explicitly, never called "blank"
  6. mixed PDF -> partial; extractable pages usable, unextractable pages named
  7. oversized file -> rejected before the reader is called
  8. page-limit violation -> rejected in preflight, before extraction
  9. encrypted PDF -> explicit failure, no password guessing
 10. idempotent re-run: unchanged healthy file is not re-embedded, manifest stable
 11. failed file is retried on the next run
 12. a failed replacement never deletes the previously indexed content
 13. a changed file replaces its own stale chunks (insert-then-delete)
 14. legacy v1 manifest: migrated, chunk count verified against the store
 15. legacy v1 entry whose vectors are missing -> repaired, not trusted
 16. corrupt manifest -> treated as empty and reported, ingestion still works
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # sibling diagnostic script

from llama_index.core import Settings  # noqa: E402
from llama_index.core.embeddings import MockEmbedding  # noqa: E402

import ingest_books  # noqa: E402

COLLECTION = "books_integrity_test"


# --------------------------------------------------------------------------
# fixtures: real PDFs without extra dependencies (raw PDF syntax + pypdf)
# --------------------------------------------------------------------------


def _esc(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _text_page(lines) -> bytes:
    ops = ["BT"]
    y = 720
    for line in lines:
        ops.append(f"/F1 11 Tf 1 0 0 1 72 {y} Tm ({_esc(line)}) Tj")
        y -= 18
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


_BLANK_PAGE = b"0.9 0.9 0.9 rg 72 400 468 200 re f"


def write_pdf(path: Path, pages) -> None:
    """Write a PDF; each page is a list of text lines, or None for a page with
    NO text layer at all (the scanned-page shape)."""
    contents = [_text_page(p) if p is not None else _BLANK_PAGE for p in pages]
    objs = {}
    page_ids = [4 + 2 * i for i in range(len(contents))]
    content_ids = [5 + 2 * i for i in range(len(contents))]
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(contents)} >>".encode()
    objs[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for i, content in enumerate(contents):
        objs[page_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_ids[i]} 0 R >>"
        ).encode()
        objs[content_ids[i]] = (
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref = len(out)
    top = max(objs)
    out += f"xref\n0 {top + 1}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, top + 1):
        out += (f"{offsets[num]:010d} 00000 n \n" if num in offsets
                else b"0000000000 65535 f \n").encode()
    out += (f"trailer\n<< /Size {top + 1} /Root 1 0 R >>\nstartxref\n{xref}\n"
            "%%EOF\n").encode()
    path.write_bytes(bytes(out))


def write_docx(path: Path, heading: str, body: str) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading(heading, level=1)
    doc.add_paragraph(body)
    doc.save(path)


def write_blank_pdf(path: Path, pages: int) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as fh:
        writer.write(fh)


def write_encrypted_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    with open(path, "wb") as fh:
        writer.write(fh)


BEE = [
    "Field Guide to Urban Beekeeping",
    "A healthy hive in midsummer can contain up to 60,000 worker bees.",
    "The queen lays up to 2,000 eggs per day at the peak of spring buildup.",
]


class _ExplodingEmbedding(MockEmbedding):
    """Deterministic embedding outage (no network)."""

    def get_text_embedding_batch(self, *args, **kwargs):
        raise RuntimeError("embedding backend unavailable")


class Harness:
    """Isolated books folder + storage + manifest for one test."""

    def __init__(self, embed_model=None):
        self.tmp = Path(tempfile.mkdtemp(prefix="ingest_m2_"))
        self.books = self.tmp / "books"
        self.books.mkdir()
        self.storage = self.tmp / "storage"
        self.storage.mkdir()
        self.manifest_path = self.storage / ".index_manifest.json"
        self.embed = embed_model or MockEmbedding(embed_dim=8)

    def run(self, **kwargs):
        return ingest_books.run_ingest(
            books_dir=self.books,
            storage_dir=self.storage,
            manifest_path=self.manifest_path,
            collection_name=COLLECTION,
            embed_model=self.embed,
            verbose=False,
            **kwargs,
        )

    def collection(self):
        import chromadb

        return chromadb.PersistentClient(path=str(self.storage)).get_or_create_collection(
            COLLECTION
        )

    def stored(self, path: Path) -> int:
        return len(self.collection().get(where={"file_path": str(path)}, include=[])["ids"] or [])

    def stored_text(self, path: Path) -> str:
        got = self.collection().get(where={"file_path": str(path)}, include=["documents"])
        return "\n".join(got["documents"] or [])

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def entry(self, path: Path) -> dict:
        return self.manifest()["files"][str(path)]

    def add(self, name: str, content) -> Path:
        path = self.books / name
        if name.endswith(".pdf") and isinstance(content, list):
            write_pdf(path, content)
        elif name.endswith(".docx"):
            write_docx(path, name, content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def touch(self, path: Path, content) -> None:
        """Rewrite a file with a NEWER mtime (content change detection).

        A list means "these text lines on ONE page" (same meaning as in `add`).
        """
        if isinstance(content, list):
            write_pdf(path, [content])
        else:
            path.write_text(content, encoding="utf-8")
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns + 10 ** 9, stat.st_mtime_ns + 10 ** 9))


class IngestTestCase(unittest.TestCase):
    """Restores LlamaIndex's global Settings fields exactly.

    `Settings.embed_model` must NOT be read to snapshot it: the untouched
    default resolves to OpenAI and raises without an API key. The fields live
    in the private `__dict__` (`_embed_model`, `_llm`, ...), so they are saved
    and restored there.
    """

    def setUp(self):
        self._saved_settings = dict(vars(Settings))

    def tearDown(self):
        vars(Settings).clear()
        vars(Settings).update(self._saved_settings)

    def assert_invariants(self, harness: Harness, summary: dict) -> None:
        """Statuses must never contradict the vector store."""
        for key, entry in summary["files"].items():
            stored = harness.stored(Path(key))
            if entry["status"] in (ingest_books.STATUS_INDEXED, ingest_books.STATUS_PARTIAL):
                self.assertGreater(
                    entry["chunks"], 0,
                    f"{Path(key).name} was marked {entry['status']} with 0 chunks",
                )
                self.assertEqual(
                    entry["chunks"], stored,
                    f"{Path(key).name}: manifest claims {entry['chunks']} chunks, "
                    f"store holds {stored}",
                )
            elif entry["status"] == ingest_books.STATUS_FAILED:
                self.assertIsNotNone(entry.get("category"))
                self.assertIsNotNone(entry.get("message"))


# --------------------------------------------------------------------------
# 1-2: supported formats still work
# --------------------------------------------------------------------------


class TestHealthyFiles(IngestTestCase):
    def test_healthy_pdf_is_indexed_with_verified_chunks(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        summary = h.run()

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_INDEXED)
        self.assertEqual(entry["unit"], "page")
        self.assertEqual(entry["documents"], 1)
        self.assertEqual(entry["nonempty_pages"], 1)
        self.assertEqual(entry["empty_pages"], [])
        self.assertGreater(entry["chunks"], 0)
        self.assertEqual(entry["chunks"], h.stored(pdf))
        self.assertIn("60,000 worker bees", h.stored_text(pdf))
        self.assertEqual(summary["counts"][ingest_books.STATUS_INDEXED], 1)
        self.assert_invariants(h, summary)

    def test_healthy_docx_is_indexed(self):
        h = Harness()
        docx = h.add("principles.docx", "A Hohmann transfer requires two engine burns.")
        summary = h.run()

        entry = h.entry(docx)
        self.assertEqual(entry["status"], ingest_books.STATUS_INDEXED)
        self.assertEqual(entry["unit"], "file")  # .docx has no page concept
        self.assertGreater(entry["chunks"], 0)
        self.assertEqual(entry["chunks"], h.stored(docx))
        self.assert_invariants(h, summary)


class TestUnsupportedFiles(IngestTestCase):
    def test_non_document_files_are_reported_ignored_and_never_indexed(self):
        h = Harness()
        h.add("guide.pdf", [BEE])
        notes = h.add("notes.txt", "scratch pad, not a document")
        summary = h.run()

        self.assertEqual(summary["ignored"], [str(notes)])
        self.assertNotIn(str(notes), h.manifest()["files"])
        self.assertEqual(h.stored(notes), 0)
        self.assert_invariants(h, summary)

    def test_run_with_no_candidates_still_reports(self):
        h = Harness()
        h.add("notes.txt", "only a note")
        summary = h.run()
        self.assertEqual(summary["files"], {})
        self.assertEqual(len(summary["ignored"]), 1)
        self.assertEqual(summary["stored_chunks"], 0)


# --------------------------------------------------------------------------
# 3-6, 9: failure reporting
# --------------------------------------------------------------------------


class TestExtractionFailures(IngestTestCase):
    def test_corrupt_pdf_is_failed_and_never_indexed(self):
        h = Harness()
        bad = h.add("corrupt.pdf", "This is not a PDF at all: no objects, no xref.")
        summary = h.run()

        entry = h.entry(bad)
        self.assertEqual(entry["status"], ingest_books.STATUS_FAILED)
        self.assertNotEqual(entry["status"], ingest_books.STATUS_INDEXED)
        self.assertIn(entry["category"], (ingest_books.CAT_UNREADABLE,))
        self.assertEqual(entry["chunks"], 0)
        self.assertEqual(h.stored(bad), 0)
        self.assertEqual(summary["counts"][ingest_books.STATUS_FAILED], 1)
        self.assert_invariants(h, summary)

    def test_reader_returning_no_documents_is_failed(self):
        """The measured defect: SimpleDirectoryReader logs 'Skipping...' for an
        unreadable file and returns [] WITHOUT raising. That must be a failure."""
        h = Harness()
        pdf = h.add("silently_skipped.pdf", [BEE])
        with mock.patch.object(ingest_books, "_load_documents", return_value=[]):
            summary = h.run()

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_FAILED)
        self.assertEqual(entry["category"], ingest_books.CAT_UNREADABLE)
        self.assertIn("returned no documents", entry["message"])
        self.assertEqual(h.stored(pdf), 0)
        self.assert_invariants(h, summary)

    def test_zero_text_pdf_is_reported_not_called_blank(self):
        h = Harness()
        pdf = h.books / "scanned.pdf"
        write_blank_pdf(pdf, 2)  # pages with no text layer at all
        summary = h.run()

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_FAILED)
        self.assertEqual(entry["category"], ingest_books.CAT_NO_TEXT)
        self.assertEqual(entry["documents"], 2)
        self.assertEqual(entry["nonempty_pages"], 0)
        self.assertEqual(len(entry["empty_pages"]), 2)
        self.assertIn("NOT assumed blank", entry["message"])
        self.assertIn("no OCR", entry["message"])
        self.assertEqual(h.stored(pdf), 0)
        self.assert_invariants(h, summary)

    def test_mixed_pdf_is_partial_and_keeps_usable_pages(self):
        h = Harness()
        pdf = h.add("mixed.pdf", [BEE, None])  # page 1 text, page 2 no text layer
        summary = h.run()

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_PARTIAL)
        self.assertEqual(entry["documents"], 2)
        self.assertEqual(entry["nonempty_pages"], 1)
        self.assertEqual(entry["empty_pages"], ["page 2"])
        self.assertGreater(entry["chunks"], 0)
        self.assertIn("60,000 worker bees", h.stored_text(pdf))  # page 1 still usable
        self.assertIn("no extractable text", entry["message"])
        self.assertEqual(summary["counts"][ingest_books.STATUS_PARTIAL], 1)
        self.assert_invariants(h, summary)

    def test_encrypted_pdf_fails_explicitly_without_guessing_passwords(self):
        h = Harness()
        pdf = h.books / "secret.pdf"
        write_encrypted_pdf(pdf)
        with mock.patch.object(ingest_books, "_load_documents") as loader:
            summary = h.run()
        loader.assert_not_called()  # rejected in preflight, before extraction

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_FAILED)
        self.assertEqual(entry["category"], ingest_books.CAT_ENCRYPTED)
        self.assertIn("password-protected", entry["message"])
        self.assertEqual(h.stored(pdf), 0)
        self.assert_invariants(h, summary)


# --------------------------------------------------------------------------
# 7-8: resource limits (enforced before the reader loads anything)
# --------------------------------------------------------------------------


class TestResourceLimits(IngestTestCase):
    def test_oversized_file_is_rejected_before_reading(self):
        h = Harness()
        pdf = h.add("big.pdf", [BEE])
        with mock.patch.object(ingest_books, "_load_documents") as loader:
            summary = h.run(max_file_mb=0.000001)  # ~1 byte cap
        loader.assert_not_called()

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_FAILED)
        self.assertEqual(entry["category"], ingest_books.CAT_OVERSIZED)
        self.assertIn("before the document reader was called", entry["message"])
        self.assertEqual(h.stored(pdf), 0)
        self.assert_invariants(h, summary)

    def test_page_limit_is_enforced_in_preflight(self):
        h = Harness()
        pdf = h.books / "three_pages.pdf"
        write_blank_pdf(pdf, 3)
        with mock.patch.object(ingest_books, "_load_documents") as loader:
            summary = h.run(max_pages=2)
        loader.assert_not_called()

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_FAILED)
        self.assertEqual(entry["category"], ingest_books.CAT_PAGE_LIMIT)
        self.assertIn("INGEST_MAX_PAGES=2", entry["message"])
        self.assert_invariants(h, summary)

    def test_disabled_caps_allow_processing(self):
        h = Harness()
        pdf = h.add("ok.pdf", [BEE])
        summary = h.run(max_file_mb=0, max_pages=0)  # <= 0 disables both caps
        self.assertEqual(h.entry(pdf)["status"], ingest_books.STATUS_INDEXED)
        self.assert_invariants(h, summary)


# --------------------------------------------------------------------------
# 10-13: idempotency and replacement safety
# --------------------------------------------------------------------------


class TestIdempotencyAndReplacement(IngestTestCase):
    def test_rerun_is_idempotent(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        first = h.run()
        chunks_after_first = h.stored(pdf)
        manifest_after_first = h.manifest_path.read_bytes()

        second = h.run()
        entry = h.entry(pdf)
        self.assertEqual(second["counts"][ingest_books.OUTCOME_UNCHANGED], 1)
        self.assertEqual(second["counts"][ingest_books.STATUS_INDEXED], 0)
        self.assertEqual(h.stored(pdf), chunks_after_first)  # no duplicate chunks
        self.assertEqual(h.manifest_path.read_bytes(), manifest_after_first)  # no drift
        self.assertEqual(entry["chunks"], chunks_after_first)
        self.assertEqual(first["stored_chunks"], second["stored_chunks"])
        self.assert_invariants(h, second)

    def test_failed_file_is_not_marked_indexed_and_is_retried(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        with mock.patch.object(ingest_books, "_load_documents", return_value=[]):
            h.run()
        self.assertEqual(h.entry(pdf)["status"], ingest_books.STATUS_FAILED)

        # Next run, reader works again: a failure must never be sticky.
        summary = h.run()
        self.assertEqual(h.entry(pdf)["status"], ingest_books.STATUS_INDEXED)
        self.assertEqual(h.stored(pdf), h.entry(pdf)["chunks"])
        self.assert_invariants(h, summary)

    def test_failed_replacement_never_deletes_the_previous_content(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        h.run()
        original_chunks = h.stored(pdf)
        self.assertGreater(original_chunks, 0)

        # Content changes AND the new ingest fails -> old version must survive.
        h.touch(pdf, "completely different bytes that no reader can parse")
        with mock.patch.object(ingest_books, "_load_documents", return_value=[]):
            summary = h.run()

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_FAILED)
        self.assertEqual(entry["previous"]["chunks"], original_chunks)
        self.assertEqual(h.stored(pdf), original_chunks)  # nothing deleted
        self.assertIn("60,000 worker bees", h.stored_text(pdf))
        self.assert_invariants(h, summary)

    def test_failed_embedding_rolls_back_partial_write_and_keeps_previous(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        h.run()
        original_chunks = h.stored(pdf)

        h2 = Harness(embed_model=_ExplodingEmbedding(embed_dim=8))
        pdf2 = h2.add("guide.pdf", [BEE])
        summary = h2.run()
        self.assertEqual(h2.entry(pdf2)["status"], ingest_books.STATUS_FAILED)
        self.assertEqual(h2.entry(pdf2)["category"], ingest_books.CAT_EMBEDDING_FAILED)
        self.assertEqual(h2.stored(pdf2), 0)  # partial write rolled back
        self.assert_invariants(h2, summary)

        # Now change the file so a previous version exists, and fail again.
        h.touch(pdf, [
            "Field Guide to Urban Beekeeping",
            "A revised edition with completely new sentences about propolis here.",
        ])
        with mock.patch.object(ingest_books, "_build_index") as build:
            store = h.collection()
            index = mock.MagicMock()
            index.insert_nodes.side_effect = RuntimeError("embedding OOM")
            build.return_value = (index, store)
            summary2 = h.run()
        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_FAILED)
        self.assertEqual(h.stored(pdf), original_chunks)  # previous version intact
        self.assertIn("previous indexed version", entry["message"])
        self.assert_invariants(h, summary2)

    def test_changed_file_replaces_its_own_stale_chunks(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        h.run()
        old_chunks = h.stored(pdf)

        h.touch(pdf, [
            "Field Guide to Urban Beekeeping",
            "Propolis is a resinous substance bees collect from plant buds.",
        ])
        summary = h.run()

        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_INDEXED)
        self.assertEqual(entry.get("replaced_chunks"), old_chunks)
        self.assertEqual(h.stored(pdf), entry["chunks"])  # no duplicates left behind
        self.assertIn("Propolis", h.stored_text(pdf))
        self.assert_invariants(h, summary)


# --------------------------------------------------------------------------
# 14-16: manifest compatibility
# --------------------------------------------------------------------------


class TestManifestCompatibility(IngestTestCase):
    def _seed_legacy(self, h: Harness, pdf: Path, mtime_ns: int) -> None:
        h.manifest_path.write_text(
            json.dumps({str(pdf): mtime_ns}, indent=2), encoding="utf-8"
        )

    def test_legacy_entry_with_stored_vectors_is_migrated_and_verified(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        h.run()  # creates the collection and real chunks
        stored = h.stored(pdf)

        self._seed_legacy(h, pdf, os.stat(pdf).st_mtime_ns)
        summary = h.run()

        entry = h.entry(pdf)
        self.assertEqual(summary["manifest_path"], str(h.manifest_path))
        self.assertEqual(entry["status"], ingest_books.STATUS_INDEXED)
        self.assertTrue(entry.get("legacy"))
        self.assertEqual(entry["chunks"], stored)  # adopted from the store, verified
        self.assertEqual(summary["counts"][ingest_books.OUTCOME_UNCHANGED], 1)
        self.assertEqual(h.stored(pdf), stored)  # no re-embedding
        self.assertTrue(any("legacy" in n for n in summary["notes"]))

        # Schema upgrade is a one-time event, not drift on every run.
        manifest_after = h.manifest_path.read_bytes()
        h.run()
        self.assertEqual(h.manifest_path.read_bytes(), manifest_after)
        self.assert_invariants(h, summary)

    def test_legacy_entry_without_stored_vectors_is_repaired_not_trusted(self):
        """The silent-loss case: v1 recorded the file as done, the store holds
        nothing. Trusting the legacy record would keep the document invisible."""
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        h.collection()  # create the empty collection
        self._seed_legacy(h, pdf, os.stat(pdf).st_mtime_ns)

        summary = h.run()
        entry = h.entry(pdf)
        self.assertEqual(entry["status"], ingest_books.STATUS_INDEXED)
        self.assertGreater(entry["chunks"], 0)
        self.assertEqual(h.stored(pdf), entry["chunks"])
        self.assert_invariants(h, summary)

    def test_corrupt_manifest_is_treated_as_empty_and_reported(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        h.manifest_path.write_text("{not json at all", encoding="utf-8")
        summary = h.run()

        self.assertTrue(any("unreadable" in n for n in summary["notes"]))
        self.assertEqual(h.entry(pdf)["status"], ingest_books.STATUS_INDEXED)
        self.assert_invariants(h, summary)

    def test_manifest_tracks_files_missing_from_disk_without_pruning_them(self):
        h = Harness()
        pdf = h.add("guide.pdf", [BEE])
        h.run()
        key = str(pdf)
        pdf.unlink()

        summary = h.run()
        self.assertEqual(summary["files"], {})
        self.assertEqual(summary.get("missing_on_disk"), [key])
        self.assertIn(key, h.manifest()["files"])  # reported, never silently dropped


# --------------------------------------------------------------------------
# the diagnostic itself: it must report problems, not produce them
# --------------------------------------------------------------------------


class TestCheckChromaDiagnostic(unittest.TestCase):
    """check_chroma stays read-only and honest about what it cannot verify."""

    def setUp(self):
        import chromadb

        import check_chroma

        self.check_chroma = check_chroma
        self._chromadb = chromadb
        self.tmp = Path(tempfile.mkdtemp(prefix="check_chroma_"))
        self.storage = self.tmp / "storage"
        self.storage.mkdir()
        self.manifest_path = self.storage / ".index_manifest.json"
        chromadb.PersistentClient(path=str(self.storage)).get_or_create_collection(COLLECTION)
        self._patches = [
            mock.patch.object(check_chroma, "STORAGE_DIR", self.storage),
            mock.patch.object(check_chroma, "INDEX_MANIFEST", self.manifest_path),
            mock.patch.object(check_chroma, "CHROMA_COLLECTION", COLLECTION),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _add_node(self, path: Path, text: str = "Some extracted prose about beekeeping.") -> None:
        client = self._chromadb.PersistentClient(path=str(self.storage))
        client.get_collection(COLLECTION).add(
            ids=[f"node-{abs(hash(str(path)))}"],
            embeddings=[[0.1] * 8],
            documents=[text],
            metadatas=[{"file_path": str(path), "file_name": path.name}],
        )

    def _write_manifest(self, files: dict) -> None:
        self.manifest_path.write_text(
            json.dumps({"version": 2, "files": files}, indent=2), encoding="utf-8"
        )

    def test_consistent_store_reports_zero_errors(self):
        pdf = self.tmp / "guide.pdf"
        pdf.write_text("x", encoding="utf-8")
        self._add_node(pdf)
        self._write_manifest({str(pdf): {
            "status": ingest_books.STATUS_INDEXED, "chunks": 1,
            "mtime_ns": os.stat(pdf).st_mtime_ns, "empty_pages": [], "unit": "page",
        }})
        self.assertEqual(self.check_chroma.main(), 0)

    def test_indexed_record_with_no_stored_vectors_is_an_error(self):
        pdf = self.tmp / "gone.pdf"
        pdf.write_text("x", encoding="utf-8")
        self._write_manifest({str(pdf): {
            "status": ingest_books.STATUS_INDEXED, "chunks": 3,
            "mtime_ns": os.stat(pdf).st_mtime_ns, "empty_pages": [], "unit": "page",
        }})
        self.assertEqual(self.check_chroma.main(), 1)  # the silently-lost case

    def test_failed_file_with_surviving_previous_version_is_a_warning(self):
        pdf = self.tmp / "guide.pdf"
        pdf.write_text("x", encoding="utf-8")
        self._add_node(pdf)
        self._write_manifest({str(pdf): {
            "status": ingest_books.STATUS_FAILED, "chunks": 0,
            "category": ingest_books.CAT_UNREADABLE, "message": "reader returned nothing",
            "mtime_ns": os.stat(pdf).st_mtime_ns, "empty_pages": [], "unit": "page",
            "previous": {"chunks": 1, "status": ingest_books.STATUS_INDEXED},
        }})
        # The old version is still served: reported, but not an integrity error.
        self.assertEqual(self.check_chroma.main(), 0)

    def test_partial_record_is_reported_with_its_unextractable_pages(self):
        pdf = self.tmp / "mixed.pdf"
        pdf.write_text("x", encoding="utf-8")
        self._add_node(pdf)
        self._write_manifest({str(pdf): {
            "status": ingest_books.STATUS_PARTIAL, "chunks": 1,
            "empty_pages": ["page 2"], "category": ingest_books.CAT_PARTIAL_TEXT,
            "message": "1 of 2 pages had no extractable text",
            "mtime_ns": os.stat(pdf).st_mtime_ns, "unit": "page",
        }})
        self.assertEqual(self.check_chroma.main(), 0)

    def test_empty_collection_fails_the_check(self):
        self.assertEqual(self.check_chroma.main(), 1)

    def test_legacy_manifest_is_reported_as_unverified(self):
        pdf = self.tmp / "guide.pdf"
        pdf.write_text("x", encoding="utf-8")
        self._add_node(pdf)
        self.manifest_path.write_text(
            json.dumps({str(pdf): os.stat(pdf).st_mtime_ns}), encoding="utf-8"
        )
        self.assertEqual(self.check_chroma.main(), 0)  # warning, not an error


if __name__ == "__main__":
    unittest.main(verbosity=2)
