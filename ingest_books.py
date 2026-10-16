"""Standalone ingest script (not part of the graph) — spec Section 3 + M2.

Reads knowledge/books/ recursively (.pdf/.docx), builds a LlamaIndex
VectorStoreIndex over a persisted ChromaDB collection in storage/, and tracks
per-file outcome in storage/.index_manifest.json.

Embeddings: Ollama nomic-embed-text (known gap — Gemini embeddings not set up).

M2 — ingestion integrity
------------------------
Every candidate file is processed ON ITS OWN and gets an honest status —
``indexed`` / ``partial`` / ``failed`` — backed by the number of chunks that
are actually persisted in the vector store, never by "the reader did not
raise".  Why this exists (measured, not assumed): a corrupt PDF makes
``SimpleDirectoryReader`` log ``Failed to load file ... Skipping...`` and
return ZERO documents *without raising*, and the previous manifest still
recorded the file as indexed — a permanently invisible document in the
knowledge base.

Invariants enforced here:

* a file is NEVER recorded as ``indexed`` with zero persisted chunks;
* a file with some extractable pages and some without is ``partial``, and the
  unextractable pages are named BY PAGE. They are never called "blank": a page
  with no text layer may be a scan or an unsupported layout, and this build has
  no OCR, so "no extractable text" is the honest statement;
* a previous valid record is never overwritten with a false success;
* replacement is insert-then-delete, so a failed re-process of a CHANGED file
  leaves the previous vectors in place and says so in the manifest;
* re-running is idempotent: unchanged healthy files are skipped without
  re-embedding and without manifest drift.

Known limits (deliberate, not hidden): the page cap applies to PDF only
(.docx has no page concept); a scanned page is reported, never OCR'd; and
"no extractable text" is a statement about the text layer, not about the page.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import (
    BOOKS_DIR,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    INDEX_MANIFEST,
    INGEST_EXTENSIONS,
    INGEST_MANIFEST_VERSION,
    INGEST_MAX_FILE_MB,
    INGEST_MAX_PAGES,
    OLLAMA_BASE_URL,
    STORAGE_DIR,
)

# --- statuses (also read by scripts/check_chroma.py) ------------------------
STATUS_INDEXED = "indexed"    # extraction + indexing succeeded, chunks > 0
STATUS_PARTIAL = "partial"    # usable content indexed, but some pages did not
STATUS_FAILED = "failed"      # nothing usable was persisted for this file

# Operational result for a file that did not need reprocessing. It is reported
# in the run summary ONLY and is never written into the entry: the stored
# status and `last_outcome` keep describing the last REAL processing, so a
# re-run of an unchanged corpus rewrites the manifest byte-for-byte.
OUTCOME_UNCHANGED = "unchanged"

# Failure/warning categories (concise, machine-readable, attributable).
CAT_OVERSIZED = "oversized"
CAT_PAGE_LIMIT = "page_limit_exceeded"
CAT_ENCRYPTED = "encrypted"
CAT_UNREADABLE = "unreadable"
CAT_NO_TEXT = "no_extractable_text"
CAT_PARTIAL_TEXT = "partial_extractable_text"
CAT_BINARY_GARBAGE = "binary_garbage"
CAT_EMBEDDING_FAILED = "embedding_failed"
CAT_EMBEDDING_INCOMPLETE = "embedding_incomplete"

_BINARY_HINT = (
    "Install the file readers: pip install llama-index-readers-file docx2txt pypdf"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mtime_ns(path) -> int:
    return os.stat(path).st_mtime_ns


# --- manifest --------------------------------------------------------------


def load_manifest(path: Path) -> "tuple[dict, list]":
    """Return ``(manifest, notes)`` for the v2 manifest at ``path``.

    Accepts the legacy v1 flat ``{path: mtime_ns}`` map and migrates it IN
    MEMORY: those entries become ``status="indexed"`` with ``chunks=None`` and
    ``legacy=True`` — i.e. "was recorded by the old code as done, chunk count
    never verified".  They are never trusted blindly: the caller re-verifies
    every legacy entry against the vector store before skipping it.
    """
    notes = []
    if not path.exists():
        return {"version": INGEST_MANIFEST_VERSION, "files": {}}, notes

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # corrupt manifest must not block ingestion
        notes.append(
            f"manifest unreadable ({type(exc).__name__}: {exc}) — treated as "
            "empty; every discovered file will be re-processed"
        )
        return {"version": INGEST_MANIFEST_VERSION, "files": {}}, notes

    if not isinstance(raw, dict):
        notes.append("manifest has an unexpected shape — treated as empty")
        return {"version": INGEST_MANIFEST_VERSION, "files": {}}, notes

    files = raw.get("files")
    if isinstance(files, dict):
        migrated = {k: dict(v) for k, v in files.items() if isinstance(v, dict)}
        unknown = len(files) - len(migrated)
        if unknown:
            notes.append(f"{unknown} manifest entry/entries had an unexpected shape and were dropped")
        if raw.get("version") != INGEST_MANIFEST_VERSION:
            notes.append(
                f"manifest version {raw.get('version')!r} migrated to "
                f"{INGEST_MANIFEST_VERSION}"
            )
        return {"version": INGEST_MANIFEST_VERSION, "files": migrated}, notes

    # legacy v1: flat {absolute_path: mtime_ns}
    legacy = {}
    for key, value in raw.items():
        if isinstance(value, int):
            legacy[key] = {
                "mtime_ns": value,
                "status": STATUS_INDEXED,
                "chunks": None,
                "legacy": True,
                "documents": None,
                "nonempty_pages": None,
                "empty_pages": [],
                "size_bytes": None,
                "category": None,
                "message": (
                    "migrated from manifest v1: recorded as done by the previous "
                    "ingest, which could not see a silently skipped file; chunk "
                    "count unverified"
                ),
                "last_outcome": None,
                "last_processed_at": None,
            }
        else:
            notes.append(f"legacy manifest entry {key!r} ignored (unexpected value)")
    if legacy:
        notes.append(
            f"{len(legacy)} legacy manifest entr{'y' if len(legacy) == 1 else 'ies'} "
            "migrated to schema v2 (chunk counts unverified)"
        )
    return {"version": INGEST_MANIFEST_VERSION, "files": legacy}, notes


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


# --- discovery / preflight / extraction ------------------------------------


def discover_files(books_dir: Path, extensions=INGEST_EXTENSIONS) -> list:
    exts = {e.lower() for e in extensions}
    return sorted(
        (p for p in books_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()),
        key=lambda p: str(p).lower(),
    )


def _pdf_preflight(path: Path, max_pages: int):
    """Cheap PDF checks before any text is decoded: encryption + page count.

    pypdf (already required by llama-index-readers-file) reads only the
    trailer/xref here, so an oversized document is rejected before the reader
    loads it. Returns ``(category, message)`` when the file must fail, else
    ``(None, None)``. Page count is None when the cap is disabled.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return (
                CAT_ENCRYPTED,
                "PDF is password-protected; encrypted documents are not supported "
                "(no password is guessed, no content is extracted)",
            )
        pages = len(reader.pages)
    except Exception as exc:  # unreadable container: the reader cannot do better
        return (
            CAT_UNREADABLE,
            f"PDF could not be opened for preflight ({type(exc).__name__}: "
            f"{str(exc)[:160]})",
        )
    if max_pages > 0 and pages > max_pages:
        return (
            CAT_PAGE_LIMIT,
            f"{pages} pages exceeds INGEST_MAX_PAGES={max_pages}; rejected before "
            "extraction (raise INGEST_MAX_PAGES to ingest it)",
        )
    return None, None


def _load_documents(path: Path) -> list:
    """Run the real production reader for ONE file (patched in tests)."""
    return SimpleDirectoryReader(input_files=[str(path)]).load_data()


def _garbage_reason(text: str) -> "str | None":
    """True-prose check (existing behavior): catches a reader that fell back to
    raw bytes when llama-index-readers-file is missing.

    Empty/whitespace-only text is NOT garbage — it is the no-text-layer page
    case, which is handled separately (assumed blank and OCR'd would both be
    lies, so it is reported as "no extractable text").
    """
    if not text.strip():
        return None
    nul = text.count("\x00")
    nonprint = sum(1 for ch in text if not (ch.isprintable() or ch.isspace()))
    letters = sum(1 for ch in text if ch.isalpha())
    binary_sig = text.lstrip().startswith(("PK", "%PDF-"))
    if nul or binary_sig or nonprint / len(text) > 0.05 or letters / len(text) < 0.5:
        return (
            f"parsed as binary data, not text (nul={nul}, magic={binary_sig}, "
            f"nonprintable={nonprint * 100 // len(text)}%, "
            f"letters={letters * 100 // len(text)}%)"
        )
    return None


def _store_ids(collection, path: Path) -> set:
    """Node ids currently persisted for this exact file path."""
    got = collection.get(where={"file_path": str(path)}, include=[])
    return set(got.get("ids") or [])


def _blocked_entry(path, size_bytes, category, message):
    return {
        "mtime_ns": _mtime_ns(path),
        "size_bytes": size_bytes,
        "status": STATUS_FAILED,
        "documents": None,
        "nonempty_pages": None,
        "empty_pages": [],
        "unit": "file",
        "chunks": 0,
        "category": category,
        "message": message,
        "last_outcome": STATUS_FAILED,
        "last_processed_at": _utc_now(),
    }


# --- the per-file pipeline -------------------------------------------------


def process_file(path: Path, index, collection, *, max_file_mb: int, max_pages: int,
                 old_entry: "dict | None" = None, verbose: bool = True) -> dict:
    """Extract, chunk and persist ONE file; return its honest manifest entry.

    Order matters: preflight (no decoding) -> extract -> chunk -> INSERT then
    delete the stale previous version. The previous version's vectors are only
    removed after the new chunks are verified present, so a failed replacement
    never loses content.
    """
    size_bytes = os.stat(path).st_size
    max_bytes = max_file_mb * 1024 * 1024
    if max_file_mb > 0 and size_bytes > max_bytes:
        return _blocked_entry(
            path, size_bytes, CAT_OVERSIZED,
            f"{size_bytes / 1048576:.2f} MB exceeds INGEST_MAX_FILE_MB={max_file_mb}; "
            "rejected before the document reader was called",
        )

    unit = "page" if path.suffix.lower() == ".pdf" else "file"
    if unit == "page":
        category, message = _pdf_preflight(path, max_pages)
        if category:
            entry = _blocked_entry(path, size_bytes, category, message)
            entry["unit"] = unit
            return entry

    try:
        documents = _load_documents(path)
    except Exception as exc:
        entry = _blocked_entry(
            path, size_bytes, CAT_UNREADABLE,
            f"reader raised {type(exc).__name__}: {str(exc)[:200]}",
        )
        entry["unit"] = unit
        return entry

    if not documents:
        # Measured behavior: the reader logs "Skipping..." for a corrupt file
        # and returns an EMPTY list without raising — the old code marked that
        # file indexed. This is the defect M2 closes.
        entry = _blocked_entry(
            path, size_bytes, CAT_UNREADABLE,
            "the document reader returned no documents for this file (corrupt, "
            "empty, or an unsupported container); it is NOT indexed",
        )
        entry["unit"] = unit
        return entry

    texts = [(doc.get_content() or "") for doc in documents]
    nonempty_idx = [i for i, text in enumerate(texts) if text.strip()]
    empty_pages = []
    for i, text in enumerate(texts):
        if text.strip():
            continue
        label = (documents[i].metadata or {}).get("page_label")
        empty_pages.append(f"page {label}" if label else f"segment {i + 1}")

    bad = next((r for r in (_garbage_reason(t) for t in texts) if r), None)
    if bad:
        entry = _blocked_entry(path, size_bytes, CAT_BINARY_GARBAGE, bad + " — " + _BINARY_HINT)
        entry["unit"] = unit
        entry["documents"] = len(documents)
        entry["nonempty_pages"] = len(nonempty_idx)
        entry["empty_pages"] = empty_pages
        return entry

    if not nonempty_idx:
        entry = _blocked_entry(
            path, size_bytes, CAT_NO_TEXT,
            f"no extractable text on any of the {len(documents)} extracted "
            f"{unit}(s). The pages are NOT assumed blank: they may be scans, "
            "images, or unsupported layouts, and this build has no OCR",
        )
        entry["unit"] = unit
        entry["documents"] = len(documents)
        entry["nonempty_pages"] = 0
        entry["empty_pages"] = empty_pages
        return entry

    nodes = [
        n for n in Settings.text_splitter.get_nodes_from_documents(
            [documents[i] for i in nonempty_idx]
        )
        if (n.text or "").strip()
    ]
    if not nodes:
        entry = _blocked_entry(
            path, size_bytes, CAT_NO_TEXT,
            "extracted text produced no chunks after splitting; nothing was indexed",
        )
        entry["unit"] = unit
        entry["documents"] = len(documents)
        entry["nonempty_pages"] = len(nonempty_idx)
        entry["empty_pages"] = empty_pages
        return entry

    # Always read the store (not just when a manifest entry exists): vectors can
    # outlive a lost/corrupt manifest, and leaving them would duplicate content.
    old_ids = _store_ids(collection, path)
    new_ids = {n.node_id for n in nodes}

    try:
        index.insert_nodes(nodes)
    except Exception as exc:
        persisted = set(collection.get(ids=sorted(new_ids), include=[])["ids"] or [])
        if persisted:                     # roll the partial write back
            collection.delete(ids=sorted(persisted))
        entry = _blocked_entry(
            path, size_bytes, CAT_EMBEDDING_FAILED,
            f"embedding/indexing failed ({type(exc).__name__}: {str(exc)[:200]}); "
            f"{len(persisted)} partial chunk(s) removed"
            + (
                f"; the previous indexed version ({len(old_ids)} chunk(s)) was left "
                "in place" if old_ids else ""
            ),
        )
        entry["unit"] = unit
        entry["documents"] = len(documents)
        entry["nonempty_pages"] = len(nonempty_idx)
        entry["empty_pages"] = empty_pages
        return entry

    persisted = set(collection.get(ids=sorted(new_ids), include=[])["ids"] or [])
    if len(persisted) != len(new_ids):
        if persisted:
            collection.delete(ids=sorted(persisted))
        entry = _blocked_entry(
            path, size_bytes, CAT_EMBEDDING_INCOMPLETE,
            f"only {len(persisted)} of {len(new_ids)} chunk(s) were persisted; "
            "partial write removed"
            + (
                f"; the previous indexed version ({len(old_ids)} chunk(s)) was left "
                "in place" if old_ids else ""
            ),
        )
        entry["unit"] = unit
        entry["documents"] = len(documents)
        entry["nonempty_pages"] = len(nonempty_idx)
        entry["empty_pages"] = empty_pages
        return entry

    # New chunks verified in the store -> only now drop the stale version.
    stale = old_ids - new_ids
    if stale:
        collection.delete(ids=sorted(stale))

    status = STATUS_PARTIAL if empty_pages else STATUS_INDEXED
    entry = {
        "mtime_ns": _mtime_ns(path),
        "size_bytes": size_bytes,
        "status": status,
        "documents": len(documents),
        "nonempty_pages": len(nonempty_idx),
        "empty_pages": empty_pages,
        "unit": unit,
        "chunks": len(new_ids),
        "category": CAT_PARTIAL_TEXT if empty_pages else None,
        "message": (
            f"{len(empty_pages)} of {len(documents)} extracted {unit}(s) yielded no "
            "extractable text and were NOT indexed (reported, not assumed blank; "
            "no OCR in this build)"
            if empty_pages else None
        ),
        "last_outcome": status,
        "last_processed_at": _utc_now(),
    }
    if stale:
        entry["replaced_chunks"] = len(stale)
    if verbose:
        print(
            f"  {path.name}: {status} — {len(documents)} {unit}(s), "
            f"{len(nonempty_idx)} with text, {len(new_ids)} chunk(s) persisted"
            + (f", {len(stale)} stale chunk(s) replaced" if stale else "")
        )
    return entry


def _skip(entry: "dict | None", store_count: int) -> bool:
    """True when an unchanged file can be skipped without re-embedding.

    Failures are ALWAYS retried (a transient embedding outage must not become
    permanent), and so are records whose stored chunk count disagrees with the
    vector store — that mismatch is exactly the silently-lost-document case.
    Legacy v1 records (chunks unknown) may only be skipped when the store
    really holds content for them.
    """
    if not entry:
        return False
    if entry.get("status") == STATUS_FAILED:
        return False
    recorded = entry.get("chunks")
    if recorded is None:
        return store_count > 0
    return store_count == recorded


def run_ingest(books_dir=BOOKS_DIR, storage_dir=STORAGE_DIR,
               manifest_path=INDEX_MANIFEST, collection_name=CHROMA_COLLECTION,
               embed_model=None, max_file_mb=None, max_pages=None,
               verbose=True) -> dict:
    """Ingest every candidate file once; return a truthful run summary.

    Injectable on purpose (dirs, manifest, collection, embed model, caps) so the
    integrity tests run offline against temporary storage with MockEmbedding.
    """
    books_dir = Path(books_dir)
    storage_dir = Path(storage_dir)
    manifest_path = Path(manifest_path)
    max_file_mb = INGEST_MAX_FILE_MB if max_file_mb is None else max_file_mb
    max_pages = INGEST_MAX_PAGES if max_pages is None else max_pages

    manifest, notes = load_manifest(manifest_path)
    files = manifest["files"]
    for note in notes:
        print(f"[ingest] {note}")

    if embed_model is None:
        print(f"Embedding with {EMBED_MODEL} via Ollama at {OLLAMA_BASE_URL} ...")
        embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    index, collection = _build_index(storage_dir, collection_name, embed_model)

    candidates = discover_files(books_dir)
    candidate_keys = {str(p) for p in candidates}
    # Unsupported/unrelated files are reported, not silently invisible: the
    # manifest only ever describes documents, so a skipped file must still be
    # visible somewhere in the run's output.
    ignored = sorted(
        str(p) for p in books_dir.rglob("*")
        if p.is_file() and str(p) not in candidate_keys
    )
    if verbose and ignored:
        print(
            f"  {len(ignored)} file(s) ignored (not "
            f"{'/'.join(INGEST_EXTENSIONS)}): "
            + ", ".join(Path(p).name for p in ignored)
        )
    summary = {
        "ignored": ignored,
        "manifest_path": str(manifest_path),
        "collection": collection_name,
        "stored_chunks": collection.count(),
        "counts": {STATUS_INDEXED: 0, STATUS_PARTIAL: 0, STATUS_FAILED: 0,
                   OUTCOME_UNCHANGED: 0},
        "files": {},
        "notes": notes,
    }
    on_disk = {str(p) for p in candidates}
    for stale_key in sorted(k for k in files if k not in on_disk):
        summary.setdefault("missing_on_disk", []).append(stale_key)

    for path in candidates:
        key = str(path)
        entry = files.get(key)
        store_count = len(_store_ids(collection, path))
        unchanged = bool(entry) and entry.get("mtime_ns") == _mtime_ns(path)

        if unchanged and _skip(entry, store_count):
            if entry.get("chunks") is None:
                # Legacy record verified against the vector store: adopt the
                # measured count instead of trusting the old "indexed" claim.
                # One-time migration — later runs are pure skips.
                files[key] = {
                    **entry,
                    "chunks": store_count,
                    "status": (STATUS_PARTIAL if entry.get("empty_pages")
                               else STATUS_INDEXED),
                    "message": "manifest v1 entry: chunk count verified against "
                               "the vector store",
                }
            summary["files"][key] = files[key]
            summary["counts"][OUTCOME_UNCHANGED] += 1
            if verbose:
                print(f"  {path.name}: unchanged ({store_count} chunk(s) already indexed)")
            continue

        if unchanged and entry and entry.get("chunks") is not None \
                and store_count != entry.get("chunks"):
            print(
                f"[ingest] {path.name}: manifest records {entry.get('chunks')} chunk(s) "
                f"but the vector store holds {store_count} — re-processing"
            )

        new_entry = process_file(
            path, index, collection,
            max_file_mb=max_file_mb, max_pages=max_pages,
            old_entry=entry, verbose=verbose,
        )
        if new_entry["status"] == STATUS_FAILED and entry and entry.get("status") != STATUS_FAILED:
            # Never lose the fact that an older version is still being served.
            new_entry["previous"] = {
                "mtime_ns": entry.get("mtime_ns"),
                "status": entry.get("status"),
                "chunks": entry.get("chunks"),
                "last_processed_at": entry.get("last_processed_at"),
            }
        files[key] = new_entry
        summary["files"][key] = new_entry
        summary["counts"][new_entry["status"]] += 1
        save_manifest(manifest_path, manifest)  # persist per file: a crash keeps truth

    save_manifest(manifest_path, manifest)
    summary["stored_chunks"] = collection.count()
    return summary


def _build_index(storage_dir, collection_name, embed_model):
    """Persisted index + raw Chroma collection (for verified per-file counts)."""
    import chromadb

    Settings.embed_model = embed_model
    client = chromadb.PersistentClient(path=str(storage_dir))
    collection = client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        store, storage_context=StorageContext.from_defaults(vector_store=store)
    )
    return index, collection


def main() -> int:
    started = time.perf_counter()
    summary = run_ingest()
    counts = summary["counts"]
    print(
        f"\n{sum(counts.values())} file(s): {counts[STATUS_INDEXED]} indexed, "
        f"{counts[STATUS_PARTIAL]} partial, {counts[STATUS_FAILED]} failed, "
        f"{counts[OUTCOME_UNCHANGED]} unchanged | chunk(s) in store: "
        f"{summary['stored_chunks']}"
    )
    if summary.get("ignored"):
        print(
            "  ignored (not a supported document type, never ingested): "
            + ", ".join(Path(k).name for k in summary["ignored"])
        )
    for key, entry in sorted(summary["files"].items()):
        if entry["status"] == STATUS_FAILED:
            print(f"  FAILED {Path(key).name}: {entry['category']} — {entry['message']}")
        elif entry["status"] == STATUS_PARTIAL and entry.get("empty_pages"):
            print(f"  PARTIAL {Path(key).name}: {entry['message']}")
    if summary.get("missing_on_disk"):
        print(
            "  manifest entries with no file on disk (kept, not pruned): "
            + ", ".join(Path(k).name for k in summary["missing_on_disk"])
        )
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Finished in {time.perf_counter() - started:.1f}s")
    if counts[STATUS_FAILED]:
        print("\nAt least one file FAILED: it is not searchable. See the lines above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
