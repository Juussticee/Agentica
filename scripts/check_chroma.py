"""Storage check: persisted Chroma collection + ingest manifest integrity.

Read-only diagnostic (M2). It verifies that what the manifest claims matches
what the vector store actually holds, and it is explicit about what it CANNOT
verify — per-file chunk counts come from the `file_path` metadata the reader
wrote on every node, so a comparison is only made when that metadata exists;
when it does not, the check is reported as unavailable instead of guessed.

Run:  python scripts/check_chroma.py
Exits 0 when the store is non-empty, its sample text is real prose, and no
integrity error was found; 1 otherwise. It never repairs, deletes, reindexes
or mutates the knowledge base.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHROMA_COLLECTION, INDEX_MANIFEST, STORAGE_DIR  # noqa: E402
from app.chroma_guard import assert_store_path_allowed  # noqa: E402
from ingest_books import STATUS_FAILED, STATUS_INDEXED, STATUS_PARTIAL  # noqa: E402


def _manifest_entries() -> "tuple[dict, list]":
    """Raw manifest entries (no migration) + notes about their shape."""
    notes = []
    if not INDEX_MANIFEST.exists():
        return {}, ["manifest missing"]
    try:
        raw = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"manifest unreadable ({type(exc).__name__}: {exc})"]
    if not isinstance(raw, dict):
        return {}, ["manifest has an unexpected shape"]
    files = raw.get("files")
    if isinstance(files, dict):
        return files, notes
    # legacy v1: flat {path: mtime_ns}
    notes.append(
        "manifest is schema v1 (flat path->mtime): per-file status and chunk "
        "counts were never recorded; mark-and-sweep below is limited to "
        "existence of stored vectors"
    )
    return {k: {"mtime_ns": v, "status": STATUS_INDEXED, "chunks": None,
                "legacy": True} for k, v in raw.items() if isinstance(v, int)}, notes


def _stored_count(col, path_key: str) -> int:
    return len(col.get(where={"file_path": path_key}, include=[])["ids"] or [])


def main() -> int:
    import chromadb

    # Read-only diagnostic, but still fail BEFORE any open when the guard is
    # armed (a manual run is unarmed and behaves exactly as before).
    assert_store_path_allowed(STORAGE_DIR)
    errors, warnings = [], []
    client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    print("collections:", [c.name for c in client.list_collections()])
    col = client.get_collection(CHROMA_COLLECTION)
    count = col.count()
    print("persisted chunks:", count)

    # --- binary-garbage sample check (existing behavior, kept) --------------
    doc = ""
    if count:
        sample = col.get(limit=1, include=["documents"])
        doc = (sample["documents"] or [""])[0] or ""
        print("first doc snippet:", repr(doc[:200]))
    else:
        print("NO DOCUMENTS — run: python ingest_books.py")

    # --- manifest vs store --------------------------------------------------
    entries, notes = _manifest_entries()
    for note in notes:
        warnings.append(note)
        print(f"manifest note: {note}")

    by_status = {}
    for key, entry in entries.items():
        status = entry.get("status") or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
    print("manifest files by status:", dict(sorted(by_status.items())))

    if entries:
        print(f"manifest entries: {len(entries)}")
        for key, entry in sorted(entries.items()):
            name = Path(key).name
            status = entry.get("status")
            stored = _stored_count(col, key)
            recorded = entry.get("chunks")
            if not Path(key).exists():
                warnings.append(f"{name}: manifest entry has no file on disk (not pruned)")
                print(f"  missing-on-disk  {name} (status={status}, stored chunks={stored})")
                continue
            if status == STATUS_INDEXED:
                if recorded == 0:
                    errors.append(f"{name}: recorded as indexed with 0 chunks")
                    print(f"  ERROR            {name}: indexed with 0 chunks")
                elif stored == 0:
                    errors.append(f"{name}: indexed but the vector store has no content for it")
                    print(f"  ERROR            {name}: indexed, but no stored vectors "
                          f"(chunks recorded={recorded}) — run: python ingest_books.py")
                elif recorded is None:
                    warnings.append(f"{name}: legacy record, chunk count unverified "
                                    f"(store holds {stored})")
                    print(f"  legacy           {name}: chunk count unverified; store holds {stored}")
                elif stored != recorded:
                    errors.append(f"{name}: manifest says {recorded} chunks, store holds {stored}")
                    print(f"  ERROR            {name}: manifest {recorded} chunk(s) != store {stored}")
                else:
                    print(f"  ok               {name}: {stored} chunk(s) verified")
            elif status == STATUS_PARTIAL:
                missing = entry.get("empty_pages") or []
                print(f"  partial          {name}: {stored} chunk(s) stored; "
                      f"{len(missing)} {entry.get('unit', 'page')}(s) without extractable text "
                      f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
                if stored == 0:
                    errors.append(f"{name}: recorded partial but nothing is stored for it")
            elif status == STATUS_FAILED:
                prev = entry.get("previous") or {}
                print(f"  failed           {name}: {entry.get('category')} — {entry.get('message')}")
                if stored:
                    if prev:
                        warnings.append(
                            f"{name}: failed re-ingest; the previous version "
                            f"({prev.get('chunks')} chunk(s)) is still served"
                        )
                        print(f"                   previous version ({prev.get('chunks')} "
                              f"chunk(s)) still in the store and still searchable")
                    else:
                        errors.append(
                            f"{name}: recorded failed but {stored} chunk(s) are stored for it"
                        )
            else:
                warnings.append(f"{name}: unknown status {status!r}")
                print(f"  unknown status   {name}: {status!r}")

        # Files present in the store but absent from the manifest. Attribution
        # needs the reader-written `file_path` metadata; without it the check is
        # UNAVAILABLE, and saying so beats reporting a wrong "missing" list.
        try:
            metas = col.get(include=["metadatas"])["metadatas"] or []
            stored_paths = {m.get("file_path") for m in metas if m and m.get("file_path")}
            note = None
        except Exception as exc:
            stored_paths = set()
            note = f"per-file attribution unavailable ({type(exc).__name__})"
        if not note and not stored_paths:
            note = ("no stored node carries `file_path` metadata, so per-file "
                    "attribution is unavailable")
        if note:
            warnings.append(f"{note}; index/manifest match could not be checked")
            print(f"  UNAVAILABLE      {note}")
        untracked = sorted(p for p in stored_paths if p not in entries)
        for p in untracked:
            warnings.append(f"{Path(p).name}: stored vectors with no manifest entry")
            print(f"  WARNING          {Path(p).name}: vectors stored but not in the manifest")
        # Manifest entries that claim content but whose nodes have no file_path.
        if stored_paths:
            for key, entry in entries.items():
                if entry.get("status") in (STATUS_INDEXED, STATUS_PARTIAL) \
                        and key not in stored_paths:
                    warnings.append(
                        f"{Path(key).name}: recorded {entry.get('status')} but no stored "
                        "node carries its file_path"
                    )
    else:
        warnings.append("manifest has no usable entries: index/manifest match unchecked")

    if doc:
        nul = doc.count("\x00")
        printable = sum(1 for ch in doc if ch.isprintable() or ch.isspace()) / max(len(doc), 1)
        print("null bytes in sample:", nul)
        print(f"printable ratio: {printable:.2%}")
        if nul or printable < 0.9:
            errors.append("chunk text looks like binary garbage")
            print("WARNING: chunk text looks like binary garbage. Re-ingest after "
                  "installing: pip install llama-index-readers-file docx2txt pypdf")

    print(f"\nintegrity: {len(errors)} error(s), {len(warnings)} warning(s)")
    for e in errors:
        print(f"  ERROR   {e}")
    for w in warnings:
        print(f"  WARNING {w}")
    if not count:
        print("RESULT: no documents in the collection — run: python ingest_books.py")
        return 1
    if errors:
        print("RESULT: integrity problems found (nothing was modified)")
        return 1
    print("RESULT: collection healthy and manifest consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
