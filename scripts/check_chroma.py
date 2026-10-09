"""Storage check: inspects the persisted Chroma collection and ingest manifest.

Verifies the index is non-empty and chunk text is real prose, not binary
garbage (the silent failure mode when llama-index-readers-file is missing).

Run:  python scripts/check_chroma.py
Exits 0 when the collection holds clean text, 1 when empty or corrupted.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHROMA_COLLECTION, INDEX_MANIFEST, STORAGE_DIR  # noqa: E402


def main() -> int:
    import chromadb

    client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    print("collections:", [c.name for c in client.list_collections()])
    col = client.get_collection(CHROMA_COLLECTION)
    count = col.count()
    print("count:", count)

    doc = ""
    if count:
        sample = col.get(limit=1, include=["documents"])
        doc = (sample["documents"] or [""])[0] or ""
        print("first doc snippet:", repr(doc[:500]))
    else:
        print("NO DOCUMENTS — run: python ingest_books.py")

    if INDEX_MANIFEST.exists():
        manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
        print("manifest files:", list(manifest.keys()))
    else:
        print("manifest missing:", INDEX_MANIFEST)

    if doc:
        nul = doc.count("\x00")
        printable = sum(
            1 for ch in doc if ch.isprintable() or ch.isspace()
        ) / max(len(doc), 1)
        print("null bytes in sample:", nul)
        print(f"printable ratio: {printable:.2%}")
        if nul or printable < 0.9:
            print(
                "WARNING: chunk text looks like binary garbage. Re-ingest after "
                "installing: pip install llama-index-readers-file docx2txt pypdf"
            )
            return 1
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
