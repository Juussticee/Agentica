"""Standalone ingest script (not part of the graph) — spec Section 3.

Reads knowledge/books/ recursively (.pdf/.docx), builds a LlamaIndex
VectorStoreIndex over a persisted ChromaDB collection in storage/, and skips
files already indexed by tracking filename + modified time in
storage/.index_manifest.json.

Embeddings: Ollama nomic-embed-text (known gap — Gemini embeddings not set up).
"""

import json
import os
import time

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
    OLLAMA_BASE_URL,
    STORAGE_DIR,
)


def _load_manifest() -> dict:
    if INDEX_MANIFEST.exists():
        return json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    INDEX_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _mtime_ns(path) -> int:
    return os.stat(path).st_mtime_ns


def main() -> None:
    manifest = _load_manifest()
    all_paths = [
        p for p in BOOKS_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".docx"}
    ]
    new_paths = [p for p in all_paths if str(p) not in manifest]
    already = len(all_paths) - len(new_paths)
    print(f"{len(all_paths)} book file(s) found; {already} already indexed, {len(new_paths)} new")

    if not new_paths:
        print("Nothing new to index. Done.")
        return

    print(f"Embedding with {EMBED_MODEL} via Ollama at {OLLAMA_BASE_URL} ...")
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    documents = SimpleDirectoryReader(
        input_files=[str(p) for p in new_paths], errors="ignore"
    ).load_data()
    print(f"Parsed {len(documents)} chunk(s) from {len(new_paths)} file(s)")

    # Guard against the silent failure mode where LlamaIndex has no reader for
    # a file type and falls back to reading raw bytes (e.g. missing
    # llama-index-readers-file): lossy utf-8 decoding makes binary look
    # half-printable, so test multiple signals, not just a ratio.
    for i, doc in enumerate(documents):
        text = doc.get_content() or ""
        if not text:
            continue
        src = doc.metadata.get("file_name", f"doc#{i}")
        nul = text.count("\x00")
        nonprint = sum(1 for ch in text if not (ch.isprintable() or ch.isspace()))
        letters = sum(1 for ch in text if ch.isalpha())
        binary_sig = text.lstrip().startswith(("PK", "%PDF-"))
        if nul or binary_sig or nonprint / len(text) > 0.05 or letters / len(text) < 0.5:
            raise RuntimeError(
                f"{src} parsed as binary data, not text "
                f"(nul={nul}, magic={binary_sig}, nonprintable={nonprint * 100 // len(text)}%, "
                f"letters={letters * 100 // len(text)}%). Install the file readers: "
                f"pip install llama-index-readers-file docx2txt pypdf"
            )

    import chromadb

    client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    collection = client.get_or_create_collection(
        CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
    )
    print(f"Persisted index to {STORAGE_DIR} (collection '{CHROMA_COLLECTION}')")

    for p in new_paths:
        manifest[str(p)] = _mtime_ns(p)
    _save_manifest(manifest)
    print(f"Indexed {len(new_paths)} file(s); manifest now tracks {len(manifest)} file(s)")


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"Finished in {time.perf_counter() - started:.1f}s")
