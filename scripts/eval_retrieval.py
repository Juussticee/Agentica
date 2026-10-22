"""RAG-1 retrieval-evaluation harness (authorized deliverable, RAG-1 only).

Measures the FROZEN current pipeline: production reader -> production
chunker (1024/200) -> Ollama nomic-embed-text -> Chroma cosine -> the
production retrieval seam (books_agent's own query engine, top_k=3),
extended with one top-5 retriever over the SAME store for Recall@5.

Isolation contract (user-mandated safeguards):
* The evaluation NEVER writes to, rebuilds, migrates, or deletes anything
  under storage/. All indexing happens in a throwaway temp directory with
  a uniquely named collection; chromadb.PersistentClient is instrumented
  so the run FAILS HARD if any client is constructed on a non-temp path.
* Before and after the run: size + mtime_ns + SHA-256 of
  storage/chroma.sqlite3 and storage/.index_manifest.json. Any change is a
  hard failure (exit 2) and the baseline file is NOT written.
* The gold set (benchmarks/rag_gold_v1.json) is read-only: its SHA-256 is
  checked before and after; the run fails if it changed.
* Synthetic-fixture metrics are reported in a SEPARATE section and never
  represent real-corpus accuracy.

Authorized model usage: local Ollama embedding calls to
OLLAMA_BASE_URL (default localhost:11434) ONLY. No chat-model calls, no
cloud services, no OCR APIs, no external network.

Run:
    env\\Scripts\\python.exe scripts\\eval_retrieval.py [--out REPORT] [--force]

Existing reports are immutable by default: if the --out file already
exists the run stops with exit 3 BEFORE any guard, gold, or Ollama work
and writes nothing. Overwriting a baseline requires the explicit --force
opt-in (or a fresh --out path); a refusal reports the target's SHA-256.

Exit codes: 0 ok | 1 gold/fixture problem | 2 guard or runtime failure
(Ollama unreachable, isolation violation, real-store change) | 3 refusing
to overwrite an existing report (pass --force or a new --out path).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import (  # noqa: E402
    BOOKS_MIN_SCORE,
    BOOKS_TOP_K,
    EMBED_MODEL,
    INDEX_MANIFEST,
    INGEST_MAX_FILE_MB,
    INGEST_MAX_PAGES,
    OLLAMA_BASE_URL,
    STORAGE_DIR,
)

from app import chroma_guard  # noqa: E402

GOLD_PATH = PROJECT_ROOT / "benchmarks" / "rag_gold_v1.json"
FIXTURE_DIR = PROJECT_ROOT / "benchmarks" / "rag_fixtures"
REAL_DIR = PROJECT_ROOT / "knowledge" / "books"
DEFAULT_OUT = PROJECT_ROOT / "benchmarks" / "rag_baseline_v1.json"
GUARD_FILES = [STORAGE_DIR / "chroma.sqlite3", INDEX_MANIFEST]

FIXTURE_FILES = [
    "fixture_headings.docx",
    "fixture_columns.pdf",
    "fixture_table.pdf",
    "fixture_scan.pdf",
    "fixture_adversarial.docx",
]


class GuardFailure(RuntimeError):
    """Isolation / immutability guard tripped: stop the run."""


# Chroma clients this process opened. They must be CLOSED before the temp
# directory can be deleted on Windows (open segment/SQLite handles block
# rmtree); the seam's internally-created client is released via
# SharedSystemClient.clear_system_cache() in release_chroma_clients().
_OPEN_CLIENTS: list = []


# --------------------------------------------------------------------------
# small utilities (pure; unit-tested)
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def package_versions(names) -> dict:
    """Installed versions for the config snapshot (spec 13.4 run record)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def guard_snapshot(paths) -> dict:
    out = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            out[str(p)] = None
            continue
        st = p.stat()
        out[str(p)] = {
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "sha256": sha256_file(p),
        }
    return out


def guard_assert_unchanged(before: dict, after: dict, label: str) -> None:
    diffs = [k for k in before if before[k] != after.get(k)]
    if diffs:
        raise GuardFailure(
            f"{label}: protected file(s) changed during the run: {diffs}. "
            f"before={ {k: before[k] for k in diffs} } "
            f"after={ {k: after.get(k) for k in diffs} }"
        )


def node_covers(meta: dict, case: dict) -> bool:
    """True when one retrieved node is gold evidence for this case."""
    if not isinstance(meta, dict):
        return False
    fname = meta.get("file_name")
    if fname not in (case.get("gold_documents") or []):
        return False
    gold_pages = case.get("gold_pages")
    if not gold_pages:                      # docx / file-only evidence
        return True
    label = meta.get("page_label")
    if label is None:
        label = meta.get("page_number")
    # R011/S016 fix (see docs/RAG_BASELINE_AUDIT.md §1.5): gold_pages
    # describes the PAGED evidence of the case; a node with no page
    # metadata at all (docx chunks carry none) cannot be page-filtered
    # without making that gold document unreachable. Enforce the page
    # restriction only on nodes that carry page metadata. A present-but-
    # unparseable label still fails: a page we cannot read is not a page
    # we confirmed.
    if label is None:
        return True
    try:
        return int(str(label).strip()) in {int(p) for p in gold_pages}
    except (TypeError, ValueError):
        return False


def compute_metrics(case: dict, ranked: "list[dict]", max_k: int = 5) -> dict:
    """Metrics for ONE case from its ordered top-max_k retrieval.

    ranked: [{"file_name":..., "page_label":...}, ...] in rank order.
    hit@k requires min_docs_covered (default 1) distinct gold documents
    among the covering nodes of the top k.
    """
    min_docs = int(case.get("min_docs_covered", 1) or 1)
    covers = [node_covers(m, case) for m in ranked[:max_k]]

    def docs_covered_upto(k: int) -> int:
        return len({ranked[i].get("file_name") for i in range(min(k, len(ranked)))
                    if covers[i]})

    out = {}
    for k in (1, 3, 5):
        if k > max_k:
            continue
        hits = sum(covers[:k])
        out[f"hit@{k}"] = docs_covered_upto(k) >= min_docs
        out[f"precision@{k}"] = round(hits / k, 6)
    first = next((i + 1 for i, c in enumerate(covers) if c), None)
    out["first_hit_rank"] = first
    out["reciprocal_rank"] = round(1.0 / first, 6) if first else 0.0
    out["ndcg@5"] = round(1.0 / math.log2(1 + first), 6) if first else 0.0
    out["coverage"] = docs_covered_upto(max_k) >= min_docs
    return out


def aggregate(per_case: "list[dict]") -> dict:
    n = len(per_case)
    if not n:
        return {"cases": 0}
    keys = ("hit@1", "hit@3", "hit@5")
    out = {"cases": n}
    for key in keys:
        out[key] = round(sum(1 for c in per_case if c["metrics"][key]) / n, 4)
    for key in ("precision@1", "precision@3", "precision@5",
                "reciprocal_rank", "ndcg@5"):
        out[key] = round(sum(c["metrics"][key] for c in per_case) / n, 4)
    out["coverage"] = round(
        sum(1 for c in per_case if c["metrics"]["coverage"]) / n, 4)
    return out


def percentile(values: "list[float]", q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(math.ceil(q * len(vals)) - 1)))
    return round(vals[idx], 3)


def abstention_proxy(top1_score, answerable: bool) -> dict:
    """Deterministic score-floor proxy of the production abstain path.

    Production refusal ALSO depends on prompt/grounding behavior that needs
    a chat model; this proxy measures ONLY the score floor
    (BOOKS_MIN_SCORE on the top-1 score) and is labeled as such.
    """
    abstain = top1_score is None or top1_score < BOOKS_MIN_SCORE
    return {
        "top1_score": None if top1_score is None else round(top1_score, 6),
        "proxy_abstain": bool(abstain),
        "correct": bool(abstain) != bool(answerable),   # abstain on unanswerable OR answer on answerable
    }


# --------------------------------------------------------------------------
# gold loading (read-only + hash guard)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# artifact schema v3 evidence (saved per case) + pre-write validation
# --------------------------------------------------------------------------

EVIDENCE_FIELDS = ("rank", "file", "page", "page_number", "node_id", "score")
EVIDENCE_MAX_K = 5


def gold_ref_of(case: dict) -> dict:
    """The exact gold inputs the scorer consumes, for embedding per case."""
    return {
        "gold_documents": list(case.get("gold_documents") or []),
        "gold_pages": (None if not case.get("gold_pages")
                       else list(case.get("gold_pages"))),
        "min_docs_covered": int(case.get("min_docs_covered", 1) or 1),
    }


def case_evidence(case: dict, t5: "list[dict]", prod_nodes: "list[dict]") -> dict:
    """Ordered top-5 evidence for ONE case (artifact_schema 3).

    Built from the SAME t5_rows retrieval already used for the k<=5
    metrics — no additional retrieval call. Identity fields come from the
    t5 rows (the fidelity check already proved their prefix equals the
    production ranking); scores on that production prefix keep the exact
    values v1/v2 recorded, ranks beyond it take the t5 retriever's score.
    """
    k_prod = len(prod_nodes)
    nodes = []
    for i, n in enumerate(t5):
        meta = n.get("metadata") or {}
        nodes.append({
            "rank": i + 1,
            "file": meta.get("file_name"),
            "page": meta.get("page_label"),
            "page_number": meta.get("page_number"),
            "node_id": str(n.get("node_id", "")),
            "score": (prod_nodes[i].get("score") if i < k_prod
                      else n.get("score")),
        })
    return {"retrieved_top5": nodes, "gold_ref": gold_ref_of(case)}


def validate_evidence(per_case: "list[dict]", cases: "list[dict]") -> None:
    """Fail loudly (GuardFailure -> exit 2, artifact NOT written) when the
    saved evidence cannot fully support the reported metrics.

    Per case: evidence present and at most max_k nodes; ranks integer,
    consecutive and 1-based; every identity field present (node_id a str);
    gold_ref equal to the source gold case; and re-deriving EVERY per-case
    metric from (gold_ref, retrieved_top5) must reproduce the recorded
    metrics and first_hit_rank exactly.
    """
    by_id = {c["id"]: c for c in cases}
    for row in per_case:
        rid = row.get("id", "?")
        nodes = row.get("retrieved_top5")
        if not isinstance(nodes, list):
            raise GuardFailure(f"evidence {rid}: retrieved_top5 missing")
        if len(nodes) > EVIDENCE_MAX_K:
            raise GuardFailure(
                f"evidence {rid}: {len(nodes)} nodes exceed "
                f"max_k={EVIDENCE_MAX_K}")
        for slot, node in enumerate(nodes):
            absent = [k for k in EVIDENCE_FIELDS if k not in node]
            if absent:
                raise GuardFailure(
                    f"evidence {rid} slot {slot + 1}: missing field(s) "
                    f"{absent}")
            rank = node["rank"]
            if (not isinstance(rank, int) or isinstance(rank, bool)
                    or rank != slot + 1):
                raise GuardFailure(
                    f"evidence {rid}: ranks must be consecutive and "
                    f"1-based; slot {slot + 1} has rank {rank!r}")
            if not isinstance(node["node_id"], str):
                raise GuardFailure(
                    f"evidence {rid} rank {rank}: node_id must be a "
                    f"string, got {type(node['node_id']).__name__}")
        source = by_id.get(rid)
        if source is None:
            raise GuardFailure(f"evidence {rid}: no such gold case")
        ref = row.get("gold_ref")
        expected = gold_ref_of(source)
        if ref != expected:
            raise GuardFailure(
                f"gold_ref {rid} does not match the source gold case: "
                f"recorded={ref!r}, source={expected!r}")
        ranked = [{"file_name": n["file"], "page_label": n["page"],
                   "page_number": n["page_number"]} for n in nodes]
        derived = compute_metrics(ref, ranked)
        recorded = row.get("metrics")
        if not isinstance(recorded, dict):
            raise GuardFailure(f"evidence {rid}: metrics block missing")
        for key, value in derived.items():
            if key not in recorded:
                raise GuardFailure(
                    f"evidence {rid}: recorded metrics lack {key!r}")
            if recorded[key] != value:
                raise GuardFailure(
                    f"evidence {rid}: recorded {key}={recorded[key]!r} is "
                    f"not reproducible from saved evidence "
                    f"(derived {value!r})")
        if row.get("first_hit_rank") != derived["first_hit_rank"]:
            raise GuardFailure(
                f"evidence {rid}: top-level first_hit_rank "
                f"{row.get('first_hit_rank')!r} != derived "
                f"{derived['first_hit_rank']!r}")


def load_gold(path: Path) -> "tuple[dict, str]":
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    cases = data.get("cases") or []
    problems = []
    for case in cases:
        for key in ("id", "section", "synthetic", "query", "answerable",
                    "gold_documents", "required_facts", "citation_expectation",
                    "labels", "provenance"):
            if key not in case:
                problems.append(f"{case.get('id', '?')}: missing field {key!r}")
        if not str(case.get("provenance", "")).strip():
            problems.append(f"{case.get('id', '?')}: empty provenance")
        if case.get("answerable") and not case.get("gold_documents"):
            problems.append(f"{case.get('id', '?')}: answerable without gold docs")
        if not case.get("answerable") and case.get("gold_documents"):
            problems.append(f"{case.get('id', '?')}: unanswerable with gold docs")
        if case.get("synthetic") != (case.get("section") == "synthetic_fixtures"):
            problems.append(f"{case.get('id', '?')}: synthetic flag/section mismatch")
    if problems:
        raise ValueError("gold set invalid: " + "; ".join(problems))
    return data, digest


def validate_gold_against_sources(gold: dict) -> "list[str]":
    """Fail loudly when a case cites evidence its source cannot contain."""
    notes = []
    from pypdf import PdfReader
    import docx2txt

    texts = {}
    for fname in sorted({d for c in gold["cases"] for d in c["gold_documents"]}):
        if fname.startswith("fixture_"):
            src = FIXTURE_DIR / fname
        else:
            src = REAL_DIR / fname
        if not src.exists():
            raise ValueError(f"gold cites missing source document: {fname}")
        if src.suffix.lower() == ".pdf":
            reader = PdfReader(str(src))
            pages = {int(reader.page_labels[i]):
                     (reader.pages[i].extract_text() or "")
                     for i in range(len(reader.pages))}
            texts[fname] = {"pages": pages}
        else:
            texts[fname] = {"text": docx2txt.process(str(src))}

    for case in gold["cases"]:
        if not case["answerable"]:
            continue
        if int(case.get("min_docs_covered", 1) or 1) > 1:
            # multi-document meta-cases state relations between documents;
            # token checks would false-positive on words like "covers".
            # They are validated by hand and asserted by the retrieval
            # metrics themselves (min_docs_covered).
            continue
        joined = " ".join(case["required_facts"]).lower()
        for token in (w for w in joined.replace(",", " ").split()
                      if len(w) > 3 and w.isalpha()):
            found = False
            for fname in case["gold_documents"]:
                info = texts[fname]
                if "pages" in info:
                    page_ids = case.get("gold_pages") or list(info["pages"])
                    haystack = " ".join(info["pages"].get(p, "")
                                        for p in page_ids).lower()
                else:
                    haystack = info["text"].lower()
                if token in haystack:
                    found = True
                    break
            if not found:
                notes.append(
                    f"{case['id']}: required-fact token {token!r} not found "
                    f"in the cited gold source(s) — REVIEW THE GOLD CASE"
                )
    return notes


# --------------------------------------------------------------------------
# evaluation index (temp collection only) + client-path instrumentation
# --------------------------------------------------------------------------

def instrument_client_paths():
    """Patch chromadb.PersistentClient to record every constructed path.

    Every instance that comes back is ALSO tracked for release: the
    production seam constructs its own client inside `_get_query_engine`,
    and each distinct Client holds +2 refs on the shared chroma System —
    unless every one is closed, the System never stops and its SQLite /
    hnswlib handles keep the temp directory locked on Windows.
    """
    import chromadb

    real = chromadb.PersistentClient
    recorded = []

    def recording(*args, **kwargs):
        path = kwargs.get("path") or (args[0] if args else None)
        chroma_guard.assert_store_path_allowed(path)  # fail BEFORE construction
        recorded.append(str(path))
        client = real(*args, **kwargs)
        _OPEN_CLIENTS.append(client)   # close() is idempotent per instance
        return client

    chromadb.PersistentClient = recording
    return real, recorded


def release_chroma_clients() -> None:
    """Close every Chroma client this process opened and drop the shared
    system cache, so temp directories are actually removable on Windows.

    `Client.close()` decrements the shared System's refcount (2 per
    client: the client plus its admin client); the System — and its
    SQLite / hnswlib file handles — is only stopped when the count hits
    zero. Anything still resident after that is force-stopped directly,
    then the cache is cleared so no cached production client can leak into
    a later run.
    """
    while _OPEN_CLIENTS:
        client = _OPEN_CLIENTS.pop()
        try:
            client.close()
        except Exception:  # noqa: BLE001 — best-effort release
            pass
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        systems = getattr(SharedSystemClient, "_identifier_to_system", None)
        if systems:
            for ident in list(systems):
                system = systems.get(ident)
                if system is None:
                    continue
                try:
                    system.stop()
                except Exception:  # noqa: BLE001 — best-effort release
                    pass
        SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001
        pass


def build_eval_index(temp_dir: Path, collection_name: str, doc_paths,
                     embed_model):
    """Index docs via the PRODUCTION per-file ingest path into TEMP storage."""
    import chromadb
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from ingest_books import process_file

    Settings.embed_model = embed_model
    chroma_guard.assert_store_path_allowed(temp_dir)
    client = chromadb.PersistentClient(path=str(temp_dir))
    _OPEN_CLIENTS.append(client)
    collection = client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"})
    store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        store, storage_context=StorageContext.from_defaults(vector_store=store))

    entries = {}
    for path in doc_paths:
        started = time.perf_counter()
        entry = process_file(
            Path(path), index, collection,
            max_file_mb=INGEST_MAX_FILE_MB, max_pages=INGEST_MAX_PAGES,
            verbose=False)
        entry["_ingest_seconds"] = round(time.perf_counter() - started, 3)
        entries[Path(path).name] = entry
    return index, collection, entries


def query_production_seam(temp_dir: Path, collection_name: str, queries,
                          embed_model):
    """Run the REAL books_agent retrieval seam against the TEMP collection.

    Patches only this process's module globals (STORAGE_DIR,
    CHROMA_COLLECTION, caches), restores them in `finally`, and calls
    engine.retrieve() — retrieval only, NO chat-model calls.
    """
    from llama_index.core import Settings

    import app.agents.books_agent as ba

    # NOTE: read the private backing field Settings._llm, never the
    # Settings.llm property: accessing the property while unset lazily
    # resolves LlamaIndex's DEFAULT LLM (OpenAI) and raises without a key.
    saved = (ba.STORAGE_DIR, ba.CHROMA_COLLECTION, ba._query_engine,
             ba._embedding_model, Settings._llm, Settings._embed_model)
    rows = []
    try:
        ba.STORAGE_DIR = Path(temp_dir)
        ba.CHROMA_COLLECTION = collection_name
        ba._query_engine = None
        ba._embedding_model = embed_model
        engine = ba._get_query_engine()          # builds on the TEMP store
        for query in queries:
            started = time.perf_counter()
            nodes = engine.retrieve(query)
            latency_ms = (time.perf_counter() - started) * 1000.0
            rows.append({
                "query": query,
                "latency_ms": round(latency_ms, 3),
                "nodes": [{
                    "node_id": str(getattr(n.node, "id_", "")),
                    "score": None if n.score is None else float(n.score),
                    "metadata": {
                        k: v for k, v in (n.node.metadata or {}).items()
                        if k in ("file_name", "page_label", "page_number")
                    },
                    "text": (getattr(n.node, "text", "") or "")[:120],
                } for n in nodes],
            })
    finally:
        (ba.STORAGE_DIR, ba.CHROMA_COLLECTION, ba._query_engine,
         ba._embedding_model, Settings._llm, Settings._embed_model) = saved
    return rows


def top5_rows(temp_dir: Path, collection_name: str, queries, embed_model):
    """Ranking up to k=5 over the SAME temp store (k=5 metrics only)."""
    import chromadb
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Settings.embed_model = embed_model
    chroma_guard.assert_store_path_allowed(temp_dir)
    client = chromadb.PersistentClient(path=str(temp_dir))
    _OPEN_CLIENTS.append(client)
    collection = client.get_collection(collection_name)
    store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        store, storage_context=StorageContext.from_defaults(vector_store=store))
    retriever = index.as_retriever(similarity_top_k=5)
    rows = []
    for query in queries:
        nodes = retriever.retrieve(query)
        rows.append([{"node_id": str(getattr(n.node, "id_", "")),
                      "score": (None if n.score is None
                                else float(n.score)),
                      "metadata": {k: v for k, v in
                                   (n.node.metadata or {}).items()
                                   if k in ("file_name", "page_label",
                                            "page_number")}}
                     for n in nodes])
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    # Fail closed for this process: every Chroma client (including the
    # production seam's own) is checked against the production store from
    # here on, before construction.
    chroma_guard.arm()
    chroma_guard.install_persistent_client_wrapper()
    parser = argparse.ArgumentParser(description="RAG-1 retrieval baseline")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing --out report "
                             "(default: refuse with exit 3)")
    args = parser.parse_args(argv)
    out_path = Path(args.out)
    started_all = time.perf_counter()

    # ---- output safety, BEFORE everything -------------------------------
    # A baseline is evidence: never overwrite one silently. Placed before
    # guards/gold/Ollama so a refusal does no work, opens no network, and
    # writes nothing (OverwriteSafetyTest asserts exactly that ordering).
    if out_path.exists() and not args.force:
        print(f"REFUSING TO OVERWRITE existing report: {out_path} "
              f"(sha256 {sha256_file(out_path)}). Existing baselines are "
              f"immutable evidence. Re-run with --force to overwrite "
              f"deliberately, or with --out <new path> to write a new one.")
        return 3

    # ---- guards BEFORE ----------------------------------------------------
    guard_before = guard_snapshot(GUARD_FILES)
    if any(v is None for v in guard_before.values()):
        print(f"GUARD FAILURE: expected store file missing before run: "
              f"{guard_before}")
        return 2
    gold, gold_sha_before = load_gold(GOLD_PATH)
    gold_notes = validate_gold_against_sources(gold)
    for note in gold_notes:
        print(f"[gold-review] {note}")

    # ---- Ollama (local embedding only) ------------------------------------
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags",
                                    timeout=5) as resp:
            tags_payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        print(f"RUNTIME UNAVAILABLE: Ollama at {OLLAMA_BASE_URL} "
              f"({type(exc).__name__}). No fallback; baseline not written.")
        return 2

    # Embedding-model identity (spec 9.2/9.3: record what was measured
    # against — digest, size, family — not just the alias). Read from the
    # LOCAL tags endpoint only; never a network lookup.
    wanted = EMBED_MODEL.lower().split(":")[0]
    model_entry = next(
        (m for m in (tags_payload.get("models") or [])
         if str(m.get("name", "")).lower().split(":")[0] == wanted), None)
    ollama_identity = {
        "name": (model_entry or {}).get("name", EMBED_MODEL),
        "digest": (model_entry or {}).get("digest"),
        "size_bytes": (model_entry or {}).get("size"),
        "modified_at": (model_entry or {}).get("modified_at"),
        "details": (model_entry or {}).get("details") or {},
        "source": f"{OLLAMA_BASE_URL}/api/tags (local only)",
    }

    import chromadb
    from llama_index.embeddings.ollama import OllamaEmbedding

    real_client_cls, client_paths = instrument_client_paths()
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL,
                                  base_url=OLLAMA_BASE_URL)

    # ---- per-query embed-call latency (spec 9.2), measured separately
    # from retrieval latency. One timed embed per unique gold query, before
    # any indexing. A failed embed call aborts here: no fallback embedding
    # source, no baseline written.
    embed_probe = []
    try:
        for q in dict.fromkeys(c["query"] for c in gold["cases"]):
            t0 = time.perf_counter()
            vec = embed_model.get_query_embedding(q)
            embed_probe.append({
                "query": q,
                "ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "dimensions": len(vec) if vec else None,
            })
    except Exception as exc:  # noqa: BLE001
        print(f"RUNTIME UNAVAILABLE: Ollama embed call failed "
              f"({type(exc).__name__}: {exc}). No fallback; baseline not "
              f"written.")
        return 2
    dims = sorted({p["dimensions"] for p in embed_probe
                   if p.get("dimensions")})
    ollama_identity["measured_dimensions"] = dims
    embed_stats = {
        "calls": len(embed_probe),
        "dimensions_observed": dims,
        "p50": percentile([p["ms"] for p in embed_probe], 0.50),
        "p90": percentile([p["ms"] for p in embed_probe], 0.90),
        "max": (round(max(p["ms"] for p in embed_probe), 3)
                if embed_probe else 0.0),
        "per_query": embed_probe,
        "label": "one embed call per unique query, measured on this "
                 "machine, not a guarantee",
    }

    temp_dir = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    collection_name = f"books_eval_{time.strftime('%Y%m%d-%H%M%S')}"
    report = {
        "run_id": collection_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": "",
        "config": {
            "production_top_k": BOOKS_TOP_K,
            "score_floor": BOOKS_MIN_SCORE,
            "score_semantics": "exp(-cosine_distance); floor is a "
                               "deterministic proxy, not a calibrated "
                               "probability",
            "embedding_model": EMBED_MODEL,
            "embedding_model_identity": ollama_identity,
            "ollama_base_url": OLLAMA_BASE_URL,
            "ollama_unreachable_behavior": "abort with exit 2 before any "
                                           "indexing — no fallback embedding "
                                           "source, baseline not written",
            "chunking": "production SentenceSplitter defaults "
                        "(1024 tokens / 200 overlap, tiktoken cl100k_base)",
            "ingest_caps": {"max_file_mb": INGEST_MAX_FILE_MB,
                            "max_pages": INGEST_MAX_PAGES},
            "chat_model_calls": "none (RAG-1 is retrieval-only)",
            # machine notes (spec 13.4: each run records its environment) —
            # recorded for comparison only, never as a quality claim.
            "machine": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "packages": package_versions([
                    "chromadb", "llama-index-core",
                    "llama-index-embeddings-ollama",
                    "llama-index-vector-stores-chroma", "tiktoken"]),
                "note": "environment recorded for comparison; latency is "
                        "machine-dependent and not a guarantee",
            },
        },
        "metric_definitions": {
            # Version of the relevance semantics: v2 applies gold_pages
            # only to nodes that carry page metadata (v1 rejected page-less
            # nodes, making R011/S016 gold docx members unreachable). See
            # docs/RAG_BASELINE_AUDIT.md §1.5.
            "version": "v2-page-metadata-aware",
            "hit@k": "top-k contains min_docs_covered (default 1) distinct "
                     "gold documents",
            "precision@k": "covering nodes / k",
            "reciprocal_rank": "1 / rank of first covering node",
            "ndcg@5": "1/log2(1+first_hit_rank), binary relevance",
            "note": "retrieval measurements on this gold set — NOT quality "
                    "guarantees; latency and score distributions are "
                    "measurements only",
        },
        # Evidence schema (NOT metric semantics — metric_definitions.version
        # stays "v2-page-metadata-aware"): v3 persists the full ordered
        # top-5 ranking plus the gold inputs per case, so every reported
        # metric is independently recomputable from the artifact alone.
        "artifact_schema": {
            "version": 3,
            "evidence": "ordered-top5-per-case",
        },
        "sections": {},
        "abstention_proxy": {},
        "latency_ms": {},
        "limitations": [],
    }

    try:
        import subprocess
        report["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
        # HEAD alone is misleading while the evaluated code is uncommitted:
        # record that fact so a corrected artifact cannot masquerade as
        # plain HEAD content.
        report["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True).stdout.strip())

        # corpus + fixtures hashes
        real_files = sorted(p for p in REAL_DIR.iterdir() if p.is_file())
        fixture_files = [FIXTURE_DIR / n for n in FIXTURE_FILES]
        missing = [str(p) for p in fixture_files if not p.exists()]
        if missing:
            print(f"FIXTURES MISSING: {missing} — run scripts/make_rag_fixtures.py")
            return 1
        report["corpus"] = {
            "real_files": [{"file": p.name, "sha256": sha256_file(p),
                            "bytes": p.stat().st_size} for p in real_files],
            "fixture_files": [{"file": p.name, "sha256": sha256_file(p),
                               "bytes": p.stat().st_size}
                              for p in fixture_files],
        }
        report["gold"] = {"path": str(GOLD_PATH.relative_to(PROJECT_ROOT)),
                          "sha256": gold_sha_before,
                          "cases": len(gold["cases"]),
                          "real_cases": sum(1 for c in gold["cases"]
                                            if not c["synthetic"]),
                          "synthetic_cases": sum(1 for c in gold["cases"]
                                                 if c["synthetic"]),
                          "gap_list": gold.get("gap_list", []),
                          "review_notes": gold_notes}

        # ---- ingest (production per-file path, temp collection only) ------
        t0 = time.perf_counter()
        index, collection, entries = build_eval_index(
            temp_dir, collection_name,
            [p for p in real_files] + fixture_files, embed_model)
        report["ingest"] = {
            "seconds": round(time.perf_counter() - t0, 3),
            "collection": collection_name,
            "temp_dir_used": True,
            "chunks_in_eval_collection": collection.count(),
            "per_file": {name: {k: e.get(k) for k in
                                ("status", "chunks", "documents",
                                 "nonempty_pages", "empty_pages", "category",
                                 "_ingest_seconds")}
                         for name, e in entries.items()},
        }
        failed = [n for n, e in entries.items() if e.get("status") == "failed"]
        if failed:
            print(f"EVAL INGEST FAILED for {failed} — inspect entries; "
                  f"baseline not written.")
            return 1

        # ---- retrieval: production seam + top-5 extension -----------------
        cases = gold["cases"]
        queries = [c["query"] for c in cases]
        prod_rows = query_production_seam(
            temp_dir, collection_name, queries, embed_model)
        t5_rows = top5_rows(temp_dir, collection_name, queries, embed_model)

        # fidelity check: production top-3 == top-5 prefix for every query.
        # Compare the production node list against the SAME-LENGTH slice of
        # the top-5 list (prod has at most BOOKS_TOP_K=3 nodes; the top-5
        # list always has up to 5, so an unsliced comparison could never
        # pass regardless of retrieval behavior).
        fidelity_mismatch = None
        for p, t5 in zip(prod_rows, t5_rows):
            prod_ids = [n["node_id"] for n in p["nodes"]]
            if prod_ids != [n["node_id"] for n in t5[:len(prod_ids)]]:
                fidelity_mismatch = p["query"]
                break
        fidelity_ok = fidelity_mismatch is None
        report["fidelity_check_production_top3_equals_top5_prefix"] = fidelity_ok
        if not fidelity_ok:
            raise GuardFailure(
                "production top-3 ranking is not a prefix of the top-5 "
                f"ranking (first mismatching query: {fidelity_mismatch!r}) — "
                "metrics would not describe the production path")

        # ---- failure/timeout counts (spec 13.3) ---------------------------
        partial = [n for n, e in entries.items()
                   if e.get("status") == "partial"]
        report["failures_and_timeouts"] = {
            "ingest_failed_files": failed,
            "ingest_partial_files": partial,
            "embed_call_failures": 0,
            "retrieval_errors": 0,
            "retrieval_timeouts": 0,
            "note": "a written baseline implies zero failures on this run: "
                    "an embed/Ollama failure aborts BEFORE indexing with "
                    "exit 2 and no baseline; an ingest failure aborts with "
                    "exit 1; a retrieval error raises out and no baseline "
                    "is written. Counts are of THIS run, not of the "
                    "pipeline under all conditions.",
        }

        # ---- per-case evaluation ------------------------------------------
        by_id = {c["id"]: c for c in cases}
        per_case = []
        for case, t5 in zip(cases, t5_rows):
            ranked = [n["metadata"] for n in t5]
            metrics = compute_metrics(case, ranked)
            prod = prod_rows[cases.index(case)]
            top1 = prod["nodes"][0]["score"] if prod["nodes"] else None
            per_case.append({
                "id": case["id"], "section": case["section"],
                "synthetic": case["synthetic"],
                "answerable": case["answerable"],
                "labels": case["labels"],
                "first_hit_rank": metrics["first_hit_rank"],
                "metrics": metrics,
                "abstention_proxy": abstention_proxy(top1, case["answerable"]),
                **case_evidence(case, t5, prod["nodes"]),
            })
        assert len(by_id) == len(cases)
        # v3 gate: the saved evidence must fully support every reported
        # metric (and gold_ref must equal the gold case) BEFORE anything is
        # aggregated or written — failure stops the run with exit 2 and no
        # artifact written.
        validate_evidence(per_case, cases)

        for section in ("real_corpus", "synthetic_fixtures"):
            rows = [c for c in per_case if c["section"] == section]
            answerable = [c for c in rows if c["answerable"]]
            unanswerable = [c for c in rows if not c["answerable"]]
            report["sections"][section] = {
                "retrieval_metrics_on_answerable": aggregate(answerable),
                "unanswerable_cases": len(unanswerable),
                "score_stats": {
                    "top1_scores_answerable": sorted(
                        c["abstention_proxy"]["top1_score"]
                        for c in answerable
                        if c["abstention_proxy"]["top1_score"] is not None),
                    "top1_scores_unanswerable": sorted(
                        c["abstention_proxy"]["top1_score"]
                        for c in unanswerable
                        if c["abstention_proxy"]["top1_score"] is not None),
                },
                "abstention_proxy": {
                    "correct_on_unanswerable": (
                        f"{sum(1 for c in unanswerable if c['abstention_proxy']['correct'])}"
                        f"/{len(unanswerable)}"),
                    "false_abstain_on_answerable": (
                        f"{sum(1 for c in answerable if not c['abstention_proxy']['correct'])}"
                        f"/{len(answerable)}"),
                },
                "per_case": rows,
            }
        report["latency_ms"] = {
            "retrieval_per_query": {
                "p50": percentile([r["latency_ms"] for r in prod_rows], 0.50),
                "p90": percentile([r["latency_ms"] for r in prod_rows], 0.90),
                "max": round(max(r["latency_ms"] for r in prod_rows), 3),
            },
            "embed_call_per_query": embed_stats,
            "label": "measurement on this machine, not a guarantee",
        }

        # ---- limitations (honest, safeguard 4) -----------------------------
        real_chunks = report["ingest"]["chunks_in_eval_collection"]
        report["limitations"] = [
            "REAL CORPUS IS THE MEASUREMENT CEILING: both real documents "
            "fit in a single chunk each, so any top-k>=2 retrieval over the "
            "real section trivially includes the gold chunk whenever the "
            "query embeds anywhere near the corpus — real-corpus metrics "
            "cannot discriminate retrieval quality (spec hypothesis H10).",
            f"The eval collection holds {real_chunks} chunks total "
            "(2 real + fixture chunks); synthetic-fixture metrics show "
            "discriminating behavior but DO NOT establish real-world "
            "retrieval accuracy.",
            "Answer-level metrics (citation correctness, unsupported-claim "
            "rate) are NOT measured: they require chat-model calls, which "
            "are outside this authorization.",
            "The abstention proxy measures only the deterministic score "
            "floor; production refusal also involves prompt and grounding "
            "behavior that requires a chat model.",
        ]

        # ---- guards AFTER --------------------------------------------------
        guard_after = guard_snapshot(GUARD_FILES)
        guard_assert_unchanged(guard_before, guard_after, "real-store guard")
        gold_sha_after = sha256_file(GOLD_PATH)
        if gold_sha_after != gold_sha_before:
            raise GuardFailure("gold set changed during the run")
        report["gold"]["sha256_after"] = gold_sha_after
        report["guards"] = {
            "real_store_before": guard_before,
            "real_store_after": guard_after,
            "real_store_unchanged": True,
            "gold_unchanged": True,
            "chroma_client_paths_constructed": client_paths,
            "all_chroma_clients_on_temp_paths": all(
                str(temp_dir) in p for p in client_paths),
            "production_query_engine_cache_used": False,
        }
        if not report["guards"]["all_chroma_clients_on_temp_paths"]:
            raise GuardFailure(
                f"a Chroma client was constructed off the temp path: "
                f"{client_paths}")

        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        report_written = True
    except GuardFailure as exc:
        print(f"GUARD FAILURE — run stopped, baseline NOT written: {exc}")
        return 2
    finally:
        try:
            chromadb.PersistentClient = real_client_cls
        except Exception:  # noqa: BLE001
            pass
        release_chroma_clients()
        shutil.rmtree(temp_dir, ignore_errors=True)

    # ---- summary -----------------------------------------------------------
    print(f"\n=== RAG-1 baseline written: {out_path} ===")
    print(f"git HEAD {report['git_head'][:12]} | eval collection "
          f"{collection_name} (deleted) | total "
          f"{time.perf_counter() - started_all:.1f}s")
    for section, data in report["sections"].items():
        m = data["retrieval_metrics_on_answerable"]
        print(f"[{section}] cases={data['retrieval_metrics_on_answerable'].get('cases', 0)} "
              f"hit@1={m.get('hit@1')} hit@3={m.get('hit@3')} hit@5={m.get('hit@5')} "
              f"MRR={m.get('reciprocal_rank')} nDCG@5={m.get('ndcg@5')}")
        print(f"  abstention proxy: correct(unanswerable)="
              f"{data['abstention_proxy']['correct_on_unanswerable']} "
              f"false-abstain(answerable)="
              f"{data['abstention_proxy']['false_abstain_on_answerable']}")
        scores = data["score_stats"]
        print(f"  top1 scores answerable    : {scores['top1_scores_answerable']}")
        print(f"  top1 scores unanswerable  : {scores['top1_scores_unanswerable']}")
    print("real-store guard: UNCHANGED (stat + sha256) | gold: unchanged")
    print("NOTE: synthetic-fixture metrics do not establish real-world "
          "retrieval accuracy; see limitations in the report.")
    return 0 if report_written else 2


if __name__ == "__main__":
    raise SystemExit(main())
