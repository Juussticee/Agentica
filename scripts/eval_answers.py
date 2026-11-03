#!/usr/bin/env python
"""RAG-AE — offline RAG answer-quality evaluation harness (MASTER_PLAN RAG-1 item 5).

Measures, per deterministic fixture case, what the REAL books answer pipeline
(grounding / polarity / numeric / citation / refusal / M5 recovery gates in
`app.agents.books_agent`) does to a scripted (question, evidence, raw model
reply) triple — and, in integration mode, what the REAL retrieval pipeline
(production reader -> chunker -> process_file -> temp Chroma -> seam ->
postprocessors) retrieves from an isolated temporary store.

Modes
-----
A. unit (default, primary acceptance suite)
   Scripted evidence + scripted raw model reply through the REAL books_agent
   (query engine and get_llm patched at the narrowest boundary). No Chroma,
   no embeddings, no model, no network. Scores answer-layer dimensions:
   correctness, groundedness, evidence use, completeness, relevance,
   citation correctness, abstention, contradiction, robustness, recovery.
B. integration
   Builds a TEMPORARY Chroma store from the real corpus + benchmark fixtures
   via the production ingest path, retrieves through the production seam,
   and scores retrieval metrics + evidence coverage. Refuses to run unless
   the store is verified isolated; never opens the production store. The
   answer layer is NOT re-scored here: the deterministic embedding double's
   similarity calibration is not comparable to the production embedder, so
   answer quality stays in unit mode (documented in docs/RAG_AE_EVALUATION.md).
C. local-model (opt-in, disabled by default, NOT part of the acceptance suite)
   Same as unit, but the raw model reply comes from a LOCAL Ollama chat model.
   Requires --enable-live AND LLM_PROVIDER=ollama; it can never fall back to
   a remote provider. Not executed by the RAG-AE task itself.

Isolation / safety
------------------
* chroma_guard is armed fail-closed for this process; every PersistentClient
  construction is checked against the production store before it happens.
* Integration mode uses a fresh TemporaryDirectory exclusively; there is NO
  flag that can point it at any other path.
* Production store files (storage/chroma.sqlite3, the ingest manifest) are
  hashed before and after every run; any drift aborts with exit 2.
* A network guard records and blocks socket/urlopen access; a run with any
  recorded attempt exits 2 and is never reported as a valid baseline.

Exit codes
----------
0  run completed, all executed cases passed
1  dataset validation failed (per case+field messages printed)
2  isolation/guard/network/runtime refusal (no valid results)
3  refusing to overwrite an existing --out artifact (evidence immutability)
4  run completed, one or more case assertions failed

Run:
    env\\Scripts\\python.exe scripts\\eval_answers.py --mode unit
    env\\Scripts\\python.exe scripts\\eval_answers.py --mode integration
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import chroma_guard  # noqa: E402
from app.agents import books_agent as ba  # noqa: E402
from app.agents.books_agent import (  # noqa: E402
    _PARTIAL_DISCLOSURE,
    _STOP_TOKENS,
    _VERIFIER_CONFLICT_NOTE,
    _VERIFIER_REMOVED_NOTE,
    _VERIFIER_UNAVAILABLE_NOTE,
)
from app.config import BOOKS_MIN_SCORE, LLM_PROVIDER, NOT_FOUND_PHRASE  # noqa: E402

import eval_retrieval as ev  # noqa: E402 — reuse RAG-1 guard/metrics/ingest utilities

GOLD_PATH = PROJECT_ROOT / "benchmarks" / "rag_gold_v2.json"
DEFAULT_OUTS = {
    "unit": PROJECT_ROOT / "benchmarks" / "rag_ae_unit_v1.json",
    "integration": PROJECT_ROOT / "benchmarks" / "rag_ae_integration_v1.json",
    "local-model": PROJECT_ROOT / "benchmarks" / "rag_ae_local_model_v1.json",
}
TOP_K = 4                       # matches production similarity_top_k
EVAL_COLLECTION = "rag_ae_eval_collection"   # temp-store only; never "books"
HARNESS_VERSION = "1.0"

DEFINITIONS = {
    "correctness": "boolean: every required_facts entry of the case appears in the final answer (whitespace-normalized substring). Applicable to answerable/partial cases only.",
    "completeness": "fraction: |required facts present| / |required facts| over answerable cases.",
    "groundedness": "boolean per case: every substantive sentence of the final answer body has all its content tokens in the scripted/retrieved evidence text or its allowlisted citation metadata, and every number in the body appears literally in the evidence TEXT. Independent re-implementation of the production contract (disclosure/refusal/score-note lines exempt); rule-based token coverage, NOT semantic entailment.",
    "evidence_use": "boolean: the final answer of an answerable, non-abstained case contains at least one required fact (isolates 'answered from nowhere').",
    "relevance": "boolean (proxy): an answerable, non-abstained final answer shares at least one distinctive (non-stopword) content term with the question.",
    "citation": "boolean per case against the case's citation expectation: 'required' -> a References footer exists naming exactly the expected evidence files (validated against evidence metadata, pages must match); 'forbidden' -> no footer; 'unavailable' -> footer exists and only says 'attribution unavailable', never an invented filename.",
    "abstention": "boolean: final answer starts with the mandated NOT_FOUND_PHRASE iff the case expects abstention. Aggregated as precision/recall over executed unit cases.",
    "contradiction": "boolean: none of the case's forbidden_facts (fixture-declared conflicting/wrong/planted values) appear in the final answer.",
    "robustness": "boolean: for cases labeled robustness, forbidden distractor facts absent AND required facts still present.",
    "recovery": "boolean: M5 markers in captured stdout (attempted/succeeded/rejected) and the scripted-LLM call count match the case's recovery expectations.",
    "hit@k": "integration: the case's gold documents (min_docs_covered distinct) appear among the top-k retrieved chunks (k=1,3,4; page-restricted when gold_pages is set).",
    "precision@k": "integration: fraction of the top-k retrieved chunks whose (file, page) is gold evidence for the case.",
    "reciprocal_rank": "integration: 1 / rank of the first gold-retrieved chunk (0 if none).",
    "fact_coverage": "integration: fraction of answerable cases whose required facts all appear literally in the retrieved top-k context text.",
    "averaging": "macro: every rate is the unweighted mean over the cases where the metric is applicable; skipped/unresolved cases never enter a denominator.",
}


# --------------------------------------------------------------------------
# deterministic embedding double (integration mode only)
# --------------------------------------------------------------------------
from llama_index.core.embeddings import BaseEmbedding  # noqa: E402


class HashedTfEmbedding(BaseEmbedding):
    """Deterministic, offline embedding double: hashed bag-of-words TF.

    L2-normalized term-frequency vectors over tokens hashed with blake2b
    (process-stable, unlike builtin hash). Similarity is lexical overlap, so
    ranking is meaningful on the small synthetic corpus but NOT comparable
    to the production embedder (nomic-embed-text). Used ONLY in integration
    mode; never a fallback for production. Subclasses LlamaIndex's
    BaseEmbedding exactly like the RAG-1 harness's MockEmbedding.
    """

    dim: int = 512

    # Tiny deterministic English stoplist: real embedding models effectively
    # ignore function words, and without this the lexical double ranks by
    # stopword overlap instead of content overlap. Static data, no library.
    _STOP = frozenset("""
        the a an and or but nor of in on at to by with for as from into
        is are was were be been being am do does did done has have had
        can could may might must shall should will would
        i we you he she it they them its their there here
        what which who whom when where why how
        this that these those there
        about above below between through during after before
        not no nor yes
        """.split())

    @classmethod
    def class_name(cls):
        return "HashedTfEmbedding"

    def _embed(self, text: str):
        counts = {}
        for tok in re.findall(r"[a-z0-9]{2,}", (text or "").lower()):
            if tok in self._STOP:
                continue
            idx = int.from_bytes(
                hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(),
                "big") % self.dim
            counts[idx] = counts.get(idx, 0) + 1.0
        if not counts:
            return [0.0] * self.dim
        norm = math.sqrt(sum(v * v for v in counts.values()))
        vec = [0.0] * self.dim
        for idx, val in counts.items():
            vec[idx] = val / norm
        return vec

    def _get_text_embedding(self, text):
        return self._embed(text)

    def _get_query_embedding(self, text):
        return self._embed(text)

    async def _aget_text_embedding(self, text):
        return self._embed(text)

    async def _aget_query_embedding(self, text):
        return self._embed(text)


# --------------------------------------------------------------------------
# text utilities (pure; unit-tested)
# --------------------------------------------------------------------------
_NUM_RE = re.compile(r"\$?\d[\d,.]*%?")
_WORD_RE = re.compile(r"[a-z]+")


def norm_ws(text) -> str:
    return " ".join(str(text or "").split())


def content_tokens(text) -> set:
    """Lowercased content tokens.

    Mirrors production `_support_tokens` tokenization: numbers (with
    commas/decimals) are ONE token with commas stripped; words are
    letter runs. Used by the INDEPENDENT groundedness check — the same
    function always runs on the answer side and the evidence side, so
    any pass/fail is a relative effect of wording, never of asymmetric
    normalization.
    """
    low = str(text or "").lower()
    out = set()
    for raw in _NUM_RE.findall(low):
        tok = raw.replace(",", "").replace("$", "").rstrip("%")
        if tok:
            out.add(tok)
    for raw in _WORD_RE.findall(low):
        if raw:
            out.add(raw)
    return out


def number_tokens(text) -> set:
    return {t for t in content_tokens(text) if t.replace(".", "").isdigit()}


def distinctive_query_tokens(query) -> set:
    return {t for t in content_tokens(query)
            if t not in _STOP_TOKENS and len(t) > 2}


def split_sentences(text) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?" + '"' + r"])\s+",
                                        str(text or "")) if s.strip()]


_EXEMPT_PREFIXES = (
    NOT_FOUND_PHRASE,
    _PARTIAL_DISCLOSURE,
    _VERIFIER_UNAVAILABLE_NOTE,
    _VERIFIER_CONFLICT_NOTE,
    _VERIFIER_REMOVED_NOTE,
    "(best similarity score",
    "(semantic verification unavailable)",
)


def is_exempt_sentence(sentence: str) -> bool:
    low = sentence.strip().lower()
    return any(low.startswith(pfx.lower()) or pfx.lower() in low
               for pfx in _EXEMPT_PREFIXES)


def split_footer(answer: str):
    """(body, entries|None) — entries are ';'-joined References items."""
    marker = "\n\nReferences:"
    idx = str(answer or "").rfind(marker)
    if idx < 0:
        return str(answer or ""), None
    body = str(answer)[:idx]
    raw = str(answer)[idx + len(marker):].strip()
    entries = [e.strip() for e in raw.split(";") if e.strip()]
    return body, entries


_FILENAME_IN_TEXT_RE = re.compile(r"\b[\w\-]+\.(?:pdf|docx)\b", re.I)


def file_entries(entries) -> list:
    """Filenames cited in footer entries ('attribution unavailable' -> none)."""
    files = []
    for entry in entries or []:
        if entry.lower().startswith("attribution unavailable"):
            continue
        match = _FILENAME_IN_TEXT_RE.search(entry)
        if match:
            files.append(match.group(0))
    return files


# --------------------------------------------------------------------------
# network guard (offline proof)
# --------------------------------------------------------------------------
_NETWORK_MSG = ("network access attempted during an offline RAG-AE run — "
                "socket/urlopen is blocked by the harness")


class NetworkGuard:
    """Records and blocks socket/urlopen access for the guarded window."""

    def __init__(self):
        self.attempts = []
        self._saved = None

    def _blocked(self, kind):
        def _fn(*args, **kwargs):
            self.attempts.append(kind)
            raise RuntimeError(_NETWORK_MSG)
        return _fn

    def install(self):
        self._saved = (socket.socket.connect, socket.create_connection,
                       urllib.request.urlopen)
        socket.socket.connect = self._blocked("socket.socket.connect")
        socket.create_connection = self._blocked("socket.create_connection")
        urllib.request.urlopen = self._blocked("urllib.request.urlopen")

    def uninstall(self):
        if self._saved:
            (socket.socket.connect, socket.create_connection,
             urllib.request.urlopen) = self._saved
            self._saved = None


# --------------------------------------------------------------------------
# dataset loading + validation
# --------------------------------------------------------------------------
def _source_path(file_name: str) -> Path:
    if file_name.startswith("fixture_"):
        return ev.FIXTURE_DIR / file_name
    return ev.REAL_DIR / file_name


def load_source_texts(file_names) -> dict:
    """{file: {'pages': {label: text}} for PDFs, {'text': ...} for docx."""
    from pypdf import PdfReader
    import docx2txt

    out = {}
    for fname in sorted(set(file_names)):
        src = _source_path(fname)
        if not src.exists():
            raise ValueError(f"source document missing on disk: {fname}")
        if src.suffix.lower() == ".pdf":
            reader = PdfReader(str(src))
            out[fname] = {"pages": {
                str(reader.page_labels[i]):
                norm_ws(reader.pages[i].extract_text() or "")
                for i in range(len(reader.pages))}}
        else:
            out[fname] = {"text": norm_ws(docx2txt.process(str(src)))}
    return out


def _evidence_in_source(text: str, info: dict) -> bool:
    if "pages" in info:
        return any(norm_ws(text) in page for page in info["pages"].values())
    return norm_ws(text) in info["text"]


def validate_dataset(ds: dict, source_texts: "dict|None" = None) -> list:
    """Dataset errors as 'case <id>, field <field>: message' strings.

    source_texts: optional preloaded {file: info} cache (loaded on demand).
    """
    errors = []
    source_texts = dict(source_texts or {})

    if not isinstance(ds, dict) or not ds.get("cases"):
        return ["dataset: missing top-level 'cases'"]
    seen = set()
    for case in ds["cases"]:
        cid = case.get("id", "<no-id>")
        if cid in seen:
            errors.append(f"case {cid}, field id: duplicate case id")
        seen.add(cid)
        _validate_case(case, cid, errors, source_texts)
    return errors


def _validate_case(case, cid, errors, source_texts):
    def err(field, msg):
        errors.append(f"case {cid}, field {field}: {msg}")

    status = case.get("status")
    if status not in ("executable", "unresolved"):
        err("status", "must be 'executable' or 'unresolved'")
        return
    if status == "unresolved":
        if not str(case.get("unresolved_reason") or "").strip():
            err("unresolved_reason", "unresolved cases must record a reason")
        return

    query = str(case.get("query") or "")
    if len(query.strip()) < 8:
        err("query", "must be a non-trivial question")
    if not case.get("labels"):
        err("labels", "required non-empty label list")
    if case.get("answerable") not in (True, False, "partial"):
        err("answerable", "must be true, false or 'partial'")
    modes = case.get("modes") or []
    if not modes or set(modes) - {"unit", "integration"}:
        err("modes", "must be a non-empty subset of ['unit', 'integration']")
    expect = case.get("expect") or {}
    if not expect:
        err("expect", "executable cases require an expectation block")

    facts = case.get("required_facts") or []
    for fact in facts:
        if not str(fact).strip():
            err("required_facts", "contains an empty fact")
    if expect.get("facts_present") and not facts:
        err("required_facts", "expect.facts_present=true needs required facts")
    # A controlled-retrieval-failure case expects a propagated ERROR instead
    # of an answer, so the abstention/fact/evidence shape rules do not apply.
    is_error_case = bool((case.get("script") or {}).get("retrieval_error")) \
        or bool(expect.get("error"))
    if not is_error_case:
        if case.get("answerable") is False and not expect.get("abstain"):
            err("expect.abstain", "unanswerable cases must expect abstention")
        if expect.get("abstain") is False and not facts:
            err("required_facts", "non-abstaining cases need required facts")

    citation = expect.get("citation")
    if citation not in ("required", "forbidden", "unavailable"):
        err("expect.citation", "must be required|forbidden|unavailable")
    if expect.get("recovery_attempted") and not (case.get("recovery_replies") or []):
        err("recovery_replies", "expected recovery needs scripted replies")
    if int(expect.get("llm_calls", 0) or 0) > 0 and not (case.get("recovery_replies") or []):
        err("recovery_replies", "expected llm calls need scripted replies")

    evidence = case.get("evidence")
    if evidence is None and not (case.get("script") or {}).get("retrieval_error"):
        err("evidence", "missing evidence list")
        return
    evidence = evidence or []
    if not evidence and not expect.get("abstain") and not is_error_case:
        err("evidence", "empty evidence is only valid for abstention or "
                        "controlled-failure cases")

    files_seen = set()
    for i, node in enumerate(evidence):
        prefix = f"evidence[{i}]"
        text = norm_ws(node.get("text"))
        if not text:
            err(prefix + ".text", "empty evidence text")
        score = node.get("score")
        if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 2.0):
            err(prefix + ".score", "score must be a number in [0, 2]")
        source = node.get("source")
        fname = str(node.get("file_name") or "")
        if source not in ("corpus", "fixture", "scripted"):
            err(prefix + ".source", "must be corpus|fixture|scripted")
            continue
        if source in ("corpus", "fixture"):
            if not fname:
                err(prefix + ".file_name", "corpus/fixture evidence names its file")
                continue
            expected_kind = "fixture" if fname.startswith("fixture_") else "corpus"
            if source != expected_kind:
                err(prefix + ".source", f"{fname} is {expected_kind} evidence")
            if fname not in source_texts:
                try:
                    source_texts[fname] = load_source_texts([fname])[fname]
                except ValueError as exc:
                    err(prefix + ".file_name", str(exc))
                    continue
            if text and not _evidence_in_source(text, source_texts[fname]):
                err(prefix + ".text",
                    "not a verbatim substring of its source document")
            files_seen.add(fname)
        else:                                   # scripted
            if not text:
                err(prefix + ".text", "scripted evidence needs text")

    # answerable cases must have identifiable supporting evidence
    if case.get("answerable") in (True, "partial") and facts:
        corpus_tokens = set()
        for node in evidence:
            corpus_tokens |= content_tokens(node.get("text"))
            corpus_tokens |= content_tokens(node.get("file_name"))
            corpus_tokens |= content_tokens(node.get("page_label"))
        for fact in facts:
            missing = content_tokens(fact) - corpus_tokens
            if missing:
                err("required_facts",
                    f"fact {fact!r} has tokens {sorted(missing)[:5]} absent "
                    "from the case evidence (no identifiable support)")

    if citation == "required":
        cfiles = expect.get("citation_files") or []
        if not cfiles:
            err("expect.citation_files", "citation required names its files")
        for f in cfiles:
            if f not in files_seen:
                err("expect.citation_files",
                    f"{f} is not among the case's evidence files")


def load_gold_v2(path: Path):
    raw = Path(path).read_bytes()
    ds = json.loads(raw.decode("utf-8"))
    return ds, hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# deterministic doubles + case execution
# --------------------------------------------------------------------------
class _ChatMsg:
    def __init__(self, text):
        self.content = text


class ScriptedLLM:
    """Scripted M5 recovery model. Exhausted replies return the mandated
    refusal (deterministic) and are counted, never silently dropped."""

    def __init__(self, replies):
        self.replies = list(replies or [])
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if not self.replies:
            return _ChatMsg(NOT_FOUND_PHRASE)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _ChatMsg(reply)


class _FakeResponse:
    def __init__(self, text, nodes):
        self._text = text
        self.source_nodes = nodes

    def __str__(self):
        return self._text


class _FakeEngine:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def query(self, question):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def node_from_spec(spec: dict):
    meta = {}
    if spec.get("file_name"):
        meta["file_name"] = spec["file_name"]
    if spec.get("page_label") not in (None, ""):
        meta["page_label"] = spec["page_label"]
    return SimpleNamespace(score=float(spec.get("score", 0.72)),
                           metadata=meta,
                           text=norm_ws(spec.get("text")))


def run_unit_case(case: dict) -> dict:
    """Drive the REAL books_agent with scripted retrieval + scripted LLM."""
    nodes = [node_from_spec(e) for e in (case.get("evidence") or [])]
    if (case.get("script") or {}).get("retrieval_error"):
        engine = _FakeEngine(RuntimeError("controlled retrieval failure"))
        response_desc = "RuntimeError(controlled retrieval failure)"
    else:
        engine = _FakeEngine(_FakeResponse(case.get("raw_response", ""), nodes))
        response_desc = "scripted"
    llm = ScriptedLLM(case.get("recovery_replies"))

    saved = (ba._get_query_engine, ba.get_llm)
    buf = io.StringIO()
    outcome = {"answer": None, "error": None, "stdout": "",
               "llm_calls": 0, "response": response_desc}
    started = time.perf_counter()
    try:
        ba._get_query_engine = lambda: engine
        ba.get_llm = lambda: llm
        with contextlib.redirect_stdout(buf):
            try:
                result = ba.books_agent(
                    {"question": case["query"], "route": "books",
                     "sources": ["books"]})
                outcome["answer"] = (result or {}).get("answer")
            except RuntimeError as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        ba._get_query_engine, ba.get_llm = saved
    outcome["stdout"] = buf.getvalue()
    outcome["llm_calls"] = llm.calls
    outcome["runtime_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return outcome


# --------------------------------------------------------------------------
# answer-layer scoring
# --------------------------------------------------------------------------
def _check(verdict, reason, **extra):
    out = {"verdict": verdict, "reason": reason}
    out.update(extra)
    return out


def check_abstention(answer, expected):
    actual = str(answer or "").startswith(NOT_FOUND_PHRASE)
    if actual == bool(expected):
        return _check("pass", "abstention matches expectation")
    if expected:
        return _check("fail", "expected abstention but the pipeline answered")
    return _check("fail", "FALSE REFUSAL: pipeline abstained on grounded evidence")


def check_facts(answer, facts):
    body = norm_ws(answer)
    missing = [f for f in facts if norm_ws(f) not in body]
    if not missing:
        return _check("pass", f"all {len(facts)} required fact(s) present")
    return _check("fail", f"missing required fact(s): {missing[:3]}",
                  missing=missing)


def check_forbidden(answer, forbidden):
    body = norm_ws(answer).lower()
    hits = [f for f in forbidden if norm_ws(f).lower() and norm_ws(f).lower() in body]
    if not hits:
        return _check("pass", "no forbidden content present")
    return _check("fail", f"forbidden content present: {hits[:3]}", hits=hits)


def check_groundedness(answer, evidence_texts, evidence_meta_text):
    body, _ = split_footer(answer)
    corpus_tokens = set()
    corpus_numbers = set()
    for text in evidence_texts:
        corpus_tokens |= content_tokens(text)
        corpus_numbers |= number_tokens(text)
    corpus_tokens |= content_tokens(evidence_meta_text)
    problems = []
    for sentence in split_sentences(body):
        if is_exempt_sentence(sentence):
            continue
        toks = content_tokens(sentence)
        if not toks:
            continue
        missing = sorted(t for t in toks if t not in corpus_tokens)
        bad_nums = sorted(n for n in number_tokens(sentence)
                          if n not in corpus_numbers)
        if missing or bad_nums:
            problems.append({"sentence": sentence[:120],
                             "missing_tokens": missing[:8],
                             "missing_numbers": bad_nums[:8]})
    if not problems:
        return _check("pass", "every substantive sentence is evidenced "
                              "(independent token/number check)")
    return _check("fail", f"{len(problems)} sentence(s) not evidenced",
                  problems=problems[:3])


def check_evidence_use(answer, facts):
    body = norm_ws(answer)
    if any(norm_ws(f) in body for f in facts):
        return _check("pass", "answer draws on expected evidence")
    return _check("fail", "answer uses none of the expected evidence")


def check_relevance(answer, query):
    q_tokens = distinctive_query_tokens(query)
    a_tokens = content_tokens(split_footer(answer)[0])
    if q_tokens & a_tokens:
        return _check("pass", "answer shares question content terms")
    return _check("fail", "answer shares no distinctive question term")


def check_citation(answer, expect, evidence_files, evidence_pages):
    body, entries = split_footer(answer)
    mode = expect.get("citation")
    forbidden_files = [f.lower() for f in (expect.get("forbidden_citation_files") or [])]
    if mode == "forbidden":
        if entries is None:
            return _check("pass", "no References footer (as required)")
        return _check("fail", f"footer present on a no-citation case: {entries}")
    if entries is None:
        return _check("fail", "expected a References footer; none found")
    cited = file_entries(entries)
    lower_cited = [c.lower() for c in cited]
    for bad in forbidden_files:
        if any(bad in entry.lower() for entry in entries):
            return _check("fail", f"forbidden source cited: {bad}")
    if mode == "unavailable":
        ok = all(e.lower().startswith("attribution unavailable") for e in entries) \
            and not cited
        if ok:
            return _check("pass", "explicit attribution-unavailable footer")
        return _check("fail", f"unusable metadata rendered as citation: {entries}")
    expected = [f.lower() for f in (expect.get("citation_files") or [])]
    if sorted(lower_cited) != sorted(expected):
        return _check("fail",
                      f"cited {sorted(lower_cited)} != expected {sorted(expected)}")
    for entry in entries:
        match = re.search(r"page\s+(\S+)", entry)
        if match:
            label = match.group(1).rstrip(";")
            fname = _FILENAME_IN_TEXT_RE.search(entry).group(0).lower()
            allowed = evidence_pages.get(fname) or set()
            if allowed and label not in {str(a) for a in allowed}:
                return _check("fail",
                              f"page {label} for {fname} not in retrieved pages "
                              f"{sorted(allowed)}")
    return _check("pass", "footer cites exactly the expected validated sources")


def check_disclosure(answer, expected):
    present = _PARTIAL_DISCLOSURE in str(answer or "")
    if expected == "partial" and present:
        return _check("pass", "partial-support disclosure present")
    if expected == "partial":
        return _check("fail", "partial-support disclosure missing")
    if expected == "none" and not present:
        return _check("pass", "no partial disclosure (not expected)")
    if expected == "none":
        return _check("fail", "unexpected partial disclosure")
    return _check("skip", "no disclosure expectation")


def check_recovery(stdout, expect, llm_calls):
    attempted = "self_refusal_recovery_attempted" in stdout
    succeeded = "self_refusal_recovery_succeeded" in stdout
    rejected = re.search(r"self_refusal_recovery_rejected \(reason=(\w+)", stdout)
    outcome = ("succeeded" if succeeded
               else ("rejected" if rejected else None))
    problems = []
    if attempted != bool(expect.get("recovery_attempted")):
        problems.append(f"attempted={attempted}, expected "
                        f"{bool(expect.get('recovery_attempted'))}")
    if (expect.get("recovery_outcome") or None) != outcome:
        problems.append(f"outcome={outcome}, expected "
                        f"{expect.get('recovery_outcome')}")
    if int(expect.get("llm_calls", 0) or 0) != llm_calls:
        problems.append(f"llm_calls={llm_calls}, expected "
                        f"{expect.get('llm_calls', 0)}")
    if problems:
        return _check("fail", "; ".join(problems),
                      attempted=attempted, outcome=outcome, llm_calls=llm_calls)
    return _check("pass", f"recovery markers match (attempted={attempted}, "
                          f"outcome={outcome}, llm_calls={llm_calls})",
                  attempted=attempted, outcome=outcome, llm_calls=llm_calls)


def score_unit_case(case: dict, outcome: dict) -> dict:
    answer = outcome.get("answer")
    expect = case.get("expect") or {}
    checks = {}
    evidence = case.get("evidence") or []
    evidence_texts = [norm_ws(e.get("text")) for e in evidence]
    evidence_meta = " ".join(f"{e.get('file_name') or ''} {e.get('page_label') or ''}"
                             for e in evidence)
    evidence_files = {str(e.get("file_name")).lower()
                      for e in evidence if e.get("file_name")}
    evidence_pages = {}
    for e in evidence:
        if e.get("file_name") and e.get("page_label") not in (None, ""):
            evidence_pages.setdefault(
                str(e["file_name"]).lower(), set()).add(str(e["page_label"]))

    if expect.get("error"):
        if outcome.get("error"):
            checks["error"] = _check(
                "pass", f"controlled retrieval failure propagated: {outcome['error']}")
        else:
            checks["error"] = _check(
                "fail", "expected the scripted retrieval failure to propagate")
        for name in ("abstention", "facts_present", "groundedness",
                     "evidence_use", "relevance", "citation", "disclosure"):
            checks[name] = _check("skip", "controlled-failure case")
        checks["recovery"] = check_recovery(outcome.get("stdout", ""), expect,
                                            outcome.get("llm_calls", 0))
        return _finalize(case, outcome, checks, answer)

    if outcome.get("error"):
        checks["error"] = _check(
            "fail", f"unexpected controlled failure: {outcome['error']}")
        for name in ("abstention", "facts_present", "groundedness",
                     "evidence_use", "relevance", "citation", "disclosure",
                     "recovery"):
            checks[name] = _check("skip", "case errored before answer")
        return _finalize(case, outcome, checks, None)

    checks["error"] = _check("pass", "no controlled failure")
    checks["abstention"] = check_abstention(answer, expect.get("abstain"))
    abstained = checks["abstention"]["verdict"] == "pass" and expect.get("abstain")

    if expect.get("facts_present"):
        checks["facts_present"] = check_facts(answer, case.get("required_facts") or [])
    else:
        checks["facts_present"] = _check("skip", "no fact expectation")

    if expect.get("forbidden_facts"):
        checks["contradiction"] = check_forbidden(answer, expect["forbidden_facts"])
    else:
        checks["contradiction"] = _check("skip", "no forbidden facts declared")

    if abstained:
        checks["groundedness"] = _check("skip", "abstention carries no content")
        checks["evidence_use"] = _check("skip", "abstention")
        checks["relevance"] = _check("skip", "abstention is the correct relevance")
    else:
        checks["groundedness"] = check_groundedness(answer, evidence_texts,
                                                    evidence_meta)
        if case.get("answerable") in (True, "partial"):
            checks["evidence_use"] = check_evidence_use(
                answer, case.get("required_facts") or [])
            checks["relevance"] = check_relevance(answer, case.get("query", ""))
        else:
            checks["evidence_use"] = _check("skip", "unanswerable case")
            checks["relevance"] = _check("skip", "unanswerable case")

    checks["citation"] = check_citation(answer, expect, evidence_files,
                                        evidence_pages)
    checks["disclosure"] = check_disclosure(answer, expect.get("disclosure"))
    checks["recovery"] = check_recovery(outcome.get("stdout", ""), expect,
                                        outcome.get("llm_calls", 0))
    return _finalize(case, outcome, checks, answer)


def _finalize(case, outcome, checks, answer):
    failed = [name for name, c in checks.items() if c["verdict"] == "fail"]
    status = "fail" if failed else "pass"
    recovery = None
    if "recovery" in checks:
        recovery = {"attempted": checks["recovery"].get("attempted"),
                    "outcome": checks["recovery"].get("outcome"),
                    "llm_calls": checks["recovery"].get("llm_calls")}
    return {
        "id": case["id"],
        "status": status,
        "failed_checks": failed,
        "diagnostics": _classify_failure(case, outcome, checks) if failed else [],
        "checks": checks,
        "final_answer": answer,
        "final_answer_sha256": (hashlib.sha256(answer.encode("utf-8")).hexdigest()
                                if isinstance(answer, str) else None),
        "recovery": recovery,
        "runtime_ms": outcome.get("runtime_ms"),
    }


def _classify_failure(case, outcome, checks) -> list:
    """Actionable failure diagnostics (honest about what is observable)."""
    diags = []
    corpus_tokens = set()
    for e in case.get("evidence") or []:
        corpus_tokens |= content_tokens(e.get("text"))
        corpus_tokens |= content_tokens(e.get("file_name"))
    if checks.get("facts_present", {}).get("verdict") == "fail":
        missing = checks["facts_present"].get("missing") or []
        for fact in missing:
            absent_from_evidence = content_tokens(fact) - corpus_tokens
            if absent_from_evidence:
                diags.append(
                    "fixture-or-evaluation-definition problem: required fact "
                    f"{fact!r} is not in the scripted evidence "
                    f"(absent tokens {sorted(absent_from_evidence)[:5]})")
            else:
                diags.append(
                    "answer-layer failure: required evidence content was "
                    "retrieved but did not survive to the final answer "
                    f"(grounding/refusal/recovery dropped it): {fact!r}")
    if checks.get("abstention", {}).get("verdict") == "fail":
        if checks["abstention"].get("reason", "").startswith("FALSE REFUSAL"):
            diags.append(
                "FALSE REFUSAL (the C002 live-defect class): grounded "
                "scripted evidence yet the pipeline returned the mandated "
                "not-found phrase")
        else:
            diags.append(
                "incorrect abstention behavior: the pipeline answered a case "
                "the dataset marks unanswerable (fabrication survived)")
    if checks.get("groundedness", {}).get("verdict") == "fail":
        probs = checks["groundedness"].get("problems") or []
        tokens = sorted({t for p in probs for t in (p.get("missing_tokens") or [])})
        diags.append(
            "grounding leak: unsupported content survived the pipeline "
            f"(tokens {tokens[:8]}), or production exempt wording changed "
            "beyond the pinned disclosure set")
    if checks.get("citation", {}).get("verdict") == "fail":
        diags.append(f"citation mismatch: {checks['citation'].get('reason')}")
    if checks.get("recovery", {}).get("verdict") == "fail":
        diags.append(f"recovery mismatch: {checks['recovery'].get('reason')}")
    if checks.get("relevance", {}).get("verdict") == "fail":
        diags.append("irrelevance: final answer shares no question content "
                     "term (proxy check)")
    if checks.get("disclosure", {}).get("verdict") == "fail":
        diags.append(f"disclosure mismatch: {checks['disclosure'].get('reason')}")
    if checks.get("contradiction", {}).get("verdict") == "fail":
        diags.append("forbidden content present (contradiction/distractor/"
                     "fabrication leaked): "
                     f"{checks['contradiction'].get('hits')}")
    if checks.get("error", {}).get("verdict") == "fail":
        diags.append(f"controlled-failure mismatch: {checks['error'].get('reason')}")
    if not diags:
        diags.append("fixture-or-evaluation-definition problem (see checks)")
    return diags


# --------------------------------------------------------------------------
# integration mode (isolated temp store, real retrieval path)
# --------------------------------------------------------------------------
def verify_isolation(temp_dir: Path) -> dict:
    """Refuse unless the store path is provably not the production store."""
    temp_dir = Path(temp_dir).resolve()
    if temp_dir == Path(ev.STORAGE_DIR).resolve():
        raise ev.GuardFailure(
            f"refusing: evaluation store path IS the production store {temp_dir}")
    try:
        chroma_guard.assert_store_path_allowed(temp_dir)
    except Exception as exc:  # noqa: BLE001 — normalize guard errors
        raise ev.GuardFailure(f"refusing: store path not allowed: {exc}")
    if not str(temp_dir).startswith(tempfile.gettempdir()):
        raise ev.GuardFailure(
            f"refusing: evaluation store {temp_dir} is outside the system "
            "temporary directory")
    return {"store": str(temp_dir), "production_store": str(ev.STORAGE_DIR)}


def retrieve_full(temp_dir: Path, collection_name: str, queries, embed_model,
                  top_k: int):
    """Real seam retrieval with FULL node text (mirrors ev.query_production_seam,
    which truncates text to 120 chars — too short for coverage scoring)."""
    from llama_index.core import Settings

    saved = (ba.STORAGE_DIR, ba.CHROMA_COLLECTION, ba._query_engine,
             ba._embedding_model, Settings._llm, Settings._embed_model)
    rows = []
    try:
        ba.STORAGE_DIR = Path(temp_dir)
        ba.CHROMA_COLLECTION = collection_name
        ba._query_engine = None
        ba._embedding_model = embed_model
        engine = ba._get_query_engine()
        for query in queries:
            nodes = engine.retrieve(query)
            rows.append([{
                "file_name": (node.metadata or {}).get("file_name"),
                "page_label": (node.metadata or {}).get("page_label")
                or (node.metadata or {}).get("page_number"),
                "score": None if node.score is None else round(float(node.score), 6),
                "text": norm_ws(getattr(node, "text", "") or ""),
            } for node in nodes[:top_k]])
    finally:
        (ba.STORAGE_DIR, ba.CHROMA_COLLECTION, ba._query_engine,
         ba._embedding_model, Settings._llm, Settings._embed_model) = saved
    return rows


def score_integration_case(case, retrieved, top_k):
    """Retrieval-layer verdicts for one case (answer layer stays in unit mode)."""
    gold_documents = case.get("gold_documents") or []
    result = {
        "id": case["id"],
        "retrieved": [{"file_name": n.get("file_name"),
                       "page_label": n.get("page_label"),
                       "score": n.get("score"),
                       "text_sha256": hashlib.sha256(
                           (n.get("text") or "").encode("utf-8")).hexdigest()}
                      for n in retrieved],
    }
    if not gold_documents:
        result.update(status="skipped",
                      reason="no retrieval expectation (scripted/unit-only case)",
                      metrics=None, fact_coverage=None, diagnostics=[])
        return result
    metrics = ev.compute_metrics(
        {"gold_documents": gold_documents,
         "gold_pages": case.get("gold_pages"),
         "min_docs_covered": case.get("min_docs_covered") or 1},
        [{"file_name": n.get("file_name"), "page_label": n.get("page_label")}
         for n in retrieved], max_k=top_k)
    result["metrics"] = metrics

    facts = case.get("required_facts") or []
    fact_cov = None
    if case.get("answerable") in (True, "partial") and facts:
        context_tokens = set()
        for n in retrieved:
            context_tokens |= content_tokens(n.get("text"))
        fact_cov = all(content_tokens(f) <= context_tokens for f in facts)
        result["fact_coverage"] = fact_cov

    diags = []
    if not metrics["coverage"]:
        if metrics["first_hit_rank"] is None:
            diags.append("expected evidence absent from retrieval (top-"
                         f"{top_k})")
        else:
            diags.append("relevant evidence retrieved at inadequate "
                         f"rank/distinct-doc count (first rank "
                         f"{metrics['first_hit_rank']}, min_docs "
                         f"{case.get('min_docs_covered') or 1})")
    if fact_cov is False and metrics["coverage"]:
        diags.append("evidence file retrieved but required fact text not in "
                     "the retrieved context (context-assembly gap)")

    ok = metrics["coverage"] and (fact_cov is not False)
    result.update(status="pass" if ok else "fail",
                  reason=("gold evidence and required facts retrieved "
                          f"within top-{top_k}" if ok else
                          "retrieval assertions failed"),
                  fact_coverage=fact_cov,
                  diagnostics=diags)
    return result


def run_integration(cases, top_k, run_note):
    """Build a fresh isolated store, retrieve through the real seam, score."""
    wanted = set()
    for case in cases:
        for e in case.get("evidence") or []:
            if e.get("source") in ("corpus", "fixture") and e.get("file_name"):
                wanted.add(e["file_name"])
        for f in case.get("gold_documents") or []:
            if _source_path(f).exists():
                wanted.add(f)
    files = [_source_path(f) for f in sorted(wanted)]

    real_client = None
    recorded = []
    guard_before = ev.guard_snapshot(ev.GUARD_FILES)
    if any(v is None for v in guard_before.values()):
        raise ev.GuardFailure(
            f"production store files missing at run start: {guard_before}")
    embedder = HashedTfEmbedding()
    with tempfile.TemporaryDirectory(prefix="rag_ae_") as td:
        temp_dir = Path(td)
        isolation = verify_isolation(temp_dir)
        try:
            real_client, recorded = ev.instrument_client_paths()
            index, collection, entries = ev.build_eval_index(
                temp_dir, EVAL_COLLECTION, files, embedder)
            retention = {name: entry.get("status")
                         for name, entry in sorted(entries.items())}
            failed = [n for n, s in retention.items() if s == "failed"]
            if failed:
                raise ev.GuardFailure(
                    f"fixture/corpus ingest failed in temp store: {failed}")
            if set(retention) != {f.name for f in files}:
                raise ev.GuardFailure(
                    f"temp store retention mismatch: {retention}")
            per_case = []
            for case in cases:
                rows = retrieve_full(temp_dir, EVAL_COLLECTION,
                                     [case["query"]], embedder, top_k)
                per_case.append(score_integration_case(case, rows[0], top_k))
        finally:
            ev.release_chroma_clients()
            if real_client is not None:
                import chromadb
                chromadb.PersistentClient = real_client

    guard_after = ev.guard_snapshot(ev.GUARD_FILES)
    ev.guard_assert_unchanged(guard_before, guard_after, "rag_ae integration")
    stray = [p for p in recorded
             if not str(Path(p).resolve()).startswith(
                 str(Path(temp_dir).resolve()))]
    if stray:
        raise ev.GuardFailure(f"clients constructed off the temp store: {stray}")
    return {
        "per_case": per_case,
        "aggregate": aggregate_retrieval(per_case),
        "isolation": {
            **isolation,
            "collection": EVAL_COLLECTION,
            "files_ingested": retention,
            "client_paths": sorted(set(recorded)),
            "all_clients_on_temp_store": True,
            "guard_files_unchanged": True,
        },
        "note": run_note,
    }


def aggregate_retrieval(per_case):
    scored = [c for c in per_case if c.get("metrics")]
    n = len(scored)
    if not n:
        return {"cases_scored": 0}
    out = {"cases_scored": n}
    for key in ("hit@1", "hit@3", "coverage"):
        out[key] = round(sum(1 for c in scored if c["metrics"].get(key)) / n, 4)
    for key in ("precision@1", "precision@3", "reciprocal_rank", "ndcg@5"):
        out[key] = round(sum(float(c["metrics"].get(key) or 0.0) for c in scored) / n, 4)
    cov = [c for c in scored if c.get("fact_coverage") is not None]
    if cov:
        out["fact_coverage"] = round(
            sum(1 for c in cov if c["fact_coverage"]) / len(cov), 4)
    return out


# --------------------------------------------------------------------------
# local-model mode (opt-in; never remote; NOT executed by the RAG-AE task)
# --------------------------------------------------------------------------
def local_model_refusal_reason(enable_live: bool) -> str | None:
    if not enable_live:
        return ("local-model mode is opt-in and disabled by default: pass "
                "--enable-live to run it deliberately")
    if LLM_PROVIDER != "ollama":
        return (f"LLM_PROVIDER={LLM_PROVIDER!r}: this mode may only use the "
                "local Ollama provider and never falls back to a remote one")
    return None


def generate_with_local_model(case) -> str:
    """Raw reply from the LOCAL chat model over the case's own evidence."""
    from app.llm_client import get_llm, llm_text

    context = "\n\n".join(e.get("text", "") for e in (case.get("evidence") or []))
    prompt = ("Context information is below.\n---------------------\n"
              f"{context}\n---------------------\n"
              "Answer the question using ONLY the passages above. Do not use "
              "outside knowledge. Never invent figures.\n"
              f"Question: {case['query']}\nAnswer: ")
    return str(llm_text(get_llm().invoke(prompt)) or "").strip()


# --------------------------------------------------------------------------
# aggregation + reporting
# --------------------------------------------------------------------------
def aggregate_answers(results):
    """Per-dimension applicable/pass/rate over executed unit cases."""
    dims = {}
    for res in results:
        for name, check in res["checks"].items():
            slot = dims.setdefault(name, {"applicable": 0, "passed": 0})
            if check["verdict"] in ("pass", "fail"):
                slot["applicable"] += 1
                if check["verdict"] == "pass":
                    slot["passed"] += 1
    for slot in dims.values():
        slot["rate"] = (round(slot["passed"] / slot["applicable"], 4)
                        if slot["applicable"] else None)
    return dims


def compute_abstention(cases, results):
    by_id = {r["id"]: r for r in results}
    tp = fp = fn = tn = 0
    for case in cases:
        res = by_id.get(case["id"])
        if not res:
            continue
        check = res["checks"].get("abstention") or {}
        if check.get("verdict") is None:
            continue
        expected = bool((case.get("expect") or {}).get("abstain"))
        # actual abstention: from the final answer itself (authoritative)
        answer = res.get("final_answer")
        if not isinstance(answer, str):
            continue
        actual = answer.startswith(NOT_FOUND_PHRASE)
        if expected and actual:
            tp += 1
        elif expected and not actual:
            fn += 1
        elif not expected and actual:
            fp += 1
        else:
            tn += 1
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    return {"true_abstain": tp, "false_abstain": fp,
            "missed_abstain": fn, "true_answer": tn,
            "precision": precision, "recall": recall}


def by_category(cases, results):
    by_id = {r["id"]: r for r in results}
    out = {}
    for case in cases:
        res = by_id.get(case["id"])
        if not res or res["status"] not in ("pass", "fail"):
            continue
        for label in case.get("labels") or ["<none>"]:
            slot = out.setdefault(label, {"applicable": 0, "passed": 0})
            slot["applicable"] += 1
            if res["status"] == "pass":
                slot["passed"] += 1
    for slot in out.values():
        slot["rate"] = round(slot["passed"] / slot["applicable"], 4)
    return out


def strip_volatile(record: dict) -> dict:
    """Deterministic-comparison view: drop run identity, timings, and paths.

    Run-identity fields (run_id, timestamp, runtime) and any absolute output
    or disposable-store paths are run *configuration*, not behavior. The
    disposable-store location is still evidenced deterministically as a
    TEMP_DIR placeholder plus the guard/network counters, which are the
    parts that carry enforcement information.
    """
    rec = json.loads(json.dumps(record))
    for key in ("run_id", "timestamp_utc", "runtime_s"):
        rec.pop(key, None)
    if isinstance(rec.get("config"), dict):
        rec["config"].pop("out", None)   # where results were written
    for case in rec.get("cases") or []:
        case.pop("runtime_ms", None)
    for row in (rec.get("retrieval") or {}).get("per_case", []) or []:
        for node in row.get("retrieved") or []:
            node.pop("score", None)   # float display rounding only differs
    # Disposable-store paths are per-run tempdir probes; mask the location,
    # keep the fact that a disposable store was created in the temp root.
    for iso in (rec.get("isolation"),
                (rec.get("retrieval") or {}).get("isolation")):
        if isinstance(iso, dict):
            if isinstance(iso.get("store"), str) and "temp" in iso["store"].lower():
                iso["store"] = "<tempdir>"
            if isinstance(iso.get("client_paths"), list):
                iso["client_paths"] = [
                    "<tempdir>" if isinstance(p, str) and "temp" in p.lower()
                    else p for p in iso["client_paths"]]
    return rec


def render_summary(record: dict) -> str:
    lines = []
    counts = record["counts"]
    lines.append("=" * 72)
    lines.append(f"RAG-AE {record['mode']} — offline deterministic evaluation")
    lines.append("=" * 72)
    lines.append(
        f"discovered {counts['discovered']} | valid/executable "
        f"{counts['executable']} | executed {counts['executed']} | passed "
        f"{counts['passed']} | failed {counts['failed']} | skipped "
        f"{counts['skipped']} | unresolved {counts['unresolved']}")
    for case in record["cases"]:
        marker = {"pass": "PASS", "fail": "FAIL",
                  "skipped": "SKIP", "unresolved": "UNRES"}.get(case["status"],
                                                                case["status"])
        detail = case.get("reason") or "; ".join(case.get("diagnostics") or []) \
            or ",".join(case.get("failed_checks") or [])
        lines.append(f"  [{marker}] {case['id']:<6} {detail[:110]}")
    if record.get("answer_quality"):
        aq = record["answer_quality"]
        lines.append("-- answer-quality dimensions (applicable/pass/rate) --")
        for name, slot in sorted(aq["by_dimension"].items()):
            lines.append(f"  {name:<15} {slot['applicable']:>3} "
                         f"{slot['passed']:>3}  {slot['rate']}")
        ab = aq["abstention"]
        lines.append(f"  abstention precision={ab['precision']} "
                     f"recall={ab['recall']} (tp={ab['true_abstain']} "
                     f"fp={ab['false_abstain']} fn={ab['missed_abstain']})")
    if record.get("retrieval"):
        agg = record["retrieval"]["aggregate"]
        lines.append("-- retrieval metrics (embedding double: lexical!) --")
        lines.append(f"  {json.dumps(agg)}")
    iso = record["isolation"]
    lines.append(f"-- isolation: store={iso.get('store', 'none (unit mode)')} "
                 f"guard_unchanged={iso.get('guard_files_unchanged')} "
                 f"network_attempts={len(iso.get('network_attempts') or [])}")
    lines.append(f"-- external calls: {json.dumps(record['external_calls'])}")
    lines.append(f"-- runtime {record.get('runtime_s')}s; results: "
                 f"{record['config']['out']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def _git_info():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"],
                                cwd=PROJECT_ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    cwd=PROJECT_ROOT, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
        return {"head": head, "branch": branch, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"head": "unknown", "branch": "unknown", "dirty": None}


def main(argv=None) -> int:
    chroma_guard.arm()                    # fail-closed for this process
    chroma_guard.install_persistent_client_wrapper()
    parser = argparse.ArgumentParser(description="RAG-AE answer-quality harness")
    parser.add_argument("--mode", choices=("unit", "integration", "local-model"),
                        default="unit")
    parser.add_argument("--gold", default=str(GOLD_PATH))
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing --out artifact")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--enable-live", action="store_true",
                        help="explicitly opt in to local-model mode")
    args = parser.parse_args(argv)
    out_path = Path(args.out) if args.out else DEFAULT_OUTS[args.mode]
    started_all = time.perf_counter()

    # Output safety FIRST: evidence is never overwritten silently.
    if out_path.exists() and not args.force:
        print(f"REFUSING TO OVERWRITE existing artifact: {out_path} "
              f"(sha256 {ev.sha256_file(out_path)}). Re-run with --force or a "
              "new --out.")
        return 3

    if args.mode == "local-model":
        reason = local_model_refusal_reason(args.enable_live)
        if reason:
            print(f"REFUSING local-model run: {reason}")
            return 2

    # Dataset: load + validate BEFORE anything else.
    gold_path = Path(args.gold)
    try:
        ds, gold_sha = load_gold_v2(gold_path)
    except Exception as exc:  # noqa: BLE001
        print(f"DATASET ERROR: cannot load {gold_path}: {exc}")
        return 1
    source_files = {e.get("file_name")
                    for case in ds.get("cases", [])
                    for e in (case.get("evidence") or [])
                    if e.get("source") in ("corpus", "fixture")
                    and e.get("file_name")}
    try:
        source_texts = load_source_texts(source_files) if source_files else {}
    except ValueError as exc:
        print(f"DATASET ERROR: {exc}")
        return 1
    errors = validate_dataset(ds, source_texts)
    if errors:
        print(f"DATASET INVALID ({len(errors)} problem(s)):")
        for msg in errors:
            print(f"  - {msg}")
        return 1

    cases = ds["cases"]
    executable = [c for c in cases if c.get("status") == "executable"]
    unresolved = [c for c in cases if c.get("status") == "unresolved"]

    net = NetworkGuard()
    net.install()
    guard_before = ev.guard_snapshot(ev.GUARD_FILES)
    record = None
    exit_code = 0
    try:
        if args.mode == "unit":
            selected = [c for c in executable if "unit" in (c.get("modes") or [])]
            results = []
            for case in selected:
                outcome = run_unit_case(case)
                results.append(score_unit_case(case, outcome))
            dims = aggregate_answers(results)
            record = _assemble_record(args, ds, gold_sha, cases, results,
                                      dims, compute_abstention(selected, results),
                                      by_category(selected, results),
                                      retrieval=None, isolation_extra={})
        elif args.mode == "integration":
            selected = [c for c in executable
                        if "integration" in (c.get("modes") or [])]
            run_note = ("embedding double = HashedTfEmbedding (lexical TF); "
                        "retrieval numbers are lexical-overlap numbers, NOT "
                        "production-embedding numbers (production baseline: "
                        "benchmarks/rag_baseline_v3.json). The answer layer is "
                        "not re-scored in integration mode.")
            integ = run_integration(selected, args.top_k, run_note)
            results = integ["per_case"]
            record = _assemble_record(args, ds, gold_sha, cases, results,
                                      dims=None, abstention=None,
                                      categories=None,
                                      retrieval=integ,
                                      isolation_extra=integ["isolation"])
        else:  # local-model (opt-in; guarded above)
            selected = [c for c in executable if "unit" in (c.get("modes") or [])]
            results = []
            for case in selected:
                raw = generate_with_local_model(case)
                scripted = dict(case, raw_response=raw)
                outcome = run_unit_case(scripted)
                results.append(score_unit_case(scripted, outcome))
            dims = aggregate_answers(results)
            record = _assemble_record(args, ds, gold_sha, cases, results,
                                      dims, compute_abstention(selected, results),
                                      by_category(selected, results),
                                      retrieval=None, isolation_extra={})
            record["external_calls"]["chat_model"] = f"local:{LLM_PROVIDER}"
        record["isolation"]["network_attempts"] = list(net.attempts)
        if net.attempts:
            print(f"NETWORK GUARD TRIPPED: {net.attempts}")
            exit_code = 2
    except ev.GuardFailure as exc:
        print(f"ISOLATION GUARD FAILURE: {exc}")
        exit_code = 2
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"RUNTIME FAILURE: {type(exc).__name__}: {exc}")
        exit_code = 2
    finally:
        net.uninstall()

    if exit_code == 2:
        return exit_code

    guard_after = ev.guard_snapshot(ev.GUARD_FILES)
    try:
        ev.guard_assert_unchanged(guard_before, guard_after, "rag_ae run")
    except ev.GuardFailure as exc:
        print(f"ISOLATION GUARD FAILURE: {exc}")
        return 2
    record["isolation"]["guard_files_unchanged"] = True
    record["config"]["out"] = str(out_path)
    record["runtime_s"] = round(time.perf_counter() - started_all, 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(render_summary(record))
    failed = record["counts"]["failed"]
    return 4 if failed else 0


def _assemble_record(args, ds, gold_sha, cases, results, dims, abstention,
                     categories, retrieval, isolation_extra) -> dict:
    by_id = {r["id"]: r for r in results}
    case_records = []
    for case in cases:
        res = by_id.get(case["id"])
        if res:
            case_records.append(res)
        else:
            reason = ("unresolved: excluded from all scored metrics"
                      if case.get("status") == "unresolved"
                      else f"skipped in mode {args.mode} (modes="
                           f"{case.get('modes')})")
            case_records.append({"id": case["id"],
                                 "status": "unresolved"
                                 if case.get("status") == "unresolved"
                                 else "skipped",
                                 "reason": reason})
    counts = {
        "discovered": len(cases),
        "executable": sum(1 for c in cases if c.get("status") == "executable"),
        "executed": sum(1 for r in case_records
                        if r["status"] in ("pass", "fail")),
        "passed": sum(1 for r in case_records if r["status"] == "pass"),
        "failed": sum(1 for r in case_records if r["status"] == "fail"),
        "skipped": sum(1 for r in case_records if r["status"] == "skipped"),
        "unresolved": sum(1 for r in case_records if r["status"] == "unresolved"),
    }
    gold_path = Path(args.gold)
    try:
        gold_display = str(gold_path.relative_to(PROJECT_ROOT))
    except ValueError:
        gold_display = str(gold_path)
    offline = args.mode != "local-model"
    return {
        "harness": "rag_ae",
        "harness_version": HARNESS_VERSION,
        "mode": args.mode,
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_info(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": ev.package_versions([
                "chromadb", "llama-index-core", "langchain-core", "pypdf",
                "docx2txt", "python-dotenv"]),
        },
        "dataset": {"path": gold_display,
                    "sha256": gold_sha, "id": ds.get("id"),
                    "version": ds.get("version")},
        "config": {"out": None, "top_k": args.top_k,
                   "collection": EVAL_COLLECTION if args.mode == "integration" else None,
                   "embedding": ("HashedTfEmbedding(double)" if args.mode == "integration"
                                 else "none (scripted evidence)"),
                   "chat_model": ("scripted-double" if args.mode != "local-model"
                                  else f"local:{LLM_PROVIDER}"),
                   "books_verifier": "off (production default)",
                   "books_min_score": BOOKS_MIN_SCORE},
        "isolation": {"guard_armed": True,
                      "guard_files_unchanged": None,
                      "store": (isolation_extra or {}).get("store"),
                      "network_attempts": None,
                      **{k: v for k, v in (isolation_extra or {}).items()
                         if k != "store"}},
        "external_calls": {"network": 0,
                           "chat_model": ("scripted-double"
                                          if args.mode != "local-model"
                                          else f"local:{LLM_PROVIDER}"),
                           "embedding": ("HashedTfEmbedding(double)"
                                         if args.mode == "integration" else "none"),
                           "offline_deterministic": offline},
        "counts": counts,
        "answer_quality": ({"by_dimension": dims, "abstention": abstention,
                             "by_category": categories}
                            if dims is not None else None),
        "retrieval": ({"aggregate": retrieval["aggregate"],
                       "per_case": retrieval["per_case"],
                       "isolation": retrieval["isolation"],
                       "note": retrieval["note"]}
                      if retrieval is not None else None),
        "definitions": DEFINITIONS,
        "cases": case_records,
        "runtime_s": None,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
