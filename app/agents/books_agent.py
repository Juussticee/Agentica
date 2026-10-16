"""Books agent: answers from the local Chroma index built by ingest_books.py.

Known gap (flagged to the user, per instructions): embeddings still come from
Ollama's nomic-embed-text; the Gemini embedding API needs separate setup and
was deliberately NOT wired in.
"""

import re
import time

from llama_index.core import (
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app import verifier
from app.config import (
    BOOKS_MIN_SCORE,
    BOOKS_TOP_K,
    BOOKS_VERIFIER,
    BOOKS_VERIFIER_EVIDENCE_MAX_CHARS,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    NOT_FOUND_PHRASE,
    OLLAMA_BASE_URL,
    STORAGE_DIR,
)
from app.llm_client import get_llm, llm_text
from app.state import AgentState, with_agent_output

_embedding_model = None
_query_engine = None

# Sentence-level refusal markers (spec section 2.1): a response counts as a
# refusal only when EVERY sentence claims the books lack the requested
# information. A mixed response ("...however the books do cover X") has
# non-marker sentences and is treated as partial support — the covered parts
# must be kept, not collapsed into a refusal (guaranteed no fabricated
# content: the covered text came from the retrieved evidence).
#
# Scope limits (deliberate): markers must stay rare, whole-claim phrases so
# that a single sentence mixing a supported fact WITH an absent-detail note
# ("...though it is not covered by the passages") is never collapsed — that
# is why "not covered by the passages" and similar are NOT markers. The
# detector classifies RESPONSE SHAPE, not truth: fabricated sentences remain
# a prompt-level contract ("do not use outside knowledge"), not a regex one.
_REFUSAL_MARKERS = (
    "not found in your books",
    "not found in the books",
    "no information about",          # "There is no information about X in the books."
    "not contain the answer",        # "(does/do) not contain the answer"
)


def _refusal_only(answer_text: str) -> bool:
    """Deterministic refusal detector: every sentence is a not-found claim.

    Robust against the small transcription/paraphrase models that reword the
    instructed refusal phrase (the exact-phrase check missed those, leaving
    reworded refusals carrying a references footer). Moving to a
    sentence-level marker check keeps determinism and never misclassifies a
    partial-support answer, because that answer contains substantive
    non-marker sentences.

    This classifies response SHAPE only — it is never proof of factual
    grounding; content support is enforced separately by
    `_enforce_grounding` below.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    if not sentences:
        return False
    return all(
        any(marker in s.lower() for marker in _REFUSAL_MARKERS)
        for s in sentences
    )


# --- citation integrity (M3) ----------------------------------------------
# Retrieved documents and model output are UNTRUSTED DATA. Two rules follow:
#
#  1. The model never authors the References footer. It is rebuilt from the
#     retrieved nodes' OWN metadata, restricted to the nodes that actually
#     supported retained claims (_reference_footer). A filename or page the
#     model wrote can therefore never be accepted as a verified citation.
#  2. Citation-shaped text INSIDE the model's output (a fabricated footer
#     copied verbatim out of a retrieved passage, or an injected instruction
#     block) is stripped before any validated footer is appended
#     (_strip_model_citations), and instruction-boundary phrases are dropped
#     as content (_INJECTION_CUES) — monotone: these rules can only remove
#     text, never add or bless it.

_FOOTER_BLOCK_RE = re.compile(
    r"^[ \t]*(?:references|sources|citations)[ \t]*:", re.I
)

# Instruction-boundary cues: machine-directed phrases that never occur in
# the sample corpus's book prose but are common inside prompt-injection
# payloads. Deliberately narrow (over-refusal risk): a paraphrase without
# any cue is still only bounded by the token support gate — that residual is
# a documented limitation, not a claimed guarantee.
_INJECTION_CUES = (
    "ignore previous", "ignore all previous", "ignore the previous",
    "ignore above", "ignore the above", "disregard previous",
    "disregard the above", "disregard these instructions",
    "system prompt", "developer message", "jailbreak",
    "reveal your instructions", "output your instructions",
    "new instructions", "you are now in", "do not follow",
)


def _strip_model_citations(answer_text: str) -> str:
    """Remove any References/Sources/Citations block the model emitted.

    Such a block is either fabricated (the model cannot know which pages
    were retrieved well enough to be trusted) or copied from text inside a
    retrieved document — in both cases it must not survive alongside the
    agent's own validated footer. Removes the marker line and its
    continuation block (until a blank line or EOF); monotone, never adds.
    """
    lines = answer_text.splitlines()
    kept = []
    i = 0
    while i < len(lines):
        if _FOOTER_BLOCK_RE.match(lines[i]):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept).strip()


# --- deterministic grounding safeguard (spec section 2.1) -----------------
# _refusal_only classifies response SHAPE, never factual grounding. This is
# the closest deterministic substitute for entailment within the current
# architecture: every SUBSTANTIVE sentence of the final answer must have ALL
# of its content tokens present in the retrieved passages (numbers exact,
# function words stop-listed, a coarse plural stem applied to both sides).
# Unsupported sentences are dropped and explicitly disclosed; an answer with
# nothing substantiated degrades to the mandated refusal (no footer).
#
# Honest limits, NOT hidden: token PRESENCE cannot prove entailment. Two
# narrow polarity gates close the two clearest overlap traps (a claim whose
# own evidence negates it, and a number whose own clause is negated) and can
# only ever mark content UNSUPPORTED — they never invent support. Residual
# gap (documented, tested): a claim that splices tokens across unrelated
# passages cannot be detected without real entailment checking, and
# tightening to single-passage-sentence coverage would refuse legitimate
# combined summaries. Granularity is one sentence (a sentence mixing
# supported and fabricated tokens is dropped whole — the conservative side).
# Real entailment would need a separate verifier model/dependency; that
# trade-off is deliberately NOT taken (no framework, no new deps).
_EXEMPT_PHRASES = (
    # "What is missing" disclosures - they talk about coverage, not content:
    "do not address", "not covered", "not covered by", "cannot be answered",
    "does not state", "does not mention", "no mention", "not in scope",
)

_PARTIAL_DISCLOSURE = (
    "Note: the answer above is limited to the parts supported by the "
    "retrieved book passages; the rest was not found in the books."
)

# M4B verifier disclosures. Fixed strings — the verifier's own "reason"
# text NEVER reaches the user (it comes from an unvalidated model), so a
# compromised or confused verifier cannot inject content through reasons.
# Each note states exactly what happened; no resolution is ever invented.
_VERIFIER_UNAVAILABLE_NOTE = (
    "Note: semantic verification was unavailable for some statements, so "
    "they could not be confirmed and were removed."
)
_VERIFIER_CONFLICT_NOTE = (
    "Note: the retrieved passages conflict with part of this answer; the "
    "conflicting statements were removed and no resolution was invented."
)
_VERIFIER_REMOVED_NOTE = (
    "Note: statements the verifier could not confirm against the retrieved "
    "passages were removed."
)

# Polarity gate inputs. Cues are deliberately few and unambiguous: "only"
# and "prevents" are NOT negations, and including them would refuse valid
# answers such as the corpus's own honey sentence.
_NEGATION_CUES = frozenset({"not", "no", "never", "cannot", "can't",
                            "without", "neither", "nor", "none"})

# Clause boundaries used only by the polarity gates: sentence end, comma,
# semicolon, "and", "but". Splitting generously makes the gates fire less
# often (permissive on ambiguity) — they must never over-refuse.
# A comma preceded by a digit is a THOUSANDS separator ("20,000"), never a
# clause boundary — splitting there tore numbers apart and silently defeated
# the number-provenance gate. A decimal point is safe because the
# sentence-boundary rule requires whitespace after the punctuation.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?<!\d)[,;]\s*|\s+(?:and|but)\s+"
)


def _clauses(text: str) -> list:
    return [part.strip() for part in _CLAUSE_SPLIT_RE.split(text) if part and part.strip()]


def _has_negation(text: str) -> bool:
    return any(word in _NEGATION_CUES
               for word in re.findall(r"[a-z']+", text.lower()))


def _number_tokens(text: str) -> set:
    return {t for t in _support_tokens(text) if t.isdigit()}


def _polarity_conflict(sentence: str, evidence_clauses: list) -> bool:
    """True when the evidence that covers this sentence states the OPPOSITE
    polarity (conservative: the claim is then treated as unsupported).

    Two narrow rules, both requiring unambiguously matching evidence:
      A. some single clause covers EVERY content token of the sentence and
         disagrees on negation;
      B. some clause containing one of the sentence's NUMBERS disagrees on
         negation (catches a figure lifted from negated/changed evidence,
         e.g. conflicting retrieved passages).
    """
    sentence_neg = _has_negation(sentence)
    toks = _support_tokens(sentence)
    numbers = _number_tokens(sentence)
    for clause, clause_toks, clause_neg in evidence_clauses:
        if clause_neg != sentence_neg:
            if toks and toks <= clause_toks:
                return True                     # rule A
            if numbers & clause_toks & _number_tokens(clause):
                return True                     # rule B
    return False


_STOP_TOKENS = frozenset("""
    the a an and or but nor of in on at to by with for as from into onto up
    down out over under above below about between within through during
    after before since until while because so than then thus hence however
    therefore also only even still too very just any all some each both few
    many much more most other another such same per via yes
    i we you he she it they them his her its their there here
    is are was were be been being am do does did done has have had having
    can could may might must shall should will would ought
    question questions answered answer answers reply responses according
    based provided given states stated says say saying mention mentions
    mentioned notes noted asks asked ask explain explains explained
    described describes adds added information info detail details topic
    topics document documents book books guide guides manual manuals
    passage passages context summaries requested request
    cover covers covered covering
    anything something nothing everything
""".split())


def _support_tokens(text: str) -> set:
    """Content tokens of a text: exact numbers, stop-filtered stemmed words."""
    tokens = set()
    for raw in re.findall(r"\$?\d[\d,.]*%?|[a-z]+", text.lower()):
        token = raw.replace(",", "").replace("$", "").rstrip("%.")
        if token.isdigit():
            if token:
                tokens.add(token)
        elif token not in _STOP_TOKENS and len(token) > 2:
            if token.endswith("s") and len(token) > 3 \
                    and not token.endswith(("ss", "us", "is")):
                token = token[:-1]
            tokens.add(token)
    return tokens


# Inline citation shapes inside MODEL text. Each is validated against the
# retrieved nodes' own metadata (filename, page) or the retrieved passage
# text (tables/figures) — a mismatch means the model invented it and the
# sentence is treated as unsupported (dropped + disclosed, never blessed).
_FILENAME_RE = re.compile(r"\b[\w][\w\-]*\.(?:pdf|docx)\b", re.I)
_PAGE_REF_RE = re.compile(r"\bpage\s+(\d+)\b", re.I)
_FIGURE_REF_RE = re.compile(
    r"\b(?:table|figure|fig\.|chart|exhibit)\s+(\d+)\b", re.I
)


def _metadata_tokens(meta: dict) -> set:
    """Support tokens contributed by a node's OWN source metadata.

    LlamaIndex inlines non-excluded metadata (page_label, file_path) into the
    LLM context, so the model legitimately sees source identity — claims
    drawn from it are evidence-backed. The literal word "page" counts too
    (the metadata string literally carries it). Metadata values are added to
    the SUPPORT corpus only; never to the polarity clauses — metadata cannot
    affirm or negate anything.
    """
    if not isinstance(meta, dict):
        return set()
    toks = set()
    for key in ("file_name", "book_title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            toks |= _support_tokens(value.strip())
    has_page = False
    for key in ("page_label", "page_number"):
        value = meta.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            toks.add(str(value))
            has_page = True
        elif isinstance(value, str) and value.strip().isdigit():
            toks.add(value.strip())
            has_page = True
    if has_page:
        toks.add("page")
    return toks


def _inline_citation_error(sentence: str, files: set, pages: set,
                           corpus_numbers: set) -> str | None:
    """Why this sentence's inline citations are unverifiable, or None.

    filenames/pages must match the RETRIEVED nodes' metadata exactly; table
    and figure identifiers must appear in the retrieved passage text (no
    table/figure metadata exists, so the passage text is the only evidence).
    """
    for match in _FILENAME_RE.finditer(sentence):
        if match.group(0).lower() not in files:
            return "filename not among retrieved sources"
    for match in _PAGE_REF_RE.finditer(sentence):
        if match.group(1) not in pages:
            return "page not among retrieved pages"
    for match in _FIGURE_REF_RE.finditer(sentence):
        if match.group(1) not in corpus_numbers:
            return "table/figure number not in retrieved passages"
    return None


def _source_ref(meta: dict) -> str | None:
    """The reference string for one node's metadata, or None when the
    metadata cannot honestly identify a source (missing/non-string file
    identity, malformed page). Never invents a placeholder filename."""
    if not isinstance(meta, dict):
        return None
    title = meta.get("file_name") or meta.get("book_title")
    if not isinstance(title, str) or not title.strip():
        return None
    title = title.strip()
    page = meta.get("page_label")
    if page is None:
        page = meta.get("page_number")
    if isinstance(page, bool):
        page = None
    elif isinstance(page, float):
        page = int(page) if page.is_integer() else None
    if isinstance(page, int):
        page_s = str(page)
    elif isinstance(page, str):
        page_s = page.strip()
        # Display guard: only short identifier-like labels (arabic/roman
        # numerals) — junk metadata must not be rendered as a citation.
        if not (0 < len(page_s) <= 16
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\- ]*", page_s)):
            page_s = None
    else:
        page_s = None
    return f"{title}, page {page_s}" if page_s else title


def _reference_footer(source_nodes: list, used_indices: list) -> str:
    """Validated References footer from exactly the supporting nodes.

    Nodes without usable source metadata are counted and disclosed as
    "attribution unavailable" (the evidence exists, the provenance does not)
    instead of being cited under an invented name. Returns "" when there is
    nothing citable and nothing to disclose.
    """
    refs, seen, uncitable = [], set(), 0
    for idx in used_indices:
        meta = getattr(source_nodes[idx], "metadata", None)
        ref = _source_ref(meta if isinstance(meta, dict) else None)
        if ref is None:
            uncitable += 1
            continue
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    if not refs and not uncitable:
        return ""
    if refs:
        body = "; ".join(refs)
        if uncitable:
            body += (f"; attribution unavailable for {uncitable} "
                     "supporting passage(s)")
    else:
        body = (f"attribution unavailable for {uncitable} "
                "supporting passage(s)")
    return "\n\nReferences: " + body


def _claim_reference_footer(source_nodes: list, retained: list) -> str:
    """M4C claim-level References footer (verification-enabled mode only).

    One numbered line per retained claim, in answer order:

        [n] <ref>[; <ref>...] (verifier-supported: <claim text>)

    The parenthetical is deliberately ATTRIBUTED ("verifier-supported"),
    never stated as fact: it reports the bounded verifier's per-claim
    judgment (BOOKS_VERIFIER=local) — a heuristic/small-model verdict on
    the shown passages — NOT independently established semantic
    entailment. The refs on the line are likewise the candidate evidence
    the verifier cited (ids validated against that claim's own evidence),
    not proof of provenance.

    Each line cites ONLY the nodes attributed to that claim (the evidence
    the verifier confirmed for it, or the deterministic covering set when
    attribution was not narrowed), so a source supporting one claim is
    never presented as supporting another. Reference strings reuse the
    same validated `_source_ref` values as the answer-level footer:
    missing or unusable metadata yields the explicit "attribution
    unavailable" marker per claim — never an invented filename, page, or
    path. Returns "" when no claim has anything citable or disclosable.
    """
    lines = []
    for n, claim in enumerate(retained, start=1):
        refs, seen, uncitable = [], set(), 0
        for idx in claim["nodes"]:
            meta = getattr(source_nodes[idx], "metadata", None)
            ref = _source_ref(meta if isinstance(meta, dict) else None)
            if ref is None:
                uncitable += 1
                continue
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
        if not refs and not uncitable:
            continue          # nothing citable and nothing to disclose
        if refs:
            body = "; ".join(refs)
            if uncitable:
                body += (f"; attribution unavailable for {uncitable} "
                         "supporting passage(s)")
        else:
            body = (f"attribution unavailable for {uncitable} "
                    "supporting passage(s)")
        lines.append(f"[{n}] {body} (verifier-supported: {claim['text']})")
    if not lines:
        return ""
    return "\n\nReferences:\n" + "\n".join(lines)


def _grounded_repair(answer_text: str, source_nodes: list):
    """Return (mode, kept_sentences, claims):
    ("refusal", [], []), ("ok", kept, claims), or ("partial", kept, claims).

    `claims` lists the RETAINED substantive sentences in order, each as
    {"text", "idx" (index into kept_sentences), "nodes"} where `nodes`
    holds the source_nodes indices that support that claim (direct cover,
    else contributing nodes of a cross-passage splice). The References
    footer is built from the nodes of claims that survive verification — a
    retrieved-but-unused passage, or a claim later removed by the M4B
    verifier, is never cited.
    """
    corpus = set()
    corpus_numbers = set()      # from passage TEXT only (tables/figures)
    evidence_clauses = []       # (clause, clause_tokens, clause_negated)
    node_toks = []              # per-node text + own-metadata tokens
    files, pages = set(), set()  # inline-citation allow-lists (retrieved)
    for node in source_nodes:
        text = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        text_toks = _support_tokens(text)
        all_toks = text_toks | _metadata_tokens(meta)
        node_toks.append(all_toks)
        corpus |= all_toks
        corpus_numbers |= {t for t in text_toks if t.isdigit()}
        for key in ("file_name", "book_title"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                files.add(value.strip().replace("\\", "/").rsplit("/", 1)[-1]
                          .lower())
        for key in ("page_label", "page_number"):
            value = meta.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                pages.add(str(value))
            elif isinstance(value, str) and 0 < len(value.strip()) <= 16:
                pages.add(value.strip())
        for clause in _clauses(text):
            evidence_clauses.append(
                (clause, _support_tokens(clause), _has_negation(clause))
            )

    kept, has_supported, removed = [], False, False
    claims = []       # supported claims: {"text", "idx" into kept, "nodes"}

    def _covering_nodes(toks: set) -> list:
        """Nodes covering this claim alone, else (splice across passages)
        every node contributing tokens to it."""
        if not toks:
            return []
        chosen = [i for i, nt in enumerate(node_toks) if toks <= nt]
        if not chosen:
            chosen = [i for i, nt in enumerate(node_toks) if toks & nt]
        return chosen

    for sentence in (s.strip() for s in
                     re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()):
        low = sentence.lower()
        toks = set()
        if any(cue in low for cue in _INJECTION_CUES):
            kind = "injection"        # untrusted document text, not content
        elif any(marker in low for marker in _REFUSAL_MARKERS):
            kind = "refusal"
        elif any(p in low for p in _EXEMPT_PHRASES):
            kind = "disclosure"
        elif _inline_citation_error(sentence, files, pages,
                                    corpus_numbers) is not None:
            # The model wrote a filename/page/table the retrieved nodes do
            # not support: fabricated attribution — never kept as fact.
            kind = "unsupported"
        else:
            toks = _support_tokens(sentence)
            if not toks:
                kind = "glue"          # no substantiable content either way
            elif toks <= corpus:
                # Token overlap is NOT proof: refuse to bless a claim whose
                # own evidence states the opposite polarity (narrow gates).
                kind = ("unsupported"
                        if _polarity_conflict(sentence, evidence_clauses)
                        else "supported")
            else:
                kind = "unsupported"
        if kind == "supported":
            has_supported = True
            claims.append({"text": sentence, "idx": len(kept),
                           "nodes": _covering_nodes(toks)})
            kept.append(sentence)
        elif kind in ("glue", "disclosure"):
            kept.append(sentence)
        else:      # unsupported / injection / refusal-shaped sentence
            removed = True       # (refusals are not claims to keep)

    if not has_supported:
        # Nothing in the answer is substantiated by the retrieved passages:
        # mandated refusal, no references footer.
        return "refusal", [], []
    if not removed:
        return "ok", kept, claims       # fully grounded: leave untouched
    return "partial", kept, claims


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OllamaEmbedding(
            model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL
        )
    return _embedding_model


class _GetLlmBridge(CustomLLM):
    """Bridge get_llm() into LlamaIndex without LangChain-bridge packages.

    LlamaIndex's response synthesizer drives the LLM with flat prompts;
    we forward each prompt to the single get_llm() factory (works for both
    the gemini and ollama providers).
    """

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=200_000,
            num_output=1024,
            model_name="get_llm() bridge",
            is_chat_model=True,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        text = llm_text(get_llm().invoke(prompt))
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        text = self.complete(prompt).text

        def _gen():
            yield CompletionResponse(text=text, delta=text)

        return _gen()


# --- M4B / G1: keep local filesystem paths out of the LLM context --------
# Metadata keys that MAY render into the synthesis prompt (citation-relevant
# only). The STORED node metadata is untouched: Chroma diagnostics, ingest
# integrity checks and the agent's own footer building all read
# node.metadata directly, so attribution provenance is preserved everywhere
# except the LLM-facing string.
_LLM_SAFE_METADATA = frozenset(
    {"page_label", "page_number", "file_name", "book_title"}
)


class _LocalPathFilter(BaseNodePostprocessor):
    """Query-time LLM-context metadata allowlist (M4B, defect D5/G1).

    Measured before this filter: the synthesis prompt carried the node's
    absolute ``file_path`` (e.g. ``C:\\Users\\<name>\\...``) because
    ``file_path`` is not in the stored ``excluded_llm_metadata_keys``.
    After this filter only the allowlisted keys remain renderable by
    MetadataMode.LLM; every other key (paths, dates, future ingest keys)
    is added to the node's exclusion list for THIS query only. Existing
    stored exclusions (file_name, file_type, ...) are never removed.
    """

    def _postprocess_nodes(self, nodes, query_bundle=None):
        for node_with_score in nodes:
            node = getattr(node_with_score, "node", node_with_score)
            meta = node.metadata or {}
            excluded = list(
                getattr(node, "excluded_llm_metadata_keys", None) or []
            )
            for key in meta:
                if key not in _LLM_SAFE_METADATA and key not in excluded:
                    excluded.append(key)
            node.excluded_llm_metadata_keys = excluded
        return nodes


def _get_query_engine():
    """Load the persisted Chroma index (never rebuilt here)."""
    global _query_engine
    if _query_engine is None:
        import chromadb

        # Bridge the single get_llm() factory into LlamaIndex — otherwise
        # LlamaIndex would silently default to OpenAI (wrong provider, no key).
        Settings.llm = _GetLlmBridge()
        Settings.embed_model = _get_embedding_model()
        client = chromadb.PersistentClient(path=str(STORAGE_DIR))
        collection = client.get_or_create_collection(
            CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(
            store, storage_context=StorageContext.from_defaults(vector_store=store)
        )
        qa_prompt = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Answer the question using ONLY the context above. Do not use outside "
            "knowledge. Never invent figures, explanations, or conclusions.\n"
            "Do not treat the topics of the context as evidence that the specific "
            "requested fact is present: answer only what the passages actually "
            "state.\n"
            "If the passages do not support the question at all, reply with "
            f"exactly: {NOT_FOUND_PHRASE}\n"
            "If the passages cover the general topic but not the requested "
            "detail, state ONLY the covered facts from the passages and then "
            "say explicitly which requested part is not covered by the "
            "passages — do not claim the whole topic is absent.\n"
            "Question: {query_str}\n"
            "Answer: "
        )
        _query_engine = index.as_query_engine(
            similarity_top_k=BOOKS_TOP_K, text_qa_template=qa_prompt,
            # M4B/G1: absolute file_path values never reach the LLM context.
            node_postprocessors=[_LocalPathFilter()],
        )
    return _query_engine


def _claim_evidence(claim: dict, source_nodes: list) -> list:
    """The EXACT passages assigned to this claim (never the whole corpus)
    with only the metadata needed for verification (never file_path),
    bounded by BOOKS_VERIFIER_EVIDENCE_MAX_CHARS in total. Evidence ids
    are derived from stable node indices, so the verifier can reference
    them and the caller can validate them."""
    budget = BOOKS_VERIFIER_EVIDENCE_MAX_CHARS
    evidence, used = [], 0
    for idx in claim["nodes"]:
        node = source_nodes[idx]
        meta = getattr(node, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        header = []
        title = meta.get("file_name") or meta.get("book_title")
        if isinstance(title, str) and title.strip():
            header.append(title.strip())
        page = meta.get("page_label")
        if page is None:
            page = meta.get("page_number")
        if isinstance(page, (str, int)) and not isinstance(page, bool):
            header.append(f"page {page}")
        text = str(getattr(node, "text", "") or "")
        block = (f"[source: {'; '.join(header)}]\n" if header else "") + text
        room = budget - used
        if room <= 0:
            break
        if len(block) > room:
            block = block[:room] + "\u2026[truncated]"
        evidence.append({"id": f"e{idx}", "text": block})
        used += len(block)
    return evidence


def _verify_claims(claims: list, source_nodes: list):
    """M4B bounded semantic verification. Returns (retained, notes, n_unavailable).

    Monotone by contract: claims are only ever REMOVED — never added,
    rewritten, or blessed beyond the deterministic gate that already ran.
    Every non-supported outcome is fail-closed (removed + disclosed), and
    the verifier's free-text reason never reaches the user: `notes` are
    fixed disclosure constants only.

    M4C claim-level attribution: a retained (supported) claim's node set
    is narrowed to the evidence ids the verifier itself cited. Those ids
    were validated against THIS claim's assigned evidence only (the
    strict parser rejects ids outside the per-claim whitelist), so one
    claim's evidence can never be attributed to another claim. If a
    scripted or unexpected result cites nothing inside the claim's own
    node set (impossible through the production parser), attribution
    falls back to the deterministic covering nodes — narrowing only, and
    never invented evidence.

    Sibling isolation / partial-response policy: each claim is judged
    ONLY by its own positional result. A result missing for a claim
    (verify_all returning fewer results than items — a partially valid
    multi-claim response) counts as UNAVAILABLE for exactly that claim:
    never a pass, never silent, and never a sibling's business. Surplus
    results are ignored. One malformed/unavailable/contradicted claim
    therefore removes exactly that claim; siblings keep their own
    verdicts, sources, and citations, and no surviving claim ever
    inherits evidence from a removed one.
    """
    items = [
        {"claim": claim["text"],
         "evidence": _claim_evidence(claim, source_nodes)}
        for claim in claims
    ]
    results = verifier.verify_all(items)
    retained, notes = [], []
    n_unavailable = n_contradicted = n_removed = 0
    for i, claim in enumerate(claims):
        result = results[i] if i < len(results) else None
        if result is None or result.get("status") != verifier.STATUS_VERIFIED:
            n_unavailable += 1
            continue
        verdict = result.get("verdict")
        if verdict == verifier.VERDICT_SUPPORTED:
            cited = {int(str(e)[1:]) for e in result.get("evidence_ids", [])
                     if str(e).startswith("e") and str(e)[1:].isdigit()}
            narrowed = [i for i in claim["nodes"] if i in cited]
            if narrowed:
                claim["nodes"] = narrowed
            retained.append(claim)
        elif verdict == verifier.VERDICT_CONTRADICTED:
            n_contradicted += 1
        else:                      # insufficient_evidence | uncertain
            n_removed += 1
    if n_unavailable:
        notes.append(_VERIFIER_UNAVAILABLE_NOTE)
    if n_contradicted:
        notes.append(_VERIFIER_CONFLICT_NOTE)
    if n_removed:
        notes.append(_VERIFIER_REMOVED_NOTE)
    return retained, notes, n_unavailable


def books_agent(state: AgentState) -> dict:
    question = state.get("question", "")
    start = time.perf_counter()

    response = _get_query_engine().query(question)
    source_nodes = list(getattr(response, "source_nodes", []) or [])
    score = source_nodes[0].score if source_nodes else None
    # M3: strip any References/Sources/Citations block the model emitted
    # (fabricated, or copied from retrieved document text) BEFORE any other
    # handling — the only footer that can reach the user is the agent's own,
    # built below from validated retrieved-node metadata.
    answer_text = _strip_model_citations(str(response).strip())
    elapsed = time.perf_counter() - start

    low_confidence = score is not None and score < BOOKS_MIN_SCORE
    # The synthesis prompt makes the model reply with the exact not-found phrase
    # when the chunks don't cover the question (nomic's similarity band is too
    # narrow for a fixed threshold alone); the score is kept as a hard floor.
    # Detection is sentence-level (see _refusal_only) so paraphrased refusals
    # cannot slip through and pick up a references footer (spec section 2.1:
    # a refusal-only response presents no evidence, so it cites none).
    refusal = bool(answer_text) and _refusal_only(answer_text)
    if not answer_text or low_confidence or refusal:
        # Spec: never invent — say explicitly it is not in the books, using
        # the mandated refusal phrase, WITHOUT a references footer.
        answer_text = NOT_FOUND_PHRASE
        if score is not None and not refusal:
            answer_text += f" (best similarity score {score:.2f})"
        print(f"[books_agent] not-found path ({elapsed:.1f}s, score={score})")
        return with_agent_output(state, "books", answer_text, ["books"])

    # Spec: include book title + page number in the answer when node
    # metadata has them. First run the deterministic grounding safeguard:
    # keep only answer content substantiated by the retrieved passages,
    # disclose whatever was dropped, and refuse when nothing substantiable
    # remains (no references footer on a refusal).
    mode, kept_sentences, claims = _grounded_repair(answer_text, source_nodes)
    if mode == "refusal":
        # No score note here: the score is NOT the reason (retrieval may
        # even have scored high) — the absence of supporting passages is.
        answer_text = NOT_FOUND_PHRASE
        print(f"[books_agent] grounding refusal ({elapsed:.1f}s, score={score})")
        return with_agent_output(state, "books", answer_text, ["books"])

    # M4B: optional bounded semantic verification (BOOKS_VERIFIER=local).
    # Monotone: it can only REMOVE claims; with the default "off" this block
    # is skipped entirely and behavior is byte-identical to M3.
    retained = claims
    notes = []
    n_unavailable = 0
    if BOOKS_VERIFIER == "local":
        retained, notes, n_unavailable = _verify_claims(claims, source_nodes)
        print(f"[books_agent] verifier: {len(retained)}/{len(claims)} "
              "claim(s) confirmed")
        if not retained:
            # Fail-closed abstention: nothing survived verification.
            answer_text = NOT_FOUND_PHRASE
            if n_unavailable >= len(claims):
                # Distinguish "verification never succeeded" from a
                # successful negative verdict — never imply verification.
                answer_text += " (semantic verification unavailable)"
            print(f"[books_agent] verifier refusal ({elapsed:.1f}s, "
                  f"score={score})")
            return with_agent_output(state, "books", answer_text, ["books"])

    dropped = {c["idx"] for c in claims} - {c["idx"] for c in retained}
    if dropped:
        remaining = [s for i, s in enumerate(kept_sentences)
                     if i not in dropped]
        parts = [" ".join(remaining)]
        if mode == "partial":
            parts.append(_PARTIAL_DISCLOSURE)
        parts.extend(notes)
        answer_text = " ".join(parts)
    elif mode == "partial":
        answer_text = " ".join(kept_sentences) + " " + _PARTIAL_DISCLOSURE
    # ("ok" with nothing dropped: the model's text stays byte-identical.)

    # Spec: include book title + page number in the answer when node
    # metadata has them — but ONLY for nodes that support a RETAINED claim
    # (M3 + M4B: a claim removed by verification loses its citation), and
    # never under a name the model wrote. Missing metadata yields an
    # explicit "attribution unavailable" marker instead of an invented name.
    used = []
    for claim in retained:
        for i in claim["nodes"]:
            if i not in used:
                used.append(i)
    footer = (_claim_reference_footer(source_nodes, retained)
              if BOOKS_VERIFIER == "local" and retained
              else _reference_footer(source_nodes, used))
    print(f"[books_agent] answered in {elapsed:.1f}s "
          f"from {len(used)} supporting chunk(s)")
    answer = answer_text
    if footer:
        answer += footer
    return with_agent_output(state, "books", answer, ["books"])
