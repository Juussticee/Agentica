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
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import (
    BOOKS_MIN_SCORE,
    BOOKS_TOP_K,
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


def _grounded_repair(answer_text: str, source_nodes: list):
    """Return (mode, text): ("refusal", None), ("partial", kept), or None
    (untouched) for the QUALIFIED case (fully supported).
    """
    corpus = set()
    evidence_clauses = []          # (clause, clause_tokens, clause_negated)
    for node in source_nodes:
        text = getattr(node, "text", "") or ""
        corpus |= _support_tokens(text)
        for clause in _clauses(text):
            evidence_clauses.append(
                (clause, _support_tokens(clause), _has_negation(clause))
            )

    kept, has_supported, removed = [], False, False
    for sentence in (s.strip() for s in
                     re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()):
        low = sentence.lower()
        if any(marker in low for marker in _REFUSAL_MARKERS):
            kind = "refusal"
        elif any(p in low for p in _EXEMPT_PHRASES):
            kind = "disclosure"
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
            kept.append(sentence)
        elif kind in ("glue", "disclosure"):
            kept.append(sentence)
        else:                          # unsupported OR refusal-shaped sentence
            removed = True             # (refusals are not claims to keep)

    if not has_supported:
        # Nothing in the answer is substantiated by the retrieved passages:
        # mandated refusal, no references footer.
        return "refusal", None
    if not removed:
        return None                    # fully grounded: leave untouched
    return "partial", " ".join(kept) + " " + _PARTIAL_DISCLOSURE


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
            similarity_top_k=BOOKS_TOP_K, text_qa_template=qa_prompt
        )
    return _query_engine


def books_agent(state: AgentState) -> dict:
    question = state.get("question", "")
    start = time.perf_counter()

    response = _get_query_engine().query(question)
    source_nodes = list(getattr(response, "source_nodes", []) or [])
    score = source_nodes[0].score if source_nodes else None
    answer_text = str(response).strip()
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
    repaired = _grounded_repair(answer_text, source_nodes)
    if repaired is not None:
        mode, kept = repaired
        if mode == "refusal":
            # No score note here: the score is NOT the reason (retrieval may
            # even have scored high) — the absence of supporting passages is.
            answer_text = NOT_FOUND_PHRASE
            print(f"[books_agent] grounding refusal ({elapsed:.1f}s, score={score})")
            return with_agent_output(state, "books", answer_text, ["books"])
        if mode == "partial":
            answer_text = kept

    # Spec: include book title + page number in the answer when node
    # metadata has them.
    refs = []
    for node in getattr(response, "source_nodes", [])[: BOOKS_TOP_K]:
        meta = node.metadata or {}
        title = meta.get("file_name") or meta.get("book_title") or "unknown file"
        page = meta.get("page_label") or meta.get("page_number")
        refs.append(f"{title}, page {page}" if page else title)

    print(f"[books_agent] answered in {elapsed:.1f}s from {len(refs)} chunk(s)")
    answer = answer_text
    if refs:
        """Spec: attach book title + page provenance when metadata has them."""
        answer += "\n\nReferences: " + "; ".join(refs)
    return with_agent_output(state, "books", answer, ["books"])
