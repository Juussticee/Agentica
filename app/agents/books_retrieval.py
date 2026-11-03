"""Retrieval: the Chroma/LlamaIndex query-engine construction and its
plumbing - embedding model, LLM bridge, LLM-context path filter.

Extracted from books_agent.py (Increment 2 decomposition). The CACHE and
the patch seam stay on the books_agent facade (tests/evals patch
`books_agent._get_query_engine`, `._query_engine`, `._embedding_model`,
`.STORAGE_DIR`, `.CHROMA_COLLECTION` there); this module only builds, and
reads facade state at call time through `_host()`.

The engine loads the persisted index built by ingest_books.py - it never
re-ingests or rebuilds documents.
"""

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

from app import chroma_guard
from app.config import (
    BOOKS_TOP_K,
    EMBED_MODEL,
    NOT_FOUND_PHRASE,
    OLLAMA_BASE_URL,
)
from app.llm_client import llm_text


def _host():
    """The books_agent facade module - the compatibility surface where
    tests and eval harnesses patch seams (get_llm, STORAGE_DIR, caches).

    Function-level import: books_agent imports THIS module at its top, so a
    module-level import would be circular. Resolving seams there keeps
    `books_agent.X = fake` patches effective after the Increment 2 split.
    """
    import app.agents.books_agent as host
    return host


def build_embedding_model():
    """Construct the Ollama embedding model - UNCACHED.

    The cache (and the `books_agent._embedding_model` patch seam) stays on
    the books_agent facade; this is only the construction.
    """
    return OllamaEmbedding(
        model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL
    )


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
        text = llm_text(_host().get_llm().invoke(prompt))
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


def _open_index():
    """Open the PERSISTED Chroma index (never re-ingests, never writes).

    The single place the store is opened: the books query engine and the
    read-only evidence retriever both build on it, so there is one guard
    call and one embedding wiring rather than two.
    """
    host = _host()
    import chromadb

    # Fail BEFORE any open: an armed test/eval context must never reach
    # the production store (no-op in normal production operation).
    chroma_guard.assert_store_path_allowed(host.STORAGE_DIR)
    # Bridge the single get_llm() factory into LlamaIndex — otherwise
    # LlamaIndex would silently default to OpenAI (wrong provider, no key).
    Settings.llm = _GetLlmBridge()
    Settings.embed_model = host._get_embedding_model()
    client = chromadb.PersistentClient(path=str(host.STORAGE_DIR))
    collection = client.get_or_create_collection(
        host.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(
        store, storage_context=StorageContext.from_defaults(vector_store=store)
    )


def build_retriever(top_k: int = 4):
    """Read-only passage retriever over the SAME persisted index.

    Used by the general route (app/evidence.py) so a general question can be
    checked against the local corpus without a second index, a second
    embedding stack, or a second retrieval policy. Returns source nodes only:
    it never synthesizes an answer, so no model is called here.
    """
    return _open_index().as_retriever(similarity_top_k=top_k)


def build_query_engine():
    """Build the persisted Chroma query engine - UNCACHED; documents are
    never re-ingested here.

    The lazy cache and the patchable seam stay on the books_agent facade
    (`_get_query_engine` / `_query_engine`): tests and eval harnesses patch
    or reset them there. The state this builder needs (STORAGE_DIR,
    CHROMA_COLLECTION, the embedding cache) is looked up on the facade at
    CALL time through `_host()` (function-level import, so importing this
    module is never circular) - runtime patches keep working.
    """
    index = _open_index()
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
    engine = index.as_query_engine(
        similarity_top_k=BOOKS_TOP_K, text_qa_template=qa_prompt,
        # M4B/G1: absolute file_path values never reach the LLM context.
        node_postprocessors=[_LocalPathFilter()],
    )
    return engine
