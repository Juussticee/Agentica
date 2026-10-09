"""Books agent: answers from the local Chroma index built by ingest_books.py.

Known gap (flagged to the user, per instructions): embeddings still come from
Ollama's nomic-embed-text; the Gemini embedding API needs separate setup and
was deliberately NOT wired in.
"""

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
            "knowledge. If the context does not contain the answer, reply with "
            f"exactly: {NOT_FOUND_PHRASE}\n"
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
    refused = NOT_FOUND_PHRASE.lower() in answer_text.lower()
    if not answer_text or low_confidence or refused:
        # Spec: never invent — say explicitly it is not in the books.
        if score is not None and not refused:
            answer_text = f"{NOT_FOUND_PHRASE} (best similarity score {score:.2f})"
        else:
            answer_text = NOT_FOUND_PHRASE
        print(f"[books_agent] not-found path ({elapsed:.1f}s, score={score})")
        return with_agent_output(state, "books", answer_text, ["books"])

    # Spec: include book title + page number in the answer when node metadata has them.
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
