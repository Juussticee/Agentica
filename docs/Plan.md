Master Agent System — Full Build Specification
Purpose of this document: a complete, unambiguous spec you can hand to a coding agent (e.g. Claude Code) and say "build this exactly." It covers environment, folder structure, every file to create, the exact contract each component must satisfy, and the order of execution with pass/fail criteria per step.

Target machine: Windows, GTX 1660 Super (4GB VRAM). Stack: Ollama, LangGraph, LlamaIndex, ChromaDB, SearXNG, Crawl4AI.

0. Global conventions
Language: Python 3.11+, all code in agents-project/app/.
State passed between LangGraph nodes is a single TypedDict called AgentState (defined once, imported everywhere — never redefined per node).
Every node function signature: def node_name(state: AgentState) -> dict — returns only the keys it changes; LangGraph merges them.
Every agent node must attach its name to state["sources"] when it contributes an answer, so the final output can show provenance.
No node calls another node directly. All routing goes through the Supervisor.
Config (model names, paths, ports) lives in one file: app/config.py. Nothing else hardcodes a path or model name.
1. Environment setup (do this first, verify before continuing)
# 1. Project + venv
mkdir agents-project && cd agents-project
python -m venv venv
venv\Scripts\activate          # Windows

# 2. Python packages
pip install langgraph langchain langchain-ollama llama-index llama-index-embeddings-ollama chromadb python-docx pypdf fastapi uvicorn requests

# 3. Ollama models
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct        # fallback if 7B is too slow on 4GB VRAM
ollama pull nomic-embed-text
Verify: ollama list shows all three models. python -c "import langgraph, chromadb, llama_index" runs without error.

Then start SearXNG:

docker run -d --name searxng -p 8080:8080 searxng/searxng
Verify: http://localhost:8080 loads a search page in the browser.

2. Folder structure to create
agents-project/
├── venv/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── state.py             # AgentState TypedDict
│   ├── graph.py              # builds and compiles the LangGraph graph
│   ├── supervisor.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── books_agent.py
│   │   ├── research_agent.py
│   │   ├── feasibility_agent.py
│   │   └── erp_agent.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── searxng_client.py
│   │   └── crawl_extractor.py
│   └── qa_node.py            # Evidence/QA check before final answer
├── knowledge/
│   ├── books/
│   └── company/
├── storage/                  # Chroma persistence — created automatically, don't hand-create
├── ingest_books.py
├── main.py                   # CLI entry point to chat with the system
└── requirements.txt
3. File-by-file contract
app/config.py
Defines constants: MAIN_MODEL = "qwen2.5:7b-instruct-q4_K_M", FALLBACK_MODEL = "qwen2.5:3b-instruct", EMBED_MODEL = "nomic-embed-text", SEARXNG_URL = "http://localhost:8080", BOOKS_DIR, STORAGE_DIR. Nothing else in the codebase hardcodes these values.

app/state.py
from typing import TypedDict, List, Optional

class AgentState(TypedDict, total=False):
    question: str
    route: str              # "books" | "research" | "feasibility" | "erp" | "multi"
    sub_routes: List[str]   # when route == "multi"
    sources: List[str]
    answer: str
    qa_flags: List[str]     # conflicts/uncertainty noted by the QA node
app/supervisor.py
Takes state["question"], decides route (and sub_routes if more than one agent is needed).
Phase 1: keyword-based routing (books-related words → books, else → research).
Phase 4 upgrade: replace with an LLM call classifying intent — the function signature stays identical so nothing downstream changes.
Must never call an agent directly — only sets route/sub_routes on the state.
app/agents/books_agent.py
Loads the persisted Chroma index from STORAGE_DIR (built by ingest_books.py, not rebuilt at query time).
Runs query_engine.query(state["question"]), returns {"answer": ..., "sources": [...] + ["books"]}.
Must include book title + page number in the answer text when the underlying node metadata has them.
Must explicitly say "not found in your books" rather than inventing an answer if similarity scores are all low.
app/tools/searxng_client.py
One function: search(query: str, num_results: int = 5) -> list[dict] hitting SEARXNG_URL/search?q=...&format=json.
Returns [{"title", "url", "snippet"}].
app/tools/crawl_extractor.py
One function: extract(url: str) -> str using Crawl4AI to fetch and return clean text, empty string on failure (never raises — Research agent must handle empty gracefully).
app/agents/research_agent.py
Calls searxng_client.search(), then crawl_extractor.extract() on the top 2-3 results, feeds the combined text + question to the LLM, returns a synthesized answer with the source URLs listed.
app/agents/feasibility_agent.py
Only active from Phase 2. Calls research_agent output + books_agent output (Supervisor sets sub_routes = ["research", "books"] first, then routes to feasibility with both prior answers in state) and produces a structured study: Market / Financial / Risk sections.
Output must be saved as a .docx via python-docx, path returned in state["answer"].
app/agents/erp_agent.py
Phase 3 only. Stub in Phase 1: return "ERP agent not yet connected."
Real version: connects to the actual ERP (vendor unspecified) through the
vendor-neutral ERPAdapter interface added in Phase 3; documentation about
the real adapter belongs in PHASE3_ERP_PLAN.md, where all ERP-specific
decisions are intentionally deferred until the ERP details are provided.
app/qa_node.py
Phase 4 only. Runs after any agent node, before Supervisor returns the final answer. Checks: does the answer cite a source? Do multiple agent outputs (in multi route) contradict each other? Appends findings to state["qa_flags"].
app/graph.py
Builds the StateGraph(AgentState), adds all nodes, wires START → supervisor, conditional edges from supervisor to each agent based on route, and each agent → END (or → qa_node → END once Phase 4 is added). Exposes one function: build_graph() -> CompiledGraph.
ingest_books.py
Standalone script (not part of the graph). Uses SimpleDirectoryReader(BOOKS_DIR, recursive=True, required_exts=[".pdf",".docx"]), builds VectorStoreIndex, persists to STORAGE_DIR. Skips files already indexed (track by filename + modified time in a small local .index_manifest.json).
main.py
CLI loop: build_graph() once, then while True: question = input(); result = graph.invoke({"question": question}); print(result["answer"]).
4. Build & test order (do not skip ahead — each step must pass before the next)
Environment (Section 1) — verified as above.

ingest_books.py — put 2 real files in knowledge/books/, run it, confirm storage/ is created and non-empty.

books_agent.py alone — write a throwaway test script that imports it directly and asks one question about the indexed books. Confirm it answers with the correct content and names a source.

searxng_client.py + crawl_extractor.py — test independently: search a query, confirm URLs come back; extract one URL, confirm non-empty text.

research_agent.py — test standalone with a factual question, confirm a synthesized answer with source URLs.

state.py + supervisor.py + graph.py — wire books + research only. Run main.py, ask one books-routed question and one research-routed question. Both must return correct, sourced answers.

✅ Phase 1 complete when step 6 passes.

feasibility_agent.py — test with a sub_routes = ["research","books"] scenario, confirm a .docx is produced with all three sections.

Wire feasibility into graph.py, test via main.py with a feasibility-style question end-to-end.

✅ Phase 2 complete when step 8 passes.

erp_agent.py — adapter-agnostic read-only checks: confirm the (unspecified)
real ERP connects, returns deterministic records, and logs each operation;
write support requires an explicit approval boundary.

Wire ERP into graph.py behind the approval flow.

✅ Phase 3 complete when step 10 passes, with logging confirmed.

qa_node.py — add after Phase 3 is stable. Test with a deliberately contradictory pair of agent outputs, confirm it flags the conflict instead of silently picking one.
Upgrade supervisor.py routing from keyword-based to LLM-based, confirm existing Phase 1/2 tests still pass.
✅ Phase 4 complete when step 12 passes.

5. Handoff instruction (for your coding agent)
Build the project exactly per the folder structure in Section 2 and the file contracts in Section 3, in the order given in Section 4. Do not proceed to the next numbered step until the current one's pass criteria are met. Use app/config.py as the single source of truth for models/paths/ports — no hardcoded values elsewhere. Stop and report if qwen2.5:7b-instruct-q4_K_M proves too slow on 4GB VRAM in testing; fall back to qwen2.5:3b-instruct by changing only config.py.