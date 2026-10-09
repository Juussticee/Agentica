# Master Agent System

A multi-agent LangGraph system that routes a natural-language question to
specialized agents and answers with provenance and honest refusals.

- **books** — retrieval-augmented answers from a local ChromaDB index of the
  company library (`knowledge/books/`), with file references and an exact
  not-found refusal instead of hallucination.
- **research** — SearXNG web search (with LLM query rewriting) → Crawl4AI
  extraction → Gemini synthesis with source URLs.
- **feasibility** — chains research + books, then synthesizes a Market /
  Financial / Risk study saved as a timestamped `.docx`. Degrades gracefully
  to a flagged **web-only** study when the library has no relevant material.
- **erp** — stub node for the (vendor-neutral, Phase 3) ERP integration.
  The actual ERP vendor is not specified; see PHASE3_ERP_PLAN.md.

## Current phase

```text
Phase 1: COMPLETE
Phase 2: COMPLETE
Phase 3: NOT STARTED   (ERP-agnostic integration boundary complete;
  actual ERP integration NOT STARTED — vendor/API/auth/data model
  intentionally deferred until the ERP is specified)
Phase 4: NOT STARTED   (QA node, LLM-based supervisor routing)
```

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (single compiled `StateGraph`) |
| Chat LLM | Google Gemini `gemini-3-flash-preview` via `langchain-google-genai` (provider switch to Ollama built in) |
| Retrieval | LlamaIndex over a persisted ChromaDB collection |
| Embeddings | Ollama `nomic-embed-text` (local; known gap — Gemini embeddings not wired) |
| Web search | SearXNG (local Docker container, JSON API) |
| Page extraction | Crawl4AI |
| Documents | python-docx (feasibility reports), pypdf + docx2txt (ingestion) |

## Setup (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

:: Ollama with the embedding model (required). Chat models are only needed
:: if you switch LLM_PROVIDER=ollama.
ollama pull nomic-embed-text

:: SearXNG — this machine maps host port 8888 to the container's 8080
docker run -d --name searxng -p 8888:8080 searxng/searxng

:: Environment
copy .env.example .env
:: then edit .env and set your key, e.g.  GOOGLE_API_KEY=...
```

**API keys.** Keys live ONLY in `.env` (git-ignored) — never in code, never in
this repository. `app/config.py` reads `.env` values FIRST and then normalizes
`GOOGLE_API_KEY`/`GEMINI_API_KEY` in the process environment, so a stale
invalid `GOOGLE_API_KEY` inherited from the OS cannot shadow the valid `.env`
key. Deleting the stale OS-level variable is still recommended.

## Run

```bat
:: 1. Index the library (incremental; skips already-indexed files)
python ingest_books.py

:: 2. Ask questions
python main.py                                  :: interactive chat
python main.py --demo                           :: canned phase-1 scenarios
python main.py --once "Is a rooftop apiary feasible in Beirut?"
```

Example questions by route:

- **Books** — "What does the orbital mechanics book say about delta-v for a
  Hohmann transfer?" → answer with `References: <file>` provenance.
- **Books refusal** — "What is quantum tunneling?" → exact not-found phrase.
- **Research** — "What is LangGraph used for?" → synthesized answer plus a
  `Sources:` URL list.
- **Feasibility** — "Do a feasibility study for launching an urban beekeeping
  venture." → `[route: feasibility]`, study written to `reports/feasibility_*.docx`.

## Outputs

| Path | Created by | Contents |
|---|---|---|
| `storage/` | `ingest_books.py` | ChromaDB persistence (`chroma.sqlite3` + collection) and `.index_manifest.json` (filename → mtime, drives incremental ingest) |
| `reports/` | feasibility agent | timestamped `feasibility_YYYYMMDD_HHMMSS.docx` studies |
| `logs/` | reserved | ERP action log path (Phase 3, not yet active) |

All three are git-ignored generated data.

## Testing

See **[docs/TESTING.md](docs/TESTING.md)** for the full test matrix. The
quick zero-quota set:

```bat
python scripts\test_phase1_routing.py
python scripts\test_phase2.py
python scripts\test_feasibility_degradation.py
python scripts\test_rewrite_step.py
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, state flow, diagrams
- [docs/TESTING.md](docs/TESTING.md) — commands, live/quota costs, failure testing
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) — factual phase status and limitations

## Known gaps

- Free-tier Gemini quota is small (20 requests/day for gemini-3-flash-preview);
  most test scripts deliberately run with fake LLMs at zero quota.
- SearXNG result quality varies; research answers depend on successfully
  crawled page content and are not guaranteed authoritative.
- Embeddings remain on Ollama `nomic-embed-text`; score distributions require
  the retrieval tuning documented in `app/config.py` (`BOOKS_MIN_SCORE`).
