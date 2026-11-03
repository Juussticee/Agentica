# GUI RAG Test Report — NIST AI Risk Management Framework (AI RMF 1.0)

Date: 2026-10-24, times local (UTC+3). Tester: real-GUI execution driven through
the Agentica Control Center web UI against the *live* backend. **No mock or
scripted path was used for any question.** Every answer below is a verbatim
capture from the GUI's "Result" panel (backend-confirmed via the app's own
execution/event API; JSON exports in `.diag/gui_rag_test_2026-10-24/`).

## 1. Initial repository / process state

- Branch `main` @ `08d44b6`, working tree 18 pre-existing entries (5 modified,
  13 untracked) — all preserved untouched; no commits, pushes, or branch
  changes during this test.
- Backend: FastAPI `app.api.server:app` on `127.0.0.1:8000` —
  `/api/v1/health` → `{"status":"ok","backend":"ok"}`.
- Frontend: Vite dev server on `http://localhost:3333` (hash-routed; GUI
  accessed as `/#/new`, `/#/executions/<id>`).
- AI settings (`/api/v1/ai/settings`): `llm_provider: ollama`,
  `main_model: qwen3:0.6b`, `ollama_base_url: http://localhost:11434`,
  source `environment`; a Gemini key is present in masked env config but was
  never called (Ollama only; no remote/paid provider used — see §4).
- Ollama `/api/tags` (passive check): `qwen3:0.6b`, `nomic-embed-text:latest`,
  `moondream:latest` present locally.
- SearXNG: listening on `127.0.0.1:8888` (passive netstat check only).
  **Invoked exactly once**, by the app itself, during strict Test 3 (see §5).

## 2. Test document

| Field | Value |
|---|---|
| Path | `knowledge/books/NIST.AI.100-1.pdf` |
| Name | NIST AI 100-1, AI RMF 1.0 (48 pages confirmed by cover text "NIST AI 100‑1 Artificial Intelligence Risk Management Framew…") |
| Size | 1,946,127 bytes |
| Pages | 48 (pypdf) |
| SHA-256 | `7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1` |

## 3. Import / ingestion details

- The GUI has **no import flow**; the application's documented ingestion path
  is the CLI `python ingest_books.py` → the live `books` Chroma collection the
  GUI reads. The user's instruction ("everything is here … let us try how the
  agent will behave under non perfect conditions") selected: ingest all three
  new PDFs as-is, keep `NIST.AI.600-1.pdf.pdf` unrenamed.
- Ingest result (exit code 1 — "at least one file FAILED", final store state
  verified with `scripts/check_chroma.py`):
  - `NIST.AI.100-1.pdf`: **indexed — 48 pages, 48 chunks persisted** (test target OK).
  - `orbital_kinetics_principles.docx`, `urban_beekeeping_field_guide.pdf`:
    unchanged (2 + 1 chunks preserved; **no re-embedding, no deletion**).
  - `NIST.AI.600-1.pdf.pdf`: FAILED `binary_garbage` (letters=20%) — not searchable.
  - `nasa_systems_engineering_handbook_0.pdf`: FAILED `binary_garbage`
    (letters=48%, magic=False) — not searchable.
- Manifest after ingest: `{'failed': 2, 'indexed': 3}`; store **healthy**,
  0 integrity errors ("collection healthy and manifest consistent").
  No code or retrieval setting was changed; the two failures were documented,
  not repaired (per test rules).

## 4. Model / provider verification

Before Test 1: the backend health and settings endpoints both name
Ollama `qwen3:0.6b`; Ollama responded locally to an otherwise-unused port
probe. Every books executions in the trace used the books agent
(`agent.started → Check internal knowledge → agent.completed`; event exports
show `route: books` for 7 of 8 executions). No Gemini/remote call occurred; no
manual `--mode local-model` evaluation was run. **Exception:** strict Test 3
was routed by the *app* to the research agent (SearXNG web search) — see §5
and §6/T3.

## 5. Routing — first-class finding (affects the whole suite)

The supervisor routes by keyword regex `app/supervisor.py::_BOOKS_RE`
(`books?|chapter|pages?|page \d|textbook|handbook|manual|guide|author|
bibliography|section|orbital|orbit|…`). Questions without any such word route
to **web research**, even when they say "this document":

- **Strict Test 3 ("Using only this document, explain the difference between
  the Map and Measure functions…") was routed to web research.** The app then
  fetched web results (the only SearXNG invocation) and returned a nonsense
  Google-Maps answer (below). This violates the document-only constraint for
  that one test, was observed *in the real GUI*, and is reported instead of
  hidden.
- Tests 4, 5, 6, 7 in their *strict* wording would misroute identically
  (predicted from the frozen, deterministic regex — no extra web calls were
  spent to re-observe the same failure). For these, a **minimally-worded
  variant** containing a routing keyword (e.g. "sections", "page") was run so
  the remaining suite stayed genuinely document-only. Variants are labeled
  below; scoring is per executed variant for 4–7 and per executed strictly-worded
  3 + a variant for comparison.
- Tests 1 and 2 routed to books naturally (contain "section"/"page").

## 6. Tests — exact questions, full verbatim responses, traces

Execution IDs are clickable in the GUI at `/#/executions/<id>`; each record+event
export sits in `.diag/gui_rag_test_2026-10-24/<id>.json`. Latencies are from the
event timeline. Retrieved passages are **not exposed** by the UI ("not
observable" where asserted); backend diagnostics below reconstruct what reached
the agent (trace, not UI display).

### Test 1 — Direct factual retrieval — `#c21103777d50` — books route — 10:52:26 → 10:53:24 (58 s)

**Question (exact):** "According to the document, what are the four core
functions of the AI Risk Management Framework? Explain each briefly and cite
the supporting section or page."

**Full verbatim response:**

> The four core functions of the AI Risk Management Framework are:
> 1. **GOVERN** – designed to inform and be infused throughout the AI system
> lifecycle. 2. 3. 4. - **MAP**, **MEASURE**, and **MANAGE** are described in
> the context as functions applied in AI system-specific contexts.
> Note: the answer above is limited to the parts supported by the retrieved
> book passages; the rest was not found in the books.
>
> References: NIST.AI.100-1.pdf, page 3; NIST.AI.100-1.pdf, page 20;
> NIST.AI.100-1.pdf, page 42

Trace: routed→books; internal knowledge 10:52:48–10:53:24; QA passed. No web
events. Web search: **not invoked**. Backend: "[books_agent] answered in 43.2s
from 3 supporting chunk(s)"; M5 self-refusal recovery attempted and succeeded
(score 0.719, 3 nodes).

**PDF verification:** The four functions ARE the document's Govern / Map /
Measure / Manage (listed together on PDF pages 5 and 25: "…to govern, map,
measure, and manage AI risks"; per-function chapters §3.1–3.4 from ~p.27).
"designed to inform and be infused throughout the other three functions"
occurs on pages 5, 25, 27 — so sentence 1's GOVERN characterization is
document-grounded, but the **cited pages 3/20/42 do not contain the cited
claims** (p.3 = NIST disclaimers; p.20 = safety practices; p.42 = governance
AI actors). Formatting is broken ("1. … 2. 3. 4. -").

### Test 2 — Definition and trustworthiness — `#8ad19e1dba9a` — books route — 10:54:52 → 10:55:35 (43 s)

**Question (exact):** "How does the document discuss trustworthy AI and the
characteristics of trustworthiness? Distinguish the document's wording from
your own explanation, and cite the supporting sections or pages."

**Full verbatim response:**

> These characteristics are defined in the same standard (ISO 9000:2015) and
> are tied to social and organizational behavior, datasets, models, and
> decisions. Human judgment is used to determine metrics and thresholds.
> Tradeoffs may arise between interpretability and privacy, or between
> accuracy and explainability. 3. **Accountability and Transparency**:
> Accountability and transparency relate to processes and activities internal
> to AI systems and their external settings. The document notes that
> neglecting these characteristics can increase risks. 4. The document does
> not mention specific details beyond the outlined characteristics and their
> definitions. Note: the answer above is limited to the parts supported by the
> retrieved book passages; the rest was not found in the books.
>
> References: NIST.AI.100-1.pdf, page 12; NIST.AI.100-1.pdf, page 16;
> NIST.AI.100-1.pdf, page 13

Trace: routed→books; internal knowledge 10:54:52–10:55:35; QA passed. Web: not
invoked. Backend: M5 recovery attempted+succeeded (score 0.719).

**PDF verification:** starts mid-thought (no list items 1–2); the trustworthiness
characteristics content (§2.2ff, "trustworthy" occurs 21×) is partially present
(ISO 9000:2015 tie-in, tradeoffs, accountability/transparency are real RMF
themes). Cited pages 12 (Risk Tolerance), 13 (risk prioritization), 16 (AI
actors figure) do not carry the stated claims. The requested
"document's wording vs. my own" distinction was **not** performed. No clearly
fabricated facts, but no complete list either.

### Test 3 — Distinguishing concepts — STRICT wording — `#bd4ce5cc7cad` — **research route** — 10:56:01 → 10:57:33 (92 s)

**Question (exact):** "Using only this document, explain the difference between
the Map and Measure functions. Give one concrete activity associated with each
and cite the supporting evidence."

**Full verbatim response:**

> The Map function refers to navigation tools that help users locate
> locations, while the Measure function involves analyzing data to identify
> patterns or gaps. For the Map function, an activity could be using Google
> Maps to find directions, supported by the image (Source: Google Maps). For
> the Measure function, an activity might be analyzing evidence maps to
> understand data trends, supported by the Methodology section (Source:
> Methodology for mapping reviews, evidence maps, and gap maps). Both examples
> rely on the provided excerpts to support the explanation.
>
> Sources:
> - https://maps.google.com/
> - https://www.cambridge.org/core/journals/research-synthesis-methods/article/methodology-for-mapping-reviews-evidence-maps-and-gap-maps/9C0C51FF65DC0D8D52CB616B08B0F986
> - https://maps.google.com/maps/search/

Trace: routed→**research** ("Routed the request to Researching the web");
research agent 10:56:02–10:57:33; QA passed. **Web search WAS invoked** (the
only occurrence). This is a hard, observed routing failure on a books-domain
question: the "document-only" constraint was defeated by the GUI's own router,
and the answer is a keyword hallucination unrelated to the AI RMF.

**PDF verification:** entirely off-document. Fails the test as a document-only
test; recorded honestly as such.

### Test 3v — same required question, books-routed variant — `#fa59003036d0` — books route — 10:59:22 → 11:00:16 (54 s)

**Question (variant, exact):** "Using only this document, explain the
difference between the MAP and MEASURE functions described in their sections.
Give one concrete activity associated with each and cite the supporting page."

**Full verbatim response:**

> The requested information was not found in the books.

Trace: routed→books; internal knowledge 10:59:22–11:00:16; QA passed. Web: not
invoked. Backend: "[books_agent] grounding refusal (35.7s, score=0.719...)";
M5 recovery attempted → regeneration **refused again** ("self_refusal_recovery_
rejected (reason=second_refusal)") → final manual refusal.

**PDF verification:** the MAP §3.1 ("MAP: Context and Risk Identification")
and MEASURE §3.2 ("MEASURE: Analysis and Tracking…") sections exist in the
document with concrete sub-activities (e.g. MAP 1.A "intended purposes",
MEASURE 2.A "Evaluation" definitions). Retrieval score was 0.72 (2.05× the
0.35 floor) — **retrieved, not answered: a false floor refusal** caused by the
grounding gate, not by missing evidence.

### Test 4 — Evidence across sections — variant — `#7084d223e392` — books route — 11:03:15 → 11:03:59 (44 s)

**Question (variant, exact):** "In the AI RMF document, what role does the
GOVERN function play in the framework's section structure, and how does it
relate to the other three functions? Cite the supporting page."

**Full verbatim response:**

> The requested information was not found in the books.

Trace: routed→books; QA passed. Web: not invoked. Backend: retrieval succeeded
(score 0.827 — high), M5 attempt → rejected (second_refusal) → not-found path.

**PDF verification:** GOVERN-as-cross-cutting is directly in the document
(pp. 5, 25, 27: "Governance is designed to be a cross-cutting function to
inform and be infused throughout the other three functions"; "GOVERN is a
cross-cutting function… enables the other functions"). Another **false
refusal** on answerable, retrieved evidence.

### Test 5 — Unanswerable question — variant — `#22af069916ec` — books route — 11:06:59 → 11:07:20 (21 s)

**Question (variant, exact):** "According to the sections of this document,
what exact chunk size in tokens does the document require for a
retrieval-augmented generation system? Cite the page if such a requirement
exists."

**Full verbatim response:**

> The requested information was not found in the books.

Trace: routed→books; QA passed. Web: not invoked. Backend: "not-found path
(20.5s, score=0.669...)"; no recovery needed after the first attempt refused.

**PDF verification:** correct calibrated behavior — the AI RMF prescribes no
RAG chunk size for retrieval systems ("tokens"-as-chunk-size appears nowhere).
No invented number, no fabricated page. **Desired abstention achieved.**

### Test 6 — Paraphrase robustness — variant — `#9b6b5d5378d3` — books route — 11:09:43 → 11:10:28 (45 s)

**Question (variant, exact):** "In this document, if a team wants to identify
and understand the context of possible AI harms before measuring them, which
section-based AI RMF function is most relevant, and why? Cite the page."

**Full verbatim response:**

> The requested information was not found in the books.

Trace: routed→books; QA passed. Web: not invoked. Backend: score 0.803, M5
recovery attempt → rejected (second_refusal) → not-found path.

**PDF verification:** the expected answer is **MAP** — the document's purpose
statement for MAP ("MAP: Identifies risks and provides methods to measure,
manage, and map risks… Context is documented…" §3.1, pp. ~25/27) matches the
question's paraphrase. A **false refusal** on a correct paraphrase; the
expected, in-corpus answer was Map.

### Test 7 — Multi-part answer — variant — `#7ed53ab21f83` — books route — 11:12:27 → 11:13:01 (34 s)

**Question (variant, exact):** "Summarize the purpose of the AI RMF described
in this document's sections, identify its four core functions, and explain one
limitation or implementation consideration the document itself discusses.
Cite the page for each part."

**Full verbatim response:**

> The AI RMF's purpose is to manage AI risks by being risk-based,
> resource-efficient, pro-innovation, and voluntary, while being
> consensus-driven and updated through an open process. Its four core
> functions are GOVERN, MAP, MEASURE, and MANAGE. - **Core Functions**: From
> page 20. Note: the answer above is limited to the parts supported by the
> retrieved book passages; the rest was not found in the books.
>
> References: NIST.AI.100-1.pdf, page 20; NIST.AI.100-1.pdf, page 42;
> NIST.AI.100-1.pdf, page 3

Trace: routed→books; internal knowledge 11:12:28–11:13:01; QA passed. Web: not
invoked. Backend: "answered in 33.1s from 3 supporting chunk(s)".

**PDF verification:** the purpose attributes ARE document text (p. 47: "The AI
RMF strives to: 1. Be risk-based, resource-efficient, pro-innovation, and
voluntary. 2. Be consensus-driven and developed and regularly updated through
an open, transparent process."). Four functions correct. **The third required
part (limitation / implementation consideration) is missing** (the doc offers
e.g. "the AI RMF is voluntary, non-prescriptive…" p.47; "does not prescribe
risk tolerance" p.12). Cited pages 20/42/3 still don't support the claims (the
purpose attributes are p.47, not 20).

## 7. Scores (0–2 per dimension; max 10)

| # | Executed as | Factual | Grounding | Citation | Completeness | Calibration | Total |
|---|---|---|---|---|---|---|---|
| T1 | strict (books) | 1 | 1 | 1 | 1 | 2 | **6/10** |
| T2 | strict (books) | 1 | 1 | 1 | 1 | 2 | **6/10** |
| T3 | **strict (research)** | 0 | 0 | 0 | 0 | 0 | **0/10** |
| T3 | variant (books) | 0 | 0 | 1 | 0 | 1 | **2/10** |
| T4 | variant (books) | 0 | 0 | 1 | 0 | 1 | **2/10** |
| T5 | variant (books) | 2 | 2 | 2 | 2 | 2 | **10/10** |
| T6 | variant (books) | 0 | 0 | 1 | 0 | 1 | **2/10** |
| T7 | variant (books) | 1 | 1 | 1 | 1 | 2 | **6/10** |

Score rationale (summary):
- Citation "1" never "2": page numbers reference the correct *file* but the
  cited pages do not contain the stated claims in any answered test.
- Calibration 2 on T1/T2/T7: honest "…the rest was not found in the books"
  limit disclosures; no invention. Calibration 2 on T5: ideal abstention.
- T5 is the only clean 10/10: refusal is the correct answer by ground truth.
- T3-variant/T4/T6 calibration "1": over-refusal on answerable content.
- Grounding "1" on T1/T2/T7: some claims match the document (verified), but
  coverage is partial and garbled.

Sum across the 9 executed runs: **36/90 (40%)**. Document-only suite (8 books
runs): 36 points from 80 → 45%. 8 book runs, 7 completion.

## 8. Critical-failure list (independent of scores)

1. **Router misroutes books questions to web research** whenever the phrase
   lacks a `_BOOKS_RE` keyword — demonstrated live on strict Test 3; predicted
   byte-identically for 4–7 strict forms. A user phrasing a document question
   like a human ("this document", "the AI RMF") gets web hallucinations.
2. **Citation misattribution in every answered test.** File+page citations are
   plausible-looking but do not locate the stated claims in the PDF; real
   supporting pages (5/25/27/38/47) were never cited. The passages retrieved
   (scores 0.67–0.83) evidently did not contain the claims the model answered
   with — claims that *are* in the document, suggesting **model prior
   knowledge** filled the gap (answers corrected-only-through-my-PDF-check).
3. **False floor refusals on in-document content** (T3v, T4, T6): retrieval
   succeeded (score ≈0.72–0.83), M5 recovery attempted, deterministic gate
   still refused ("second_refusal"). With qwen3:0.6b, 3 of 4 books questions
   wanting synthesis refused.
4. **Degenerate output formatting** (T1 numbered list collapse; T2 mid-sentence
   opening, orphaned "3./4." items) — a small-model synthesis weakness that
   passed QA unchanged.
5. **Two of three user PDFs failed ingestion** as `binary_garbage`
   (`NIST.AI.600-1.pdf.pdf`, `nasa_systems_engineering_handbook_0.pdf`) — the
   honest data gate worked, but the GenAI profile and NASA handbook are
   currently unsearchable.

## 9. Failed / incomplete / unobservable checks

- Retrieved passages per question: **not observable** (the UI exposes no
  passage viewer; number-of-supporting-chunks from agent logs is recorded
  instead).
- Whether citations were generated from the retrieved chunks' metadata or
  embellished: not observable from the UI; the PDF page-mismatch evidence
  (§8.2) suggests misattribution rather than metadata corruption.
- Optional second-PDF downloads (GenAI profile, NASA handbook): already present
  in `knowledge/books/` but deemed out of scope BEFORE ingestion failed on
  both (§3); no re-download attempted.
- Full pytest suite was not part of this task's scope (real-GUI testing only).

## 10. Verdict — **FAIL (as a document-grounded RAG demonstration on this corpus)**

Reason: the only perfect test was the one correct refusal (T5). Answered
questions contained true-but-uncited claims (prior-knowledge leakage), cited
pages that don't support them, broke formatting, and 3 of 8 books-route runs
falsely refused answerable content. One strictly-worded question escaped the
document entirely into web research. **Agentica is not deployment-ready on
this evidence.** The deterministic safety layer (no fabrication of pageless
refs, no invented numbers, honest limit notes, ingest gate) performed well and
is the reason failure modes are visible rather than silent.

## 11. Proposed next tests (not executed — need approval)

1. A larger/debus model (e.g. an Ollama 7–8B instruct) to separate model
   capability from retrieval quality in the false-refusals.
2. `BOOKS_MIN_SCORE` / grounding-gate calibration pass on real-corpus
   paraphrases (would be a *configuration* change — currently out of scope).
3. Router improvements with a human phrasing (this is the documented, frozen
   Phase 1 behavior; changing it is a scope decision).
4. Re-ingest the two failed PDFs after byte-level inspection of
   `nasa_systems_engineering_handbook_0.pdf` (magic=False indicates possibly
   corrupted/HTML download rather than a PDF).
5. Repeat with the separate, authorized `local-model` RAG-AE opt-in evaluation
   for measurement the harness can score offline.

## 12. Final Git state

Working tree identical to the start of the task plus this report file and the
`.diag/gui_rag_test_2026-10-24/` exports (all inside the already-untracked
`.diag/`); `08d44b6` unchanged; no committed, pushed, deployed, or destructive
commands; no settings or code changes; the two failed-ingestion manifest
entries are the only store state change, both recorded (nothing deleted).
