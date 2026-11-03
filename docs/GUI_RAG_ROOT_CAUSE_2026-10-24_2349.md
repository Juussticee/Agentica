# GUI RAG Root-Cause Investigation — Read-Only

Date: 2026-10-24 23:49 local. **No code, prompts, configuration, datasets, PDFs,
manifests, indexes or store contents were modified. No ingestion, no SearXNG, no
network, no model calls. No commits or pushes.** This is an investigation report;
nothing here was implemented.

---

## Exec summary

The read-only pass **overturns the GUI test report's central citation finding** and
locates the true causes of the other failures:

1. **Citations were most likely correct** — the app cites each retrieved chunk's
   *printed page label* (from the PDF's own `/PageLabels`), while the test report
   verified citations against *PDF page indices* (cover = 1). For NIST AI 100-1
   those differ by 4 (printed label N lives at PDF index N+4 for numbered body
   pages; front matter is labeled `1, 1, i, i, ii`). Re-checking the cited labels
   against the correct pages shows the claimed content is present there.
2. **Routing misses are structural**: the GUI runs the supervisor's frozen keyword
   router, whose `books` pattern has no document-deictic phrase ("this document").
   The/user questions phrased like the strict tests route to `research`.
3. **False refusals are the token-coverage grounding gate colliding with a 0.6B
   synthesizer**, with M5's one regeneration also refusing again. The gate worked
   as designed; the combination (tiny model + narrow fixed token families) is the
   failure mode.
4. **Degenerate formatting is a model-capability limit** (qwen3:0.6b), not a
   downstream parsing defect; nothing in the app corrupts well-formed text.
5. The GUI report itself contains internal inconsistencies (a "9 executed runs"
   sum over an 8-run suite; a narration order at T3v that the code cannot emit).

---

## Repository state

- Start of this investigation: branch `main`, HEAD `08d44b6`,
  20 working-tree entries (5 modified, 15 untracked) — exactly the pre-task tree
  plus the previous GUI-report outputs (`docs/GUI_RAG_TEST_REPORT_2026-10-24_1114.md`,
  `.diag/gui_rag_test_2026-10-24/<8 records>.json`, the three knowledge PDFs),
  i.e. the GUI test's own artifacts. Verified with `git status --porcelain=v1`.
- End of this investigation: unchanged except this report file. Compare with
  `git diff` / `git status` at any time; only this new file was created.

## Artifacts used

- `docs/GUI_RAG_TEST_REPORT_2026-10-24_1114.md` (read in full).
- All 8 saved execution/event records in `.diag/gui_rag_test_2026-10-24/`.
- Current source: `app/supervisor.py`, `app/agents/books_agent.py` (all 1,103
  lines), `app/config.py`, `ingest_books.py`, `app/qa_node.py`, `main.py`,
  `app/tools/searxng_client.py`; installed `llama_index/readers/file/docs/base.py`.
- The user PDF `knowledge/books/NIST.AI.100-1.pdf` read **read-only via pypdf**
  (48 pages confirmed; SHA-256 unchanged from the GUI report's record).

> What the saved records can and cannot prove: event payloads carry `route` and
> the coarse step timeline only. The retrieval scores, `books_agent` console
> lines (e.g. "grounding refusal (35.7s, score=0.719)") and M5 outcomes quoted by
> the GUI report were observed in the backend console during that earlier
> session and were **not** persisted as artifacts. They are cited below as
> *reported* (sample consistent with code), never re-measured here.

---

## Investigation 1 — Router misclassification (books → research)

### Exact decision path

1. `app/api/server.py` receives the GUI question and builds executor state.
2. `app/supervisor.py::supervisor` → `keyword_route` (default `ROUTING_MODE`
   is `"keyword"` — `app/config.py L104-107`). In keyword mode the LLM
   classifier is **never consulted**.
3. `keyword_route` (`app/supervisor.py L66-80`): `feasibility` regex →
   `erp` regex → **`_BOOKS_RE`** → else `research`. Order is: neither
   feasibility nor ERP matched the strict T3 wording, and `_BOOKS_RE` did not
   match either, so the question fell to the final fallback.
4. Saved record `bd4ce5cc7cad.json`: `routing.decided` →
   `"Routed the request to Researching the web"`, `route: research` — observed.

### The rule that decided it

`_BOOKS_RE` (`app/supervisor.py L33-41`):

```python
\b(books?|chapter|pages?|page \d|textbook|handbook|manual|guide|
  author|bibliography|section|orbital|orbit|spacecraft|speed|altitude|
  bee|beekeeping|urban|hive|queen|drone|nectar|pollen|swarm)\b
```

All its members are either corpus-specific words (beekeeping, orbital) or
generic book-objects. There is **no member for "document", "framework",
"this file"** or any other deictic way the strict tests referred to NIST.AI.100-1.
"Measure", "Map", "GOVERN" are likewise absent (and were expected to be: the
corpus the regex was originally built for contained beekeeping and orbital-
kinetics books). This is a **brittle keyword mechanism, not a prompt-interpretation
failure**: routing never reached the LLM in the GUI run.

### Natural-language questions the current rules would misroute

Derived from the regex without submitting anything:

| Question | `_BOOKS_RE` hit? | Route |
|---|---|---|
| "According to the NIST AI RMF, what are the four core functions of the framework?" | no | research (wrong) |
| "How does this document define trustworthy AI?" | no | research (wrong) |
| "Summarize the AI RMF's purpose and its limitations." | no | research (wrong) |
| "What is the weather like in Beirut today?" | no | research (right, incidental) |

The books route is *not* reachable with natural document wording unless the
user injects `document`'s/section/page/handbook/... — confirmed by the GUI
session where every saved books run's question contains at least one authored
routing word ("sections", "page", "section").

> Also note the converse false-positive risk the keyword design carries: a
> research-style question that literally contains "guide"/"manual"/"speed"
> ("What speed does NASA recommend for ...?") routes to books.

### Existing architecture analysis (no "yet more keywords"-only proposal)

- `ROUTING_MODE="llm"` (`app/config.py L104-107`, `app/supervisor.py L82-101`)
  already frames two improvements but is default-off. It consults
  `get_llm(fast=True)` with a four-class prompt and falls back to keywords on
  any error/garbage/unknown label. Note the design's limit: when the LLM
  *succeeds* its label wins — with `qwen3:0.6b` classification is unreliable
  (mentally evidenced by the synthesis behavior), so llm routing over Ollama
  is only as good as the configured model.
- The keyword route is fenced by frozen tests (`scripts/test_phase1_routing.py`,
  `scripts/test_agent_architecture.py` freeze Phase 1-5 behavior). Any regex
  change must update those tests *deliberately*, not silently.

### Proposed minimal routing fix (NOT implemented)

1. **Config-only, first**: set `ROUTING_MODE=llm` in the environment and pin a
   classification-capable model for `fast=True` routing (e.g. an Ollama 7-8B or
   a reliable fallback), keeping the keyword fallback untouched. No code change;
   flips one documented opt-in.
2. **Then, one targeted regex widening** (code change, test-gated): add the
   document-deictic members the GUI conversation actually produced —
   `"this document", "the document", "according to the document"` — to
   `_BOOKS_RE` with an accompanying unit-test row in the routing tests. This
   is deliberately *adequate*, not exhaustive: doc-deictic phrases are the
   real-world shape that misrouted; "everything mentions documents" drift is
   the false-positive risk to guard against with those exact tests.
3. Regression tests: direct `keyword_route()` unit cases for the deictic
   phrasings (expect `"books"`) and `research`-guard cases for the existing labs
   ("What speed... NASA recommends..." stays research if desired); plus an
   end-to-end routing assertion mirroring strict T3's wording. Accept: no
   question in the GUI corpus routes to research unless it contains no deictic
   and no book-corpus word.

---

## Investigation 2 — Citation page "misattribution"

**Result: the primary claim of the GUI report (cited pages wrong) is itself
wrong at its verification basis — the app's citations carry the document's own
printed page labels, and the report re-checked them as if they were 1-based
PDF indices.**

### Metadata flow (code-verified end to end)

1. **Ingestion** (`ingest_books.py::process_file` → `_load_documents`):
   `SimpleDirectoryReader(input_files=[...])` loads one `Document` per PDF page.
   Installed `llama_index/readers/file/docs/base.py L92-94`:

   ```python
   page_label = pdf.page_labels[page]     # pypdf label of THAT page
   metadata = {"page_label": page_label, "file_name": file.name}
   ```

   `pdf.page_labels` is pypdf's expansion of the PDF's `/PageLabels` state —
   i.e. **the printed page numbers the document itself announces**, not
   1-based indices.
2. **Chunking**: `Settings.text_splitter.get_nodes_from_documents(...)` splits
   texts *after* page documents are built; node metadata is inherited from its
   source page's metadata (`page_label`, `file_name`). So each persisted chunk
   keeps the printed label of its page — one chunk, one label. Multiple
   chunks per page (repeated header labels, e.g. `['1','1','i','i','ii',...]` for
   NIST AI 100-1) are possible but never shown to the answer by page.
3. **Retrieval** (`app/agents/books_agent.py::_get_query_engine`): the persisted
   Chroma's top-`BOOKS_TOP_K=3` nodes are returned with their stored metadata
   intact (`node.metadata`). `_LocalPathFilter` only *removes* non-allowlisted
   metadata keys (`path/date/...`) from the LLM-facing string; `page_label` is
   on the M4B allowlist `_LLM_SAFE_METADATA` and still visible.
4. **Footer** (`books_agent.py::_reference_footer` and `_claim_reference_footer`,
   both invoked from the tail of `books_agent()`): the References footer is
   built **exclusively** from `source_nodes[idx].metadata` for nodes that
   support retained claims. `_source_ref` renders `"file_name, page X"`.
   The model cannot inject a page here: `_strip_model_citations` removes any
   model-emitted footer first, and `_inline_citation_error` rejects inline
   page/filename references outside the retrieved allow-list.
5. Therefore **a citation string can only name a page that a genuinely retrieved
   chunk legitimately owned**. It is *metadata-accurate by construction*.

### What the PDF itself proves about the cited labels

`pypdf.PdfReader.page_labels` for NIST.AI.100-1:

```
['1','1','i','i','ii','1','2','3', ... '43']   (48 PDF pages)
```

i.e. printed label N corresponds to PDF index N+4 for numbered body pages.

| Cited label | PDF index | What the label page really contains |
|---|---|---|
| 3 | 8 | "…valid and reliable, safe, secure and resilient, accountable and transparent, explainable and interpretable... Part 2 comprises the 'Core' of the Framework. It describes four specific functions…" |
| 20 | 25 | "Part 2: Core and Profiles / 5. AI RMF Core … the Core is composed of four functions" — **and the word cross-cutting** |
| 42 | 47 | "Appendix D: Attributes of the AI RMF … The AI RMF strives to: [risk-based, resource-efficient, pro-innovation, voluntary / consensus-driven …]" |
| 47 (expected, GUI report) | 51 — **out of range** | — (the GUI report pointed at index 47 and read the same Appendix-D text; that page is *labeled* 42) |

`grep`-checked and pypdf-verified. **Citations "page 20" for the four-function /
cross-cutting claim and "page 42" for the purpose attributes are consistent
with the actual label pages.** The "cited pages do not contain claims"
conclusion in the GUI report came from reading PDF indices 3/20/42 (1-based
index, yielding disclaimers / safety / governance-actors pages) instead of
labeled pages 3/20/42.

What remains a *hypothesis* (correctly labeled as such):

- Whether **every** cited-page→claim pair is correct under the label scheme —
  the GUI report's page-content claims about indices 12/13/16/3/etc. were not
  re-derived here sentence by sentence; a full replay would re-run the GUI
  suite, which this task does not do. The mechanism is proven by construction;
  the four key labels are proven correct by the table above.
- Claim-level attribution (which of the top-3 chunks *actually* backs each
  sentence) may still be loose: `_grounded_covering_nodes`' splice fallback
  (any node whose token set overlaps) can attribute a claim to all three
  chunks. That degrades *attributability*, not page correctness.

### Strongest evidence-backed root cause (bookkeeping, not app logic)

The **citations are accurate for the metadata they are built from.** What
misfired is the *verification basis* of the test report: comparing
`page_label` values (printed folio, offset by front matter) with PDF-index
numbers (0-based or 1-based offsets from the cover). Off-by-four systematically
made every correct label look wrong.

### Smallest proposed fix (NOT implemented)

1. Add a one-page appendix note to docs: "the books agent reports the page
   label of the PDF (`/PageLabels`), which is not the PDF viewer index for
   front-matter-bearing documents", and re-check any future scoring against
   labels. Zero app-code change.
2. Optional hardening (only if project wants viewer-consistent refs): retain
   both `page_label` (printed) and `page_index` (1-based PDF) per chunk and
   emit `"page 20 (viewer p. 25)"` in footers. Requires ingest-metadata change
   plus footer formatting and tests; defer as scope decision.
3. Tests that would have caught the loop:

   ```python
   # (a) label-vs-index acceptance test
   def test_footer_cites_printed_label_not_viewer_index(...): ...
   # (b) claim-support spot check: for each cited page, at least one claim's
   #     tokens must occur in THAT page's text — catches attribution looseness,
   #     which is a separate bug from off-by-one errors.
   ```

---

## Investigation 3 — False refusals and the grounding gate

### The actual criteria (`app/agents/books_agent.py`)

- Retrieval: `BOOKS_TOP_K=3` nodes from the whole `books` collection
  (single shared collection; no per-PDF subsetting).
- Score floor: `BOOKS_MIN_SCORE=0.35` (`app/config.py L118`). Score note says
  the floor check happens in parallel with the prompt's own refusal contract:
  "the nomic similarity band is too narrow for a fixed threshold alone."
- Gate `_grounded_repair`: every substantive sentence of the model's answer
  must have **all** of its content tokens (stop-words removed, plural/`-ed`/`-ing`
  stem, `\d+`-exact numbers, a bounded `_PARAPHRASE_GROUPS` alias set for a few
  domain families) present in the union of the three retrieved chunks' text and
  their allowlisted metadata. Any uncovered sentence is *dropped and disclosed*;
  an answer with **no** supported sentence is the mandated refusal
  (no footer, deterministic — `NOT_FOUND_PHRASE`).
- M5 recovery (`_self_refusal_retry`, `_RECOVERY_MIN_COVERAGE = 0.5`): runs
  exactly once only when the model's *first* answer is a pure refusal
  (`_refusal_only`), the score clears the floor, retrieval is non-empty, AND
  `_question_covered_by_evidence` confirms ≥50% deterministic question-to-evidence
  topical coverage; the regenerated text is then re-run through the *unchanged*
  gate. Any second refusal = abstain (`second_refusal`).
- Verifier M4B is `off` by default (`BOOKS_VERIFIER = "off"`); the GUI run's
  prints (per report) show M3-style deterministic gating only — consistent.

### Trace reconstruction per case

Saved event records (7-9 events each) confirm only: route, `agent.started/
completed`, `qa.completed`, timestamps/latency. The report additionally cites
backend console lines (scores 0.719/0.827/0.803, `second_refusal`) — those are
*reported* evidence, not replayable from artifacts.

**T3v `fa59003036d0`** (Map-vs-Measure): narrative has both "grounding refusal
(score=0.719)" and "second_refusal". **This combination is not producible by the
current code order** — recovery happens before the final gate, and
`second_refusal` prints only when recovery is *entered* (first response was a
pure refusal). One of the two console lines was either misremembered or
transcribed out of order. Classifying the earliest divergence point as
"model refuses question-evidence pair at synthesis" is the strongest defensible
reading; whether the *gate* or the *model* refused twice cannot be settled from
saved data. Label: **unresolvable from saved artifacts**; M5 eligibility
(topical coverage ≥50%, good score) was met, so retrieval was *topically* fine.

**T4 `7084d223e392`** (GOVERN role): printed `second_refusal` after recovery.
Since `_question_covered_by_evidence` was satisfied and the *actual* document
support lives on **label page 20** (PDF index 25 — "the Core is composed of
four functions ... GOVERN is a cross-cutting function"), the evidence for the
asked claim was *in corpus and plausibly retrieved*; the divergence is at
**model synthesis** (0.6B answered refusal / unsupported text), not retrieval
and not topology. This is the M5-documented shape: a small instruction model
producing a self-refusal even with covering passages.

**T6 `9b6b5d5378d3`** (paraphrase → MAP): same `second_refusal`. Evidence for
"MAP identifies context and risks" exists in corpus (§3.1, printed ~p. 23-27);
the paraphrase "harms" vs corpus "risks" is outside `_PARAPHRASE_GROUPS`, so
even a well-meaning answer that used "harm(s)" would be gate-refused unless
the model happened to echo "risks". Two compounding factors:
*(a)* model paraphrase not covered by the token-coverage alias families →
grounding-gate false negative, and *(b)* identical M5 second-refusal shape.

So per case the earliest divergence: |
|---|---|
| T3v | Model output refused/unsupported at synthesis; could not fully localize from saved artifacts |
| T4 | Model synthesis (twice: initial + M5 regeneration) |
| T6 | Model synthesis AND gate's paraphrase-family gap (likely a combination) |

None of the three shows a retrieval or context-construction failure as the
*earliest* defect. **The validator behaved as designed**; its engineering
limit (token-covered coverage aliases are corpus-specific to bee/orbital
families, and the ModelSet of NIST vocabulary is not represented) turned a
good retrieval into an abstention.

### Minimal remediation proposals (NOT implemented; no gate weakening)

1. **Widen `_PARAPHRASE_GROUPS` with corpus-verified, review-approved families
   discovered from the real corpus**, not arbitrary synonyms — specifically
   `{identif, context, risk, harm}` etc. — each addition frozen by unit tests
   exactly like the existing families (the file's own contract).
2. **Bound the required coverage further instead of relaxing it**: allow
   "≥ all content tokens except one open-vocabulary adjective" — requires a
   precise definition and adversarial tests. Only take if (1) is still
   insufficient. Zero gate disabling.
3. **Raise the recovery budget deterministically**: allow exactly `2`
   regenerations *only* when topology is unambiguous (`_question_covered_by_evidence`
   passes AND `_grounded_repair` previously returned "partial" with at least one
   supported sentence) so it cannot be combined with weak retrieval. Bounded,
   testable, never a loop.
4. Tests: extend `scripts/test_self_refusal_recovery.py` (already constrained)
   with real-corpus paraphrase fixtures for NIST phrasing; keep the mandate:
   **no weakening of `_grounded_repair`**; the alternative is the honest
   abstention it already guarantees.

---

## Investigation 4 — Small-model output quality (`qwen3:0.6b`)

Sanity check first: no downstream transformation breaks numbering —
`_strip_model_citations` only deletes footer-shaped blocks; `_grounded_repair`
keeps/discloses whole sentences; footer appends *after* the model text.
Therefore everything reported by the GUI under "1. … 2. 3. 4." formatting,
mid-sentence starts, orphan list items, came **pre-formatted** by the model.

| Defect | Most likely source |
|---|---|
| "1. **GOVERN** … 2. 3. 4." — numbering collapses | Model capability (0.6B) + dense instructions QA prompt with "Answer ONLY the first/third/second/third items and do use/trust/pass hedges… A3" |
| T2 starts mid-thought ("These characteristics …") | Model; the QA prompt's request to "state the covered facts first" plus truncation of a failure to plan a list |
| T7 partially answers, drops part 3 | Model; also legitimate if the retrieval really lacked a `limitation` passage tokens-wise (dated for any *top-3* chunking limitation: `BOOKS_TOP_K=3` cannot hold enough spans of the real corpus for multi-part synthesis) |
| Token budget / truncation | Simulated context is generous (200k in `_GetLlmBridge` metadata), so not truncation; the 0.6B decoder is the limit |
| T3 web answer unreadably literal | Different agent; again `research` agent behavior, not the synthesis path |

**Fixable in app code** (no model replacement): none of the numbering/format
defects are code-fixable without pretending generation quality is app
behavior. The tractable app-level items are: cleaner instruction order in the
QA prompt (priority: answer-then-hedges), potentially `BOOKS_TOP_K` raise
(config, not code) for multi-part questions, and gate-family widening for the
specific paraphrases. **Model-inherent**: transcription quality itself, list
numbering, and grated "2. 3. 4." patterns; these need a bigger model to
demonstrably fix (the report's own next-test proposal, unexecuted here).

---

## Investigation 5 — Report cross-check and integrity

### Verifiable now and confirmed

- **8 executions** in `.diag/gui_rag_test_2026-10-24/` — 7 books, 1 research
  (`bd4ce5cc7cad`) — matching the §4 sentence "for 7 of 8 executions".
  ⚠️ The report's score section says **"Sum across the 9 executed runs:
  36/90"** (line 327) and elsewhere "8 book runs, 7 completion": both numbers
  conflict with the saved artifact set and with each other. Denominator
  correction: **8 executed runs, 36/80 points over that set (45%)**.
- Provider/model claims (`ollama` + `qwen3:0.6b`): consistent with the config
  defaults and with Ollama-tag usage; **not independently re-verifiable** from
  artifacts (no provider metadata is written into saved records).
- Score/refusal observations (0.67–0.83, `second_refusal`): 
  *reported* from live console only.
- Enforcement of T5 (calibration test): the saved record alone suffices to
  confirm the final not-found answer (`"The requested information was not
  found in the books."` for `22af069916ec`).
- Git-state claims: head/branch/tree counts line up with the current tree
  (this pass's own `git status` at top).

### Internal inconsistencies (flagged, not corrected in place)

1. "9 executed runs" vs 8 saved executions (denominator change).
2. T3v narration order (recovery-after-grounding-refusal) that code cannot
   emit; preserve as-is in the original file per "do not overwrite any
   existing artifact" — this report documents it instead.

---

## Consolidated cause table

| # | Phenomenon | Confirmed cause | Likely cause | Unresolved hypothesis |
|---|---|---|---|---|
| 1 | Books questions → research | Frozen keyword regex lacks deictic phrases; ROUTING_MODE=keyword in GUI | — | — |
| 2 | Citations "wrong pages" | **Verification artifact**: label-vs-PDF-index off-by-4; app emits genuine printed labels | — | Whether every remaining label pairing is correct (only label 3/20/42 re-derived here) |
| 2b | Loose claim attribution | Splice fallback (`covering_nodes`) may vote for all retrieved chunks | — | Scope of case-level impact |
| 3 | False refusals (T4/T6) | Deterministic token-coverage gate + 0.6B model's refusal-shaped answers | `_PARAPHRASE_GROUPS` coverage gap for NIST vocabulary ("harm" vs "risk") | T3v exact split: model- vs gate-initiated refusal |
| 4 | Degenerate formatting | qwen3:0.6b synthesis (no app defect in the append/strip path) | Prompt ordering contributes | — |

## Prioritized remediation plan (nothing executed)

1. **Citation page-vs-index documentation + acceptance test** — cost ≈ zero;
   eliminates the misleading test-report finding that produced the wrong
   overall score impressions.
2. **Routing**: (a) flip `ROUTING_MODE=llm` config with a sane classifier
   model (no code), (b) expand `_BOOKS_RE` with 3 deictic members + unit tests.
3. **Refusal calibration**: corpus-driven `_PARAPHRASE_GROUPS` expansion with
   adversarial tests; optionally one extra M5 generation only when partial
   support existed.
4. **Capability**: replaces `qwen3:0.6b` in `MAIN_MODEL` (config) for GUI
   sessions where quality matters; keep 0.6B for latency-sensitive paths.
5. **Only after 1-4**: re-run the 7-test GUI suite on the same PDF to obtain
   numbers the earlier report's corrections recommend (this plan, not this
   task).

## No-op confirmation

No code/data/file was modified except this report; no ingestion, no SearXNG,
no network, no model calls, no production-store interaction; no commits/pushes.
