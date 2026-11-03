# RAG-AE — Answer-Quality Evaluation Harness

Status: **implemented; offline deterministic baseline established** (see
`benchmarks/rag_ae_unit_v1.json` and `benchmarks/rag_ae_integration_v1.json`).
This harness does **not** make Agentica production-ready and does not measure
live-model answer quality (see *Limitations*).

Owner task: MASTER_PLAN RAG-1 §7 item 5 / PROJECT_STATUS "outstanding item 2".
Implementing files: `scripts/eval_answers.py` (harness),
`scripts/test_eval_answers.py` (63 offline tests),
`benchmarks/rag_gold_v2.json` (dataset, sha256-pinned).

## 1. What it measures (and what it does not)

RAG-AE separates **retrieval quality** from **answer quality**. A single
aggregate score is never reported.

**Answer layer (unit mode — the primary suite).** Each case is a scripted
(question, evidence, raw model reply) triple driven through the REAL
`books_agent` (query engine and `get_llm` patched at the narrowest boundary;
no Chroma, no model, no network). The harness then scores the final answer
against deterministic expectations across nine dimensions: correctness
(required facts present), groundedness (independent token/number support
check), evidence use, completeness, relevance (term-overlap proxy),
citation/source correctness, abstention, contradiction handling (forbidden
facts absent), and robustness (distractors ignored). M5 refusal recovery is
scored separately from captured branch markers and scripted LLM call counts.

**Retrieval layer (integration mode).** Cases run through the REAL pipeline:
production reader → chunker → `process_file` → temp Chroma → production
`books_agent` seam → postprocessors, over the real corpus plus benchmark
fixtures in a fresh temporary store. Metrics: hit@k, precision@k,
reciprocal rank, nDCG@5, and fact coverage (required facts literally in the
retrieved top-k context).

**It does NOT measure:** live-model semantic answer quality or instruction
following; the production embedder's ranking (integration uses a labeled
deterministic lexical double); model-level entity discrimination between
lookalike values that are both in the retrieved text; M5 effectiveness with
a live model. Mocked answer tests establish the pipeline contract, not the
model.

## 2. Dataset: `benchmarks/rag_gold_v2.json`

19 cases: 16 executable (C001–C016), 3 unresolved (C101–C103, excluded from
every metric). Schema (full field list in the file's `schema` block):

- `id`, `section`, `synthetic`, `status` (`executable`|`unresolved`)
- `query`, `answerable` (`true`|`false`|`partial`)
- `gold_documents`, `gold_pages`, `min_docs_covered` — retrieval-layer labels
- `required_facts` — verbatim-anchored answer facts (substring-checked)
- `evidence[]` — scripted retrieval output for unit mode:
  `{source: corpus|fixture|scripted, file_name, page_label, text, score}`.
  `corpus`/`fixture` evidence text MUST be a verbatim whitespace-normalized
  substring of the named source file (validated at load); `scripted` is
  self-contained synthetic text.
- `raw_response` — the deterministic model double's reply; drives the real
  grounding/citation/refusal/recovery gates
- `recovery_replies`, `script` (e.g. `{"retrieval_error": true}`)
- `expect` — abstain, facts_present, disclosure (`none`|`partial`),
  forbidden_facts, citation (`required`|`forbidden`|`unavailable`),
  citation_files, forbidden_citation_files, recovery_attempted,
  recovery_outcome, llm_calls, error
- `modes` — `["unit"]`, `["integration"]`, or both
- `labels`, `rationale`, `review_notes`, `provenance`

**Adding a case:** author it in the same shape, anchor every corpus/fixture
evidence text verbatim from the source file, then run
`env\Scripts\python.exe -m pytest scripts/test_eval_answers.py -q` —
`DatasetValidationTests.test_real_dataset_is_valid` rejects duplicate IDs,
missing/renamed documents, non-verbatim evidence, answerable cases without
identifiable supporting evidence, empty/contradictory expectations, and
invalid citation targets, each with `case <id>, field <field>` messages.
After the first recorded baseline run the dataset is evidence: pin its new
sha256 in `test_eval_answers.py::test_dataset_sha_pinned` and do not edit;
changes go in a NEW version file.

## 3. Running

```
env\Scripts\python.exe scripts\eval_answers.py --mode unit            # default suite
env\Scripts\python.exe scripts\eval_answers.py --mode integration     # isolated retrieval
env\Scripts\python.exe -m pytest scripts\test_eval_answers.py -q      # harness self-tests
```

Exit codes: `0` all executed cases passed · `1` dataset invalid (per
case+field messages) · `2` isolation/guard/network/runtime refusal ·
`3` refusing to overwrite an existing `--out` artifact · `4` one or more case
assertions failed. `--out` defaults to
`benchmarks/rag_ae_<mode>_v1.json`; existing artifacts are immutable evidence
(`--force` to overwrite deliberately).

**Modes.** `unit` (deterministic doubles, offline, default),
`integration` (real retrieval path over a temp store), `local-model`
(OPT-IN ONLY: requires `--enable-live` AND `LLM_PROVIDER=ollama`; refuses
remote providers and never falls back to one; not part of the acceptance
suite; not executed by the RAG-AE task).

## 4. How test-store isolation is enforced

1. `chroma_guard` armed fail-closed for the harness process; every
   `chromadb.PersistentClient` construction is checked BEFORE construction
   (production store always refused).
2. Integration mode creates a fresh `TemporaryDirectory`; `verify_isolation`
   refuses any path equal to the production store, disallowed by the guard,
   or outside the system temp directory. There is no flag to point it
   elsewhere.
3. Production store files (`storage/chroma.sqlite3`, the ingest manifest) are
   hashed before/after; drift aborts with exit 2.
4. A network guard records and BLOCKS socket/urlopen; any recorded attempt
   exits 2 and invalidates the run.
5. After the run: all Chroma clients closed (Windows temp cleanup), every
   constructed client path asserted to be inside the temp dir.

## 5. Interpreting metrics

All rates are macro-averages over applicable cases; skipped and unresolved
cases never enter a denominator. Definitions are embedded in every artifact
(`definitions` block) and summarized in the run's stdout summary.
Groundedness is rule-based token/number coverage with production's own
disclosure/refusal strings exempt — the closest deterministic substitute for
entailment, NOT entailment (see `docs/RAG_EVAL_GAPS.md` known limitations).
Integration-mode retrieval numbers come from the deterministic lexical
embedding double (`HashedTfEmbedding`, hashed TF-IDF-like vectors with a
tiny stoplist): they measure the pipeline mechanics, and are NOT comparable
to the production-embedding baseline in `benchmarks/rag_baseline_v3.json`.

Failed cases carry actionable diagnostics classifying the failure as:
expected evidence absent from retrieval · evidence retrieved at inadequate
rank · evidence retrieved but lost during context assembly · answer-layer
grounding/refusal drop (with the missing tokens) · fabricated content
surviving (grounding leak) · FALSE REFUSAL (the C002 live-defect class) ·
citation mismatch · recovery-marker mismatch · fixture/evaluation-definition
problem (required fact not in the case's own evidence).

## 6. Unresolved cases

`status: "unresolved"` with a mandatory `unresolved_reason` and
`resolution_findings`. They are discovered, validated, reported — and
excluded from every executed count, metric and exit code.

> **UPDATE 2026-10-23 (v2 → v3 resolution pass).** The three unresolved
> records were re-investigated with source+test verification and carried
> into `benchmarks/rag_gold_v3.json` (sha256-pinned): **C102 is now
> EXECUTABLE** there against a new deterministic fixture
> (`benchmarks/rag_fixtures/fixture_multipage_table.pdf`), while **C101
> and C103 stay unresolved with VERIFIED-FINAL findings** (citations
> recorded in the file; summarized below). The v2 file keeps its own sha
> pin untouched — v3 is the additive resolution record and only C101/
> C102/C103 + the header differ from v2 (asserted in
> `GoldV3DatasetTests.test_additive_only_diff_against_v2`).

v2-era status note (kept as history):

- **C101** (two-passage different-value conflict) — VERIFIED unresolved:
  `test_case3_conflict_claim_with_affirmative_evidence_kept`
  (`scripts/test_books_grounding.py:331`) pins that the deterministic layer
  KEEPS a claim its own evidence affirms when two passages disagree
  affirmatively; exactly the affirm-vs-negate mirror shape
  (`TestConflictingSources`, `scripts/test_citation_integrity.py:324`) is
  deterministically resolvable, and executable **C009 already scores
  it**. Executing C101 as drafted would manufacture a known-limitation
  failure (an expected abstention contradicting pinned production
  behavior) or pass vacuously. Adjudicating between two affirmative
  values is model-knowledge entailment — live-model territory, the same
  residual class as `not_covered_here.model-entity-discrimination`.
- **C102** (multi-page table) — EXECUTABLE in v3: page-2 row
  identification among page-1 lookalike rows with a repeated column
  header (`gold_pages=[2]`, required fact `2.84`, forbidden adjacent-row
  `2.71 metres` and page-1 lookalike figures). Verified pipeline facts
  before promotion: per-page PDF documents with preserved `page_label`,
  and `Settings.text_splitter` = `SentenceSplitter(1024/200)` so each
  small page is ONE chunk and chunks never span pages — the page-2 chunk
  therefore carries the repeated header itself, and the cross-chunk
  row-split sub-case stays a documented limitation (forcing it would
  require production chunker changes, which RAG-AE does not make).
  Single-page table behavior remains frozen v1 evidence (gold-v1 S007).
- **C103** (scan "intake date") — VERIFIED unresolved: the prior report's
  framing stands, now evidence-pinned — an existing scan-like fixture DOES
  exist (`fixture_scan.pdf`, page 1 image-only; verified against
  `build_fixture_scan_pdf` + extraction), contrary to the draft's
  `blocked_on` note, and the behavior class is executable today as
  **C008** (gold-v1 S013's query; the image-only page cannot produce any
  chunk, so a fabricated "sign" answer must abstain). C103 itself stays
  unresolved: its query references content
  present in no fixture/corpus document, and its expected
  "no-extractable-text" wording is an ingest-manifest category
  (`ingest_books.py CAT_NO_TEXT`), not producible answer text.
  No production behavior was changed.

## 6b. v3 baselines (C101–C103 pass)

```
env\Scripts\python.exe -m pytest scripts\test_eval_answers.py -q
    # 68 passed (63 v2-era + 5 gold-v3 focused; both sha pins enforced)
env\Scripts\python.exe scripts\eval_answers.py --mode unit \
    --gold benchmarks/rag_gold_v3.json --out benchmarks/rag_ae_unit_v3.json
env\Scripts\python.exe scripts\eval_answers.py --mode integration \
    --gold benchmarks/rag_gold_v3.json \
    --out benchmarks/rag_ae_integration_v3.json
```

Observed (both exit 0, both modes deterministic under `strip_volatile`
comparisons, zero network, guard unchanged):

- **unit v3**: 17/17 executable passed (C102 now scores across all its
  applicable dimensions); C101/C103 reported unresolved.
- **integration v3**: 10/10 scored cases passed — the page-2 chunk of the
  new fixture is retrieved and the 2000 figure is covered in context;
  aggregate moved honestly: hit@1 0.889 → 0.8 (the correct chunk is not
  rank-1 under the lexical embedding double), hit@3 stays 1.0,
  fact_coverage 1.0, precision@1 0.9.
- One iteration was needed: the first C102 draft refused at the answer
  layer (mandated not-found) because the fixture carried no natural-
  language sentence, so the model double's fluent reply had token
  support nowhere — the same failure class C009 encodes. Fixed by adding
  one evidenced sentence to the fixture page 2; decoy rows remain bare
  table cells so they cannot ground a wrong-row answer.

## 7. Reproducing the baseline

```
git status --porcelain        # note pre-existing state
env\Scripts\python.exe -m pytest scripts\test_eval_answers.py -q     # 63 passed
env\Scripts\python.exe scripts\eval_answers.py --mode unit
env\Scripts\python.exe scripts\eval_answers.py --mode integration
```

Both runs are deterministic: `strip_volatile()`-compared artifacts from two
runs are byte-equivalent apart from run_id/timestamp/runtime, the output path,
and the disposable-store tempdir probe path (all masked by `strip_volatile`;
pinned by `CliTests.test_deterministic_rerun`). The dataset sha256 in each
artifact ties results to the exact fixture set.

## 8. Limitations and remaining gaps

- Offline/mocked results establish the deterministic PIPELINE contract
  (grounding gates, refusal shape, citation rebuild, M5 bounds) — they
  cannot establish live-model answer quality, and the live M5 effectiveness
  question stays open (mode `local-model` exists for a separately documented
  optional run).
- The corpus is two real documents + five synthetic fixtures; passing all
  cases is evidence about these cases, not about general RAG quality.
- Deterministic groundedness cannot detect entity misattribution between
  lookalike values in the same context (C010 documents this residual),
  cross-passage token splicing, or same-value conflicting evidence (C101);
  see `docs/RAG_EVAL_GAPS.md`.
- Retrieval-layer numbers use a lexical embedding double; production-embedder
  retrieval quality remains the frozen gold-v1 baseline
  (`benchmarks/rag_baseline_v3.json`, `scripts/eval_retrieval.py`).
- The books retrieval path has no metadata filter stage, so
  metadata-filter correctness is scored at the answer layer only (C001
  file-scoped question, C007/C013/C016 attribution).
