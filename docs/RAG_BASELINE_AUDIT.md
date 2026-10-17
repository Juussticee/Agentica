# RAG-1 Baseline Audit (metric integrity, abstention proxy, isolation, reproducibility)

**Audit date:** 2026-10-17 · **Audited artifact:** `benchmarks/rag_baseline_v1.json`
(run `books_eval_20261016-184730`, written 2026-10-16 18:47:46, file sha256
`2c1d4ae9de438a25c761680b06fe083022af4a8c7c27872e8f6a1e13027c659d`)

**Scope:** audit-only. No application behavior changed, no RAG-2 work started,
no baseline/gold/fixture/test/plan file modified. This document is the only
file created inside the repo; scratch artifacts live in the system temp dir.

> **Status note (2026-10-17, RAG-1 closeout):** §§1–6 are the historical
> audit of **v1** and are preserved as written; only status markers were
> added where a sentence says a proposal was *not yet* implemented. The
> current frozen artifact is `benchmarks/rag_baseline_v3.json` — see
> **§7** for the v2/v3 addendum (revised relevance rule, affected cases
> R011/S016, top-five evidence schema, artifact hashes, comparability
> rule).

---

## 1. Metric formulas and independent recalculation

All formulas below are stated as documented in the report's
`metric_definitions` block and then verified against an independent
re-implementation (written from scratch in `%TEMP%\audit_recompute.py`;
it does **not** import `eval_retrieval`).

### 1.1 Relevance rule (`node_covers`)

A retrieved node is *gold evidence* for a case iff:

1. `node.file_name ∈ case.gold_documents`, **and**
2. page constraint: if `case.gold_pages` is empty/absent → any page passes
   (file-only evidence, e.g. `.docx`); otherwise the node's
   `page_label`/`page_number` must parse to an integer ∈ `gold_pages`.

A node with no page metadata **fails** a non-empty `gold_pages` constraint
(int(`"None"`) raises → `False`). This detail matters — see §1.5.

### 1.2 Per-metric formulas (all per query, then macro-averaged over queries)

Let `covers[i]` = relevance of the node at rank `i+1`, and
`min_docs = case.min_docs_covered` (default 1).

| Metric | Formula | Denominator |
|---|---|---|
| `hit@k` | `1` iff #**distinct gold file names** among covering nodes of ranks ≤ k `≥ min_docs` | n/a (boolean per query) |
| `precision@k` | `(# covering *nodes* in top k) / k` | `k` (fixed; duplicates count per node) |
| `reciprocal_rank` (MRR component) | `1 / r` where `r` = rank of the **first covering node**; `0` if none | per query |
| `ndcg@5` | `1 / log2(1 + r)` with `r` = first-covering-node rank (binary relevance, single relevant item in the ideal ranking); `0` if none | per query |
| `coverage` | `1` iff #distinct gold files in top **5** `≥ min_docs` | boolean per query |

- **Averaging:** macro per query — every case weighs `1/n`. Never per
  relevant document. Section aggregates use **answerable cases only**
  (`n=11` real, `n=14` synthetic).
- **Unanswerable questions:** excluded from every aggregate. They are
  listed with all-zero metrics per case and counted as
  `unanswerable_cases: 4 / 3`. A query with no relevant result scores
  `reciprocal_rank = 0`, `ndcg@5 = 0` — but that path is only reachable in
  this run for unanswerable cases (all answerable cases hit), so it never
  enters an aggregate.
- **Duplicate chunks:** `precision@k` counts each covering node, so two
  gold chunks from the same file each contribute; `hit@k`/`coverage` count
  **distinct file names only**, so duplicates can never inflate them.
- **Multiple relevant documents:** only `hit@k`/`coverage` honor
  `min_docs_covered`. MRR/nDCG use the first covering node and are blind
  to the multi-document requirement (§1.5).

### 1.3 Independent recalculation result — reported numbers are correct

Recomputed from `benchmarks/rag_gold_v1.json` + the report's own
`retrieved_top3` rankings, using an independent implementation:

- **Per-case:** every recorded `hit@1`, `hit@3`, `precision@1`,
  `precision@3`, `first_hit_rank` reproduced — **0 mismatches** across all
  32 cases.
- **Real corpus (11 answerable):** hit@1 = 10/11 = **0.9091** ✓ ·
  hit@3 = 10/11 = **0.9091** ✓ · precision@1 = **1.0** ✓ ·
  precision@3 = **0.3333** ✓ · MRR = **1.0** ✓ · nDCG@5 = **1.0** ✓
- **Synthetic (14 answerable):** hit@1 = 12/14 = **0.8571** ✓ ·
  hit@3 = 13/14 = **0.9286** ✓ · precision@1 = 12/14 = **0.9286** ✓ ·
  precision@3 = **0.3333** ✓ · MRR = (13·1 + 0.5)/14 = **0.9643** ✓ ·
  nDCG@5 = (13·1 + 0.63093)/14 = **0.9736** ✓

**No arithmetic or aggregation error exists.** Every reported figure is
reproducible from the recorded per-case data under the stated formulas,
including the aggregate denominators (11 and 14).

### 1.4 Why hit@1 = 0.9091 but MRR = 1.0 (the suspected inconsistency)

This is **not a numeric error — it is a definitional split**, and it is
real:

- `hit@1` requires `min_docs_covered` **distinct gold documents** in the
  top-1. For **R011** (`min_docs_covered = 2`) that is *impossible by
  construction* — one slot cannot hold two documents — so `hit@1 = false`.
- `reciprocal_rank`/`ndcg@5` only ask for the first covering **node**;
  R011's rank-1 node covers, so RR = 1 and nDCG@5 = 1.

Hence MRR (1.0) > hit@1 (0.9091) legitimately: the two metric families
measure different success criteria in multi-doc cases. The same pattern
appears in the synthetic set (S016). The report's `metric_definitions`
block *does* state both definitions, but it does not warn that they can
diverge; a reader comparing MRR against hit-rate across phases will draw
wrong conclusions if this is not carried forward.

### 1.5 Found: both multi-doc gold cases have a page-gating defect (report only — no code changed)

`gold_pages = [1]` in R011 and S016 gates **every** gold document in the
case, including the page-less `.docx` (`orbital_kinetics_principles.docx`
for R011, `fixture_headings.docx` for S016). A `.docx` node has no page
metadata, so `node_covers` can **never** return true for it whenever
`gold_pages` is non-empty. Consequences:

- **R011:** the docx gold document is un-coverable → top-3 yields at most
  1 distinct gold doc vs `min_docs = 2` → `hit@3` and `coverage` are
  **structurally predetermined false regardless of retrieval quality**,
  while its RR/nDCG stay 1.0. (The retrieved orbital docx in fact sat at
  rank 2.)
- **S016:** same — `fixture_headings.docx` sat at rank 2 but cannot cover →
  `hit@3`/`coverage` false regardless.

Independent re-run of the evaluator under the loosened rule "a node with
no page metadata satisfies the page constraint" (page-labeled nodes still
gated) changes **only** these two cases:

| Metric | Reported | Under loosened page rule |
|---|---|---|
| real hit@3 | 0.9091 | **1.0000** |
| real coverage | 0.9091 | **1.0000** |
| synthetic hit@3 | 0.9286 | **1.0000** |
| synthetic coverage | 0.9286 | **1.0000** |
| hit@1, MRR, nDCG@5, precision (both sets) | — | unchanged |

This does **not** mean the reported values are wrong — they are correct
under the implemented rule, which itself is what production-era retrieval
was held to. It means two cases' coverage-family scores are properties of
the relevance rule, not of the retriever, so they cannot serve as a
retrieval-quality signal for RAG-2. The minimal proposed fix (per-doc page
applicability in `node_covers`, e.g. skip the page filter for nodes
without page metadata) is listed in §6 and, at audit time, **required
approval before any implementation** (approval given; implemented 2026-10-17
in v2/v3 — see §7); the existing baseline and gold set were not touched.

### 1.6 Transparency gap: hit@5/coverage are not independently verifiable

The report stores only `retrieved_top3`, but `hit@5` and `coverage` are
computed from the top-5 list. For min_docs = 1 cases with a hit inside
top-3 (all of them) and for unanswerable cases (no gold docs exist), the
values are decidable and consistent; for the two multi-doc cases the
recorded `hit@5 = false` is consistent with the top-3 evidence but cannot
be re-derived from the artifact alone. **Proposed (needs approval): record
`retrieved_top5` rankings in the report next run.** *(implemented in v3 —
`artifact_schema.version=3`; see §7.2)*

---

## 2. Abstention-proxy audit

### 2.1 Mechanism

`abstention_proxy(top1_score, answerable)`:

```
proxy_abstain := (top1_score is None) OR (top1_score < BOOKS_MIN_SCORE)
correct       := proxy_abstain XOR answerable
```

- `top1_score` = production top-1 retrieval score from the production
  seam (llama-index → Chroma cosine; confirmed in installed source:
  `similarity_score = math.exp(-distance)`), matching the report's
  recorded `score_semantics`.
- `BOOKS_MIN_SCORE = 0.35` (`app/config.py:116`); it is a **hard score
  floor**, explicitly "a deterministic proxy, not a calibrated
  probability" (recorded in `config`).
- Production refusal is strictly richer: `books_agent.py:795` computes
  `low_confidence = score is not None and score < 0.35`, and line 802
  *additionally* gates on prompt-level refusal detection and (line 817)
  on the deterministic grounding-repair gate. The proxy models **only
  the score floor**.

### 2.2 Every unanswerable case, with evidence

Real corpus (threshold 0.35 — should the system abstain? does the proxy agree?):

| Case | top-1 score | floor | should abstain (floor) | proxy says | outcome |
|---|---|---|---|---|---|
| R012 | 0.650900 | 0.35 | no | no abstain | incorrect (expected `abstain`) |
| R013 | 0.729732 | 0.35 | no | no abstain | incorrect |
| R014 | 0.681883 | 0.35 | no | no abstain | incorrect |
| R015 | 0.785900 | 0.35 | no | no abstain | incorrect |

Synthetic:

| Case | top-1 score | floor | should abstain (floor) | proxy says | outcome |
|---|---|---|---|---|---|
| S013 | 0.747266 | 0.35 | no | no abstain | incorrect |
| S014 | 0.712682 | 0.35 | no | no abstain | incorrect |
| S015 | 0.726903 | 0.35 | no | no abstain | incorrect |

**0/4 and 0/3 recomputed — matches the report.** Also 0/11 and 0/14 false
abstains on answerable cases, as recorded.

### 2.3 Interpretation (the honest version)

All 32 top-1 scores fall in **[0.6509, 0.8307]** — roughly 2–3× the 0.35
floor. Under `exp(-cosine_distance)` the floor corresponds to a cosine
distance of ≈1.05, far outside anything a single-chunk corpus query can
produce; dense retrieval always returns *something* above the floor.
Therefore:

- The proxy **never fires for any query in this run**, answerable or not.
  0/4 and 0/3 mean "the score floor is unreachable and proved nothing
  about unanswerability", **not** "the system will hallucinate 7/7
  answers".
- The proxy does **not** measure answer-level abstention. Production
  refusal also depends on the synthesis prompt, refusal-phrase detection,
  and the grounding gate — all chat-model behavior, explicitly outside
  RAG-1's authorization. No claim about answer refusal can be made from
  this baseline.
- Asymmetry noted: the proxy treats `top1_score is None` as *abstain*,
  while production (`score is not None and ...`) treats `None` as *pass
  the floor*. Not exercised in this run (no `None` top-1), but the two
  implementations are not identical by design.

No threshold change was made; no chat-model calls were run.

---

## 3. Isolation and store-safety evidence

### 3.1 What the harness isolation proves (verified in code, report, and tests)

- **Temp-only clients:** the report's `guards.chroma_client_pathsConstructed`
  lists 3 client constructions, all under
  `%TEMP%\rag_eval_bqdavnh6` — `all_chroma_clients_on_temp_paths: true`;
  the audit rerun likewise recorded only `rag_eval_kap7br68`.
  `chromadb.PersistentClient` is monkey-patched for the whole run, and any
  off-temp path raises → exit 2, report not written.
- **Production engine cache never reused:** the seam snapshots and nulls
  `books_agent._query_engine`/`STORAGE_DIR`/`CHROMA_COLLECTION` and
  restores in `finally` (including on failure — asserted by
  `test_production_seam_restores_namespace_on_failure`);
  `production_query_engine_cache_used: false`.
- **Deterministic isolation proof:** `IsolationProofTest` builds an eval
  index on temp paths with `MockEmbedding`, asserts every constructed
  client path contains the temp dir and never `STORAGE_DIR`, asserts
  production globals restored *by identity*, proves the temp dir is fully
  removable (no leaked handles), and stat+hash-guards the real store
  across the test. It passed in this audit (§5).
- **Fidelity check:** production top-3 == top-5 prefix for all 32 queries
  (`true`), so top-3 metrics describe the production path.
- **Gold separation:** gold SHA-256 recorded before *and* after the run;
  current file hash `16e25677f2effbd65bba0475754f7b9f5c85344d468497dd322234b155bbb25d`
  **matches the recorded value exactly** — gold untouched by audit.
- **Synthetic≠real:** fixtures live in `benchmarks/rag_fixtures/`, are
  separately hashed in `corpus.fixture_files`, cases carry
  `synthetic:true` + `section:"synthetic_fixtures"` (validated on load),
  and all fixture metrics are reported in a **separate section** with an
  explicit warning line. Fixture corpus hashes vs report: 0 mismatches;
  real corpus hashes: 0 mismatches (7 files).
- **Run metadata:** `git_head` =
  `7a44a9a7f747a983d2ffeb2df767f7f4f265b5f2` = **current HEAD**; config
  snapshot (top_k=3, floor 0.35 — matches `app/config.py`),
  embedding-model identity (nomic-embed-text digest
  `0a109f42…5e59f`, 768 dims, size/family/quantization from the local
  Ollama tags endpoint), package versions, ingest caps, and 4 honest
  `limitations` all present.

### 3.2 Found: production `chroma.sqlite3` is byte-churned by *any* production-path read (outside eval windows)

The baseline's guards prove the store was byte-identical across its own
run window (before == after
`e3edcfc9…a0372`). However the file's **current** bytes no longer match
that record, and by timestamp+experiment the cause is isolated:

| When | Event | Recorded sha (first 16) |
|---|---|---|
| 2026-10-16 18:47:30→18:47:46 | baseline eval run — guard before==after ✓ | `e3edcfc9dd12408d` |
| 2026-10-16 18:48:31 | ~45 s after baseline, one write (RAG-1 verification ran `check_chroma` then) | → `555d8acfa3e0917` |
| 2026-10-17 09:09:36 | audit ran `check_chroma.py` | → `c3d378c481013fe4` |
| 2026-10-17 09:09:44→09:11 | audit eval **rerun** — guard before==after `c3d378…` ✓ | unchanged |
| 2026-10-17 09:13:56 | controlled experiment: before-stat → `scripts/check_chroma.py` (exit 0) → after-stat | `c3d378…` → `cb0e25647c81a9e0` |

**Controlled experiment result:** running the "read-only diagnostic"
`scripts/check_chroma.py` changes `chroma.sqlite3` bytes by itself
(chronadb client open/migration bookkeeping), so *any* production-path
`PersistentClient` read — including `check_chroma`'s own, and the ones in
RAG-1's verification sequence — bumps the byte hash. The eval harness
never does (both guarded windows prove before==after; all its clients were
on temp paths).

**Content integrity remains sound:** after every read,
`check_chroma` reports manifest entries = 2 with 1+1 stored chunks
verified, sample text 100% printable, 0 errors/0 warnings;
`.index_manifest.json` is byte-unchanged since 2026-10-13; the rerun
reproduced the baseline exactly (§4). No vector, manifest entry, or
metadata row was found altered.

**Disclosure:** the audit itself caused two of the byte-churn events
(today's two `check_chroma` runs, one of them the deliberate attribution
experiment), each verified to leave logical content intact. No write,
migration, reindex, or deletion was performed; the harness, gold,
manifest, and baseline files were not modified.

**Consequence for guard design (proposed only):** the byte-level store
guard is sound *within* a run but cannot distinguish "data mutated" from
"client opened" across runs — RAG-2 may want a logical-content probe
(counts + per-file `file_path` attribution + sample hash) alongside the
byte hash. Not implemented here.

---

## 4. Reproducibility

Performed with output redirected to
`%TEMP%\rag_rerun_audit.json` — `--out` is a harness-supported flag, so
**no harness change and no baseline overwrite**; the baseline file's
mtime/sha are untouched (`18:47:46`, `2c1d4ae9…`).

```
env\Scripts\python.exe scripts\eval_retrieval.py --out %TEMP%\rag_rerun_audit.json
→ exit 0, total 31.4s, collection books_eval_20261017-091012 (temp, deleted)
```

**Diff vs the existing baseline (field-by-field):**

- Every metric in both sections: **identical**.
- Every per-case metric object, every order/ranking, and **every score
  value** (to full recorded precision): **identical**.
- Abstention proxy (both ratio strings + per-case values): identical.
- `git_head`, gold SHA, all corpus/fixture hashes, embedding-model
  digest: identical.
- Guards in the rerun: `real_store_unchanged=true`,
  `gold_unchanged=true`, `all_chroma_clients_on_temp_paths=true`,
  `production_query_engine_cache_used=false`, fidelity check `true`,
  chunk count 8.

**Expected (and observed) variation is latency only:**

| Measurement | Baseline | Rerun |
|---|---|---|
| retrieval p50/p90/max (ms) | 69.3 / 85.3 / 94.6 | 79.9 / 96.0 / 106.3 |
| embed per-query p50/p90 (ms) | 64.6 / 75.0 | 74.5 / 93.6 |
| ingest (s) | 11.16 | 8.15 |

Retrieval scores needed no tolerance at all: embeddings are deterministic
(model digest identical), and even the rebuilt temp HNSW index produced
byte-identical rankings and scores. Latency differences are ordinary
machine-load variation (+~15% retrieval, −27% ingest) and are recorded in
the reports as machine-dependent measurements, as they should be.

---

## 5. Files read/changed and commands executed

**Files read (no modifications):** `scripts/eval_retrieval.py`,
`benchmarks/rag_baseline_v1.json`, `benchmarks/rag_gold_v1.json`,
`scripts/test_rag_eval_harness.py`, `scripts/check_chroma.py`,
`app/config.py` (lines 116–117), `app/agents/books_agent.py` (lines
640–660, 795–825), `frontend/vite.config.ts`, `docs/TESTING.md`,
`docs/RAG_IMPLEMENTATION_PLAN.md`, installed llama-index Chroma source
(score semantics), storage-dir metadata (stat/hash only).

**Files changed:** created `docs/RAG_BASELINE_AUDIT.md` (this document)
— only repo change. Scratch outside the repo:
`%TEMP%\audit_recompute.py`, `%TEMP%\rag_rerun_audit.json`.
Side effect disclosed: two `check_chroma` runs byte-churned
`storage/chroma.sqlite3` (content-verified intact; §3.2).

**Commands actually run (audit):**

| Command | Exit | Result |
|---|---|---|
| `%TEMP%\audit_recompute.py <root>` (independent formulas) | 0 | 0 per-case mismatches; all 12 aggregates match; proxy 0/4, 0/3 match |
| `pytest scripts/test_rag_eval_harness.py -q` | 0 | 20 passed, 22.31 s |
| `scripts/eval_retrieval.py --out %TEMP%\rag_rerun_audit.json` | 0 | metrics identical to baseline; guards GREEN |
| `scripts/check_chroma.py` (×2: verification + attribution experiment) | 0 | 0 errors, 0 warnings; manifest consistent; caused the documented byte churn |
| hash/diff probes (gold, corpus, guards, rerun-diff, store re-stat) | 0 | see §§1–4 |
| `git status --porcelain` / `git rev-parse` | 0 | see below |
| combined A–E probe (timed out at 120 s on the rg step) | timeout | re-run split into parts, all exit 0 |

**Git / frontend config re-check (audit-time snapshot; superseded for the
current status by §7.4):**

```
 M docs/TESTING.md
?? benchmarks/rag_baseline_v1.json
?? benchmarks/rag_fixtures/
?? benchmarks/rag_gold_v1.json
?? docs/RAG_BASELINE_AUDIT.md
?? docs/RAG_IMPLEMENTATION_PLAN.md
?? scripts/eval_retrieval.py
?? scripts/make_rag_fixtures.py
?? scripts/test_rag_eval_harness.py
branch main | HEAD 7a44a9a7f747a983d2ffeb2df767f7f4f265b5f2 (recorded in baseline = current)
frontend/vite.config.ts:10 port: 3333 | :11 strictPort: true | :14 target: 'http://127.0.0.1:8000'
```

Identical to the pre-audit status plus this audit document itself:
nothing staged, committed, or pushed; gold set untouched (SHA match
`16e25677…b25d`) — status otherwise matches RAG-1's delivery record.

---

## 6. Verdict + minimal proposed fixes (approval required — none implemented)

**Verdict: the RAG-1 baseline IS trustworthy enough to serve as the
comparison point for RAG-2** — with three caveats that must travel with
it:

1. **Numbers are correct and exactly reproducible.** Independent
   recalculation reproduced every per-case value and every aggregate
   (12/12), and a full rerun reproduced all metrics and all scores
   byte-identically under identical guards. The suspected MRR/hit-rate
   inconsistency is a *definitional* split (node-level RR/nDCG vs
   distinct-doc `min_docs` hit), not an arithmetic error — but it must be
   documented so RAG-2 comparisons don't misread it.
2. **Coverage-family metrics on the two multi-doc cases (R011, S016) are
   properties of the relevance rule, not the retriever:** `gold_pages`
   gates page-less `.docx` gold documents, making their `hit@3`/
   `coverage` structurally false. Hit@1 (0.9091 / 0.8571), MRR, and
   nDCG@5 are unaffected. Minimal fix — **APPROVED AND IMPLEMENTED
   2026-10-17** (original proposal retained): in `node_covers`, skip the page filter for nodes without page
   metadata (effect verified offline: flips exactly R011 and S016
   `hit@3`→true and coverage→true → real hit@3/coverage 1.0000,
   synthetic hit@3/coverage 1.0000, everything else unchanged), then
   re-run the baseline under the normal regeneration procedure
   (performed: v2 `books_eval_20261017-092323`, v3
   `books_eval_20261017-095829` — deltas in §7.2).
   Secondary proposal: record `retrieved_top5` so hit@5/coverage are
   auditable from the artifact — **implemented in v3**
   (`ordered-top5-per-case`; see §7.2).
3. **The baseline proves retrieval-ranking behavior only.** The
   abstention proxy is a provably-unreachable score floor (all scores
   ≥ 0.65 vs floor 0.35): it establishes nothing about answer-level
   refusal — the prompt/grounding refusal path remains unmeasured until
   answer-level evaluation is separately authorized. Real-corpus metrics
   remain non-discriminating (2 single-chunk documents) — recorded in the
   report's own `limitations`.

Store safety: within every eval window the store is proven unchanged;
*between* runs, byte hashes drift whenever any tool opens the production
store (proven for `check_chroma`), while logical content verifies intact.
RAG-2's regression gate should compare logical content, not raw bytes, or
accept the documented churn.

---

## 7. Addendum (2026-10-17, RAG-1 closeout) — v2/v3: revised relevance rule, affected cases, current frozen artifact

This addendum supersedes the *status* statements in §1.5, §1.6, §5 and
§6.2; the v1 findings in §§1–6 remain historically correct **for the v1
artifact** (`2c1d4ae9…`) and are not rewritten. Nothing here changes the
§2 conclusion (the score floor is unreachable) or the §3 store-safety
record.

### 7.1 Artifact lineage (all four hashes re-verified at closeout)

| Artifact | `run_id` | sha256 (first 16) | `artifact_schema` | `metric_definitions.version` | top-5 evidence |
|---|---|---|---|---|---|
| `rag_baseline_v1.json` | `books_eval_20261016-184730` | `2c1d4ae9de438a25…` | — | *(absent — v1 relevance rule)* | no (top-3 only) |
| `rag_baseline_v2.json` | `books_eval_20261017-092323` | `cfe43ee2f803eba0…` | — | `v2-page-metadata-aware` | no (top-3 only) |
| `rag_baseline_v3.json` | `books_eval_20261017-095829` | `aa89c37bc23b0dc7…` | `3 / ordered-top5-per-case` | `v2-page-metadata-aware` | **yes** (`retrieved_top5` + `gold_ref`) |
| `rag_gold_v1.json` | — | `16e25677f2effbd6…` | — | — | — |

Gold and all three baselines are SHA-256-pinned by
`scripts/test_rag_eval_harness.py` and must never be rewritten. As part
of the closeout the evaluator gained an output-safety gate: an existing
`--out` file is refused with **exit 3** before any guard/gold/Ollama
work, and `--force` is the only opt-in to overwrite — the previously
documented bare command could silently clobber the pinned v1 artifact
(this §7.4 records the fixed state).

### 7.2 What changed between v1 and v2 — the page-metadata-aware rule

`node_covers` now **skips the page filter for nodes that carry no page
metadata** (page-labelled nodes remain gated), making page-less `.docx`
gold members reachable in multi-doc cases. This is exactly the §6.2
minimal fix, approved and implemented on 2026-10-17. Aggregation,
abstention proxy, guards and the case set are otherwise unchanged.

**Affected cases: exactly R011 and S016** (full per-case diff, v1 vs v3):

| Metric | v1 (old rule — historical) | v2/v3 (current rule) |
|---|---|---|
| real hit@3 | 0.9091 | **1.0000** |
| real coverage | 0.9091 | **1.0000** |
| real precision@3 | 0.3333 | **0.3636** |
| synthetic hit@3 | 0.9286 | **1.0000** |
| synthetic coverage | 0.9286 | **1.0000** |
| synthetic precision@3 | 0.3333 | **0.3571** |
| hit@1, precision@1, MRR, nDCG@5, hit@5 (both sections) | — | **identical to v1** |

v2 and v3 have **identical per-case metrics** (verified); their only
difference is evidence. v3 additionally stores `retrieved_top5`, which
closes the §1.6 transparency gap — `hit@5`/`coverage` are now recomputable
from the artifact alone, enforced by the harness suite's independent
recalculation test. v2 remains unauditable for ranks 4–5; v3 is the
reference.

### 7.3 Comparability rule for later phases

- **Stable across v1/v2/v3** — safe for cross-version comparison: hit@1
  (0.9091 real / 0.8571 synthetic), precision@1 (1.0 / 0.9286), MRR
  (1.0 / 0.9643), nDCG@5 (1.0 / 0.9736), hit@5 (1.0 / 1.0).
- **Rule-dependent** — compare v3 ↔ future runs only: hit@3, coverage,
  precision@3. v1's values are historical, not arithmetic errors (they
  were correct under the rule then in force, §1.5).
- The §1.4 MRR-vs-hit divergence is definitional and unaffected: R011/S016
  still have `hit@1 = false` by construction (min_docs = 2 in one slot).
- `unanswerable_cases` (4 real / 3 synthetic) and the §2 proxy ratios
  (0/4, 0/3, 0/11, 0/14) are unchanged in v2/v3 — still score-floor-only
  evidence, still **not** answer-level refusal.

### 7.4 State verified at closeout (2026-10-17)

- HEAD `7a44a9a7f747a983d2ffeb2df767f7f4f265b5f2`, nothing staged or
  committed. Pending work = the RAG-1 deliverables (gold, fixtures,
  baselines v1/v2/v3, harness + tests, this audit, the implementation
  plan) + this closeout's documentation updates (`docs/TESTING.md`,
  `docs/MASTER_PLAN.md`, `docs/PROJECT_STATUS.md`,
  `docs/CURRENT_STATE_AND_HANDOFF.md`, `docs/RAG_1_COMPLETION_REPORT.md`)
  + the evaluator's output-safety gate. No unrelated pending changes.
- Harness suite: **38 passed, 11 subtests, exit 0** — includes the v3
  pin, the evidence recalculation, and the new overwrite-safety tests.
- Frozen-baseline gate `benchmark_verification.py --check`:
  `baseline MATCH` (exit 0).
- v3 guards re-checked from the artifact: real store unchanged, gold
  unchanged, temp-only Chroma clients, production engine cache unused,
  fidelity true; `git_dirty: true` inside v3 remains deliberate (the
  evaluated code was uncommitted at run time).
- Production store byte-stable through the closeout (`cb0e25647c81a9e0…`,
  size/mtime unchanged); `check_chroma.py` **not** run; no model, API or
  network calls made.
- Still unmeasured by design: answer-level correctness and refusal →
  **RAG-AE (answer-quality evaluation, not started)**, which is tracked
  separately from the RAG-1…RAG-6 implementation sequence in both
  roadmap docs (naming reconciled at this closeout).