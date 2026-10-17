# RAG-1 Completion Report — Baseline & Evaluation

**Track:** RAG-1 (`docs/MASTER_PLAN.md` §5.1 / `docs/RAG_IMPLEMENTATION_PLAN.md` §14)
**Closeout date:** 2026-10-17 · **HEAD:** `7a44a9a7f747a983d2ffeb2df767f7f4f265b5f2`
**Verdict:** RAG-1 is **complete and review-ready**. Everything below is
**uncommitted** — no commit has been authorized.

This is the phase completion report required by
`docs/RAG_IMPLEMENTATION_PLAN.md` §14.1 rule (5): changes, metrics delta,
deviations, new risks.

---

## 1. Changes (complete file inventory)

### Created (all RAG-1 deliverables)

| File | Role |
|---|---|
| `benchmarks/rag_gold_v1.json` | Gold set: 32 cases (25 answerable / 7 unanswerable; 17 synthetic), `required_facts`, `citation_expectation`, per-case provenance, 9-item `gap_list`, `immutability_rule` |
| `benchmarks/rag_fixtures/` (5 files) | Deterministic synthetic corpus (columns, table, scan, headings, adversarial) built by the fixture generator |
| `scripts/make_rag_fixtures.py` | Fixture generator — byte-identical rebuild asserted in tests |
| `scripts/eval_retrieval.py` | Evaluation harness: temp eval collection, instrumented Chroma clients, store/gold before-after guards, Ollama identity capture, `--out`/`--force`, exit codes 0/1/2/**3** |
| `scripts/test_rag_eval_harness.py` | Offline suite: **38 tests, 11 subtests** (metrics math, guards, gold immutability, fixture determinism, isolation proof, artifact-schema-3 evidence + independent recalculation, **overwrite safety**, SHA-256 pins for gold + v1/v2/v3) |
| `benchmarks/rag_baseline_v1.json` | First baseline (historical, `2c1d4ae9…`) |
| `benchmarks/rag_baseline_v2.json` | Post-relevance-rule baseline (`cfe43ee2…`) |
| `benchmarks/rag_baseline_v3.json` | **Frozen reference** (`aa89c37b…`, `artifact_schema.version=3`) |
| `docs/RAG_BASELINE_AUDIT.md` | Independent audit of v1 (§§1–6, preserved) + dated v2/v3 addendum (§7) |
| `docs/RAG_IMPLEMENTATION_PLAN.md` | Implementation plan (phase table, mapping, RAG-AE reconciliation) |
| `docs/RAG_1_COMPLETION_REPORT.md` | This report |

### Modified

| File | Change |
|---|---|
| `docs/TESTING.md` | Harness row corrected (20 → **38** tests, overwrite-safety behavior, frozen v3 identity) + new subsection "RAG-1 frozen baseline (v3) — recorded metrics" with denominators and caveats |
| `docs/MASTER_PLAN.md` | §5.1 naming-reconciliation note + new **RAG-AE** entry (answer-quality evaluation as a separate track) |
| `docs/PROJECT_STATUS.md` | RAG-1 completion state, limitation, outstanding work |
| `docs/CURRENT_STATE_AND_HANDOFF.md` | RAG-1 closeout section (state, limitations, outstanding work) |
| `scripts/eval_retrieval.py` | Output-safety gate: refuses to overwrite an existing `--out` report with **exit 3** before any guard/gold/Ollama work; `--force` is the only opt-in |
| `scripts/test_rag_eval_harness.py` | +4 overwrite-safety tests; v3 added to the SHA-256 pins |

No file under `app/`, `ingest_books.py`, `requirements.txt`, prompts,
thresholds, or the gold set was touched. No retrieval or answer logic
changed anywhere.

## 2. Verification evidence (2026-10-17 closeout)

| Check | Result |
|---|---|
| `pytest scripts/test_rag_eval_harness.py -q` | **38 passed, 11 subtests, exit 0** (29.4 s) |
| `pytest scripts/test_books_grounding.py scripts/test_citation_integrity.py -q` | **46 passed, 8 subtests, exit 0** (answer-level behavior unchanged) |
| `python scripts/benchmark_verification.py --check` | `all cases: baseline MATCH, integrated target MATCH`, **exit 0** |
| Bare `python scripts/eval_retrieval.py` (CLI) | **exit 3**, refusal printed, v1 bytes unchanged, store untouched |
| Pinned hashes before/after | gold `16e25677…`, v1 `2c1d4ae9…`, v2 `cfe43ee2…`, v3 `aa89c37b…` — all identical |
| Production store | `chroma.sqlite3` `cb0e25647c81a9e0…`, 245760 bytes, mtime unchanged across the whole closeout |
| Git | HEAD unchanged; nothing staged/committed/pushed; staged diff empty |
| v3 internal guards (from the artifact) | real store unchanged · gold unchanged · temp-only Chroma clients · production query-engine cache unused · fidelity true · `git_dirty: true` (deliberate: evaluated code was uncommitted) |

**Not verified this session (reported, not claimed):** the full 383-test
regression and the ingest suite were not re-run — last recorded green
2026-10-16 (`docs/RAG_IMPLEMENTATION_PLAN.md` §13.5); no suite that could
touch the production store was executed, and `scripts/check_chroma.py` was
never run.

## 3. Metrics delta

v3 = frozen reference (denominators and caveats in
`docs/TESTING.md` § "RAG-1 frozen baseline (v3)"):

| Section | n | hit@1 | hit@3 | precision@1 | precision@3 | coverage | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|---|
| real corpus | 11 answerable | 0.9091 (10/11) | 1.0 (11/11) | 1.0 | 0.3636 | 1.0 | 1.0 | 1.0 |
| synthetic fixtures | 14 answerable | 0.8571 (12/14) | 1.0 (14/14) | 0.9286 (13/14) | 0.3571 | 1.0 | 0.9643 | 0.9736 |

**v1 → v3 delta:** exactly two cases changed (R011, S016) when the
relevance rule became page-metadata-aware — real hit@3/coverage 0.9091 →
1.0, precision@3 0.3333 → 0.3636; synthetic hit@3/coverage 0.9286 → 1.0,
precision@3 0.3333 → 0.3571. Every other metric value is identical across
v1/v2/v3; v2 and v3 metrics are byte-identical to each other. Full lineage
and the comparability rule: `docs/RAG_BASELINE_AUDIT.md` §7.

**Abstention proxy (unchanged):** 0/4 and 0/3 correct on unanswerable,
0/11 and 0/14 false-abstains — proof only that the 0.35 score floor is
unreachable, **not** an answer-level refusal measurement.

## 4. Deviations from the plan

1. **Delivered file names** differ from `docs/MASTER_PLAN.md` §5.1, which
   predicted `scripts/test_rag1_evaluation.py`; the delivered harness tests
   are `scripts/test_rag_eval_harness.py`, and `scripts/make_rag_fixtures.py`
   was added (fixture generator) beyond the plan's file list.
2. **Relevance rule changed mid-track** (`v2-page-metadata-aware`): the
   audit's §6.2 minimal fix was approved and implemented, so hit@3 /
   coverage / precision@3 are not comparable between v1 and v2/v3 (§7.3 of
   the audit). Recorded, not silent.
3. **`retrieved_top5` evidence** was added only in v3; v2 lacks ranks 4–5
   evidence and is superseded.
4. **MASTER_PLAN RAG-1 scope item "tune/document `BOOKS_MIN_SCORE` from
   measurements" was NOT done** — measurement showed the floor is
   unreachable instead (audit §2.3); threshold calibration remains with the
   plan's RAG-3 "threshold-calibration report … for user approval".
5. **Baseline metric values were not originally recorded in
   `docs/TESTING.md`** (MASTER_PLAN acceptance text); fixed in this
   closeout (recorded with denominators + caveats).
6. **`docs/TESTING.md` said "20 passed"** while the suite had grown;
   corrected to the verified 38.
7. **Naming collision** ("RAG-2" = three things) resolved as **RAG-AE**,
   outside the numbered sequence, with the mapping documented in both
   roadmap docs; architecture references inspected and still accurate.

## 5. New risks and limitations

- **Answer-level quality/refusal remains unmeasured** — the single biggest
  gap; addressed by RAG-AE (not started, needs its own authorization and
  model budget).
- **Real-corpus metrics cannot discriminate** (2 single-chunk documents;
  8-chunk eval collection) — synthetic metrics do not establish real-world
  accuracy.
- **`hit@1` vs MRR divergence is definitional** (distinct-doc `min_docs` vs
  first-covering-node) — must be carried into every later comparison.
- **Coverage-family metrics are rule-dependent** — compare v3 ↔ future only.
- **`git_dirty: true` is recorded inside v3** — the artifact honestly
  reflects that evaluated code was uncommitted; if the code changes before
  commit, the hash stays valid but the reproducibility story requires this
  report.
- **Latency figures are machine-local measurements**, not guarantees.
- **Gold set has no partially-answerable cases** and no real
  table/scan/multi-column examples (its `gap_list`) — any extension must be
  a NEW gold file (gold v1 is pinned immutable).
- **Full regression not re-run this session** (§2 above).

## 6. Safety record

Production Chroma store was never opened: guards were file-level stat +
SHA-256 only; `scripts/check_chroma.py` was never run; the harness builds
only temp collections (proven by `IsolationProofTest`); no model, API or
network call was made in this closeout; no baseline or gold byte was
written; nothing was staged, committed, pushed, reset or discarded.

## 7. Outstanding work

1. **Commit authorization** for the whole RAG-1 change set (10 new paths +
   `docs/TESTING.md`, plus this report and the closeout doc updates).
2. **RAG-AE** — answer-quality evaluation (spec in MASTER_PLAN §5.1).
3. **Threshold calibration** for `BOOKS_MIN_SCORE` (plan RAG-3) — report +
   user approval; no silent change.
4. **Optional gold extension** (`rag_gold_v2.json`) for partial-answer
   cases and the gap-list categories — user approval required.
5. **Re-run the full regression** before or alongside the RAG-1 commit if
   a green floor beyond the suites in §2 is wanted.
