# RAG Evaluation — What Is Measured, What Is Missing (Finalization Audit, 2026-10-18)

Status: honest inventory. **No answer-quality evaluation (RAG-AE) has been
implemented or run.** Nothing below claims otherwise.

## 1. Measured today (with evidence)

| Capability | Status | Evidence |
|---|---|---|
| Retrieval quality on real+synthetic corpus (hit@k, precision@k, MRR, nDCG@5, abstention-proxy) | **Measured**, frozen | `benchmarks/rag_baseline_v3.json` (sha256 `aa89c37b…`), recorded in `docs/TESTING.md` §C |
| Retrieval harness correctness (guards, metrics semantics, isolation, overwrite safety) | **Measured** | `scripts/test_rag_eval_harness.py` (38 passed) |
| Deterministic grounding/polarity/numeric/citation gates | **Measured** (unit level) | grounding/paraphrase/adversarial/citation suites (77 + subtests) |
| Verification-benchmark consumption (scripted verdicts) | **Measured** (pipeline behavior, NOT model accuracy) | `scripts/benchmark_verification.py --check` 38/38 |
| Bounded self-refusal recovery (eligibility, single retry, gates) | **Measured** (unit level) | `scripts/test_self_refusal_recovery.py` (18 + subtests) |
| Latency of retrieval/embedding on this machine | Recorded (p50/p90/max in v3 baseline) | `docs/TESTING.md` §C |
| Answer-level latency in live runs | Anecdotal only (execution records) | execution store; not a benchmark |

## 2. NOT implemented / NOT measured

| Capability | Gap |
|---|---|
| **Grounded answer correctness (RAG-AE)** | No answer-quality harness exists. Retrieval metrics never evaluate generated text. |
| **Refusal quality: false-refusal rate** | Known live defect class (delta-v question); only unit-level probes exist (candidate C001/C002). |
| **Correct-abstention rate on unanswerable questions** | The v3 "abstention proxy" only proves the 0.35 score floor is unreachable — it does NOT measure answer refusal (documented in v3 caveats). |
| **Citation correctness at answer level** | Gate-level tests exist; no corpus-level citation audit. |
| **Threshold calibration (BOOKS_MIN_SCORE)** | Untested beyond "floor unreachable on this corpus" (all top-1 in [0.65, 0.83]). |
| **Complex-document cases** | No real tables/scans/multi-page docs in corpus (gold v1 gap_list). |
| **Conflicting-evidence behavior end-to-end** | Unit-tested only with synthetic in-test passages; no corpus fixture. |
| **Partially-answered-question behavior** | Unit-tested; no corpus-level measurement. |
| **Multi-chunk questions** | Every current document fits in ONE 1024-token chunk — chunk-boundary behavior is unmeasurable on this corpus. |
| **Small-corpus top-k saturation** | `BOOKS_TOP_K=3` > corpus size 2: every query returns ALL chunks; retrieval cannot discriminate at k=3. Measured fact (Phase 3/4 probes), not yet acted on. |

## 3. Proposed additive candidate set (not a replacement)

`benchmarks/rag_gold_v2_candidate.json` (draft, requires human review before
first execution):

- C001/C002 — the live false-refusal delta-v question + a filename-free paraphrase (paraphrased-answerable, false-refusal-probe).
- C003/C004 — numerical variants (comma-formatted figure; units-explicit).
- C005 — partially answerable (covered fact + absent fact in one question; expects disclosure, not whole-topic refusal).
- C006 — unanswerable domain-adjacent (correct abstention, no footer).
- C007 — multi-doc paraphrase (per-claim attribution must not swap).
- C101-C103 — **blocked-on-fixture** cases (conflicting pair, multi-page table, scan-like): explicitly not executable until fixtures exist.

Frozen v1/v2/v3 baselines and `rag_gold_v1.json` remain untouched; the
candidate file is validated offline by
`scripts/test_rag_gold_v2_candidate.py` (schema/coverage/no-execution-claims).

## 4. Metrics to implement next (ordered by leverage)

1. **False-refusal rate on answerable paraphrases** (C001-C004): needs an answer-level harness driving `books_agent` with a scripted-free local model and scoring: answered-with-citation vs abstained. Blocked on live-model authorization, not on code.
2. **Correct-abstention rate** (C006 + v1 unanswerables): same harness, inverted expectation.
3. **Citation correctness**: footer document must be in gold_documents; forbidden-document set for unanswerables.
4. **Partially-answered disclosure correctness** (C005): covered fact present AND explicit not-covered disclosure present AND no fabricated half.
5. **Latency per answer** (wall time, retry-count as a dimension — recovery attempts must be ≤1).

All five run against a *disposable* store copy per the guard contract; none
may touch `storage/`.
