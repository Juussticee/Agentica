# Retrieval reranking, claim-level verification and evidence provenance (2026-11-01)

The previous milestone (`docs/EVIDENCE_GROUNDED_ANSWERING.md`) made the
`general` route consult the local corpus and cite what it used. Its known gap
was precise: **a citation is not proof**. The grounding gate worked over the
whole retrieved set, so a sentence could be "supported" by passage A while
citing passage B; similarity ranking put topical material above answering
material; duplicates filled the context; and the passage text was not stored,
so a past answer could not be audited. This milestone fixes those three things
with deterministic code, one new module per concern, and no new dependency.

## 1. Existing architecture (what was reused, not rebuilt)

```
question
  └─ app.evidence.retrieve()          -> app.agents.books_agent._get_retriever
       └─ app.agents.books_retrieval.build_retriever -> _open_index()
            (the SAME persisted Chroma index, embedding model and chroma_guard
             the books route has always used; no second store, no re-ingest)
  └─ app.evidence.select()   [NEW]    -> rerank, dedupe, diversify
  └─ app.evidence.assess()            -> grounded | partial | none
  └─ app.graph.general_node
       ├─ evidence.compose_prompt -> model
       ├─ evidence.postprocess -> books_grounding._grounded_repair (unchanged)
       └─ app.claim_check.verify() [NEW] -> per-claim verdicts vs the CITED passage
  └─ app.qa_node                     -> figures, licences, citations, claim findings
  └─ app.execution.runner            -> app.evidence_store.write() [NEW] (sidecar)
  └─ GET /api/v1/executions/{id}/evidence [NEW]
```

Nothing was re-implemented: the tokenizer, paraphrase families, polarity
detection, clause splitter and citation validators all come from
`app.agents.books_grounding`, so the evidence route and the books route agree
about what "covered" means.

## 2. Retrieval improvements (`app/evidence.py::select`)

Similarity alone is not a ranking. The reranker is a deterministic score:

```
rerank = similarity
       + 0.40 × share of the question's content words the passage covers
       + 0.20 × share of the question's technical words it states literally
```

then:

* **near-duplicates are dropped** (token Jaccard ≥ 0.9 against an already-kept
  passage — one chunk retrieved twice no longer fills the context);
* **selection is diverse**: the next passage is chosen by rerank plus a small
  bonus for question words no better-ranked passage already covers
  (`_DIVERSITY_WEIGHT = 0.05`, capped at 3 words), so a two-part question can
  be answered from two documents;
* the decision is recorded as a `ranking` audit trail (similarity, rerank,
  coverage, exact-term share, original position) per kept passage.

No model was introduced for reranking, and no dependency: a cross-encoder
would add latency and a failure mode that could not be measured inside this
system, so it was not adopted (see §8).

## 3. Claim-level verification (`app/claim_check.py`)

Each substantive sentence is checked against the passage it **cites** (inline
`file, page`), not against the union of everything retrieved.

| verdict | meaning |
| --- | --- |
| `supported` | every content token (numbers exact) is in the cited passage and nothing there contradicts it |
| `partial` | the cited passage covers ≥ 60% of the claim; the missing words are named |
| `contradicted` | the cited passage states the opposite polarity, or a different licence for the same subject |
| `unsupported` | the cited passage does not state it — including a cited file/page that was never retrieved (fabricated attribution) |
| `unable` | no passages were consulted, so nothing can be checked |

Rules that keep the verdicts honest:

* **Numbers must match exactly.** A version, port, percentage or capacity the
  passage does not state makes the claim unsupported even when every other
  word overlaps (`C5`, `C7`, `C11` in the benchmark).
* **A contradiction needs the same subject.** Two documents that merely mention
  different licences is a *disclosed conflict*, not proof the answer is wrong:
  a claim corroborated by one passage is `supported`, and only a claim that no
  passage supports, where one states a different licence for the same subject,
  is `contradicted` (regression E5 — the first implementation got this wrong and
  the answer benchmark caught it).
* **No model judges its own answer.** Every verdict is deterministic. The
  optional `BOOKS_VERIFIER` model step stays where it was (books route only,
  monotone, defaults to `off`) and is deliberately not consulted here.
* Disclosures, the generated References line and the "Answered from my own
  knowledge" label are not claims and are skipped, never counted as failures.

QA (`app/qa_node.py::_claim_verification_findings`) reports only the
non-clean verdicts — contradicted, unsupported, partly covered — naming the
sentence and its reason, and at most two examples per class. A single weak
claim still never fails the execution.

## 4. Evidence provenance (`app/evidence_store.py`)

One immutable JSON sidecar per execution:

```
storage/evidence_snapshots/<execution_id>.json
```

Written by `GraphRunner.run` after the terminal state is persisted
(`tempfile` + `os.replace`, best-effort, wrapped so it can never fail a run).
It records:

* the question, route, answer, sources, QA flags;
* **the index identity as used** — collection plus a digest of the ingest
  manifest — so a later re-ingest is visible rather than silently reinterpreted;
* the reranking ranking, dropped-duplicate count and coverage verdict;
* **the passage text as retrieved**, with file name, page, chunk id, retrieval
  and rerank scores, and the document's own creation/modification stamp;
* the claim-to-passage verdicts and their reasons.

Both answer paths use the same writer (2026-11-02): the `general` route records
its record in `state["evidence"]`, the `books` route in
`state["books_evidence"]` (a separate channel so the general route's QA claim
checks keep exactly one answer path). `snapshot_evidence()` picks whichever
holds passages, and the snapshot's `route` field says which route produced it.

**Completeness is explicit.** `passage_details` records EVERY retrieved
passage, one row each, and `EVIDENCE_SNAPSHOT_CHARS` is a **per-passage** cap.
The earlier implementation used one cumulative budget for the prompt *and* for
persistence, so the fourth of four ~2 KB passages was silently skipped once the
first three had consumed it — a snapshot that implied a passage was never
consulted. A passage longer than the cap is now stored cut and flagged
`truncated`. The snapshot reports:

| field | meaning |
| --- | --- |
| `passages_retrieved` | every passage the retriever returned for this run |
| `passages_supplied` | how many were actually placed in the model's prompt (`supplied` per row; prompt budget `EVIDENCE_PASSAGE_CHARS` still bounds the prompt) |
| `storage_complete` | a row exists for every retrieved passage (a missing record is storage loss) |
| `content_complete` | every stored passage's text is whole — `truncated`/empty rows make this false |
| `complete` | `storage_complete AND content_complete`, so it can never imply full text when the text was cut; naming each affected rank |
| `partial_reason` | one human-readable line per affected rank, or `null` |

Never stored: `file_path`, absolute paths, credentials. A snapshot is what the
run consulted, not a claim that the document's current contents match it — the
manifest digest makes the difference checkable.

Exposed read-only as `GET /api/v1/executions/{id}/evidence` (200 with
`evidence: null` and a reason when a route consulted no passages — never
fabricated). **No schema change or migration**: the executions/events tables
are untouched.

## 5. Benchmark methodology

| file | suite |
| --- | --- |
| `benchmarks/evidence_general_v1.json` | 15 end-to-end answer/routing/follow-up cases (unchanged, still 12/12) |
| `benchmarks/evidence_claims_v1.json` | 6 retrieval + 11 claim cases (adversarial) |

`app/evidence_claims_eval.py` + `scripts/eval_claims.py` run the new suite
offline; `app/evidence_eval.py` + `scripts/eval_evidence.py` remain the answer
suite. Fixtures are authoritative: licence values come from the projects'
published terms, expected verdicts are hand-classified, and **no model output is
ever used as an expected answer**. The answer suite's provider double repeats a
prepared reply, so it measures the pipeline, not the model.

## 6. Before / after (measured, reproducible)

```bat
env\Scripts\python.exe scripts\eval_claims.py   --fail-on-regression
env\Scripts\python.exe scripts\eval_evidence.py --fail-on-regression
    :: storage\evidence_eval\evidence_claims_report.{json,md}
    :: storage\evidence_eval\evidence_general_report.{json,md}
```

**Retrieval** (6 cases, similarity order vs reranked):

| metric | similarity order | reranked |
| --- | --- | --- |
| hit@1 | 0.333 | **1.000** |
| MRR | 0.625 | **1.000** |
| precision@K | 0.57 | 0.57 (K=4 returns every fixture passage; unchanged) |
| recall@K | 1.0 | 1.0 (same reason) |
| duplicates dropped | — | 1 |
| mean selection latency | — | **0.08 ms** |

Improved cases R1 (answering passage ranked below a topical distractor), R2
(answering passage buried last), R3 (Redis material competing with Memcached),
R5 (two-document question); **no case regressed**. Precision/recall are
unchanged because these fixtures have fewer passages than K, which is reported
rather than presented as a win.

**Claims** (11 cases): verdict accuracy **11/11**, claim-support precision
**1.0**, recall **1.0**, contradiction detection **1.0**, unsupported detection
**1.0**, false-verification rate **0.0**, mean verification latency **0.10 ms**.

**End-to-end answers** (unchanged dataset): cases passed 7/12 baseline → **12/12**
grounded, citation accuracy **1.000**, no regressions — including the two cases
the reranker changed (E5 conflicting documents, E6 swapped licence).

## 7. Regression results

* Full backend suite: **1341 passed, 2 skipped, 396 subtests** (145 s).
  `scripts/test_claim_verification.py` contributes 32 tests and
  `scripts/test_evidence_snapshot.py` 14 more (four-passage completeness, the
  per-passage cap, explicit `partial` status, `retrieved` vs `supplied`, books
  snapshot creation/refusal/no-passages, a real subprocess restart, and the
  runner's record selection). Three `test_citation_integrity.py` shape
  assertions were updated to the new node contract (the extra
  `books_evidence` reporting key) — no assertion was weakened.
* Preserved and exercised by those tests: books RAG grounding and citations,
  local-only restrictions, explicit research authorization, numeric grounding,
  licence checks, instruction-echo stripping, learning/feedback, follow-up
  routing, execution lifecycle, Control Center endpoints, existing evidence
  labels.
* Frontend untouched; `tsc -b` remained clean in the previous milestone and no
  frontend file changed here.
* `git diff --check` clean.

**Live verification (real Ollama, real index, real API)**

| check | result |
| --- | --- |
| `POST /api/v1/executions` "What is the GOVERN function?" | `a49f58678a01`, route `general`, completed, citation `NIST.AI-100-1.pdf, page 22`, claim `supported` |
| `GET /api/v1/executions/a49f58678a01/evidence` | `passages_retrieved: 4`, `passages_supplied: 2`, `complete: true`; 4 rows with file/page/chunk id/score/`supplied`; the 4th passage is now recorded (it was dropped by the old shared budget) |
| `POST /api/v1/executions` "According to my documents, what does the GOVERN function say about risk culture?" | `18bff98789c9`, route `books`, refused after consulting 3 passages (0.6B self-refusal), research proposal gated |
| `GET /api/v1/executions/18bff98789c9/evidence` | books snapshot present: status `none`, reason `refused: self_refusal_after_recovery`, `passages_retrieved: 3` (pages 22/23/7, scores 0.784/0.736/0.722), `complete: true` |
| path leak check on both responses | no `file_path`, no `C:`, no `Users` |
| restart the backend, re-read both snapshots | present and identical after restart |

## 8. Known limitations

1. **Token coverage is not entailment.** `supported` means "the cited passage
   states these words, with this polarity and these numbers". It does not
   establish that the claim is true in the world, and the UI/QA wording never
   says it does.
2. **Reranking is lexical + similarity.** A passage that answers the question in
   different words (no shared content tokens and no paraphrase family) can
   still rank low. No cross-encoder was adopted because its benefit, latency and
   failure behaviour were not measured here — that is the honest reason, not a
   claim that it would not help.
3. **Claim checks are per sentence.** A claim split across two sentences, or a
   conclusion drawn from two passages, is judged sentence by sentence; a
   cross-sentence inference can pass as `partial` rather than being caught.
4. **Research and planner routes write no snapshot.** They cite URLs/agent
   answers rather than local passages, so their provenance stays in `sources`,
   events and the answer itself.
5. **Snapshot growth is unbounded.** One JSON per answering execution, no
   retention policy yet; each passage is capped at `EVIDENCE_SNAPSHOT_CHARS`
   (8000) and a cut passage is flagged, never dropped.
6. **The prompt still sees fewer passages than the retriever returns.**
   `EVIDENCE_PASSAGE_CHARS` (6000) bounds the evidence prompt; the live general
   run supplied 2 of 4 passages while recording all 4. That is now visible per
   row (`supplied`), not silent, but the model genuinely did not read two of
   them.
7. **Fixtures are small and authored.** 11 claim cases and 6 retrieval cases
   prove the classification logic, not field accuracy; the model double in the
   answer suite is not a model.
8. **Not verified live:** a corpus that disagrees across documents, a model
   strong enough to answer reliably from passages (only `qwen3:0.6b` installed),
   and Arabic/RTL text (no RTL layer exists in the codebase).

## 9. Remaining risks

* A wrong-but-florid answer whose words still appear in a cited passage is
  indistinguishable from a correct one for this checker; the residual risk is
  **citation-faithful hallucination**.
* QA findings are advisory: an execution with a contradicted claim stays
  `completed` (by design — QA has never gated status), so a user who ignores the
  flag sees a confident answer.
* The manifest digest is a change *signal*, not a content hash of each chunk; a
  document edited without touching its mtime would not be distinguished.
* Snapshots contain passage text, so they inherit whatever sensitivity the
  corpus already has; they are local files, not exposed beyond the loopback API.

## 10. Next highest-value work

Add a retention/size policy (age + total-byte ceiling, keep-first-N per route)
so the audit trail stays usable as it grows, and expose the snapshot in the
Control Center instead of only the loopback API. A budget that lets the prompt
carry every retrieved passage — or a documented passage-selection rule — is the
remaining half of "the snapshot matches what the model read".
