# Evidence-grounded general answering (2026-11-01)

The reliability audit (`docs/FULL_SYSTEM_RELIABILITY_AUDIT.md`) left one class
of defect open: the `general` route answered from model knowledge with nothing
to check it against. The local 0.6B model produced a confident wrong Redis
licence, invented throughput figures and wrong Memcached licensing; QA could
flag an unsourced *number* but nothing else, because there was no evidence in
the run to compare a claim with.

This milestone points the **existing** RAG index and the **existing** grounding
gate at general questions. No second retriever, no second index, no new
database, no new model, no new prompt-only policy: one node
(`app.graph.general_node`), one new module (`app/evidence.py`), one new
read-only seam on the books facade, and claim-level checks added to the
existing QA node.

## 1. What changed

| Layer | File | Change |
| --- | --- | --- |
| Retrieval seam | `app/agents/books_retrieval.py` | `_open_index()` extracted; new `build_retriever(top_k)` returning **source nodes only** (no synthesis, no model call). The store is opened in one place, so `chroma_guard` and the embedding wiring are shared with the books query engine. |
| Facade | `app/agents/books_agent.py` | `_get_retriever()` / `_reset_retrievers()` — cached, patchable exactly like `_get_query_engine`. |
| Evidence | `app/evidence.py` (new) | `retrieve`, `assess`, `passages`, `evidence_text`, `references`, `compose_prompt`, `postprocess`, `licence_values`, `licence_conflict`, `evidence_files`. |
| General route | `app/graph.py` | `general_node` now retrieves, assesses, answers from evidence when the coverage verdict allows it, and otherwise labels the reply as model knowledge. Writes `state["evidence"]`. |
| QA | `app/qa_node.py` | Figures the answered-from passages state are supported; licence claims are classified; citations are validated against the retrieved file set; an evidence-built answer with no provenance, or a memory answer that points at "your documents", is flagged. |
| State | `app/state.py` | `evidence` channel declared. |
| Config | `app/config.py` | `EVIDENCE_TOP_K` (4), `EVIDENCE_MIN_SCORE` (0.45), `EVIDENCE_MIN_COVERAGE` (0.6), `EVIDENCE_PASSAGE_CHARS` (6000) — all env-overridable, all cheap guards rather than tuning knobs. |

## 2. Evidence flow

```
question ──► evidence.retrieve()      read-only top-k passages (fail-soft: [] on any error)
          └► evidence.assess()         deterministic coverage verdict
                 status = none        no relevant evidence
                        | partial     relevant, but some requested topics are missing
                        └ grounded    relevant and every requested content word is covered

none      ──► model knowledge  + "Answered from my own knowledge; not checked
                                 against any source."            (no sources at all)
grounded  ──► evidence.compose_prompt()  ──► model
partial        └► evidence.postprocess() ──► books_grounding._grounded_repair
                        kept sentences only, fabricated citations dropped,
                        References built from the nodes that supported a claim
                        missing topics disclosed, licence conflicts disclosed
              └► if nothing survives the gate: fall back to the labelled
                 memory answer ("no passage in your local document library
                 covers this")
```

`assess()` deliberately does **not** treat a similarity score as support. A
passage is evidence only when it clears the score floor **and** shares enough of
the question's own content words (`_support_tokens` — stop words removed,
numbers exact). Measured live on the real corpus:

| question | best score | coverage | verdict |
| --- | --- | --- | --- |
| What is the GOVERN function? | 0.735 | 1.00 | `grounded` |
| What is the purpose of the MEASURE function? | 0.638 | 0.67 | `partial` (missing "purpos") |
| What is a checksum? | 0.612 | 0.00 | `none` |
| What licence is Redis distributed under? | 0.666 | 0.00 | `none` |

The last row is the point of the coverage gate: retrieval *always* returns its
top-k, and "Redis" scored 0.67 against NIST AI-RMF text. Without a coverage
check that passage would have been cited as the source of a Redis answer.

## 3. Claim verification — and its deliberate limits

QA classifies only what this process can actually decide:

| Claim class | How it is decided | Reported as |
| --- | --- | --- |
| Figures (%, multiplier, data rate, IOPS) | present in the answer but in none of the question, the cited provenance or the answered-from passages | "cannot support … treat them as unverified" |
| Figures the answer labels as an estimate | explicit hedge word in the same sentence (`approximately`, `roughly`, `estimate`, `ballpark`, `order of magnitude`) | not flagged (labelled, not hidden) |
| Arithmetic on user-supplied numbers | plain counts are not figures; only percentages/multipliers/rates are considered | not flagged |
| Licence claims | closed vocabulary (MIT, Apache, BSD, GPL, LGPL, MPL, SSPL, public domain, proprietary, closed source) compared **per subject**: a sentence is contradicted only when no passage states the same licence about the same subject words and at least one states a different one | `unsupported` (no source), `contradicted`, or nothing |
| Citations | filename-like tokens matched **whole** against the retrieved file names | "cites file(s) this run did not retrieve" |
| Attribution | an evidence-built answer with no `evidence:` provenance, or a memory answer referring to "your documents" while nothing relevant was found | "cannot be attributed" / "refers to local documents" |

**Not attempted:** general factual correctness, version compatibility, security
assertions, dates. No deterministic check here can establish those, and a check
that guessed would be worse than the honest label. A single weak claim never
fails the answer: these are findings, not a verdict, and a QA finding has never
changed an execution's status.

## 4. Tests

`scripts/test_general_evidence.py` — 35 tests, deterministic, offline, no
network, no vector store, no Ollama (scripted provider double + fixed passages):

* coverage: grounded / partial / none / below-floor / keyword-mention-is-not-support / conflicting licences / no passages;
* answers: citation from evidence, partial disclosure, conflict disclosure, fabricated filename dropped, fabricated table reference dropped, `[excerpt N]` normalization;
* fallbacks: no documents labelled, insufficient evidence labelled, gate refusal falls back labelled, degenerate output, provider outage, learned requirements reach the evidence prompt;
* claim QA: contradicted licence, unsupported licence, memory licence unverified, the audit's performance/throughput fixture, evidence-supported figure not flagged, arithmetic not flagged, disclosed estimate not flagged, fabricated citation, missing provenance, dotted filename false positive (live regression), "according to your documents" overclaim;
* routing unchanged: local-only → `books`, explicit web authorization → `research`, ordinary question → `general`; follow-up still cites `execution:<id>`;
* state contract: `evidence` is a declared `AgentState` key.

Two existing tests were **updated to the new contract, not weakened**:

* `test_agent_architecture.py::test_agentstate_has_expected_keys` — `evidence` joins the declared keys (LangGraph drops undeclared state keys, which is exactly the bug found live below);
* `test_e2e_assistant.py::…general_question_answers_once_without_plan_or_sources` — still one model call, still no plan, still no sources; the answer must now also carry the "Answered from my own knowledge" label.

## 5. Evaluation: does grounding actually help?

`app/evidence_eval.py` + `scripts/eval_evidence.py` + `benchmarks/evidence_general_v1.json`
(15 cases; 12 measured through the node, 3 through routing/follow-up). The same
node, the same QA gate and the same scripted provider run twice: retrieval
**off** (baseline) and **on** (grounded). Expected facts, licence values and
figures are fixtures in the dataset, never model output.

```
env\Scripts\python.exe scripts\eval_evidence.py --print
    :: evidence: storage\evidence_eval\evidence_general_report.{json,md}
```

| metric | baseline | grounded | delta |
| --- | --- | --- | --- |
| factual support rate | 0.667 | **1.000** | +0.333 |
| unsupported claim rate | 0.083 | 0.083 | 0.000 |
| contradicted claim rate | 0.000 | **0.083** | +0.083 (detection, not prevalence) |
| citation accuracy | n/a (nothing to cite) | **1.000** | — |
| evidence coverage | 0.000 | 0.740 | +0.740 |
| partial-answer quality | 0.000 | **1.000** | +1.000 |
| correct abstention | 0.667 | 1.000 | +0.333 |
| unnecessary abstention | 0.333 | **0.000** | −0.333 |
| execution reliability | 1.000 | 1.000 | 0.000 |
| **cases passed** | **7/12** | **12/12** | **+5, no regressions** |

Improved cases: E1 (Redis licence), E2 (partial coverage), E4 (relevant-looking
but insufficient), E5 (conflicting documents), E6 (value swapped between two
subjects — undetectable by token coverage, caught by claim-level QA).

The harness measures **the pipeline, not the model**: the provider double
repeats each case's prepared reply. A real model that cannot answer from
passages still fails — and the pipeline then refuses to present that answer as
sourced, which is the property being tested.

## 6. Live Ollama verification

Executions created through the running API (`POST /api/v1/executions`) — the
same endpoint the Control Center uses:

| id | question | route | outcome |
| --- | --- | --- | --- |
| `62125bc639d5` | What are the four functions of the AI Risk Management Framework? | `books` | completed, cited `NIST.AI.100-1.pdf, page 3`, QA clean |
| `7d9aadc7a069` | What is the purpose of the MEASURE function? | `general` | evidence retrieved, the model's composed answer did not survive the gate → labelled memory answer, QA clean |
| `5a73002d7a37` | What licence is Redis distributed under? | `general` | no relevant passages → labelled memory answer, **QA flagged**: "states licence(s) GPL but no source was consulted … unverified" |

In-process, the same Redis question answers "Redis is distributed under the
GNU General Public License version 3" — a wrong claim, now visibly unverified
instead of authoritative, which is the whole point.

Two defects were found by these live runs and fixed:

1. **False positive citation finding.** `NIST.AI.100-1.pdf` was split at its dots
   and reported as the fabricated file `100-1.pdf` against a correct, properly
   attributed answer. The matcher now consumes the whole dotted token; pinned by
   `test_dotted_retrieved_filename_is_not_a_fabricated_citation`.
2. **Undeclared state channel.** LangGraph persists only keys declared on
   `AgentState`, so the `evidence` record was silently dropped before QA and the
   licence check stayed silent (execution `5e23351c94e0` had no flag; the same
   question after the fix produced `5a73002d7a37` with one). Fixed in
   `app/state.py`, pinned by a contract test and by the agent-architecture key
   test.

## 7. Before/after, same question, same model

Question: *"What is the purpose of the MEASURE function?"* (the corpus is the
NIST AI RMF — the passage set genuinely covers MEASURE).

* **Before** (retrieval off): "The MEASURE function is typically used to
  calculate a specific value, such as time, data, or other metrics. It is
  commonly used in programming or data processing contexts…" — presented with
  full confidence, no label, no sources, no flag.
* **After**: the retrieved NIST passages were used, the composed answer above
  did not pass the grounding gate, and the user gets the same text **labelled**
  "Answered from my own knowledge: no passage in your local document library
  covers this, so treat specific figures and claims as unverified." — and for
  the Redis licence question, the QA flag in the execution record.

## 8. Remaining limitations (honest list)

1. **The pipeline cannot make a weak model correct.** A 0.6B model often fails
   to answer from supplied passages; the gate then falls back to a labelled
   memory answer. Grounding improves *accountability*, not knowledge.
2. **Answering from evidence is not the same as verifying it.** A passage the
   model quotes wrongly can still pass token coverage when the wording overlaps;
   claim-level checks cover licences and figures only.
3. **Non-licence factual claims are not checked.** "Redis is written in Rust"
   from memory is labelled but not detected.
4. **Licence contradiction needs subject overlap.** A claim about an entity
   whose words do not appear in the passage clause is reported as *unsupported*
   rather than *contradicted*.
5. **No corpus refresh.** Evidence quality is whatever `ingest_books.py` put in
   the store; stale documents produce stale citations.
6. **Retrieval is similarity-only.** No reranking, no hybrid search; the
   coverage floor is the only defence against the "mentions the keyword"
   failure, and thresholds are constants.
7. **Passage text lives in memory only.** QA reads it in-process; it is not
   persisted, so a post-hoc audit of a run cannot re-check its claims.
8. **Not tested live:** an evidence path with a model strong enough to answer
   reliably from passages (only `qwen3:0.6b` is installed), a corpus covering
   licensing questions, and conflicting-document behaviour on real files.

## 9. Next highest-value work

Give the general route's evidence a reranker plus a per-sentence entailment
check against the passages it cites, so "the passage does not say this" is
detected for prose claims too — the residual gap between *cited* and *correct*.
