# Grounding Gate Design Note — Directional (Asymmetric) Paraphrase Aliases

**Status: DESIGN ONLY.** Nothing in this note is implemented. The shipped
gate in `app/agents/books_agent.py` uses the symmetric `_PARAPHRASE_GROUPS`
frozensets, and that behavior is pinned by the regression suites
(`scripts/test_books_grounding_adversarial.py`,
`scripts/test_books_grounding_paraphrase.py`, `scripts/test_books_grounding.py`,
`scripts/test_citation_integrity.py`). Directional aliases must not be
enabled until the tests in §7 exist and pass. This note records the design
so the known subject-swap limitation is closed deliberately, not silently.

Related, already-shipped lessons (do not regress them):

- duration `time` vs multiplication `times` — a family entry must be a
  POST-normalization surface form (`_support_tokens` emits `times`, not
  `time`);
- polarity rule A must use the SAME support relation as coverage, or a
  negated clause stops blocking its own affirmative claim;
- `modifi` is the normal form of modifies/modified/modifying (the dead
  `modif` entry over-refused every natural inflection).

## 1. Why symmetric groups produce false positives

`_sentence_covered(toks, corpus)` accepts a non-literal answer token when
it belongs to a group that some evidence token also belongs to. The test
is symmetric by construction: if evidence token `e` and answer token `a`
share a group, `e` admits `a` exactly as `a` would admit `e`.

Support, however, is a directional relation. A corpus that says
"propulsion **systems**" genuinely supports an answer that rewords it as
"propulsion **spacecraft**" (this is the verified delta-v case and must
keep passing). The reverse is not entailment: evidence describing a
generic **system** ("Each system carries a backup radio.") does not
support an answer that names the subject a **spacecraft** ("Each
spacecraft carries a backup radio.") — yet the symmetric group admits it.
The same asymmetry applies to concepts, not only subjects: the swap is
evaluated word-to-word with no notion of which side is the evidence, so
any pair inside a family is mutually interchangeable regardless of whether
both substitutions preserve truth. This is the concrete, currently pinned
false positive:

- `TestBFamilySwapDepth::test_family3_subject_swap_pinned_as_known_limitation`
  — evidence `Each system carries a backup radio.` + answer
  `Each spacecraft carries a backup radio.` → `mode == "ok"` (pinned).

## 2. Directional support vs a symmetric synonym group

A symmetric group asserts *equivalence within this corpus*: "these words
mean the same thing, full stop." A directional alias asserts *support*:
"an answer may phrase the claim with form `a` when the evidence uses form
`e`." Formally:

- group (today): `covered(a, E) ⇔ ∃e ∈ E : group(a) ∋ e` — one undirected
  set, checked for membership on both sides;
- directional (proposed): `covered(a, E) ⇔ a ∈ E ∨ ∃e ∈ ALIASES[a] : e ∈ E`
  — a lookup keyed by the **answer-side** token; the alias table maps an
  answer form to the set of evidence forms that justify it.

The evidence side never consults the table: evidence tokens only provide
literal matches (and their own normalization). Asymmetry lives in the
table, not in normalization — both sides are still normalized by the same
`_support_tokens`, so the "identical normalization" invariant of the
current contract is preserved; only the *matching relation* becomes
directed.

## 3. Normalization and application (grounded in the family-3 case)

- **Keys** are answer-side tokens exactly as `_support_tokens` emits them;
  **targets** are evidence-side tokens in the same normalized space. No
  second stemmer, no inflection engine — every entry is a literal
  post-normalization string, listed explicitly (the discipline that made
  `times` and `modifi` fixes necessary and sufficient).
- **Lookup is one hop**: literal check first (unchanged), then
  `ALIASES.get(tok, ())` against the corpus. Nothing else changes —
  sentence granularity, stop-list, metadata tokens, the glue/disclosure
  handling all stay as they are.
- **Family-3 example** (the motivating case):

  | direction | mapping | outcome |
  |---|---|---|
  | answer `system` ← evidence `spacecraft` | `ALIASES["system"] = {"spacecraft", "vehicle", "vehicl"}` | verified delta-v answer ("propulsion systems" over corpus "spacecraft") stays admitted |
  | answer `spacecraft` ← evidence `system` | **no entry** | `Each system…` → `Each spacecraft…` becomes a refusal; the pinned limitation test flips to its loudly-marked new expectation |

  Every symmetric family would be decomposed the same way: each direction
  must be registered on its own merits, with the evidence that justifies
  it (the delta-v raw answer justifies `system ← spacecraft, vehicle`;
  nothing justifies the reverse). Directions with no justification are
  dropped rather than mirrored.

## 4. Protections that must remain intact

- **Polarity/negation**: `_polarity_conflict` keeps running after coverage
  and before acceptance, and its rule A must call the SAME support
  relation used by coverage (today `_sentence_covered`; under directional
  aliases, the directional variant). Otherwise the shipped FP-1 regression
  (`test_negated_evidence_blocks_affirmative_family_swap`) reopens:
  negated evidence could again admit its own affirmative claim through a
  family word. Same-polarity family swaps must stay admitted
  (`test_same_polarity_family_swap_still_admitted`).
- **Numbers**: aliases are word-only. The numeric branch of coverage
  (`tok.isdigit()` → literal match mandatory, never group-covered) is
  kept verbatim, and the structural invariant test
  (`test_no_group_contains_a_numeric_member`) is extended to the alias
  table: no key and no target may contain a digit.
- **Units, signs, decimal values**: never aliased (a unit token like
  `meters` must not become an alias for `kilometers`; sign words like
  `negative` are ordinary content tokens with no alias). The E-series
  regressions (changed value 3.9→4.1/3.95, wrong unit, sign word,
  family-word-plus-changed-number) must stay refused under the new
  relation — they are the acceptance tests for this section.
- **Citations and metadata**: alias lookup applies only inside content-token
  coverage. `_inline_citation_error`, `_FILENAME_RE`/`_PAGE_REF_RE`/
  `_FIGURE_REF_RE` validation, `_metadata_tokens`, and both References
  footer builders run before/outside coverage and are untouched: a
  fabricated page or filename is still refused by exact metadata
  comparison, never by alias (`test_fabricated_page_dropped_real_page_kept`,
  `test_plural_page_stem_does_not_bypass_page_number_check`).

## 5. Avoiding transitive alias chains

Transitivity is how a narrow table silently becomes a broad one: with
`a→b` and `b→c` registered, a careless evaluator that re-checks targets
as keys yields `a→c`, re-admitting claims nobody justified. Rules:

- **One hop, always.** `ALIASES[a]` is consulted only against literal
  evidence tokens; a target that matches evidence satisfies the check —
  a target that does NOT match is never expanded through its own entry.
- **No derived entries.** The table is hand-written data; no closure,
  symmetric closure, or "family completion" step may generate rows.
  Converting today's symmetric groups must not mirror each pair
  automatically — that would rebuild the false positive by construction.
- **Guard test.** Before implementation: build the transitive closure of
  the real table in the test and assert `closure == one_hop` (i.e. the
  table contains no chain `a→b→c` at all), plus a synthetic
  three-node table proving the evaluator never uses `b`'s entry when
  matching `a`.

## 6. Migration path (current behavior until tests prove the contract)

1. **Ship nothing.** Today's symmetric groups remain the production
   contract; this note changes no runtime behavior.
2. **Side-by-side evaluator.** Add `_sentence_covered_v2` (literal ∪
   directional lookup) next to `_sentence_covered`, behind an
   opt-in/environment switch defaulting to OFF. With the switch off, the
   old path runs — byte-identical behavior, full suites unchanged.
3. **Shadow comparison.** Run the entire grounding suite set against v2
   explicitly. Required result: identical outcomes on every case except
   the *designated flip set* — exactly one behavioral delta is allowed:
   the family-3 subject-swap pin flips from `ok` to `refusal`. The
   delta-v control (`TestAPositiveControl`, `TestRealVerifiedEvidence`),
   the paraphrase suite, and every citation test must be byte-identical.
4. **Table build.** Derive candidate rows from today's families, then
   delete every direction without a justifying test (see §7). Only
   directions backed by a named regression survive.
5. **Flip and enable.** Update the subject-swap pin to its documented
   refusal expectation (its comment pre-authorizes exactly this flip),
   enable v2 by default, keep v1 behind the switch for one green cycle.
6. **Retire v1.** Remove the symmetric evaluator once the full batch is
   green with v2 only; the group definitions survive as source data for
   the alias rows, not as an active code path.

Each step is independently revertible; no step modifies thresholds,
prompts, retrieval, scoring, persistence, or numeric/citation checks.

## 7. Required tests before any implementation

Positive (must stay/be admitted):

1. delta-v control: both investigated sentences retained (evidence
   contains `spacecraft`, answer uses `system`) — `TestAPositiveControl`.
2. `orbital mechanics` answer over `orbital kinetics` evidence
   (`mechanic ← kinetic`), `propulsion systems` over `propulsion engine`,
   `needs` over `requires`, `calculate` over `derive`, `times` over
   `multiplied` — the existing `_FAMILY_CASES` admitted rows, re-run
   against v2.
3. `modify`/`modified` inflections over `alters` evidence
   (`test_modify_inflection_paraphrase_is_admitted`).
4. same-polarity family swap with both sides negated still admitted.

Negative (must stay/be refused):

5. **the flip**: evidence `Each system carries a backup radio.` → answer
   `Each spacecraft carries a backup radio.` refused (pinned test
   changes expectation, loudly).
6. every existing refusal stays a refusal under v2 — full adversarial
   suite run, zero new admissions beyond flip #5 (assert the diff of
   outcomes, not just green).
7. polarity across a family swap still refused
   (`test_negated_evidence_blocks_affirmative_family_swap`) — proves
   rule A uses the directional relation too.
8. numeric E-series unchanged (value/unit/sign/decimal/missing),
   extended structural invariant over the alias table (§4).
9. citations unchanged: fabricated page/filename/table still dropped;
   real page kept; `pages` stem still gated by the page number.
10. **transitivity guard**: synthetic `a→b`, `b→c` table proves `a` never
    matches evidence `c`; real table asserted chain-free (§5).
11. **directionality (metamorphic)**: for a single-registered direction
    (`system ← spacecraft`), swapping evidence and answer changes the
    outcome — admitted one way, refused the other. This test is the
    defining property no symmetric group can satisfy.

Only after 1–11 exist and pass may the switch default change.

## 8. Known splice limitation (directional aliases do not solve it)

Coverage evaluates an answer sentence against the UNION of all retrieved
nodes' tokens (plus metadata tokens), at answer-sentence granularity.
Consequently a sentence that splices tokens across unrelated passages — or
across two sentences of one passage — is admitted today, with or without
aliases: `Propolis determines transfer budgets.` is accepted from
`Orbital kinetics determines transfer budgets.` + `Propolis is collected
from plant buds.` Whether the match is symmetric, directional, literal, or
aliased, the relation answers only *"is this token evidenced SOMEWHERE in
the retrieved set"*, never *"is it evidenced by the passage relevant to
this claim."* Both variants of that gap are pinned in
`TestGEvidenceBoundariesPinned` and `TestGSamePassageSplicePinned`, and
the agent code documents the same residual honestly.

Closing it requires binding each claim to its own evidence — per-claim
evidence selection (the M4C attribution machinery already narrows
*citations* this way) or real entailment checking — which is a separate,
larger redesign with a known over-refusal cost: single-passage
requirements would refuse legitimate combined summaries. Directional
aliases change *which wordings* an evidence token supports; they do not
change *which evidence* is consulted. Both axes are needed, and they must
land as independent, separately-tested steps.
