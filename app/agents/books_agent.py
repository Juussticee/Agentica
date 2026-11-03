"""Books agent facade + orchestrator: answers from the local Chroma index
built by ingest_books.py.

Increment 2 decomposition (behavior-identical refactor) - the implementation
now lives in three responsibility modules, all re-exported here:

  * app.agents.books_retrieval  - query-engine/embedding construction, the
    LlamaIndex LLM bridge, the LLM-context path filter;
  * app.agents.books_grounding  - refusal/citation detection, the
    deterministic grounding gate, evidence building, the verifier step;
  * app.agents.books_recovery   - the one bounded M5 self-refusal
    regeneration.

This module keeps two things:

  1. the orchestrator `books_agent(state)` - one run's retrieval -> strip ->
     refuse/recover -> ground -> verify -> footer sequence;
  2. the COMPATIBILITY SURFACE. Tests and eval harnesses patch seams on
     THIS module (`_get_query_engine`, `_query_engine`, `_embedding_model`,
     `get_llm`, `STORAGE_DIR`, `CHROMA_COLLECTION`, `BOOKS_VERIFIER`,
     `BOOKS_VERIFIER_EVIDENCE_MAX_CHARS`, ...) and import internals from it.
     Every seam lookup therefore stays here at call time, and the moved
     code reads patchable state back through this facade (function-level
     `_host()` lookups), so `books_agent.X = fake` behaves exactly as it
     did before the split. Names imported above that this module's own code
     does not use are deliberate re-exports, not dead imports.

Known gap (flagged to the user, per instructions): embeddings still come from
Ollama's nomic-embed-text; the Gemini embedding API needs separate setup and
was deliberately NOT wired in.
"""

import time

from app.agents import books_retrieval as _retrieval
from app.agents.books_grounding import (
    _claim_evidence,
    _claim_reference_footer,
    _clauses,
    _CLAUSE_SPLIT_RE,
    _EXEMPT_PHRASES,
    _FIGURE_REF_RE,
    _FILENAME_RE,
    _FOOTER_BLOCK_RE,
    _grounded_repair,
    _has_negation,
    _inline_citation_error,
    _INJECTION_CUES,
    _metadata_tokens,
    _NEGATION_CUES,
    _number_tokens,
    _PAGE_REF_RE,
    _PARAPHRASE_GROUPS,
    _PARTIAL_DISCLOSURE,
    _polarity_conflict,
    _REFUSAL_MARKERS,
    _refusal_only,
    _reference_footer,
    _sentence_covered,
    _source_ref,
    _STOP_TOKENS,
    _strip_model_citations,
    _support_tokens,
    _VERIFIER_CONFLICT_NOTE,
    _VERIFIER_REMOVED_NOTE,
    _VERIFIER_UNAVAILABLE_NOTE,
    _verify_claims,
)
from app.agents.books_recovery import (
    _QUESTION_TASK_TOKENS,
    _question_covered_by_evidence,
    _RECOVERY_MIN_COVERAGE,
    _RECOVERY_PROMPT,
    _self_refusal_retry,
)
from app.agents.books_retrieval import (
    _GetLlmBridge,
    _LLM_SAFE_METADATA,
    _LocalPathFilter,
)
from app.config import (
    BOOKS_MIN_SCORE,
    BOOKS_TOP_K,
    BOOKS_VERIFIER,
    BOOKS_VERIFIER_EVIDENCE_MAX_CHARS,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    NOT_FOUND_PHRASE,
    OLLAMA_BASE_URL,
    STORAGE_DIR,
)
from app.execution.events import RETRIEVAL_COMPLETED, RECOVERY_OUTCOME
from app.execution.instrumentation import emit
from app.llm_client import get_llm, llm_text
from app.state import AgentState, with_agent_output

# Lazy singletons - FACADE-OWNED patch seams (unchanged contract): tests
# and evals reset `books_agent._query_engine = None`, preset
# `books_agent._embedding_model = <mock>` and redirect `STORAGE_DIR` /
# `CHROMA_COLLECTION` here. Construction itself lives in books_retrieval.
_embedding_model = None
_query_engine = None
# Read-only passage retriever cache for the general route's evidence check
# (app/evidence.py). Same facade-ownership rule as the engine above: tests and
# evals patch `_get_retriever` on THIS module.
_retrievers = {}


def _get_retriever(top_k: int = 4):
    """Cached read-only retriever (seam: patch/replace `_get_retriever`)."""
    if top_k not in _retrievers:
        _retrievers[top_k] = _retrieval.build_retriever(top_k=top_k)
    return _retrievers[top_k]


def _get_embedding_model():
    """Cached embedding model (seam: preset books_agent._embedding_model)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _retrieval.build_embedding_model()
    return _embedding_model


def _reset_retrievers():
    """Drop cached retrievers (used with patch seams in tests/evals)."""
    _retrievers.clear()


def _get_query_engine():
    """Cached Chroma query engine - THE documented lazy-engine seam.

    Tests and eval harnesses replace this function outright, reset the
    `_query_engine` cache, and redirect `STORAGE_DIR` /
    `CHROMA_COLLECTION` on this module; the construction itself (including
    the chroma_guard fail-closed check, which must run BEFORE anything is
    built) stays inside books_retrieval.build_query_engine().
    """
    global _query_engine
    if _query_engine is None:
        _query_engine = _retrieval.build_query_engine()
    return _query_engine

def _evidence_record(nodes, *, status: str, reason: str, score,
                     mode: str) -> dict:
    """The passages THIS books run actually consulted (2026-11-02).

    One mechanism for both routes: `app.evidence.passage_details` renders the
    same citation-safe provenance the general route persists (file, page,
    chunk id, retrieval score, document identity, text). Every retrieved node
    is recorded and every one of them was in the query engine's prompt, hence
    `supplied=True` for all — this route has no separate reranking, so
    `ranking` stays empty rather than inventing one.

    Written to `state["books_evidence"]`, NOT `state["evidence"]`: QA's
    claim checks are calibrated for the general route's record, and adding a
    second answer path to that channel would change QA behavior. The runner
    persists either record with the same snapshot writer.
    """
    from app import evidence as evidence_mod

    if not nodes:
        return None
    return {
        "status": status,
        "answered_from": "books",
        "coverage": None,
        "best_score": score,
        "missing": [],
        "conflict": False,
        "reason": reason,
        "grounding": mode,
        "ranking": [],
        "dropped_duplicates": 0,
        "passages_detail": evidence_mod.passage_details(
            nodes, supplied=list(range(len(nodes)))),
        "claim_checks": {},
    }


def _refusal_update(state: AgentState, answer_text: str,
                     diagnosis: dict, nodes=None) -> dict:
    """Refusal output + the structured diagnosis the orchestrator's
    books->research fallback decision reads (2026-10-25 additional phase).

    The user-facing refusal stays byte-identical (mandated phrase, never a
    references footer); only the machine-readable books_refusal dict is
    added. Diagnosis values are scalars ONLY — refusal stage, score, node
    count, bounded-recovery outcome — never passage text, never model
    output (privacy + the M5 no-leak contract).

    A refusal still CONSULTED the retrieved passages (2026-11-02), so the
    evidence record is attached when there were any: an audit needs to see
    that the answer was refused in spite of them.
    """
    update = with_agent_output(state, "books", answer_text, ["books"])
    update["books_refusal"] = diagnosis
    record = _evidence_record(
        nodes, status="none", reason=f"refused: {diagnosis.get('reason')}",
        score=diagnosis.get("score"), mode="refusal")
    if record is not None:
        update["books_evidence"] = record
    return update


def books_agent(state: AgentState) -> dict:
    question = state.get("question", "")
    start = time.perf_counter()

    t_retrieve = time.perf_counter()
    response = _get_query_engine().query(question)
    source_nodes = list(getattr(response, "source_nodes", []) or [])
    score = source_nodes[0].score if source_nodes else None
    # W2: the retrieval that actually happened, recorded while it happens —
    # identifiers/scores/duration only, NEVER passage text (privacy rule).
    # No-op outside an execution run (CLI, unit tests).
    emit(
        RETRIEVAL_COMPLETED,
        f"Retrieved {len(source_nodes)} supporting passage(s)",
        agent_id="books", step_id="books",
        payload={
            "nodes": len(source_nodes),
            "score": score,
            "duration_ms": round((time.perf_counter() - t_retrieve) * 1000),
            "chunks": [
                {
                    "file_name": (getattr(n, "metadata", None) or {}).get("file_name"),
                    "page_label": (getattr(n, "metadata", None) or {}).get("page_label"),
                    "score": getattr(n, "score", None),
                }
                for n in source_nodes
            ],
        },
    )
    # M3: strip any References/Sources/Citations block the model emitted
    # (fabricated, or copied from retrieved document text) BEFORE any other
    # handling — the only footer that can reach the user is the agent's own,
    # built below from validated retrieved-node metadata.
    answer_text = _strip_model_citations(str(response).strip())
    elapsed = time.perf_counter() - start

    low_confidence = score is not None and score < BOOKS_MIN_SCORE
    # The synthesis prompt makes the model reply with the exact not-found phrase
    # when the chunks don't cover the question (nomic's similarity band is too
    # narrow for a fixed threshold alone); the score is kept as a hard floor.
    # Detection is sentence-level (see _refusal_only) so paraphrased refusals
    # cannot slip through and pick up a references footer (spec section 2.1:
    # a refusal-only response presents no evidence, so it cites none).
    refusal = bool(answer_text) and _refusal_only(answer_text)

    # Orchestrated knowledge-fallback diagnosis (2026-10-25 additional
    # phase): which bounded-recovery outcome applies if this run ends in a
    # refusal — recorded on state["books_refusal"] and read by
    # app.supervisor.fallback_decision (never a score/refusal-sentence
    # guess). "not_attempted" = the first output was not a pure
    # self-refusal (grounding/verifier refusals below); a pure refusal
    # starts as "skipped_not_eligible" and becomes "attempted" /
    # "succeeded" / "rejected" exactly where M5 runs. User-facing output,
    # log markers and gates stay byte-identical.
    recovery = "not_attempted" if not refusal else "skipped_not_eligible"
    recovery_reason = None
    initial_refusal = refusal

    # M5: exactly one bounded regeneration for a provable pure self-refusal
    # (never for empty output, low score, exceptions, or grounding failures).
    if (refusal and score is not None and score >= BOOKS_MIN_SCORE
            and source_nodes
            and _question_covered_by_evidence(question, source_nodes)):
        recovery = "attempted"
        print(f"[books_agent] self_refusal_recovery_attempted "
              f"(score={score}, nodes={len(source_nodes)})")
        recovered = _self_refusal_retry(question, source_nodes)
        if recovered:
            recovered = _strip_model_citations(recovered)
        if recovered and not _refusal_only(recovered):
            # Re-enter the UNCHANGED downstream path: _grounded_repair
            # (support/polarity/numeric/citation gates) and the optional
            # verifier still decide what may be said.
            answer_text, refusal = recovered, False
            recovery = "succeeded"
            print(f"[books_agent] self_refusal_recovery_succeeded "
                  f"(chars={len(recovered)})")
        else:
            recovery = "rejected"
            reason = ("second_refusal" if recovered else "empty_or_error")
            recovery_reason = reason
            print(f"[books_agent] self_refusal_recovery_rejected "
                  f"(reason={reason}, score={score})")

    # W2: one durable record of the bounded-recovery outcome for every
    # refusal path (attempted->succeeded / attempted->rejected / denied by
    # eligibility), where stdout only had transient print lines. Scalars
    # only — no prompt, no passage, no model output.
    if initial_refusal:
        emit(
            RECOVERY_OUTCOME,
            f"Recovery outcome: {recovery}"
            + (f" ({recovery_reason})" if recovery_reason else ""),
            agent_id="books", step_id="books",
            payload={"outcome": recovery, "reason": recovery_reason},
        )

    if not answer_text or low_confidence or refusal:
        # Spec: never invent — say explicitly it is not in the books, using
        # the mandated refusal phrase, WITHOUT a references footer.
        # Approved 2026-10-24 RAG fix: the refusal reason is recorded (log
        # only — the user-facing refusal phrase stays exactly as mandated).
        # Distinguishes weak retrieval from a model self-refusal (M5 has
        # already had its ONE bounded regeneration at this point), so
        # post-mortems never have to guess between the two.
        reason = ("empty_or_error" if not answer_text else
                  "low_confidence" if low_confidence else
                  "self_refusal_after_recovery")
        answer_text = NOT_FOUND_PHRASE
        if score is not None and not refusal:
            answer_text += f" (best similarity score {score:.2f})"
        print(f"[books_agent] not-found path "
              f"({elapsed:.1f}s, score={score}, reason={reason})")
        return _refusal_update(state, answer_text,
                               {"reason": reason, "score": score,
                                "nodes": len(source_nodes),
                                "recovery": recovery}, source_nodes)

    # Spec: include book title + page number in the answer when node
    # metadata has them. First run the deterministic grounding safeguard:
    # keep only answer content substantiated by the retrieved passages,
    # disclose whatever was dropped, and refuse when nothing substantiable
    # remains (no references footer on a refusal).
    gate_diagnostics: list = []
    mode, kept_sentences, claims = _grounded_repair(
        answer_text, source_nodes, diagnostics=gate_diagnostics)
    if mode == "refusal":
        # No score note here: the score is NOT the reason (retrieval may
        # even have scored high) — the absence of supporting passages is.
        answer_text = NOT_FOUND_PHRASE
        print(f"[books_agent] grounding refusal ({elapsed:.1f}s, score={score})")
        # 2026-10-25 diagnostics (log-only, fixed at decision time — never
        # derived from a later-overwritten variable): the stage is the
        # grounding gate and each line is one dropped sentence's reason.
        for line in gate_diagnostics:
            print(f"[books_agent] gate_drop: {line}")
        return _refusal_update(state, answer_text,
                               {"reason": "grounding", "score": score,
                                "nodes": len(source_nodes),
                                "recovery": recovery}, source_nodes)

    # M4B: optional bounded semantic verification (BOOKS_VERIFIER=local).
    # Monotone: it can only REMOVE claims; with the default "off" this block
    # is skipped entirely and behavior is byte-identical to M3.
    retained = claims
    notes = []
    n_unavailable = 0
    if BOOKS_VERIFIER == "local":
        retained, notes, n_unavailable = _verify_claims(claims, source_nodes)
        print(f"[books_agent] verifier: {len(retained)}/{len(claims)} "
              "claim(s) confirmed")
        if not retained:
            # Fail-closed abstention: nothing survived verification.
            answer_text = NOT_FOUND_PHRASE
            if n_unavailable >= len(claims):
                # Distinguish "verification never succeeded" from a
                # successful negative verdict — never imply verification.
                answer_text += " (semantic verification unavailable)"
            print(f"[books_agent] verifier refusal ({elapsed:.1f}s, "
                  f"score={score})")
            return _refusal_update(state, answer_text,
                                   {"reason": "verifier", "score": score,
                                    "nodes": len(source_nodes),
                                    "recovery": recovery}, source_nodes)

    dropped = {c["idx"] for c in claims} - {c["idx"] for c in retained}
    if dropped:
        remaining = [s for i, s in enumerate(kept_sentences)
                     if i not in dropped]
        parts = [" ".join(remaining)]
        if mode == "partial":
            parts.append(_PARTIAL_DISCLOSURE)
        parts.extend(notes)
        answer_text = " ".join(parts)
    elif mode == "partial":
        answer_text = " ".join(kept_sentences) + " " + _PARTIAL_DISCLOSURE
    # ("ok" with nothing dropped: the model's text stays byte-identical.)

    # Spec: include book title + page number in the answer when node
    # metadata has them — but ONLY for nodes that support a RETAINED claim
    # (M3 + M4B: a claim removed by verification loses its citation), and
    # never under a name the model wrote. Missing metadata yields an
    # explicit "attribution unavailable" marker instead of an invented name.
    used = []
    for claim in retained:
        for i in claim["nodes"]:
            if i not in used:
                used.append(i)
    footer = (_claim_reference_footer(source_nodes, retained)
              if BOOKS_VERIFIER == "local" and retained
              else _reference_footer(source_nodes, used))
    print(f"[books_agent] answered in {elapsed:.1f}s "
          f"from {len(used)} supporting chunk(s)")
    answer = answer_text
    if footer:
        answer += footer
    update = with_agent_output(state, "books", answer, ["books"])
    # The passages this answered from, for the same immutable snapshot the
    # general route writes (2026-11-02). "grounding" records the gate's own
    # mode; the verifier's removals are already visible in the answer.
    record = _evidence_record(
        source_nodes,
        status="grounded" if mode == "ok" else "partial",
        reason="answered from the retrieved passages", score=score, mode=mode)
    if record is not None:
        update["books_evidence"] = record
    return update
