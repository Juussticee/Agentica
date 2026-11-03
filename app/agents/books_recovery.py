"""Recovery: the ONE bounded regeneration for a provable model self-refusal
(M5).

Extracted from books_agent.py (Increment 2 decomposition); the eligibility
screen and its bounded contract moved verbatim. Grounding/verification of
the recovered text is unchanged and still lives in books_grounding -
recovery can only surface content those gates already accept.
"""

from app.agents.books_grounding import (
    _claim_evidence,
    _metadata_tokens,
    _PARAPHRASE_GROUPS,
    _support_tokens,
)
from app.config import NOT_FOUND_PHRASE
from app.llm_client import llm_text


def _host():
    """The books_agent facade module - the compatibility surface where
    tests and eval harnesses patch seams (get_llm, STORAGE_DIR, caches).

    Function-level import: books_agent imports THIS module at its top, so a
    module-level import would be circular. Resolving seams there keeps
    `books_agent.X = fake` patches effective after the Increment 2 split.
    """
    import app.agents.books_agent as host
    return host


# --- M5: bounded recovery from a pure model self-refusal ------------------
# A small instruction-tuned model can emit the mandated not-found phrase
# even when the retrieved passages DO cover the question (observed live:
# score 0.78, on-topic corpus, refusal). This is response SHAPE, not an
# evidence judgment — so when the refusal is provably a self-refusal (good
# score, real nodes, question topically covered by the evidence), the agent
# regenerates ONCE from the SAME retrieved passages and re-runs the result
# through the unchanged grounding/polarity/numeric/citation gates held
# in books_grounding (app.agents.books_grounding).
#
# Deliberate bounds (each one is a contract, not a tunable):
#   * at most ONE extra generation — a second refusal abstains immediately,
#     there is no third attempt and no retry loop anywhere;
#   * eligibility requires a pure `_refusal_only` response, non-empty
#     source nodes with text, score >= BOOKS_MIN_SCORE, and deterministic
#     question-to-evidence topical coverage — empty retrieval, exceptions,
#     low score and ordinary grounding failures NEVER reach this path;
#   * the retry prompt is built from the ORIGINAL source nodes' real text
#     via `_claim_evidence` (the audited, file_path-free, size-bounded
#     evidence builder) — never reconstructed or fabricated passages, and
#     no second retrieval cycle;
#   * the recovered text is untrusted model output exactly like the first
#     answer: citations stripped, then `_grounded_repair` (token support,
#     polarity, numeric provenance, metadata citation gates) and the
#     optional verifier all run unchanged — recovery can only surface
#     content that the normal gates already accept.
# Honest limit: topical COVERAGE is not entailment. It only decides whether
# ONE regeneration is worth attempting; what may be SAID stays gated.
_RECOVERY_MIN_COVERAGE = 0.5

_RECOVERY_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "The passages above are the retrieved evidence for the question below. "
    "Answer the question directly using ONLY the passages above. Do not use "
    "outside knowledge. Never invent figures, explanations, or conclusions. "
    "A previous answer declined to answer; that declination is not evidence "
    "— decide from the passages themselves what they actually state.\n"
    "If the passages genuinely do not support the question at all, reply "
    f"with exactly: {NOT_FOUND_PHRASE}\n"
    # 2026-10-25 fix cycle: the main QA prompt's partial-coverage clause was
    # missing here, so a weak model could only choose between full answer and
    # NOT_FOUND — observed live: eligible retries for answerable questions
    # still returned the pure refusal twice (A1/A4). Byte-identical clause
    # to the main prompt above; the NOT_FOUND contract stays reserved for
    # passages that support nothing.
    "If the passages cover the general topic but not the requested detail, "
    "state ONLY the covered facts from the passages and then say explicitly "
    "which requested part is not covered by the passages — do not claim the "
    "whole topic is absent.\n"
    "Question: {question}\n"
    "Answer: "
)


# --- question-structure tokens for the M5 ELIGIBILITY screen only --------
# (authorized W1, 2026-10-27 — the coverage design decision the fix report
# asked for rather than another stop word.)
#
# Words that state WHAT THE USER ASKED THE ANSWER TO DO — instruction verbs
# ("give", "cite") and the comparison task noun ("difference") — which an
# evidence passage never echoes, exactly like the interrogatives and task
# verbs already removed from the required set (FR-2). They are subtracted
# HERE ONLY: `_grounded_repair`, `_support_tokens` and every answer-side
# gate are untouched, so nothing may be SAID on account of this list. The
# screen's own contract is unchanged — it decides whether ONE bounded
# regeneration is worth attempting, never what it may claim, and the
# grounding/polarity/numeric/citation gates still run on its output.
#
# Evidence (.diag/w1_diagnosis/diagnose.py, real top-3 Table 2/3 chunks):
# the T3 Map/Measure question scored 5/12 = 0.42 (recovery denied) and
# becomes 5/9 = 0.56 (eligible). The genuinely absent CONTENT nouns stay
# required — "activity", "evidenc", "concret", "one" are NOT in this set —
# so the B1 "missing detail" screen and the off-topic screen keep refusing
# (pinned by scripts/test_self_refusal_recovery.py).
_QUESTION_TASK_TOKENS = frozenset({"give", "cite", "differenc"})


def _question_covered_by_evidence(question: str, source_nodes: list) -> bool:
    """Deterministic question-to-evidence TOPICAL coverage (M5 eligibility).

    True when at least half of the question's own content tokens — after
    subtracting the instruction/task vocabulary in `_QUESTION_TASK_TOKENS`
    (eligibility screen only) — occur in the retrieved passages' text or
    their allowlisted citation metadata, with word-level relief through the
    existing `_PARAPHRASE_GROUPS` and the same `_support_tokens`
    normalization as the grounding gate (numbers must match literally).
    Topicality screen for whether ONE regeneration is worth attempting —
    never proof that the evidence ANSWERS the question (that stays
    `_grounded_repair`'s job).
    """
    q_toks = _support_tokens(question) - _QUESTION_TASK_TOKENS
    if not q_toks:
        return False
    corpus = set()
    for node in source_nodes:
        corpus |= _support_tokens(getattr(node, "text", "") or "")
        meta = getattr(node, "metadata", None)
        if isinstance(meta, dict):
            corpus |= _metadata_tokens(meta)
    if not corpus or not any(
            _support_tokens(getattr(n, "text", "") or "")
            for n in source_nodes):
        return False            # empty retrieval: metadata alone is no
        # usable evidence, so a regeneration could only ever re-refuse.
    covered = 0
    for tok in q_toks:
        if tok in corpus:
            covered += 1
        elif not tok.isdigit() and any(
                tok in group and not group.isdisjoint(corpus)
                for group in _PARAPHRASE_GROUPS):
            covered += 1
    return covered * 2 >= len(q_toks)


def _self_refusal_retry(question: str, source_nodes: list) -> str | None:
    """The ONE bounded regeneration for an eligible pure self-refusal.

    Returns the stripped model text, or None on any failure (invoke error,
    non-text output). Never logs the prompt, the passages, or the response;
    never retrieves again; never raises.
    """
    blocks = _claim_evidence(
        {"nodes": [i for i, n in enumerate(source_nodes)
                   if str(getattr(n, "text", "") or "").strip()]},
        source_nodes)
    if not blocks:
        return None
    context = "\n\n".join(block["text"] for block in blocks)
    prompt = _RECOVERY_PROMPT.format(context=context, question=question)
    try:
        return str(llm_text(_host().get_llm().invoke(prompt)) or "").strip()
    except Exception:
        return None
