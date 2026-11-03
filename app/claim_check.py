"""Claim-level verification: does the CITED passage actually state the claim?

The evidence layer (app/evidence.py) decides *whether* passages cover a
question and the books grounding gate drops sentences no passage supports. That
gate works over the whole retrieved set, which leaves a real hole: a sentence
can be covered by passage A and still cite passage B — "supported" by the run,
attributed to the wrong document, and unverifiable for a reader who follows the
citation. This module checks each claim against the passage it actually cites.

STATUSES
  supported     every content token (numbers exact) is in the cited passage and
                nothing there contradicts it
  partial       the cited passage covers most of it (>= 60% of its content
                words); the rest is named
  contradicted  the cited passage states the opposite polarity, or a different
                licence for the same subject
  unsupported   the cited passage does not state it (or the cited file/page was
                never retrieved — a fabricated attribution)
  unable        no passages were consulted at all, so nothing can be checked

HONESTY
  * Token coverage is NOT entailment. "Supported" means "the cited passage
    states these words, with the same polarity and the same numbers" — it does
    not establish that the claim is true in the world, and it is never
    presented that way.
  * No model judges its own answer here: every verdict is deterministic and
    reproducible. The optional LLM verifier (`BOOKS_VERIFIER`) stays where it
    was, used by the books route only, and is not consulted for this.
  * Uncertainty is reported as `unable`/`partial`, never upgraded to
    `supported`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.agents.books_grounding import (
    _FILENAME_RE,
    _PARAPHRASE_GROUPS,
    _PAGE_REF_RE,
    _has_negation,
    _clauses,
    _metadata_tokens,
    _polarity_conflict,
    _source_ref,
    _support_tokens,
)
from app.evidence import licence_values

SUPPORTED = "supported"
PARTIAL = "partial"
CONTRADICTED = "contradicted"
UNSUPPORTED = "unsupported"
UNABLE = "unable"

PARTIAL_FLOOR = 0.6
MAX_CLAIM_CHARS = 240

# Sentences that are not claims to check: the answer's own disclosure and the
# generated References line.
_SKIP_MARKERS = (
    "not stated by the passages",
    "different licences",
    "answered from my own knowledge",
    "references:",
    "no passage in your local document library",
    "this was not checked",
)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or ""))
            if s.strip()]


def _claim_text(sentence: str) -> str:
    text = re.sub(r"\s+", " ", sentence).strip()
    return text[:MAX_CLAIM_CHARS] + ("…" if len(text) > MAX_CLAIM_CHARS else "")


def _cited_nodes(sentence: str, nodes: list) -> tuple:
    """(matching nodes, cited but unknown) for the sentence's inline citations.

    No citations at all means "whole retrieved set is the evidence base" —
    which is how the gate already read the answer.
    """
    files = {m.group(0).lower() for m in _FILENAME_RE.finditer(sentence)}
    pages = {m.group(1) for m in _PAGE_REF_RE.finditer(sentence)}
    if not files and not pages:
        return list(nodes), False

    matched, unknown = [], False
    for node in nodes:
        meta = getattr(node, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        name = meta.get("file_name") or meta.get("book_title")
        name = (str(name).strip().lower()
                if isinstance(name, str) and name.strip() else None)
        page = meta.get("page_label")
        if page is None:
            page = meta.get("page_number")
        page = str(page).strip() if page is not None else None
        name_hit = name is not None and name in files
        page_hit = page is not None and page in pages
        if files and not name_hit:
            if not pages:
                unknown = True
            continue
        if pages and not page_hit:
            if not files:
                unknown = True
            continue
        matched.append(node)
    if not matched:
        unknown = True
    return matched, unknown


def _corpus(nodes: list) -> tuple:
    corpus: set = set()
    numbers: set = set()
    clauses: list = []
    for node in nodes:
        text = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", None)
        toks = _support_tokens(text) | _metadata_tokens(
            meta if isinstance(meta, dict) else {})
        corpus |= toks
        numbers |= {t for t in _support_tokens(text) if t.isdigit()}
        for clause in _clauses(text):
            clauses.append((clause, _support_tokens(clause),
                            _has_negation(clause)))
    return corpus, numbers, clauses


def _covered(token: str, corpus: set) -> bool:
    if token in corpus:
        return True
    if token.isdigit():
        return False
    return any(token in group and not group.isdisjoint(corpus)
               for group in _PARAPHRASE_GROUPS)


def _licence_contradiction(sentence: str, nodes: list) -> Optional[str]:
    """A licence claim contradicted by EVERY passage that addresses it.

    The supporting half is essential: two documents naming different licences
    are a DISCLOSED conflict (the node says so), not proof that the answer is
    wrong — an answer corroborated by one of them must not be reported as
    contradicted. Only a claim no passage supports, where at least one states
    a different licence for the same subject, is a contradiction.
    """
    mine = licence_values(sentence)
    if not mine:
        return None
    subject = {t for t in _support_tokens(sentence)
               if not t.isdigit() and len(t) > 3
               and "licen" not in t and "sourc" not in t}
    if not subject:
        return None
    support, against = False, set()
    for node in nodes:
        text = str(getattr(node, "text", "") or "")
        for clause in _clauses(text):
            theirs = licence_values(clause)
            if not theirs:
                continue
            if not (subject & {t for t in _support_tokens(clause)
                               if len(t) > 3}):
                continue
            if theirs == mine:
                support = True
            else:
                against |= theirs
    if against and not support:
        return "passage states " + ", ".join(sorted(against)) + \
               " for the same subject"
    return None


def verify_sentence(sentence: str, nodes: list) -> Dict[str, Any]:
    """Classify one sentence against the passages it cites."""
    candidates, unknown = _cited_nodes(sentence, nodes)
    claim: Dict[str, Any] = {
        "text": _claim_text(sentence),
        "status": UNABLE,
        "reason": "",
        "evidence": [],
        "missing": [],
    }
    if unknown:
        claim["status"] = UNSUPPORTED
        claim["reason"] = "cited source was not retrieved in this run"
        cited = sorted({m.group(0) for m in _FILENAME_RE.finditer(sentence)})
        if cited:
            claim["reason"] += " (" + ", ".join(cited) + ")"
        return claim

    if not candidates:
        claim["status"] = UNABLE
        claim["reason"] = "no passage was consulted for this claim"
        return claim

    claim["evidence"] = [ref for ref in (_source_ref(
        getattr(n, "metadata", None)
        if isinstance(getattr(n, "metadata", None), dict) else None)
        for n in candidates) if ref]

    corpus, numbers, clauses = _corpus(candidates)
    tokens = _support_tokens(sentence)

    if _polarity_conflict(sentence, clauses):
        claim["status"] = CONTRADICTED
        claim["reason"] = "the cited passage states the opposite"
        return claim

    licence_conflict = _licence_contradiction(sentence, candidates)
    if licence_conflict:
        claim["status"] = CONTRADICTED
        claim["reason"] = licence_conflict
        return claim

    missing_numbers = sorted(t for t in tokens if t.isdigit() and t not in numbers)
    if missing_numbers:
        claim["status"] = UNSUPPORTED
        claim["reason"] = ("figure(s) not stated by the cited passage: "
                           + ", ".join(missing_numbers))
        claim["missing"] = missing_numbers
        return claim

    missing = sorted(t for t in tokens if not _covered(t, corpus))
    claim["missing"] = missing
    if not missing:
        claim["status"] = SUPPORTED
        claim["reason"] = "stated by the cited passage"
        return claim

    covered = len(tokens) - len(missing)
    share = (covered / len(tokens)) if tokens else 0.0
    if share >= PARTIAL_FLOOR:
        claim["status"] = PARTIAL
        claim["reason"] = "cited passage covers part of it; missing: " + \
                          ", ".join(missing)
    else:
        claim["status"] = UNSUPPORTED
        claim["reason"] = "the cited passage does not state: " + \
                          ", ".join(missing)
    return claim


def verify(answer: str, nodes: list) -> Dict[str, Any]:
    """Verify every checkable claim in an answer.

    Returns {"claims": [...], "summary": {status: count, ...}, "checkable":
    bool}. Non-claim sentences (disclosures, the References line, refusals) are
    skipped, not counted as failures.
    """
    claims: List[Dict[str, Any]] = []
    for sentence in _sentences(answer):
        low = sentence.lower()
        if any(marker in low for marker in _SKIP_MARKERS):
            continue
        if not _support_tokens(sentence):
            continue
        claims.append(verify_sentence(sentence, nodes))

    summary = {status: 0 for status in
               (SUPPORTED, PARTIAL, CONTRADICTED, UNSUPPORTED, UNABLE)}
    for claim in claims:
        summary[claim["status"]] = summary.get(claim["status"], 0) + 1
    return {"claims": claims, "summary": summary,
            "checkable": bool(nodes) and bool(claims)}
