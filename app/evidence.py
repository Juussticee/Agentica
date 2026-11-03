"""Evidence for the GENERAL route: retrieve, assess, compose, verify.

Why this module exists (2026-11-01): the general route answered from model
knowledge alone, and the reliability audit showed what that costs — incorrect
Redis/Memcached licensing, invented throughput, no source to check. The local
index and its grounding gate already existed for the books route; this module
points the SAME index and the SAME gate at general questions instead of
building a second retrieval system.

RESPONSIBILITIES
  retrieve()      read-only passages from the persisted Chroma index (no
                  model, no writes, no second embedding stack).
  assess()        deterministic evidence coverage for the QUESTION: relevance,
                  which requested topics the passages cover, which are
                  missing, and whether the passages contradict each other.
  passages()      the bounded, path-free passage text handed to the model.
  compose_prompt()the evidence-first instruction block.
  postprocess()   strip fabricated citations, run the existing
                  books_grounding `_grounded_repair` gate over the composed
                  answer, and build citations from the nodes that ACTUALLY
                  supported a retained sentence.
  licence_values()/licence_conflict()  narrow, honest claim checks: a licence
                  name is a checkable fact with a fixed vocabulary, so a
                  licensing claim can be compared with the evidence instead
                  of being silently trusted.

HONESTY CONTRACT
  * A high similarity score is NOT support: assess() requires lexical
    coverage of the question too, and postprocess() still refuses sentences
    the passages do not state.
  * A passage that merely mentions the topic never becomes evidence for a
    specific figure or licence.
  * When evidence is absent or insufficient the caller says so; this module
    never invents passages, citations, scores or coverage.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from app.agents.books_grounding import (
    _grounded_repair,
    _reference_footer,
    _source_ref,
    _strip_model_citations,
    _support_tokens,
    _PARAPHRASE_GROUPS,
    _metadata_tokens,
)
from app.config import (
    EVIDENCE_MIN_COVERAGE,
    EVIDENCE_MIN_SCORE,
    EVIDENCE_PASSAGE_CHARS,
    EVIDENCE_SNAPSHOT_CHARS,
    EVIDENCE_TOP_K,
)

# Marker the scripted model double (and any caller) uses to detect that real
# evidence was supplied. Kept in this module so the benchmark and the node
# cannot drift apart.
EVIDENCE_MARKER = "Excerpts from the user's local documents"

# Provenance prefix for a passage the run actually consulted. QA checks that a
# grounded answer names at least one of these.
SOURCE_PREFIX = "evidence:"

LOCAL_ONLY_NOTE = (
    "Answered from my own knowledge: no passage in your local document "
    "library covers this, so treat specific figures and claims as unverified."
)


# --- licence vocabulary ---------------------------------------------------
# Fixed, small, and deliberately closed: a licence NAME is a fact this
# process can compare against a passage. It is not a general fact checker.
_LICENCES = {
    "mit": "MIT",
    "apache": "Apache",
    "bsd": "BSD",
    "gpl": "GPL",
    "lgpl": "LGPL",
    "mpl": "MPL",
    "sspl": "SSPL",
    "public domain": "public domain",
    "proprietary": "proprietary",
    "closed source": "closed source",
}
_LICENCE_RE = re.compile(
    r"\b(mit|apache(?:[- ]2(?:\.0)?)?|bsd(?:[- ]\d)?|lgpl|gpl(?:v?\d)?|mpl|"
    r"sspl|public domain|proprietary|closed source)\b",
    re.IGNORECASE,
)


def _canonical_licence(raw: str) -> str:
    low = raw.strip().lower()
    for key, value in _LICENCES.items():
        if low.startswith(key):
            return value
    return raw.strip()


def licence_values(text: str) -> set:
    """Canonical licence names mentioned in a piece of text."""
    return {_canonical_licence(m.group(1)) for m in _LICENCE_RE.finditer(text or "")}


def asks_about_licence(question: str) -> bool:
    return bool(re.search(r"\blicen[cs]|\bpermit(?:ted|s)?\b|\bterms of use\b",
                          str(question or ""), re.IGNORECASE))


def licence_conflict(nodes: list) -> bool:
    """True when the retrieved passages name DIFFERENT licences.

    Deliberately coarse: it does not try to match subjects. It answers only
    "do these passages tell one story about licensing, or several?" — enough
    for the answer to disclose disagreement instead of picking one.
    """
    found: set = set()
    for node in nodes:
        found |= licence_values(getattr(node, "text", "") or "")
    return len(found) > 1


# --- retrieval ------------------------------------------------------------

def retrieve(question: str, top_k: int = EVIDENCE_TOP_K) -> list:
    """Passages from the persisted index; [] on any failure (fail-soft).

    Never raises: a missing index, a stopped embedding model or a broken
    dependency means "no local evidence", which the caller reports honestly
    rather than turning into a failed execution.
    """
    text = str(question or "").strip()
    if not text:
        return []
    try:
        import app.agents.books_agent as books

        getter = getattr(books, "_get_retriever", None)
        if getter is None:            # very old facade: no evidence, no crash
            return []
        return list(getter(top_k=top_k).retrieve(text) or [])
    except Exception:  # noqa: BLE001 — no local evidence is a valid answer
        return []


def _corpus(nodes: list) -> Tuple[set, List[set]]:
    """Union of support tokens and per-node token sets (text + metadata)."""
    corpus: set = set()
    per_node: List[set] = []
    for node in nodes:
        text = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        toks = _support_tokens(text) | _metadata_tokens(meta)
        per_node.append(toks)
        corpus |= toks
    return corpus, per_node


def _covered(token: str, corpus: set) -> bool:
    if token in corpus:
        return True
    if token.isdigit():
        return False
    return any(token in group and not group.isdisjoint(corpus)
               for group in _PARAPHRASE_GROUPS)


def _score(node: Any) -> Optional[float]:
    value = getattr(node, "score", None)
    return float(value) if isinstance(value, (int, float)) else None


# --- reranking (2026-11-01) ------------------------------------------------
# Cosine similarity alone ranks passages that merely share the topic above the
# passage that answers the question, and it happily returns the same chunk four
# times. Reranking here is deliberately deterministic and offline:
#
#   rerank = similarity
#          + 0.40 * share of the QUESTION's content words this passage covers
#          + 0.20 * share of the question's TECHNICAL words it states literally
#
# and the selection then drops near-duplicates and prefers passages that add
# question words no better-ranked passage already covers (so a two-part
# question can be answered from two documents). No model, no new dependency:
# a reranker whose cost and failure mode cannot be measured would not be worth
# the latency.
_DUPLICATE_JACCARD = 0.9
_DIVERSITY_WEIGHT = 0.05
_DIVERSITY_CAP = 3


def _node_tokens(node: Any) -> set:
    text = getattr(node, "text", "") or ""
    meta = getattr(node, "metadata", None)
    return _support_tokens(text) | _metadata_tokens(
        meta if isinstance(meta, dict) else {})


def _node_ref(node: Any) -> Optional[str]:
    meta = getattr(node, "metadata", None)
    return _source_ref(meta if isinstance(meta, dict) else None)


def _technical(tokens: set) -> set:
    """Question words worth exact-matching: numbers and long terms."""
    return {t for t in tokens if t.isdigit() or len(t) >= 5}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select(question: str, nodes: list,
           top_k: int = EVIDENCE_TOP_K) -> Dict[str, Any]:
    """Reorder, deduplicate and diversify the retrieved passages.

    Returns {"nodes": [...], "ranking": [...], "dropped_duplicates": n}.
    `ranking` is the audit trail of the decision (one row per KEPT passage with
    its similarity, its rerank score and its query coverage), never a claim
    that the passage supports an answer.
    """
    q_tokens = _support_tokens(str(question or ""))
    technical = _technical(q_tokens)
    scored: List[Dict[str, Any]] = []
    kept_sets: List[set] = []
    dropped = 0
    for position, node in enumerate(nodes):
        tokens = _node_tokens(node)
        if any(_jaccard(tokens, other) >= _DUPLICATE_JACCARD
               for other in kept_sets):
            dropped += 1
            continue
        kept_sets.append(tokens)
        similarity = _score(node)
        covered = {t for t in q_tokens if _covered(t, tokens)}
        exact = {t for t in technical if t in tokens}
        coverage = (len(covered) / len(q_tokens)) if q_tokens else 0.0
        exact_share = (len(exact) / len(technical)) if technical else 0.0
        rerank = ((similarity or 0.0) + 0.40 * coverage
                  + 0.20 * exact_share)
        scored.append({
            "node": node,
            "position": position,
            "tokens": tokens,
            "similarity": similarity,
            "coverage": round(coverage, 3),
            "exact": round(exact_share, 3),
            "rerank": round(rerank, 4),
            "ref": _node_ref(node),
        })

    scored.sort(key=lambda row: (-row["rerank"], row["position"]))

    chosen: List[Dict[str, Any]] = []
    covered_all: set = set()
    remaining = list(scored)
    while remaining and len(chosen) < top_k:
        best = max(
            remaining,
            key=lambda row: (
                row["rerank"] + _DIVERSITY_WEIGHT * min(
                    len((row["tokens"] & q_tokens) - covered_all), _DIVERSITY_CAP),
                -row["position"],
            ),
        )
        remaining.remove(best)
        chosen.append(best)
        covered_all |= best["tokens"] & q_tokens

    ranking = [
        {"ref": row["ref"], "similarity": row["similarity"],
         "rerank": row["rerank"], "coverage": row["coverage"],
         "exact": row["exact"], "position": row["position"]}
        for row in chosen
    ]
    return {"nodes": [row["node"] for row in chosen],
            "ranking": ranking,
            "dropped_duplicates": dropped}


def passage_details(nodes: list, ranking: Optional[List[Dict[str, Any]]] = None,
                    supplied: Optional[List[int]] = None,
                    limit: int = EVIDENCE_SNAPSHOT_CHARS
                    ) -> List[Dict[str, Any]]:
    """Per-passage provenance for the execution snapshot.

    Every retrieved passage is recorded — one row each, in retrieval order.
    `limit` is a PER-PASSAGE cap, never a shared budget: the previous shared
    budget silently dropped the fourth of four passages once the first three
    had used it up, so a snapshot claimed a passage was never consulted
    (2026-11-02 live 3-of-4 truncation). A passage longer than the cap is
    stored cut and flagged `truncated`, never omitted.

    `supplied` (0-based node indices) marks the passages that were actually
    placed in the model's prompt; None means "not tracked", not "all".

    Citation-safe metadata only (file name, page, chunk id, document
    timestamps and size as a version identity) — never `file_path`, never
    anything that identifies the machine.
    """
    scores = {(row.get("ref"), row.get("position")): row
              for row in (ranking or [])}
    given = set(supplied) if supplied is not None else None
    out: List[Dict[str, Any]] = []
    for index, node in enumerate(nodes):
        meta = getattr(node, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        full = re.sub(r"\s+", " ", str(getattr(node, "text", "") or "")).strip()
        row = scores.get((_node_ref(node), index)) or {}
        out.append({
            "rank": index + 1,
            "ref": _node_ref(node),
            "file_name": meta.get("file_name") or meta.get("book_title"),
            "page": meta.get("page_label") or meta.get("page_number"),
            "chunk_id": getattr(node, "node_id", None)
                        or getattr(node, "id_", None),
            "document_modified": meta.get("last_modified_date"),
            "document_created": meta.get("creation_date"),
            "document_size": meta.get("file_size"),
            "score": _score(node),
            "rerank": row.get("rerank"),
            "coverage": row.get("coverage"),
            "supplied": (index in given) if given is not None else None,
            "truncated": len(full) > limit,
            "chars": len(full),
            "text": full[:limit],
        })
    return out


def assess(question: str, nodes: list) -> Dict[str, Any]:
    """Deterministic evidence coverage for one question.

    status:
      none      no relevant evidence (nothing retrieved, low best score, or
                the question's own content words are not in the passages)
      partial   relevant, but some requested topics are missing
      grounded  relevant and every requested content word is covered
    """
    q_tokens = _support_tokens(str(question or ""))
    result: Dict[str, Any] = {
        "status": "none",
        "passages": len(nodes),
        "best_score": None,
        "asked": sorted(q_tokens),
        "covered": [],
        "missing": sorted(q_tokens),
        "conflict": False,
        "reason": "no passages retrieved",
    }
    if not nodes:
        return result

    scores = [s for s in (_score(n) for n in nodes) if s is not None]
    best = max(scores) if scores else None
    result["best_score"] = best
    corpus, _ = _corpus(nodes)
    covered = [t for t in q_tokens if _covered(t, corpus)]
    missing = sorted(t for t in q_tokens if t not in covered)
    result["covered"] = sorted(covered)
    result["missing"] = missing
    result["conflict"] = licence_conflict(nodes) if asks_about_licence(question) else False

    # A score alone is not relevance (the audit's lesson): the passage must
    # also share the question's subject vocabulary.
    coverage = (len(covered) / len(q_tokens)) if q_tokens else 0.0
    result["coverage"] = round(coverage, 3)

    if best is not None and best < EVIDENCE_MIN_SCORE:
        result["reason"] = f"best similarity {best:.2f} below floor"
        return result
    if q_tokens and coverage < EVIDENCE_MIN_COVERAGE:
        result["reason"] = ("passages share too little of the question "
                            f"({coverage:.0%} of its content words)")
        return result

    result["status"] = "partial" if missing else "grounded"
    result["reason"] = ("relevant evidence; some requested topics are not "
                        "in the passages" if missing
                        else "relevant evidence covers the question")
    return result


def prompt_passages(nodes: list, limit: int = EVIDENCE_PASSAGE_CHARS
                    ) -> Tuple[str, List[int]]:
    """Bounded passage block for the prompt, plus WHICH nodes it contains.

    Only the citation-safe metadata is rendered (never absolute paths), and
    the whole block is cut at a fixed size so a long passage cannot crowd out
    the instructions. The returned indices are the passages the model was
    actually given — the snapshot records the difference between those and
    everything that was retrieved, instead of implying the model saw all of
    them (2026-11-02).
    """
    parts: List[str] = []
    supplied: List[int] = []
    used = 0
    for index, node in enumerate(nodes):
        meta = getattr(node, "metadata", None)
        ref = _source_ref(meta if isinstance(meta, dict) else None)
        text = re.sub(r"\s+", " ", str(getattr(node, "text", "") or "")).strip()
        if not text:
            continue
        block = f"[{len(parts) + 1}] ({ref or 'source unknown'})\n{text}"
        if used + len(block) > limit:
            break
        parts.append(block)
        supplied.append(index)
        used += len(block)
    return "\n\n".join(parts), supplied


def passages(nodes: list, limit: int = EVIDENCE_PASSAGE_CHARS) -> str:
    """Bounded passage block for the prompt (text + validated source label)."""
    return prompt_passages(nodes, limit)[0]


def evidence_text(nodes: list, limit: int = EVIDENCE_PASSAGE_CHARS) -> str:
    """Raw passage text for QA (figures are checked against this)."""
    joined = "\n".join(str(getattr(n, "text", "") or "") for n in nodes)
    return joined[:limit]


def references(nodes: list, indices: List[int]) -> List[str]:
    """Provenance strings for the passages a retained sentence used.

    `attribution unavailable` is exposed rather than a made-up filename, so a
    grounded answer never cites a source it cannot name.
    """
    out: List[str] = []
    for index in indices:
        if not (0 <= index < len(nodes)):
            continue
        meta = getattr(nodes[index], "metadata", None)
        ref = _source_ref(meta if isinstance(meta, dict) else None)
        out.append(SOURCE_PREFIX + (ref or "unattributed passage"))
    return out


# --- answer composition ---------------------------------------------------

# The INSTRUCTION half of the evidence prompt, exported so the caller can
# strip a model's echo of it (app.llm_client.strip_instruction_echo) without
# duplicating the wording.
EVIDENCE_INSTRUCTIONS = (
    "You are answering a user's question using excerpts from their local "
    "document library below.\n"
    "Rules:\n"
    "- Use ONLY the excerpts. Do not add facts from your own knowledge, "
    "and do not guess missing numbers, licences, versions or prices.\n"
    "- When you state something from an excerpt, cite it inline as "
    "[excerpt N].\n"
    "- If the excerpts cover the topic but not a requested detail, say "
    "plainly which detail they do not state.\n"
    "- If the excerpts do not answer the question at all, say exactly "
    "that instead of answering from memory.\n"
)


def compose_prompt(question: str, nodes: list, learned: str = "",
                   block: Optional[str] = None) -> str:
    """Evidence-first prompt for the general route.

    `block` lets the caller reuse the exact `prompt_passages()` text it already
    computed (so the snapshot can record which passages the model was really
    given without rendering them twice).
    """
    body = passages(nodes) if block is None else block
    return (
        EVIDENCE_INSTRUCTIONS
        + f"\n{EVIDENCE_MARKER}:\n{body}\n\n"
        + f"Question: {question}{learned}\n\nAnswer:"
    )


def evidence_files(nodes: list) -> List[str]:
    """Lowercased file names of the consulted passages (citation allow-list)."""
    out: List[str] = []
    for node in nodes:
        meta = getattr(node, "metadata", None)
        if not isinstance(meta, dict):
            continue
        name = meta.get("file_name") or meta.get("book_title")
        if isinstance(name, str) and name.strip():
            name = name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
            if name not in out:
                out.append(name)
    return out


def postprocess(answer_text: str, nodes: list,
                diagnostics: Optional[list] = None
                ) -> Tuple[str, List[str], Dict[str, Any]]:
    """Apply the books grounding gate to a composed general answer.

    Returns (answer, sources, detail):
      * mode ok/partial -> only sentences the passages support survive, with a
        References list built from the nodes that supported them;
      * mode refusal    -> ("", [], detail) so the caller can be honest
        instead of presenting an unsubstantiated reply as evidence-backed.
    """
    text = str(answer_text or "").strip()
    # Our own prompt convention ("[excerpt N]") is a reference to the passage
    # block, not content: normalize it away before the support gate so the
    # marker itself can never make a supported sentence look unsupported.
    text = re.sub(r"\[\s*excerpt\s*\d+\s*\]", "", text)
    cleaned = _strip_model_citations(text.strip())
    dropped = diagnostics if diagnostics is not None else []
    detail: Dict[str, Any] = {"mode": "refusal", "claims": 0,
                              "dropped": dropped}
    if not cleaned:
        return "", [], detail

    mode, kept, claims = _grounded_repair(cleaned, nodes,
                                          diagnostics=dropped)
    detail["mode"] = mode
    detail["claims"] = len(claims)
    if mode == "refusal" or not kept:
        return "", [], detail

    body = " ".join(kept).strip()
    used: List[int] = []
    for claim in claims:
        for index in claim["nodes"]:
            if index not in used:
                used.append(index)
    footer = _reference_footer(nodes, used) if used else ""
    return (body + footer), references(nodes, used), detail
