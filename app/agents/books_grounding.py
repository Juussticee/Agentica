"""Grounding: refusal/citation detection, the deterministic grounding gate,
evidence building and the bounded verifier step.

Extracted from books_agent.py (Increment 2 decomposition): a responsibility
move only - every function body, constant and comment moved verbatim, and
app.agents.books_agent re-exports all of these names, so existing imports
(`from app.agents.books_agent import _grounded_repair`, `ba._STOP_TOKENS`,
...) and the books/grounding/citation/verifier suites keep working
unchanged.

Pure functions over retrieved nodes: no retrieval, no LLM, no network, no
mutable state. The single facade lookup (`_host()`) reads the verifier
evidence budget that test_verifier tunes on books_agent.
"""

import re

from app import verifier


def _host():
    """The books_agent facade module - the compatibility surface where
    tests and eval harnesses patch seams (get_llm, STORAGE_DIR, caches).

    Function-level import: books_agent imports THIS module at its top, so a
    module-level import would be circular. Resolving seams there keeps
    `books_agent.X = fake` patches effective after the Increment 2 split.
    """
    import app.agents.books_agent as host
    return host


# Sentence-level refusal markers (spec section 2.1): a response counts as a
# refusal only when EVERY sentence claims the books lack the requested
# information. A mixed response ("...however the books do cover X") has
# non-marker sentences and is treated as partial support — the covered parts
# must be kept, not collapsed into a refusal (guaranteed no fabricated
# content: the covered text came from the retrieved evidence).
#
# Scope limits (deliberate): markers must stay rare, whole-claim phrases so
# that a single sentence mixing a supported fact WITH an absent-detail note
# ("...though it is not covered by the passages") is never collapsed — that
# is why "not covered by the passages" and similar are NOT markers. The
# detector classifies RESPONSE SHAPE, not truth: fabricated sentences remain
# a prompt-level contract ("do not use outside knowledge"), not a regex one.
_REFUSAL_MARKERS = (
    "not found in your books",
    "not found in the books",
    "no information about",          # "There is no information about X in the books."
    "not contain the answer",        # "(does/do) not contain the answer"
)


def _refusal_only(answer_text: str) -> bool:
    """Deterministic refusal detector: every sentence is a not-found claim.

    Robust against the small transcription/paraphrase models that reword the
    instructed refusal phrase (the exact-phrase check missed those, leaving
    reworded refusals carrying a references footer). Moving to a
    sentence-level marker check keeps determinism and never misclassifies a
    partial-support answer, because that answer contains substantive
    non-marker sentences.

    This classifies response SHAPE only — it is never proof of factual
    grounding; content support is enforced separately by
    `_enforce_grounding` below.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    if not sentences:
        return False
    return all(
        any(marker in s.lower() for marker in _REFUSAL_MARKERS)
        for s in sentences
    )


# --- citation integrity (M3) ----------------------------------------------
# Retrieved documents and model output are UNTRUSTED DATA. Two rules follow:
#
#  1. The model never authors the References footer. It is rebuilt from the
#     retrieved nodes' OWN metadata, restricted to the nodes that actually
#     supported retained claims (_reference_footer). A filename or page the
#     model wrote can therefore never be accepted as a verified citation.
#  2. Citation-shaped text INSIDE the model's output (a fabricated footer
#     copied verbatim out of a retrieved passage, or an injected instruction
#     block) is stripped before any validated footer is appended
#     (_strip_model_citations), and instruction-boundary phrases are dropped
#     as content (_INJECTION_CUES) — monotone: these rules can only remove
#     text, never add or bless it.

_FOOTER_BLOCK_RE = re.compile(
    r"^[ \t]*(?:references|sources|citations)[ \t]*:", re.I
)

# Instruction-boundary cues: machine-directed phrases that never occur in
# the sample corpus's book prose but are common inside prompt-injection
# payloads. Deliberately narrow (over-refusal risk): a paraphrase without
# any cue is still only bounded by the token support gate — that residual is
# a documented limitation, not a claimed guarantee.
_INJECTION_CUES = (
    "ignore previous", "ignore all previous", "ignore the previous",
    "ignore above", "ignore the above", "disregard previous",
    "disregard the above", "disregard these instructions",
    "system prompt", "developer message", "jailbreak",
    "reveal your instructions", "output your instructions",
    "new instructions", "you are now in", "do not follow",
)


def _strip_model_citations(answer_text: str) -> str:
    """Remove any References/Sources/Citations block the model emitted.

    Such a block is either fabricated (the model cannot know which pages
    were retrieved well enough to be trusted) or copied from text inside a
    retrieved document — in both cases it must not survive alongside the
    agent's own validated footer. Removes the marker line and its
    continuation block (until a blank line or EOF); monotone, never adds.
    """
    lines = answer_text.splitlines()
    kept = []
    i = 0
    while i < len(lines):
        if _FOOTER_BLOCK_RE.match(lines[i]):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept).strip()


# --- deterministic grounding safeguard (spec section 2.1) -----------------
# _refusal_only classifies response SHAPE, never factual grounding. This is
# the closest deterministic substitute for entailment within the current
# architecture: every SUBSTANTIVE sentence of the final answer must have ALL
# of its content tokens present in the retrieved passages (numbers exact,
# function words stop-listed, a coarse plural stem applied to both sides).
# Unsupported sentences are dropped and explicitly disclosed; an answer with
# nothing substantiated degrades to the mandated refusal (no footer).
#
# Honest limits, NOT hidden: token PRESENCE cannot prove entailment. Two
# narrow polarity gates close the two clearest overlap traps (a claim whose
# own evidence negates it, and a number whose own clause is negated) and can
# only ever mark content UNSUPPORTED — they never invent support. Residual
# gap (documented, tested): a claim that splices tokens across unrelated
# passages cannot be detected without real entailment checking, and
# tightening to single-passage-sentence coverage would refuse legitimate
# combined summaries. Granularity is one sentence (a sentence mixing
# supported and fabricated tokens is dropped whole — the conservative side).
# Real entailment would need a separate verifier model/dependency; that
# trade-off is deliberately NOT taken (no framework, no new deps).
_EXEMPT_PHRASES = (
    # "What is missing" disclosures - they talk about coverage, not content:
    "do not address", "not covered", "not covered by", "cannot be answered",
    "does not state", "does not mention", "no mention", "not in scope",
)

_PARTIAL_DISCLOSURE = (
    "Note: the answer above is limited to the parts supported by the "
    "retrieved book passages; the rest was not found in the books."
)

# M4B verifier disclosures. Fixed strings — the verifier's own "reason"
# text NEVER reaches the user (it comes from an unvalidated model), so a
# compromised or confused verifier cannot inject content through reasons.
# Each note states exactly what happened; no resolution is ever invented.
_VERIFIER_UNAVAILABLE_NOTE = (
    "Note: semantic verification was unavailable for some statements, so "
    "they could not be confirmed and were removed."
)
_VERIFIER_CONFLICT_NOTE = (
    "Note: the retrieved passages conflict with part of this answer; the "
    "conflicting statements were removed and no resolution was invented."
)
_VERIFIER_REMOVED_NOTE = (
    "Note: statements the verifier could not confirm against the retrieved "
    "passages were removed."
)

# Polarity gate inputs. Cues are deliberately few and unambiguous: "only"
# and "prevents" are NOT negations, and including them would refuse valid
# answers such as the corpus's own honey sentence.
_NEGATION_CUES = frozenset({"not", "no", "never", "cannot", "can't",
                            "without", "neither", "nor", "none"})

# Clause boundaries used only by the polarity gates: sentence end, comma,
# semicolon, "and", "but". Splitting generously makes the gates fire less
# often (permissive on ambiguity) — they must never over-refuse.
# A comma preceded by a digit is a THOUSANDS separator ("20,000"), never a
# clause boundary — splitting there tore numbers apart and silently defeated
# the number-provenance gate. A decimal point is safe because the
# sentence-boundary rule requires whitespace after the punctuation.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?<!\d)[,;]\s*|\s+(?:and|but)\s+"
)


def _clauses(text: str) -> list:
    return [part.strip() for part in _CLAUSE_SPLIT_RE.split(text) if part and part.strip()]


def _has_negation(text: str) -> bool:
    return any(word in _NEGATION_CUES
               for word in re.findall(r"[a-z']+", text.lower()))


def _number_tokens(text: str) -> set:
    return {t for t in _support_tokens(text) if t.isdigit()}


def _polarity_conflict(sentence: str, evidence_clauses: list) -> bool:
    """True when the evidence that covers this sentence states the OPPOSITE
    polarity (conservative: the claim is then treated as unsupported).

    Two narrow rules, both requiring unambiguously matching evidence:
      A. some single clause covers EVERY content token of the sentence and
         disagrees on negation;
      B. some clause containing one of the sentence's NUMBERS disagrees on
         negation (catches a figure lifted from negated/changed evidence,
         e.g. conflicting retrieved passages).
    """
    sentence_neg = _has_negation(sentence)
    toks = _support_tokens(sentence)
    numbers = _number_tokens(sentence)
    for clause, clause_toks, clause_neg in evidence_clauses:
        if clause_neg != sentence_neg:
            # Rule A must use the SAME bounded family coverage as the
            # main gate: a raw subset check missed a need→require swap,
            # so a clause that NEGATED the claim failed to block the
            # affirmative wording of it (adversarial regression:
            # polarity across a family swap). Family-aware coverage can
            # only widen this gate toward UNSUPPORTED — it never blesses.
            if toks and _sentence_covered(toks, clause_toks):
                return True                     # rule A
            if numbers & clause_toks & _number_tokens(clause):
                return True                     # rule B
    return False


# 2026-10-25 fix cycle additions (confirmed GUI false-refusal causes,
# Phase 1) — explanation lives OUTSIDE the string because `.split()` would
# otherwise turn comment words into stop words (that mistake silently
# disabled the "not" requirement and broke a polarity regression):
#  - "like" / "content" / "emphasiz*" are non-assertive meta and
#    reporting words: an evidenced sentence was dropped solely because the
#    model's reporting verb or a filler was not book prose (A5 offline
#    repro: the cross-cutting GOVERN sentence dropped on "emphasiz" alone).
#  - interrogatives and demonstratives are QUESTION-structure words the
#    evidence never echoes; counting them against M5 coverage rejected
#    answerable questions (A6 offline repro: coverage 0.45 with "how/what"
#    among the missing tokens).
#  - "summariz*" joins "explain/describe" as a task verb (A6).
# Symmetric by contract: these forms are removed from answer sentences,
# evidence corpus and questions alike, so no token can gain support from an
# asymmetric rule — only non-assertive requirement tokens stop being
# required.
#
# 2026-10-27 (authorized W1): "main" / "essential" / "highlight" join the
# same non-assertive class above. Each was named VERBATIM by a GUI
# `gate_drop:` line as the ONLY missing token of an otherwise fully
# evidenced sentence (A1: "... with its main parts including the Core and
# functions GOVERN, MAP, MEASURE, and MANAGE."; A2's two dropped sentences
# on "essential" and "highlight"). Read-only re-check against the REAL
# top-3 chunks (.diag/w1_diagnosis/diagnose.py): with these three no longer
# required, both sentences become fully supported while every other content
# token in them — framework, govern, map, measur, trustworthy, risk, ... —
# stays required, and the abstention cases (chunk-size / bee-supers) are
# untouched because none of these three words appears in them.
#
# 2026-10-27 (authorized W2.1): "essentially" joins the same non-assertive
# class. It is the adverbial form of the already-listed "essential", and a
# model writing it as filler makes an OTHERWISE FULLY EVIDENCED sentence
# fail: the identical sentence with and without the word, against the same
# verbatim A2 page-12 chunk, returns mode="ok" and mode="refusal"
# respectively (scripts/test_books_grounding.py,
# test_essentially_adverb_does_not_refuse_an_evidenced_sentence — offline
# repro in .diag/w1_essentially_check.py). Symmetric, as everywhere above:
# the token stops being required on the answer, evidence and question sides
# alike, so it can never make an unsupported claim pass (pinned by
# test_essentially_does_not_ground_an_unsupported_claim). Only this one word
# was added — no speculative expansion of the list.
_STOP_TOKENS = frozenset("""
    the a an and or but nor of in on at to by with for as from into onto up
    down out over under above below about between within through during
    after before since until while because so than then thus hence however
    therefore also only even still too very just any all some each both few
    many much more most other another such same per via yes
    i we you he she it they them his her its their there here
    is are was were be been being am do does did done has have had having
    can could may might must shall should will would ought
    question questions answered answer answers reply responses according
    based provided given states stated says say saying mention mentions
    mentioned notes noted asks asked ask explain explains explained
    described describes adds added information info detail details topic
    topics document documents book books guide guides manual manuals
    passage passages context summaries requested request
    cover covers covered covering
    use uses used using
    anything something nothing everything
    like content emphasizes emphasize emphasized emphasizing
    summarize summarizes summarized summarizing summary
    what how why when where which who whom whose this these those
    main essential essentials highlight highlights highlighted highlighting
    essentially
""".split())


def _support_tokens(text: str) -> set:
    """Content tokens of a text: exact numbers, stop-filtered stemmed words.

    Normalization only (applied IDENTICALLY to answer sentences, evidence
    clauses and corpus text, so any pass is a relative effect of wording,
    never of asymmetric rules): plural "s" as before, plus light
    derivational strips (-ed / -ing when the remaining stem is long enough
    to stay a word, then a final "e"). Numbers are never stemmed.
    """
    tokens = set()
    for raw in re.findall(r"\$?\d[\d,.]*%?|[a-z]+", text.lower()):
        token = raw.replace(",", "").replace("$", "").rstrip("%.")
        if token.isdigit():
            if token:
                tokens.add(token)
        elif token not in _STOP_TOKENS and len(token) > 2:
            # "times" (the multiplication word) is exact-matched, never
            # collapsed to its duration homonym "time": otherwise
            # evidence that merely spends TIME satisfies the multiply
            # family for a MULTIPLIED claim (adversarial regression:
            # duration-time vs multiplied claim). Symmetric — both sides
            # of coverage normalize identically, as everywhere above.
            if (token.endswith("s") and len(token) > 3
                    and token != "times"
                    and not token.endswith(("ss", "us", "is"))):
                token = token[:-1]
            if token.endswith("ed") and len(token) >= 6:
                token = token[:-2]            # required -> requir
            elif token.endswith("ing") and len(token) >= 6:
                token = token[:-3]            # moving -> mov
            if token.endswith("e") and len(token) >= 5:
                token = token[:-1]            # determine -> determin; the
                # >= 5 floor deliberately spares 4-letter e-words (page,
                # move) so they stay identical to the unstemmed forms
                # _metadata_tokens adds verbatim ("page").
            tokens.add(token)
    return tokens


# --- bounded phrase/synonym coverage (verified false-refusal repair) ------
# Small, explicit families of domain words that mean the same thing in this
# corpus where the model paraphrases ("orbital mechanics" over source
# "orbital kinetics", "propulsion systems" over "propulsion engine",
# "calculated" over "determines/derives"). Each entry lists SURFACE FORMS
# AFTER the normalization above — including a member's OWN normalized form
# where the rule applies (change -> chang, engine -> engin, vehicle ->
# vehicl), so membership matches what `_support_tokens` actually emits.
#
# Contract (deliberately not a semantic matcher):
#   * a sentence token counts as covered without a literal hit ONLY when
#     another member of its own group literally appears in the evidence —
#     a group can never invent support, it can only relabel wording;
#   * NUMBERS are exempt from groups entirely (exact match stays mandatory);
#   * coverage remains per sentence — one supported sentence still never
#     carries an unrelated one, and the polarity gates still run before any
#     group is consulted;
#   * no embeddings, no LLM, no network: static data, constant-time lookup.
# Documented residual: a genuinely novel concept with no group and no
# literal token still refuses — that is the gate working, not a defect.
_PARAPHRASE_GROUPS = (
    frozenset({"mechanic", "kinetic", "dynamic", "motion"}),
    frozenset({"propulsion", "engine", "engin", "thrust", "rocket"}),
    frozenset({"system", "spacecraft", "vehicle", "vehicl"}),
    frozenset({"need", "requir", "necessit"}),
    frozenset({"calculat", "comput", "estim", "deriv", "determin"}),
    # Multiplication family ONLY — deliberately excludes "equal": an
    # equality word alone must never alias a multiplication claim
    # (adversarial regression: inverted "multiplied by" formula).
    # Entry is "times" — the multiplication word's POST-normalization
    # form — not the duration homonym "time": with the singular spelling
    # in the family, evidence that merely spends TIME admitted a
    # MULTIPLIED claim (adversarial regression: duration-time vs
    # multiplied claim).
    frozenset({"multipli", "multiply", "times"}),
    frozenset({"refer", "relat", "describ", "express"}),
    # Approved 2026-10-24 RAG fix (root-cause Investigation 3: false refusals
    # on the verified NIST AI-RMF GUI runs). In this domain's own vocabulary
    # a possible harm and a risk are the same operational concept — the AI
    # RMF defines its functions over "AI risks" while users routinely ask
    # about "AI harms" (observed reproducibly: a correct, in-corpus Map
    # answer paraphrasing "possible AI harms" was refused although the
    # retrieved passages say "risks"). Same contract as every family above:
    # relabel-only, one token at a time, numbers never covered, and the
    # polarity gates still run first (a negated "risks" clause continues to
    # cover-refuse an affirmative "harms" claim — actually FASTER now, since
    # rule A route through the family still requires full clause coverage).
    # Frozen by tests in scripts/test_books_grounding_adversarial.py (no
    # group invents support without a sibling member in the evidence).
    frozenset({"risk", "harm"}),

    # "delta" is the domain's own word for a change in velocity, so it
    # anchors this family in the verified corpus. Deliberately excludes
    # motion verbs ("moves"/"moving"): moving and CHANGING an orbit are
    # materially different claims (adversarial regression D1).
    # "modifi" is what `_support_tokens` actually emits for modifies /
    # modified / modifying — the former "modif" entry was DEAD (no input
    # produces it), which over-refused every natural "modify" inflection
    # against sibling-member evidence (adversarial regression:
    # test_modify_inflection_paraphrase_is_admitted).
    frozenset({"change", "chang", "alter", "adjust", "modifi", "modify",
               "vari", "delta"}),
)


def _sentence_covered(toks: set, corpus: set) -> bool:
    """Every content token of ONE sentence evidenced: literally, or via its
    bounded paraphrase group when a sibling member occurs in the evidence.
    Numeric tokens must always match literally (never group-covered)."""
    for tok in toks:
        if tok in corpus:
            continue
        if tok.isdigit():
            return False
        for group in _PARAPHRASE_GROUPS:
            if tok in group and not group.isdisjoint(corpus):
                break
        else:
            return False
    return True


# Inline citation shapes inside MODEL text. Each is validated against the
# retrieved nodes' own metadata (filename, page) or the retrieved passage
# text (tables/figures) — a mismatch means the model invented it and the
# sentence is treated as unsupported (dropped + disclosed, never blessed).
_FILENAME_RE = re.compile(r"\b[\w][\w\-]*\.(?:pdf|docx)\b", re.I)
_PAGE_REF_RE = re.compile(r"\bpage\s+(\d+)\b", re.I)
_FIGURE_REF_RE = re.compile(
    r"\b(?:table|figure|fig\.|chart|exhibit)\s+(\d+)\b", re.I
)


def _metadata_tokens(meta: dict) -> set:
    """Support tokens contributed by a node's OWN source metadata.

    LlamaIndex inlines non-excluded metadata (page_label, file_path) into the
    LLM context, so the model legitimately sees source identity — claims
    drawn from it are evidence-backed. The literal word "page" counts too
    (the metadata string literally carries it). Metadata values are added to
    the SUPPORT corpus only; never to the polarity clauses — metadata cannot
    affirm or negate anything.
    """
    if not isinstance(meta, dict):
        return set()
    toks = set()
    for key in ("file_name", "book_title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            toks |= _support_tokens(value.strip())
    has_page = False
    for key in ("page_label", "page_number"):
        value = meta.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            toks.add(str(value))
            has_page = True
        elif isinstance(value, str) and value.strip().isdigit():
            toks.add(value.strip())
            has_page = True
    if has_page:
        toks.add("page")
    return toks


def _inline_citation_error(sentence: str, files: set, pages: set,
                           corpus_numbers: set) -> str | None:
    """Why this sentence's inline citations are unverifiable, or None.

    filenames/pages must match the RETRIEVED nodes' metadata exactly; table
    and figure identifiers must appear in the retrieved passage text (no
    table/figure metadata exists, so the passage text is the only evidence).
    """
    for match in _FILENAME_RE.finditer(sentence):
        if match.group(0).lower() not in files:
            return "filename not among retrieved sources"
    for match in _PAGE_REF_RE.finditer(sentence):
        if match.group(1) not in pages:
            return "page not among retrieved pages"
    for match in _FIGURE_REF_RE.finditer(sentence):
        if match.group(1) not in corpus_numbers:
            return "table/figure number not in retrieved passages"
    return None


def _source_ref(meta: dict) -> str | None:
    """The reference string for one node's metadata, or None when the
    metadata cannot honestly identify a source (missing/non-string file
    identity, malformed page). Never invents a placeholder filename."""
    if not isinstance(meta, dict):
        return None
    title = meta.get("file_name") or meta.get("book_title")
    if not isinstance(title, str) or not title.strip():
        return None
    title = title.strip()
    page = meta.get("page_label")
    if page is None:
        page = meta.get("page_number")
    if isinstance(page, bool):
        page = None
    elif isinstance(page, float):
        page = int(page) if page.is_integer() else None
    if isinstance(page, int):
        page_s = str(page)
    elif isinstance(page, str):
        page_s = page.strip()
        # Display guard: only short identifier-like labels (arabic/roman
        # numerals) — junk metadata must not be rendered as a citation.
        if not (0 < len(page_s) <= 16
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\- ]*", page_s)):
            page_s = None
    else:
        page_s = None
    return f"{title}, page {page_s}" if page_s else title


def _reference_footer(source_nodes: list, used_indices: list) -> str:
    """Validated References footer from exactly the supporting nodes.

    Nodes without usable source metadata are counted and disclosed as
    "attribution unavailable" (the evidence exists, the provenance does not)
    instead of being cited under an invented name. Returns "" when there is
    nothing citable and nothing to disclose.
    """
    refs, seen, uncitable = [], set(), 0
    for idx in used_indices:
        meta = getattr(source_nodes[idx], "metadata", None)
        ref = _source_ref(meta if isinstance(meta, dict) else None)
        if ref is None:
            uncitable += 1
            continue
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    if not refs and not uncitable:
        return ""
    if refs:
        body = "; ".join(refs)
        if uncitable:
            body += (f"; attribution unavailable for {uncitable} "
                     "supporting passage(s)")
    else:
        body = (f"attribution unavailable for {uncitable} "
                "supporting passage(s)")
    return "\n\nReferences: " + body


def _claim_reference_footer(source_nodes: list, retained: list) -> str:
    """M4C claim-level References footer (verification-enabled mode only).

    One numbered line per retained claim, in answer order:

        [n] <ref>[; <ref>...] (verifier-supported: <claim text>)

    The parenthetical is deliberately ATTRIBUTED ("verifier-supported"),
    never stated as fact: it reports the bounded verifier's per-claim
    judgment (BOOKS_VERIFIER=local) — a heuristic/small-model verdict on
    the shown passages — NOT independently established semantic
    entailment. The refs on the line are likewise the candidate evidence
    the verifier cited (ids validated against that claim's own evidence),
    not proof of provenance.

    Each line cites ONLY the nodes attributed to that claim (the evidence
    the verifier confirmed for it, or the deterministic covering set when
    attribution was not narrowed), so a source supporting one claim is
    never presented as supporting another. Reference strings reuse the
    same validated `_source_ref` values as the answer-level footer:
    missing or unusable metadata yields the explicit "attribution
    unavailable" marker per claim — never an invented filename, page, or
    path. Returns "" when no claim has anything citable or disclosable.
    """
    lines = []
    for n, claim in enumerate(retained, start=1):
        refs, seen, uncitable = [], set(), 0
        for idx in claim["nodes"]:
            meta = getattr(source_nodes[idx], "metadata", None)
            ref = _source_ref(meta if isinstance(meta, dict) else None)
            if ref is None:
                uncitable += 1
                continue
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
        if not refs and not uncitable:
            continue          # nothing citable and nothing to disclose
        if refs:
            body = "; ".join(refs)
            if uncitable:
                body += (f"; attribution unavailable for {uncitable} "
                         "supporting passage(s)")
        else:
            body = (f"attribution unavailable for {uncitable} "
                    "supporting passage(s)")
        lines.append(f"[{n}] {body} (verifier-supported: {claim['text']})")
    if not lines:
        return ""
    return "\n\nReferences:\n" + "\n".join(lines)


def _grounded_repair(answer_text: str, source_nodes: list,
                     diagnostics: list | None = None):
    """Return (mode, kept_sentences, claims):
    ("refusal", [], []), ("ok", kept, claims), or ("partial", kept, claims).

    `claims` lists the RETAINED substantive sentences in order, each as
    {"text", "idx" (index into kept_sentences), "nodes"} where `nodes`
    holds the source_nodes indices that support that claim (direct cover,
    else contributing nodes of a cross-passage splice — see
    `_covering_nodes`). The References footer is built from the nodes of
    claims that survive verification — a retrieved-but-unused passage, or a
    claim later removed by the M4B verifier, is never cited.

    `diagnostics` (optional, caller-owned list): when given, one entry per
    DROPPED sentence with its disqualification kind and — for unsupported
    sentences — the exact missing tokens. Log-only: it never reaches the
    user and never alters the decision (2026-10-25 fix cycle, requirement:
    refusal diagnostics must identify the actual stage/reason without
    relying on an overwritten variable).
    """
    corpus = set()
    corpus_numbers = set()      # from passage TEXT only (tables/figures)
    evidence_clauses = []       # (clause, clause_tokens, clause_negated)
    node_toks = []              # per-node text + own-metadata tokens
    files, pages = set(), set()  # inline-citation allow-lists (retrieved)
    for node in source_nodes:
        text = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        text_toks = _support_tokens(text)
        all_toks = text_toks | _metadata_tokens(meta)
        node_toks.append(all_toks)
        corpus |= all_toks
        corpus_numbers |= {t for t in text_toks if t.isdigit()}
        for key in ("file_name", "book_title"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                files.add(value.strip().replace("\\", "/").rsplit("/", 1)[-1]
                          .lower())
        for key in ("page_label", "page_number"):
            value = meta.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                pages.add(str(value))
            elif isinstance(value, str) and 0 < len(value.strip()) <= 16:
                pages.add(value.strip())
        for clause in _clauses(text):
            evidence_clauses.append(
                (clause, _support_tokens(clause), _has_negation(clause))
            )

    kept, has_supported, removed = [], False, False
    claims = []       # supported claims: {"text", "idx" into kept, "nodes"}

    def _covering_nodes(toks: set) -> list:
        """Nodes that actually support this claim.

        Tier 1 (2026-10-25 citation-noise fix): a single node whose own
        tokens cover EVERY sentence token — literally, or via that token's
        bounded paraphrase-group sibling in THIS node. This is the honest
        attribution for a group-covered sentence (evidence: B2 GUI run —
        "thrust"/"changes" were group-covered by the orbital passage's
        "rocket"/"delta", but the old literal-subset check missed it and
        the intersection fallback also cited a NIST disclaimer chunk that
        shared one incidental token).

        Tier 2 (unchanged contract, tightened): a genuine cross-passage
        splice — a node is a contributing node only when it covers at
        least HALF of the sentence's tokens; a noise chunk sharing one
        incidental token never becomes a citation.
        """
        if not toks:
            return []

        def _node_covers(nt: set) -> bool:
            for t in toks:
                if t in nt:
                    continue
                if t.isdigit():
                    return False
                if not any(t in g and not g.isdisjoint(nt)
                           for g in _PARAPHRASE_GROUPS):
                    return False
            return True

        chosen = [i for i, nt in enumerate(node_toks) if _node_covers(nt)]
        if chosen:
            return chosen
        return [i for i, nt in enumerate(node_toks)
                if len(toks & nt) * 2 >= len(toks)]

    for sentence in (s.strip() for s in
                     re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()):
        low = sentence.lower()
        toks = set()
        detail = None               # why it was dropped (diagnostics only)
        if any(cue in low for cue in _INJECTION_CUES):
            kind = "injection"        # untrusted document text, not content
            detail = "injection cue"
        elif any(marker in low for marker in _REFUSAL_MARKERS):
            kind = "refusal"
            detail = "refusal-shaped sentence"
        elif any(p in low for p in _EXEMPT_PHRASES):
            kind = "disclosure"
        else:
            cite_err = _inline_citation_error(sentence, files, pages,
                                              corpus_numbers)
            if cite_err is not None:
                # The model wrote a filename/page/table the retrieved nodes
                # do not support: fabricated attribution — never kept.
                kind = "unsupported"
                detail = cite_err
            else:
                toks = _support_tokens(sentence)
                if not toks:
                    kind = "glue"      # no substantiable content either way
                elif _sentence_covered(toks, corpus):
                    # Token overlap is NOT proof: refuse to bless a claim
                    # whose own evidence states the opposite polarity.
                    if _polarity_conflict(sentence, evidence_clauses):
                        kind = "unsupported"
                        detail = "polarity conflict with own evidence"
                    else:
                        kind = "supported"
                else:
                    kind = "unsupported"
                    missing = sorted(
                        t for t in toks
                        if t not in corpus
                        and not (not t.isdigit() and any(
                            t in g and not g.isdisjoint(corpus)
                            for g in _PARAPHRASE_GROUPS)))
                    detail = "missing tokens: " + ", ".join(missing)
        if kind == "supported":
            has_supported = True
            claims.append({"text": sentence, "idx": len(kept),
                           "nodes": _covering_nodes(toks)})
            kept.append(sentence)
        elif kind in ("glue", "disclosure"):
            kept.append(sentence)
        else:      # unsupported / injection / refusal-shaped sentence
            removed = True       # (refusals are not claims to keep)
            if diagnostics is not None:
                diagnostics.append(
                    f"{kind} ({detail or 'not classified'}): "
                    f"{sentence[:160]}")

    if not has_supported:
        # Nothing in the answer is substantiated by the retrieved passages:
        # mandated refusal, no references footer.
        return "refusal", [], []
    if not removed:
        return "ok", kept, claims       # fully grounded: leave untouched
    return "partial", kept, claims


def _claim_evidence(claim: dict, source_nodes: list) -> list:
    """The EXACT passages assigned to this claim (never the whole corpus)
    with only the metadata needed for verification (never file_path),
    bounded by BOOKS_VERIFIER_EVIDENCE_MAX_CHARS in total. Evidence ids
    are derived from stable node indices, so the verifier can reference
    them and the caller can validate them."""
    # Facade seam: test_verifier tunes
    # books_agent.BOOKS_VERIFIER_EVIDENCE_MAX_CHARS at runtime.
    budget = _host().BOOKS_VERIFIER_EVIDENCE_MAX_CHARS
    evidence, used = [], 0
    for idx in claim["nodes"]:
        node = source_nodes[idx]
        meta = getattr(node, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        header = []
        title = meta.get("file_name") or meta.get("book_title")
        if isinstance(title, str) and title.strip():
            header.append(title.strip())
        page = meta.get("page_label")
        if page is None:
            page = meta.get("page_number")
        if isinstance(page, (str, int)) and not isinstance(page, bool):
            header.append(f"page {page}")
        text = str(getattr(node, "text", "") or "")
        block = (f"[source: {'; '.join(header)}]\n" if header else "") + text
        room = budget - used
        if room <= 0:
            break
        if len(block) > room:
            block = block[:room] + "\u2026[truncated]"
        evidence.append({"id": f"e{idx}", "text": block})
        used += len(block)
    return evidence


def _verify_claims(claims: list, source_nodes: list):
    """M4B bounded semantic verification. Returns (retained, notes, n_unavailable).

    Monotone by contract: claims are only ever REMOVED — never added,
    rewritten, or blessed beyond the deterministic gate that already ran.
    Every non-supported outcome is fail-closed (removed + disclosed), and
    the verifier's free-text reason never reaches the user: `notes` are
    fixed disclosure constants only.

    M4C claim-level attribution: a retained (supported) claim's node set
    is narrowed to the evidence ids the verifier itself cited. Those ids
    were validated against THIS claim's assigned evidence only (the
    strict parser rejects ids outside the per-claim whitelist), so one
    claim's evidence can never be attributed to another claim. If a
    scripted or unexpected result cites nothing inside the claim's own
    node set (impossible through the production parser), attribution
    falls back to the deterministic covering nodes — narrowing only, and
    never invented evidence.

    Sibling isolation / partial-response policy: each claim is judged
    ONLY by its own positional result. A result missing for a claim
    (verify_all returning fewer results than items — a partially valid
    multi-claim response) counts as UNAVAILABLE for exactly that claim:
    never a pass, never silent, and never a sibling's business. Surplus
    results are ignored. One malformed/unavailable/contradicted claim
    therefore removes exactly that claim; siblings keep their own
    verdicts, sources, and citations, and no surviving claim ever
    inherits evidence from a removed one.
    """
    items = [
        {"claim": claim["text"],
         "evidence": _claim_evidence(claim, source_nodes)}
        for claim in claims
    ]
    results = verifier.verify_all(items)
    retained, notes = [], []
    n_unavailable = n_contradicted = n_removed = 0
    for i, claim in enumerate(claims):
        result = results[i] if i < len(results) else None
        if result is None or result.get("status") != verifier.STATUS_VERIFIED:
            n_unavailable += 1
            continue
        verdict = result.get("verdict")
        if verdict == verifier.VERDICT_SUPPORTED:
            cited = {int(str(e)[1:]) for e in result.get("evidence_ids", [])
                     if str(e).startswith("e") and str(e)[1:].isdigit()}
            narrowed = [i for i in claim["nodes"] if i in cited]
            if narrowed:
                claim["nodes"] = narrowed
            retained.append(claim)
        elif verdict == verifier.VERDICT_CONTRADICTED:
            n_contradicted += 1
        else:                      # insufficient_evidence | uncertain
            n_removed += 1
    if n_unavailable:
        notes.append(_VERIFIER_UNAVAILABLE_NOTE)
    if n_contradicted:
        notes.append(_VERIFIER_CONFLICT_NOTE)
    if n_removed:
        notes.append(_VERIFIER_REMOVED_NOTE)
    return retained, notes, n_unavailable
