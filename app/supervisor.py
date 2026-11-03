"""Supervisor: decides routing only — it NEVER calls an agent node directly.

Phase 1: keyword-based routing (books-related words -> books, else -> research).
Phase 2: feasibility-style questions matched first (sub_routes chain).
Phase 3: a tight ERP keyword list routes to the erp agent. Checked BEFORE
books because ERP terms (invoices, inventory, ...) are specific while book
questions usually self-identify ("book", "manual", "chapter").

2026-10-24 (approved RAG fix): _BOOKS_RE additionally recognizes
document-deictic phrasings — "this/that/the document(s)", "the pdf" — so
ordinary document-grounded questions ("Using only this document, explain..."
— the exact strict wording that misrouted to web research in the verified GUI
failure) reach the books agent without artificial trigger words. Feasibility
and ERP patterns keep their higher precedence, so an ERP/feasibility question
containing "document" still routes there first.

2026-10-25 (approved fix cycle, Phase 2): two narrow, corpus-grounded
additions — no bare "AI"/"framework" terms, no exact-question rules:
  * _BOOKS_IDENTITY_RE: terms that identify the INDEXED corpus itself
    (the NIST AI RMF document family by name/acronym + "trustworthy AI",
    its distinctive §2/§3 vocabulary). Verified GUI failures A2/A3 —
    natural questions with no deictic wording — name these terms, and T6
    names "AI RMF", so all three become eligible without routing generic
    AI questions ("transformers in modern AI", "AI art") to books.
  * _CURRENT_INFO_RE: checked BEFORE the books patterns (after
    feasibility/ERP) so genuine current-information requests keep the
    research path even when they also name a corpus document
    ("the latest news about the NIST AI framework" -> research).
Both are frozen by scripts/test_phase1_routing.py, which also pins that
keyword-free research queries stay on research.

Phase 6: optional LLM-assisted classification behind ROUTING_MODE=llm
(config; default "keyword" keeps byte-identical behavior). The LLM label is
restricted to the four valid routing classes; any LLM error, malformed
output, or unknown label falls back to the keyword result — default mode
never depends on a working AI provider. The function signature and return
contract stay identical so nothing downstream changes.

Phase 7: additionally marks genuinely multi-objective requests as
"complex" (new key: \'complex\': bool — only ever True, never written False
so pre-Phase-7 test states are untouched). The planner (app/planner.py)
decides whether to actually run via PLANNER_MODE; the supervisor only
detects the need so the graph can take one conditional hop.

2026-10-25 (orchestrated knowledge fallback, additional phase):
  * _LOCAL_ONLY_RE: explicit no-internet / document-scoped source
    constraints ("using only this document", "do not search the web") —
    checked after feasibility/ERP but BEFORE every source cue, so such a
    request can never reach research directly, and the fallback decision
    (fallback_decision below) never silently converts it into an
    internet-sourced answer.
  * _EXPLICIT_WEB_RE: explicit web-research requests ("Google the four
    functions of the AI RMF") checked with the current-information cues so
    they reach research even when they name a corpus document.
  * fallback_decision(): the orchestrator's books->research decision,
    reading the structured state["books_refusal"] diagnosis (never a score
    or the refusal sentence alone) — called by app/graph._after_books.
"""

import re

from app.config import BOOKS_WEB_FALLBACK, ROUTING_MODE
from app.learning import signals as learning_signals
from app.planner import needs_planning
from app.state import AgentState

_ROUTING_PROMPT = """Classify the user question into exactly one of these labels:
books: the question can be answered from the user's own book library
research: the question needs general web research
feasibility: the question asks for a business feasibility study / go-no-go analysis
erp: the question is about ERP business data (invoices, inventory, orders, customers)

Answer with ONLY the label (one word).

Question: {question}

Label:"""

_VALID_LLM_ROUTES = {"books", "research", "feasibility", "erp"}

_BOOKS_RE = re.compile(
    r"\b(books?|chapter|pages?|page \d|textbook|handbook|manual|guide|"
    r"author|bibliography|section|orbital|orbit|spacecraft|speed|altitude|"
    r"bee|beekeeping|urban|hive|queen|drone|nectar|pollen|swarm|"
    # Possessive scoping (2026-10-28, Priority A): "my/our/your documents"
    # is the same local-source request as "this/the document" — without it
    # "Find the information in my documents" missed every rule and fell
    # through to the old research default. Narrow on purpose: document
    # nouns only, never "my report"/"my file list".
    r"(?:knowledge\s*base|"
    # 2026-10-30: a quantifier/qualifier between the article and the noun is
    # still the same referent — "compare the TWO documents", "my UPLOADED
    # files" — and without this the whole request fell through to general.
    r"(?:this|that|the|my|our|your)(?:\s+(?:two|three|four|five|both|several|"
    r"all|other|uploaded|stored|indexed|saved))? documents?|"
    r"(?:these|those) documents?|"
    # 2026-10-30 (end-to-end trace): "my STORED documents", "the uploaded
    # files", "my indexed pdfs" are the same local-source request with one
    # qualifier in between, and every one of them fell through to
    # `unsupported`. Qualifier + document noun only — never "my saved
    # settings" or a bare "stored".
    r"(?:my|our|your|the|these|those) (?:stored|uploaded|indexed|saved) "
    r"(?:documents?|files?|pdfs?|papers?|reports?|"
    r"books?|manuals?|guides?)|"
    r"(?:my|our|your) (?:pdf|book|manual|guide)|the pdf))\b",
    re.IGNORECASE,
)

# Corpus-identity anchors (2026-10-25): terms that identify documents the
# collection actually contains — the NIST AI Risk Management Framework by
# full name or its own acronym ("AI RMF", "RMF"), plus the distinctive
# §2/§3 vocabulary "trustworthy AI" and the corpus owner "NIST". Deliberately
# NOT bare "AI" or "framework": "transformers in modern AI" and "which web
# framework should I learn" must not become document questions (pinned by
# tests; since 2026-10-28 they take the general route).
_BOOKS_IDENTITY_RE = re.compile(
    r"\b(?:ai\s+)?risk management framework\b|"
    r"\b(?:ai\s+)?rmf\b|"
    r"\btrustworthy ai\b|"
    r"\bnist\b",
    re.IGNORECASE,
)

# Current-information cues (2026-10-25): genuine "what's happening now"
# requests keep the research path even when they name a corpus document.
# Deliberately narrow: no "current" ("the current version of the manual"
# is a document question), no bare years, no "who is" ("who is the author
# of this book" is a document question).
_CURRENT_INFO_RE = re.compile(
    r"\b(latest|breaking|today|this\s+week|this\s+month|overnight|"
    r"news|weather|stock\s+price|exchange\s+rate)\b",
    re.IGNORECASE,
)

# --- Orchestrated knowledge fallback (2026-10-25 additional phase) --------
# Document nouns a question can be scoped to.
_DOC_NOUN = r"(?:document|pdf|book|manual|guide|file|text|report|paper)s?"

# A document noun with an optional provenance qualifier. "Using only the
# local documents" and "in the indexed documents" scope a request to the
# local corpus exactly as "the documents" does (2026-10-31 live audit:
# without this, case B2 asked for local-only and the run still proposed an
# outbound web search).
_DOC_NOUN_QUAL = rf"(?:(?:local|indexed|uploaded|stored)\s+)?{_DOC_NOUN}"

# EXPLICIT local-only / no-internet constraints. Checked after feasibility
# and ERP (their spec-pinned precedence) but BEFORE every source cue: the
# user forbade external research or scoped the request to one local
# document, so books is the only permitted source.
_LOCAL_ONLY_RE = re.compile(
    rf"\b(?:"
    rf"(?:using|with|from|by|answer(?:ing)?|respond(?:ing)?)\s+only\s+"
    rf"(?:this|that|the|these|those|your|our)\s+{_DOC_NOUN_QUAL}|"
    rf"only\s+(?:from\s+|using\s+|with\s+|in\s+|by\s+)?(?:this|that|the)\s+{_DOC_NOUN_QUAL}|"
    rf"(?:this|that|the)\s+{_DOC_NOUN_QUAL}\s+only|"
    rf"without\s+(?:using\s+)?(?:any\s+)?(?:the\s+)?"
    rf"(?:internet|web|online|search(?:ing)?)|"
    rf"(?:no|not)\s+(?:any\s+)?(?:web|internet|online)\s+search|"
    rf"(?:do\s+not|don't|never|avoid)\s+(?:use\s+)?(?:the\s+)?"
    rf"(?:internet|web|search(?:ing)?|online)|"
    rf"offline|"
    rf"from\s+the\s+books?\s+only"
    rf")\b",
    re.IGNORECASE,
)

# Document-DEICTIC scoping — used by fallback_decision (routing already
# sends these to books via _BOOKS_RE). The question asks what THIS local
# document says, so a web answer cannot satisfy it: fallback stays blocked
# and the books abstention survives (GUI RAG test T5).
_DOC_DEICTIC_RE = re.compile(
    rf"\b(?:this|that|these|those|the)\s+{_DOC_NOUN_QUAL}\b",
    re.IGNORECASE)

# Explicit web-research requests, checked with the current-information
# cues. Deliberately request-shaped forms only: a document question ABOUT
# web search ("what does the manual say about searching the web") must stay
# on books, so progressive forms ("searching the web") are excluded.
_EXPLICIT_WEB_RE = re.compile(
    r"\b(?:"
    r"search\s+(?:the\s+)?(?:web|internet)|"
    r"google|"
    r"look\s+(?:it\s+)?up\s+on(?:line)?|"
    r"browse\s+the\s+(?:web|internet)|"
    r"web\s+search"
    r")\b",
    re.IGNORECASE,
)

# --- Priority A (2026-10-28): the rest of the routing policy -----------
# Until now EVERYTHING that missed the patterns above fell through to the
# research DEFAULT, so "What is a vector database?" cost a live web search.
# The three deterministic checks below split that fall-through into:
#   * unsupported  — a request we have no tool for, a request for memory we
#     deliberately do not keep, or an underspecified fragment (-> clarify);
#   * general      — a question answerable from model knowledge, frictionless.
# They run AFTER every source cue (feasibility/ERP/local-only/current-info/
# books), so no substantive routed request is ever re-classified by them.

# Requests for conversation memory we do not have. Checked FIRST among the
# new rules: "What did I tell you earlier?" starts with "what" and must not
# reach the general model, which could only hallucinate a history. Deliberately
# past-tense only — "What do I tell my client about pricing?" is a real
# question and stays general.
_NO_HISTORY_RE = re.compile(
    r"\b(?:"
    r"what (?:did|has) (?:i|we) (?:tell|say|ask|mention|discuss)|"
    r"what did we (?:tell|say|ask|mention|discuss)|"
    r"do you remember (?:what|when|my|our)|"
    r"(?:my|our) (?:previous|earlier|last) (?:conversation|chat|message)|"
    r"as i (?:said|told|mentioned) (?:before|earlier)|"
    r"i (?:already )?(?:told|said|asked) you"
    r")\b",
    re.IGNORECASE,
)

# Authoring answered IN THE REPLY (2026-10-29 pipeline trace):
# "Write a Python function that parses a CSV file" starts with "write", so the
# ACTION rule below refused it with "I can't do that" — although producing
# that code needs no tool at all, just model knowledge. Code/prose authoring
# is therefore classified as GENERAL, while anything that would create or
# change something on the user's machine (a file, the repo, an install, a
# deploy) stays UNSUPPORTED (`_SYSTEM_ACTION_RE`).
_CODE_ARTIFACT = (
    r"(?:functions?|scripts?|programs?|code|snippets?|class|classes|"
    r"method|methods|module|modules|quer(?:y|ies)|regex|expressions?|"
    r"algorithm|loop|loops|tests?|unit\s+tests?|commands?|endpoint|"
    r"api\s+call|component|components|query\s+parameters)"
)
_CODING_LANGUAGE = (
    r"(?:python|javascript|typescript|java|c\+\+|c#|golang|rust|sql|"
    r"bash|shell|powershell|html|css|react|pandas|numpy|django|flask)"
)
_TEXT_AUTHORING_RE = re.compile(
    r"^\s*(?:write|create|draft|generate|produce|compose|make|build|"
    r"give\s+me|show\s+me)\b[\s\S]*\b(?:"
    + _CODE_ARTIFACT + r"|" + _CODING_LANGUAGE +
    r")\b",
    re.IGNORECASE,
)

# The authoring exception above never covers an operation on the user's
# machine: those need a tool this assistant does not have.
_SYSTEM_ACTION_RE = re.compile(
    r"\b(?:save|write)\s+(?:it\s+)?(?:to|into)\s+(?:a\s+)?"
    r"(?:file|disk|my\b|the\s+(?:repo|repository|project|workspace))|"
    r"\b(?:to|into|on)\s+(?:a\s+|the\s+|my\s+)?"
    r"(?:file|disk|filesystem|hard\s+drive)\b|"
    r"\bon\s+(?:my|the)\s+(?:computer|machine|desktop)\b|"
    r"\bin\s+(?:my|the)\s+(?:repo|repository|project|workspace|codebase)\b|"
    r"\brun\s+(?:it|the\s+script|the\s+command)\b|"
    r"\b(?:install|deploy|commit|push)\b",
    re.IGNORECASE,
)

# --- Workspace coding (2026-10-30) ----------------------------------------
# A request that names a SOURCE FILE (or the workspace/repo itself) is a
# workspace task, not a general question and not an unsupported action. It is
# checked AFTER every source cue, so "search the web for docker-compose.yml"
# stays research and a books question stays books; and it only applies when a
# workspace is configured, so the default installation routes exactly as
# before (CODING_WORKSPACE_ROOT empty -> capability disabled).
_CODING_INTENT_RE = re.compile(
    r"\b(?:workspace|repository|repo|codebase)\b"
    r"|\b[\w./\\-]+\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|json|md|css|scss|"
    r"html|yml|yaml|toml|sql|sh|ps1|rb|go|rs|java|cs)\b",
    re.IGNORECASE,
)


# --- Dataset analysis (2026-10-30) ---------------------------------------
# A request that NAMES a dataset file is a data-analysis task: the analyst
# needs an actual file to compute over, so "what is a dataset?" (no file) is
# untouched by this rule. Checked after every source cue, like coding.
#
# 2026-10-30 (end-to-end trace): "Analyze the dataset and tell me which
# products performed best" names NO file and was answered by the GENERAL
# route from model knowledge — i.e. it invented an analysis. A definite
# reference to the user's own data ("the dataset", "my sales data") is a
# data request whether or not a workspace is configured: the data node is
# the only place allowed to say "not enabled" or "name the file", and both
# of those are honest. A bare "the data" is NOT enough — it appears in
# ordinary prose ("how is the data stored?") — so a plain data noun needs
# an analysis verb in the same request, in either order.
_ANALYSIS_VERB = (
    r"(?:analy[sz]e\w*|calculat\w*|comput\w*|summari[sz]\w*|"
    r"break\s+down|trend\w*|total\w*|average\w*|correlat\w*|"
    r"performed\s+best|best[-\s]performing|top[-\s]\w+)"
)
# One optional qualifier is allowed between the determiner and the noun:
# "my sales data", "the quarterly spreadsheet" are the same definite
# reference. Bare "the data" still needs an analysis verb (see below).
_DATA_REF = (r"(?:the|my|our|this|that)\s+(?:[\w-]+\s+)?"
             r"(?:data|spreadsheet)")
_DATA_INTENT_RE = re.compile(
    r"\b[\w][\w./\\-]*\.(?:csv|tsv)\b"
    r"|\b(?:the|my|our|this|that)\s+(?:dataset|data\s*set)\b"
    rf"|\b{_ANALYSIS_VERB}\b[\s\S]{{0,80}}?\b{_DATA_REF}\b"
    rf"|\b{_DATA_REF}\b[\s\S]{{0,80}}?\b{_ANALYSIS_VERB}\b",
    re.IGNORECASE,
)


# --- Explicit research request (2026-10-30) -------------------------------
# "Research three suitable open-source models, compare their hardware
# requirements, and recommend one" is the textbook multi-step request, and it
# was answered with "I'm not sure what you'd like me to do": the verb
# "research" is not in _CURRENT_INFO_RE (which wants latest/news/today) and
# _ACTION_REQUEST_RE only lists create/send/deploy-style verbs, so the request
# fell through to the unsupported shape rules. Only REQUEST-SHAPED forms
# qualify, and "compare" is deliberately absent: "compare the two documents
# in my knowledge base" is a books question, which is checked first. Checked
# AFTER books so "research what the manual says" stays local.
_EXPLICIT_RESEARCH_RE = re.compile(
    r"^\s*(?:"
    r"research|investigate|look\s+into|find\s+out\s+about|"
    r"gather\s+(?:information|data)\s+(?:on|about|regarding)|"
    r"do\s+(?:some\s+)?research\s+(?:on|about|into)|"
    r"survey"
    r")\b",
    re.IGNORECASE,
)

# --- Contextual follow-up (2026-10-30) ------------------------------------
# A request that refers BACK to earlier work instead of naming its subject.
# Without this rule "What about the one you recommended?" reached the GENERAL
# route, which answered it from model knowledge — inventing a recommendation
# the user was never given. Detection only decides the route; the `followup`
# node grounds the answer in the RECORDED execution (app/followup.py) and
# says plainly when there is none. Deliberately anaphoric forms only: a
# question that names its own subject is a new request, not a follow-up.
_FOLLOWUP_RE = re.compile(
    r"\b(?:the|that|this)\s+one\b"
    r"|\b(?:the|that)\s+(?:option|recommendation|suggestion|answer|result|"
    r"plan|study|list|comparison)\s+(?:you|above|before|earlier)\b"
    r"|\byour\s+(?:recommendation|suggestion|previous\s+answer|"
    r"last\s+answer|earlier\s+answer)\b"
    # 2026-10-31 live audit (case C4): "Can you include those in the previous
    # answer?" names nothing of its own — it points at the answer above. It
    # reached the GENERAL route and produced a confident claim about an
    # answer that was never inspected. A bare reference to the previous
    # answer, or an instruction to fold something unnamed INTO it, is a
    # follow-up and is grounded in the recorded execution (app/followup.py).
    r"|\b(?:the|your)\s+(?:previous|last|earlier)\s+answer\b"
    r"|\b(?:include|add|insert|apply|fold|expand|extend|flesh\s+out)\s+"
    r"(?:those|these|them|that)\b"
    r"|\b(?:those|these|them)\s+(?:in|into|to)\s+(?:the|your)\s+"
    r"(?:previous|last|earlier|above|prior)\b"
    r"|\b(?:which|what)\s+of\s+those\b"
    r"|\bwhat\s+about\s+(?:it|them|those|that|the\s+first|the\s+second)\b"
    r"|\bthe\s+(?:first|second|third|last)\s+(?:one|option|model|approach)\b"
    r"|\bsame\s+(?:thing|model|option|approach)\b",
    re.IGNORECASE,
)


def workspace_coding_enabled() -> bool:
    """Whether a coding workspace is configured (read at call time)."""
    from app.config import CODING_WORKSPACE_ROOT

    return bool(str(CODING_WORKSPACE_ROOT or "").strip())


# Imperative requests for capabilities that do not exist (no report writer,
# no mailer, no file system tools). Anchored at the start and followed by a
# word boundary so "Create a report" matches while "creator of the manual"
# (already a books question above) never does.
_ACTION_REQUEST_RE = re.compile(
    r"^\s*(?:"
    r"create|make|build|write|draft|generate|produce|compose|"
    r"send|email|mail|schedule|purchase|buy|"
    r"delete|remove|erase|rename|move|copy|upload|download|install|"
    r"compile|execute|deploy|publish|fix|repair|troubleshoot|"
    r"prepare|design|draw|paint|edit|format|convert|translate|calculate"
    r")(?:\s|$)",
    re.IGNORECASE,
)

# Bare "do it"-style imperatives. Narrow on purpose: an open "do" prefix
# would swallow genuine questions ("Do you know whether ... is safe?").
_BARE_IMPERATIVE_RE = re.compile(
    r"^\s*(?:do|handle|deal with|take care of)\s+"
    r"(?:it|this|that|everything|the (?:rest|needful)|"
    # 2026-10-31 live audit (case A5): "do the thing we discussed" matched
    # the QUESTION shape on its leading "do" and was answered as a general
    # question ("the question is unclear and unclear"). It is an imperative
    # with no subject, so the honest answer is to ask what was meant.
    r"the (?:thing|stuff|task|job|work)(?: (?:we|you) discussed)?|"
    r"what (?:we|you) discussed)\b",
    re.IGNORECASE,
)

# Shapes of a question answerable from model knowledge: a question word or
# auxiliary at the start, a known explanatory verb, or a trailing question
# mark. Anything that matches none of these AND none of the rules above is
# underspecified -> keyword_route asks for detail instead of guessing.
_QUESTION_SHAPE_RE = re.compile(
    r"^\s*(?:"
    r"what|why|how|who|whom|whose|when|where|which|"
    r"is|are|was|were|am|does|did|do|can|could|would|should|will|shall|"
    r"has|have|had|"
    r"explain|describe|define|compare|contrast|summarize|summarise|list|"
    r"give me|tell me|show me|discuss|clarify|elaborate|review|"
    r"analyse|analyze|walk me through"
    r")\b|\?\s*$",
    re.IGNORECASE,
)

_FEASIBILITY_RE = re.compile(
    r"\b(feasibility|feasible|business case|go/no-go|go no go|"
    r"market study|market analysis|risk assessment|venture)\b",
    re.IGNORECASE,
)

# --- Conversational openers (2026-10-27 greeting-misrouting fix) ----------
# Root cause of execution 3fc3bc8460c9: "hello" matched none of the
# patterns above and fell through to keyword_route's research DEFAULT, so a
# greeting cost a live web search (8 results, 3 page crawls) and produced an
# irrelevant sourced answer that QA then accepted.
#
# Deliberately FULL-STRING matching: the normalized input must EQUAL one of
# these phrases (lowercased, punctuation stripped, whitespace collapsed).
# Any extra content — "hello, what does the NIST AI RMF say..." — is NOT a
# greeting and keeps the existing precedence chain below, so no broad
# keyword rule can ever mis-classify a substantive question.
_CONVERSATIONAL_OPENER_CHARS = re.compile(r"[^\w\s']+")

_GREETING_OPENERS = frozenset({
    "hello", "hi", "hey", "hiya", "howdy", "yo", "greetings",
    "good morning", "good afternoon", "good evening", "good day",
    "hello there", "hi there", "hey there",
    "what's up", "whats up",
})

# Simple social exchanges (2026-10-28, Priority A): a wellbeing question is
# conversation, not a research query. Same full-string discipline as the
# greeting set — "how are you after reading chapter 3?" is NOT one of these.
_WELLBEING_OPENERS = frozenset({
    "how are you", "how are you doing", "how is it going", "how's it going",
})

_PARTING_OPENERS = frozenset({
    "bye", "goodbye", "see you", "see you later",
    "thanks", "thank you", "thanks a lot",
    "ok", "okay", "yes", "no",
    # acknowledgements (2026-10-28): short confirmations are conversation.
    "got it", "sounds good", "no problem",
})

_GREETING_REPLY = (
    "Hi! I'm Agentica. Ask me anything and I'll route it to the right "
    "source: your indexed documents (Books), web research, ERP business "
    "data, or a feasibility study."
)
_WELLBEING_REPLY = (
    "Doing well, thanks — I'm Agentica, a local assistant. I don't keep "
    "conversation history between runs, but I can answer questions from "
    "your indexed documents, research the web when you ask, look up ERP "
    "business data, or run a feasibility study."
)
_PARTING_REPLY = (
    "Noted. I don't keep conversation history between runs, so send your "
    "request as a new message and I'll route it to the right source."
)

# Acknowledgements (2026-10-29): social exchanges that are not bare openers —
# "thanks, that helps", "great, thank you", "cheers, got it". Without this,
# such a reply fell through to `unsupported` and was answered with "I'm not
# sure what you'd like me to do", which is plainly wrong. The rule is
# deliberately ALL-words deterministic: the normalized input must be short
# and contain ONLY acknowledgement filler plus at least one anchor word, so
# "thanks, how do I sort a list?" (real question) can never match.
_ACK_ANCHORS = frozenset({
    "thanks", "thank", "thx", "cheers", "appreciate", "appreciated",
    "great", "perfect", "awesome", "brilliant", "excellent", "nice",
    "cool", "got", "sounds", "helps", "helped", "works", "worked",
})
_ACK_FILLER = frozenset({
    "it", "that", "this", "you", "so", "much", "very", "a", "lot",
    "all", "good", "well", "for", "the", "help", "now", "then",
    "yes", "no", "ok", "okay", "and", "again", "really", "that's",
})
_ACK_MAX_WORDS = 6
_ACK_REPLY = (
    "You're welcome. I don't keep conversation history between runs, so "
    "include whatever I need in each new message."
)


def _is_acknowledgement(normalized: str) -> bool:
    """True for a short social acknowledgement (no question content).

    Every word must be acknowledgement filler and at least one must be an
    anchor; longer or substantive text never matches.
    """
    words = normalized.split()
    if not words or len(words) > _ACK_MAX_WORDS:
        return False
    if any(word not in _ACK_ANCHORS and word not in _ACK_FILLER
           for word in words):
        return False
    return any(word in _ACK_ANCHORS for word in words)


def _normalized_opener(question: str) -> str:
    """Lowercased, punctuation-stripped, whitespace-collapsed input."""
    cleaned = _CONVERSATIONAL_OPENER_CHARS.sub(" ", str(question or "")).lower()
    return " ".join(cleaned.split())


_WELLBEING_TRAILERS = (" today", " now", " these days")


def _is_wellbeing(opener: str) -> bool:
    """True for a wellbeing question, including after a greeting.

    2026-10-30 (end-to-end trace): "Hi, how are you?" is ONE social
    exchange, but it matched neither the greeting set (extra words) nor the
    wellbeing set (extra greeting), so it reached the general route and
    spent a model call on a hello. The rule stays full-string: a greeting
    token (one or two words, e.g. "good morning") followed by an EXACT
    wellbeing phrase, optionally followed by "today"/"now". Any other
    content — "how are you after reading chapter 3?" — still fails.
    """
    text = opener
    for trailer in _WELLBEING_TRAILERS:
        if text.endswith(trailer):
            text = text[: -len(trailer)].strip()
            break
    if text in _WELLBEING_OPENERS:
        return True
    words = text.split()
    for head_len in (1, 2):
        if len(words) <= head_len:
            continue
        head = " ".join(words[:head_len])
        tail = " ".join(words[head_len:])
        if head in _GREETING_OPENERS and tail in _WELLBEING_OPENERS:
            return True
    return False


def is_conversational(question: str) -> bool:
    """True ONLY for a bare greeting / conversational opener (no LLM).

    Deterministic full-string match, the greeting+wellbeing combination
    rule (`_is_wellbeing`), plus the short-acknowledgement rule
    (`_is_acknowledgement`) — see the block comment above. Used by
    keyword_route, by supervisor() (so ROUTING_MODE=llm can never route a
    greeting to a model either), and by the QA gate, which must not accept
    a sourced agent answer as a reply to a greeting.
    """
    opener = _normalized_opener(question)
    return (opener in _GREETING_OPENERS or opener in _PARTING_OPENERS
            or _is_wellbeing(opener) or _is_acknowledgement(opener))


def conversational_reply(question: str) -> str:
    """The fixed, source-free reply for a conversational opener.
    No LLM, no retrieval, no tools, no URLs — provenance stays empty."""
    opener = _normalized_opener(question)
    if _is_wellbeing(opener):
        return _WELLBEING_REPLY
    if opener in _GREETING_OPENERS:
        return _GREETING_REPLY
    if opener in _PARTING_OPENERS:
        return _PARTING_REPLY
    return _ACK_REPLY

_ERP_RE = re.compile(
    r"\b(erp|inventory|stock|invoices?|purchase orders?|sales orders?|"
    r"customer records?|vendor records?|procurement)\b",
    re.IGNORECASE,
)


def _llm_route(question: str) -> str | None:
    """Ask the LLM for a routing label; return None on ANY failure.

    Failures are swallowed here by design (fail-soft): the caller falls back
    to keyword routing, so a missing/broken/billing-dead provider can never
    crash routing.
    """
    try:
        from app.llm_client import get_llm, llm_text

        response = get_llm(fast=True).invoke(
            _ROUTING_PROMPT.format(question=question)
        )
        label = llm_text(response).strip().lower()
    except Exception:
        return None
    return label if label in _VALID_LLM_ROUTES else None


def keyword_route(state: AgentState) -> dict:
    """Deterministic keyword routing — the ONE source of routing policy.

    Precedence (unchanged through the books cue, Phases 1–5 + the
    2026-10-24/25 RAG fixes): conversational > feasibility > ERP >
    explicit local-only > current-info / explicit web > books.

    2026-10-28 (Priority A): the old `return {"route": "research"}`
    fall-through is gone. What neither a source cue nor a document keyword
    matches is now classified deterministically as unsupported (no such
    tool / no such memory / too vague) or general (model-knowledge answer),
    so a missed keyword rule can never trigger an external web search again.
    """
    question = state.get("question", "")
    # Bare conversational openers never reach an agent: answered locally,
    # no Books, no Web Research (2026-10-27 greeting-misrouting fix).
    if is_conversational(question):
        return {"route": "conversational"}
    if _FEASIBILITY_RE.search(question):
        # Spec (feasibility_agent): Supervisor sets sub_routes=["research", "books"]
        # first so both agents produce prior answers in state.
        return {"route": "feasibility", "sub_routes": ["research", "books"]}
    if _ERP_RE.search(question):
        return {"route": "erp"}
    # Explicit local-only / no-internet constraints outrank every source
    # cue below: books is the only permitted source for such a request.
    if _LOCAL_ONLY_RE.search(question):
        return {"route": "books"}
    # 2026-10-25: current-information cues outrank corpus anchors so
    # "latest news about the NIST AI framework" stays on research, while
    # plain corpus-identity questions fall through to books below.
    # Explicit web requests ("Google the four functions of the AI RMF")
    # join this precedence level for the same reason.
    if _CURRENT_INFO_RE.search(question) or _EXPLICIT_WEB_RE.search(question):
        return {"route": "research"}
    if _BOOKS_RE.search(question) or _BOOKS_IDENTITY_RE.search(question):
        return {"route": "books"}
    if _EXPLICIT_RESEARCH_RE.search(question):
        # 2026-10-30: an explicit "research X" request is a research task
        # even when it is long and multi-objective. Checked after books (a
        # document question wins) and before the shape rules below, which
        # used to answer it with "I'm not sure what you'd like me to do".
        return {"route": "research"}
    if _DATA_INTENT_RE.search(question):
        # Dataset analysis: read-only and workspace-confined. It must be
        # checked before the coding rule so a CSV question never looks like a
        # source-file edit. 2026-10-30: NOT gated on a configured workspace
        # any more — the data node is the single place that decides whether
        # it can run, and "no workspace is configured" from it is honest
        # while an invented analysis from the general route is not.
        return {"route": "data"}
    if workspace_coding_enabled() and _CODING_INTENT_RE.search(question):
        # The capability itself enforces the path boundary; routing only
        # decides that this is a workspace task.
        return {"route": "coding"}
    # --- 2026-10-30: feedback / correction -------------------------------
    # A message that CORRECTS earlier work ("that's wrong", "not enough",
    # "you misunderstood", "next time ...") goes to the feedback route: it
    # revises the current answer and records a lesson. Checked before the
    # follow-up rule because a correction must be learned from, not merely
    # answered; both sit after every source cue, so "the manual is wrong about
    # X?" still belongs to the capability its subject names. Detection is
    # deterministic and question-shaped messages are excluded, so an ordinary
    # question can never be mistaken for a correction.
    if learning_signals.is_feedback(question):
        return {"route": "feedback"}
    # --- 2026-10-30: contextual follow-up --------------------------------
    # A bare reference to earlier work goes to the follow-up route, which
    # answers from the recorded execution (never from model knowledge). It is
    # checked here — after every substantive source cue — so a question that
    # names its own subject is still routed on that subject, and it fails
    # honestly when no earlier result was recorded.
    if _FOLLOWUP_RE.search(question):
        return {"route": "followup"}
    # --- 2026-10-28 Priority A: no silent research fall-through ----------
    if _NO_HISTORY_RE.search(question):
        # We keep no conversation history; answer honestly instead of
        # letting a model invent one ("unsupported" = capability we lack).
        return {"route": "unsupported"}
    if (_TEXT_AUTHORING_RE.search(question)
            and not _SYSTEM_ACTION_RE.search(question)):
        # "Write a Python function ..." is answered in the reply, not
        # performed on the machine (see _TEXT_AUTHORING_RE).
        return {"route": "general"}
    # 2026-10-30: an ACTION on a source file ("fix the bug in app/graph.py")
    # when no workspace is configured used to answer "I have no tool for it",
    # which is untrue — the capability exists and is switched off. Route it to
    # the coding node, whose one job in that state is to say exactly that.
    # Questions about code ("how do I parse a .py file?") never match the
    # action rule, so they keep the general route.
    if _CODING_INTENT_RE.search(question) and _ACTION_REQUEST_RE.search(question):
        return {"route": "coding"}
    if _ACTION_REQUEST_RE.search(question) or _BARE_IMPERATIVE_RE.search(question):
        return {"route": "unsupported"}
    if _QUESTION_SHAPE_RE.search(question):
        return {"route": "general"}
    # Underspecified fragment ("Do it", "beekeeping stuff"): ask, don't guess.
    return {"route": "unsupported"}


# --- Priority B (2026-10-28): outbound research approval -------------------
# There is no approval mechanism in the application yet (docs/CONTROL_CENTER_
# PLAN.md: "Approvals: None anywhere ... Approval Center is net-new backend
# capability (Section 12 / GUI-10)"), and no state survives between runs, so
# the only approval signal a run can read is the user's own message. The
# contract implemented here is therefore:
#   * a request that authorizes external research in THIS message runs it;
#   * anything else gets an approval REQUEST — the external action is never
#     executed, never simulated, and the answer offers local/knowledge
#     alternatives (GUI-10 will add persisted approve/reject endpoints).

def _document_scoped(question: str) -> bool:
    """True when the request names a local document/corpus as its subject."""
    text = str(question or "")
    return bool(_BOOKS_RE.search(text) or _BOOKS_IDENTITY_RE.search(text)
                or _DOC_DEICTIC_RE.search(text) or _LOCAL_ONLY_RE.search(text))


def research_authorization(question: str, route: str = "") -> tuple[bool, str]:
    """May this run perform outbound web research? -> (allowed, code).

    Fail-closed: only two in-message authorizations exist (an explicit web
    request, a current-information request), plus the feasibility route,
    whose own contract is a study built from external data the user asked
    for. Every other path — the LLM routing label, a planner-dispatched
    research task, the books->research fallback — is NOT authorized, and a
    message that forbids web access can never authorize it either.
    """
    if route == "feasibility":
        return True, "requested_operation:feasibility_study"
    q = str(question or "")
    if _LOCAL_ONLY_RE.search(q):
        # "Do not search the web" must deny even though it contains the
        # words "search the web" — the constraint outranks the cue.
        return False, "local_only_request"
    if _EXPLICIT_WEB_RE.search(q):
        return True, "explicit_web_request"
    if _CURRENT_INFO_RE.search(q):
        return True, "current_information_request"
    if _EXPLICIT_RESEARCH_RE.search(q) and not _document_scoped(q):
        # 2026-10-30: "Research three suitable models..." is the user asking
        # for external research in this very message, which is the same
        # authorization as "search the web for ...". Without it the route
        # said research while the gate demanded an approval, so a request
        # that explicitly asked to research could never do it.
        # Deliberately NOT for document-scoped wording: "research what the
        # manual says" names a local source, so a books refusal must not turn
        # into an outbound search on the strength of the verb alone.
        return True, "explicit_research_request"
    return False, "no_research_authorization_in_this_request"


_CAPABILITIES_LIST = (
    "answer questions from your indexed documents (Books), research the web "
    "when you ask for it, look up ERP business data, or run a feasibility "
    "study"
)


def unsupported_reply(question: str) -> str:
    """Deterministic reply for the `unsupported` route (no LLM, no tools).

    Three shapes, chosen by the same rules keyword_route used, so the reply
    always explains WHY the request was not executed: missing conversation
    memory, a capability that does not exist, or an underspecified request.
    """
    q = str(question or "")
    if _NO_HISTORY_RE.search(q):
        return (
            "I don't keep conversation history between runs, so I can't "
            "tell you what you told me earlier. Send the details again in "
            "this message and I'll route it to the right source."
        )
    if _ACTION_REQUEST_RE.search(q):
        return (
            "I can't do that — I have no tool for it yet. I can: "
            f"{_CAPABILITIES_LIST}. I can also answer general questions "
            "from my own knowledge."
        )
    # Bare imperatives ("Do it") are UNSPECIFIED, not a known-missing
    # capability: asking what was meant is the honest answer.
    return (
        "I'm not sure what you'd like me to do. I can: "
        f"{_CAPABILITIES_LIST}. Add a bit more detail — for example "
        "\"What does chapter 3 say about propolis?\" or "
        "\"Search the latest AI news\" — and I'll get started."
    )


# Sentence every approval-REQUEST answer carries (2026-10-29). It marks an
# answer that ASKED for permission instead of reporting work, so the runner's
# rejected-local substitution and QA's contradiction check both recognise it
# from ONE definition rather than copying the literal.
APPROVAL_PENDING_MARKER = "I have NOT searched the web"


def approval_request_text(question: str, code: str,
                          diagnosis: dict | None = None,
                          request_id: str | None = None) -> str:
    """The fail-closed answer when research needs approval it does not have.

    Identifies the proposed external action and why it is needed, states
    plainly that nothing was executed, how to authorize exactly this
    operation (an explicit request in a message — scoped to that request,
    never a blanket permission), and offers the local alternatives. When
    `diagnosis` is a books refusal, the local limitation is disclosed too,
    so the user sees both outcomes instead of one.

    With `request_id` (GUI-10, when an approval store is in scope) the reply
    additionally carries the PERSISTED request id and its decision/resume
    endpoints, so a decision can be bound to this exact operation.
    """
    q = str(question or "").strip()
    local_note = ""
    if isinstance(diagnosis, dict) and diagnosis:
        local_note = (
            "Note: your indexed documents were consulted first and did not "
            f"provide a sufficient grounded answer (reason: "
            f"{diagnosis.get('reason', 'unknown')}).\n\n"
        )
    why = ("your documents did not contain a sufficient answer, and web "
           "research is an external action"
           if str(code).startswith("books_insufficient")
           else "your message does not ask for web research")
    binding = ""
    if request_id:
        binding = (
            f"\n\nApproval request ID: {request_id}\n"
            "Decide it with POST /api/v1/approvals/" + str(request_id) +
            "/approve (or /reject), then POST /api/v1/approvals/" +
            str(request_id) +
            "/resume to run (or honestly close) THIS operation — the "
            "approval applies only to this exact request."
        )
    return (
        f"{local_note}"
        f"{APPROVAL_PENDING_MARKER} for this yet.\n\n"
        "External action proposed: a web search (up to 8 results, reading "
        f"up to 3 pages) for: {q}\n"
        f"Why: {why}.\n\n"
        "To approve it, send an explicit request such as \"Search the web "
        "for ...\" or \"Look up ... online\" — that approval applies only "
        "to the request you send, never to future ones.\n"
        "Alternatively I can answer from my own knowledge, or from your "
        "indexed documents, if you prefer."
        f"{binding}"
    )


# Routes answered without the planner: a single direct step (2026-10-28).
# Keeps a general/unsupported/conversational request away from the planner's
# research-shaped fallback plan, so routing decides, not plan generation.
# Routes answered in one step: the follow-up route reads one recorded
# execution (no plan can improve it) and the three answer-only routes never
# decompose. Everything else may be marked `complex` for the planner.
# 2026-10-30: `feedback` joins them — revising one recorded answer and
# recording one lesson is a single step, and a plan would add model calls
# without adding work.
_SINGLE_STEP_ROUTES = frozenset({"conversational", "general", "unsupported",
                                 "followup", "feedback"})


def supervisor(state: AgentState) -> dict:
    """Return only the keys this node changes: route (and sub_routes if multi).

    ROUTING_MODE="keyword" (default): pure keyword routing, unchanged.
    ROUTING_MODE="llm": consult the LLM first; deterministic keyword result is
    the fallback whenever the LLM fails, returns garbage, or an unknown label.

    Phase 7: also sets "complex": True when app.planner.needs_planning()
    detects a multi-objective request (planner gating itself stays behind
    PLANNER_MODE in app/planner.py — keyword routing values are unchanged).
    """
    question = state.get("question", "")

    base = None
    if is_conversational(question):
        # Deterministic pre-check: a greeting is answered locally, and in
        # llm mode it must not consume a classification call either.
        base = {"route": "conversational"}
    elif ROUTING_MODE == "llm":
        label = _llm_route(question)
        if label == "feasibility":
            base = {"route": "feasibility", "sub_routes": ["research", "books"]}
        elif label in _VALID_LLM_ROUTES:
            base = {"route": label}
        # LLM missing/malformed/unknown -> deterministic keyword fallback.

    if base is None:
        base = keyword_route(state)

    if needs_planning(state) and base.get("route") not in _SINGLE_STEP_ROUTES:
        base["complex"] = True
    return base


# --- Orchestrated knowledge-fallback decision (2026-10-25) -----------------
# Failure-state classification for the books refusal reasons (app/agents/
# books_agent.py): which orchestration outcome each reason maps to. Every
# reason below has ALREADY exhausted the bounded local recovery that the M5
# eligibility contract allows for it (recovery never runs twice, never for
# low score, never after the grounding gate), so the orchestrator's only
# remaining choice is abstain-vs-research.
_FAILURE_CLASS = {
    "empty_or_error": "local_synthesis_error",
    "low_confidence": "no_local_evidence",
    "self_refusal_after_recovery": "recovery_exhausted",
    "grounding": "synthesis_validation_failed",
    "verifier": "validation_failed",
}


def fallback_decision(state: AgentState) -> tuple[bool, str]:
    """Orchestrator decision: may a direct books refusal fall back to the
    EXISTING research agent? Returns (allowed, code); code is diagnostic
    (logged by app.graph._after_books).

    Decision inputs, deliberately NOT the ones the spec forbids:
      * the structured state["books_refusal"] diagnosis — never a similarity
        score alone, never "the agent returned a refusal sentence" alone;
      * the ORIGINAL question's source constraints (_LOCAL_ONLY_RE explicit
        local-only, _DOC_DEICTIC_RE document-scoped) — such requests abstain
        locally instead of being silently answered from the internet;
      * run context: only the DIRECT books route falls back (feasibility
        chains and plan-dispatched runs are excluded), and a run where
        research already answered never falls back again.
    Combined with the graph edge (books -> research -> END), fallback
    transitions are bounded to ONE per run — books<->research loops are
    structurally impossible. The research query is always the original
    question (research_agent reads state["question"]), never the refusal
    sentence and never retrieved passage text.
    """
    if not BOOKS_WEB_FALLBACK:
        return False, "fallback_disabled"
    diagnosis = state.get("books_refusal")
    if not isinstance(diagnosis, dict) or not diagnosis:
        return False, "books_answered"
    if state.get("route") != "books":
        return False, "not_direct_books_route"
    question = state.get("question", "")
    if _LOCAL_ONLY_RE.search(question) or _DOC_DEICTIC_RE.search(question):
        return False, "local_only_request"
    if "research" in (state.get("agent_answers") or {}):
        return False, "research_already_ran"
    reason = str(diagnosis.get("reason") or "unknown")
    failure_class = _FAILURE_CLASS.get(reason, "unclassified")
    return True, f"books_insufficient:{failure_class}:{reason}"
