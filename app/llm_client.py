"""Single factory for chat LLMs (user-requested deviation from the original spec).

Every agent imports get_llm() from here — no agent ever instantiates a model
directly. Switching providers is a one-line change in app/config.py:

    LLM_PROVIDER = "gemini"   #  ->  "ollama"  (original spec behavior)
"""

import re

from app.config import (
    FALLBACK_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    MAIN_MODEL,
    OLLAMA_BASE_URL,
)

_TEMPERATURE = 0.2  # deterministic-ish synthesis; raised no further per Phase 1


def get_llm(fast: bool = False, timeout: float | None = None):
    """Return a LangChain-compatible chat model for the configured provider.

    fast=True: for short classification/rewrite calls. On gemini this minimizes
    thinking overhead (Gemini 3 thinks by default, which dominates latency on
    tiny prompts); on ollama it is simply the normal model. All instantiation
    stays inside this factory — agents never build models directly.

    timeout: OPTIONAL transport-level request timeout in seconds (M4B-H).
    None — the default — preserves the historical construction exactly for
    every existing caller. When set it becomes the httpx timeout, which
    bounds EACH of connect/read/write/pool individually: the installed
    ollama client passes it straight through (its own default is NO
    timeout at all) and the Gemini SDK exposes the same field as its
    per-request deadline. Per-operation semantics mean it bounds GAPS for
    a streamed response, not the total — that is why app/verifier.py
    forces non-streaming on the ollama path (M4B-H closure audit). This
    is what reclaims a hung request; the caller-side thread wrapper in
    app/verifier.py still bounds waiting time separately.
    """
    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER='gemini' but no API key was found. "
                "Put GOOGLE_API_KEY=... in your .env file (never in code)."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        _timeout_kw = {} if timeout is None else {"timeout": timeout}
        if fast:
            try:
                return ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=_TEMPERATURE,
                    thinking_config={"thinking_level": "minimal"},
                    **_timeout_kw,
                )
            except Exception:
                # Older SDK versions reject thinking_config; degrade silently.
                pass
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=_TEMPERATURE,
            **_timeout_kw,
        )

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=MAIN_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=_TEMPERATURE,
            **({"sync_client_kwargs": {"timeout": timeout}} if timeout else {}),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r} (expected 'gemini' or 'ollama'); "
        f"fallback model {FALLBACK_MODEL!r} is only used manually per spec."
    )


_ECHO_MIN_WORDS = 5
# 0.5 because a local model PARAPHRASES the instruction it echoes ("the request
# asks for code" -> "the question requires code"): the live paraphrase scored
# 0.56 against its source sentence while genuine content scored under 0.2.
_ECHO_COVERAGE = 0.5
# The model also narrates its own answer ("**Note**: The revised answer
# explicitly covers ..."), which shares no wording with the instruction. Such a
# line states what the answer is supposed to contain, never what it contains.
# Lines that are ONLY meta: an empty "**Final Check:**" heading, or the
# model reporting that it followed formatting rules. Neither carries content.
_META_ONLY_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\*{0,2}\s*(?:final\s+)?(?:note|check)\s*\*{0,2}"
    r"\s*:?\s*\*{0,2}\s*$"
    r"|^\s*(?:[-*]\s*)?the answer (?:is|has been|was) "
    r"(?:formatted|stated|provided) as requested",
    re.IGNORECASE,
)

_META_NOTE_RE = re.compile(
    r"^\s*(?:\*\*|-\s*)?(?:final\s+)?(?:note|check)\b.*\b(?:revised answer|"
    r"requires code|cannot provide|one sentence|instruction|user named|"
    r"formatted as requested|every requirement)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]{4,}")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")


def strip_instruction_echo(text: str, instructions: str) -> str:
    """Drop lines that only repeat the prompt's own instructions.

    A small local model copies the prompt back into its answer: the
    2026-10-31 live audit had a general answer end with "Note: If the question
    requires code or a draft, I cannot provide that here.", and a feedback
    revision end with "If something cannot be supplied without running the
    task again, say so in one sentence at the end." Those are our instructions
    to ourselves, not an answer to the user, so they are removed before the
    text is stored.

    Deterministic and deliberately conservative: a line is dropped only when
    it has at least five words AND at least 70% of those words appear in a
    single instruction sentence, so normal content (which reuses at most a
    few instruction words) is never touched. Returns the text unchanged when
    nothing matches.
    """
    body = str(text or "")
    source = str(instructions or "")
    if not body.strip() or not source.strip():
        return body
    sentences = [set(_WORD_RE.findall(part.lower()))
                 for part in _SENTENCE_SPLIT_RE.split(source)]
    sentences = [s for s in sentences if len(s) >= _ECHO_MIN_WORDS]
    if not sentences:
        return body

    kept: list = []
    for line in body.splitlines():
        if _META_ONLY_RE.search(line):  # an empty meta heading, or a claim
            continue                  # that the answer is nicely formatted
        words = set(_WORD_RE.findall(line.lower()))
        if len(words) >= _ECHO_MIN_WORDS and (
                _META_NOTE_RE.search(line)
                or any(len(words & sentence) / len(words) >= _ECHO_COVERAGE
                       for sentence in sentences)):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    # Never return an empty answer: if the whole reply echoed the prompt, the
    # caller's own empty-answer handling is the honest path, but blanking a
    # mixed answer would lose the part that did answer.
    return cleaned if cleaned.strip() else body


def llm_text(response) -> str:
    """Normalize a LangChain chat response to plain text.

    Gemini 3 models return content as a list of typed blocks
    ([{'type': 'text', 'text': ...}, ...]); Ollama returns a plain string.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)
