"""Single factory for chat LLMs (user-requested deviation from the original spec).

Every agent imports get_llm() from here — no agent ever instantiates a model
directly. Switching providers is a one-line change in app/config.py:

    LLM_PROVIDER = "gemini"   #  ->  "ollama"  (original spec behavior)
"""

from app.config import (
    FALLBACK_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    MAIN_MODEL,
    OLLAMA_BASE_URL,
)

_TEMPERATURE = 0.2  # deterministic-ish synthesis; raised no further per Phase 1


def get_llm(fast: bool = False):
    """Return a LangChain-compatible chat model for the configured provider.

    fast=True: for short classification/rewrite calls. On gemini this minimizes
    thinking overhead (Gemini 3 thinks by default, which dominates latency on
    tiny prompts); on ollama it is simply the normal model. All instantiation
    stays inside this factory — agents never build models directly.
    """
    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER='gemini' but no API key was found. "
                "Put GOOGLE_API_KEY=... in your .env file (never in code)."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        if fast:
            try:
                return ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=_TEMPERATURE,
                    thinking_config={"thinking_level": "minimal"},
                )
            except Exception:
                # Older SDK versions reject thinking_config; degrade silently.
                pass
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=_TEMPERATURE,
        )

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=MAIN_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=_TEMPERATURE,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r} (expected 'gemini' or 'ollama'); "
        f"fallback model {FALLBACK_MODEL!r} is only used manually per spec."
    )


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
