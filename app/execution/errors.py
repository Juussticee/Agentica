"""Structured, user-facing error model (GUI-1; reused by GUI-16 AI readiness).

Every error raised to the Control Center maps to:
  WHAT happened (human message) / WHY (cause) / WHAT NOW (recoverability
  + suggested actions). Raw exception text stays in `details` and is shown
  only behind a "technical details" disclosure (CONTROL_CENTER_PLAN §32).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# AI/provider classification (GUI-16 taxonomy)
AI_AUTH_REJECTED = "auth_rejected"          # provider answered 401/403: bad key
AI_CONNECTION_REFUSED = "connection_refused"  # nothing listening on the endpoint
AI_DNS_UNREACHABLE = "dns_unreachable"      # host could not be resolved/routed
AI_TIMEOUT = "timeout"                      # endpoint accepted but never answered
AI_WRONG_SERVICE = "wrong_service"          # something answered, but not the AI service
AI_MODEL_MISSING = "model_missing"          # endpoint works but model not installed
AI_BAD_RESPONSE = "bad_response"            # endpoint answered in an unexpected format
AI_MALFORMED_ENDPOINT = "malformed_endpoint"  # URL itself unparseable
AI_NOT_CONFIGURED = "not_configured"        # no key/base URL at all

AI_CATEGORIES = {
    AI_AUTH_REJECTED,
    AI_CONNECTION_REFUSED,
    AI_DNS_UNREACHABLE,
    AI_TIMEOUT,
    AI_WRONG_SERVICE,
    AI_MODEL_MISSING,
    AI_BAD_RESPONSE,
    AI_MALFORMED_ENDPOINT,
    AI_NOT_CONFIGURED,
}

# Human, action-oriented explanations (UX language rule: WHAT/WHY/WHAT NOW)
AI_MESSAGES = {
    AI_AUTH_REJECTED: (
        "The AI provider rejected the API key.",
        "The configured key was not accepted.",
        "Check the API key in Settings → AI connection, then test the connection.",
    ),
    AI_CONNECTION_REFUSED: (
        "The AI service could not be reached.",
        "Nothing is listening at the configured address.",
        "Check that the AI service is running and the address in Settings is correct.",
    ),
    AI_DNS_UNREACHABLE: (
        "The AI service address could not be found.",
        "The hostname could not be resolved from this machine.",
        "Check the address/network configuration in Settings.",
    ),
    AI_TIMEOUT: (
        "The AI service did not respond in time.",
        "The request waited too long for an answer.",
        "Retry; if it keeps failing, check the service or use a different provider.",
    ),
    AI_WRONG_SERVICE: (
        "The configured address is not an AI service.",
        "Something else answered at that address.",
        "Verify the endpoint URL in Settings → AI connection.",
    ),
    AI_MODEL_MISSING: (
        "The configured AI model is not available.",
        "The service is running but the requested model is not installed/accessible.",
        "Install or select an available model, then test the connection again.",
    ),
    AI_BAD_RESPONSE: (
        "The AI service returned an unexpected response.",
        "The answer format did not match what Agentica expects.",
        "Retry; if it persists, report the endpoint format issue.",
    ),
    AI_MALFORMED_ENDPOINT: (
        "The AI service address is malformed.",
        "The configured URL could not be used as an endpoint.",
        "Correct the endpoint URL in Settings (e.g. http://host:port).",
    ),
    AI_NOT_CONFIGURED: (
        "AI is not configured yet.",
        "No API key or endpoint is set, so AI-backed steps cannot run.",
        "Configure a provider in Settings → AI connection.",
    ),
}


@dataclass
class ExecutionError:
    """A structured, user-facing error record."""

    category: str                  # AI_* above or an execution category
    human_message: str             # WHAT happened (plain language)
    why: str                       # WHY (plain language)
    recovery: str                  # suggested action (plain language)
    recoverable: bool = True
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    severity: str = "error"
    details: Dict[str, Any] = field(default_factory=dict)  # raw/technical, hidden by default

    def to_dict(self) -> Dict[str, Any]:
        what_now, why, what_next = self.human_message, self.why, self.recovery
        return {
            "category": self.category,
            "human_message": what_now,
            "why": why,
            "recovery": what_next,
            "recoverable": self.recoverable,
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "severity": self.severity,
            "details": dict(self.details),
        }


def classify_ai_exception(exc: Exception, provider: Optional[str] = None) -> str:
    """Map a provider/HTTP exception to an AI_* category (bounded, no retry)."""
    text = str(exc).lower()
    type_name = type(exc).__name__.lower()

    if (isinstance(exc, ValueError) and "url" in text) or "invalid url" in text:
        return AI_MALFORMED_ENDPOINT
    if "unauthorized" in text or "401" in text or "invalid api key" in text or "permission denied" in text:
        return AI_AUTH_REJECTED
    if "timeout" in text or "timed out" in text or "timeout" in type_name:
        return AI_TIMEOUT
    if "not found" in text and ("model" in text or "ollama" in text):
        return AI_MODEL_MISSING
    if "404" in text or "not_found" in type_name:
        return AI_MODEL_MISSING
    if "connection refused" in text or "connectionrefused" in text:
        return AI_CONNECTION_REFUSED
    if "getaddrinfo failed" in text or "name or service not known" in text or "nodename nor servname" in text:
        return AI_DNS_UNREACHABLE
    if ("no connection" in text or "unreachable" in text or "network" in text
            or "connectionerror" in type_name):
        return AI_CONNECTION_REFUSED
    return AI_BAD_RESPONSE


def ai_error_for_provider_unavailable(provider: str, category: str) -> ExecutionError:
    human, why, recovery = AI_MESSAGES.get(category, AI_MESSAGES[AI_BAD_RESPONSE])
    return ExecutionError(
        category=category,
        human_message=human,
        why=why,
        recovery=recovery,
        recoverable=True,
        details={"provider": provider, "category": category},
    )
