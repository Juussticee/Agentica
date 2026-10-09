"""TEST-ONLY dummy agent demonstrating the standardized pattern.

Run via:  python scripts/test_agent_architecture.py

This is a *test-only* double, never a permanent production agent. It exists
solely to prove that adding an agent is:

  1. one module,
  2. one public entry point (def dummy_agent(state: AgentState) -> dict),
  3. a dict of ONLY changed keys,
  4. provenance recorded through state["sources"] + the shared helper,
  5. no direct calls to another agent,
  6. no new AgentState key,

so a hypothetical future agent (email, calendar, CRM, finance, HR, real ERP,
...) can be plugged in without redesigning the system.
"""

from app.state import AgentState, with_agent_output


def dummy_agent(state: AgentState) -> dict:
    return with_agent_output(state, "dummy", "DUMMY ANSWER", ["dummy"])
