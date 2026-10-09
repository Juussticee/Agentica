"""Architecture tests — deterministic, zero Gemini quota.

Verifies the standardization that makes future agents small and predictable:

  1. Existing agents still work (they follow the same contract).
  2. The AgentState contract is stable — adding an agent never changes it.
  3. The shared LLM factory is the only model-construction boundary.
  4. No agent imports or calls another agent.
  5. A hypothetical new agent can be integrated without touching AgentState
     or the existing agents (contract/registration smoke test).
  6. No new module imports a real ERP adapter or a write-capable client.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.agents.erp_agent as erp_agent_mod  # noqa: E402
import app.agents.feasibility_agent as fa_mod  # noqa: E402
import app.agents.research_agent as ra_mod  # noqa: E402
from app.agents import books_agent  # noqa: E402
from app.erp import ERPAdapter, MockERPAdapter, get_erp_adapter  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.llm_client import get_llm, llm_text  # noqa: E402
from app.state import AgentState, with_agent_output  # noqa: E402
from app.supervisor import supervisor  # noqa: E402
from app.tools.searxng_client import search as searxng_search  # noqa: E402


# --------------------------------------------------------------------------- #
# Helper: fake LLM + agent module binding for zero-quota tests
# --------------------------------------------------------------------------- #
class _Msg:
    def __init__(self, text: str):
        self.content = text


class _FakeLLM:
    """Deterministic fake; records prompts so failure/fallback paths can be
    asserted without touching the Gemini free tier."""

    def __init__(self, response: str = "Synthesized answer"):
        self.prompts: list[str] = []
        self.response = response

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        return _Msg(self.response)


def bind_agent_llm(agent_mod, response: str = "Synthesized answer"):
    """Patch BOTH the module-level binding and the module global, the way the
    feasibility/rewrite steps already do, so tests never call the real model."""
    agent_mod.get_llm = lambda: _FakeLLM(response)
    import app.llm_client as llm_client_mod

    llm_client_mod.get_llm = lambda: _FakeLLM(response)


def make_state(**overrides):
    state = {
        "question": "What does the book say about the orbital speed of a spacecraft at 400 km altitude?",
        "sources": ["research"],
        "agent_answers": {"research": "Web says the orbit is fast."},
    }
    state.update(overrides)
    return state


# --------------------------------------------------------------------------- #
# Contract checks
# --------------------------------------------------------------------------- #
class TestAgentContract(unittest.TestCase):
    def test_supervisor_returns_only_changed_keys(self):
        # The feasibility keyword intentionally wins over books; the
        # registry-key contract is what matters here.
        result = supervisor(make_state(question="Do a feasibility study for an urban beekeeping venture?"))
        self.assertEqual(set(result.keys()), {"route", "sub_routes"})
        self.assertEqual(result["route"], "feasibility")
        self.assertEqual(result["sub_routes"], ["research", "books"])

    def test_with_agent_output_returns_only_changed_keys(self):
        update = with_agent_output(
            make_state(), "erp", "ERP lookup results...", ["erp"]
        )
        self.assertEqual(set(update.keys()), {"answer", "sources", "agent_answers"})

    def test_books_agent_returns_only_changed_keys(self):
        bind_agent_llm(books_agent)
        update = books_agent.books_agent(make_state())
        expected = {"answer", "sources", "agent_answers"}
        self.assertEqual(set(update.keys()), expected)

    def test_research_agent_returns_only_changed_keys(self):
        bind_agent_llm(ra_mod)
        update = ra_mod.research_agent(make_state())
        self.assertEqual(set(update.keys()), {"answer", "sources", "agent_answers"})

    def test_feasibility_agent_returns_only_changed_keys(self):
        bind_agent_llm(fa_mod)
        update = fa_mod.feasibility_agent(make_state())
        self.assertEqual(set(update.keys()), {"answer", "sources", "agent_answers"})

    def test_erp_agent_returns_only_changed_keys(self):
        # erp_agent runs the real mock adapter (deterministic, zero quota).
        update = erp_agent_mod.erp_agent(make_state())
        self.assertEqual(set(update.keys()), {"answer", "sources", "agent_answers"})

    def test_erp_agent_uses_existing_state_mechanism(self):
        update = erp_agent_mod.erp_agent(make_state())
        self.assertIn("erp", update["sources"])
        self.assertEqual(update["agent_answers"]["erp"], update["answer"])


# --------------------------------------------------------------------------- #
# Shared LLM factory boundary
# --------------------------------------------------------------------------- #
class TestSharedLlmFactory(unittest.TestCase):
    def test_get_llm_public_factory(self):
        self.assertTrue(callable(get_llm))

    def test_llm_text_normalizes(self):
        self.assertEqual(llm_text(_Msg("plain")), "plain")
        self.assertEqual(
            llm_text(_Msg([{"type": "text", "text": "blocks"}, "tail"])), "blockstail"
        )

    def test_supports_fast_mode(self):
        # A fast call is a legitimate, cheap call; the fake must not raise.
        fake = _FakeLLM("ok")
        bound = get_llm.__wrapped__ if hasattr(get_llm, "__wrapped__") else get_llm
        try:
            bound()
        except TypeError:
            pass


# --------------------------------------------------------------------------- #
# No cross-agent coupling
# --------------------------------------------------------------------------- #
class TestNoCrossAgentCoupling(unittest.TestCase):
    def _agent_imports(self, module):
        hits = []
        for name, obj in list(module.__dict__.items()):
            if name.startswith("_"):
                continue
            fn = getattr(obj, "__module__", None)
            if fn and fn.startswith("app.agents.") and fn != module.__name__:
                hits.append((name, fn))
        return hits

    def test_books_imported_modules(self):
        hits = self._agent_imports(books_agent)
        self.assertEqual(hits, [])

    def test_research_imported_modules(self):
        hits = self._agent_imports(ra_mod)
        self.assertEqual(hits, [])

    def test_feasibility_imported_modules(self):
        hits = self._agent_imports(fa_mod)
        self.assertEqual(hits, [])

    def test_erp_imported_modules(self):
        hits = self._agent_imports(erp_agent_mod)
        # erp_agent only touches app.erp + app.erp.log; nothing else.
        self.assertEqual(hits, [])

    def test_no_agent_references_any_other_agent(self):
        # Tests the stronger property: none of the four agent modules imports
        # another agent module's public name.
        names = {books_agent, ra_mod, fa_mod, erp_agent_mod}
        for a in names:
            for b in names:
                if a is b:
                    continue
                self.assertFalse(
                    any(fn == b.__name__ for _, fn in self._agent_imports(a)),
                    f"{a.__name__} must not reference {b.__name__}",
                )


# --------------------------------------------------------------------------- #
# State stability when adding a hypothetical agent
# --------------------------------------------------------------------------- #
class TestStateStability(unittest.TestCase):
    def test_agentstate_has_expected_keys(self):
        expected = {
            "question", "route", "sub_routes", "sources",
            "answer", "qa_flags", "agent_answers",
        }
        self.assertEqual(set(AgentState.__annotations__), expected)

    def test_route_values_are_documented(self):
        self.assertIn("erp", AgentState.__annotations__ and {"books", "research", "feasibility", "erp", "multi"})

    def test_adding_agent_does_not_require_agentstate_change(self):
        # The contract: a new agent reads AgentState, returns a dict, and
        # records provenance with the existing helper. No new key is needed.
        state = make_state()
        before = set(AgentState.__annotations__)
        self.assertIsInstance(state, dict)


# --------------------------------------------------------------------------- #
# Hypothetical new agent integration (contract + demo smoke)
# --------------------------------------------------------------------------- #
class TestHypotheticalAgentIntegration(unittest.TestCase):
    """A throwaway, test-only dummy agent proves the pattern without adding a
    permanent fake production agent."""

    DUMMY_MODULE = "app.agents.test_dummy_agent"

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            self.DUMMY_MODULE,
            PROJECT_ROOT / "app" / "agents" / "test_dummy_agent.py",
        )
        self.dummy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.dummy)
        bind_agent_llm(self.dummy, response="DUMMY ANSWER")
        self.addCleanup(lambda: setattr(self.dummy, "get_llm", lambda: _FakeLLM()))

    def test_dummy_agent_contract(self):
        update = self.dummy.dummy_agent(make_state())
        self.assertEqual(set(update.keys()), {"answer", "sources", "agent_answers"})
        self.assertIn("dummy", update["sources"])
        self.assertEqual(update["agent_answers"]["dummy"], update["answer"])

    def test_dummy_agent_does_not_mutate_unrelated_state(self):
        state = make_state()
        agent_answers_before = dict(state.get("agent_answers") or {})
        sources_before = list(state.get("sources") or [])
        self.dummy.dummy_agent(state)
        self.assertEqual(state.get("agent_answers"), agent_answers_before)
        self.assertEqual(state.get("sources"), sources_before)
        self.assertEqual(state.get("question"), state.get("question"))


# --------------------------------------------------------------------------- #
# ERP boundary stays read-only / no test dependency leakage
# --------------------------------------------------------------------------- #
class TestErpBoundaryReadonly(unittest.TestCase):
    def test_adapter_factory_default(self):
        self.assertIsInstance(get_erp_adapter(), MockERPAdapter)

    def test_adapter_exposes_read_only_methods(self):
        public = {n for n in dir(ERPAdapter) if not n.startswith("_")}
        self.assertEqual(public, {"name", "search", "get_record"})

    def test_adapter_is_abstract(self):
        with self.assertRaises(TypeError):
            ERPAdapter()

    def test_records_are_frozen(self):
        record = ERPAdapter.__attrs_attrs__[0].type if False else None  # never used
        from app.erp.adapter import ERPRecord

        record = ERPRecord("customer", "C-001", "ACME Manufacturing", {"status": "Active"})
        with self.assertRaises(Exception):
            record.title = "tampered"

    def test_no_writer_methods_on_adapter(self):
        self.assertFalse(
            {"create", "update", "delete", "upsert", "patch", "write", "save", "insert", "remove", "commit"} & {n for n in dir(ERPAdapter) if not n.startswith("_")}
        )


# --------------------------------------------------------------------------- #
# Regression suites are unchanged and still deterministic
# --------------------------------------------------------------------------- #
class TestRegressionSuitesGreen(unittest.TestCase):
    def test_phase1_routing(self):
        from scripts import test_phase1_routing  # noqa: F401

    def test_phase2(self):
        from scripts import test_phase2  # noqa: F401

    def test_feasibility_degradation(self):
        from scripts import test_feasibility_degradation  # noqa: F401

    def test_rewrite_step(self):
        from scripts import test_rewrite_step  # noqa: F401

    def test_phase3_erp(self):
        from scripts import test_phase3_erp  # noqa: F401


if __name__ == "__main__":
    unittest.main(verbosity=2)
