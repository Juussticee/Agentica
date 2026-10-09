"""Phase 3 ERP-boundary tests — fully deterministic, zero Gemini quota.

Covers the 8 required behaviors:
  1. ERP routing (supervisor keywords + graph wiring Supervisor->erp->END)
  2. Mock ERP adapter (search/get_record/invalid input/determinism)
  3. Successful read (agent formats mock records)
  4. Record-not-found (adapter + agent message)
  5. Adapter failure (unavailable / timeout / unexpected, safe messages)
  6. Permission denied (safe message, no internal detail leakage)
  7. Provenance (sources tagged "erp" via the existing state mechanism)
  8. Write protection (interface has NO write methods; records frozen)

No network, no real ERP, no credentials — error paths use stub adapters.
"""

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.erp import (  # noqa: E402
    ERPAdapter,
    ERPAdapterUnavailableError,
    ERPError,
    ERPInvalidRequestError,
    ERPPermissionDeniedError,
    ERPRecord,
    ERPRecordNotFoundError,
    ERPTimeoutError,
    MockERPAdapter,
    get_erp_adapter,
)
import app.agents.erp_agent as erp_agent_module  # noqa: E402
import app.erp.log as erp_log  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.supervisor import supervisor  # noqa: E402


class _StubAdapter(ERPAdapter):
    """Read-only adapter stub whose search behavior is injected per test."""

    def __init__(self, search_behavior=None):
        self._search_behavior = search_behavior or (lambda query, limit=5: [])

    def name(self):
        return "stub"

    def search(self, query, limit=5):
        return self._search_behavior(query, limit)

    def get_record(self, entity, record_id):
        raise ERPRecordNotFoundError(f"stub has no {entity} records")


def _run_agent(behavior, question="Show me open invoices", prior_sources=None):
    """Run erp_agent against a stub adapter, returning its state update."""
    state = {"question": question}
    if prior_sources is not None:
        state["sources"] = list(prior_sources)
    stub = _StubAdapter(behavior)
    with mock.patch.object(erp_agent_module, "get_erp_adapter", return_value=stub):
        return erp_agent_module.erp_agent(state)


# --------------------------------------------------------------------------- #
# 1. ERP routing
# --------------------------------------------------------------------------- #
class TestRouting(unittest.TestCase):
    def test_erp_keywords_route_to_erp(self):
        for question in (
            "Show me open invoices",
            "What is the current inventory level?",
            "Check stock for P-9001",
            "ERP customer records please",
        ):
            with self.subTest(question=question):
                self.assertEqual(supervisor({"question": question})["route"], "erp")

    def test_books_research_feasibility_routing_unchanged(self):
        self.assertEqual(
            supervisor({"question": "What does the orbital book say about delta-v?"})["route"],
            "books",
        )
        self.assertEqual(
            supervisor({"question": "What is LangGraph used for?"})["route"], "research"
        )
        result = supervisor({"question": "Do a feasibility study for a beekeeping venture"})
        self.assertEqual(result["route"], "feasibility")
        self.assertEqual(result["sub_routes"], ["research", "books"])

    def test_graph_erp_route_end_to_end(self):
        """Supervisor -> erp_agent -> END with the REAL mock adapter (no network)."""
        result = build_graph().invoke({"question": "Show me open invoices"})
        self.assertEqual(result.get("route"), "erp")
        self.assertIn("erp", result.get("sources", []))
        self.assertIn("INV-1001", result.get("answer", ""))
        self.assertIn("erp", result.get("agent_answers", {}))


# --------------------------------------------------------------------------- #
# 2. Mock ERP adapter
# --------------------------------------------------------------------------- #
class TestMockAdapter(unittest.TestCase):
    def test_search_finds_expected_records(self):
        records = MockERPAdapter().search("acme")
        self.assertEqual([r.record_id for r in records], ["C-001", "INV-1001"])

    def test_search_is_case_insensitive_and_deterministic(self):
        adapter = MockERPAdapter()
        first = adapter.search("GLOBEX")
        second = adapter.search("globex")
        self.assertEqual([r.record_id for r in first], ["C-002", "INV-1002"])
        self.assertEqual(first, second)

    def test_search_no_match_returns_empty_list(self):
        self.assertEqual(MockERPAdapter().search("zzz-no-such-term"), [])

    def test_search_limit(self):
        records = MockERPAdapter().search("inv", limit=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(MockERPAdapter().search("inv")), 2)

    def test_get_record_found(self):
        record = MockERPAdapter().get_record("invoice", "INV-1002")
        self.assertEqual(record.entity, "invoice")
        self.assertEqual(record.fields["status"], "Paid")
        self.assertEqual(record.fields["amount_due"], "340.50")

    def test_get_record_not_found(self):
        with self.assertRaises(ERPRecordNotFoundError):
            MockERPAdapter().get_record("customer", "C-999")
        with self.assertRaises(ERPRecordNotFoundError):
            MockERPAdapter().get_record("nonsense_entity", "X-1")

    def test_invalid_requests_rejected(self):
        adapter = MockERPAdapter()
        for call in (
            lambda: adapter.search(""),
            lambda: adapter.search("   "),
            lambda: adapter.search("x", limit=0),
            lambda: adapter.search("x", limit=-1),
            lambda: adapter.get_record("customer", ""),
            lambda: adapter.get_record("", "C-001"),
        ):
            with self.subTest(call=call):
                with self.assertRaises(ERPInvalidRequestError):
                    call()

    def test_adapter_name(self):
        self.assertEqual(MockERPAdapter().name(), "mock")


# --------------------------------------------------------------------------- #
# 3+4. Successful read / record-not-found through the agent
# --------------------------------------------------------------------------- #
class TestAgentReads(unittest.TestCase):
    def test_successful_read_formats_records(self):
        with mock.patch.object(
            erp_agent_module, "get_erp_adapter", return_value=MockERPAdapter()
        ):
            result = erp_agent_module.erp_agent({"question": "Show me open invoices"})
        self.assertIn("ERP lookup results", result["answer"])
        self.assertIn("INV-1001", result["answer"])
        self.assertIn("Open", result["answer"])
        self.assertIn("mock ERP adapter", result["answer"])  # transparency note

    def test_empty_result_is_safe_message(self):
        result = _run_agent(lambda q, limit=5: [])
        self.assertIn("no records matching", result["answer"])
        self.assertIn("mock ERP adapter", result["answer"])

    def test_blank_question_maps_to_invalid_request(self):
        # Real mock validates the blank query after the agent strips it.
        with mock.patch.object(
            erp_agent_module, "get_erp_adapter", return_value=MockERPAdapter()
        ):
            result = erp_agent_module.erp_agent({"question": "   "})
        self.assertEqual(
            result["answer"], "This request could not be processed as a valid ERP query."
        )

    def test_agent_returns_only_changed_keys(self):
        result = _run_agent(lambda q, limit=5: [])
        self.assertEqual(set(result.keys()), {"answer", "sources", "agent_answers"})


# --------------------------------------------------------------------------- #
# 5+6. Adapter failure / permission denied — safe messages, no leakage
# --------------------------------------------------------------------------- #
class TestAgentErrorPaths(unittest.TestCase):
    def test_adapter_unavailable(self):
        def boom(query, limit=5):
            raise ERPAdapterUnavailableError("connection refused http://internal-erp.local:4711")

        result = _run_agent(boom)
        self.assertEqual(
            result["answer"], "The ERP system is currently unavailable. Please try again later."
        )
        self.assertNotIn("internal-erp", result["answer"])

    def test_permission_denied_safe_and_no_leak(self):
        def denied(query, limit=5):
            raise ERPPermissionDeniedError("role read_all denied token=supersecret123")

        result = _run_agent(denied)
        self.assertIn("insufficient permissions", result["answer"])
        self.assertNotIn("supersecret123", result["answer"])

    def test_record_not_found_message(self):
        def missing(query, limit=5):
            raise ERPRecordNotFoundError("customer C-999 absent")

        result = _run_agent(missing)
        self.assertEqual(result["answer"], "No matching record was found in the ERP.")

    def test_timeout_message(self):
        def slow(query, limit=5):
            raise ERPTimeoutError("operation exceeded 30s")

        result = _run_agent(slow)
        self.assertEqual(
            result["answer"], "The ERP request timed out. Please try again later."
        )

    def test_unexpected_failure_safe_and_no_leak(self):
        def broken(query, limit=5):
            raise ValueError("bad connection string Server=secret-db-host")

        result = _run_agent(broken)
        self.assertEqual(
            result["answer"], "An unexpected error occurred while accessing the ERP."
        )
        self.assertNotIn("secret-db-host", result["answer"])

    def test_base_erp_error_generic_message(self):
        def base(query, limit=5):
            raise ERPError("internal stack detail boom")

        result = _run_agent(base)
        self.assertEqual(
            result["answer"], "An ERP error occurred while processing this request."
        )
        self.assertNotIn("boom", result["answer"])

    def test_error_answers_carry_no_mock_data_note(self):
        result = _run_agent(lambda q, limit=5: (_ for _ in ()).throw(ERPTimeoutError("t")))
        self.assertNotIn("mock ERP adapter", result["answer"])


# --------------------------------------------------------------------------- #
# 7. Provenance
# --------------------------------------------------------------------------- #
class TestProvenance(unittest.TestCase):
    def test_sources_appended_via_existing_mechanism(self):
        result = _run_agent(lambda q, limit=5: [], prior_sources=["research"])
        self.assertEqual(result["sources"], ["research", "erp"])

    def test_sources_fresh_when_none_prior(self):
        result = _run_agent(lambda q, limit=5: [])
        self.assertEqual(result["sources"], ["erp"])

    def test_agent_answers_records_erp(self):
        result = _run_agent(lambda q, limit=5: [])
        self.assertEqual(result["agent_answers"]["erp"], result["answer"])


# --------------------------------------------------------------------------- #
# 8. Write protection — enforced by the interface itself
# --------------------------------------------------------------------------- #
class TestWriteProtection(unittest.TestCase):
    _FORBIDDEN = {
        "create",
        "update",
        "delete",
        "upsert",
        "patch",
        "post",
        "put",
        "write",
        "insert",
        "remove",
        "save",
        "commit",
    }

    def test_interface_exposes_only_read_methods(self):
        public = {n for n in dir(ERPAdapter) if not n.startswith("_")}
        self.assertEqual(public, {"name", "search", "get_record"})
        self.assertFalse(public & self._FORBIDDEN)

    def test_mock_exposes_only_read_methods(self):
        public = {n for n in dir(MockERPAdapter) if not n.startswith("_")}
        self.assertEqual(public, {"name", "search", "get_record"})
        self.assertFalse(public & self._FORBIDDEN)

    def test_abstract_interface_not_instantiable(self):
        with self.assertRaises(TypeError):
            ERPAdapter()  # abstract methods: cannot be used without implementing reads

    def test_records_are_frozen(self):
        record = ERPRecord("customer", "C-001", "ACME Manufacturing", {"status": "Active"})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.title = "tampered"


# --------------------------------------------------------------------------- #
# Factory + logging
# --------------------------------------------------------------------------- #
class TestFactory(unittest.TestCase):
    def test_default_factory_returns_mock(self):
        self.assertIsInstance(get_erp_adapter(), MockERPAdapter)

    def test_unknown_adapter_rejected(self):
        import app.config as config

        with mock.patch.object(config, "ERP_ADAPTER", "future_real_adapter"):
            with self.assertRaises(ValueError):
                get_erp_adapter()


class TestLogging(unittest.TestCase):
    def _with_temp_log(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "erp_actions.log"
        patcher = mock.patch.object(erp_log, "ERP_LOG_PATH", path)
        patcher.start()
        self.addCleanup(patcher.stop)
        return path

    def test_success_log_line_fields(self):
        path = self._with_temp_log()
        with mock.patch.object(
            erp_agent_module, "get_erp_adapter", return_value=MockERPAdapter()
        ):
            erp_agent_module.erp_agent({"question": "Show me open invoices"})
        line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
        self.assertIn("op=search", line)
        self.assertIn("adapter=mock", line)
        self.assertIn("status=success", line)
        self.assertIn("duration_ms=", line)
        self.assertIn("records=1", line)

    def test_failure_log_line_has_category(self):
        path = self._with_temp_log()
        result = _run_agent(
            lambda q, limit=5: (_ for _ in ()).throw(ERPPermissionDeniedError("denied"))
        )
        self.assertIn("insufficient permissions", result["answer"])
        line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
        self.assertIn("status=failure", line)
        self.assertIn("error_category=permission_denied", line)

    def test_unexpected_failure_logged_as_category(self):
        path = self._with_temp_log()
        _run_agent(lambda q, limit=5: (_ for _ in ()).throw(ValueError("boom")))
        line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
        self.assertIn("error_category=unexpected_failure", line)

    def test_log_never_contains_query_or_record_data(self):
        path = self._with_temp_log()
        with mock.patch.object(
            erp_agent_module, "get_erp_adapter", return_value=MockERPAdapter()
        ):
            erp_agent_module.erp_agent({"question": "Show me ACME invoices NOW-SECRET-QUERY"})
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("ACME", content)
        self.assertNotIn("NOW-SECRET-QUERY", content)
        self.assertNotIn("INV-1001", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
