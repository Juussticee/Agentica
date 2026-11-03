"""Increment 2 tests: shared diagnostic core (app/diagnostics_core.py).

Verifies the consolidation goal — /api/v1/health, /api/v1/services and
POST /api/v1/ai/test now answer the SAME underlying questions from ONE
definition, while each endpoint keeps its own response schema.

Deterministic and offline:
  * provider readiness is config-only (asserted with a tripwire on the
    HTTP helper — any network attempt fails the test),
  * Ollama probes hit a local fake HTTP server (127.0.0.1) or a closed
    loopback port, never an external host,
  * no live Agentica endpoint, no model call, no database writes
    (temp ExecutionStore only).
"""

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import ai_settings  # noqa: E402
from app import diagnostics_core as core  # noqa: E402
from app import services_health  # noqa: E402
from app.execution.errors import (  # noqa: E402
    AI_CONNECTION_REFUSED,
    AI_DNS_UNREACHABLE,
    AI_MODEL_MISSING,
    AI_TIMEOUT,
)
from app.execution.store import ExecutionStore  # noqa: E402


def _no_network():
    """Tripwire: any attempt to open an HTTP helper fails the test."""
    return mock.patch.object(
        core, "_http_json",
        side_effect=AssertionError("network attempted in a config-only check"),
    )


class TestProviderReadinessConfigOnly(unittest.TestCase):
    """provider_readiness(): one rule, no network, ever."""

    def test_cases(self):
        cases = [
            # eff, expected provider, expected state
            ({"llm_provider": "gemini", "gemini_api_key": "k"}, "gemini", "ready"),
            ({"llm_provider": "gemini"}, "gemini", "not_configured"),
            # The historical /health-vs-/services disagreement: provider unset
            # with a saved Gemini key. Core normalizes unset -> gemini.
            ({"gemini_api_key": "k"}, "gemini", "ready"),
            ({"llm_provider": "ollama",
              "ollama_base_url": "http://localhost:11434",
              "main_model": "qwen3:0.6b"}, "ollama", "ready"),
            ({"llm_provider": "ollama", "ollama_base_url": ""},
             "ollama", "not_configured"),
            ({"llm_provider": "mystery", "gemini_api_key": "k"},
             "mystery", "not_configured"),
            ({}, "gemini", "not_configured"),
        ]
        for eff, want_provider, want_state in cases:
            with self.subTest(eff=eff), _no_network():
                r = core.provider_readiness(eff)
                self.assertEqual(r["provider"], want_provider)
                self.assertEqual(r["state"], want_state)
                self.assertEqual(bool(r["configured"]), want_state == "ready")
                # route-specific extras stay intact
                self.assertTrue(r["what_will_be_tested"])
                self.assertTrue(r["state_hint"])

    def test_raw_setting_preserved_for_services_display(self):
        r = core.provider_readiness({"gemini_api_key": "k"})  # unset provider
        self.assertIsNone(r["provider_setting"])   # raw value...
        self.assertEqual(r["provider"], "gemini")  # ...normalized value


class TestEndpointAgreement(unittest.TestCase):
    """/health's `ai` block and /services' `ai_provider` rollup must agree.

    Historically the rollup hard-coded its own condition and could read
    `not_configured` while /health read `ready` for the same settings.
    """

    def setUp(self):
        import app.api.server as server

        self.server = server
        self.store = ExecutionStore(Path(tempfile.mkdtemp()) / "diag.sqlite3")
        server._store = self.store
        self.client = TestClient(server.build_server())
        self.addCleanup(self._close)

    def _close(self):
        self.server._store = None
        self.store.close()

    @staticmethod
    def _snapshot():
        """services_snapshot with all network/in-process probes stubbed."""
        empty = {"id": "stub", "state": "unknown"}
        with mock.patch.object(services_health, "check_ollama",
                               return_value=dict(empty, id="ollama")), \
             mock.patch.object(services_health, "check_searxng",
                               return_value=dict(empty, id="searxng")), \
             mock.patch.object(services_health, "check_storage",
                               return_value=dict(empty, id="storage")), \
             mock.patch.object(services_health, "check_erp",
                               return_value=dict(empty, id="erp")):
            return services_health.services_snapshot()

    def _both_surfaces(self, eff):
        with mock.patch.object(ai_settings, "effective_settings",
                               return_value=dict(eff)):
            health = self.client.get("/api/v1/health")
            self.assertEqual(health.status_code, 200)
            snap = self._snapshot()
        return health.json()["ai"], snap["ai_provider"]

    def test_agree_on_ready(self):
        for eff in (
            {"llm_provider": "gemini", "gemini_api_key": "k"},
            {"gemini_api_key": "k"},                    # unset provider case
            {"llm_provider": "ollama",
             "ollama_base_url": "http://localhost:11434"},
            {"llm_provider": "gemini"},                 # not configured
            {"llm_provider": "ollama", "ollama_base_url": ""},
            {"llm_provider": None, "gemini_api_key": None},
        ):
            with self.subTest(eff=eff):
                health_ai, rollup = self._both_surfaces(eff)
                self.assertEqual(health_ai["state"], rollup["state"])
                self.assertEqual(health_ai["provider"], rollup["provider"])

    def test_health_schema_unchanged(self):
        health_ai, _ = self._both_surfaces(
            {"llm_provider": "gemini", "gemini_api_key": "k"})
        self.assertEqual(
            set(health_ai),
            {"provider", "state", "what_will_be_tested", "state_hint"},
        )

    def test_services_rollup_schema_unchanged(self):
        # frontend/src/api/dashboard.ts: {provider, state, settings_url, note}
        _, rollup = self._both_surfaces({"llm_provider": "gemini",
                                         "gemini_api_key": "k"})
        self.assertEqual(
            set(rollup), {"provider", "state", "settings_url", "note"})
        self.assertEqual(rollup["settings_url"], "/#/settings")

    def test_snapshot_top_level_schema_unchanged(self):
        snap = self._snapshot()
        self.assertEqual(set(snap), {"checked_at", "services", "ai_provider"})

    def test_readiness_is_config_only(self):
        """Neither surface may open a socket to answer provider readiness."""
        with _no_network():
            self._both_surfaces({"llm_provider": "gemini", "gemini_api_key": "k"})


class TestAiTestSchemaUnchanged(unittest.TestCase):
    """/ai/test keeps returning the diagnostics dict untouched."""

    def setUp(self):
        import app.api.server as server

        self.server = server
        self.store = ExecutionStore(Path(tempfile.mkdtemp()) / "diag.sqlite3")
        server._store = self.store
        self.client = TestClient(server.build_server())
        self.addCleanup(self._close)
        ai_settings._session_overrides.clear()
        self.addCleanup(ai_settings._session_overrides.clear)

    def _close(self):
        self.server._store = None
        self.store.close()

    def test_ollama_result_passed_through(self):
        canned = {"state": "model_missing", "phases": {"configured": {"ok": True}},
                  "category": "model_missing"}
        with mock.patch("app.ai_diagnostics.check_ollama",
                        return_value=dict(canned)) as probe:
            r = self.client.post("/api/v1/ai/test",
                                 json={"llm_provider": "ollama",
                                       "ollama_base_url": "http://127.0.0.1:1"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), canned)
        probe.assert_called_once()
        # a non-ready test must not silently persist session overrides
        self.assertEqual(ai_settings._session_overrides, {})

    def test_unknown_provider_rejected(self):
        # pydantic's Literal["gemini", "ollama"] rejects before the handler
        r = self.client.post("/api/v1/ai/test", json={"llm_provider": "nope"})
        self.assertEqual(r.status_code, 422)


class _TagsHandler(BaseHTTPRequestHandler):
    """Local fake Ollama: GET /api/tags with a per-test mode + counter."""

    protocol_version = "HTTP/1.1"
    mode = "ok"          # ok | missing-model | garbage
    requests = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        cls = type(self)
        cls.requests += 1
        if cls.mode == "garbage":
            body = b"<html>not json</html>"
            code = 200
        else:
            models = (["nomic-embed-text:latest"] if cls.mode == "missing-model"
                      else ["qwen3:0.6b", "nomic-embed-text:latest"])
            body = json.dumps({"models": [{"name": m} for m in models]}).encode()
            code = 200
        self.send_response(code)
        self.send_header("Content-Type", "text/html" if cls.mode == "garbage"
                         else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


class TestCheckOllamaSingleProbe(unittest.TestCase):
    """/services' Ollama row: ONE request per check, shared presence rule."""

    @classmethod
    def setUpClass(cls):
        _TagsHandler.mode = "ok"
        _TagsHandler.requests = 0
        cls.server = HTTPServer(("127.0.0.1", 0), _TagsHandler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _TagsHandler.mode = "ok"
        _TagsHandler.requests = 0
        # the OLD code re-probed via _http_status/_latency_ms; those must be gone
        self._no_second_probe = mock.patch.object(
            services_health, "_http_status",
            side_effect=AssertionError("duplicate probe path reactivated"))
        self._no_second_probe.start()
        self.addCleanup(self._no_second_probe.stop)

    def _check(self, mode, model="qwen3:0.6b"):
        _TagsHandler.mode = mode
        _TagsHandler.requests = 0
        with mock.patch("app.config.OLLAMA_BASE_URL", self.url), \
             mock.patch("app.config.MAIN_MODEL", model):
            return services_health.check_ollama()

    def test_model_present_is_running_with_one_request(self):
        r = self._check("ok")
        self.assertEqual(r["state"], "running")
        self.assertEqual(_TagsHandler.requests, 1)
        self.assertIn("Test connection", r["next_action"])
        self.assertIn("qwen3:0.6b", r["config"]["available_models"])

    def test_model_missing_is_degraded_with_one_request(self):
        r = self._check("missing-model", model="qwen3:0.6b")
        self.assertEqual(r["state"], "degraded")
        self.assertIn("not installed", r["reason"])
        self.assertEqual(_TagsHandler.requests, 1)

    def test_unreadable_body_is_unknown_not_running(self):
        r = self._check("garbage")
        self.assertEqual(r["state"], "unknown")
        self.assertEqual(_TagsHandler.requests, 1)

    def test_bare_model_name_matches_tagged_variant(self):
        r = self._check("ok", model="qwen3")   # installed as qwen3:0.6b
        self.assertEqual(r["state"], "running")

    def test_not_configured_never_touches_the_network(self):
        with mock.patch("app.config.OLLAMA_BASE_URL", ""):
            r = services_health.check_ollama()
        self.assertEqual(r["state"], "not_configured")
        self.assertEqual(_TagsHandler.requests, 0)

    def test_closed_port_is_unreachable(self):
        closed = HTTPServer(("127.0.0.1", 0), _TagsHandler)
        url = f"http://127.0.0.1:{closed.server_address[1]}"
        closed.server_close()   # released -> connection refused
        with mock.patch("app.config.OLLAMA_BASE_URL", url), \
             mock.patch("app.config.MAIN_MODEL", "qwen3:0.6b"):
            r = services_health.check_ollama()
        self.assertEqual(r["state"], "unreachable")
        self.assertTrue(r["reason"])


class TestFetchOllamaTagsContract(unittest.TestCase):
    def test_dead_port_categories(self):
        s = HTTPServer(("127.0.0.1", 0), _TagsHandler)
        url = f"http://127.0.0.1:{s.server_address[1]}"
        s.server_close()
        status, payload, category = core.fetch_ollama_tags(url, timeout=1.0)
        self.assertIsNone(status)
        self.assertIsNone(payload)
        # Loopback-to-closed-port can surface as refused, a DNS-class
        # failure, or (Windows) a bounded timeout — all mean "no answer".
        # Same tolerance as test_gui16_settings.test_connection_refused.
        self.assertIn(category, (AI_CONNECTION_REFUSED, AI_DNS_UNREACHABLE,
                                 AI_TIMEOUT))

    def test_http_404_keeps_status_and_classifies(self):
        class NotFound(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        s = HTTPServer(("127.0.0.1", 0), NotFound)
        t = threading.Thread(target=s.serve_forever, daemon=True)
        t.start()
        try:
            status, payload, category = core.fetch_ollama_tags(
                f"http://127.0.0.1:{s.server_address[1]}", timeout=1.0)
        finally:
            s.shutdown()
        self.assertEqual(status, 404)
        self.assertIsNone(payload)
        self.assertEqual(category, AI_MODEL_MISSING)

    def test_timeout_is_forwarded(self):
        seen = {}

        def fake_http_json(url, method="GET", body=None, timeout=10.0):
            seen["timeout"] = timeout
            return 200, {"models": []}

        with mock.patch.object(core, "_http_json", side_effect=fake_http_json):
            status, payload, category = core.fetch_ollama_tags(
                "http://127.0.0.1:9", timeout=3.5)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"models": []})
        self.assertIsNone(category)
        self.assertEqual(seen["timeout"], 3.5)


class TestSharedPresenceRule(unittest.TestCase):
    def test_ollama_model_names(self):
        self.assertEqual(
            core.ollama_model_names(
                {"models": [{"name": "a:latest"}, {"model": "b"}, {},
                            "junk", {"name": ""}]}),
            ["a:latest", "b"],
        )
        self.assertEqual(core.ollama_model_names(None), [])
        self.assertEqual(core.ollama_model_names([1, 2]), [])
        self.assertEqual(core.ollama_model_names({"models": None}), [])

    def test_model_present_rule(self):
        names = ["qwen3:0.6b", "nomic-embed-text:latest"]
        self.assertTrue(core.model_present("qwen3:0.6b", names))  # exact
        self.assertTrue(core.model_present("qwen3", names))       # bare tag
        self.assertFalse(core.model_present("llama3", names))
        self.assertFalse(core.model_present("", names))
        self.assertFalse(core.model_present("qwen3", []))

    def test_both_surfaces_share_the_rule(self):
        """ai_diagnostics.check_ollama uses core.model_present too."""
        from app import ai_diagnostics as diag

        tags = {"models": [{"name": "qwen3:0.6b"}]}
        with mock.patch.object(diag, "_http_json",
                               return_value=(200, tags)):
            ready = diag.check_ollama("http://127.0.0.1:1", "qwen3",
                                      with_generation=False)
            missing = diag.check_ollama("http://127.0.0.1:1", "qwen9",
                                        with_generation=False)
        self.assertTrue(ready["phases"]["model_present"]["ok"])
        self.assertEqual(ready["state"], "partial")  # no generation check
        self.assertEqual(missing["state"], "model_missing")


if __name__ == "__main__":
    unittest.main()
