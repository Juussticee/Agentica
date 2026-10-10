"""GUI-16 tests: AI settings, readiness, and provider diagnostics.

All tests are deterministic; provider checks use a local fake HTTP server or
monkeypatched diagnostics. One optional live test is clearly marked and
skipped unless OLLAMA_LIVE=1.
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

from app import ai_diagnostics as diag  # noqa: E402
from app import ai_settings  # noqa: E402


def fresh_settings():
    td = tempfile.mkdtemp()
    p = Path(td) / "ai_settings.json"
    with mock.patch.object(ai_settings, "SETTINGS_PATH", p):
        ai_settings._session_overrides.clear()
        yield p


class TestPrecedenceAndMasking(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch.object(
            ai_settings, "SETTINGS_PATH",
            Path(tempfile.mkdtemp()) / "ai_settings.json",
        )
        self.patcher.start()
        ai_settings._session_overrides.clear()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(ai_settings._session_overrides.clear)

    def test_env_only(self):
        with mock.patch.object(ai_settings, "load_persisted", return_value={}):
            eff = ai_settings.effective_settings()
        self.assertEqual(eff["source"] if "source" in eff else None, None)
        masked = ai_settings.masked_settings()
        self.assertNotIn("gemini_api_key", masked)  # full key never returned
        self.assertIn("gemini_api_key_masked", masked)
        self.assertEqual(masked["source"]["gemini_api_key"], "environment")

    def test_saved_beats_env(self):
        ai_settings.persist({"main_model": "saved-model"})
        eff = ai_settings.effective_settings()
        self.assertEqual(eff["main_model"], "saved-model")
        masked = ai_settings.masked_settings()
        self.assertEqual(masked["source"]["main_model"], "saved")

    def test_session_beats_saved(self):
        ai_settings.persist({"main_model": "saved-model"})
        ai_settings.set_session_overrides({"main_model": "session-model"})
        self.assertEqual(ai_settings.effective_settings()["main_model"], "session-model")
        self.assertEqual(ai_settings.masked_settings()["source"]["main_model"], "session")

    def test_masked_key_never_full(self):
        ai_settings.set_session_overrides({"gemini_api_key": "sk-test-1234567890abcdef"})
        masked = ai_settings.masked_settings()
        self.assertNotIn("sk-test-1234567890abcdef", json.dumps(masked))

    def test_rejects_unknown_keys(self):
        ai_settings.persist({"evil_key": "x", "main_model": "ok"})
        self.assertNotIn("evil_key", ai_settings.load_persisted())
        self.assertIn("main_model", ai_settings.load_persisted())

    def test_clear_deletes_file(self):
        ai_settings.persist({"main_model": "m"})
        self.assertTrue(ai_settings.SETTINGS_PATH.exists())
        ai_settings.clear_persisted()
        self.assertFalse(ai_settings.SETTINGS_PATH.exists())

    def test_corrupt_file_treated_as_absent(self):
        ai_settings.SETTINGS_PATH.write_text("{not json", encoding="utf-8")
        self.assertEqual(ai_settings.load_persisted(), {})


class TestAiDiagnosticsOllama(unittest.TestCase):
    """Fake Ollama HTTP server for bounded deterministic diagnostics tests."""

    @classmethod
    def setUpClass(cls):
        cls.mode = "ok"

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"   # avoid stale keep-alive/10054 resets

            def log_message(self, *a):
                pass

            def _reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/api/tags"):
                    if cls.mode == "embed-only":
                        models = ["nomic-embed-text:latest"]
                    else:  # ok / garbage / any other POST-only scenario
                        models = ["qwen3:0.6b", "nomic-embed-text:latest"]
                    self._reply(200, {"models": [{"name": n} for n in models]})
                else:
                    self._reply(404, {})

            def do_POST(self):
                # Consume the request body before replying: with HTTP/1.1
                # keep-alive an unread body desyncs the next response.
                try:
                    self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                except Exception:
                    pass
                if self.path.startswith("/api/generate"):
                    if cls.mode == "garbage":
                        self._reply(200, {"response": "I don't understand."})
                    else:
                        self._reply(200, {"response": "**OLLAMA_TEST_OK**",
                                          "done_reason": "stop"})
                else:
                    self._reply(404, {})

        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.t = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_full_ready(self):
        r = diag.check_ollama(f"http://127.0.0.1:{self.port}", "qwen3:0.6b")
        self.assertEqual(r["state"], "ready")
        self.assertIn("verified_at", r)
        for phase in ("configured", "endpoint_parsed", "reachable",
                      "authorized", "model_present", "generation_ok", "verified"):
            self.assertTrue(r["phases"][phase]["ok"], phase)

    def test_generation_only_model_missing(self):
        self.__class__.mode = "embed-only"
        self.addCleanup(lambda: setattr(self.__class__, "mode", "ok"))
        r = diag.check_ollama(f"http://127.0.0.1:{self.port}", "qwen3:0.6b")
        self.assertEqual(r["state"], "model_missing")
        # available models surfaced so the user can pick one
        self.assertIn("nomic-embed-text:latest",
                      r["phases"]["model_present"]["available_models"])
        # the historical failure mode: an embeddings model is NOT a chat model
        self.assertIn("embedding", r.get("hint", ""))

    def test_garbage_generation_reported_not_success(self):
        self.__class__.mode = "garbage"
        self.addCleanup(lambda: setattr(self.__class__, "mode", "ok"))
        r = diag.check_ollama(f"http://127.0.0.1:{self.port}", "qwen3:0.6b")
        self.assertEqual(r["state"], "error")
        self.assertEqual(r["category"], "bad_response")
        self.assertFalse(r["phases"]["generation_ok"]["ok"])

    def test_connection_refused(self):
        r = diag.check_ollama("http://127.0.0.1:9", "any")
        self.assertIn(r["state"], ("not_configured", "error"))
        self.assertIn(r["category"], ("dns_unreachable", "connection_refused"))

    def test_malformed_endpoint(self):
        r = diag.check_ollama("ftp://not-http", "any")
        self.assertEqual(r["state"], "error")
        self.assertEqual(r["category"], "malformed_endpoint")

    def test_not_configured(self):
        r = diag.check_ollama("", "")
        self.assertEqual(r["state"], "not_configured")

    def test_tagless_model_name_matches(self):
        r = diag.check_ollama(f"http://127.0.0.1:{self.port}", "qwen3", with_generation=False)
        self.assertEqual(r["state"], "partial")
        self.assertTrue(r["phases"]["model_present"]["ok"])

    def test_skip_generation_is_partial_not_ready(self):
        r = diag.check_ollama(f"http://127.0.0.1:{self.port}", "qwen3:0.6b",
                              with_generation=False)
        self.assertEqual(r["state"], "partial")
        self.assertIn("Generation check skipped", r["note"])


@unittest.skipUnless(
    __import__("os").getenv("OLLAMA_LIVE") == "1",
    "live provider test — set OLLAMA_LIVE=1 to run (uses the real endpoint)",
)
class TestLiveOllama(unittest.TestCase):
    def test_live_ollama_ready(self):
        from app.config import MAIN_MODEL, OLLAMA_BASE_URL
        r = diag.check_ollama(OLLAMA_BASE_URL, MAIN_MODEL)
        self.assertEqual(r["state"], "ready", json.dumps(r, indent=2))


class TestSettingsApi(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.td = Path(tempfile.mkdtemp())
        import app.api.server as server

        self.server = server
        self._store = ExecutionStore(self.td / "t.sqlite3")
        self.server._store = self._store
        self._persist = mock.patch.object(
            ai_settings, "SETTINGS_PATH", self.td / "ai_settings.json"
        )
        self._persist.start()
        ai_settings._session_overrides.clear()
        from fastapi.testclient import TestClient

        self.client = TestClient(server.build_server())

    def tearDown(self):
        self._persist.stop()
        ai_settings._session_overrides.clear()
        ai_settings.clear_persisted()
        self._store.close()
        self.server._store = None

    def test_get_masked(self):
        r = self.client.get("/api/v1/ai/settings")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertNotIn("gemini_api_key", body)
        self.assertIn("gemini_api_key_masked", body)

    def test_put_persists_and_get_reflects(self):
        r = self.client.put("/api/v1/ai/settings",
                            json={"main_model": "saved-via-api", "llm_provider": "ollama"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["main_model"], "saved-via-api")
        r2 = self.client.get("/api/v1/ai/settings")
        self.assertEqual(r2.json()["source"]["main_model"], "saved")

    def test_delete_clears(self):
        self.client.put("/api/v1/ai/settings", json={"main_model": "x"})
        r = self.client.delete("/api/v1/ai/settings")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["cleared"], True)
        self.assertEqual(ai_settings.load_persisted(), {})

    def test_test_endpoint_reports_missing_model(self):
        # point at a dead port so connectivity fails fast and honestly
        r = self.client.post("/api/v1/ai/test",
                             json={"llm_provider": "ollama",
                                   "ollama_base_url": "http://127.0.0.1:9",
                                   "main_model": "none:latest",
                                   "with_generation": False})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn(body["state"], ("not_configured", "error"))

    def test_test_endpoint_does_not_persist(self):
        r = self.client.post("/api/v1/ai/test",
                             json={"main_model": "tested-not-saved",
                                   "ollama_base_url": "http://127.0.0.1:9"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(ai_settings.load_persisted(), {})

    def test_health_uses_effective_settings(self):
        self.client.put("/api/v1/ai/settings",
                        json={"llm_provider": "ollama", "main_model": "m1",
                              "ollama_base_url": "http://x:1"})
        r = self.client.get("/api/v1/health")
        self.assertEqual(r.json()["ai"]["provider"], "ollama")


from app.execution.store import ExecutionStore  # noqa: E402  (used above)


if __name__ == "__main__":
    unittest.main(verbosity=2)
