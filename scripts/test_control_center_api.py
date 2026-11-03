"""Control Center projections through the REAL API (2026-10-30) — offline.

Proves the management UI can only show true things:

  * `/api/v1/agents` lists the REGISTERED specialists (registry-backed), each
    with its tools, permissions and a config-only availability verdict —
    `coding`/`data` are DISABLED without a workspace, not silently "available";
  * `general`/`feasibility` become unavailable when no provider is configured;
  * the payload contains no secret value and no absolute filesystem path;
  * `/api/v1/skills` reports documented procedures with real file size/mtime
    and states that nothing is installable;
  * `/api/v1/memory` names the sections that DO persist and the ones that do
    not (`preferences`, `procedures`), instead of faking a store;
  * `/api/v1/dashboard` counts come from the store (executions/approvals/
    events), the attention queue holds real failed/paused rows, and the
    activity timeline is merged from recorded events only;
  * `/api/v1/executions` filters are additive and backward compatible, and an
    unknown status is a 422 rather than an empty list;
  * approvals now name the requesting agent, without changing any decision
    behaviour.

Temp SQLite + temp workspace, patched provider readiness, no model, no
network, no Chroma.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def os_utime(path, timestamp):
    os.utime(path, (timestamp, timestamp))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

from app.execution.store import ExecutionStore  # noqa: E402

READY = {"provider": "ollama", "provider_setting": "ollama", "configured": True,
         "state": "ready",
         "what_will_be_tested": "Ollama (http://localhost:11434), model qwen3:0.6b",
         "state_hint": "Configured. Agentica will use it for AI-backed steps."}
NOT_READY = {"provider": "gemini", "provider_setting": "gemini",
             "configured": False, "state": "not_configured",
             "what_will_be_tested": "Gemini, model gemini-3-flash-preview",
             "state_hint": "No API key or endpoint set."}


class ControlCenterCase(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.api.server as server

        self.server = server
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_store = server._store
        self.store = ExecutionStore(Path(self._tmp.name) / "cc.sqlite3")
        server._store = self.store
        self.addCleanup(self._restore)
        self.addCleanup(self.store.close)
        self.client = TestClient(server.build_server())

        readiness = mock.patch("app.diagnostics_core.provider_readiness",
                               return_value=dict(READY))
        readiness.start()
        self.addCleanup(readiness.stop)

    def _restore(self):
        self.server._store = self._orig_store

    # -- seeding -----------------------------------------------------------

    def seed(self, eid="e1", question="What does chapter 3 say about propolis?",
             status="completed", route="books", error=None, plan=None,
             events=()):
        self.store.create_execution(eid, question)
        self.store.update_execution(
            eid, status=status, route=route, answer="an answer",
            error=error, plan=plan)
        for index, (type_, agent, message) in enumerate(events, start=1):
            self.store.append_event(eid, index, type_, message, agent_id=agent)

    def get_json(self, path):
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()


class TestAgentsCatalog(ControlCenterCase):
    def test_lists_every_registered_agent_with_its_contract(self):
        body = self.get_json("/api/v1/agents")
        ids = [a["id"] for a in body["agents"]]
        from app.agents import registry

        self.assertEqual(sorted(ids), sorted(registry.AGENTS))
        self.assertEqual(body["counts"]["registered"], len(registry.AGENTS))
        data = next(a for a in body["agents"] if a["id"] == "data")
        self.assertTrue(data["implemented"])
        self.assertTrue(data["tools"])
        self.assertEqual(set(data["permissions"]),
                         {"network", "filesystem", "writes", "approval_required"})
        self.assertFalse(data["permissions"]["writes"])
        self.assertEqual(data["planner_capability"], "data")
        self.assertIn("rejects", data)

    def test_workspace_agents_are_disabled_without_a_workspace(self):
        with mock.patch("app.config.CODING_WORKSPACE_ROOT", ""):
            body = self.get_json("/api/v1/agents")
        by_id = {a["id"]: a for a in body["agents"]}
        for agent_id in ("coding", "data"):
            self.assertEqual(by_id[agent_id]["availability"], "disabled")
            self.assertIn("CODING_WORKSPACE_ROOT",
                          by_id[agent_id]["availability_reason"])

    def test_workspace_agents_become_available_with_a_real_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("app.config.CODING_WORKSPACE_ROOT", tmp):
                body = self.get_json("/api/v1/agents")
        by_id = {a["id"]: a for a in body["agents"]}
        self.assertEqual(by_id["data"]["availability"], "available")
        self.assertEqual(by_id["coding"]["availability"], "available")
        self.assertIn("approved", by_id["coding"]["availability_reason"])

    def test_missing_workspace_directory_is_unavailable_not_available(self):
        missing = str(Path(self._tmp.name) / "nope")
        with mock.patch("app.config.CODING_WORKSPACE_ROOT", missing):
            body = self.get_json("/api/v1/agents")
        by_id = {a["id"]: a for a in body["agents"]}
        self.assertEqual(by_id["data"]["availability"], "disabled")
        self.assertIn("does not exist", by_id["data"]["availability_reason"])

    def test_provider_dependent_agents_go_unavailable_without_a_provider(self):
        with mock.patch("app.diagnostics_core.provider_readiness",
                        return_value=dict(NOT_READY)):
            body = self.get_json("/api/v1/agents")
        by_id = {a["id"]: a for a in body["agents"]}
        self.assertEqual(by_id["general"]["availability"], "unavailable")
        self.assertEqual(by_id["feasibility"]["availability"], "unavailable")
        # These need no provider at all.
        self.assertEqual(by_id["conversational"]["availability"], "available")
        self.assertEqual(by_id["unsupported"]["availability"], "available")

    def test_recent_activity_comes_from_real_executions(self):
        self.seed()
        self.seed(eid="e2", question="Compare two frameworks and pick one",
                  status="completed", route="multi",
                  plan={"tasks": [
                      {"task_id": "t1", "capability": "data",
                       "objective": "total by region", "depends_on": [],
                       "status": "completed", "result": "table"},
                  ]})
        body = self.get_json("/api/v1/agents")
        by_id = {a["id"]: a for a in body["agents"]}
        self.assertEqual(by_id["books"]["recent_activity"][0]["execution_id"],
                         "e1")
        data_activity = by_id["data"]["recent_activity"]
        self.assertEqual(data_activity[0]["execution_id"], "e2")
        self.assertEqual(data_activity[0]["task_status"], "completed")
        self.assertEqual(by_id["erp"]["recent_activity"], [])

    def test_planned_roles_are_listed_separately_not_as_agents(self):
        body = self.get_json("/api/v1/agents")
        planned = {p["id"] for p in body["planned"]}
        self.assertIn("personal_assistant", planned)
        self.assertIn("automation", planned)
        self.assertNotIn("personal_assistant", [a["id"] for a in body["agents"]])

    def test_no_secret_or_absolute_path_is_exposed(self):
        with mock.patch("app.config.CODING_WORKSPACE_ROOT",
                        str(Path(self._tmp.name))):
            raw = self.client.get("/api/v1/agents").text
        lowered = raw.lower()
        for forbidden in ("api_key", "apikey", "token", "password",
                          "gsk_", "sk-", "ghp_"):
            self.assertNotIn(forbidden, lowered)
        self.assertNotIn(str(PROJECT_ROOT), raw)
        self.assertNotIn(str(Path.home()), raw)

    def test_unknown_agent_is_a_structured_404(self):
        response = self.client.get("/api/v1/agents/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertIn("human_message", response.json()["detail"])

    def test_agent_detail_matches_the_list_entry(self):
        listed = next(a for a in self.get_json("/api/v1/agents")["agents"]
                      if a["id"] == "research")
        detail = self.get_json("/api/v1/agents/research")["agent"]
        self.assertEqual(detail, listed)


class TestSkills(ControlCenterCase):
    def test_documented_skills_are_real_files_and_not_installable(self):
        body = self.get_json("/api/v1/skills")
        self.assertFalse(body["can_install"])
        self.assertGreater(body["count"], 0)
        for skill in body["skills"]:
            self.assertTrue(skill["installed"], skill["path"])
            self.assertTrue(skill["size_bytes"] > 0)
            self.assertFalse(skill["executable"])
            self.assertEqual(skill["required_tools"], [])
            self.assertIn("documentation", skill["permissions"].lower())
        self.assertIn("no skill runtime", body["note"].lower())


class TestMemory(ControlCenterCase):
    def test_persisted_and_unimplemented_sections_are_distinguished(self):
        self.seed()
        body = self.get_json("/api/v1/memory")
        sections = {s["id"]: s for s in body["sections"]}
        # 2026-10-30 (learning): preferences and procedures are now backed by
        # the `lessons` table, so an empty store reports "empty" with an
        # explanation — no longer "not_implemented", and still no fabrication.
        self.assertEqual(sections["preferences"]["status"], "empty")
        self.assertIsNone(sections["preferences"]["provenance"])
        self.assertIn("never inferred", sections["preferences"]["summary"])
        self.assertEqual(sections["procedures"]["status"], "empty")
        self.assertIn("lessons", sections["procedures"]["provenance"])
        self.assertEqual(sections["procedures"]["items"], [])
        self.assertEqual(sections["executions"]["status"], "available")
        self.assertEqual(sections["executions"]["counts"]["total"], 1)
        self.assertEqual(sections["executions"]["items"][0]["id"], "e1")
        self.assertIn("sqlite", sections["executions"]["provenance"].lower())

    def test_persisted_lessons_show_up_in_the_learning_sections(self):
        self.seed()
        self.store.create_lesson({
            "lesson_id": "L1", "status": "active", "category": "preference",
            "capability": "research", "scope": "research comparisons",
            "problem": "Answer lacked benchmarks",
            "root_cause": "Instruction did not ask for them",
            "corrective_behavior": "Include benchmarks and VRAM figures",
            "validation": "Answer states a benchmark figure per model",
            "confidence": 0.9, "evidence": ["user correction"],
            "source_execution_id": "e1", "origin": "user_feedback",
        })
        self.store.create_lesson({
            "lesson_id": "L2", "status": "candidate", "category": "procedure",
            "capability": "coding", "scope": "patching",
            "problem": "Patch touched unrelated lines",
            "root_cause": None,
            "corrective_behavior": "Keep hunks minimal",
            "validation": "diff touches one function",
            "confidence": 0.5, "evidence": ["execution failure"],
            "source_execution_id": "e1", "origin": "execution_failure",
        })
        sections = {s["id"]: s for s in self.get_json("/api/v1/memory")["sections"]}
        prefs = sections["preferences"]
        self.assertEqual(prefs["status"], "available")
        self.assertEqual([p["id"] for p in prefs["items"]], ["L1"])
        self.assertIn("lessons", prefs["provenance"])
        procs = sections["procedures"]
        self.assertEqual(procs["status"], "available")
        self.assertEqual(procs["counts"]["active"], 1)
        self.assertEqual(procs["counts"]["candidate"], 1)
        ids = {i["id"]: i for i in procs["items"]}
        self.assertEqual(ids["L2"]["behavior"], "Keep hunks minimal")
        self.assertEqual(ids["L2"]["origin"], "execution_failure")

    def test_knowledge_section_reflects_the_corpus_on_disk(self):
        body = self.get_json("/api/v1/memory")
        knowledge = next(s for s in body["sections"] if s["id"] == "knowledge")
        from app.config import BOOKS_DIR

        if BOOKS_DIR.is_dir() and any(BOOKS_DIR.iterdir()):
            self.assertEqual(knowledge["status"], "available")
            self.assertTrue(knowledge["items"])
            self.assertTrue(knowledge["items"][0]["path"].startswith("knowledge/"))
        else:
            self.assertEqual(knowledge["status"], "empty")

    def test_empty_store_reports_empty_not_fabricated(self):
        body = self.get_json("/api/v1/memory")
        sections = {s["id"]: s for s in body["sections"]}
        self.assertEqual(sections["executions"]["status"], "empty")
        self.assertEqual(sections["approvals"]["status"], "empty")


class TestDashboard(ControlCenterCase):
    def test_counts_match_the_store(self):
        self.seed(eid="e1", status="completed")
        self.seed(eid="e2", status="failed",
                  error={"category": "timeout",
                         "human_message": "The provider timed out.",
                         "why": "no response", "recovery": "retry",
                         "recoverable": True})
        self.seed(eid="e3", status="waiting_for_approval", route="research")
        body = self.get_json("/api/v1/dashboard")
        counts = self.store.counts()
        self.assertEqual(body["executions"]["total"],
                         counts["executions"]["total"])
        self.assertEqual(body["executions"]["failed"], 1)
        self.assertEqual(body["executions"]["waiting_approval"], 1)
        self.assertEqual(body["executions"]["active"], 0)
        self.assertEqual(body["agents"]["registered"],
                         len(__import__("app.agents.registry",
                                        fromlist=["AGENTS"]).AGENTS))

    def test_attention_queue_holds_real_failed_and_paused_rows(self):
        self.seed(eid="e1", status="failed",
                  error={"category": "provider_unavailable",
                         "human_message": "The AI provider is unavailable.",
                         "why": "no key", "recovery": "configure", "recoverable": True})
        self.seed(eid="e2", status="completed")
        self.seed(eid="e3", status="waiting_for_approval", route="research")
        attention = self.get_json("/api/v1/dashboard")["attention"]
        ids = [a["id"] for a in attention]
        self.assertIn("e1", ids)
        self.assertIn("e3", ids)
        self.assertNotIn("e2", ids)
        e1 = next(a for a in attention if a["id"] == "e1")
        self.assertEqual(e1["category"], "provider_unavailable")
        self.assertEqual(e1["human_message"], "The AI provider is unavailable.")

    def test_activity_is_merged_from_recorded_events_only(self):
        self.seed(eid="e1", events=[
            ("execution.started", "supervisor", "Routing the request"),
            ("agent.completed", "books", "Answered from the books"),
        ])
        activity = self.get_json("/api/v1/dashboard")["activity"]
        self.assertEqual([a["type"] for a in activity][:2],
                         ["agent.completed", "execution.started"])
        self.assertEqual(activity[0]["execution_id"], "e1")
        self.assertEqual(activity[0]["agent_id"], "books")

    def test_empty_dashboard_is_honest(self):
        body = self.get_json("/api/v1/dashboard")
        self.assertEqual(body["executions"]["total"], 0)
        self.assertEqual(body["attention"], [])
        self.assertEqual(body["activity"], [])
        self.assertEqual(body["approvals"]["pending"], 0)
        self.assertEqual(body["skills"]["installed"], body["skills"]["total"])

    def test_learning_endpoint_is_read_only_and_honest_when_empty(self):
        body = self.get_json("/api/v1/learning")
        self.assertEqual(body["total"], 0)
        self.assertEqual(body["active"], [])
        self.assertEqual(body["repeated_patterns"], [])
        self.assertIn("does not prove behaviour changed", body["note"])

    def test_learning_endpoint_reports_stored_lessons_without_secrets(self):
        self.seed()
        self.store.create_lesson({
            "lesson_id": "L1", "status": "active", "category": "preference",
            "capability": "research", "scope": "research comparisons",
            "problem": "Answer lacked benchmarks",
            "corrective_behavior": "Include benchmarks",
            "confidence": 0.9, "evidence": ["user correction"],
            "source_execution_id": "e1", "origin": "user_feedback",
        })
        self.store.create_lesson({
            "lesson_id": "L2", "status": "candidate", "category": "preference",
            "capability": "research", "scope": "depth",
            "problem": "Answer lacked licensing",
            "corrective_behavior": "Include licensing",
            "confidence": 0.5, "evidence": ["user correction"],
            "origin": "user_feedback",
        })
        body = self.get_json("/api/v1/learning")
        self.assertEqual(body["total"], 2)
        self.assertEqual([l["id"] for l in body["active"]], ["L1"])
        self.assertEqual([l["id"] for l in body["candidates"]], ["L2"])
        self.assertEqual(body["active"][0]["source_execution_id"], "e1")
        # Two lessons for the same capability are a repeated pattern.
        self.assertEqual(body["repeated_patterns"][0]["capability"], "research")
        raw = self.client.get("/api/v1/learning").text
        self.assertNotIn(str(PROJECT_ROOT), raw)

    def test_payload_is_json_serialisable_and_path_free(self):
        self.seed()
        raw = self.client.get("/api/v1/dashboard").text
        json.loads(raw)
        self.assertNotIn(str(PROJECT_ROOT), raw)


class TestExecutionFilters(ControlCenterCase):
    def setUp(self):
        super().setUp()
        self.seed(eid="a1", question="What does chapter 3 say about propolis?",
                  status="completed", route="books")
        self.seed(eid="a2", question="Fix the off-by-one bug in calc.py",
                  status="failed", route="coding")
        self.seed(eid="a3", question="Compare the AI risk frameworks",
                  status="completed", route="multi")

    def test_no_parameters_keeps_the_original_contract(self):
        body = self.get_json("/api/v1/executions")
        self.assertEqual(body["count"], 3)
        self.assertEqual(body["total"], 3)
        self.assertEqual({e["id"] for e in body["executions"]},
                         {"a1", "a2", "a3"})

    def test_status_filter(self):
        body = self.get_json("/api/v1/executions?status=failed")
        self.assertEqual([e["id"] for e in body["executions"]], ["a2"])
        self.assertEqual(body["total"], 3)

    def test_text_query_filter_is_case_insensitive(self):
        body = self.get_json("/api/v1/executions?q=PROPOLIS")
        self.assertEqual([e["id"] for e in body["executions"]], ["a1"])
        self.assertEqual(self.get_json("/api/v1/executions?q=zzz")["count"], 0)

    def test_limit_and_date_filters(self):
        self.assertEqual(self.get_json("/api/v1/executions?limit=2")["count"], 2)
        future = "2999-01-01T00:00:00+00:00"
        self.assertEqual(
            self.get_json(f"/api/v1/executions?since={future}")["count"], 0)
        past = "2000-01-01T00:00:00+00:00"
        self.assertEqual(
            self.get_json(f"/api/v1/executions?since={past}")["count"], 3)

    def test_malformed_date_bound_is_rejected_not_silently_empty(self):
        response = self.client.get("/api/v1/executions?since=yesterday")
        self.assertEqual(response.status_code, 422)
        self.assertIn("human_message", response.json()["detail"])

    def test_unknown_status_is_rejected_not_silently_empty(self):
        response = self.client.get("/api/v1/executions?status=bogus")
        self.assertEqual(response.status_code, 422)
        self.assertIn("human_message", response.json()["detail"])

    def test_activity_endpoint_uses_the_registered_inventory(self):
        body = self.get_json("/api/v1/agents/activity")
        nodes = {a["node"] for a in body["agents"]}
        self.assertIn("data", nodes)
        self.assertIn("coding", nodes)
        self.assertIn("qa", nodes)


class TestLearningEvaluation(ControlCenterCase):
    """The Learning page's impact section reads report files, nothing more."""

    def _report(self, delta_pp=25.0, version="learning_eval_v1"):
        return {
            "benchmark_version": version, "harness_version": 1,
            "mode": "offline_deterministic", "live_model": False,
            "provider": {"description": "scripted"},
            "started_at": "2026-10-30T10:00:00+00:00",
            "finished_at": "2026-10-30T10:00:04+00:00",
            "task_count": 2,
            "summary": {
                "tasks": 2,
                "baseline": {"requirements": 4, "satisfied": 2,
                             "completeness": 0.5, "qa_findings": 0},
                "learning": {"requirements": 4, "satisfied": 3,
                             "completeness": 0.75, "qa_findings": 1},
                "delta": {"completeness_pp": delta_pp,
                          "requirements_satisfied": 1, "qa_findings": 1,
                          "succeeded": 0, "empty_answers": 0,
                          "regressions": 1, "improved_tasks": 1,
                          "regressed_tasks": 1, "unchanged_tasks": 0},
                "regressions": [{"task_id": "t2", "kind": "qa_regression"}],
                "by_capability": [], "lessons": [],
            },
            "tasks": [{
                "task_id": "t1", "capability": "general", "request": "q",
                "baseline_execution_id": "b1", "learning_execution_id": "l1",
                "lesson_fixtures": ["l1"],
                "comparison": {"outcome": "improved",
                               "completeness_baseline": 0.5,
                               "completeness_learning": 1.0,
                               "completeness_delta_pp": 50.0,
                               "applied_lessons": ["l1"],
                               "retrieved_not_applied": [],
                               "regressions": []},
            }],
            "notes": ["offline mode"],
        }

    def test_missing_report_directory_is_reported_as_unavailable(self):
        import app.api.control_center as cc

        missing = Path(self._tmp.name) / "no_eval"
        with mock.patch.object(cc, "_eval_dir", return_value=missing):
            body = self.get_json("/api/v1/learning/evaluation")
        self.assertFalse(body["available"])
        self.assertIsNone(body["latest"])
        self.assertIn("No benchmark has been run", body["reason"])
        self.assertIn("eval_learning.py", body["run_command"])

    def test_latest_report_and_history_are_read_from_the_files(self):
        import app.api.control_center as cc

        directory = Path(self._tmp.name) / "eval"
        directory.mkdir()
        older = directory / "a.json"
        newer = directory / "b.json"
        older.write_text(json.dumps(self._report(10.0, "v-old")),
                         encoding="utf-8")
        newer.write_text(json.dumps(self._report(25.0, "v-new")),
                         encoding="utf-8")
        os_utime(older, 1_700_000_000)
        os_utime(newer, 1_700_000_100)
        with mock.patch.object(cc, "_eval_dir", return_value=directory):
            body = self.get_json("/api/v1/learning/evaluation")
        self.assertTrue(body["available"])
        self.assertEqual(body["latest"]["benchmark_version"], "v-new")
        self.assertEqual([h["benchmark_version"] for h in body["history"]],
                         ["v-new", "v-old"])
        self.assertEqual(body["latest"]["tasks"][0]["task_id"], "t1")
        self.assertEqual(
            body["latest"]["tasks"][0]["applied_lessons"], ["l1"])
        self.assertEqual(body["latest"]["summary"]["delta"]["completeness_pp"],
                         25.0)
        self.assertNotIn(str(PROJECT_ROOT), json.dumps(body))

    def test_a_malformed_report_is_skipped_and_named(self):
        import app.api.control_center as cc

        directory = Path(self._tmp.name) / "eval2"
        directory.mkdir()
        (directory / "broken.json").write_text("{not json", encoding="utf-8")
        (directory / "wrong.json").write_text("{\"hello\": 1}",
                                               encoding="utf-8")
        (directory / "good.json").write_text(json.dumps(self._report()),
                                              encoding="utf-8")
        with mock.patch.object(cc, "_eval_dir", return_value=directory):
            body = self.get_json("/api/v1/learning/evaluation")
        self.assertTrue(body["available"])
        skipped = {entry["file"] for entry in body["skipped"]}
        self.assertEqual(skipped, {"broken.json", "wrong.json"})


class TestDateFilterSemantics(unittest.TestCase):
    """The range filter must compare INSTANTS, not strings.

    The store writes aware ISO strings, sometimes with microseconds and
    sometimes at second precision; the UI sends `Date#toISOString()`
    (milliseconds + `Z`). Lexical comparison silently drops a row sitting
    exactly on the bound, which is the bug these tests pin.
    """

    def test_boundary_row_is_kept_when_the_instants_are_equal(self):
        from app.api.control_center import filter_executions

        rows = [{"id": "r1", "created_at": "2026-10-30T09:00:00+00:00"}]
        kept = filter_executions(rows, since="2026-10-30T09:00:00.000Z")
        self.assertEqual([r["id"] for r in kept], ["r1"])

    def test_microseconds_and_mixed_offsets_compare_correctly(self):
        from app.api.control_center import filter_executions

        rows = [
            {"id": "old", "created_at": "2026-10-29T23:59:59.999999+00:00"},
            {"id": "new", "created_at": "2026-10-30T00:00:00.000001+00:00"},
        ]
        kept = filter_executions(rows, since="2026-10-30T00:00:00.000Z")
        self.assertEqual([r["id"] for r in kept], ["new"])

    def test_row_without_a_parsable_timestamp_is_excluded_from_a_range(self):
        from app.api.control_center import filter_executions

        rows = [{"id": "bad", "created_at": "not-a-date"}]
        self.assertEqual(
            filter_executions(rows, since="2026-01-01T00:00:00Z"), [])
        self.assertEqual(
            filter_executions(rows, until="2026-01-01T00:00:00Z"), [])
        # No range requested → no filtering, so the row stays visible.
        self.assertEqual(len(filter_executions(rows)), 1)


class TestApprovalAgentAttribution(ControlCenterCase):
    def test_requesting_agent_is_added_without_changing_decisions(self):
        self.store.create_execution("e1", "Compare two frameworks")
        created = self.store.create_or_reuse_approval(
            operation="research", description="Search the web for X",
            payload={"question": "Compare two frameworks", "route": "research"},
            execution_id="e1")
        pending = self.get_json("/api/v1/approvals")["approvals"]
        self.assertEqual(pending[0]["requesting_agent"], "research")
        self.assertEqual(pending[0]["approval_id"], created["approval_id"])
        detail = self.get_json(
            f"/api/v1/approvals/{created['approval_id']}")["approval"]
        self.assertEqual(detail["requesting_agent"], "research")
        # The decision path is untouched: approve still works and stays bound.
        decided = self.client.post(
            f"/api/v1/approvals/{created['approval_id']}/approve")
        self.assertEqual(decided.status_code, 200, decided.text)
        self.assertEqual(decided.json()["decision"]["outcome"], "approved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
