"""Multi-approval plan through the REAL API (2026-10-29) — offline.

Proves the lifecycle you get when ONE execution contains TWO research tasks,
each needing its own decision, driven entirely through the live endpoints:

  POST /api/v1/executions                     -> pauses at the first gate
  GET  /api/v1/approvals                      -> the first pending request
  POST /api/v1/approvals/{id}/approve|resume  -> task 1 runs, plan pauses
                                                 again for task 2's OWN request
  POST /api/v1/approvals/{id2}/approve|resume -> task 2 runs, synthesis runs,
                                                 execution `completed`

Real code paths exercised end to end: the supervisor's routing, the planner's
parse+validate of the model's plan, the Phase 8 dispatcher, the research
approval gate, the runner's plan continuation, the persisted store, the
execution status machine and the deterministic QA gate.

Two seams only, both at provider/external boundaries:
  * `app.llm_client.get_llm` — answers the REAL planner prompt with a fixed
    plan and the REAL synthesis prompt with fixed text. Any other prompt
    raises, so an unexpected model call fails the suite instead of reaching a
    provider (this is also the regression guard for the earlier accidental
    live call).
  * `app.graph.research_agent` — the research node after the gate. It returns
    a distinct result per call, so "which task ran" is provable.
SearXNG/crawl are additionally stubbed to raise: no code here may search.

Offline by construction: no Chroma, no model, no network.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

import app.graph as gm  # noqa: E402
from app.execution.store import ExecutionStore  # noqa: E402
from app.state import with_agent_output  # noqa: E402
from app.tools import crawl_extractor, searxng_client  # noqa: E402

# Two research needs, no web authorization in the message (no "search the web"
# and no current-information cue), multi-objective enough for the REAL
# complexity heuristic (>=12 words, >=2 coordinating signals) and routed to
# books by the corpus anchor "risk management framework".
QUESTION = ("Compare the AI risk management framework with ISO 42001 and then "
            "evaluate which one suits our startup better")

# A valid plan: research -> research -> synthesis. Objectives are real
# sub-requests, so each research task exists for its own reason.
PLAN_JSON = json.dumps({
    "goal": "Compare the AI RMF with ISO 42001 for our startup",
    "tasks": [
        {"task_id": "t1", "capability": "research",
         "objective": "Gather the AI RMF governance and risk-tolerance requirements",
         "depends_on": [], "required_inputs": ["question"],
         "expected_output": "AI RMF requirements summary",
         "done_when": "requirements are summarised"},
        {"task_id": "t2", "capability": "research",
         "objective": "Gather the ISO 42001 certification requirements",
         "depends_on": ["t1"], "required_inputs": ["research"],
         "expected_output": "ISO 42001 requirements summary",
         "done_when": "requirements are summarised"},
        {"task_id": "t3", "capability": "synthesis",
         "objective": "Recommend which framework suits the startup",
         "depends_on": ["t1", "t2"], "required_inputs": ["research"],
         "expected_output": "Recommendation with reasoning",
         "done_when": "a recommendation is given"},
    ],
})

COMBINED_ANSWER = "COMBINED_RECOMMENDATION: adopt the AI RMF first."
APPROVAL_MARKER = "I have NOT searched the web"


def _blocked(*_args, **_kwargs):
    raise AssertionError(
        "network blocked: the multi-approval suite must make zero external "
        "requests — a real search here means the approval gate failed")


_ORIG_SEAMS: dict = {}


def setUpModule():
    _ORIG_SEAMS["search"] = searxng_client.search
    _ORIG_SEAMS["extract"] = crawl_extractor.extract
    searxng_client.search = _blocked
    crawl_extractor.extract = _blocked


def tearDownModule():
    searxng_client.search = _ORIG_SEAMS["search"]
    crawl_extractor.extract = _ORIG_SEAMS["extract"]


class _ScriptedLLM:
    """Answers only the two prompts this workflow may send; else raises."""

    def __init__(self):
        self.prompts: list = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        text = str(prompt)
        if "You are a task planner" in text:
            return SimpleNamespace(content=PLAN_JSON)
        if "Combine the completed work" in text:
            return SimpleNamespace(content=COMBINED_ANSWER)
        raise AssertionError(
            "unexpected model call in an offline suite: "
            + text[:120].replace("\n", " "))


class _ResearchRecorder:
    """The research node after the gate: one distinct result per execution."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, state):
        self.calls.append(dict(state))
        n = len(self.calls)
        return with_agent_output(
            state, "research", f"RESEARCH_RESULT_{n}",
            ["research", f"https://example.org/source-{n}"])


class MultiApprovalApiTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        import app.api.server as server

        self.server = server
        self._orig_store = server._store
        self.store = ExecutionStore(
            Path(tempfile.mkdtemp()) / "multi_approval.sqlite3")
        server._store = self.store
        self.client = TestClient(server.build_server())
        self.addCleanup(self._restore_store)
        self.addCleanup(self.store.close)

        self.llm = _ScriptedLLM()
        self.research = _ResearchRecorder()
        self._orig_research = gm.research_agent
        gm.research_agent = self.research
        self.addCleanup(setattr, gm, "research_agent", self._orig_research)
        patch = mock.patch("app.llm_client.get_llm",
                           side_effect=lambda **_kwargs: self.llm)
        patch.start()
        self.addCleanup(patch.stop)

    def _restore_store(self):
        self.server._store = self._orig_store

    # -- helpers ----------------------------------------------------------

    def _start(self) -> dict:
        response = self.client.post("/api/v1/executions",
                                    json={"question": QUESTION})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def _execution(self, eid: str) -> dict:
        response = self.client.get(f"/api/v1/executions/{eid}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _pending(self) -> list:
        response = self.client.get("/api/v1/approvals")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["count"], len(body["approvals"]))
        return body["approvals"]

    def _tasks(self, execution: dict) -> dict:
        self.assertIsNotNone(execution.get("plan"), "the plan must be persisted")
        return {t["task_id"]: t for t in execution["plan"]["tasks"]}

    def _decide(self, approval_id: str, action: str):
        return self.client.post(f"/api/v1/approvals/{approval_id}/{action}")

    def _resume(self, approval_id: str):
        return self.client.post(f"/api/v1/approvals/{approval_id}/resume")

    def _events(self, eid: str) -> list:
        response = self.client.get(f"/api/v1/executions/{eid}/events")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["events"]

    # -- the workflow ------------------------------------------------------

    def test_two_research_tasks_each_need_their_own_decision(self):
        created = self._start()
        eid = created["id"]

        # --- 1. initial submission: paused at the FIRST gate ---------------
        self.assertEqual(created["status"], "waiting_for_approval",
                         "a paused plan must not report completed")
        self.assertIn(APPROVAL_MARKER, created["answer"])
        self.assertEqual(self.research.calls, [],
                         "no external research may run before any decision")
        tasks = self._tasks(created)
        self.assertEqual({k: v["status"] for k, v in tasks.items()},
                         {"t1": "failed", "t2": "blocked", "t3": "blocked"})
        self.assertIn("not authorized", tasks["t1"]["error"])

        first_pending = self._pending()
        self.assertEqual(len(first_pending), 1)
        first = first_pending[0]
        self.assertEqual(first["execution_id"], eid)
        self.assertEqual(first["operation"], "research")
        self.assertIn(first["approval_id"], created["answer"])

        # --- 2. a resume before the decision is refused ---------------------
        early = self._resume(first["approval_id"])
        self.assertEqual(early.status_code, 409, early.text)
        self.assertEqual(self.research.calls, [])

        # --- 3. first approval + resume: task 1 runs, task 2 gates ---------
        approved = self._decide(first["approval_id"], "approve")
        self.assertEqual(approved.status_code, 200, approved.text)
        self.assertEqual(approved.json()["approval"]["status"], "approved")

        resumed = self._resume(first["approval_id"])
        self.assertEqual(resumed.status_code, 200, resumed.text)
        body = resumed.json()
        self.assertEqual(body["outcome"], "executed")
        self.assertEqual(len(self.research.calls), 1,
                         "the approved task must run exactly once")
        self.assertEqual(self.research.calls[0]["question"], QUESTION,
                         "the approved search is the request")

        after_first = self._execution(eid)
        self.assertEqual(after_first["status"], "waiting_for_approval")
        self.assertIsNone(after_first["error"])
        self.assertIn(APPROVAL_MARKER, after_first["answer"])
        tasks = self._tasks(after_first)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t1"]["result"], "RESEARCH_RESULT_1")
        self.assertEqual(tasks["t2"]["status"], "failed",
                         "the second research task must pause, not run")
        self.assertNotEqual(tasks["t3"]["status"], "completed")

        second_pending = self._pending()
        self.assertEqual(len(second_pending), 1,
                         "the second operation needs its OWN request")
        second = second_pending[0]
        self.assertNotEqual(second["approval_id"], first["approval_id"])
        self.assertEqual(second["execution_id"], eid)
        self.assertIn(second["approval_id"], after_first["answer"])
        self.assertEqual(len(self.research.calls), 1,
                         "approving the first task must not authorize the second")

        # --- 4. decision idempotency + no second execution ------------------
        decided_at = approved.json()["approval"]["decided_at"]
        repeat = self._decide(first["approval_id"], "approve")
        self.assertEqual(repeat.status_code, 200, repeat.text)
        self.assertTrue(repeat.json()["decision"]["already_decided"])
        self.assertEqual(repeat.json()["approval"]["decided_at"], decided_at)

        events_before_replay = len(self._events(eid))
        replayed = self._resume(first["approval_id"])
        self.assertEqual(replayed.status_code, 200, replayed.text)
        self.assertEqual(replayed.json()["outcome"], "denied",
                         "a spent approval reports its single-execution claim")
        self.assertEqual(len(self.research.calls), 1,
                         "a spent approval never executes again")
        self.assertEqual(self._execution(eid)["status"], "waiting_for_approval")
        # Exactly ONE honest note records the refusal; no continuation events
        # (no step/dispatcher/resolution spam) and no duplicate request.
        replayed_events = self._events(eid)[events_before_replay:]
        self.assertEqual(len(replayed_events), 1, replayed_events)
        self.assertEqual(replayed_events[0]["type"], "approval.finalized")
        self.assertEqual(replayed_events[0]["payload"]["outcome"], "denied")
        self.assertEqual(len(self._pending()), 1,
                         "and must not coin a duplicate request")

        # --- 5. second approval + resume: both tasks done, plan completes ---
        second_approved = self._decide(second["approval_id"], "approve")
        self.assertEqual(second_approved.status_code, 200, second_approved.text)
        final = self._resume(second["approval_id"])
        self.assertEqual(final.status_code, 200, final.text)
        self.assertEqual(final.json()["outcome"], "executed")
        self.assertEqual(len(self.research.calls), 2,
                         "each approved task runs exactly once")
        self.assertEqual(self.research.calls[1]["question"], QUESTION)

        finished = self._execution(eid)
        self.assertEqual(finished["status"], "completed")
        self.assertIsNone(finished["error"])
        self.assertEqual(finished["answer"], COMBINED_ANSWER)
        tasks = self._tasks(finished)
        self.assertEqual({k: v["status"] for k, v in tasks.items()},
                         {"t1": "completed", "t2": "completed",
                          "t3": "completed"})
        self.assertEqual(tasks["t1"]["result"], "RESEARCH_RESULT_1")
        self.assertEqual(tasks["t2"]["result"], "RESEARCH_RESULT_2")
        self.assertEqual(tasks["t1"]["objective"],
                         "Gather the AI RMF governance and risk-tolerance "
                         "requirements")
        self.assertEqual(tasks["t2"]["objective"],
                         "Gather the ISO 42001 certification requirements")
        self.assertEqual(finished["qa_flags"], [],
                         "a finished plan must carry no QA findings")
        self.assertEqual(self._pending(), [],
                         "no request may remain pending")

        # --- 6. the timeline tells the same story ---------------------------
        types = [e["type"] for e in self._events(eid)]
        self.assertEqual(types.count("approval.created"), 2)
        self.assertEqual(types.count("approval.executed"), 2)
        self.assertEqual(types.count("execution.completed"), 1)
        self.assertNotIn("execution.failed", types)

    def test_interruption_between_the_gates_is_recovered_through_the_api(self):
        """Crash after the SECOND approved research, before its tail runs.

        The retry goes through the same resume endpoint, continues from the
        durably persisted plan, repeats no research, re-runs no completed task
        and reports the truthful terminal status.
        """
        from app.execution.runner import GraphRunner

        eid = self._start()["id"]
        first = self._pending()[0]
        self.assertEqual(self._decide(first["approval_id"], "approve").status_code,
                         200)
        self.assertEqual(self._resume(first["approval_id"]).status_code, 200)
        second = self._pending()[0]
        self.assertEqual(self._decide(second["approval_id"], "approve").status_code,
                         200)

        def _crash(*_args, **_kwargs):
            raise RuntimeError("process died mid-continuation")

        with mock.patch.object(GraphRunner, "_continue_plan", _crash):
            crashed = self._resume(second["approval_id"])
        self.assertEqual(crashed.status_code, 500, crashed.text)
        self.assertIn("could not be executed",
                      crashed.json()["detail"]["human_message"])

        self.assertEqual(len(self.research.calls), 2,
                         "the second approved search ran exactly once")
        paused = self._execution(eid)
        self.assertEqual(paused["status"], "waiting_for_approval")
        tasks = self._tasks(paused)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t2"]["status"], "completed",
                         "the approved result is persisted before the tail")
        self.assertNotEqual(tasks["t3"]["status"], "completed")
        self.assertEqual(self._pending(), [])

        recovered = self._resume(second["approval_id"])

        self.assertEqual(recovered.status_code, 200, recovered.text)
        body = recovered.json()
        self.assertEqual(body["outcome"], "executed")
        self.assertTrue(body.get("recovered"))
        self.assertEqual(len(self.research.calls), 2,
                         "recovery must never repeat an approved search")
        self.assertEqual(
            sum(1 for p in self.llm.prompts if "Combine the completed work"
                in p), 1,
            "the synthesis task runs exactly once")

        finished = self._execution(eid)
        self.assertEqual(finished["status"], "completed")
        self.assertIsNone(finished["error"])
        self.assertEqual(finished["answer"], COMBINED_ANSWER)
        tasks = self._tasks(finished)
        self.assertEqual(tasks["t1"]["result"], "RESEARCH_RESULT_1")
        self.assertEqual(tasks["t2"]["result"], "RESEARCH_RESULT_2")
        self.assertEqual(
            {k: v["status"] for k, v in tasks.items()},
            {"t1": "completed", "t2": "completed", "t3": "completed"})
        self.assertEqual(finished["qa_flags"], [])
        self.assertEqual(self._pending(), [])

    def test_rejecting_the_second_request_leaves_the_run_blocked_and_honest(self):
        created = self._start()
        eid = created["id"]
        first = self._pending()[0]
        self.assertEqual(self._decide(first["approval_id"], "approve").status_code,
                         200)
        self.assertEqual(self._resume(first["approval_id"]).status_code, 200)
        second = self._pending()[0]

        rejected = self._decide(second["approval_id"], "reject")
        self.assertEqual(rejected.status_code, 200, rejected.text)
        self.assertEqual(rejected.json()["approval"]["status"], "rejected")

        closed = self._resume(second["approval_id"])
        self.assertEqual(closed.status_code, 200, closed.text)
        self.assertEqual(closed.json()["outcome"], "rejected")
        self.assertEqual(len(self.research.calls), 1,
                         "a rejected operation never searches")

        execution = self._execution(eid)
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["error"]["category"], "approval_rejected")
        tasks = self._tasks(execution)
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertNotEqual(tasks["t2"]["status"], "completed")
        self.assertNotEqual(execution["status"], "completed")

    def test_unknown_approval_is_404_and_never_executes(self):
        self._start()
        self.assertEqual(self._resume("ffffffffffff").status_code, 404)
        self.assertEqual(
            self.client.get("/api/v1/approvals/ffffffffffff").status_code, 404)
        self.assertEqual(self.research.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
