"""End-to-end assistant behaviour (2026-10-30 milestone).

Every test runs a REAL request through the real pipeline:
GraphRunner -> supervisor -> (planner) -> dispatcher -> specialist node ->
synthesis -> QA -> persisted execution row. Only the model provider and the
specialist nodes are doubles, which is what keeps the suite offline.

The assertions are about OUTCOMES (route, delegation, task statuses, what the
specialists received, what the final answer claims, what QA proved), not about
"a function was called".

Scenarios covered (milestone Phase 6):
  1  simple greeting            -> no delegation, no model call
  2  short acknowledgement      -> no delegation
  3  general question           -> one model answer, no plan, no sources
  4  RAG request                -> books specialist + provenance in the answer
  5  research without authority -> pause for approval, nothing external
  6  two approvals              -> independent, neither pre-executed
  7  data request               -> honest "not enabled", never invented figures
  8  coding request, no workspace -> honest "switched off", no patch claimed
  9  multi-step task            -> plan, dependency order, synthesis, QA
 10  independent tasks         -> executed once each, in plan order (no dupes)
 11  specialist failure        -> dependents blocked, failure reported
 12  missing output            -> QA flags the empty result
 13  conflicting results       -> synthesis receives BOTH, no silent pick
 14  follow-up request         -> context taken from the RECORDED execution
 15  resumed plan              -> completed tasks are never re-executed
 16  approval continuation     -> the approved operation runs exactly once
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

import app.graph as gm  # noqa: E402
from app.execution.runner import GraphRunner  # noqa: E402
from app.execution.store import ExecutionStore  # noqa: E402


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeLLM:
    """Provider double: records every prompt, answers via a responder."""

    def __init__(self, responder):
        self._respond = responder
        self.prompts = []

    def invoke(self, prompt):
        text = str(prompt)
        self.prompts.append(text)
        return SimpleNamespace(content=self._respond(text))

    def calls_matching(self, marker):
        return [p for p in self.prompts if marker in p]


def default_responder(plan=None, general="MODEL_ANSWER",
                      synthesis="COMBINED_ANSWER", followup="FOLLOWUP_ANSWER"):
    def respond(text):
        if "You are a task planner" in text:
            return json.dumps(plan) if plan else "not a plan"
        if "Combine the completed work" in text:
            return synthesis
        if "answering a FOLLOW-UP" in text:
            return followup
        return general
    return respond


class _Real:
    """Sentinel: spy on the node but let the REAL implementation run."""

    def __repr__(self):
        return "REAL"


REAL = _Real()


class Recorder:
    """Installs deterministic specialist doubles on app.graph and records calls.

    synthesis/followup default to REAL: they are the composition steps whose
    behaviour the tests actually verify, so a spy is installed instead of a
    fake. A specialist passed in `behavior` replaces the node entirely.
    """

    NAMES = ("books_agent", "research_agent", "feasibility_agent", "erp_agent",
             "synthesis_node", "coding_node", "data_agent", "followup_node")

    def __init__(self, behavior=None):
        self.calls = []
        self._behavior = behavior or {}
        self._snapshot = {name: getattr(gm, name) for name in self.NAMES}

    def __enter__(self):
        for name in self.NAMES:
            setattr(gm, name, self._make(name))
        return self

    def __exit__(self, *exc):
        for name, fn in self._snapshot.items():
            setattr(gm, name, fn)
        return False

    def _make(self, name):
        key = name.removesuffix("_agent").removesuffix("_node")

        real = self._snapshot[name]

        def node(state):
            self.calls.append((key, dict(state)))
            behavior = self._behavior.get(key)
            if behavior is REAL or (behavior is None and key in ("synthesis",
                                                                 "followup")):
                return real(state)
            if behavior is None:
                answer = f"{key.upper()}_OUT"
                return _out(state, key, answer, [key])
            if callable(behavior):
                return behavior(state)
            if isinstance(behavior, BaseException):
                raise behavior
            return _out(state, key, str(behavior), [key])
        return node

    def called(self, key):
        return [state for name, state in self.calls if name == key]


def _out(state, key, answer, sources):
    return {
        "answer": answer,
        "sources": list(sources),
        "agent_answers": {**(state.get("agent_answers") or {}), key: answer},
    }


def plan(*tasks, goal="do the thing"):
    return {"goal": goal, "tasks": list(tasks)}


def task(tid, capability, objective, depends_on=(), outputs="an answer"):
    return {
        "task_id": tid, "capability": capability, "objective": objective,
        "depends_on": list(depends_on),
        "required_inputs": ["question"] if not depends_on else ["question"],
        "expected_output": outputs, "done_when": "produced",
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class AssistantCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ExecutionStore(Path(self._tmp.name) / "assistant.sqlite3")
        self.addCleanup(self.store.close)
        self.runner = GraphRunner(self.store)
        self.llm = None

    def use_llm(self, responder=None, **kwargs):
        self.llm = FakeLLM(responder or default_responder(**kwargs))
        patcher = mock.patch("app.llm_client.get_llm", return_value=self.llm)
        patcher.start()
        self.addCleanup(patcher.stop)
        return self.llm

    def ask(self, question, agents=None):
        behavior = agents if agents is not None else {}
        with Recorder(behavior) as recorder:
            eid = self.runner.create(question)["id"]
            row = self.runner.run(eid)
        return row, recorder

    def events(self, eid):
        return self.store.get_events(eid)


# ---------------------------------------------------------------------------
# 1-3: conversation vs. work
# ---------------------------------------------------------------------------


class TestConversationNeedsNoWork(AssistantCase):
    def test_greeting_is_answered_locally_without_a_model_call(self):
        llm = self.use_llm()
        row, rec = self.ask("Hi, how are you?")

        self.assertEqual(row["route"], "conversational")
        self.assertTrue(row["answer"].strip())
        self.assertEqual(row["sources"], [])
        self.assertIsNone(row["plan"])
        self.assertEqual(rec.calls, [])          # no specialist ran
        self.assertEqual(llm.prompts, [])        # no model call either
        self.assertEqual(row["status"], "completed")

    def test_acknowledgement_does_not_start_work(self):
        llm = self.use_llm()
        row, rec = self.ask("Thanks, that helps.")

        self.assertEqual(row["route"], "conversational")
        self.assertEqual(rec.calls, [])
        self.assertEqual(llm.prompts, [])
        self.assertEqual(row["sources"], [])

    def test_general_question_answers_once_without_plan_or_sources(self):
        llm = self.use_llm()
        row, rec = self.ask("Explain how RAG works.")

        self.assertEqual(row["route"], "general")
        # 2026-11-01 evidence grounding: the answer is still the model's, but
        # it now carries the honest label (no source was consulted). The
        # structural contract is unchanged: one completion, no plan, no
        # provenance that would imply a source.
        self.assertTrue(row["answer"].startswith("MODEL_ANSWER"))
        self.assertIn("Answered from my own knowledge", row["answer"])
        self.assertIsNone(row["plan"])
        self.assertEqual(row["sources"], [])
        self.assertEqual(rec.calls, [])
        self.assertEqual(len(llm.prompts), 1)     # exactly one completion
        self.assertEqual(row["status"], "completed")

    def test_document_request_reaches_the_books_specialist(self):
        self.use_llm()
        row, rec = self.ask(
            "Answer this question using my stored documents and cite the sources.",
            agents={"books": "Answer from chapter 3 with citations."})

        self.assertEqual(row["route"], "books")
        self.assertEqual(len(rec.called("books")), 1)
        self.assertEqual(row["answer"], "Answer from chapter 3 with citations.")
        self.assertIn("books", row["sources"])


# ---------------------------------------------------------------------------
# 5-6: approvals
# ---------------------------------------------------------------------------


class TestApprovalGate(AssistantCase):
    QUESTION = "What does chapter 3 say about propolis?"

    def _refusing_books(self):
        def books(state):
            return {
                "answer": "The requested information was not found in the books.",
                "sources": [],
                "agent_answers": {**(state.get("agent_answers") or {}),
                                  "books": "not found"},
                "books_refusal": {"reason": "low_confidence", "score": 0.12,
                                  "nodes": 0, "recovery": "none"},
            }
        return books

    def test_unauthorized_research_pauses_instead_of_searching(self):
        self.use_llm()
        row, rec = self.ask(self.QUESTION, agents={"books": self._refusing_books()})

        self.assertEqual(rec.called("research"), [])   # nothing external ran
        pending = self.store.list_approvals("pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["execution_id"], row["id"])
        self.assertEqual(row["status"], "waiting_for_approval")

    def test_two_pending_approvals_are_independent(self):
        self.use_llm()
        first, _ = self.ask(self.QUESTION, agents={"books": self._refusing_books()})
        second, _ = self.ask(self.QUESTION, agents={"books": self._refusing_books()})

        pending = self.store.list_approvals("pending")
        self.assertEqual(len(pending), 2)
        self.assertEqual({p["execution_id"] for p in pending},
                         {first["id"], second["id"]})
        self.assertNotEqual(pending[0]["approval_id"], pending[1]["approval_id"])
        # Neither run consumed the other's request.
        self.assertEqual(first["status"], "waiting_for_approval")
        self.assertEqual(second["status"], "waiting_for_approval")

    def test_approved_request_runs_the_operation_exactly_once(self):
        self.use_llm()
        row, rec = self.ask(self.QUESTION, agents={"books": self._refusing_books()})
        self.assertEqual(rec.called("research"), [])
        approval = self.store.list_approvals("pending")[0]

        self.store.decide_approval(approval["approval_id"], "approved")
        with Recorder({"research": "SEARCHED_RESULT"}) as resumed:
            first = self.runner.resume_approval(approval["approval_id"])
            second = self.runner.resume_approval(approval["approval_id"])

        self.assertEqual(len(resumed.called("research")), 1)   # never replayed
        self.assertIn("SEARCHED_RESULT", first.get("answer", ""))
        # The repeat is refused by the durable single-execution claim, and the
        # real result stays on the execution.
        self.assertEqual(second.get("outcome"), "denied")
        self.assertIn("already executed", second.get("reason", ""))
        stored = self.store.get_execution(row["id"])
        self.assertIn("SEARCHED_RESULT", stored["answer"])


# ---------------------------------------------------------------------------
# 7-8: honest capability limits
# ---------------------------------------------------------------------------


class TestHonestLimits(AssistantCase):
    def test_dataset_request_without_a_workspace_is_refused_not_invented(self):
        llm = self.use_llm()
        row, rec = self.ask(
            "Analyze the dataset and tell me which products performed best.",
            agents={"data": gm.data_agent})

        self.assertEqual(row["route"], "data")
        self.assertIn("not enabled", row["answer"])
        self.assertIn("CODING_WORKSPACE_ROOT", row["answer"])
        # The general route was never asked, so no figures could be invented.
        self.assertEqual(llm.prompts, [])
        self.assertEqual(row["sources"], [])

    def test_coding_request_without_a_workspace_names_the_missing_setting(self):
        llm = self.use_llm()
        row, _ = self.ask("Fix the bug in app/graph.py", agents={"coding": gm.coding_node})

        self.assertEqual(row["route"], "coding")
        self.assertIn("CODING_WORKSPACE_ROOT", row["answer"])
        self.assertIn("not enabled", row["answer"])
        self.assertEqual(llm.prompts, [])
        # No patch was prepared, so nothing may look like editable work.
        self.assertNotIn("approval", (row["answer"] or "").lower())


# ---------------------------------------------------------------------------
# 9-13: planning, delegation, failure, QA, synthesis
# ---------------------------------------------------------------------------


MULTI = ("Research three suitable open-source models, compare their hardware "
         "requirements, and recommend one for my machine.")


class TestMultiStepExecution(AssistantCase):
    def _plan(self):
        return plan(
            task("t1", "research", "Research the options on the web"),
            task("t2", "books", "Check the library for requirements"),
            task("t3", "synthesis", "Combine both findings into a recommendation",
                 depends_on=["t1", "t2"]),
        )

    def test_plan_delegates_in_dependency_order_and_synthesises(self):
        llm = self.use_llm(plan=self._plan())
        row, rec = self.ask(MULTI)

        self.assertTrue(row["plan"], "a multi-objective request must be planned")
        statuses = {t["task_id"]: t["status"] for t in row["plan"]["tasks"]}
        self.assertEqual(statuses, {"t1": "completed", "t2": "completed",
                                    "t3": "completed"})
        order = [name for name, _ in rec.calls]
        self.assertIn("research", order)
        self.assertIn("books", order)
        self.assertLess(order.index("research"), order.index("synthesis"))
        self.assertLess(order.index("books"), order.index("synthesis"))

        # Information exchange: synthesis received BOTH specialists' results.
        self.assertEqual(len(llm.calls_matching("RESEARCH_OUT")), 1)
        self.assertEqual(len(llm.calls_matching("BOOKS_OUT")), 1)
        self.assertEqual(row["answer"], "COMBINED_ANSWER")
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["qa_flags"], [])

    def test_independent_tasks_run_once_each(self):
        self.use_llm(plan=self._plan())
        _, rec = self.ask(MULTI)

        self.assertEqual(len(rec.called("research")), 1)
        self.assertEqual(len(rec.called("books")), 1)
        self.assertEqual(len(rec.called("synthesis")), 1)

    def test_specialist_failure_blocks_dependents_and_is_reported(self):
        self.use_llm(plan=self._plan())
        row, rec = self.ask(MULTI, agents={"research": RuntimeError("provider down")})

        tasks = {t["task_id"]: t for t in row["plan"]["tasks"]}
        self.assertEqual(tasks["t1"]["status"], "failed")
        self.assertIn("provider down", tasks["t1"]["error"])
        self.assertEqual(tasks["t3"]["status"], "blocked")
        self.assertEqual(rec.called("synthesis"), [])       # never ran
        self.assertEqual(len(rec.called("books")), 1)       # independent work continued
        self.assertTrue(any("plan incomplete" in f for f in row["qa_flags"]))
        self.assertNotEqual(row["status"], "completed")

    def test_missing_output_is_flagged_by_qa(self):
        self.use_llm(plan=plan(task("t1", "research", "Research it"),
                               task("t2", "synthesis", "Answer", depends_on=["t1"])))
        row, _ = self.ask(MULTI, agents={"research": lambda state: {"answer": "   "}})

        tasks = {t["task_id"]: t for t in row["plan"]["tasks"]}
        self.assertEqual(tasks["t1"]["status"], "failed")
        self.assertEqual(tasks["t1"]["error"], "the task produced no output")
        self.assertTrue(any("plan incomplete" in f for f in row["qa_flags"]))

    def test_conflicting_results_reach_synthesis_without_being_flattened(self):
        llm = self.use_llm(plan=self._plan())
        row, _ = self.ask(MULTI, agents={
            "research": "Option A needs 16 GB VRAM.",
            "books": "The manual says Option A needs 8 GB VRAM.",
        })

        synthesis_prompts = llm.calls_matching("Combine the completed work")
        self.assertEqual(len(synthesis_prompts), 1)
        prompt = synthesis_prompts[0]
        # Both figures are present, so a disagreement is visible to the model
        # (and to a reader when the provider is down).
        self.assertIn("16 GB", prompt)
        self.assertIn("8 GB", prompt)
        self.assertEqual(row["answer"], "COMBINED_ANSWER")


# ---------------------------------------------------------------------------
# 14: contextual follow-up
# ---------------------------------------------------------------------------


class TestFollowUpContext(AssistantCase):
    def test_follow_up_uses_the_recorded_execution_not_model_memory(self):
        llm = self.use_llm()
        first, _ = self.ask("Which local model should I use for coding?")
        self.assertEqual(first["status"], "completed")

        second, rec = self.ask("What about the one you recommended?")

        self.assertEqual(second["route"], "followup")
        self.assertEqual(rec.called("books"), [])
        self.assertEqual(rec.called("research"), [])
        prompt = llm.calls_matching("answering a FOLLOW-UP")[0]
        self.assertIn("MODEL_ANSWER", prompt)          # the recorded result
        self.assertIn(first["question"], prompt)       # and its request
        self.assertEqual(second["sources"], [f"execution:{first['id']}"])
        self.assertEqual(second["answer"], "FOLLOWUP_ANSWER")

    def test_follow_up_without_history_refuses_to_invent(self):
        self.use_llm()
        row, rec = self.ask("What about the one you recommended?")

        self.assertEqual(row["route"], "followup")
        self.assertIn("no completed execution", row["answer"])
        self.assertEqual(row["sources"], [])           # no source claimed
        self.assertEqual(row["qa_flags"], [])          # an honest refusal is not a defect
        # No specialist was consulted (the follow-up node itself is spied, not
        # faked, so it legitimately appears in the call log).
        self.assertEqual([c for c in rec.calls if c[0] != "followup"], [])

    def test_prior_context_never_leaks_a_failed_run(self):
        self.use_llm()
        good, _ = self.ask("What is the difference between an agent and a tool?")
        # A failing run after it must not replace the recorded result.
        with Recorder({}):
            broken = self.runner.create("Explain how RAG works.")["id"]
            with mock.patch.object(gm, "general_node", side_effect=RuntimeError("boom")):
                self.runner.run(broken)
        self.assertEqual(self.store.get_execution(broken)["status"], "failed")

        row, _ = self.ask("What about the one you recommended?")
        self.assertEqual(row["sources"], [f"execution:{good['id']}"])


# ---------------------------------------------------------------------------
# 15-16: recovery / continuation
# ---------------------------------------------------------------------------


class TestResumedWork(AssistantCase):
    def test_completed_tasks_are_not_re_executed_on_continuation(self):
        self.use_llm()  # the real synthesis node runs; never a live provider
        resume_plan = plan(
            task("t1", "books", "Answer from the library"),
            task("t2", "research", "Research the rest", depends_on=["t1"]),
            task("t3", "synthesis", "Combine", depends_on=["t1", "t2"]),
        )
        resume_plan["tasks"][0]["status"] = "completed"
        resume_plan["tasks"][0]["result"] = "ALREADY_DONE"

        with Recorder({}) as rec:
            result = gm.continue_plan({"question": MULTI, "plan": resume_plan})

        self.assertEqual(rec.called("books"), [])      # t1 never re-ran
        self.assertEqual(len(rec.called("research")), 1)
        self.assertEqual(len(rec.called("synthesis")), 1)
        statuses = {t["task_id"]: t["status"] for t in result["plan"]["tasks"]}
        self.assertEqual(statuses["t1"], "completed")
        self.assertEqual(result["answer"], "COMBINED_ANSWER")


if __name__ == "__main__":
    unittest.main(verbosity=2)
