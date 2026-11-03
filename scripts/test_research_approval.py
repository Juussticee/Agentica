"""Outbound research approval (2026-10-28, Priority B) — fail-closed, offline.

The application has NO approval mechanism yet (docs/CONTROL_CENTER_PLAN.md:
"Approvals: None anywhere — Approval Center is net-new backend capability
(Section 12 / GUI-10)"), and no state survives between runs. The only
approval signal a run can therefore read is the user's own message, so the
contract proven here is:

  1. external research does NOT execute before approval;
  2. approval permits ONLY the intended operation (per-request, never a
     blanket permission, never carried over between runs);
  3. declining (a message that does not authorize research) prevents it;
  4. local RAG never asks for approval;
  5. general questions never trigger web research;
  6. missing/absent approval state fails closed;
  7. existing approval/execution behavior (routing topology, QA, fallback
     ELIGIBILITY) stays compatible.

The search and crawl seams are replaced module-wide for this suite, so ANY
attempt at an external call fails the run — the same guard the knowledge
fallback suite uses.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

import app.graph as gm  # noqa: E402
import app.supervisor as sup  # noqa: E402
from app.qa_node import run_qa  # noqa: E402
from app.state import with_agent_output  # noqa: E402
from app.tools import crawl_extractor, searxng_client  # noqa: E402

APPROVAL_MARKER = "NOT searched"


def _blocked(*_args, **_kwargs):
    raise AssertionError(
        "network blocked: research approval tests must make zero external "
        "requests — an executed search here means the gate failed")


_ORIG_SEAMS: dict = {}


def setUpModule():
    _ORIG_SEAMS["search"] = searxng_client.search
    _ORIG_SEAMS["extract"] = crawl_extractor.extract
    searxng_client.search = _blocked
    crawl_extractor.extract = _blocked


def tearDownModule():
    searxng_client.search = _ORIG_SEAMS["search"]
    crawl_extractor.extract = _ORIG_SEAMS["extract"]


class _ResearchRecorder:
    """Stands in for the research node: records every state it receives."""

    def __init__(self, calls):
        self.calls = calls
        self.states = []

    def __call__(self, state):
        self.calls.append("research")
        self.states.append(dict(state))
        return with_agent_output(state, "research",
                                 "EXTERNAL_RESEARCH_ANSWER",
                                 ["research", "https://example.org/x"])


class _ApprovalGraphCase(unittest.TestCase):
    """Real graph (supervisor/QA/general/unsupported/gate all REAL) with the
    four agent nodes faked, per the routing-suite convention."""

    def setUp(self):
        self.calls = []
        self.recorder = _ResearchRecorder(self.calls)
        self._snapshot = {n: getattr(gm, n) for n in
                          ("books_agent", "research_agent",
                           "feasibility_agent", "erp_agent")}

        def _fake(name):
            def fn(state):
                self.calls.append(name)
                return {"answer": f"FAKE_{name.upper()}",
                        "sources": [name],
                        "agent_answers": {**(state.get("agent_answers") or {}),
                                          name: f"FAKE_{name.upper()}"}}
            return fn

        gm.books_agent = _fake("books")
        gm.research_agent = self.recorder
        gm.feasibility_agent = _fake("feasibility")
        gm.erp_agent = _fake("erp")
        self.graph = gm.build_graph()
        self.addCleanup(self._restore)

    def _restore(self):
        for name, fn in self._snapshot.items():
            setattr(gm, name, fn)

    def _llm_route_as(self, label):
        """Force ROUTING_MODE=llm to hand back `label` (the only way a
        question with no web cue can reach the research route at all)."""
        return mock.patch.multiple(
            sup, ROUTING_MODE="llm",
            _llm_route=mock.Mock(return_value=label))


class TestNoExecutionBeforeApproval(_ApprovalGraphCase):
    def test_direct_research_route_never_runs_without_authorization(self):
        with self._llm_route_as("research"):
            result = self.graph.invoke(
                {"question": "What is a vector database?"})
        self.assertEqual(self.calls, [],
                         "research executed before approval was granted")
        self.assertEqual(result["route"], "research")
        self.assertEqual(result["sources"], [],
                         "a gated run must claim no provenance")
        self.assertIn(APPROVAL_MARKER, result["answer"])
        self.assertNotIn("Sources:", result["answer"])
        self.assertNotIn("https://", result["answer"])
        # identifies the proposed external action and why
        self.assertIn("External action proposed", result["answer"])
        self.assertIn("What is a vector database?", result["answer"])
        self.assertIn("does not ask for web research", result["answer"])
        # offers the local alternatives
        self.assertIn("my own knowledge", result["answer"])
        self.assertIn("indexed documents", result["answer"])

    def test_books_fallback_never_searches_without_authorization(self):
        """The 2026-10-25 knowledge fallback: eligibility still holds, but
        the external step now stops at the approval gate."""
        state = {"question": "What year was the AI Risk Management Framework "
                             "released?",
                 "route": "books",
                 "books_refusal": {"reason": "low_confidence", "score": 0.30,
                                   "nodes": 1, "recovery": "skipped"}}
        update = gm.research_entry(state)
        self.assertEqual(self.calls, [],
                         "fallback executed an unauthorized web search")
        self.assertIn(APPROVAL_MARKER, update["answer"])
        # the local limitation is disclosed alongside the proposal
        self.assertIn("indexed documents were consulted first",
                      update["answer"])
        self.assertIn("reason: low_confidence", update["answer"])
        self.assertEqual(update["sources"], [])
        self.assertIn("research", update["agent_answers"])

    def test_plan_dispatched_research_never_becomes_a_back_door(self):
        state = {
            "question": "q",
            "route": "research",
            "plan": {
                "goal": "q",
                "source": "test",
                "tasks": [{
                    "task_id": "t1", "capability": "research",
                    "objective": "Research the request on the web",
                    "depends_on": [], "required_inputs": ["question"],
                    "expected_output": "Web research summary",
                    "done_when": "summary available",
                }],
            },
        }
        for _ in range(5):
            update = gm.dispatcher_node(state)
            state.update(update or {})
            tasks = state["plan"]["tasks"]
            if all(t.get("status") in
                   ("completed", "failed", "blocked", "unsupported")
                   for t in tasks):
                break
        task = state["plan"]["tasks"][0]
        self.assertEqual(self.calls, [],
                         "a plan task executed unauthorized web research")
        # honest bookkeeping: it did NOT complete, and the reason is on it
        self.assertEqual(task["status"], "failed")
        self.assertIn("not authorized", task["error"])
        self.assertIsNone(task["result"])
        self.assertIn(APPROVAL_MARKER, state.get("answer", ""))


class TestApprovalIsScopedToTheIntendedOperation(_ApprovalGraphCase):
    def test_authorization_codes_are_explicit(self):
        self.assertEqual(
            sup.research_authorization(
                "Search the web for beekeeping guides", "research"),
            (True, "explicit_web_request"))
        self.assertEqual(
            sup.research_authorization("Search the latest AI news",
                                       "research"),
            (True, "current_information_request"))
        self.assertEqual(
            sup.research_authorization("What is the weather in Berlin today?",
                                       "research"),
            (True, "current_information_request"))
        self.assertEqual(
            sup.research_authorization("anything at all", "feasibility"),
            (True, "requested_operation:feasibility_study"))
        # a message that forbids web access can never authorize it
        self.assertEqual(
            sup.research_authorization(
                "Do not search the web; use only this document", "research"),
            (False, "local_only_request"))

    def test_one_authorized_request_never_authorizes_the_next(self):
        # 1st run: the message itself asks for research -> it runs.
        with self._llm_route_as("research"):
            first = self.graph.invoke(
                {"question": "Search the latest AI news"})
        self.assertEqual(self.calls, ["research"])
        self.assertEqual(first["answer"], "EXTERNAL_RESEARCH_ANSWER")

        # 2nd run on the SAME compiled graph: no authorization in the
        # message -> gated. Nothing carries over (no session permission).
        with self._llm_route_as("research"):
            second = self.graph.invoke(
                {"question": "What is a vector database?"})
        self.assertEqual(self.calls, ["research"],
                         "approval leaked into an unrelated request")
        self.assertEqual(second["sources"], [])
        self.assertIn(APPROVAL_MARKER, second["answer"])

    def test_feasibility_chain_is_still_authorized_by_the_request(self):
        result = self.graph.invoke(
            {"question": "Do a feasibility study for an urban beekeeping "
                         "venture?"})
        self.assertEqual(result["route"], "feasibility")
        self.assertEqual(self.calls, ["research", "books", "feasibility"])


class TestDecliningPreventsTheOperation(_ApprovalGraphCase):
    def test_a_declining_message_never_reaches_research(self):
        for question in ("No thanks", "No thank you",
                         "Do not search the web; just tell me what you know"):
            with self.subTest(question=question):
                self.calls.clear()
                result = self.graph.invoke({"question": question})
                self.assertNotIn("research", self.calls, question)
                self.assertFalse(
                    sup.research_authorization(question, result["route"])[0])
                self.assertNotIn("https://", result.get("answer", ""))

    def test_declined_run_still_answers_clearly(self):
        result = self.graph.invoke({"question": "No thanks"})
        self.assertTrue(result["answer"].strip())
        self.assertEqual(result.get("qa_flags") or [], [])


class TestLocalRagNeedsNoApproval(_ApprovalGraphCase):
    def test_books_route_runs_directly_with_no_prompt(self):
        result = self.graph.invoke(
            {"question": "What does chapter 3 say about propolis?"})
        self.assertEqual(result["route"], "books")
        self.assertEqual(self.calls, ["books"])
        self.assertEqual(result["answer"], "FAKE_BOOKS")
        self.assertEqual(result["sources"], ["books"])
        self.assertNotIn(APPROVAL_MARKER, result["answer"])

    def test_explicit_local_only_request_untouched(self):
        result = self.graph.invoke(
            {"question": "Using only this document, explain the Map "
                         "function."})
        self.assertEqual(result["route"], "books")
        self.assertEqual(self.calls, ["books"])


class TestGeneralQuestionsNeverResearch(_ApprovalGraphCase):
    def test_general_question_is_answered_without_web_access(self):
        fake_llm = mock.Mock(return_value="A vector database stores "
                                          "embeddings.")
        with mock.patch("app.llm_client.get_llm", return_value=fake_llm):
            result = self.graph.invoke(
                {"question": "What is a vector database?"})
        self.assertEqual(result["route"], "general")
        self.assertEqual(self.calls, [],
                         "a general question triggered web research")
        self.assertEqual(result["sources"], [])
        self.assertNotIn(APPROVAL_MARKER, result["answer"])
        self.assertEqual(result.get("qa_flags") or [], [])


class TestMissingApprovalStateFailsClosed(_ApprovalGraphCase):
    def test_empty_or_absent_state_is_denied(self):
        self.assertEqual(sup.research_authorization("", ""),
                         (False, "no_research_authorization_in_this_request"))
        self.assertEqual(sup.research_authorization(None, None),
                         (False, "no_research_authorization_in_this_request"))
        self.assertEqual(sup.research_authorization("anything", ""),
                         (False, "no_research_authorization_in_this_request"))

    def test_gated_run_with_no_question_still_produces_an_honest_answer(self):
        update = gm.research_entry({"route": "research"})
        self.assertEqual(self.calls, [])
        self.assertIn(APPROVAL_MARKER, update["answer"])

    def test_gated_run_without_a_books_refusal_does_not_crash(self):
        update = gm.research_entry(
            {"question": "What is a vector database?", "route": "books"})
        self.assertEqual(self.calls, [])
        self.assertIn(APPROVAL_MARKER, update["answer"])
        self.assertNotIn("indexed documents were consulted", update["answer"])

    def test_unknown_route_value_fails_closed(self):
        self.assertFalse(sup.research_authorization("Explain RAG", "mystery")[0])


class TestExistingBehaviorStaysCompatible(_ApprovalGraphCase):
    def test_qa_does_not_treat_an_approval_request_as_a_failure(self):
        with self._llm_route_as("research"):
            result = self.graph.invoke(
                {"question": "What is a vector database?"})
        self.assertEqual(run_qa(result), [])

    def test_fallback_eligibility_layer_is_unchanged(self):
        """Priority B gates EXECUTION; the 2026-10-25 eligibility decision
        (fallback_decision) keeps its exact contract."""
        refused = {"route": "books",
                   "question": "What year was the AI Risk Management "
                               "Framework released?",
                   "books_refusal": {"reason": "low_confidence"}}
        self.assertEqual(
            sup.fallback_decision(refused),
            (True, "books_insufficient:no_local_evidence:low_confidence"))
        local_only = dict(refused, question="Using only this document, what "
                                            "chunk size is required?")
        self.assertEqual(sup.fallback_decision(local_only),
                         (False, "local_only_request"))

    def test_graph_topology_is_unchanged(self):
        from langgraph.graph import END

        self.assertEqual(gm._after_research({"route": "research"}), END)
        self.assertEqual(gm._after_research({"route": "feasibility"}),
                         "books")
        refused = {"route": "books", "question": "q",
                   "books_refusal": {"reason": "low_confidence"}}
        # the transition itself still exists — the GATE is what stops it
        self.assertEqual(gm._after_books(refused), "research")
        self.assertEqual(gm._after_books({"route": "feasibility"}),
                         "feasibility")
        self.assertEqual(gm._after_books({"route": "books",
                                          "question": "anything"}), END)

    def test_conversation_still_needs_no_approval(self):
        result = self.graph.invoke({"question": "hello"})
        self.assertEqual(result["route"], "conversational")
        self.assertEqual(self.calls, [])
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
