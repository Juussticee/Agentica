"""Routing policy (2026-10-28, Priority A) — table-driven, deterministic, offline.

The old router had ONE defect class: anything that missed every keyword rule
fell through to the research DEFAULT, so a greeting (execution 3fc3bc8460c9),
"Create a report", or "What is a vector database?" all cost a live web search.
The policy now distinguishes, with no LLM classifier:

  * conversation       — greetings, thanks, acknowledgements, wellbeing
  * general            — questions answerable from model knowledge
  * local knowledge    — books / ERP / feasibility (unchanged precedence)
  * research           — requested in THIS message (explicit web / current info)
  * unsupported        — no such tool, no such memory, or too vague (clarify)

Everything here is deterministic: zero network, zero Chroma, zero live model
(the model seam is faked only where a general answer is produced, and boom
stubs prove the search/crawl/retrieval seams are never touched).
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Fail-closed Chroma isolation (docs/TESTING.md §F), same as every suite that
# imports application modules directly.
from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

import app.supervisor as supervisor_module  # noqa: E402
from app.qa_node import run_qa  # noqa: E402
from app.supervisor import (  # noqa: E402
    conversational_reply,
    is_conversational,
    research_authorization,
    supervisor,
    unsupported_reply,
)

# The representative inputs from the task, with the route each MUST take
# given Agentica's actual capabilities (no report writer, no mailer, no
# conversation memory; books, ERP, feasibility and web research exist).
ROUTE_TABLE = [
    # conversation
    ("Hi", "conversational"),
    ("Thanks", "conversational"),
    # acknowledgements (2026-10-29): a short "thanks, that helps" is social
    # exchange, not an underspecified request.
    ("Thanks, that helps", "conversational"),
    ("Great, thank you so much", "conversational"),
    ("Cheers, got it", "conversational"),
    ("Good morning", "conversational"),
    ("How are you?", "conversational"),
    # general model-knowledge questions (previously: research default)
    ("Hi, can you explain RAG?", "general"),
    ("What is a vector database?", "general"),
    ("Who won the 2024 F1 championship?", "general"),
    ("Explain how transformer neural networks work in modern AI systems.",
     "general"),
    # no conversation memory we actually have -> honest unsupported
    ("What did I tell you earlier?", "unsupported"),
    # requested in THIS message -> research
    ("Search the latest AI news", "research"),
    ("What is the latest news on fusion energy startups?", "research"),
    # local knowledge
    ("Find the information in my documents", "books"),
    ("What does chapter 3 say about propolis?", "books"),
    ("Show me open invoices", "erp"),
    ("Do a feasibility study for an urban beekeeping venture?", "feasibility"),
    # capability we do not have / underspecified -> unsupported
    ("Create a report", "unsupported"),
    ("Send an email to my team", "unsupported"),
    ("Do it", "unsupported"),
    ("beekeeping stuff", "books"),   # corpus keyword still wins (pre-existing)
]

GENERAL_QUESTIONS = [
    "What is a vector database?",
    "Who won the 2024 F1 championship?",
    "Tell me about functions.",
    "What are people saying about AI art these days?",
    "How do raptor migrations work?",
]


def _faked_graph(calls, boom=None):
    """Real compiled graph with every AGENT node faked, recording calls.

    Supervisor, planner, QA, the conversational/general/unsupported nodes
    stay REAL — only the nodes that would touch LLM/search/crawl/Chroma are
    replaced (the established convention across the routing suites).
    `boom` is installed as app.llm_client.get_llm when a test must prove no
    model call happens at all.
    """
    import app.graph as gm

    def fake(name):
        def fn(state):
            calls.append(name)
            return {"answer": f"FAKE_{name.upper()}",
                    "sources": [name],
                    "agent_answers": {**(state.get("agent_answers") or {}),
                                      name: f"FAKE_{name.upper()}"}}
        return fn

    snapshot = {n: getattr(gm, n) for n in
                ("books_agent", "research_agent", "feasibility_agent",
                 "erp_agent")}
    for name, attr in (("books", "books_agent"),
                       ("research", "research_agent"),
                       ("feasibility", "feasibility_agent"),
                       ("erp", "erp_agent")):
        setattr(gm, attr, fake(name))
    return gm, snapshot


class _FakedGraphCase(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.gm, self._snapshot = _faked_graph(self.calls)
        self.graph = self.gm.build_graph()

    def tearDown(self):
        for name, fn in self._snapshot.items():
            setattr(self.gm, name, fn)


class TestRouteTable(unittest.TestCase):
    """Every representative input lands on its designated route (no LLM)."""

    def test_table_routes(self):
        for question, expected in ROUTE_TABLE:
            with self.subTest(question=question):
                result = supervisor({"question": question})
                self.assertEqual(result["route"], expected, question)
                # supervisor returns only the keys it changes (existing contract)
                self.assertLessEqual(set(result),
                                     {"route", "sub_routes", "complex"})

    def test_general_questions_never_fall_through_to_research(self):
        """The defect this policy fixes: a missed keyword rule must not
        trigger an external web search."""
        for question in GENERAL_QUESTIONS:
            with self.subTest(question=question):
                self.assertFalse(is_conversational(question))
                self.assertNotEqual(
                    supervisor({"question": question})["route"], "research")


class TestConversationIsFrictionless(_FakedGraphCase):
    def test_conversational_runs_no_agent(self):
        for question in ("Hi", "Thanks", "Good morning", "How are you?"):
            with self.subTest(question=question):
                self.calls.clear()
                result = self.graph.invoke({"question": question})
                self.assertEqual(result["route"], "conversational")
                self.assertEqual(self.calls, [])
                self.assertEqual(result["sources"], [])
                self.assertEqual(result["answer"],
                                 conversational_reply(question))
                self.assertEqual(result.get("qa_flags") or [], [])
                self.assertEqual(run_qa(result), [])

    def test_conversation_touches_no_tool_seam(self):
        import app.agents.books_agent as ba
        import app.llm_client
        import app.tools.crawl_extractor as ce
        import app.tools.searxng_client as sc

        boom = mock.Mock(side_effect=AssertionError("tool was invoked"))
        with mock.patch.object(ba, "_get_query_engine", boom), \
             mock.patch.object(app.llm_client, "get_llm", boom), \
             mock.patch.object(sc.requests, "get", boom), \
             mock.patch.object(ce, "_crawl", boom):
            graph = self.gm.build_graph()
            for question in ("Hi", "How are you?", "Thanks"):
                result = graph.invoke({"question": question})
                self.assertEqual(result["route"], "conversational")
        boom.assert_not_called()


class TestAcknowledgementsAreConversation(_FakedGraphCase):
    """"thanks, that helps" must not be answered as an unknown request."""

    def test_acknowledgement_gets_a_social_reply_not_a_clarification(self):
        for question in ("Thanks, that helps", "Great, thank you so much",
                         "Cheers, got it"):
            with self.subTest(question=question):
                self.assertTrue(is_conversational(question))
                result = self.graph.invoke({"question": question})
                self.assertEqual(result["route"], "conversational")
                self.assertEqual(self.calls, [])
                self.assertEqual(result["sources"], [])
                self.assertIn("You're welcome", result["answer"])
                self.assertNotIn("not sure what you'd like", result["answer"])
                self.assertEqual(run_qa(result), [])

    def test_acknowledgement_plus_a_real_question_keeps_its_route(self):
        """The rule is all-words: one substantive word disqualifies it."""
        for question in ("Thanks, how do I sort a list in Python?",
                         "Thanks, what does chapter 3 say about propolis?",
                         "Great, summarize the latest AI news"):
            with self.subTest(question=question):
                self.assertFalse(is_conversational(question))
                self.assertNotEqual(
                    supervisor({"question": question})["route"],
                    "conversational")


class TestGeneralIsAnsweredFromModelKnowledge(_FakedGraphCase):
    GENERAL_Q = "What is a vector database?"

    def _invoke_general(self, question):
        from types import SimpleNamespace

        fake_llm = mock.Mock()
        fake_llm.invoke.return_value = SimpleNamespace(
            content="A vector database stores embeddings for similarity "
                    "search.")
        import app.llm_client as llm_mod
        with mock.patch.object(llm_mod, "get_llm", return_value=fake_llm):
            result = self.gm.build_graph().invoke({"question": question})
        return result, fake_llm

    def test_general_route_answers_from_the_model_only(self):
        result, fake_llm = self._invoke_general(self.GENERAL_Q)
        self.assertEqual(result["route"], "general")
        self.assertEqual(fake_llm.invoke.call_count, 1)
        self.assertIn("vector database", result["answer"])
        self.assertEqual(result["sources"], [],
                         "a model-knowledge answer must carry no provenance")
        self.assertEqual(self.calls, [],
                         "no agent may run for a general question")
        self.assertEqual(result.get("qa_flags") or [], [])

    def test_general_never_touches_search_crawl_or_retrieval(self):
        import app.tools.crawl_extractor as ce
        import app.tools.searxng_client as sc
        from types import SimpleNamespace

        boom = mock.Mock(side_effect=AssertionError("external tool invoked"))
        fake_llm = mock.Mock()
        fake_llm.invoke.return_value = SimpleNamespace(
            content="Model knowledge answer.")
        import app.llm_client as llm_mod
        with mock.patch.object(sc.requests, "get", boom), \
             mock.patch.object(ce, "_crawl", boom), \
             mock.patch.object(llm_mod, "get_llm", return_value=fake_llm), \
             mock.patch("app.agents.books_agent._get_query_engine", boom):
            result = self.gm.build_graph().invoke(
                {"question": self.GENERAL_Q})
        self.assertEqual(result["route"], "general")
        boom.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_general_fails_soft_when_the_provider_is_unavailable(self):
        """Provider down -> an honest message, NEVER a fallback to research."""
        import app.llm_client as llm_mod
        with mock.patch.object(llm_mod, "get_llm",
                               side_effect=RuntimeError("provider down")):
            result = self.gm.build_graph().invoke(
                {"question": self.GENERAL_Q})
        self.assertEqual(result["route"], "general")
        self.assertIn("unavailable", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(self.calls, [])
        self.assertEqual(result.get("qa_flags") or [], [])


class TestCodeAuthoringIsAnsweredFromKnowledge(_FakedGraphCase):
    """Trace (2026-10-29): "Write a Python function that parses a CSV file"
    was routed `unsupported` and answered "I can't do that" — although the
    code is produced IN THE REPLY, with no tool at all. Code/draft authoring
    is model-knowledge work; operations on the user's machine stay refused.
    """

    CODING = (
        "Write a Python function that parses a CSV file and returns a list "
        "of dicts.",
        "Write a SQL query that joins orders and customers",
        "Draft a short bash script that lists the largest files",
    )

    def test_coding_requests_route_general_not_unsupported(self):
        for question in self.CODING:
            with self.subTest(question=question):
                result = supervisor({"question": question})
                self.assertEqual(result["route"], "general")
                # a single direct answer — never a multi-step plan
                self.assertNotIn("complex", result)

    def test_coding_answer_uses_the_model_and_no_agent(self):
        from types import SimpleNamespace

        fake_llm = mock.Mock()
        fake_llm.invoke.return_value = SimpleNamespace(
            content="def parse_csv(path):\n    ...")
        import app.llm_client as llm_mod
        with mock.patch.object(llm_mod, "get_llm", return_value=fake_llm):
            result = self.graph.invoke({"question": self.CODING[0]})
        self.assertEqual(result["route"], "general")
        self.assertEqual(fake_llm.invoke.call_count, 1)
        self.assertIn("def parse_csv", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(self.calls, [], "no agent may run for a coding request")
        self.assertIsNone(result.get("plan"), "a coding question is not planned")
        self.assertEqual(run_qa(result), [])

    def test_machine_actions_stay_unsupported(self):
        """The authoring exception never covers writing files or running
        things: those need a capability this assistant does not have."""
        for question in (
            "Write a Python script to a file on disk",
            "Create a report",
            "Send an email to my team",
            "Delete the temporary files",
        ):
            with self.subTest(question=question):
                self.assertEqual(supervisor({"question": question})["route"],
                                 "unsupported")

    def test_machine_actions_never_consult_a_model(self):
        import app.llm_client as llm_mod

        boom = mock.Mock(side_effect=AssertionError("model was consulted"))
        with mock.patch.object(llm_mod, "get_llm", boom):
            for question in ("Write a Python script to a file on disk",
                             "Create a report"):
                with self.subTest(question=question):
                    result = self.gm.build_graph().invoke(
                        {"question": question})
                    self.assertEqual(result["route"], "unsupported")
                    self.assertIn("I can't do that", result["answer"])
        boom.assert_not_called()
        self.assertEqual(self.calls, [])


class TestUnsupportedRequestsGetGuidance(_FakedGraphCase):
    def test_unsupported_action_explains_the_missing_capability(self):
        for question in ("Create a report", "Send an email to my team",
                         "Delete the temporary files"):
            with self.subTest(question=question):
                self.calls.clear()
                result = self.gm.build_graph().invoke({"question": question})
                self.assertEqual(result["route"], "unsupported")
                self.assertEqual(self.calls, [])
                self.assertEqual(result["sources"], [])
                self.assertIn("I can't do that", result["answer"])
                # it offers what DOES exist, honestly
                self.assertIn("indexed documents", result["answer"])
                self.assertEqual(result.get("qa_flags") or [], [])

    def test_underspecified_request_asks_for_detail(self):
        result = self.gm.build_graph().invoke({"question": "Do it"})
        self.assertEqual(result["route"], "unsupported")
        self.assertIn("I'm not sure what you'd like me to do",
                      result["answer"])
        self.assertEqual(self.calls, [])

    def test_missing_memory_is_answered_without_a_model(self):
        import app.llm_client as llm_mod

        boom = mock.Mock(side_effect=AssertionError("model was consulted"))
        with mock.patch.object(llm_mod, "get_llm", boom):
            result = self.gm.build_graph().invoke(
                {"question": "What did I tell you earlier?"})
        self.assertEqual(result["route"], "unsupported")
        boom.assert_not_called()
        self.assertIn("conversation history", result["answer"])
        self.assertEqual(self.calls, [])
        self.assertEqual(result.get("qa_flags") or [], [])

    def test_unsupported_replies_exist_for_each_shape(self):
        self.assertIn("conversation history",
                      unsupported_reply("What did I tell you earlier?"))
        self.assertIn("I can't do that", unsupported_reply("Create a report"))
        self.assertIn("I'm not sure", unsupported_reply("Do it"))


class TestSubstantiveRequestsKeepTheirRoute(_FakedGraphCase):
    """Priority A must not disturb any existing routing decision."""

    def test_precedence_chain_is_unchanged(self):
        pins = {
            "hello, what does the NIST AI RMF say about risk management?":
                "books",
            "Show me open invoices": "erp",
            "Using only this document, explain the difference between the Map "
            "and Measure functions.": "books",
            "Do not search the web; use only this document about beekeeping":
                "books",
            "What is the latest news about the NIST AI framework?": "research",
            "Google the four functions of the AI Risk Management Framework":
                "research",
            "What happened in AI research this week?": "research",
            "Do a feasibility study for an urban beekeeping venture?":
                "feasibility",
        }
        for question, expected in pins.items():
            with self.subTest(question=question):
                self.assertEqual(
                    supervisor({"question": question})["route"], expected)

    def test_greeting_words_inside_a_question_are_not_a_greeting(self):
        for question in (
            "hello, what does the NIST AI RMF say about risk management?",
            "hi there, summarize chapter 3 of the manual",
            "what does the guide say about greetings between colonies?",
            "good morning rituals in a bee hive",
            "how are you supposed to inspect a hive in winter?",
        ):
            with self.subTest(question=question):
                self.assertFalse(is_conversational(question))

    def test_local_rag_still_runs_without_any_approval_prompt(self):
        result = self.graph.invoke(
            {"question": "What does chapter 3 say about propolis?"})
        self.assertEqual(result["route"], "books")
        self.assertEqual(self.calls, ["books"])
        self.assertEqual(result["answer"], "FAKE_BOOKS")
        self.assertNotIn("approval", result["answer"].lower())

    def test_requested_research_still_reaches_the_research_node(self):
        result = self.graph.invoke(
            {"question": "Search the latest AI news"})
        self.assertEqual(result["route"], "research")
        self.assertEqual(self.calls, ["research"])
        self.assertEqual(result["answer"], "FAKE_RESEARCH")

    def test_qa_accepts_the_new_routes(self):
        for question, _expected in ROUTE_TABLE:
            with self.subTest(question=question):
                result = self.graph.invoke({"question": question})
                self.assertEqual(
                    result.get("qa_flags") or [],
                    [],
                    f"{question!r} -> {result.get('route')}: "
                    f"{result.get('qa_flags')}")


class TestLlmRoutingModeIsPreserved(unittest.TestCase):
    """ROUTING_MODE=llm keeps its 4-label contract (Priority A adds no LLM
    classifier of its own — only the label vocabulary is unchanged)."""

    def test_llm_label_still_routes_a_substantive_request(self):
        import app.supervisor as sup

        with mock.patch.object(sup, "ROUTING_MODE", "llm"), \
             mock.patch.object(sup, "_llm_route", return_value="research"):
            result = sup.supervisor({"question": "What is a vector database?"})
        self.assertEqual(result["route"], "research")

    def test_llm_mode_never_classifies_a_conversational_opener(self):
        import app.supervisor as sup

        with mock.patch.object(sup, "ROUTING_MODE", "llm"), \
             mock.patch.object(sup, "_llm_route",
                               side_effect=AssertionError("LLM consulted")):
            for question in ("Hi", "Thanks", "How are you?", "Good morning"):
                with self.subTest(question=question):
                    self.assertEqual(
                        sup.supervisor({"question": question})["route"],
                        "conversational")


# ---------------------------------------------------------------------------
# 2026-10-30 (end-to-end milestone): the gaps a real request trace exposed.
# Each case is a request a user would actually type; before this increment the
# route it took made the answer wrong (a model hello, an invented analysis, a
# refusal to research, a fabricated "recommendation" from an empty memory).
# ---------------------------------------------------------------------------


class TestAssistantRoutingGaps(unittest.TestCase):
    def _route(self, question):
        return supervisor({"question": question})["route"]

    def test_greeting_plus_wellbeing_is_one_social_exchange(self):
        for question in ("Hi, how are you?", "Hello, how are you doing?",
                         "Good morning, how's it going?", "Hey there! how are you"):
            with self.subTest(question=question):
                self.assertTrue(is_conversational(question))
                self.assertEqual(self._route(question), "conversational")
                self.assertIn("Doing well", conversational_reply(question))

    def test_a_real_question_after_a_greeting_is_still_a_question(self):
        question = "Hi, how are you after reading chapter 3?"
        self.assertFalse(is_conversational(question))
        self.assertEqual(self._route(question), "books")

    def test_document_referents_reach_the_library(self):
        for question in (
            "Answer this question using my stored documents and cite the sources.",
            "Summarise these documents for me.",
            "Search the uploaded files for the pricing rules.",
            "What do my indexed PDFs say about governance?",
        ):
            with self.subTest(question=question):
                self.assertEqual(self._route(question), "books")

    def test_documents_without_a_referent_stay_general(self):
        for question in ("What is a document database?",
                         "Explain how document stores handle indexing."):
            with self.subTest(question=question):
                self.assertEqual(self._route(question), "general")

    def test_explicit_research_request_is_routed_and_authorized(self):
        question = ("Research three suitable open-source models, compare their "
                    "hardware requirements, and recommend one for my machine.")
        self.assertEqual(self._route(question), "research")
        allowed, code = research_authorization(question, "research")
        self.assertTrue(allowed)
        self.assertEqual(code, "explicit_research_request")

    def test_a_document_question_never_becomes_external_research(self):
        # "research what the manual says" is a local-library request: books is
        # checked first, so no outbound search is authorized by that wording.
        question = "Research what the manual says about hive inspections."
        self.assertEqual(self._route(question), "books")
        self.assertFalse(research_authorization(question, "books")[0])

    def test_dataset_reference_is_a_data_request_not_a_general_answer(self):
        for question in (
            "Analyze the dataset and tell me which products performed best.",
            "What is the monthly revenue trend in my sales data?",
        ):
            with self.subTest(question=question):
                self.assertEqual(self._route(question), "data")

    def test_prose_about_data_stays_general(self):
        for question in ("How is data stored in a relational database?",
                         "What is a dataset?"):
            with self.subTest(question=question):
                self.assertEqual(self._route(question), "general")

    def test_back_references_route_to_the_follow_up_capability(self):
        for question in ("What about the one you recommended?",
                         "Which of those is cheaper?",
                         "Compare the first one with the second.",
                         "Is your recommendation still valid?"):
            with self.subTest(question=question):
                self.assertEqual(self._route(question), "followup")

    def test_a_question_that_names_its_subject_is_not_a_follow_up(self):
        for question, expected in (
            ("What about the EU AI Act?", "general"),
            ("Compare the two documents in my knowledge base", "books"),
            ("Which database should I learn?", "general"),
            ("What does chapter 3 say about propolis?", "books"),
        ):
            with self.subTest(question=question):
                self.assertEqual(self._route(question), expected)

    # --- 2026-10-31 full-system audit (live cases C4, A5, B2) -------------- #
    def test_folding_unnamed_material_into_the_previous_answer_is_a_follow_up(self):
        # Live case C4: "Can you include those in the previous answer?" reached
        # the GENERAL route and claimed the previous answer already covered
        # things that were never inspected. It names no subject of its own.
        # `feedback` is the other correct answer for this family: a message
        # shaped as a correction revises the earlier result AND records it.
        # What must never happen is the general route answering it blind.
        for question in ("Can you include those in the previous answer?",
                         "Please add those to the previous answer.",
                         "What did the previous answer say about licensing?"):
            with self.subTest(question=question):
                self.assertIn(self._route(question), ("followup", "feedback"))

    def test_an_imperative_with_no_subject_asks_instead_of_answering(self):
        # Live case A5: "do the thing we discussed" matched the QUESTION shape
        # on its leading "do" and was answered as a general question.
        for question in ("do the thing we discussed",
                         "Do the thing you discussed.",
                         "do what we discussed"):
            with self.subTest(question=question):
                self.assertEqual(self._route(question), "unsupported")

    def test_a_local_document_request_never_falls_back_to_web_research(self):
        # Live case B2: "Using only the local documents, ..." was routed to
        # books, then the run proposed an outbound web search — contradicting
        # the user's own source constraint.
        for question in (
            "Using only the local documents, what does NIST AI 100-1 say "
            "about trustworthiness?",
            "In the indexed documents, what is the capital of France?",
        ):
            with self.subTest(question=question):
                self.assertEqual(self._route(question), "books")
                allowed, code = supervisor_module.fallback_decision({
                    "question": question,
                    "route": "books",
                    "books_refusal": {"reason": "self_refusal_after_recovery"},
                    "agent_answers": {},
                })
                self.assertFalse(allowed)
                self.assertEqual(code, "local_only_request")

    def test_a_source_file_action_reports_the_disabled_capability(self):
        # With no workspace configured the coding node is the ONLY place that
        # may explain this; "I have no tool for it" was untrue.
        with mock.patch("app.config.CODING_WORKSPACE_ROOT", ""):
            self.assertEqual(self._route("Fix the bug in app/graph.py"),
                             "coding")
            # A question ABOUT code keeps the general route.
            self.assertEqual(
                self._route("How does Python parse an indented block?"),
                "general")


if __name__ == "__main__":
    unittest.main(verbosity=2)
