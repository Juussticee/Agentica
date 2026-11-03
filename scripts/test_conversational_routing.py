"""Greeting / conversational-opener routing + QA acceptance (2026-10-27).

Recorded failure (execution 3fc3bc8460c9): the user typed `hello`, the
keyword router matched nothing and fell through to its research DEFAULT, so
a greeting cost a live web search (8 results, 3 page crawls), produced an
irrelevant sourced answer about search engines, and the deterministic QA
gate accepted it because it only checks structure (route provenance +
non-empty answer).

Scope of this suite (all deterministic, zero network, zero LLM, zero Chroma):

  1. `hello`, `hi`, `good morning` -> conversational route, fixed reply,
     NO research/search/crawl/retrieval/LLM call of any kind.
  2. A greeting followed by real content keeps its existing route
     ("hello, what does the NIST AI RMF say about risk management?" -> books).
  3. Genuine research / ERP / feasibility / local-only requests are
     untouched.
  4. The recorded failure pattern (greeting + irrelevant web-summary answer)
     is now FLAGGED by QA; the fixed conversational reply is not.
  5. ROUTING_MODE=llm can never route a greeting to the model either.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fail-closed Chroma isolation (docs/TESTING.md §F), same as every suite
# that imports application modules directly.
from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

from app.qa_node import run_qa  # noqa: E402
from app.supervisor import (  # noqa: E402
    conversational_reply,
    is_conversational,
    keyword_route,
    supervisor,
)

GREETINGS = ["hello", "hi", "good morning"]
RECORDED_ANSWER = (
    "The provided sources highlight features like a human-first optimization "
    "engine for modern search, Google's search tools (including sustainability "
    "and privacy policies), and other services such as Gmail and Google Maps. "
    "The answer synthesizes these key points to reflect the main topics "
    "covered in the given web excerpts.\n\nSources:\n"
    "- https://www.hellosearch.io/\n- https://www.google.com/xhtml/search\n"
    "- https://hallo.zone/"
)


def _faked_graph(calls):
    """Real compiled graph with every AGENT node faked, recording calls.

    The conversational node, supervisor, planner and QA stay REAL — only the
    agent nodes (the ones that would touch LLM / search / crawl / Chroma) are
    replaced, which is the established convention across the routing suites.
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


class TestGreetingsRouteConversationally(unittest.TestCase):
    def setUp(self):
        self.calls = []
        gm, self._snapshot = _faked_graph(self.calls)
        self.gm = gm
        self.graph = gm.build_graph()

    def tearDown(self):
        for name, fn in self._snapshot.items():
            setattr(self.gm, name, fn)

    def test_hello_and_hi_and_good_morning_get_a_conversational_reply(self):
        for greeting in GREETINGS:
            with self.subTest(greeting=greeting):
                self.calls.clear()
                result = self.graph.invoke({"question": greeting})
                # routed to the conversational route, never an agent route
                self.assertEqual(result["route"], "conversational")
                # NO research, NO books, NO erp, NO feasibility ran
                self.assertEqual(self.calls, [], f"{greeting} invoked {self.calls}")
                # fixed reply, no provenance (nothing external was consulted)
                self.assertEqual(result["sources"], [])
                self.assertFalse(
                    any(tok in (result.get("answer") or "").lower()
                        for tok in ("http", "sources:", "references:")),
                    result.get("answer"))
                self.assertTrue((result.get("answer") or "").strip())
                # QA accepts the legitimate conversational exchange
                self.assertEqual(result.get("qa_flags") or [], [])

    def test_greeting_variants_and_casing_normalize(self):
        for raw in ("Hello!", "HI", "Good Morning?", "  hello  ", "hey there",
                    "thanks", "bye"):
            with self.subTest(raw=raw):
                self.assertTrue(is_conversational(raw), raw)
                self.assertEqual(
                    keyword_route({"question": raw})["route"],
                    "conversational", raw)

    def test_greeting_run_invokes_no_llm_search_crawl_or_retrieval(self):
        """Prove the tool seams are never touched, using the REAL agent
        modules (nothing faked): any contact with an LLM, the SearXNG
        client, the crawler or the books query engine fails the test."""
        import app.agents.books_agent as ba
        import app.agents.feasibility_agent as fa
        import app.agents.research_agent as ra
        import app.llm_client
        import app.tools.crawl_extractor as ce
        import app.tools.searxng_client as sc

        boom = mock.Mock(side_effect=AssertionError("tool was invoked"))
        with mock.patch.object(ba, "_get_query_engine", boom), \
             mock.patch.object(ba, "get_llm", boom), \
             mock.patch.object(ra, "get_llm", boom), \
             mock.patch.object(fa, "get_llm", boom), \
             mock.patch.object(app.llm_client, "get_llm", boom), \
             mock.patch.object(sc.requests, "get", boom), \
             mock.patch.object(ce, "_crawl", boom):
            # rebuild so the REAL agent nodes (not the fakes) are compiled in
            graph = self.gm.build_graph()
            result = graph.invoke({"question": "hello"})

        self.assertEqual(result["route"], "conversational")
        boom.assert_not_called()

    def test_llm_routing_mode_never_classifies_a_greeting(self):
        with mock.patch("app.supervisor.ROUTING_MODE", "llm"), \
             mock.patch("app.supervisor._llm_route",
                        side_effect=AssertionError("LLM consulted")) as route_mock:
            result = supervisor({"question": "good morning"})
        self.assertEqual(result["route"], "conversational")
        route_mock.assert_not_called()


class TestSubstantiveRequestsKeepTheirRoute(unittest.TestCase):
    """The fix must not swallow real questions that merely START with a
    greeting, nor alter any existing route."""

    def test_greeting_prefixed_document_question_still_goes_to_books(self):
        calls = []
        gm, snapshot = _faked_graph(calls)
        try:
            graph = gm.build_graph()
            result = graph.invoke({
                "question": "hello, what does the NIST AI RMF say about "
                            "risk management?",
            })
            self.assertEqual(result["route"], "books")
            self.assertEqual(calls, ["books"])
            self.assertEqual(result["answer"], "FAKE_BOOKS")
            self.assertEqual(result.get("qa_flags") or [], [])
        finally:
            for name, fn in snapshot.items():
                setattr(gm, name, fn)

    def test_genuine_web_research_question_still_routes_to_research(self):
        calls = []
        gm, snapshot = _faked_graph(calls)
        try:
            graph = gm.build_graph()
            result = graph.invoke(
                {"question": "What is the latest news on fusion energy "
                             "startups?"})
            self.assertEqual(result["route"], "research")
            self.assertEqual(calls, ["research"])
            self.assertEqual(result.get("qa_flags") or [], [])
        finally:
            for name, fn in snapshot.items():
                setattr(gm, name, fn)

    def test_existing_route_pins_are_unchanged(self):
        pins = {
            "What does chapter 3 say about propolis?": "books",
            "Show me open invoices": "erp",
            "Do a feasibility study for an urban beekeeping venture?":
                "feasibility",
            "Using only this document, explain the difference between the Map "
            "and Measure functions.": "books",
            # 2026-10-28 (Priority A): cue-less questions answer from model
            # knowledge instead of hitting the old research default (they
            # used to be pinned to "research" here).
            "Tell me about functions.": "general",
            "Who won the 2024 F1 championship?": "general",
        }
        for question, route in pins.items():
            with self.subTest(question=question):
                self.assertFalse(is_conversational(question), question)
                self.assertEqual(supervisor({"question": question})["route"],
                                 route, question)

    def test_greeting_words_inside_a_question_are_not_a_greeting(self):
        for question in (
            "hello, what does the NIST AI RMF say about risk management?",
            "hi there, summarize chapter 3 of the manual",
            "what does the guide say about greetings between colonies?",
            "good morning rituals in a bee hive",
        ):
            with self.subTest(question=question):
                self.assertFalse(is_conversational(question), question)


class TestQaRejectsTheRecordedFailure(unittest.TestCase):
    """The exact pattern from execution 3fc3bc8460c9 must not pass QA."""

    @staticmethod
    def _state(**overrides):
        state = {"question": "hello", "route": "research",
                 "answer": RECORDED_ANSWER,
                 "sources": ["research", "https://www.hellosearch.io/",
                             "https://www.google.com/xhtml/search",
                             "https://hallo.zone/"],
                 "agent_answers": {"research": RECORDED_ANSWER},
                 "qa_flags": []}
        state.update(overrides)
        return state

    def test_recorded_greeting_plus_web_summary_is_flagged(self):
        flags = run_qa(self._state())
        self.assertTrue(flags, "the recorded failure pattern passed QA")
        joined = " | ".join(flags)
        self.assertIn("conversational greeting", joined)
        self.assertIn("research", joined)

    def test_greeting_answered_by_books_is_flagged_too(self):
        flags = run_qa(self._state(route="books", sources=["books"],
                                   agent_answers={"books": "BOK"}))
        self.assertTrue(flags)
        self.assertIn("conversational greeting", " | ".join(flags))

    def test_legitimate_conversational_reply_is_not_flagged(self):
        state = self._state(route="conversational",
                            answer=conversational_reply("hello"),
                            sources=[],
                            agent_answers={"conversational":
                                           conversational_reply("hello")})
        self.assertEqual(run_qa(state), [])

    def test_substantive_research_answer_is_not_flagged(self):
        # A real research answer to a real question must stay clean — the new
        # check is scoped to bare conversational openers only.
        state = self._state(
            question="What is the latest news on fusion energy startups?",
            answer="A fusion energy startup announced a new reactor milestone "
                   "this week, according to three industry reports.",
        )
        self.assertEqual(run_qa(state), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
