"""Orchestrated knowledge fallback (Books -> internet research) — regression.

Covers the 10 required regression cases of the 2026-10-25 additional phase
("Orchestrated Knowledge Fallback"), driving the REAL books_agent (faked
query engine) through the REAL compiled graph:

  1. answerable-from-PDF question        -> local answer + citations, 0 web calls
  2. absent from books            -> fallback reaches the approval gate; no
                                      search runs without in-message approval
  3. relevant passages, failed synthesis -> bounded M5 recovery BEFORE fallback
  4. grounding-gate failure              -> gate never bypassed; research as appropriate
  5. explicit local-only request         -> abstain, research NEVER invoked
  6. explicit web-research request       -> routed directly to research AND (2026-10-28)
                                             the authorized research node really executes
  7. approval gate                      -> the REAL research node is reachable but is
                                            never reached; both limitations disclosed
  8. partly-local/partly-research        -> provenance stays distinguishable
  9. retry/fallback limits               -> one transition, one search, no loops
 10. privacy + external-call guards      -> query is the ORIGINAL question only;
                                            a fail-closed network guard is armed
                                            for this whole module (zero real
                                            network calls, asserted by test).

Offline by construction: setUpModule replaces app.tools.searxng_client.search
and app.tools.crawl_extractor.extract with raising stubs; every test either
fakes the research node or fakes those seams explicitly. No Chroma (the
fail-closed conftest guard stays armed), no Ollama/Gemini, no SearXNG.

2026-10-28 (Priority B, outbound research approval): the fallback's
EXTERNAL step stops at the approval gate — eligibility (fallback_decision),
graph topology, failure-class logging, the bounded recovery order and the
privacy rules are all unchanged, but a request that does not authorize web
research gets an approval REQUEST instead of a search. Full approval proof
lives in scripts/test_research_approval.py.
"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fail-closed Chroma isolation (docs/TESTING.md §F) for direct script runs.
from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

import app.agents.books_agent as ba  # noqa: E402
import app.agents.research_agent as ra  # noqa: E402
import app.graph as gm  # noqa: E402
import app.supervisor as sup  # noqa: E402
from app.config import NOT_FOUND_PHRASE  # noqa: E402
from app.state import with_agent_output  # noqa: E402
from app.tools import crawl_extractor, searxng_client  # noqa: E402
from langgraph.graph import END  # noqa: E402

GUIDE = "urban_beekeeping_field_guide.pdf"
RESEARCH_URL = "https://example.org/rmf"

BEE_PASSAGE = (
    "Field Guide to Urban Beekeeping. Chapter 1: Colony Structure. "
    "A healthy hive in midsummer can contain up to 60,000 worker bees, plus "
    "one queen and a few hundred drones depending on the season. The queen "
    "lays up to 2,000 eggs per day at the peak of spring buildup."
)
PROPOLIS_PASSAGE = (
    "Chapter 3: Hive Products. Propolis is a resinous substance bees collect "
    "from plant buds and use to seal small gaps and sterilize the interior "
    "of the hive. Honey is harvested only when its moisture content drops "
    "below 18 percent, which prevents fermentation during storage."
)
PROP_CLAIM = (
    "Propolis is a resinous substance bees collect from plant buds and use "
    "to seal small gaps and sterilize the interior of the hive."
)
ORBITAL_PASSAGE = (
    "Orbital Kinetics Principles. A Hohmann transfer between circular orbits "
    "requires two engine burns and depends on the delta-v budget of the "
    "spacecraft."
)
# M5-proven coverage-eligible fixture (scripts/test_self_refusal_recovery.py):
# no "the <docnoun>" wording, so it is NOT document-scoped and may fall back.
DELTA_QUESTION = (
    "According to orbital_kinetics_principles.docx, what does delta-v mean, "
    "and how is it used in orbital mechanics?"
)
RESEARCH_ANSWER = (
    "The AI Risk Management Framework was released in January 2023 by NIST."
    f"\n\nSources:\n- {RESEARCH_URL}"
)
# Non-deictic, corpus-identity question: routes to books, eligible for
# fallback (used by cases 2, 7, 8, 9 and the privacy case).
FALLBACK_QUESTION = ("What year was the AI Risk Management Framework "
                     "released?")


# ---------------------------------------------------------------------------
# Fail-closed network guard (case 10): zero real network calls in this module
# ---------------------------------------------------------------------------

def _blocked(*_args, **_kwargs):
    raise AssertionError(
        "network blocked: automated fallback tests must make zero external "
        "requests (searxng/crawl seams are faked per test)")


_ORIG_SEAMS: dict = {}


def setUpModule():
    _ORIG_SEAMS["search"] = searxng_client.search
    _ORIG_SEAMS["extract"] = crawl_extractor.extract
    searxng_client.search = _blocked
    crawl_extractor.extract = _blocked


def tearDownModule():
    # Restore the real seams for other suites in the same pytest session.
    searxng_client.search = _ORIG_SEAMS["search"]
    crawl_extractor.extract = _ORIG_SEAMS["extract"]


# ---------------------------------------------------------------------------
# Deterministic stand-ins (same conventions as the existing books suites)
# ---------------------------------------------------------------------------

def _node(score, text=BEE_PASSAGE, file_name=GUIDE, page="1"):
    return SimpleNamespace(score=score,
                           metadata={"file_name": file_name, "page_label": page},
                           text=text)


class _FakeResponse:
    def __init__(self, text, nodes):
        self._text = text
        self.source_nodes = nodes

    def __str__(self):
        return self._text


class _FakeEngine:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def query(self, question):
        self.calls += 1
        return self.response


class _ChatMsg:
    def __init__(self, text):
        self.content = text


class _ScriptedLLM:
    """Counts every (recovery) invocation; appends its marker to `order`."""

    def __init__(self, replies, order):
        self.replies = list(replies)
        self.calls = 0
        self.order = order

    def invoke(self, prompt):
        self.calls += 1
        self.order.append("recovery")
        if not self.replies:
            raise AssertionError("recovery beyond the scripted bound")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _ChatMsg(reply)


class _ResearchRecorder:
    """Stands in for the research node: records the full state it receives."""

    def __init__(self, order, answer=RESEARCH_ANSWER):
        self.order = order
        self.answer = answer
        self.calls = []

    def __call__(self, state):
        self.calls.append(dict(state))
        self.order.append("research")
        return with_agent_output(state, "research", self.answer,
                                 ["research", RESEARCH_URL])


class FallbackGraphBase(unittest.TestCase):
    """Real graph + real books_agent (faked engine/LLM) + recorded research."""

    def setUp(self):
        self._orig_engine = ba._get_query_engine
        self._orig_get_llm = ba.get_llm
        self._orig_research = gm.research_agent
        self.order = []
        self.research = _ResearchRecorder(self.order)
        gm.research_agent = self.research
        self.addCleanup(self._restore)

    def _restore(self):
        ba._get_query_engine = self._orig_engine
        ba.get_llm = self._orig_get_llm
        gm.research_agent = self._orig_research

    def _run(self, question, response, replies=(), research=None):
        """Compile the REAL graph (real supervisor/books/_after_books edges)
        and invoke it once; captures stdout log. `research` overrides the
        recorded research node for tests that need the real one."""
        if research is not None:
            gm.research_agent = research
        self.llm = _ScriptedLLM(list(replies), self.order)
        self.engine = _FakeEngine(response)
        ba._get_query_engine = lambda: self.engine
        ba.get_llm = lambda: self.llm
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.result = gm.build_graph().invoke({"question": question})
        self.log = buf.getvalue()
        return self.result


# ---------------------------------------------------------------------------
# Case 1 — answerable from an indexed PDF: local answer, no unnecessary web
# ---------------------------------------------------------------------------

class TestAnswerableLocalNoWeb(FallbackGraphBase):
    def test_answers_locally_with_citations_and_zero_research(self):
        result = self._run(
            "What does chapter 3 say about propolis?",
            _FakeResponse(PROP_CLAIM, [_node(0.74, PROPOLIS_PASSAGE,
                                             page="3")]))
        self.assertEqual(result["route"], "books")
        self.assertIn("References: urban_beekeeping_field_guide.pdf, page 3",
                      result["answer"])
        self.assertIn(PROP_CLAIM, result["answer"])
        self.assertEqual(result["sources"], ["books"])
        self.assertNotIn("books_refusal", result)   # answered runs carry none
        self.assertEqual(self.research.calls, [],
                         "answerable PDF question must make ZERO web calls")
        self.assertEqual(self.order, [])
        self.assertEqual(self.llm.calls, 0)         # no recovery needed


# ---------------------------------------------------------------------------
# Case 2 — absent from books, answerable through research: fallback fires
# ---------------------------------------------------------------------------

class TestFallbackToResearch(FallbackGraphBase):
    def test_refusal_reaches_the_approval_gate_instead_of_searching(self):
        # Below-floor score on an irrelevant chunk: pure refusal, M5
        # recovery ineligible by contract (zero model calls).
        result = self._run(FALLBACK_QUESTION,
                           _FakeResponse(NOT_FOUND_PHRASE,
                                         [_node(0.30, BEE_PASSAGE)]))
        self.assertEqual(result["route"], "books")
        self.assertEqual(result["books_refusal"],
                         {"reason": "low_confidence", "score": 0.30,
                          "nodes": 1, "recovery": "skipped_not_eligible"})
        # 2026-10-28 (Priority B): the transition is still taken, but the
        # EXTERNAL step stops at the approval gate — no search, no crawl,
        # no fabricated web answer.
        self.assertEqual(self.research.calls, [],
                         "fallback executed an unauthorized web search")
        answer = result["answer"]
        self.assertIn("NOT searched", answer)
        self.assertIn("External action proposed", answer)
        self.assertIn(FALLBACK_QUESTION, answer)
        # Both limitations are disclosed: local refusal + nothing searched.
        self.assertIn("indexed documents were consulted first", answer)
        self.assertTrue(
            result["agent_answers"]["books"].startswith(NOT_FOUND_PHRASE))
        # Failure-state classification is logged, not guessed from a score.
        self.assertIn(
            "knowledge fallback books->research "
            "(books_insufficient:no_local_evidence:low_confidence)",
            self.log)
        # No external provenance may be claimed for a search that never ran.
        self.assertEqual(result["sources"], ["books"])
        self.assertEqual(self.llm.calls, 0)
        self.assertEqual(self.engine.calls, 1)

    def test_authorized_request_still_executes_the_research_node(self):
        """2026-10-28 (Priority B): the gate must not disable research —
        a message that ITSELF asks for web research runs the node."""
        question = "Search the web for beekeeping guides"
        result = self._run(question,
                           _FakeResponse(NOT_FOUND_PHRASE, [_node(0.30)]))
        self.assertEqual(result["route"], "research")
        self.assertEqual(len(self.research.calls), 1)
        self.assertEqual(self.research.calls[0]["question"], question)
        self.assertEqual(result["answer"], RESEARCH_ANSWER)
        self.assertEqual(result["sources"], ["research", RESEARCH_URL])

class TestRecoveryBeforeFallback(FallbackGraphBase):
    def test_bounded_local_recovery_runs_before_the_fallback(self):
        # Proven coverage-eligible self-refusal; the recovery itself refuses
        # again, so the orchestrator may then fall back — in that order.
        result = self._run(DELTA_QUESTION,
                           _FakeResponse(NOT_FOUND_PHRASE,
                                         [_node(0.78, ORBITAL_PASSAGE)]),
                           replies=[NOT_FOUND_PHRASE])
        self.assertIn("self_refusal_recovery_attempted", self.log)
        self.assertIn("self_refusal_recovery_rejected", self.log)
        self.assertEqual(self.llm.calls, 1, "exactly ONE bounded recovery")
        self.assertEqual(self.engine.calls, 1, "no second retrieval cycle")
        self.assertEqual(self.order, ["recovery"],
                         "recovery completes, then the approval gate stops the "
                         "external step (2026-10-28)")
        self.assertEqual(result["books_refusal"],
                         {"reason": "self_refusal_after_recovery",
                          "score": 0.78, "nodes": 1, "recovery": "rejected"})
        self.assertIn(
            "knowledge fallback books->research "
            "(books_insufficient:recovery_exhausted:"
            "self_refusal_after_recovery)", self.log)
        self.assertEqual(self.research.calls, [],
                         "the external step must not run without approval")
        self.assertIn("NOT searched", result["answer"])
        self.assertIn("reason: self_refusal_after_recovery",
                      result["answer"])


# ---------------------------------------------------------------------------
# Case 4 — grounding failure: the gate is never bypassed
# ---------------------------------------------------------------------------

class TestGroundingGateNotBypassed(FallbackGraphBase):
    def test_unsupported_claim_never_reaches_the_answer_or_the_query(self):
        q = "What annual profit does chapter 3 report for propolis sales?"
        fabricated = "Chapter 3 reports annual propolis sales profit of $12,000."
        result = self._run(q, _FakeResponse(fabricated,
                                            [_node(0.74, PROPOLIS_PASSAGE)]))
        self.assertIn("gate_drop", self.log)
        self.assertEqual(result["books_refusal"],
                         {"reason": "grounding", "score": 0.74,
                          "nodes": 1, "recovery": "not_attempted"})
        # Local outcome stays an honest refusal with no footer…
        books_answer = result["agent_answers"]["books"]
        self.assertTrue(books_answer.startswith(NOT_FOUND_PHRASE))
        self.assertNotIn("References:", books_answer)
        # …the fabricated claim is in the final answer NOR sent anywhere:
        # the approval gate stopped the external step before a query existed.
        self.assertNotIn("$12,000", result["answer"])
        self.assertEqual(self.research.calls, [],
                         "no search may run to carry a rejected claim")
        self.assertIn("books_insufficient:"
                      "synthesis_validation_failed:grounding", self.log)
        self.assertIn("NOT searched", result["answer"])
        self.assertNotIn("Sources:", result["answer"])


# ---------------------------------------------------------------------------
# Case 5 — explicit local-only request: abstain, research never invoked
# ---------------------------------------------------------------------------

class TestLocalOnlyAbstains(FallbackGraphBase):
    def test_local_only_request_never_reaches_research(self):
        q = ("Using only this document, what exact chunk size in characters "
             "does the ingestion pipeline require?")
        result = self._run(q, _FakeResponse(NOT_FOUND_PHRASE, [_node(0.30)]))
        self.assertEqual(result["route"], "books")
        self.assertTrue(result["answer"].startswith(NOT_FOUND_PHRASE))
        self.assertEqual(self.research.calls, [],
                         "local-only request must never invoke research")
        self.assertNotIn("[graph] knowledge fallback", self.log)
        self.assertEqual(sup.fallback_decision(result),
                         (False, "local_only_request"))


# ---------------------------------------------------------------------------
# Case 6 — explicit web-research request: routed directly to research
# ---------------------------------------------------------------------------

class TestExplicitWebRouting(unittest.TestCase):
    WEB_REQUESTS = [
        # names a corpus document yet explicitly asks for web research
        "Google the four functions of the AI Risk Management Framework",
        "Search the web for beekeeping guides",          # books keywords present
        "Look up online the NIST AI RMF governance function",
        "What is the latest news about the NIST AI framework?",
    ]
    LOCAL_CONSTRAINTS = [
        # explicit no-internet wording outranks explicit web wording
        ("Do not search the web; use only this document about beekeeping",
         "books"),
        ("Using only this document, google the four functions", "books"),
        # document questions ABOUT web search stay on books
        ("What does the manual say about searching the web?", "books"),
    ]

    def test_explicit_web_requests_route_to_research(self):
        for q in self.WEB_REQUESTS:
            with self.subTest(q=q):
                self.assertEqual(sup.keyword_route({"question": q})["route"],
                                 "research")

    def test_local_only_constraints_outrank_web_and_current_info(self):
        for q, expected in self.LOCAL_CONSTRAINTS:
            with self.subTest(q=q):
                self.assertEqual(sup.keyword_route({"question": q})["route"],
                                 expected)

    def test_corpus_questions_still_route_to_books(self):
        # The new precedence must not steal ordinary document questions.
        for q in ("What does chapter 3 say about propolis?",
                  "Explain the four core functions of the AI Risk "
                  "Management Framework and what each one is meant to "
                  "accomplish."):
            with self.subTest(q=q):
                self.assertEqual(sup.keyword_route({"question": q})["route"],
                                 "books")


# ---------------------------------------------------------------------------
# Case 7 — research returns nothing: transparent limitation
# ---------------------------------------------------------------------------

class TestResearchFailureTransparent(FallbackGraphBase):
    def test_fallback_is_blocked_before_any_search_happens(self):
        """REAL research agent, search seam faked and RECORDED, zero network:
        the approval gate must stop the run before the seam is ever touched,
        and the reply must disclose BOTH limitations instead of one."""
        search_calls = []
        with mock.patch.object(ra, "_rewrite_query",
                               side_effect=lambda text: text), \
             mock.patch.object(searxng_client, "search",
                               side_effect=lambda text, num_results=8:
                               (search_calls.append(text) or [])):
            result = self._run(
                FALLBACK_QUESTION,
                _FakeResponse(NOT_FOUND_PHRASE, [_node(0.30, BEE_PASSAGE)]),
                research=ra.research_agent)
        # No search at all — approval was never granted.
        self.assertEqual(search_calls, [],
                         "an unauthorized search left the process")
        answer = result["answer"]
        self.assertIn("NOT searched", answer)
        self.assertIn("indexed documents were consulted first", answer)
        self.assertIn("reason: low_confidence", answer)
        self.assertNotIn("January 2023", answer)

    def test_real_research_node_is_never_reached_without_approval(self):
        """Case 8 (disclosure half): even with the REAL research node wired
        in and every seam faked to SUCCEED, a fallback run must stop at the
        approval gate — nothing external happens, and the answer claims
        nothing external happened."""
        search_calls = []
        fake_llm = SimpleNamespace(
            invoke=lambda _prompt: _ChatMsg(
                "The framework was published in January 2023."))
        with mock.patch.object(ra, "_rewrite_query",
                               side_effect=lambda text: text), \
             mock.patch.object(searxng_client, "search",
                               side_effect=lambda text, num_results=8:
                               (search_calls.append(text) or [
                                   {"url": RESEARCH_URL,
                                    "title": "NIST AI RMF",
                                    "snippet": "AI RMF overview"}])), \
             mock.patch.object(crawl_extractor, "extract",
                               side_effect=lambda url: "NIST AI RMF page "
                                                        "text."), \
             mock.patch.object(ra, "get_llm", return_value=fake_llm):
            result = self._run(
                FALLBACK_QUESTION,
                _FakeResponse(NOT_FOUND_PHRASE,
                              [_node(0.30, BEE_PASSAGE)]),
                research=ra.research_agent)
        self.assertEqual(search_calls, [],
                         "the real research node executed without approval")
        answer = result["answer"]
        self.assertIn("NOT searched", answer)
        self.assertNotIn(f"Sources:\n- {RESEARCH_URL}", answer)
        # Local refusal preserved separately; never merged as content.
        self.assertTrue(
            result["agent_answers"]["books"].startswith(NOT_FOUND_PHRASE))
        self.assertNotIn(NOT_FOUND_PHRASE, answer)


# ---------------------------------------------------------------------------
# Case 8 — partly local, partly research: provenance stays distinguishable
# ---------------------------------------------------------------------------

class TestProvenanceDistinguishable(FallbackGraphBase):
    def test_local_and_external_provenance_never_merge(self):
        result = self._run(FALLBACK_QUESTION,
                           _FakeResponse(NOT_FOUND_PHRASE,
                                         [_node(0.30, BEE_PASSAGE)]))
        answers = result["agent_answers"]
        # Local outcome preserved verbatim, separate from the gated reply.
        self.assertTrue(answers["books"].startswith(NOT_FOUND_PHRASE))
        # The research slot holds the approval REQUEST — it must carry no
        # external URL, because no external source was ever consulted.
        self.assertIn("NOT searched", answers["research"])
        self.assertNotIn(RESEARCH_URL, answers["research"])
        self.assertNotIn("Sources:", answers["research"])
        # The local refusal sentence is never absorbed into the answer body
        # as if it were content, and no external provenance is claimed.
        self.assertNotIn(NOT_FOUND_PHRASE, result["answer"])
        self.assertEqual(result["answer"], answers["research"])
        self.assertEqual(result["sources"], ["books"])
        self.assertNotIn("research", result["sources"])


# ---------------------------------------------------------------------------
# Case 9 — retry / fallback limits: one transition, no loops, no repeats
# ---------------------------------------------------------------------------

class TestFallbackLimits(unittest.TestCase):
    def test_graph_topology_allows_exactly_one_fallback_transition(self):
        self.assertEqual(gm._after_research({"route": "books"}), END,
                         "fallback research run must terminate, never loop")
        self.assertEqual(gm._after_research({"route": "research"}), END)
        self.assertEqual(gm._after_research({"route": "feasibility"}), "books",
                         "feasibility chain unchanged")

    def test_after_books_guards(self):
        refused = {"route": "books", "question": FALLBACK_QUESTION,
                   "books_refusal": {"reason": "low_confidence"}}
        self.assertEqual(gm._after_books(refused), "research")
        # research already answered -> never a second fallback
        already = dict(refused, agent_answers={"research": "x"})
        self.assertEqual(sup.fallback_decision(already),
                         (False, "research_already_ran"))
        self.assertEqual(gm._after_books(already), END)
        # feasibility chain precedence untouched
        self.assertEqual(gm._after_books({"route": "feasibility"}),
                         "feasibility")
        # answered books run -> terminal
        self.assertEqual(gm._after_books({"route": "books",
                                          "question": "anything"}), END)

    def test_config_switch_disables_fallback(self):
        refused = {"route": "books", "question": FALLBACK_QUESTION,
                   "books_refusal": {"reason": "low_confidence"}}
        with mock.patch.object(sup, "BOOKS_WEB_FALLBACK", False):
            self.assertEqual(sup.fallback_decision(refused),
                             (False, "fallback_disabled"))
            self.assertEqual(gm._after_books(refused), END)


class TestInvocationCounts(FallbackGraphBase):
    def test_each_run_invokes_each_node_exactly_once(self):
        self.llm = _ScriptedLLM([], self.order)
        self.engine = _FakeEngine(
            _FakeResponse(NOT_FOUND_PHRASE, [_node(0.30)]))
        ba._get_query_engine = lambda: self.engine
        ba.get_llm = lambda: self.llm
        graph = gm.build_graph()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            first = graph.invoke({"question": FALLBACK_QUESTION})
            second = graph.invoke({"question": FALLBACK_QUESTION})
        # Exactly one books query per run and NO research execution — the
        # approval gate holds on every run, never a repeated transition.
        self.assertEqual(self.engine.calls, 2)
        self.assertEqual(self.research.calls, [],
                         "a later run must not inherit approval")
        self.assertEqual(
            buf.getvalue().count("[graph] knowledge fallback"), 2)
        for result in (first, second):
            self.assertEqual(result["sources"].count("research"), 0)
            self.assertIn("NOT searched", result["answer"])


# ---------------------------------------------------------------------------
# Case 10 — privacy: only the original question leaves the process
# ---------------------------------------------------------------------------

class TestPrivacyGuards(FallbackGraphBase):
    def test_no_document_content_or_refusal_text_reaches_the_outbound_answer(self):
        result = self._run(FALLBACK_QUESTION,
                           _FakeResponse(NOT_FOUND_PHRASE,
                                         [_node(0.30, BEE_PASSAGE)]))
        # Nothing was sent anywhere: the external step never ran …
        self.assertEqual(self.research.calls, [],
                         "no state may leave the process without approval")
        answer_blob = json.dumps({"answer": result["answer"],
                                  **(result.get("books_refusal") or {})})
        # … and the reply that WILL be shown (and persisted) carries no
        # passage text and no refusal sentence — only the diagnosis.
        for marker in ("60,000", "worker bees", "queen"):
            self.assertNotIn(marker, answer_blob)
        self.assertNotIn(NOT_FOUND_PHRASE, result["answer"])
        self.assertIn(FALLBACK_QUESTION, result["answer"])
        diagnosis = result.get("books_refusal") or {}
        for key, value in diagnosis.items():
            with self.subTest(key=key):
                self.assertIsInstance(value, (str, int, float, type(None)))
                self.assertLess(len(str(value)), 40)

    def test_network_guard_is_armed_and_failing_closed(self):
        with self.assertRaises(AssertionError):
            searxng_client.search("must never execute")
        with self.assertRaises(AssertionError):
            crawl_extractor.extract("https://example.org")


if __name__ == "__main__":
    unittest.main(verbosity=2)
