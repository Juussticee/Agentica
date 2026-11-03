"""Evidence-grounded general answering: deterministic offline tests.

Covers the behaviour the general route must show (docs/EVIDENCE_GROUNDED_
ANSWERING.md), with NO model, NO network and NO vector store: a scripted
provider double and a fixed passage list stand in for both. The point of
these tests is the pipeline's honesty — which claims survive, which are
disclosed, and what QA reports — not the quality of any model.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import evidence
from app.followup import followup_node
from app.graph import general_node
from app.qa_node import run_qa
from app.supervisor import supervisor

REDIS_BSD = (
    "Redis is distributed under the BSD-3 licence and is written in C. "
    "It keeps its dataset in memory."
)
REDIS_MIT = (
    "Redis is licensed under the MIT licence, which permits commercial "
    "redistribution with attribution."
)
MEMCACHED = (
    "Memcached is distributed under the Apache 2.0 licence and is also an "
    "in-memory key-value store."
)


def passage(text, score=0.82, name="guide.pdf", page="3"):
    return SimpleNamespace(
        text=text,
        metadata={"file_name": name, "page_label": page},
        score=score,
    )


class ScriptedModel:
    """Provider double: picks its reply from the prompt alone."""

    def __init__(self, from_evidence="", from_memory=""):
        self.from_evidence = from_evidence
        self.from_memory = from_memory
        self.prompts = []

    def invoke(self, prompt):
        text = str(prompt)
        self.prompts.append(text)
        if evidence.EVIDENCE_MARKER in text:
            return SimpleNamespace(content=self.from_evidence)
        return SimpleNamespace(content=self.from_memory)


def run_general(question, nodes, model, lessons=None):
    state = {"question": question, "route": "general", "sources": [],
             "qa_flags": [], "agent_answers": {}}
    if lessons is not None:
        state["lessons"] = lessons
    with patch.object(evidence, "retrieve", return_value=nodes), \
            patch("app.llm_client.get_llm", return_value=model):
        update = general_node(state)
    state.update(update)
    return state


class TestCoverageAssessment(unittest.TestCase):
    def test_relevant_passage_is_grounded(self):
        result = evidence.assess("What licence is Redis distributed under?",
                                 [passage(REDIS_BSD)])
        self.assertEqual(result["status"], "grounded")
        self.assertIn("redis", result["covered"])
        self.assertEqual(result["missing"], [])

    def test_topic_passage_without_the_detail_is_partial(self):
        result = evidence.assess(
            "How much VRAM does Redis need?",
            [passage("Redis keeps its dataset in memory.")])
        # "vram" and "need" are absent -> below the coverage floor -> none,
        # never presented as an answer to the VRAM question.
        self.assertEqual(result["status"], "none")
        self.assertIn("vram", result["missing"])

    def test_low_similarity_never_counts_as_evidence(self):
        result = evidence.assess(
            "What licence is Redis distributed under?",
            [passage(REDIS_BSD, score=0.2)])
        self.assertEqual(result["status"], "none")
        self.assertIn("below floor", result["reason"])

    def test_keyword_mention_is_not_support(self):
        """"Redis" appears, the question is about benchmarks: not evidence."""
        result = evidence.assess(
            "What is Redis throughput?",
            [passage("Redis is an in-memory data store.")])
        self.assertEqual(result["status"], "none")
        self.assertLess(result["coverage"], 0.6)

    def test_conflicting_licences_are_detected(self):
        result = evidence.assess(
            "What licence is Redis distributed under?",
            [passage(REDIS_BSD), passage(REDIS_MIT, page="9")])
        self.assertTrue(result["conflict"])

    def test_no_passages_is_none_not_an_error(self):
        result = evidence.assess("What licence is Redis distributed under?", [])
        self.assertEqual(result["status"], "none")
        self.assertEqual(result["passages"], 0)


class TestGroundedAnswers(unittest.TestCase):
    def test_complete_coverage_answers_from_evidence_with_citation(self):
        model = ScriptedModel(
            from_evidence="Redis is distributed under the BSD-3 licence.",
            from_memory="Redis uses the SSPL.")
        state = run_general("What licence is Redis distributed under?",
                            [passage(REDIS_BSD)], model)
        self.assertTrue(state["answer"].startswith(
            "Redis is distributed under the BSD-3 licence."))
        self.assertEqual(state["evidence"]["answered_from"], "evidence")
        self.assertTrue(any(s.startswith("evidence:")
                            for s in state["sources"]))
        self.assertIn("guide.pdf", state["answer"])
        self.assertNotIn("SSPL", state["answer"])

    def test_partial_coverage_names_the_missing_detail(self):
        model = ScriptedModel(
            from_evidence="Redis keeps its dataset in memory.")
        state = run_general(
            "Does Redis keep its dataset in memory and what is its VRAM use?",
            [passage(REDIS_BSD)], model)
        self.assertIn("Not stated by the passages", state["answer"])
        self.assertIn("vram", state["answer"].lower())

    def test_conflicting_passages_are_disclosed(self):
        model = ScriptedModel(
            from_evidence="Redis is distributed under the BSD-3 licence.")
        state = run_general("What licence is Redis distributed under?",
                            [passage(REDIS_BSD), passage(REDIS_MIT)],
                            model)
        self.assertTrue(state["evidence"]["conflict"])
        self.assertIn("different licences", state["answer"])

    def test_fabricated_citation_is_never_kept(self):
        diagnostics = []
        answer, _sources, detail = evidence.postprocess(
            "Redis is distributed under the BSD-3 licence "
            "(secret_internal.pdf, page 2).", [passage(REDIS_BSD)],
            diagnostics=diagnostics)
        self.assertEqual(detail["mode"], "refusal")
        self.assertEqual(answer, "")
        self.assertIn("filename", " ".join(diagnostics))

    def test_fabricated_table_reference_is_dropped(self):
        answer, _sources, detail = evidence.postprocess(
            "Redis is distributed under the BSD-3 licence, see table 7.",
            [passage(REDIS_BSD)])
        self.assertEqual(detail["mode"], "refusal")
        self.assertEqual(answer, "")

    def test_excerpt_markers_do_not_break_grounding(self):
        """Our own "[excerpt N]" convention is normalized away, not treated
        as unsupported content."""
        answer, sources, detail = evidence.postprocess(
            "Redis is distributed under the BSD-3 licence [excerpt 1].",
            [passage(REDIS_BSD)])
        self.assertEqual(detail["mode"], "ok")
        self.assertIn("BSD-3", answer)
        self.assertTrue(sources and sources[0].startswith("evidence:"))


class TestHonestFallbacks(unittest.TestCase):
    def test_no_documents_labels_the_answer_as_unchecked(self):
        model = ScriptedModel(from_memory="Redis uses the SSPL.")
        state = run_general("What licence is Redis distributed under?",
                            [], model)
        self.assertEqual(state["evidence"]["status"], "none")
        self.assertEqual(state["evidence"]["answered_from"], "model")
        self.assertEqual(state["sources"], [])
        self.assertIn("not checked against any source", state["answer"])
        self.assertIn("SSPL", state["answer"])   # useful, just labelled

    def test_insufficient_evidence_says_the_passages_do_not_state_it(self):
        model = ScriptedModel(from_memory="Redis needs about 2 GB of VRAM.")
        state = run_general("How much VRAM does Redis need?",
                            [passage("Redis keeps its dataset in memory.")],
                            model)
        self.assertEqual(state["evidence"]["answered_from"], "model")
        self.assertIn("no passage in your local document library covers "
                      "this", state["answer"])

    def test_evidence_that_fails_the_gate_falls_back_labelled(self):
        """Passages exist but the composed answer is not supported by them."""
        model = ScriptedModel(
            from_evidence="Redis is written in Rust and uses 12 GB of VRAM.",
            from_memory="Redis is written in C.")
        state = run_general("What language is Redis written in?",
                            [passage(REDIS_BSD)], model)
        self.assertEqual(state["evidence"]["answered_from"], "model")
        self.assertIn("no passage in your local document library covers "
                      "this", state["answer"])
        self.assertIn("Redis is written in C.", state["answer"])

    def test_degenerate_model_output_does_not_crash(self):
        model = ScriptedModel(from_evidence="...", from_memory="")
        state = run_general("What licence is Redis distributed under?",
                            [passage(REDIS_BSD)], model)
        self.assertTrue(state["answer"].strip())
        self.assertEqual(state["route"], "general")

    def test_provider_failure_is_reported_not_fabricated(self):
        class Broken:
            def invoke(self, prompt):
                raise RuntimeError("down")

        state = run_general("What licence is Redis distributed under?",
                            [passage(REDIS_BSD)], Broken())
        self.assertEqual(state["evidence"]["status"], "error")
        self.assertIn("provider is unavailable", state["answer"])
        self.assertEqual(state["sources"], [])

    def test_learned_requirements_reach_the_evidence_prompt(self):
        lesson = {
            "lesson_id": "l1", "status": "active", "capability": "general",
            "corrective_behavior": "Name the licence explicitly.",
            "validation": "check that the next answer covers licensing",
            "keywords": ["licence", "licensing"],
        }
        model = ScriptedModel(
            from_evidence="Redis is distributed under the BSD-3 licence.")
        state = run_general("What licence is Redis distributed under?",
                            [passage(REDIS_BSD)], model,
                            lessons=[lesson])
        self.assertIn("Name the licence explicitly", model.prompts[-1])
        self.assertEqual(state.get("lessons_applied"), ["l1"])


class TestClaimLevelQa(unittest.TestCase):
    def state_for(self, answer, ev, sources=None):
        return {"question": "What licence is Redis distributed under?",
                "route": "general", "answer": answer,
                "sources": sources if sources is not None else list(
                    ev.get("used") or []),
                "qa_flags": [], "agent_answers": {"general": answer},
                "evidence": ev}

    def test_licence_contradicting_its_own_passage_is_contradicted(self):
        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 1, "files": ["guide.pdf"], "text": REDIS_BSD,
              "used": ["evidence:guide.pdf, page 3"], "grounding": "ok"}
        flags = run_qa(self.state_for(
            "Redis is licensed under the MIT licence.", ev))
        self.assertTrue(any("conflicts with the consulted passages" in f
                            for f in flags), flags)

    def test_licence_about_an_entity_the_passages_never_cover(self):
        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 1, "files": ["guide.pdf"], "text": REDIS_BSD,
              "used": ["evidence:guide.pdf, page 3"], "grounding": "ok"}
        flags = run_qa(self.state_for(
            "Nginx is proprietary software for this use.", ev))
        self.assertTrue(any("unsupported" in f for f in flags), flags)

    def test_licence_contradicted_by_a_consulted_passage(self):
        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 2, "files": ["guide.pdf"],
              "text": REDIS_BSD + " " + MEMCACHED,
              "used": ["evidence:guide.pdf, page 3"], "grounding": "ok"}
        flags = run_qa(self.state_for(
            "Memcached is distributed under the BSD-3 licence.", ev))
        self.assertTrue(any("conflicts with the consulted passages" in f
                            for f in flags), flags)

    def test_licence_from_memory_is_flagged_unverified(self):
        ev = {"status": "none", "answered_from": "model", "passages": 0,
              "files": [], "text": "", "used": [], "grounding": "none"}
        flags = run_qa(self.state_for(
            "Redis is licensed under the MIT licence.", ev, sources=[]))
        self.assertTrue(any("no source was consulted" in f for f in flags),
                        flags)

    def test_audit_regression_unsourced_performance_and_throughput(self):
        ev = {"status": "none", "answered_from": "model", "passages": 0,
              "files": [], "text": "", "used": [], "grounding": "none"}
        flags = run_qa(self.state_for(
            "Redis is 10x faster than Memcached and can reduce load by "
            "30-50%, with NVMe delivering 300-400 MB/s.", ev, sources=[]))
        joined = " | ".join(flags)
        self.assertIn("cannot support", joined)
        self.assertIn("10x", joined)
        self.assertIn("400mb/s", joined)

    def test_figure_supported_by_the_passages_is_not_flagged(self):
        passage_text = "The cache reduces database load by 40% in tests."
        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 1, "files": ["guide.pdf"], "text": passage_text,
              "used": ["evidence:guide.pdf, page 3"], "grounding": "ok"}
        flags = run_qa(self.state_for(
            "The cache reduces database load by 40%.", ev))
        self.assertFalse([f for f in flags if "cannot support" in f], flags)

    def test_arithmetic_on_user_numbers_is_not_flagged(self):
        state = self.state_for(
            "100 ms.", {"status": "none", "answered_from": "model",
                        "passages": 0, "files": [], "text": "",
                        "used": [], "grounding": "none"}, sources=[])
        state["question"] = "We measured 40 ms and 60 ms. What is the total?"
        self.assertEqual([f for f in run_qa(state) if "cannot support" in f],
                         [])

    def test_disclosed_estimate_is_not_an_unsupported_claim(self):
        flags = run_qa(self.state_for(
            "Roughly 40% of reads hit the cache, by my estimate.",
            {"status": "none", "answered_from": "model", "passages": 0,
             "files": [], "text": "", "used": [], "grounding": "none"},
            sources=[]))
        self.assertFalse([f for f in flags if "cannot support" in f], flags)

    def test_dotted_retrieved_filename_is_not_a_fabricated_citation(self):
        """Regression (live 2026-11-01): "NIST.AI.100-1.pdf" was split at its
        dots and reported as the fabricated file "100-1.pdf"."""
        name = "NIST.AI.100-1.pdf"
        p = passage(REDIS_BSD, name=name)
        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 1, "files": [name.lower()],
              "text": REDIS_BSD, "used": [f"evidence:{name}, page 3"],
              "grounding": "ok"}
        flags = run_qa(self.state_for(
            "Redis is distributed under the BSD-3 licence "
            f"({name}, page 3).", ev))
        self.assertFalse([f for f in flags if "did not retrieve" in f], flags)
        self.assertEqual(p.metadata["file_name"], name)

    def test_fabricated_citation_in_the_final_answer_is_flagged(self):
        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 1, "files": ["guide.pdf"], "text": REDIS_BSD,
              "used": ["evidence:guide.pdf, page 3"], "grounding": "ok"}
        flags = run_qa(self.state_for(
            "Redis is distributed under the BSD-3 licence "
            "(architecture_handbook.pdf, page 4).", ev))
        self.assertTrue(any("did not retrieve" in f for f in flags), flags)

    def test_evidence_answer_without_provenance_is_flagged(self):
        ev = {"status": "grounded", "answered_from": "evidence",
              "passages": 1, "files": ["guide.pdf"], "text": REDIS_BSD,
              "used": [], "grounding": "ok"}
        flags = run_qa(self.state_for(
            "Redis is distributed under the BSD-3 licence.", ev,
            sources=[]))
        self.assertTrue(any("cites none" in f for f in flags), flags)

    def test_memory_answer_claiming_documents_is_flagged(self):
        ev = {"status": "none", "answered_from": "model", "passages": 0,
              "files": [], "text": "", "used": [], "grounding": "none"}
        flags = run_qa(self.state_for(
            "According to your documents, Redis is BSD-licensed.", ev,
            sources=[]))
        self.assertTrue(any("refers to local documents" in f for f in flags),
                        flags)
        self.assertTrue(any("licence" in f.lower() for f in flags))


class TestStateContract(unittest.TestCase):
    def test_evidence_channel_is_declared(self):
        """Regression (live 2026-11-01): an UNDECLARED state key is dropped by
        LangGraph, so QA never received the evidence record and the licence
        check stayed silent on a run that made an unfounded claim."""
        from app.state import AgentState

        self.assertIn("evidence", AgentState.__annotations__)


class TestRoutingUnchanged(unittest.TestCase):
    def test_local_only_instruction_still_uses_the_documents(self):
        self.assertEqual(
            supervisor({"question": "Using only the local documents, what "
                                  "does the AI RMF say about risk?"})["route"],
            "books")

    def test_explicit_web_authorization_still_uses_research(self):
        self.assertEqual(
            supervisor({"question": "Search the web and compare Redis and "
                                  "Memcached licences."})["route"],
            "research")

    def test_general_question_still_routes_to_general(self):
        self.assertEqual(
            supervisor({"question": "What licence is Redis distributed "
                                  "under?"})["route"], "general")


class TestFollowUpStillGroundedInRecords(unittest.TestCase):
    def test_follow_up_cites_the_recorded_execution(self):
        prior = {"id": "abc123", "question": "Compare caches",
                 "answer": "Redis was recommended.", "completed_at": "now"}
        state = {"question": "What about the one you recommended?",
                 "route": "followup", "sources": [], "qa_flags": [],
                 "agent_answers": {}, "prior_execution": prior}
        model = ScriptedModel(from_memory="Redis.")
        with patch("app.llm_client.get_llm", return_value=model):
            update = followup_node(state)
        self.assertEqual(update["sources"], ["execution:abc123"])


if __name__ == "__main__":
    unittest.main()
