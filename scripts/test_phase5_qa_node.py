"""Phase 5 tests: production deterministic QA node (zero quota).

Covers MASTER_PLAN Phase 5 acceptance criteria:
  * qa_node appends to qa_flags, merging (never clearing) existing flags
  * returns ONLY qa_flags
  * fail-soft: an internal QA error becomes a flag, never a crash
  * end-to-end: every route's terminal hop passes through the gate
"""

import unittest
from unittest.mock import patch

from app.qa_node import has_incomplete_plan, qa_node, run_qa


def make_state(**overrides):
    state = {
        "question": "q",
        "route": "research",
        "sources": ["research"],
        "answer": "ANSWER",
        "qa_flags": [],
        "agent_answers": {"research": "R"},
    }
    state.update(overrides)
    return state


class TestQaNodeUnit(unittest.TestCase):
    def test_returns_only_qa_flags_key(self):
        update = qa_node(make_state())
        self.assertEqual(set(update.keys()), {"qa_flags"})

    def test_valid_single_route_has_no_flags(self):
        self.assertEqual(qa_node(make_state())["qa_flags"], [])

    def test_appends_to_existing_flags_never_clears(self):
        state = make_state(route="erp", sources=["books"], qa_flags=["QA: pre-existing"])
        result = qa_node(state)["qa_flags"]
        self.assertIn("QA: pre-existing", result)
        self.assertIn("QA: ERP route missing required erp provenance", result)

    def test_feasibility_missing_books_flagged(self):
        state = make_state(route="feasibility", agent_answers={"research": "R"}, sources=["research"])
        flags = qa_node(state)["qa_flags"]
        joined = " | ".join(flags)
        self.assertIn("missing required books output", joined)
        self.assertIn("missing upstream agent output(s): books", joined)

    def test_empty_answer_flagged(self):
        flags = qa_node(make_state(answer="   "))["qa_flags"]
        self.assertEqual(len(flags), 1)
        self.assertIn("final answer is empty", flags[0])

    def test_fail_soft_on_internal_error(self):
        with patch("app.qa_node.run_qa", side_effect=RuntimeError("boom")):
            flags = qa_node(make_state())["qa_flags"]
        self.assertEqual(len(flags), 1)
        self.assertIn("QA: internal error", flags[0])
        self.assertIn("boom", flags[0])

    def test_flag_order_deterministic(self):
        state = make_state(
            route="feasibility", agent_answers={}, sources=[], answer=""
        )
        flags = qa_node(state)["qa_flags"]
        self.assertEqual(
            flags,
            [
                "QA: feasibility route missing required research output",
                "QA: feasibility route missing required books output",
                "QA: feasibility route incomplete — missing upstream agent output(s): research, books",
                "QA: final answer is empty — workflow produced no output",
            ],
        )

    def test_completed_task_without_a_result_is_flagged(self):
        # 2026-10-29: the plan must not claim a step succeeded while recording
        # nothing it produced — that is a contradiction, not a warning about
        # the answer text.
        plan = {"source": "default", "tasks": [
            {"task_id": "t1", "capability": "books", "depends_on": [],
             "status": "completed"},
            {"task_id": "t2", "capability": "synthesis", "depends_on": ["t1"],
             "status": "completed", "result": "   "},
        ]}
        flags = qa_node(make_state(plan=plan))["qa_flags"]
        self.assertEqual(len(flags), 1)
        self.assertIn("completed task(s) with no result: t1, t2", flags[0])

    def test_completed_tasks_with_results_are_not_flagged(self):
        plan = {"source": "default", "tasks": [
            {"task_id": "t1", "capability": "books", "depends_on": [],
             "status": "completed", "result": "FINDINGS"},
        ]}
        self.assertEqual(qa_node(make_state(plan=plan))["qa_flags"], [])

    def test_run_qa_contract_unchanged_from_phase4(self):
        # Same expectations the 30 Phase 4 tests froze.
        self.assertEqual(run_qa(make_state(route="erp", sources=["erp"])), [])
        self.assertEqual(run_qa(make_state()), [])
        s = make_state(route="erp", sources=["books"])
        self.assertIn("missing required erp provenance", run_qa(s)[0])


class TestObjectiveLevelQa(unittest.TestCase):
    """2026-10-29: QA judges what each step ACCOMPLISHED, not string presence.

    Structural contradictions are findings; genuinely ambiguous cases are not.
    No entity/substring matching is attempted, so an equivalently named entity
    can never fail a task.
    """

    LONG_A = ("The AI Risk Management Framework is organized around the "
              "govern, map, measure and manage functions.")
    LONG_B = ("Invoice INV-2213 is 30 days overdue; the vendor contact is "
              "Northwind Supply, terms net 30.")

    def _state(self, tasks, answer="FINAL ANSWER", route="books"):
        return make_state(
            route=route, answer=answer, plan={"source": "llm", "tasks": tasks})

    def _task(self, tid, capability, objective, status="completed",
              depends_on=(), **extra):
        task = {"task_id": tid, "capability": capability,
                "objective": objective, "depends_on": list(depends_on),
                "status": status, "result": None}
        task.update(extra)
        return task

    def test_distinct_objectives_with_distinct_results_are_clean(self):
        state = self._state([
            self._task("t1", "books", "What does the framework say about the "
                       "govern function?", result=self.LONG_A),
            self._task("t2", "erp", "Which invoice is overdue?",
                       result=self.LONG_B),
        ])
        self.assertEqual(qa_node(state)["qa_flags"], [])

    def test_identical_results_for_different_capabilities_is_a_warning(self):
        state = self._state([
            self._task("t1", "books", "What does the framework say?",
                       result=self.LONG_A),
            self._task("t2", "research", "What does the framework say per "
                       "web sources?", result=self.LONG_A),
        ])
        flags = qa_node(state)["qa_flags"]
        self.assertEqual(len(flags), 1)
        self.assertIn("t1, t2", flags[0])
        self.assertIn("possibility, not proof", flags[0])
        self.assertFalse(has_incomplete_plan(state),
                         "a warning must not turn a finished plan into a "
                         "failure")

    def test_same_capability_identical_results_are_legitimate(self):
        state = self._state([
            self._task("t1", "erp", "Which invoices are overdue?",
                       result=self.LONG_B),
            self._task("t2", "erp", "Summarise overdue invoices by vendor",
                       result=self.LONG_B),
        ])
        self.assertEqual(qa_node(state)["qa_flags"], [],
                         "two lookups over the same data legitimately agree")

    def test_short_identical_results_are_not_compared(self):
        state = self._state([
            self._task("t1", "books", "a", result="OK"),
            self._task("t2", "research", "b", result="OK"),
        ])
        self.assertEqual(qa_node(state)["qa_flags"], [])

    def test_completed_dependent_of_a_failed_prerequisite_is_flagged(self):
        state = self._state([
            self._task("t1", "books", "a", status="failed",
                       error="boom"),
            self._task("t2", "erp", "b", depends_on=["t1"],
                       result=self.LONG_B),
        ])
        flags = qa_node(state)["qa_flags"]
        joined = " | ".join(flags)
        self.assertIn("recorded completed although a prerequisite", joined)
        self.assertIn("plan incomplete", joined)
        self.assertTrue(has_incomplete_plan(state))

    def test_blocked_task_with_completed_prerequisites_is_flagged(self):
        state = self._state([
            self._task("t1", "books", "a", result=self.LONG_A),
            self._task("t2", "erp", "b", depends_on=["t1"],
                       status="blocked", block_reason="dependency"),
        ])
        joined = " | ".join(qa_node(state)["qa_flags"])
        self.assertIn("blocked although every prerequisite completed", joined)

    def test_blocked_task_on_a_genuine_dependency_failure_is_not_double_flagged(self):
        state = self._state([
            self._task("t1", "books", "a", status="failed", error="boom"),
            self._task("t2", "erp", "b", depends_on=["t1"],
                       status="blocked", block_reason="dependency"),
        ])
        joined = " | ".join(qa_node(state)["qa_flags"])
        self.assertNotIn("blocked although every prerequisite", joined)
        self.assertIn("plan incomplete", joined)

    def test_approval_request_recorded_as_a_result_is_flagged(self):
        state = self._state([
            self._task("t1", "research", "look it up",
                       result=("I have NOT searched the web for this yet.\n\n"
                               "Approval request ID: abc")),
        ])
        joined = " | ".join(qa_node(state)["qa_flags"])
        self.assertIn("recorded an approval request as their result", joined)

    def test_task_left_running_is_flagged(self):
        state = self._state([self._task("t1", "erp", "a", status="running")])
        joined = " | ".join(qa_node(state)["qa_flags"])
        self.assertIn("still running when the execution ended", joined)

    def test_equivalent_entity_naming_is_never_a_failure(self):
        # The objective says "the framework"; the result names "the AI RMF".
        # No substring/entity matching is performed, so this is clean.
        state = self._state([
            self._task("t1", "books", "What does the framework require for "
                       "risk tolerance?",
                       result=("The AI RMF expects risk tolerance to be set by "
                               "senior leadership and reviewed periodically.")),
        ])
        self.assertEqual(qa_node(state)["qa_flags"], [])

    def test_insufficient_evidence_produces_no_invented_finding(self):
        # Objectives recorded, results too vague to judge: nothing is claimed.
        state = self._state([
            self._task("t1", "books", "anything about propolis",
                       result="See above."),
        ])
        self.assertEqual(qa_node(state)["qa_flags"], [])

    def test_step_that_echoes_its_instruction_is_flagged(self):
        # The specialist returned its own prompt: no work was done.
        objective = "Gather the ISO 42001 certification requirements"
        state = self._state([
            self._task("t1", "research", objective, result=objective),
        ])
        joined = " | ".join(qa_node(state)["qa_flags"])
        self.assertIn("echoed the question instead of doing work", joined)
        self.assertFalse(has_incomplete_plan(state),
                         "the check is diagnostic, not a failure verdict")

    def test_result_equal_to_the_request_is_flagged_as_an_echo(self):
        question = "Gather the ISO 42001 certification requirements"
        state = self._state(
            [self._task("t1", "research", "look it up", result=question)],
            answer=question)
        state["question"] = question
        joined = " | ".join(qa_node(state)["qa_flags"])
        self.assertIn("echoed the question instead of doing work", joined)

    def test_short_result_equal_to_a_short_objective_is_not_an_echo(self):
        state = self._state([self._task("t1", "erp", "OK", result="OK")])
        self.assertEqual(qa_node(state)["qa_flags"], [])

    def test_non_planned_state_has_no_objective_findings(self):
        self.assertEqual(qa_node(make_state())["qa_flags"], [])


class TestNumericGrounding(unittest.TestCase):
    """2026-10-31 live audit (phase E): fabricated benchmark figures passed QA.

    This is NOT a fact checker — a general answer consults no source, so the
    only honest check is whether the run can support the figure at all.
    Deliberately narrow (percentages, multipliers, data rates) and deliberately
    silent about arithmetic on user-supplied values.
    """

    def _flags(self, question, answer):
        return " | ".join(run_qa(make_state(
            question=question, route="general", sources=[], answer=answer,
            agent_answers={"general": answer})))

    def test_unsourced_percentage_is_flagged(self):
        joined = self._flags(
            "Is Redis a good session store?",
            "Redis is 10x faster and cuts load by 30-50%.")
        self.assertIn("cannot support", joined)
        self.assertIn("50%", joined)

    def test_unsourced_data_rate_is_flagged(self):
        joined = self._flags("How much faster is NVMe than SATA? Give a number.",
                             "NVMe offers 300-400 MB/s, SATA about 100-200 MB/s.")
        self.assertIn("cannot support", joined)

    def test_arithmetic_on_user_numbers_is_not_flagged(self):
        self.assertEqual(
            self._flags("If a request takes 40 ms and I add 60 ms, what is the "
                       "total latency in milliseconds?",
                       "The total latency is 100 ms."),
            "")

    def test_a_figure_the_user_supplied_is_supported(self):
        self.assertEqual(
            self._flags("We spend 40% of the budget on hosting. Is that high?",
                       "Yes, 40% is high for hosting at this scale."),
            "")

    def test_a_revision_may_reuse_the_figures_of_the_answer_it_revises(self):
        state = make_state(route="feedback", sources=[], question="Not enough.",
                          answer="Caching cuts load by 25%.",
                          agent_answers={"feedback": "Caching cuts load by 25%."},
                          prior_execution={"answer": "Caching cuts load by 25%."})
        self.assertEqual(run_qa(state), [])


class TestInstructionEcho(unittest.TestCase):
    """A local model echoes the prompt into its answer (live audit: the note
    "If the question requires code or a draft, I cannot provide that here"
    appeared in unrelated answers). The echo is removed before the answer is
    stored; genuine content is untouched."""

    PROMPT = ("Answer the user's general question concisely and factually from "
              "your own knowledge. If the request asks for code or a draft, put "
              "it in your reply: you cannot create files or run commands.\n\n"
              "Question: What is a checksum?")

    def test_an_echoed_instruction_line_is_removed(self):
        from app.llm_client import strip_instruction_echo

        answer = ("A checksum detects errors in data.\n\n**Note:** If the "
                  "question requires code or a draft, I cannot provide files "
                  "or run commands.")
        self.assertEqual(strip_instruction_echo(answer, self.PROMPT),
                         "A checksum detects errors in data.")

    def test_a_self_review_note_is_removed(self):
        from app.llm_client import strip_instruction_echo

        answer = ("Redis is faster.\n**Note**: The revised answer explicitly "
                  "covers concrete trade-offs.")
        self.assertEqual(strip_instruction_echo(answer, self.PROMPT),
                         "Redis is faster.")

    def test_real_content_survives(self):
        from app.llm_client import strip_instruction_echo

        answer = ("TCP is connection-oriented; UDP is connectionless. Both run "
                  "over IP, and applications choose based on delivery needs.")
        self.assertEqual(strip_instruction_echo(answer, self.PROMPT), answer)

    def test_a_fully_echoed_answer_is_never_blanked_to_nothing(self):
        from app.llm_client import strip_instruction_echo

        answer = ("You cannot create files or run commands; put the code in "
                  "your reply instead.")
        self.assertTrue(strip_instruction_echo(answer, self.PROMPT).strip())


class TestQaNodeInGraph(unittest.TestCase):
    """End-to-end: every route ends through the qa gate (faked agents)."""

    @classmethod
    def setUpClass(cls):
        import app.graph as graph_module
        cls.graph_module = graph_module
        # Save the real agents once; restore in tearDownClass so later suites
        # (test_phase3_erp with the REAL erp agent) never see these fakes.
        cls._snapshot = {
            name: getattr(graph_module, name)
            for name in ("books_agent", "research_agent", "feasibility_agent", "erp_agent")
        }
        cls._install_fakes()

    @classmethod
    def tearDownClass(cls):
        for name, fn in cls._snapshot.items():
            setattr(cls.graph_module, name, fn)

    @classmethod
    def _install_fakes(cls):
        graph_module = cls.graph_module

        def fake_books(state):
            return {
                "answer": "BOOKS",
                "sources": list(state.get("sources") or []) + ["books"],
                "agent_answers": {**(state.get("agent_answers") or {}), "books": "BOOKS"},
            }

        def fake_research(state):
            return {
                "answer": "RESEARCH",
                "sources": list(state.get("sources") or []) + ["research"],
                "agent_answers": {**(state.get("agent_answers") or {}), "research": "RESEARCH"},
            }

        def fake_feasibility(state):
            return {
                "answer": "STUDY",
                "sources": list(state.get("sources") or []) + ["feasibility"],
                "agent_answers": {**(state.get("agent_answers") or {}), "feasibility": "STUDY"},
            }

        def fake_erp(state):
            return {
                "answer": "ERP",
                "sources": list(state.get("sources") or []) + ["erp"],
                "agent_answers": {**(state.get("agent_answers") or {}), "erp": "ERP"},
            }

        cls.graph_module.books_agent = fake_books
        cls.graph_module.research_agent = fake_research
        cls.graph_module.feasibility_agent = fake_feasibility
        cls.graph_module.erp_agent = fake_erp

    def test_books_route_has_qa_flags_key(self):
        g = self.graph_module.build_graph()
        r = g.invoke({"question": "What does chapter 3 say about propolis?"})
        self.assertEqual(r["route"], "books")
        self.assertEqual(r["qa_flags"], [])

    def test_research_route_has_qa_flags_key(self):
        g = self.graph_module.build_graph()
        r = g.invoke({"question": "What happened in AI research this week?"})
        self.assertEqual(r["route"], "research")
        self.assertEqual(r["qa_flags"], [])

    def test_erp_route_qa_does_not_demand_research_books(self):
        g = self.graph_module.build_graph()
        r = g.invoke({"question": "Show me open invoices"})
        self.assertEqual(r["route"], "erp")
        self.assertEqual(r["qa_flags"], [])

    def test_feasibility_route_qa_flags_present(self):
        g = self.graph_module.build_graph()
        r = g.invoke(
            {"question": "Do a feasibility study for an urban beekeeping venture?"}
        )
        self.assertEqual(r["route"], "feasibility")
        self.assertEqual(r["qa_flags"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
