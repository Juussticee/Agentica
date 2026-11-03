"""The `data` node — Data Analyst behaviour (2026-10-30) — offline.

Proves the CONTRACT, not just that a function ran:

  * the analyst reads ONLY a dataset inside the configured workspace (escape
    refused, unsupported extension refused, absent workspace refused);
  * `profile` is always computed by code, so a request is still answered
    usefully when the provider is down — and the answer then says that no
    further analysis was planned;
  * the model CHOOSES operations: a step it invents outside the closed set is
    reported as NOT run, never approximated; every number printed comes from
    app/data/analysis.py;
  * interpretation is labelled and separated from computed results, and the
    answer is explicit that nothing was written;
  * routing sends a named dataset to `data` only when a workspace is
    configured, and leaves ordinary questions, coding and web requests alone;
  * a planned `data` task executes through the existing dispatcher and lands
    in its own result channel.

Temporary workspaces, one scripted model seam (`app.llm_client.get_llm`) and
stubbed search/crawl: no network, no live model, no Chroma, no app startup.
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
from app.agents.data_agent import data_agent, dataset_paths  # noqa: E402
from app.coding.workspace import WorkspaceUnavailable  # noqa: E402
from app.data.dataset import load_dataset  # noqa: E402
from app.execution.runner import GraphRunner  # noqa: E402
from app.execution.store import ExecutionStore, STATUS_COMPLETED  # noqa: E402
from app.supervisor import keyword_route  # noqa: E402
from app.tools import crawl_extractor, searxng_client  # noqa: E402

SALES = (
    "region,units,revenue,note\n"
    "north,10,1000,ok\n"
    "south,5,500,\n"
    "east,7,700,ok\n"
    "north,3,300,ok\n"
    "west,,200,missing units\n"
)

QUESTION = "Which region has the highest revenue in sales.csv?"

PLAN_REPLY = json.dumps({
    "summary": "group revenue by region",
    "steps": [{"op": "group_agg", "group_by": "region",
               "measure": "revenue", "agg": "sum"}],
})
EXPLAIN_REPLY = "North leads on revenue across the two orders it has."


def _blocked(*_args, **_kwargs):
    raise AssertionError("network blocked: dataset analysis is offline")


class _ScriptedLLM:
    """Answers the plan + explanation prompts; raises on anything else."""

    def __init__(self, plan_reply=PLAN_REPLY, explain_reply=EXPLAIN_REPLY):
        self.plan_reply = plan_reply
        self.explain_reply = explain_reply
        self.prompts = []

    def invoke(self, prompt, *args, **kwargs):
        text = str(prompt)
        self.prompts.append(text)
        if "choosing analysis operations" in text:
            if isinstance(self.plan_reply, Exception):
                raise self.plan_reply
            return SimpleNamespace(content=self.plan_reply)
        if "explaining computed dataset results" in text:
            if isinstance(self.explain_reply, Exception):
                raise self.explain_reply
            return SimpleNamespace(content=self.explain_reply)
        raise AssertionError(
            "unexpected model call: " + text[:120].replace("\n", " "))


class DataCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "project"
        self.root.mkdir()
        (self.root / "sales.csv").write_text(SALES, encoding="utf-8")
        self.workspace = mock.patch("app.config.CODING_WORKSPACE_ROOT",
                                    str(self.root))
        self.workspace.start()
        self.addCleanup(self.workspace.stop)
        self.llm = _ScriptedLLM()
        self.patch_llm(lambda **_: self.llm)

    def patch_llm(self, factory):
        patcher = mock.patch("app.llm_client.get_llm", side_effect=factory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_node(self, question=QUESTION):
        return data_agent({"question": question})


class TestNode(DataCase):
    def test_computes_profile_and_the_chosen_operation(self):
        update = self.run_node()
        answer = update["answer"]
        self.assertIn("sales.csv", answer)
        self.assertIn("5 row(s)", answer)
        self.assertIn("Computed results", answer)
        # The grouped sum is computed, not paraphrased by the model.
        self.assertIn("| north | 1300 | 2 |", answer)
        self.assertIn("| east | 700 | 1 |", answer)
        # Interpretation is labelled and sourced from the tables.
        self.assertIn(EXPLAIN_REPLY, answer)
        self.assertIn("not a new calculation", answer)
        self.assertIn("read-only", answer)
        self.assertIn("data", update["agent_answers"])
        self.assertEqual(update["sources"], [],
                         "a local dataset is not an external source")

    def test_the_model_never_sees_a_data_row(self):
        self.run_node()
        plan_prompt = [p for p in self.llm.prompts
                       if "choosing analysis operations" in p][0]
        self.assertIn("region", plan_prompt)          # schema is shared
        for row_value in ("1000", "missing units", "north,10"):
            self.assertNotIn(row_value, plan_prompt,
                             "rows must never be sent to the provider")

    def test_provider_down_still_answers_with_the_computed_profile(self):
        self.patch_llm(lambda **_: (_ for _ in ()).throw(
            RuntimeError("provider down")))
        answer = self.run_node()["answer"]
        self.assertIn("Computed results", answer)
        self.assertIn("6.25", answer)                 # units mean, computed
        self.assertIn("No additional analysis was planned", answer)
        self.assertIn("provider is unavailable", answer)
        self.assertIn("not interpreted here", answer)

    def test_garbage_model_reply_never_invents_steps(self):
        self.llm.plan_reply = "I think you should look at the north region."
        answer = self.run_node()["answer"]
        self.assertIn("No additional analysis was planned", answer)
        self.assertIn("not usable JSON", answer)
        self.assertIn("Computed results", answer)

    def test_unsupported_step_is_reported_not_approximated(self):
        self.llm.plan_reply = json.dumps({
            "summary": "trend",
            "steps": [{"op": "regression", "x": "units", "y": "revenue"}],
        })
        answer = self.run_node()["answer"]
        self.assertIn("`regression` was NOT run", answer)
        self.assertIn("unsupported operation", answer)
        self.assertIn("Computed results", answer)

    def test_text_column_asked_for_a_sum_is_refused(self):
        self.llm.plan_reply = json.dumps({
            "summary": "bad aggregate",
            "steps": [{"op": "group_agg", "group_by": "region",
                       "measure": "note", "agg": "sum"}],
        })
        answer = self.run_node()["answer"]
        self.assertIn("`group_agg` was NOT run", answer)
        self.assertIn("not numbers", answer)

    def test_unknown_column_is_refused(self):
        self.llm.plan_reply = json.dumps({
            "summary": "wrong column",
            "steps": [{"op": "value_counts", "column": "profit"}],
        })
        answer = self.run_node()["answer"]
        self.assertIn("`value_counts` was NOT run", answer)
        self.assertIn("no column named", answer)

    def test_absent_dataset_asks_for_a_file(self):
        answer = self.run_node("Summarise the numbers for me")["answer"]
        self.assertIn("Name the dataset file", answer)
        self.assertNotIn("Computed results", answer)

    def test_missing_file_and_escape_are_reported(self):
        answer = self.run_node("Analyse forecast.csv please")["answer"]
        self.assertIn("I could not read a dataset", answer)
        self.assertIn("No analysis was performed", answer)

        outside = Path(self._tmp.name) / "outside.csv"
        outside.write_text("a\n1\n", encoding="utf-8")
        escaped = self.run_node("Analyse ../outside.csv")["answer"]
        self.assertIn("cannot contain '..' segments", escaped)
        self.assertIn("No analysis was performed", escaped)

    def test_disabled_workspace_says_so(self):
        with mock.patch("app.config.CODING_WORKSPACE_ROOT", ""):
            answer = self.run_node()["answer"]
        self.assertIn("not enabled", answer)
        self.assertIn("CODING_WORKSPACE_ROOT", answer)

    def test_two_datasets_analyses_the_first_and_says_so(self):
        (self.root / "other.csv").write_text("a\n1\n", encoding="utf-8")
        answer = self.run_node(
            "Compare sales.csv with other.csv")["answer"]
        self.assertIn("only `sales.csv`", answer)
        self.assertIn("ask again for the others", answer)


class TestRouting(DataCase):
    def test_dataset_request_routes_to_data(self):
        self.assertEqual(keyword_route({"question": QUESTION})["route"],
                         "data")

    def test_without_a_workspace_it_still_routes_here_to_say_so(self):
        """2026-10-30: the route no longer hides a disabled capability.

        The old contract sent this request to another route when no workspace
        was configured; in practice that meant the GENERAL route answered
        "analyse the dataset" from model knowledge — an invented analysis of
        data it never read. The data node is the only place allowed to decide
        whether it can run, so routing keeps it here and the node reports the
        real reason.
        """
        with mock.patch("app.config.CODING_WORKSPACE_ROOT", ""):
            self.assertEqual(keyword_route({"question": QUESTION})["route"],
                             "data")
            answer = self.run_node(QUESTION)["answer"]
        self.assertIn("not enabled", answer)
        self.assertIn("CODING_WORKSPACE_ROOT", answer)
        self.assertNotIn("Computed results", answer)

    def test_ordinary_questions_are_untouched(self):
        cases = {
            "What is a CSV file, and when should I use one?":
                "general",
            "Search the web for the latest pandas release":
                "research",
            "Explain the off-by-one bug in calc.py": "coding",
        }
        for question, expected in cases.items():
            self.assertEqual(keyword_route({"question": question})["route"],
                             expected, question)

    def test_dataset_paths_extraction(self):
        self.assertEqual(
            dataset_paths("Compare sales.csv and ../x.tsv"), 
            ["sales.csv", "../x.tsv"])
        self.assertEqual(dataset_paths("no file mentioned"), [])


class TestPlanDispatch(DataCase):
    def test_planned_data_task_runs_through_the_dispatcher(self):
        task = {
            "task_id": "t1",
            "capability": "data",
            "objective": "Total revenue by region in sales.csv",
            "depends_on": [],
            "required_inputs": ["question"],
            "expected_output": "revenue table",
            "done_when": "the table exists",
            "status": "pending",
        }
        state = {"question": QUESTION, "plan": {"tasks": [task]}}
        update = gm.dispatcher_node(state)
        executed = update["plan"]["tasks"][0]
        self.assertEqual(executed["status"], "completed")
        self.assertIn("1300", executed["result"])
        self.assertIn("data", update["agent_answers"])
        self.assertEqual(gm.CAPABILITY_AGENT_KEYS["data"], "data")


class TestMultiSpecialistPlan(DataCase):
    """The analyst's result must REACH the next specialist, not just exist."""

    def test_data_result_feeds_the_synthesis_task(self):
        self.llm = _ScriptedLLM(explain_reply="North leads.")
        plan = {"tasks": [
            {"task_id": "t1", "capability": "data",
             "objective": "Total revenue by region in sales.csv",
             "depends_on": [], "required_inputs": ["question"],
             "expected_output": "revenue table", "status": "pending"},
            {"task_id": "t2", "capability": "synthesis",
             "objective": "Recommend where to focus sales effort",
             "depends_on": ["t1"], "required_inputs": ["data"],
             "expected_output": "recommendation", "status": "pending"},
        ]}
        state = {"question": QUESTION, "route": "data", "plan": plan,
                 "agent_answers": {}}

        state.update(gm.dispatcher_node(state))
        self.assertEqual(state["plan"]["tasks"][0]["status"], "completed")
        self.assertEqual(state["plan"]["tasks"][1]["status"], "pending",
                         "the dependent task waits for its prerequisite")

        state.update(gm.dispatcher_node(state))
        synthesis = state["plan"]["tasks"][1]
        self.assertEqual(synthesis["status"], "completed")
        self.assertIn("1300", synthesis["result"],
                      "synthesis must carry the computed figure, not invent one")
        self.assertIn("data", state["agent_answers"])
        self.assertIn("synthesis", state["agent_answers"])


class TestThroughTheRunner(DataCase):
    def setUp(self):
        super().setUp()
        self._orig = {"search": searxng_client.search,
                      "extract": crawl_extractor.extract}
        searxng_client.search = _blocked
        crawl_extractor.extract = _blocked
        self.addCleanup(self._restore)
        self.store = ExecutionStore(Path(self._tmp.name) / "store.sqlite3")
        self.addCleanup(self.store.close)
        self.runner = GraphRunner(self.store)

    def _restore(self):
        searxng_client.search = self._orig["search"]
        crawl_extractor.extract = self._orig["extract"]

    def test_execution_completes_with_a_real_analysis_and_no_qa_flags(self):
        row = self.runner.run(self.runner.create(QUESTION)["id"])
        self.assertEqual(row["route"], "data")
        self.assertEqual(row["status"], STATUS_COMPLETED)
        self.assertIn("1300", row["answer"])
        self.assertEqual(row["qa_flags"], [])
        types = [e["type"] for e in self.store.get_events(row["id"])]
        self.assertIn("agent.completed", types)

    def test_failed_dataset_read_is_an_honest_completion(self):
        row = self.runner.run(self.runner.create(
            "Analyse missing.csv please")["id"])
        self.assertEqual(row["status"], STATUS_COMPLETED)
        self.assertIn("could not read a dataset", row["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
