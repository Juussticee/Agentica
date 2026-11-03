"""Dataset reader + deterministic analysis (2026-10-30) — offline.

Proves the layer that every Data Analyst figure comes from:

  * the file is read THROUGH the existing workspace boundary (escape refused),
    size-capped, row-capped (with the truncation recorded) and typed by
    inference, not by coercion;
  * `profile`, `value_counts`, `group_agg` and `filter_count` compute exact
    numbers over the parsed rows;
  * anything that cannot be computed exactly — a text column asked for a sum,
    an unknown column, an unsupported operation, a bad limit — is REFUSED
    with a reason, never approximated or silently dropped;
  * a refused step is recorded in `run_steps` output instead of raising, so a
    caller can report exactly which part of a request did not run.

Temporary workspaces only: no model, no network, no Chroma, no app startup.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.coding.workspace import Workspace  # noqa: E402
from app.data import analysis as analysis_module  # noqa: E402
from app.data import dataset as dataset_module  # noqa: E402
from app.data.analysis import AnalysisError, run_step, run_steps  # noqa: E402
from app.data.dataset import DatasetError, load_dataset  # noqa: E402

SALES = (
    "region,units,revenue,note\n"
    "north,10,1000,ok\n"
    "south,5,500,\n"
    "east,7,700,ok\n"
    "north,3,300,ok\n"
    "west,,200,missing units\n"
)


class DatasetCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "project"
        self.root.mkdir()
        (self.root / "sales.csv").write_text(SALES, encoding="utf-8")
        self.ws = Workspace(self.root)

    def load(self, name="sales.csv"):
        return load_dataset(self.ws, name)


class TestReader(DatasetCase):
    def test_header_types_and_missing_values(self):
        data = self.load()
        self.assertEqual([c.name for c in data.columns],
                         ["region", "units", "revenue", "note"])
        self.assertEqual([c.kind for c in data.columns],
                         ["text", "integer", "integer", "text"])
        self.assertEqual(data.row_count, 5)
        self.assertFalse(data.truncated)
        units = data.column("units")
        self.assertEqual((units.non_null, units.missing), (4, 1))
        self.assertEqual(data.column("note").missing, 1)
        self.assertEqual(data.column("region").distinct, 4)

    def test_coverage_note_names_the_file_and_row_count(self):
        self.assertIn("all 5 rows", self.load().coverage_note())

    def test_row_cap_is_recorded_not_hidden(self):
        with mock.patch.object(dataset_module, "MAX_ROWS", 3):
            data = self.load()
        self.assertEqual(data.row_count, 3)
        self.assertTrue(data.truncated)
        self.assertIn("first 3 rows", data.coverage_note())
        self.assertIn("more", data.coverage_note())

    def test_tsv_uses_tab_delimiter(self):
        (self.root / "orders.tsv").write_text(
            "id\tamount\n1\t10\n2\t20\n", encoding="utf-8")
        data = self.load("orders.tsv")
        self.assertEqual(data.delimiter, "\t")
        self.assertEqual(data.row_count, 2)
        self.assertEqual(data.column("amount").kind, "integer")

    def test_quoted_thousands_separator_is_numeric(self):
        (self.root / "money.csv").write_text(
            'name,total\na,"1,000"\nb,"2,500"\n', encoding="utf-8")
        data = self.load("money.csv")
        self.assertEqual(data.column("total").kind, "integer")
        self.assertEqual(data.numbers("total"), [1000.0, 2500.0])

    def test_ragged_rows_are_preserved(self):
        (self.root / "ragged.csv").write_text(
            "a,b,c\n1,2\n3,4,5,6\n", encoding="utf-8")
        data = self.load("ragged.csv")
        self.assertEqual(list(data.rows[0]), ["1", "2", ""])
        # An overflowing cell is joined into the last column, never dropped.
        self.assertEqual(data.rows[1][2], "5,6")

    def test_blank_lines_and_all_empty_rows_are_skipped(self):
        (self.root / "sparse.csv").write_text(
            "a,b\n1,2\n,\n\n3,4\n", encoding="utf-8")
        self.assertEqual(self.load("sparse.csv").row_count, 2)

    def test_header_only_file_has_no_rows(self):
        (self.root / "empty.csv").write_text("a,b\n", encoding="utf-8")
        self.assertEqual(self.load("empty.csv").row_count, 0)

    def test_empty_file_and_non_dataset_extension_are_refused(self):
        (self.root / "zero.csv").write_text("", encoding="utf-8")
        with self.assertRaises(DatasetError):
            self.load("zero.csv")
        (self.root / "notes.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(DatasetError) as caught:
            self.load("notes.txt")
        self.assertIn("not a dataset file", str(caught.exception))

    def test_escape_and_symlink_style_paths_are_refused(self):
        outside = Path(self._tmp.name) / "outside.csv"
        outside.write_text("a\n1\n", encoding="utf-8")
        with self.assertRaises(DatasetError):
            self.load("../outside.csv")
        with self.assertRaises(DatasetError):
            self.load(str(outside))

    def test_size_cap_is_refused_before_reading(self):
        with mock.patch.object(dataset_module, "MAX_DATASET_BYTES", 5):
            with self.assertRaises(DatasetError) as caught:
                self.load()
        self.assertIn("too large", str(caught.exception))

    def test_duplicate_and_blank_headers_become_unique_names(self):
        (self.root / "dupes.csv").write_text(
            "a,a,,a\n1,2,3,4\n", encoding="utf-8")
        data = self.load("dupes.csv")
        self.assertEqual([c.name for c in data.columns],
                         ["a", "a (2)", "column_3", "a (3)"])


class TestProfile(DatasetCase):
    def test_profile_computes_exact_numeric_figures(self):
        result = run_step(self.load(), {"op": "profile"})
        self.assertEqual(result["status"], "ok")
        table = result["table"]
        rows = {row[0]: row for row in table["rows"]}
        # units: 10, 5, 7, 3 (one missing)
        self.assertEqual(rows["units"][1], "integer")
        self.assertEqual(rows["units"][2], "4")     # non_null
        self.assertEqual(rows["units"][3], "1")     # missing
        self.assertEqual(rows["units"][5], "3")     # min
        self.assertEqual(rows["units"][6], "10")    # max
        self.assertEqual(rows["units"][7], "6.25")  # mean
        self.assertEqual(rows["units"][8], "25")    # sum
        # text columns carry no numeric aggregates at all (never 0)
        self.assertEqual(rows["region"][5:], ["", "", "", ""])
        self.assertIn("all 5 rows", " ".join(result["notes"]))

    def test_profile_of_row_less_dataset_lists_columns_with_zero_counts(self):
        (self.root / "none.csv").write_text("a,b\n", encoding="utf-8")
        result = run_step(self.load("none.csv"), {"op": "profile"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["table"]["rows"], [
            ["a", "empty", "0", "0", "0", "", "", "", ""],
            ["b", "empty", "0", "0", "0", "", "", "", ""],
        ], "no rows means empty columns, never a fabricated 0")


class TestValueCounts(DatasetCase):
    def test_counts_are_sorted_with_counts(self):
        result = run_step(self.load(), {"op": "value_counts",
                                        "column": "region"})
        self.assertEqual(result["table"]["rows"],
                         [["north", "2"], ["east", "1"], ["south", "1"],
                          ["west", "1"]])

    def test_missing_values_are_not_a_category(self):
        result = run_step(self.load(), {"op": "value_counts",
                                        "column": "note"})
        values = [row[0] for row in result["table"]["rows"]]
        self.assertEqual(values, ["ok", "missing units"])

    def test_bad_limit_is_refused(self):
        for limit in (0, 99, "many"):
            with self.assertRaises(AnalysisError):
                run_step(self.load(), {"op": "value_counts",
                                       "column": "region", "limit": limit})


class TestGroupAgg(DatasetCase):
    def test_sum_by_group(self):
        result = run_step(self.load(), {
            "op": "group_agg", "group_by": "region",
            "measure": "revenue", "agg": "sum"})
        self.assertEqual(result["table"]["rows"], [
            ["north", "1300", "2"],
            ["east", "700", "1"],
            ["south", "500", "1"],
            ["west", "200", "1"],
        ])
        self.assertEqual(result["table"]["columns"],
                         ["region", "sum(revenue)", "rows"])

    def test_mean_ignores_missing_measure_values(self):
        result = run_step(self.load(), {
            "op": "group_agg", "group_by": "region",
            "measure": "units", "agg": "mean"})
        rows = dict((row[0], row[1]) for row in result["table"]["rows"])
        self.assertEqual(rows["north"], "6.5")   # (10 + 3) / 2
        # west has NO units at all: it is omitted and disclosed, not shown
        # as a zero (which would be a fabricated figure).
        self.assertNotIn("west", rows)
        self.assertIn("omitted", " ".join(result["notes"]))
        self.assertIn("not zero", " ".join(result["notes"]))

    def test_count_needs_no_measure(self):
        result = run_step(self.load(), {
            "op": "group_agg", "group_by": "region", "agg": "count"})
        rows = dict((row[0], row[1]) for row in result["table"]["rows"])
        self.assertEqual(rows["north"], "2")

    def test_text_measure_for_sum_is_refused(self):
        with self.assertRaises(AnalysisError) as caught:
            run_step(self.load(), {"op": "group_agg", "group_by": "region",
                                   "measure": "note", "agg": "sum"})
        self.assertIn("not numbers", str(caught.exception))

    def test_unknown_aggregation_and_column_are_refused(self):
        with self.assertRaises(AnalysisError):
            run_step(self.load(), {"op": "group_agg", "group_by": "region",
                                   "measure": "revenue", "agg": "median"})
        with self.assertRaises(AnalysisError):
            run_step(self.load(), {"op": "group_agg", "group_by": "nope",
                                   "measure": "revenue", "agg": "sum"})

    def test_group_cap_is_reported(self):
        rows = "region,revenue\n" + "".join(
            f"r{i},{i}\n" for i in range(30))
        (self.root / "many.csv").write_text(rows, encoding="utf-8")
        result = run_step(self.load("many.csv"), {
            "op": "group_agg", "group_by": "region",
            "measure": "revenue", "agg": "sum"})
        self.assertEqual(len(result["table"]["rows"]), 20)
        self.assertIn("top 20", " ".join(result["notes"]))


class TestFilterCount(DatasetCase):
    def test_numeric_comparison_counts_and_share(self):
        result = run_step(self.load(), {
            "op": "filter_count", "column": "revenue", "compare": "gt",
            "value": "500"})
        self.assertEqual(result["table"]["rows"], [["2", "5", "40.0%"]])

    def test_text_contains(self):
        result = run_step(self.load(), {
            "op": "filter_count", "column": "note", "compare": "contains",
            "value": "ok"})
        # "missing units" does not contain "ok"; "ok" appears 3 times.
        self.assertEqual(result["table"]["rows"][0][0], "3")

    def test_missing_values_are_not_compared(self):
        result = run_step(self.load(), {
            "op": "filter_count", "column": "units", "compare": "ge",
            "value": "1"})
        self.assertEqual(result["table"]["rows"], [["4", "4", "100.0%"]])

    def test_bad_operator_value_and_column_are_refused(self):
        for step in (
            {"op": "filter_count", "column": "units", "compare": "like",
             "value": "1"},
            {"op": "filter_count", "column": "units", "compare": "gt"},
            {"op": "filter_count", "column": "ghost", "compare": "eq",
             "value": "1"},
            # `op` is the OPERATION, so a comparison placed there is refused
            # instead of silently being read as the operator.
            {"op": "filter_count", "column": "units", "op_": "gt",
             "value": "1"},
        ):
            with self.assertRaises(AnalysisError):
                run_step(self.load(), step)


class TestRunSteps(DatasetCase):
    def test_supported_step_runs_and_refused_step_is_recorded(self):
        results = run_steps(self.load(), [
            {"op": "value_counts", "column": "region"},
            {"op": "regression", "column": "region"},
            {"op": "group_agg", "group_by": "region", "measure": "note",
             "agg": "sum"},
            "not even an object",
        ])
        self.assertEqual([r["status"] for r in results],
                         ["ok", "rejected", "rejected", "rejected"])
        self.assertIn("unsupported operation",
                      results[1]["notes"][0])
        self.assertIn("not numbers", results[2]["notes"][0])
        self.assertIn("must be an object", results[3]["notes"][0])
        # Every refusal names the operation it refused.
        self.assertEqual(results[1]["op"], "regression")
        self.assertEqual(results[2]["op"], "group_agg")

    def test_operation_set_is_closed_and_documented(self):
        self.assertEqual(sorted(analysis_module.OPS),
                         ["filter_count", "group_agg", "profile",
                          "value_counts"])
        # The name is normalized, but nothing outside the set runs.
        run_step(self.load(), {"op": "PROFILE"})
        for unknown in ("regression", "sql", "join", "chart"):
            with self.assertRaises(AnalysisError):
                run_step(self.load(), {"op": unknown})


if __name__ == "__main__":
    unittest.main(verbosity=2)
