"""Deterministic analysis operations for the Data Analyst (2026-10-30).

Every figure a Data Analyst answer contains is produced HERE, by plain
Python over the parsed rows. The model's only job (see
`app/agents/data_agent.py`) is to choose WHICH operations to run and to
explain their results; it never supplies a number, and it never sees the raw
rows.

Four operations, a closed set:

  profile        every column: inferred type, non-null, missing, distinct,
                 and for numeric columns min / max / mean / sum
  value_counts   one column: the most frequent values with counts
  group_agg      group by one column, aggregate another (sum/mean/min/max/
                 count), top groups by the aggregate
  filter_count   how many rows match one simple condition, and their share
                 (step: {"op": "filter_count", "column", "compare",
                 "value"} — `compare` is the operator, `op` the operation)

Each operation validates its own parameters and RAISES `AnalysisError` for
anything it cannot compute exactly — an unsupported column, a text column
asked for a sum, an unknown op, a bad operator. The caller reports that
refusal; nothing is approximated, sampled or invented.
"""

from typing import Dict, List, Optional, Sequence

from app.data.dataset import Column, Dataset, DatasetError

OPS = ("profile", "value_counts", "group_agg", "filter_count")

_FILTER_OPS = ("eq", "ne", "gt", "ge", "lt", "le", "contains")
_AGGS = ("sum", "mean", "min", "max", "count")

MAX_GROUPS = 20
MAX_VALUE_COUNTS = 10


class AnalysisError(Exception):
    """An operation cannot be computed exactly as requested."""


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def _pct(part: int, whole: int) -> str:
    if not whole:
        return "0%"
    return f"{100.0 * part / whole:.1f}%"


def _require_column(dataset: Dataset, name: object) -> Column:
    text = str(name or "").strip()
    if not text:
        raise AnalysisError("a column name is required")
    try:
        return dataset.column(text)
    except DatasetError as exc:
        raise AnalysisError(str(exc)) from exc


def _result(op: str, params: Dict[str, object], columns: Sequence[str],
            rows: Sequence[Sequence[str]], notes: Sequence[str] = ()
            ) -> Dict[str, object]:
    return {
        "op": op,
        "status": "ok",
        "params": dict(params),
        "table": {"columns": list(columns), "rows": [list(r) for r in rows]},
        "notes": list(notes),
    }


def _profile(dataset: Dataset, _step: Dict[str, object]) -> Dict[str, object]:
    rows: List[List[str]] = []
    for column in dataset.columns:
        numbers = dataset.numbers(column.name) if column.numeric else []
        rows.append([
            column.name,
            column.kind,
            str(column.non_null),
            str(column.missing),
            str(column.distinct),
            _fmt(min(numbers)) if numbers else "",
            _fmt(max(numbers)) if numbers else "",
            _fmt(sum(numbers) / len(numbers)) if numbers else "",
            _fmt(sum(numbers)) if numbers else "",
        ])
    return _result(
        "profile", {},
        ["column", "type", "non_null", "missing", "distinct",
         "min", "max", "mean", "sum"],
        rows,
        [dataset.coverage_note()])


def _value_counts(dataset: Dataset, step: Dict[str, object]) -> Dict[str, object]:
    column = _require_column(dataset, step.get("column"))
    counts: Dict[str, int] = {}
    for value in dataset.values(column.name):
        if value.strip().lower() in ("", "na", "n/a", "null", "none", "nan"):
            continue
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    raw_limit = step.get("limit")
    if raw_limit in (None, ""):
        limit = MAX_VALUE_COUNTS
    else:
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            raise AnalysisError(
                f"value_counts limit must be a whole number, not "
                f"{raw_limit!r}")
    if limit < 1 or limit > MAX_VALUE_COUNTS:
        raise AnalysisError(
            f"value_counts limit must be between 1 and {MAX_VALUE_COUNTS}")
    rows = [[value, str(count)] for value, count in ordered[:limit]]
    notes = [dataset.coverage_note(),
             f"{len(counts)} distinct value(s) in {column.name}"]
    if len(ordered) > limit:
        notes.append(f"showing the top {limit} of {len(ordered)} values")
    return _result("value_counts", {"column": column.name}, ["value", "count"],
                   rows, notes)


def _aggregate(values: Sequence[float], agg: str) -> float:
    if agg == "count":
        return float(len(values))
    if not values:
        raise AnalysisError("no numeric values to aggregate")
    if agg == "sum":
        return float(sum(values))
    if agg == "mean":
        return float(sum(values)) / len(values)
    if agg == "min":
        return float(min(values))
    return float(max(values))


def _group_agg(dataset: Dataset, step: Dict[str, object]) -> Dict[str, object]:
    group = _require_column(dataset, step.get("group_by"))
    agg = str(step.get("agg") or "").strip().lower()
    if agg not in _AGGS:
        raise AnalysisError(
            f"unsupported aggregation {agg or '(none)'!r} (use "
            + ", ".join(_AGGS) + ")")
    measure_name = str(step.get("measure") or "").strip()
    measure: Optional[Column] = None
    if agg != "count":
        measure = _require_column(dataset, measure_name)
        if not measure.numeric:
            raise AnalysisError(
                f"cannot {agg} column {measure.name!r}: it holds "
                f"{measure.kind} values, not numbers")
    elif measure_name:
        measure = _require_column(dataset, measure_name)

    buckets: Dict[str, List[float]] = {}
    group_position = dataset.index(group.name)
    measure_position = dataset.index(measure.name) if measure else None
    for row in dataset.rows:
        key = row[group_position].strip()
        if not key:
            continue
        buckets.setdefault(key, [])
        if measure_position is None:
            buckets[key].append(1.0)
            continue
        raw = row[measure_position]
        if str(raw).strip().lower() in ("", "na", "n/a", "null", "none", "nan"):
            continue
        try:
            buckets[key].append(float(str(raw).replace(",", "").strip()))
        except ValueError:
            raise AnalysisError(
                f"value {raw!r} in {measure.name} is not a number")

    computed = []
    empty_groups = 0
    for key, values in buckets.items():
        if agg == "count" and measure_position is not None:
            computed.append((key, float(len(values)), len(values)))
        elif agg != "count" and not values:
            # A group whose measure column is entirely missing cannot be
            # aggregated. It is OMITTED and counted, never shown as 0.
            empty_groups += 1
        else:
            computed.append((key, _aggregate(values, agg), len(values)))
    computed.sort(key=lambda item: (-item[1], item[0]))
    rows = [[key, _fmt(value), str(size)]
            for key, value, size in computed[:MAX_GROUPS]]
    notes = [dataset.coverage_note(),
             f"{len(computed)} group(s) in {group.name}"]
    if empty_groups:
        notes.append(
            f"{empty_groups} group(s) had no numeric {measure.name} values "
            "and are omitted (they are not zero)")
    if len(computed) > MAX_GROUPS:
        notes.append(f"showing the top {MAX_GROUPS} groups by {agg}")
    label = agg if measure is None else f"{agg}({measure.name})"
    return _result("group_agg",
                   {"group_by": group.name, "agg": agg,
                    "measure": measure.name if measure else None},
                   [group.name, label, "rows"], rows, notes)


def _compare(raw_value: str, column: Column, op: str, target: str) -> bool:
    if op == "contains":
        return target.lower() in raw_value.lower()
    if column.numeric:
        try:
            left = float(str(raw_value).replace(",", "").strip())
            right = float(str(target).replace(",", "").strip())
        except ValueError:
            raise AnalysisError(
                f"cannot compare {column.name!r} (numeric) with {target!r}")
    else:
        left, right = raw_value.strip().lower(), str(target).strip().lower()
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    if op == "lt":
        return left < right
    return left <= right


def _filter_count(dataset: Dataset, step: Dict[str, object]) -> Dict[str, object]:
    column = _require_column(dataset, step.get("column"))
    # `compare` (not `op`) carries the comparison: `op` already names the
    # operation itself, and overloading it silently dropped the comparison.
    op = str(step.get("compare") or "").strip().lower()
    if op not in _FILTER_OPS:
        raise AnalysisError(
            f"unsupported comparison {op or '(none)'!r} (use "
            + ", ".join(_FILTER_OPS) + ")")
    if "value" not in step or str(step.get("value")) == "":
        raise AnalysisError("a comparison value is required")
    target = str(step.get("value"))
    matched = 0
    considered = 0
    for raw in dataset.values(column.name):
        if str(raw).strip().lower() in ("", "na", "n/a", "null", "none", "nan"):
            continue
        considered += 1
        if _compare(raw, column, op, target):
            matched += 1
    return _result("filter_count",
                   {"column": column.name, "op": op, "value": target},
                   ["matched", "compared", "share"],
                   [[str(matched), str(considered), _pct(matched, considered)]],
                   [dataset.coverage_note()])


_RUNNERS = {
    "profile": _profile,
    "value_counts": _value_counts,
    "group_agg": _group_agg,
    "filter_count": _filter_count,
}


def run_step(dataset: Dataset, step: object) -> Dict[str, object]:
    """Run ONE validated step. Raises AnalysisError if it cannot be exact."""
    if not isinstance(step, dict):
        raise AnalysisError("each step must be an object")
    op = str(step.get("op") or "").strip().lower()
    if op not in _RUNNERS:
        raise AnalysisError(
            f"unsupported operation {op or '(none)'!r} (supported: "
            + ", ".join(OPS) + ")")
    return _RUNNERS[op](dataset, step)


def run_steps(dataset: Dataset, steps: Sequence[object]) -> List[Dict[str, object]]:
    """Run each step; a refusal is recorded, never replaced by a guess."""
    results: List[Dict[str, object]] = []
    for step in steps:
        try:
            results.append(run_step(dataset, step))
        except (AnalysisError, DatasetError) as exc:
            op = str((step or {}).get("op") if isinstance(step, dict) else "")
            results.append({
                "op": op or "(invalid step)",
                "status": "rejected",
                "params": dict(step) if isinstance(step, dict) else {},
                "table": {"columns": [], "rows": []},
                "notes": [str(exc)],
            })
    return results
