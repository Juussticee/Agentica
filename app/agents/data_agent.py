"""The `data` node — Data Analyst specialist (2026-10-30).

Contract (what makes this a specialist rather than a prompt variant):

  ACCEPTS   a question about ONE named dataset file (`.csv`/`.tsv`) inside the
            configured workspace: what is in it, counts, totals, averages,
            distributions, "how many rows have X", comparisons across groups.
  REJECTS   files outside the workspace, non-dataset formats, and any
            operation outside the closed set in app/data/analysis.py — it
            reports the refusal instead of approximating.
  RECEIVES  the question (or, as a plan task, its own objective) — the
            original request travels separately as `request`.
  TOOLS     app/data/dataset.py (bounded read + typing) and
            app/data/analysis.py (deterministic computation). Nothing else.
  PERMISSIONS  read-only, workspace-confined, no network, no writes.
  OUTPUT    `answer` (schema, computed tables, coverage note, and clearly
            labelled interpretation) + `agent_answers["data"]`.
  FAILURE   provider down / unusable reply -> the DETERMINISTIC profile is
            still computed and reported; extra analysis is named as not
            planned. Never a fabricated number, never a silent fallback.
  VERIFICATION  `status` in the answer names exactly what ran (`profile` is
            always computed by code), and every table states whether it
            covered the whole file.

The model is consulted at TWO narrow seams only: choosing which operations to
run (as JSON, validated against the closed set) and explaining their results.
It never sees a row of data — only the schema — and it never produces a
number: the answer's figures are formatted from `run_steps` output.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from app.coding.workspace import (
    Workspace,
    WorkspaceError,
    configured_workspace,
    describe_workspace_problem,
    redact_secrets,
)
from app.data.analysis import OPS, run_steps
from app.data.dataset import Dataset, DatasetError, load_dataset
from app.state import AgentState, with_agent_output

# A dataset name, INCLUDING any leading ../ segments, so an escape is
# handed to the boundary intact instead of being silently reduced to an
# in-workspace name (which could analyse the wrong file).
_DATASET_RE = re.compile(
    r"(?:\.{1,2}/)*[\w][\w./\\-]*\.(?:csv|tsv)\b", re.IGNORECASE)

MAX_STEPS = 4

_PLAN_PROMPT = (
    "You are choosing analysis operations for a dataset. You are given ONLY "
    "the schema of {path} — never the rows — and the user's request.\n\n"
    "Choose a SMALL set of operations that answers the request. Reply with "
    "ONLY this JSON object:\n"
    '{{"summary": "one sentence", "steps": [ ... ]}}\n'
    "Allowed steps (each a JSON object, no others):\n"
    '- {{"op": "value_counts", "column": "<name>", "limit": <1-10>}}\n'
    '- {{"op": "group_agg", "group_by": "<name>", "measure": "<numeric '
    'name or omit>", "agg": "sum"|"mean"|"min"|"max"|"count"}}\n'
    '- {{"op": "filter_count", "column": "<name>", "compare": "eq"|"ne"|'
    '"gt"|"ge"|"lt"|"le"|"contains", "value": "<text>"}}\n'
    "Rules: use ONLY columns listed below, spelled exactly. Aggregating a "
    "column requires it to be numeric. If the request cannot be answered "
    'with these operations, return {{"summary": "why not", "steps": []}}.\n'
    "The `profile` operation is already computed; do not request it.\n\n"
    "Schema of {path}:\n{schema}\n\nUser request: {request}\n"
)

_EXPLAIN_PROMPT = (
    "You are explaining computed dataset results. Every number below was "
    "computed in Python; you must NOT recalculate, extrapolate, round "
    "differently, or add any figure that is not in these tables. Point out "
    "what the results mean for the user's request, note anything the data "
    "cannot answer, and be explicit about the coverage limit if one is "
    "stated. Keep it under 150 words.\n\nUser request: {request}\n\n"
    "Dataset: {path}\nSchema: {schema}\n\nComputed results:\n{tables}\n"
)


# -- helpers ------------------------------------------------------------------

def dataset_paths(question: str) -> List[str]:
    """Dataset files named in the request (workspace-relative, deduped)."""
    found: List[str] = []
    for match in _DATASET_RE.finditer(str(question or "")):
        token = match.group(0).replace("\\", "/")
        # Strip exactly one leading "./" — never a character set, which would
        # silently turn "../outside.csv" into an in-workspace name and hide
        # the escape from the boundary check.
        if token.startswith("./"):
            token = token[2:]
        if token and token.lower() not in [f.lower() for f in found]:
            found.append(token)
    return found


def _llm_text(prompt: str) -> str:
    from app.llm_client import get_llm, llm_text

    return str(llm_text(get_llm().invoke(prompt)) or "").strip()


def _parse_plan(text: object) -> Optional[dict]:
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.DOTALL)
    if fence:
        body = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", body, re.DOTALL)
        if brace:
            body = brace.group(0)
    try:
        parsed = json.loads(body)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _table_markdown(table: dict) -> List[str]:
    columns = [str(c) for c in (table.get("columns") or [])]
    rows = [list(r) for r in (table.get("rows") or [])]
    if not columns:
        return []
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = [str(cell) for cell in row] + [""] * (len(columns) - len(row))
        lines.append("| " + " | ".join(cells[:len(columns)]) + " |")
    return lines


def _render_result(result: dict) -> List[str]:
    op = str(result.get("op") or "?")
    if result.get("status") != "ok":
        reason = "; ".join(str(n) for n in result.get("notes") or []) or \
            "refused"
        return [f"- `{op}` was NOT run: {reason}"]
    params = result.get("params") or {}
    described = ", ".join(
        f"{key}={value}" for key, value in params.items() if value is not None)
    lines = [f"- `{op}`" + (f" ({described})" if described else "")]
    lines.extend(_table_markdown(result.get("table") or {}))
    for note in result.get("notes") or []:
        lines.append(f"  - {note}")
    return lines


def _request_steps(dataset: Dataset, request: str) -> Tuple[Optional[dict], str]:
    """Ask the model which extra operations to run. -> (plan, problem)."""
    try:
        reply = _llm_text(_PLAN_PROMPT.format(
            path=dataset.path, schema=dataset.schema_line(),
            request=str(request)[:800]))
    except Exception as exc:  # noqa: BLE001 — reported, never fatal
        return None, f"the AI provider is unavailable ({type(exc).__name__})"
    parsed = _parse_plan(reply)
    if parsed is None:
        return None, "the model's reply was not usable JSON"
    steps = parsed.get("steps")
    if steps is None:
        return None, "the model returned no steps field"
    if not isinstance(steps, list):
        return None, "the model's steps were not a list"
    return parsed, ""


def _interpret(dataset: Dataset, request: str, results: List[dict]) -> str:
    blocks: List[str] = []
    for result in results:
        if result.get("status") != "ok":
            continue
        table = result.get("table") or {}
        blocks.append(f"[{result.get('op')}] " + json.dumps(
            {"columns": table.get("columns"), "rows": table.get("rows")},
            ensure_ascii=False))
        for note in result.get("notes") or []:
            blocks.append(f"note: {note}")
    if not blocks:
        return ""
    try:
        return _llm_text(_EXPLAIN_PROMPT.format(
            request=str(request)[:800], path=dataset.path,
            schema=dataset.schema_line(), tables="\n".join(blocks)))
    except Exception:  # noqa: BLE001 — the computed tables stand on their own
        return ""


# -- the node -----------------------------------------------------------------

def data_agent(state: AgentState) -> dict:
    """Analyse ONE named dataset inside the workspace; report only computed numbers."""
    question = str(state.get("question") or "")
    request = str(state.get("request") or question)
    try:
        workspace = configured_workspace()
    except WorkspaceError as exc:
        return with_agent_output(
            state, "data", describe_workspace_problem(exc), [])
    if workspace is None:
        return with_agent_output(
            state, "data",
            "Dataset analysis is not enabled: no workspace root is "
            "configured, so no dataset can be read. Ask the operator to set "
            "CODING_WORKSPACE_ROOT.", [])

    paths = dataset_paths(question)
    if not paths:
        return with_agent_output(
            state, "data",
            "Name the dataset file to analyse (for example `sales.csv`) and "
            "I will read it from the configured workspace.", [])

    wording: List[str] = []
    if len(paths) > 1:
        wording.append(
            "You named " + ", ".join(paths) + f"; I analysed only "
            f"`{paths[0]}` — ask again for the others separately.")

    try:
        dataset = load_dataset(workspace, paths[0])
    except DatasetError as exc:
        answer = redact_secrets(
            "I could not read a dataset: " + str(exc) + " No analysis was "
            "performed.")
        return with_agent_output(state, "data", answer, [])

    if not dataset.rows:
        answer = redact_secrets(
            f"`{dataset.path}` has a header but no data rows, so there is "
            "nothing to compute.")
        return with_agent_output(state, "data", answer, [])

    # 1. Deterministic baseline — no model involved, always computed.
    results: List[dict] = run_steps(dataset, [{"op": "profile"}])

    # 2. Extra operations the request genuinely needs (model CHOOSES them).
    plan, problem = _request_steps(dataset, request)
    chosen: List[object] = []
    if plan is not None:
        chosen = list(plan.get("steps") or [])[:MAX_STEPS]
    summary = str((plan or {}).get("summary") or "").strip()
    if chosen:
        results.extend(run_steps(dataset, chosen))

    # 3. Interpretation over the COMPUTED tables only.
    explanation = _interpret(dataset, request, results)

    header = (
        f"Dataset: `{dataset.path}` — {dataset.row_count} row(s), "
        f"{len(dataset.columns)} column(s).")
    if dataset.truncated:
        header += (" The file is larger than the row cap, so every figure "
                   "below covers only the first "
                   f"{dataset.row_count} rows.")
    lines = [header,
             "",
             "Schema: " + dataset.schema_line(),
             "",
             "Computed results (all figures calculated in Python over the "
             "loaded rows):"]
    for result in results:
        lines.extend(_render_result(result))
    if plan is None:
        lines.append("")
        lines.append(
            f"- No additional analysis was planned: {problem}. The profile "
            "above is still computed and complete; ask again with the "
            "provider available for group/filter analysis.")
    elif summary:
        lines.append("")
        lines.append(f"Planned analysis: {summary}")
    for note in wording:
        lines.append("")
        lines.append(note)
    if explanation:
        lines.append("")
        lines.append("Interpretation (based only on the computed tables above; "
                     "not a new calculation):")
        lines.append(explanation)
    else:
        lines.append("")
        lines.append(
            "Interpretation: none produced (the AI provider is unavailable or "
            "returned nothing), so only the computed tables above are "
            "reported — they are not interpreted here.")
    lines.append("")
    lines.append("Nothing was written or modified; dataset analysis is "
                 "read-only.")

    answer = redact_secrets("\n".join(lines))
    return with_agent_output(state, "data", answer, [])


__all__ = ["data_agent", "dataset_paths", "OPS", "MAX_STEPS"]
