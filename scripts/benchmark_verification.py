"""M4B-H deterministic verification benchmark runner (offline, zero quota).

Runs every case in benchmarks/verification_cases.json through the REAL
books_agent with a faked query engine, on the three layers the milestone
requires each case to distinguish:

  Layer 1 - deterministic M3 baseline: BOOKS_VERIFIER=off (pure gate).
            Measured outcome compared with the case's `recorded_baseline`
            (drift gate: MATCH / CHANGED / UNRECORDED), plus an
            informational baseline-gap marker vs target_outcomes (the KNOWN
            deterministic defect surface — D1/D2 style — that never fails
            --check on its own).
  Layer 2 - mocked verifier: BOOKS_VERIFIER=local with
            app.verifier.verify_all SCRIPTED to the case's `mock_verifier`
            verdict (no model, no network). The scripted verdict is a
            fixed input; this layer measures how the PIPELINE consumes each
            verdict class, never model quality.
  Layer 3 - integrated pipeline outcome: the user-visible result of layer 2
            (accept | accept_disclosed | refusal) plus target flags,
            compared with the design `target_outcomes`.

The deterministic QUALITY REPORT is computed from `expected_verdict` (the
semantically correct classification by construction): false-acceptance,
false-rejection, contradiction (numeric/relational/negation split),
paraphrase, malformed, timeout/unavailable, citation integrity, and budget
exclusion counts — always raw counts AND denominators AND percentage.
It measures deterministic pipeline behavior against constructed truth; it
is NOT a measurement of live model quality (that is the optional
scripts/eval_verifier_local.py experiment).

No network, no embeddings, no LLM, no mutation of the live index: the
query engine is faked per case and the verifier is scripted per case.

Usage:
  python scripts/benchmark_verification.py            # table + quality report, exit 0
  python scripts/benchmark_verification.py --check    # exit 1 on baseline CHANGED
                                                      # or integrated-layer GAP
  python scripts/benchmark_verification.py --record   # print id -> measured lines
                                                      # for null recorded baselines
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import app.agents.books_agent as ba  # noqa: E402
import app.verifier as verifier  # noqa: E402
from app.agents.books_agent import books_agent  # noqa: E402
from app.config import BOOKS_VERIFIER, NOT_FOUND_PHRASE  # noqa: E402

CASES_PATH = PROJECT_ROOT / "benchmarks" / "verification_cases.json"

_FILE_RE = re.compile(r"\b[\w][\w\-]*\.(?:pdf|docx)\b", re.I)

# Claims fed to the mocked verifier layer (budget accounting for §5).
_SCRIPTED_CLAIMS = 0

# Effective per-claim scripted verdicts AFTER id-whitelist validation
# (M4C): what the pipeline actually consumed — a mis-attached evidence id
# shows up here as unavailable(malformed), never as a pass.
_SCRIPTED_EFFECTS = []


def _nodes_for(case):
    nodes = []
    for evidence in case["evidence"]:
        meta = {k: v for k, v in evidence.items()
                if k != "text" and v is not None}
        nodes.append(SimpleNamespace(score=0.7, metadata=meta,
                                     text=evidence["text"]))
    return nodes


def _scripted_verify_all(case):
    """Scripted verify_all faithful to the production strict parser.

    Per-claim overrides (M4C): `mock_verifier_per_claim` (index-aligned
    verdict specs) and `mock_evidence_ids` (index-aligned id lists).
    Ids are validated against each item's OWN evidence exactly like
    `_parse_output` would (decisive verdicts must cite a non-empty subset
    of the per-claim whitelist); a violation degrades to
    `unavailable(malformed)` — a mis-attached id is treated exactly like
    a fabricated one, never silently accepted.
    """
    spec_default = case.get("mock_verifier")
    per_claim = case.get("mock_verifier_per_claim")
    ids_per_claim = case.get("mock_evidence_ids")
    if spec_default is None and not per_claim:
        raise ValueError(f"case {case.get('id')!r} has no scripted "
                         "verifier verdict")

    def _verify_all(items):
        global _SCRIPTED_CLAIMS
        _SCRIPTED_CLAIMS += len(items)
        out = []
        for i, item in enumerate(items):
            ids = [str(e.get("id")) for e in (item.get("evidence") or [])]
            spec = (per_claim[i] if per_claim and i < len(per_claim)
                    else spec_default)
            if spec is None:
                raise ValueError(f"case {case.get('id')!r}: "
                                 f"mock_verifier_per_claim too short for "
                                 f"item {i}")
            cited = (ids_per_claim[i] if ids_per_claim
                     and i < len(ids_per_claim) else ids)
            if spec.startswith("unavailable:"):
                _SCRIPTED_EFFECTS.append(spec)
                out.append({"status": verifier.STATUS_UNAVAILABLE,
                            "verdict": None, "evidence_ids": [],
                            "reason": "", "error": spec.split(":", 1)[1]})
                continue
            decisive = spec in (verifier.VERDICT_SUPPORTED,
                                verifier.VERDICT_CONTRADICTED)
            if (decisive and (not cited
                              or not set(cited) <= set(ids))):
                _SCRIPTED_EFFECTS.append(
                    "unavailable:" + verifier.ERR_MALFORMED)
                out.append({"status": verifier.STATUS_UNAVAILABLE,
                            "verdict": None, "evidence_ids": [],
                            "reason": "", "error": verifier.ERR_MALFORMED})
                continue
            _SCRIPTED_EFFECTS.append(spec)
            out.append({"status": verifier.STATUS_VERIFIED,
                        "verdict": spec, "evidence_ids": list(cited),
                        "reason": "scripted", "error": None})
        return out

    return _verify_all


def run_layer(case, mode):
    """Run the real books_agent for one case on one layer.

    mode="off"  -> layer 1 (deterministic baseline, verifier skipped).
    mode="local" -> layer 2/3: verifier enabled with verify_all SCRIPTED
    to the case's mock_verifier verdict (zero model calls).
    """

    class _Resp:
        def __init__(self):
            self.source_nodes = _nodes_for(case)

        def __str__(self):
            # M4C multi-claim cases provide the full synthesized answer;
            # single-claim cases use the claim text itself.
            return case.get("answer_text") or case["claim"]

    orig_engine = ba._get_query_engine
    orig_mode = ba.BOOKS_VERIFIER
    orig_verify = verifier.verify_all
    ba.BOOKS_VERIFIER = mode
    ba._get_query_engine = lambda: SimpleNamespace(query=lambda q: _Resp())
    if mode != "off":
        verifier.verify_all = _scripted_verify_all(case)
    _SCRIPTED_EFFECTS.clear()
    try:
        out = books_agent({"question": case.get("question", "?")})
    finally:
        ba._get_query_engine = orig_engine
        ba.BOOKS_VERIFIER = orig_mode
        verifier.verify_all = orig_verify
    effects = list(_SCRIPTED_EFFECTS)
    _SCRIPTED_EFFECTS.clear()

    answer = out["answer"]
    refusal = (answer.startswith(NOT_FOUND_PHRASE)
               and "References:" not in answer)
    if refusal:
        outcome = "refusal"
    elif "Note:" in answer:
        outcome = "accept_disclosed"
    else:
        outcome = "accept"
    return {
        "outcome": outcome,
        "answer": answer,
        "claim_lines": _parse_claim_lines(answer),
        "claim_present": case["claim"] in answer,
        "disclosure": "Note:" in answer,
        "attribution_unavailable": "attribution unavailable" in answer,
        "footer_present": "References:" in answer,
        "unavailable_marker": "semantic verification unavailable" in answer,
        "files_in_answer": {m.lower() for m in _FILE_RE.findall(answer)},
        "scripted_effects": effects,
    }


def _parse_claim_lines(answer):
    """M4C claim-level footer lines -> [(refs_str, supports_text)].

    Returns [] for answer-level (off-mode) footers or refusals, so the
    claim-level mapping check only applies where the per-claim format is
    expected.
    """
    if "References:\n" not in answer:
        return []
    block = answer.split("References:\n", 1)[1]
    lines = []
    for ln in block.strip().splitlines():
        m = re.match(r"^\[(\d+)\] (.*?) \(verifier-supported: (.*)\)$", ln)
        if m:
            lines.append((m.group(2), m.group(3)))
    return lines


def _mapping_ok(case, layer2):
    """M4C: the verified layer's claim->evidence lines must match the
    case's expected mapping exactly (refs per claim, claim order). True
    when the case defines no expected mapping.
    """
    expected = case.get("expected_mapping")
    if not expected:
        return True
    got = layer2["claim_lines"]
    if len(got) != len(expected):
        return False
    for (refs, supports), exp in zip(got, expected):
        if supports != exp["supports"]:
            return False
        if refs.split("; ") != exp["refs"]:
            return False
    return True


def _flags_ok(case, layer2):
    for flag, expected in (case.get("target_flags") or {}).items():
        if bool(layer2.get(flag)) != bool(expected):
            return False
    return True


def _effective_ok(case, layer2):
    """M4C adversarial contract: a case may declare `expected_effective` —
    the exact per-claim verdicts the pipeline must consume AFTER
    per-claim id validation (an id-whitelist rejection must surface as
    unavailable(malformed), never as a pass). Checked against the
    runner's recorded effective verdicts; when undeclared, no check
    (backward compatible with existing cases)."""
    expected = case.get("expected_effective")
    if not expected:
        return True
    return list(layer2.get("scripted_effects") or []) == list(expected)


def _declared_specs(case):
    """Every scripted verdict spec a case declares, from ANY channel —
    case-level mock, per-claim mock, or post-validation expectations —
    so the quality subsets cannot miss a failure declared indirectly."""
    specs = []
    if case.get("mock_verifier"):
        specs.append(case["mock_verifier"])
    specs += list(case.get("mock_verifier_per_claim") or [])
    specs += list(case.get("expected_effective") or [])
    return specs


def _mock_label(case):
    """Scripted-verdict table column (M4C): per-claim aware."""
    per_claim = case.get("mock_verifier_per_claim")
    if per_claim:
        return "+".join(per_claim)
    return case.get("mock_verifier") or "?"


def _render_effects(effects):
    """Label for the EFFECTIVE scripted verdicts the pipeline consumed
    (after per-claim id validation). A single verdict keeps the original
    wording; mixed results are counted explicitly, so an id-whitelist
    rejection on one claim never hides behind a 'confirmed' label."""
    if not effects:
        return "-"
    if len(effects) == 1:
        spec = effects[0]
        if spec == verifier.VERDICT_SUPPORTED:
            return "confirmed"
        if spec.startswith("unavailable:"):
            return f"unavailable({spec.split(':', 1)[1]})"
        return "removed"
    conf = sum(1 for s in effects if s == verifier.VERDICT_SUPPORTED)
    unavs = {}
    for s in effects:
        if s.startswith("unavailable:"):
            unavs[s] = unavs.get(s, 0) + 1
    rem = len(effects) - conf - sum(unavs.values())
    parts = []
    if conf:
        parts.append(f"{conf} confirmed")
    if rem:
        parts.append(f"{rem} removed")
    for spec, n in unavs.items():
        parts.append(f"{n} unavailable({spec.split(':', 1)[1]})")
    return "mixed: " + ", ".join(parts)


def _rate(num, den):
    if not den:
        return "0/0 (n/a)"
    return f"{num}/{den} ({100.0 * num / den:.1f}%)"


def quality_report(rows):
    """§5 deterministic quality report: raw counts + denominators + %.

    `rows` = [(case, layer1, layer2), ...]. Every metric is computed on
    BOTH layers so the deterministic baseline and the verified pipeline
    are compared directly: layer 1 = BOOKS_VERIFIER=off (pure gate),
    layer 2 = integrated outcome with the scripted verifier verdict.
    """
    lines = []
    lines.append("QUALITY REPORT (deterministic; constructed truth, not model quality)")
    lines.append("Each cell: raw/total (percent). off = verifier off (M3 baseline), "
                 "verified = scripted-verifier pipeline (integrated layer).")
    lines.append(f"{'metric':46} {'off':>14} {'verified':>14}")
    lines.append("-" * 78)

    def cell(layer_idx, pred_subset, want):
        """rate over cases in pred_subset where accepted==want (layer)."""
        sub = [r for r in rows if pred_subset(r[0])]
        n = 0
        for r in sub:
            layer = r[layer_idx]
            acc = layer["claim_present"]
            out = layer["outcome"]
            if want == "keep" and acc:
                n += 1
            elif want == "drop" and not acc:
                n += 1
            elif (want == "fail_closed_refusal" and out == "refusal"
                  and layer["unavailable_marker"] and not acc):
                n += 1
            elif (want == "not_success"
                  and out not in ("accept", "accept_disclosed")):
                n += 1
        return _rate(n, len(sub))

    def cite(layer_idx):
        checks = passed = 0
        for c, l1, l2 in rows:
            layer = (l1, l2)[layer_idx]
            checks += 1
            # (a) a refusal never carries a references footer.
            if layer["outcome"] == "refusal" and layer["footer_present"]:
                continue
            passed += 1
            checks += 1
            # (b) every filename in the answer came from the case's evidence.
            evidence_files = {e.get("file_name", "").lower()
                              for e in c["evidence"] if e.get("file_name")}
            if layer["files_in_answer"] <= evidence_files:
                passed += 1
        return _rate(passed, checks)

    unsup = lambda c: c["expected_verdict"] == "insufficient_evidence"
    supp_ok = lambda c: (c["expected_verdict"] == "supported"
                         and ("accept" in c["target_outcomes"]
                              or "accept_disclosed" in c["target_outcomes"]))
    contra = lambda c: c["expected_verdict"] == "contradicted"
    para = lambda c: (c["category"] == "paraphrase"
                      and c["expected_verdict"] == "supported")
    mal = lambda c: ("unavailable:malformed_output"
                     in _declared_specs(c))
    unav = lambda c: any(s.startswith("unavailable:")
                         for s in _declared_specs(c))

    def row(label, subset, want):
        lines.append(f"{label:46} {cell(1, subset, want):>14} "
                     f"{cell(2, subset, want):>14}")

    row("unsupported claims accepted (false accept)", unsup, "keep")
    row("supported claims falsely rejected", supp_ok, "drop")
    row("contradictions detected (claim removed)", contra, "drop")
    for cat, label in (("numeric", "numeric"),
                       ("relation", "relational"),
                       ("negation", "negation")):
        row(f"  ... {label}", lambda c, k=cat: contra(c)
            and c["category"] == k, "drop")
    row("  ... other", lambda c: contra(c)
        and c["category"] not in ("numeric", "relation", "negation"),
        "drop")
    row("paraphrases accepted", para, "keep")
    row("malformed output -> fail-closed refusal", mal,
        "fail_closed_refusal")
    row("timeout/unavailable -> fail-closed refusal", unav,
        "fail_closed_refusal")
    row("unavailable NEVER shown as successful verdict", unav,
        "not_success")
    lines.append(f"{'citation integrity checks passed (2/case)':46} "
                 f"{cite(0):>14} {cite(1):>14}")

    # M4C claim-level attribution: exact claim->evidence mapping on
    # the verified layer, for the cases that define one.
    mapping_cases = [r for r in rows if r[0].get("expected_mapping")]
    n_map_ok = sum(1 for r in mapping_cases if _mapping_ok(r[0], r[2]))
    lines.append(f"claim-level mapping exact (verified layer): "
                 f"{_rate(n_map_ok, len(mapping_cases))}")

    # Claims processed / excluded by the budget (§5 accounting).
    lines.append(f"claims through mocked verifier layer      : "
                 f"{_SCRIPTED_CLAIMS} processed, 0 budget-excluded "
                 f"(the mock consumes no budget by design)")
    return "\n".join(lines)


def budget_probe():
    """Deterministic §3/§5 budget enforcement probe (no threads, no sleeps).

    Returns (probeA, probeB) tallies of (processed, cap, deadline) over 10
    claims: A with the normal 30s deadline, B with zero remaining budget.
    """
    orig_call = verifier._call_llm
    payload = json.dumps({"verdict": verifier.VERDICT_SUPPORTED,
                          "evidence_ids": ["e0"], "reason": "x"})

    def _instant(prompt, timeout):
        return "ok", payload

    verifier._call_llm = _instant
    try:
        items = [{"claim": f"claim {i}",
                  "evidence": [{"id": "e0", "text": "ev"}]}
                 for i in range(10)]
        res_a = verifier.verify_all(items)
        orig_deadline = verifier.BOOKS_VERIFIER_TOTAL_DEADLINE
        verifier.BOOKS_VERIFIER_TOTAL_DEADLINE = 0.0
        try:
            res_b = verifier.verify_all(items)
        finally:
            verifier.BOOKS_VERIFIER_TOTAL_DEADLINE = orig_deadline
    finally:
        verifier._call_llm = orig_call

    def tally(res):
        processed = sum(1 for r in res if r["status"] == verifier.STATUS_VERIFIED)
        cap = sum(1 for r in res if r.get("error") == verifier.ERR_CAP)
        deadline = sum(1 for r in res if r.get("error") == verifier.ERR_DEADLINE)
        return processed, cap, deadline

    return tally(res_a), tally(res_b)


def main(argv) -> int:
    check = "--check" in argv
    record = "--record" in argv
    if BOOKS_VERIFIER != "off":
        print("ERROR: run the benchmark with BOOKS_VERIFIER=off "
              f"(current: {BOOKS_VERIFIER})")
        return 2
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    print(f"{len(cases)} case(s) from {CASES_PATH.name}\n")
    header = (f"{'case':34} {'baseline':10} {'scripted':26} {'verifier':28} "
              f"{'integrated':17} {'target':24} status")
    print(header)
    print("-" * len(header))

    rows, changed, gaps, unrecorded = [], [], [], []
    baseline_gaps = 0
    for case in cases:
        layer1 = run_layer(case, "off")          # deterministic M3 baseline
        layer2 = run_layer(case, "local")        # scripted verifier + integrated
        recorded = case.get("recorded_baseline")
        target = case["target_outcomes"]

        # Layer 1 status vs recorded baseline (drift gate).
        if recorded is None:
            base_status = "UNRECORDED"
            unrecorded.append(case["id"])
        elif recorded != layer1["outcome"]:
            base_status = "CHANGED"
            changed.append((case["id"], recorded, layer1["outcome"]))
        else:
            base_status = "MATCH"

        # Layer 3 status vs design target (+ flags + M4C mapping +
        # post-id-validation effective verdicts).
        integrated_ok = (layer2["outcome"] in target
                         and _flags_ok(case, layer2)
                         and _mapping_ok(case, layer2)
                         and _effective_ok(case, layer2))
        int_status = "MATCH" if integrated_ok else "GAP"
        if not integrated_ok:
            gaps.append((case["id"], layer2["outcome"], target))

        # Informational: layer-1 misses the design target (known gap surface).
        bl_gap = layer1["outcome"] not in target
        if bl_gap:
            baseline_gaps += 1

        # Layer 2 rendering: what the scripted verdict did in the pipeline
        # (effective per-claim verdicts after id validation — M4C).
        mock = _mock_label(case)
        if layer1["outcome"] == "refusal":
            effect = "gate-refusal (not reached)"
        else:
            effect = _render_effects(layer2.get("scripted_effects") or [])

        status = ("GAP" if int_status == "GAP"
                  else "CHANGED" if base_status == "CHANGED"
                  else "UNRECORDED" if base_status == "UNRECORDED"
                  else "MATCH")
        if bl_gap and status == "MATCH":
            status = "MATCH (baseline-gap*)"

        print(f"{case['id']:34} {layer1['outcome']:10} {mock:26} "
              f"{effect:28} {layer2['outcome']:17} {','.join(target):24} "
              f"{status}")
        rows.append((case, layer1, layer2))

    print("\n* baseline-gap: the DETERMINISTIC layer alone misses the design "
          "target (known defect surface;\nclosed by enabling "
          "BOOKS_VERIFIER=local, not by changing the gate). "
          f"Count: {baseline_gaps}/{len(cases)} case(s).")

    print()
    print(quality_report(rows))

    print()
    a, b = budget_probe()
    print("BUDGET PROBE (deterministic, scripted LLM, 10 claims per probe)")
    print(f"  normal 30s deadline : {a[0]} processed, {a[1]} cap-excluded, "
          f"{a[2]} deadline-excluded")
    print(f"  zero remaining budget: {b[0]} processed, {b[1]} cap-excluded, "
          f"{b[2]} deadline-excluded")

    print()
    if record:
        for case in cases:
            if case.get("recorded_baseline") is None:
                print(f"  {case['id']} -> {run_layer(case, 'off')['outcome']}")
        return 0
    if changed:
        print("CHANGED baselines (implementation moved — update records only "
              "deliberately):")
        for cid, old, new in changed:
            print(f"  {cid}: recorded={old} measured={new}")
    if gaps:
        print("INTEGRATED-PIPELINE GAP(s) (deterministic verdict handling "
              "misses the design target):")
        for cid, got, target in gaps:
            print(f"  {cid}: integrated={got} target={target}")
    if unrecorded:
        print(f"{len(unrecorded)} case(s) with no recorded baseline yet "
              "(run with --record, then verify each value by review).")
    if not (changed or gaps):
        print("all cases: baseline MATCH, integrated target MATCH")
    if check and (changed or gaps):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
