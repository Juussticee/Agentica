"""M4B-H optional local-model experiment (NOT part of the regression suite).

Runs the FULL expanded deterministic benchmark set
(benchmarks/verification_cases.json) through the REAL installed local model
(qwen3:0.6b via Ollama — no downloads, no paid quota) using the same bounded
path production uses (app/verifier.py: bounded evidence, daemon-thread
caller bound, transport timeout, in-flight semaphore, strict parser,
fail-closed mapping). Per case it records:

  * expected verdict (constructed truth from the benchmark)
  * actual verdict / parse failure / timeout / busy
  * per-claim latency (cold first call vs warm rest)
  * whether the verdict would change the deterministic baseline outcome
  * false acceptance and false rejection counts

After the run it probes the runtime again and reports whether timed-out
requests kept running (follow-up probe latency) and whether any verifier
threads remain alive (resource accumulation check).

The timeout used is the PRODUCTION bound (BOOKS_VERIFIER_CLAIM_TIMEOUT,
10s) — it is deliberately NOT loosened to improve the success rate. Any
proposal to change a limit must come from these measurements, not from
this script.

This is an EVALUATION, not proof of production readiness: small samples
are not statistically conclusive and live-model correctness is never
asserted by the deterministic suites. Run manually:

    python scripts/eval_verifier_local.py

Requires the local Ollama runtime; reports a clean error and exits 2 if it
is unavailable (never falls back to a paid provider).
"""

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.verifier as verifier  # noqa: E402
from app.config import (  # noqa: E402
    BOOKS_VERIFIER_CLAIM_TIMEOUT,
    BOOKS_VERIFIER_MAX_INFLIGHT,
    OLLAMA_BASE_URL,
)

CASES_PATH = PROJECT_ROOT / "benchmarks" / "verification_cases.json"

NOT_SUPPORTED = ("contradicted", "insufficient_evidence", "uncertain")


def _ps_loaded():
    """Models currently loaded in the Ollama runtime (None if unreachable)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/ps", timeout=5) as resp:
            return [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except Exception:  # noqa: BLE001
        return None


def _run_one(case) -> dict:
    """One claim through the production-bounded verifier path."""
    evidence = [{"id": f"e{i}", "text": item["text"]}
                for i, item in enumerate(case["evidence"])]
    prompt = verifier._build_prompt(
        case["claim"], verifier._bounded_evidence(evidence))
    started = time.monotonic()
    call_status, raw = verifier._call_llm(prompt, BOOKS_VERIFIER_CLAIM_TIMEOUT)
    latency = time.monotonic() - started
    out = {"call_status": call_status, "raw": raw, "latency": latency,
           "verdict": None, "parse_ok": False}
    if call_status == "ok":
        parsed = verifier._parse_output(raw, [e["id"] for e in evidence])
        if parsed is None:
            out["call_status"] = "malformed"
        else:
            out["parse_ok"] = True
            out["verdict"] = parsed["verdict"]
    return out


def main() -> int:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5):
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"LOCAL RUNTIME UNAVAILABLE ({type(exc).__name__}): {exc}")
        print("Reported as unavailable; no paid model substituted.")
        return 2

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    print(f"local-model experiment | model path via get_llm(timeout=ollama) "
          f"| claim timeout = {BOOKS_VERIFIER_CLAIM_TIMEOUT}s (production) "
          f"| in-flight budget = {BOOKS_VERIFIER_MAX_INFLIGHT}")
    print(f"loaded before run: {_ps_loaded()}\n")

    rows, started = [], time.monotonic()
    for idx, case in enumerate(cases):
        res = _run_one(case)
        expected = case["expected_verdict"]
        verdict = res["verdict"]
        status = res["call_status"]
        correct = status == "ok" and verdict == expected
        false_accept = (status == "ok" and verdict == "supported"
                        and expected in NOT_SUPPORTED)
        false_reject = (status == "ok" and verdict in NOT_SUPPORTED
                        and expected == "supported")
        would_change = (case.get("recorded_baseline") == "accept"
                        and verdict != "supported")
        rows.append({**res, "id": case["id"], "expected": expected,
                     "correct": correct, "false_accept": false_accept,
                     "false_reject": false_reject,
                     "would_change": would_change})
        warm = "cold" if idx == 0 else "warm"
        print(f"{case['id']:36} exp={expected:20} got={str(verdict):20} "
              f"{status:9} {warm:4} {res['latency']:6.2f}s "
              f"correct={correct} false_accept={false_accept} "
              f"false_reject={false_reject} would_change={would_change}")
    elapsed = time.monotonic() - started

    ok = [r for r in rows if r["call_status"] == "ok"]
    timeouts = [r for r in rows if r["call_status"] == "timeout"]
    busy = [r for r in rows if r["call_status"] == "busy"]
    malformed = [r for r in rows if r["call_status"] == "malformed"]
    errors = [r for r in rows if r["call_status"] == "error"]
    lat = [r["latency"] for r in rows]
    warm_lat = [r["latency"] for r in rows[1:]] or lat

    print("\n=== SUMMARY ===")
    print(f"samples              : {len(rows)} ({CASES_PATH.name})")
    print(f"correct verdicts     : {sum(1 for r in rows if r['correct'])}/{len(rows)}")
    print(f"false acceptances    : {sum(1 for r in rows if r['false_accept'])}"
          f"/{sum(1 for r in rows if r['expected'] in NOT_SUPPORTED)}"
          f" (expected not-supported cases)")
    print(f"false rejections     : {sum(1 for r in rows if r['false_reject'])}"
          f"/{sum(1 for r in rows if r['expected'] == 'supported')}"
          f" (expected supported cases)")
    print(f"parse failures       : {len(malformed)} (of {len(ok)} completed calls)")
    print(f"timeouts             : {len(timeouts)} "
          f"(> {BOOKS_VERIFIER_CLAIM_TIMEOUT}s production bound)")
    print(f"busy (budget)        : {len(busy)}   provider errors: {len(errors)}")
    if lat:
        print(f"latency cold first   : {lat[0]:.2f}s (model load + first generate)")
        print(f"latency warm         : min {min(warm_lat):.2f}s / "
              f"median {statistics.median(warm_lat):.2f}s / "
              f"max {max(warm_lat):.2f}s "
              f"(completed calls: max "
              f"{max((r['latency'] for r in ok), default=0):.2f}s)")
    print(f"total elapsed        : {elapsed:.1f}s")
    print(f"would change baseline: {sum(1 for r in rows if r['would_change'])} "
          f"claim(s) accepted by the deterministic gate that the model "
          f"would not confirm")
    print(f"loaded after run     : {_ps_loaded()}")

    # Resource accounting: a timed-out worker can linger only until the
    # transport bound (timeout + 5s). Give stragglers that window, then
    # count survivors and probe the runtime.
    if timeouts:
        print(f"waiting {verifier._TRANSPORT_MARGIN:.0f}s transport window "
              "for timed-out workers to die...")
        time.sleep(verifier._TRANSPORT_MARGIN + 0.5)
    alive = [t for t in __import__("threading").enumerate()
             if t.name == "books-verifier" and t.is_alive()]
    print(f"live verifier threads after run: {len(alive)} "
          f"(budget {BOOKS_VERIFIER_MAX_INFLIGHT}; accumulation check)")
    probe_started = time.monotonic()
    probe = _run_one({"claim": "Propolis seals small gaps.",
                      "evidence": [{"text": "Propolis seals small gaps."}]})
    alive_after = [t for t in __import__("threading").enumerate()
                   if t.name == "books-verifier" and t.is_alive()]
    print(f"post-run probe call   : {probe['call_status']} "
          f"({time.monotonic() - probe_started:.2f}s; the model's own warm "
          f"median is the reference — extra queueing would show as "
          f"~N x 10s steps) live threads after probe: {len(alive_after)}")
    print("Evaluation only — small sample, no production readiness claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
