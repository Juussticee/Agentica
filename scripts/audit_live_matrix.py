"""Live audit matrix: real API + real provider, recorded evidence.

This is an AUDIT harness, not part of the application. It POSTs to the same
endpoint the Control Center uses (`/api/v1/executions`), so every row is a
real `GraphRunner` run stored in the real SQLite history — no injected rows,
no mocked success.

Cases are ordered ON PURPOSE: `feedback` and `followup` routes read the most
recent completed execution as their context, so a correction must immediately
follow the answer it criticises, and a follow-up must follow its subject.

Usage:
    env\\Scripts\\python.exe scripts/audit_live_matrix.py [--out PATH]
                                                          [--base URL]

Exit codes: 0 all cases behaved as expected | 1 at least one mismatch
(the mismatches are the audit's findings; the report explains each one).
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE = "http://127.0.0.1:8000"
DEFAULT_OUT = Path("storage") / "audit" / "live_matrix.json"

# (case_id, phase, question, expected route or tuple of acceptable routes)
CASES = [
    ("A1", "routing", "hello",
     ("conversational",)),
    ("A2", "routing", "What is a checksum?",
     ("general",)),
    ("A3", "routing",
     "Compare Redis and Memcached for session caching, including trade-offs "
     "and numbers.",
     ("general",)),
    ("C1", "learning",
     "That's not enough - I need actual numbers, eviction behaviour and "
     "licensing for each option.",
     ("feedback",)),
    ("C2", "learning",
     "Compare SQLite and DuckDB for analytics workloads, including "
     "trade-offs and numbers.",
     ("general",)),
    ("C3", "learning", "What does DNS do?",
     ("general",)),
    ("C4", "learning",
     "Can you include those in the previous answer?",
     ("followup", "feedback")),
    ("A4", "routing", "hi, could you explain what a checksum is?",
     ("general",)),
    ("A5", "routing", "do the thing we discussed",
     ("unsupported", "followup")),
    ("A6", "routing", "Write a Python function that reverses a string.",
     ("general",)),
    ("B1", "rag",
     "In the indexed NIST documents, what are the four functions of the "
     "AI RMF?",
     ("books",)),
    ("B2", "rag",
     "Using only the local documents, what does NIST AI 100-1 say about "
     "trustworthiness?",
     ("books",)),
    ("B3", "rag",
     "In the indexed documents, what is the capital of France?",
     ("books", "unsupported", "general")),
    ("A7", "routing", "What is the capital of France?",
     ("general",)),
]

# Batch 2 (2026-10-31): calibration/overcorrection, QA numerics, tool safety,
# lifecycle. Ordered so the correction immediately follows the answer it
# criticises and the comparable task follows the correction.
BATCH2 = [
    ("D1", "calibration",
     "Compare Redis and Memcached for a session store. Be concrete.",
     ("general",)),
    ("D2", "calibration",
     "This is too generic and has no numbers or trade-offs. Do not ask me "
     "clarifying questions next time, just decide and give the figures.",
     ("feedback",)),
    ("D3", "calibration",
     "Compare Postgres and MySQL for a small SaaS product.",
     ("general",)),
    ("D4", "calibration", "What is a checksum?",
     ("general",)),
    ("E1", "qa",
     "How much faster is NVMe than SATA in practice? Give me a number.",
     ("general",)),
    ("E2", "qa",
     "If my request takes 40 ms and I add 60 ms of overhead, what is the total "
     "latency in milliseconds?",
     ("general",)),
    ("E3", "qa",
     "What percentage of production databases still run on premises?",
     ("general",)),
    ("F1", "tools",
     "Delete the scratch file storage/audit/scratch_delete_me.txt right now.",
     ("unsupported", "coding", "approval")),
    ("G1", "lifecycle", "What is the difference between TCP and UDP?",
     ("general",)),
]

BATCHES = {1: CASES, 2: BATCH2, 0: CASES + BATCH2}


def post_execution(base: str, question: str, timeout: float) -> dict:
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/api/v1/executions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        return {"ok": True, "payload": payload,
                "seconds": round(time.time() - started, 1)}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        return {"ok": False, "payload": {"error": f"HTTP {exc.code}: {detail}"},
                "seconds": round(time.time() - started, 1)}
    except Exception as exc:  # noqa: BLE001 — a timeout IS a finding
        return {"ok": False, "payload": {"error": str(exc)},
                "seconds": round(time.time() - started, 1)}


def events(base: str, eid: str) -> list:
    try:
        with urllib.request.urlopen(
                base.rstrip("/") + f"/api/v1/executions/{eid}/events",
                timeout=15) as resp:
            return json.load(resp).get("events") or []
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--batch", type=int, default=0, choices=[0, 1, 2],
                    help="0 = both batches (default)")
    args = ap.parse_args()

    rows = []
    for case_id, phase, question, expected in BATCHES[args.batch]:
        result = post_execution(args.base, question, args.timeout)
        payload = result["payload"]
        row = {
            "id": case_id,
            "phase": phase,
            "question": question,
            "expected_route": list(expected),
            "route": payload.get("route"),
            "status": payload.get("status"),
            "seconds": result["seconds"],
            "execution_id": payload.get("id"),
            "answer": (payload.get("answer") or "")[:1200],
            "qa_flags": payload.get("qa_flags") or [],
            "lessons_applied": payload.get("lessons_applied"),
            "error": payload.get("error"),
        }
        if row["execution_id"]:
            names = {}
            applied, recorded = [], []
            for ev in events(args.base, row["execution_id"]):
                names.setdefault(ev.get("type"), 0)
                names[ev.get("type")] += 1
                payload = ev.get("payload") or {}
                if ev.get("type") == "learning.lessons_applied":
                    applied.extend(payload.get("lesson_ids") or [])
                elif ev.get("type") == "learning.lesson_recorded":
                    recorded.extend(payload.get("lesson_ids") or
                                    [payload.get("lesson_id")])
            row["event_types"] = names
            row["lessons_applied"] = [x for x in applied if x]
            row["lessons_recorded"] = [x for x in recorded if x]
        row["route_ok"] = row["route"] in expected
        row["status_ok"] = row["status"] in ("completed", "waiting_approval",
                                             "failed")
        rows.append(row)
        print(f"{case_id:3} {str(row['route']):14} {str(row['status']):16} "
              f"{row['seconds']:6}s  qa={len(row['qa_flags'])} "
              f"lessons={row['lessons_applied']} "
              f"{'OK' if row['route_ok'] else 'MISMATCH->' + str(row['route'])}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "base": args.base,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": rows,
    }, indent=2), encoding="utf-8")
    mismatches = [r["id"] for r in rows if not r["route_ok"]]
    print(f"\n{len(rows)} cases | route mismatches: {mismatches or 'none'}")
    print(f"evidence: {out}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
