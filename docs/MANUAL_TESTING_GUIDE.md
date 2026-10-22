# Agentica — Manual Testing Guide (prepared 2026-10-18)

For the tester: you do not need to read any source code. Follow the steps exactly.
Copy-paste the questions verbatim — wording matters.

**Ground rules for the whole session**

- Submit **one question at a time** and wait for it to reach a final status before
  the next one. (This is what makes the log correlation in Test G unambiguous.)
- Do **not** click **"Run sample execution"** on the Overview page — it submits its
  own question and would interleave with your test runs.
- Do **not** restart Ollama, the backend, or the frontend, and do **not** run any
  ingestion, unless a stop condition in Test J tells you to.
- Open the app at **http://localhost:3333** (not `http://127.0.0.1:3333` — the
  preview binds to the IPv6 loopback and `127.0.0.1:3333` will fail).

---

## Test A — Application startup

**Purpose:** confirm the page loads and the backend is healthy.

1. Open `http://localhost:3333` in your browser.
2. The **Dashboard** must appear with the heading "Dashboard" and the sidebar
   (Overview / New Execution / Executions / Tasks & Sequence / Services / Agents /
   Settings).
3. Under **System health** you must see: `Agent system: Working` and
   `Backend version: 0.1.0`.
4. Under **AI connection** you must see: `Configured — Ollama
   (http://localhost:11434), model qwen3:0.6b`.
5. Under **Recent executions** the history table must list previous runs
   (the newest should be `#c2a52bcb3b1c`).

**PASS:** all of the above are visible and no error card is shown.

**FAIL:** page does not load at all, `Agent system` is not "Working", the AI
connection says anything other than "Configured … qwen3:0.6b", or the history
table is empty.

**Evidence:** screenshot of the Dashboard.

**Continue?** Only if the failure is a page-load problem, go to the startup
checklist at the end. If the backend reports unhealthy, **stop and report** —
do not submit any question.

---

## Test B — Answerable books question

**Purpose:** a normal, grounded answer with correct citation.

**Navigation:** sidebar → **New Execution** → paste into the **Task** box:

```
Which bee is mentioned in the Urban Beekeeping Guide?
```

Click **Start execution**. You are taken to the execution page automatically.

**Source verification (already done during preparation, from the document
itself):** the indexed guide (`urban_beekeeping_field_guide.pdf`, 1 page) talks
about **worker bees** (up to 60,000 in a midsummer hive), **one queen**, and
**a few hundred drones**. It names **no bee species** — do not expect
"Apis mellifera" or similar.

**PASS:**

- Status reaches **Completed**.
- The answer mentions worker bees (optionally the queen and/or drones) and is
  supported by the guide.
- The answer ends with a `References:` footer naming
  `urban_beekeeping_field_guide.pdf` (a page number, e.g. page 1, is expected
  here because the PDF has page metadata).

**FAIL:**

- The answer says the information was not found (this question IS answerable).
- The answer names a bee species or fact not in the guide.
- The answer cites a document other than the beekeeping guide.
- No References footer at all.

**Evidence:** screenshot of the Result card including the footer; record the
execution ID from the page heading (`Execution #<id>`).

**Continue after failure?** No — record and go to Test J, stop condition 3.

---

## Test C — Out-of-corpus question

**Purpose:** safe refusal instead of financial fabrication.

**Navigation:** New Execution → paste:

```
What does the Q3 balance sheet say about server fleet costs?
```

**Source verification (already done):** the indexed corpus contains only the
beekeeping guide and the orbital-mechanics document. Neither contains "balance
sheet", "Q3", "fleet", or "server" — the information is genuinely absent.

**PASS:**

- Status **Completed**.
- The answer is exactly (or begins with) the mandated refusal:
  **"The requested information was not found in the books."**
  (A similarity-score note may follow in parentheses.)
- **No** `References:` footer, **no** financial numbers, **no** invented facts
  or citations.

**FAIL:** any number, figure, "Q3 balance sheet" content, citation, or a
different wording that still asserts financial information.

**Evidence:** screenshot of the full answer.

**Continue after failure?** No — this is a fabrication defect. Record evidence
and stop (Test J, stop condition 4).

---

## Test D — Former delta-v self-refusal failure (PRIMARY REGRESSION TEST)

**Purpose:** the exact live question that used to self-refuse must now produce a
useful, evidence-supported answer.

**Navigation:** New Execution → paste:

```
According to orbital_kinetics_principles.docx, what does delta-v mean, and how is it used in orbital mechanics?
```

**Source verification (already done):** the indexed document
(`orbital_kinetics_principles.docx`) states that delta-v relates to exhaust
velocity and mass ratio through the Tsiolkovsky rocket equation (derived 1903),
and that it drives maneuvers — e.g. a Hohmann transfer from low Earth orbit to
geostationary transfer orbit needs **roughly 2.4 km/s**, circularizing adds
**~1.5 km/s**.

**PASS:**

- Status **Completed** and the answer is a **useful explanation** (not the
  not-found refusal): delta-v defined via the rocket-equation / mass-ratio
  relationship and its use in maneuvers, consistent with the facts above.
- `References:` footer names `orbital_kinetics_principles.docx`.
- Claims match the source: the 1903 year, the ~2.4 km/s and ~1.5 km/s figures
  (if figures appear at all) — no other numbers.

**ACCEPTABLE VARIANT:** if the model first declined and recovery kicked in, the
answer can still be correct — Test G will show the recovery markers, which is
**expected behavior**, not a failure.

**FAIL:**

- The answer is still the not-found refusal (this was the original bug).
- The answer contains figures/claims not in the document.
- The footer cites a different document or invents page numbers.

**Evidence:** screenshot of the full Result card; note the execution ID.

**Continue after failure?** No — this is THE regression. Record everything
(screenshot, ID, log excerpt via Test G) and stop.

---

## Test E — Hohmann transfer question

**Purpose:** grounded answer on a specific orbital-mechanics fact, or an honest
refusal — never unsupported detail.

**Navigation:** New Execution → paste:

```
How is delta-v used when planning a Hohmann transfer between circular orbits?
```

**Source verification (already done):** the document defines the Hohmann
transfer as "the minimum-energy two-impulse maneuver connecting two circular,
coplanar orbits" and gives the ~2.4 km/s LEO→GTO figure.

**PASS (either of these):**

1. A grounded answer describing a two-impulse / minimum-energy transfer between
   circular orbits, possibly with the ~2.4 km/s figure, with a
   `References:` footer naming `orbital_kinetics_principles.docx`; **or**
2. The safe not-found refusal (if you judge the evidence insufficient for the
   question as asked).

**FAIL:** specific transfer details that are NOT in the document (other
numbers of burns with invented values, other Δv figures, plane-change claims,
etc.), or citations to unrelated documents.

**Evidence:** screenshot including the footer.

**Continue after failure?** No — record and stop (fabrication).

---

## Test F — Citation correctness (run after B, D, E)

**Purpose:** verify citations are real, relevant, and not fabricated.

For each completed books answer (Tests B, D, E), check:

1. **Right document:** bee answers cite `urban_beekeeping_field_guide.pdf`;
   orbital answers cite `orbital_kinetics_principles.docx`.
2. **Real sources:** every cited file must be one of those two names — no other
   filenames, no chapter/section names that don't exist.
3. **No fabricated page numbers on the .docx:** the orbital document has NO
   page metadata, so its footer must be **file-only**. A page number attached to
   `orbital_kinetics_principles.docx` is a fabrication.
   (The PDF may legitimately show a page number — it has exactly 1 page.)
4. **Refusals cite nothing:** Tests C (and any refusal) must have NO
   `References:` footer.
5. **No unrelated documents:** a bee question must never cite the orbital
   document or vice versa.

**PASS:** all five hold for every answer. **FAIL:** any violation.

**Evidence:** screenshots of each footer. **Continue after failure?** No.

---

## Test G — Recovery retry-limit inspection (log)

**Purpose:** confirm at most ONE recovery generation per execution, and see the
recovery markers for Test D.

**Log location (current backend):**

```
C:\Users\hashe\Documents\Agentic_System\.diag\backend_20261018_m5loaded.log
```

(The pre-restart log is preserved as `.diag\backend_pre_m5restart_preserved.log`
for comparison; the OLD stale-code log is `.diag\backend_20261018_133803.log`.)

**How to read it (PowerShell):**

```powershell
cd C:\Users\hashe\Documents\Agentic_System
Select-String -Path .diag\backend_20261018_m5loaded.log -Pattern "self_refusal_recovery|not-found path|grounding refusal"
```

**Marker meanings (mutually exclusive outcomes after an attempt):**

| Marker | Meaning |
|---|---|
| `self_refusal_recovery_attempted (score=…, nodes=…)` | One regeneration was started (eligible pure self-refusal). |
| `self_refusal_recovery_succeeded (chars=…)` | Regeneration produced a non-refusal answer; it then went through the unchanged gates. |
| `self_refusal_recovery_rejected (reason=second_refusal, score=…)` | Regeneration refused again → immediate abstain. |
| `self_refusal_recovery_rejected (reason=empty_or_error, score=…)` | Regeneration failed/empty → abstain. |
| `not-found path (…s, score=…)` | Final refusal path (also the pre-recovery behavior). |
| `grounding refusal (…s, score=…)` | Gates found nothing substantiable → refusal. |

**Correlating markers with your execution:**

1. Find the marker lines emitted while your question ran. In the log, uvicorn
   writes the marker lines **during** request handling, and the access line
   `POST /api/v1/executions … 201 Created` **immediately after** them; the UI
   then polls `GET /api/v1/executions/<your-id>` right after that. Because you
   submit one question at a time, the marker block directly above your
   execution's POST/GET lines belongs to your run.
2. **Retry bound check:** count `attempted` markers for your execution — there
   must be **at most ONE**. If you ever see two `attempted` markers in a row
   (or an `attempted` followed by a second `attempted` without a terminal
   `succeeded`/`rejected` between them), that is a violation.
3. For Test D: you want `attempted` → `succeeded` (typical good path), or
   `attempted` → `rejected` followed by the safe refusal (also correct, bounded
   behavior). `not-found path` with **no** `attempted` line at all on the
   delta-v question means the recovery did not trigger — record it as a finding.

**Precise observability gap (honest limitation):** the marker lines contain
**no execution ID and no timestamp** (they are plain stdout prints), and the
uvicorn access lines also lack timestamps. Correlation therefore relies on
submission ordering — which is why the one-question-at-a-time rule matters. No
extra logging was added for this, deliberately.

**FAIL:** more than one `attempted` for a single execution.

---

## Test H — Normal-answer regression (no recovery expected)

**Purpose:** an easy, clearly answerable question must pass WITHOUT any
recovery attempt.

**Navigation:** New Execution → paste:

```
What is the width of the bee space that bees keep free of comb and propolis?
```

**Source verification (already done):** the guide states the bee space is
"a corridor of 6 to 9 millimeters" (page 1).

**PASS:**

- Status **Completed**; answer contains **6 to 9 millimeters**.
- `References:` footer cites `urban_beekeeping_field_guide.pdf` (page 1).
- Test G shows **NO** `self_refusal_recovery_*` markers for this execution.

**FAIL:** refusal, wrong value, or any recovery marker in the log for this run.

**Evidence:** screenshot + the log query from Test G run for this execution's
time window.

**Continue after failure?** Record first; continue to Test I, but if the answer
was fabricated, stop (Test J, condition 4).

---

## Test I — Evidence collection (record for EVERY test)

For each test (A–H), record:

1. **Execution ID** — from the page heading `Execution #<id>` or the
   `#/executions/<id>` URL.
2. **Question** (copy-paste confirmation).
3. **Final answer** (full text, screenshot).
4. **References/citations** (exact footer text).
5. **Route/agent shown** — the "What happened" timeline labels the step
   (books questions show **"Internal Knowledge"**; also note "Understanding the
   request" = supervisor, "Result Check" = QA).
6. **Recovery markers** from the log (Test G query), with `attempted` count.
7. **Duration + final status** — status badge (Completed/Failed) and the
   timeline timestamps (start → completion).
8. **Screenshot** of the Result card; **log excerpt** if anything failed.

**Results template**

| Test | Exec ID | Question (abbrev.) | Status | Answer OK? (Y/N) | Footer cites | Route (timeline) | Recovery markers (attempted/succeeded/rejected) | Duration | Screenshot? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| A | — | startup | | | — | — | — | | | |
| B | | Which bee… | | | | | | | | |
| C | | Q3 balance sheet… | | | | | | | | |
| D | | delta-v mean… | | | | | | | | |
| E | | Hohmann… | | | | | | | | |
| F | (uses B/D/E) | footer audit | | | | | | | | |
| G | | log audit | | | | | | | | |
| H | | bee space width… | | | | | | | | |

---

## Test J — Stop conditions

**Stop immediately (do not continue the session) if:**

1. **The page does not load.** Do not restart anything yet — run the startup
   checklist below once; if it still fails, report (browser error text +
   screenshot).
2. **The backend is unhealthy** (`Agent system` not "Working", or
   `GET http://127.0.0.1:8000/api/v1/health` not `status: ok`). Do not submit
   questions. Capture the health JSON and the last ~50 lines of the current
   backend log.
3. **An answerable question (B or H) returns the not-found response.** Record
   ID + screenshot + the log lines around it. Do not retry the question.
4. **The system invents an answer** (Test C content, or figures not in the
   sources). Record everything. Do not retry, do not ingest, do not repair.
5. **Citations are wrong** (wrong document, fabricated page number on the
   .docx, footer on a refusal). Record and stop.
6. **More than one recovery attempt** for a single execution. Record the exact
   log lines (with surrounding context) and stop.
7. **ChromaDB integrity observation differs from baseline.** Run
   `powershell -ExecutionPolicy Bypass -File .diag\chroma_hash.ps1` before and
   after your session and compare. **Baseline:** `chroma.sqlite3` size
   `245760`, sha256 starts `ffded36120a03dc2…`, mtime `2026-10-18T10:49:17Z`;
   segment files (`data_level0.bin` 321200, `header.bin` 100, `length.bin` 400,
   `link_lists.bin` 0) with mtime `2026-10-18T10:49:21Z`; no
   `chroma.sqlite3-wal`/`-shm` sidecars.
   **If anything differs: STOP testing, report the evidence. Do NOT retry the
   question, run ingestion, repair the store, or restore any files.**

In every stop case: preserve screenshots, execution IDs, log excerpts, and the
`chroma_hash.ps1` before/after outputs, then report.

---

## Service health commands (no questions, no Chroma queries)

```powershell
# Backend health (safe, read-only)
curl.exe -s http://127.0.0.1:8000/api/v1/health

# Ollama alive + expected models (no generation request)
curl.exe -s http://127.0.0.1:11434/api/tags

# Frontend alive (expect HTTP 200)
curl.exe -s -o NUL -w "%{http_code}" http://localhost:3333/
```

Expected: backend `"status":"ok"`, `"provider":"ollama"`, `"state":"ready"`,
`model qwen3:0.6b`; Ollama lists `qwen3:0.6b` and `nomic-embed-text`; frontend
returns `200`.

**Confirming the frontend serves the current working tree:** the preview is a
`vite` dev server started from
`C:\Users\hashe\Documents\Agentic_System\frontend` (verified via its process
command line) — dev servers read source files from disk on every request, so
the preview always serves the current tree. As a sanity check, the Dashboard
must show the CURRENT execution history (newest `#c2a52bcb3b1c` as of
preparation time). If in doubt, hard-refresh with **Ctrl+Shift+R**.

## Where to find things

- **Backend log (current):** `.diag\backend_20261018_m5loaded.log`
- **Execution IDs:** page heading `Execution #<id>`, the URL
  `http://localhost:3333/#/executions/<id>`, or Dashboard → Recent executions
- **Recovery markers:** `Select-String -Path .diag\backend_20261018_m5loaded.log -Pattern "self_refusal_recovery"`
- **Chroma integrity:** `powershell -ExecutionPolicy Bypass -File .diag\chroma_hash.ps1`

## Startup checklist (only if services are down tomorrow)

Services are ordinary local processes — they do **not** survive machine
restart, sleep, or logout, and persistence overnight is NOT guaranteed. If
anything is down:

1. **Ollama:** start "Ollama" from the Start menu; verify with
   `curl.exe -s http://127.0.0.1:11434/api/tags` (models listed).
2. **Backend** (run from the project root):
   ```powershell
   cd C:\Users\hashe\Documents\Agentic_System
   & "C:\Program Files\Python311\python.exe" -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000
   ```
   Wait ~5 s, then verify `curl.exe -s http://127.0.0.1:8000/api/v1/health`
   returns `"status":"ok"`. (The backend has NO auto-reload: if
   `books_agent.py` is ever edited, this restart is mandatory before testing.)
3. **Frontend:**
   ```powershell
   cd C:\Users\hashe\Documents\Agentic_System\frontend
   npm run dev
   ```
   Port 3333 is strict — if it fails, the port is occupied; report rather than
   killing processes.
4. Re-run **Test A** before submitting any question.
