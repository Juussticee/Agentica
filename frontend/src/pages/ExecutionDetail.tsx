import { useEffect, useState } from 'react'
import type { ApprovalRequest, Execution, ExecutionEvent } from '../api/types'
import { getExecution, getEvents, listApprovals } from '../api/client'
import { Loading, ErrorCard, StatusBadge } from '../components/ui'
import PlanSection from '../components/PlanSection'
import { allowlistedClass } from '../lib/classSuffix'

// Event severities recorded by the backend serializer (see
// app/execution/events.py); anything else renders the neutral base class.
const EVENT_SEVERITIES = ['info', 'warning', 'error'] as const

type ApiError = Error & { why?: string; recovery?: string }

const AGENT_NAMES: Record<string, string> = {
  supervisor: 'Supervisor — understanding the request',
  planner: 'Planner — decomposing the work',
  dispatcher: 'Dispatcher — running the tasks',
  research: 'Web Research',
  books: 'Internal Knowledge (RAG)',
  erp: 'Business Data',
  feasibility: 'Feasibility Review',
  coding: 'Workspace Coding',
  data: 'Data Analyst',
  followup: 'Contextual Follow-up',
  synthesis: 'Synthesis',
  conversational: 'Conversation',
  general: 'General Knowledge',
  unsupported: 'Capability Guidance',
  qa: 'Quality Gate',
}

function humanAgent(id?: string | null): string {
  if (!id) return 'System'
  return AGENT_NAMES[id] ?? id
}

export default function ExecutionDetail({ id }: { id: string }) {
  const [exec, setExec] = useState<Execution | null>(null)
  const [events, setEvents] = useState<ExecutionEvent[] | null>(null)
  const [approvals, setApprovals] = useState<ApprovalRequest[] | null>(null)
  const [loadErr, setLoadErr] = useState<ApiError | null>(null)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const [e, ev, ap] = await Promise.all([
          getExecution(id),
          getEvents(id),
          // Every decision state for THIS run (read-only union).
          listApprovals('all').catch(() => null),
        ])
        if (!alive) return
        setExec(e)
        setEvents(ev.events)
        setApprovals((ap?.approvals ?? []).filter((a) => a.execution_id === id))
        setLoadErr(null)
      } catch (err) {
        if (!alive) return
        setLoadErr(
          Object.assign(err instanceof Error ? err : new Error(String(err)), {
            why: (err as ApiError).why,
            recovery: (err as ApiError).recovery,
          }),
        )
      }
    }
    load()
    // Poll while the execution is (possibly) still active; every second like
    // the original preview. Once terminal, stop automatically.
    const t = setInterval(async () => {
      try {
        const e = await getExecution(id)
        if (!alive) return
        setExec(e)
        if (e.status === 'running' || e.status === 'queued' || e.status === 'waiting_for_approval') {
          const ev = await getEvents(id)
          if (alive) setEvents(ev.events)
        }
      } catch {
        /* transient poll error: keep last good state */
      }
    }, 1000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [id])

  if (loadErr) {
    return (
      <ErrorCard
        message={loadErr.message}
        why={loadErr.why}
        recovery={loadErr.recovery ?? 'Go back to the executions list and retry.'}
      />
    )
  }
  if (!exec) return <Loading label="Loading execution…" />

  const running = exec.status === 'running' || exec.status === 'queued'
  const waitingForApproval = exec.status === 'waiting_for_approval'

  return (
    <div>
      <h1>Execution #{exec.id}</h1>
      <p className="muted">{exec.question}</p>

      <div className="detail-head">
        <StatusBadge status={exec.status} />
        {running && <span className="pulse" aria-hidden />}
        <span className="muted">
          {exec.status === 'running' ? 'In progress…' : ''}{' '}
          {exec.status === 'failed' ? 'This execution did not finish.' : ''}
          {waitingForApproval ? 'External research is paused until you decide in Approval Center.' : ''}
        </span>
      </div>
      {waitingForApproval && (
        <section className="card card--warn" aria-label="Waiting for approval">
          <h2>Waiting for your approval</h2>
          <p className="muted">No external research has run. Review the proposed operation before it continues.</p>
          <a href="#/approvals">Open Approval Center</a>
        </section>
      )}

      {approvals && approvals.length > 0 && (
        <section className="card" aria-label="Approvals for this execution">
          <h2>Approvals for this execution</h2>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Approval</th><th>Requested by</th><th>Operation</th>
                  <th>Status</th><th>Expires</th><th>Decided</th><th>Executed</th>
                </tr>
              </thead>
              <tbody>
                {approvals.map((a) => (
                  <tr key={a.approval_id}>
                    <td><code>{a.approval_id.slice(0, 8)}…</code></td>
                    <td>{a.requesting_agent ? humanAgent(a.requesting_agent) : '—'}</td>
                    <td>{a.operation}</td>
                    <td>{a.status}</td>
                    <td>{new Date(a.expires_at).toLocaleString()}</td>
                    <td>{a.decided_at ? new Date(a.decided_at).toLocaleString() : '—'}</td>
                    <td>{a.executed_at ? new Date(a.executed_at).toLocaleString() : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">
            Decisions are made in <a href="#/approvals">Approval Center</a>;
            this table is read-only.
          </p>
        </section>
      )}

      {exec.error && (
        <ErrorCard
          message={exec.error.human_message}
          why={exec.error.why}
          recovery={exec.error.recovery}
          technical={
            exec.error.details
              ? JSON.stringify(exec.error.details, null, 2)
              : exec.error.category
          }
        />
      )}

      {/* GUI-12: the real persisted plan. Null for simple routes and legacy
          rows, so nothing renders there. While a run is still going, the
          plan is only recorded when it finishes — say so honestly. */}
      <PlanSection plan={exec.plan} />
      {exec.plan && (
        <p className="muted plan-attribution">
          Each task above names the specialist that handled it; results and
          statuses are the persisted dispatcher data, not a summary.
        </p>
      )}
      {running && !exec.plan && (
        <section className="card plan plan--pending" aria-label="Execution plan">
          <h2>Plan (what the agent intended)</h2>
          <p className="muted">The plan appears here once it has been recorded.</p>
        </section>
      )}

      {exec.answer && (
        <section className="card">
          <h2>Result</h2>
          <div className="answer">{exec.answer}</div>
          {exec.qa_flags && exec.qa_flags.length > 0 && (
            <div className="qa-flags">
              {exec.qa_flags.map((f, i) => (
                <div key={i} className="qa-flag">{f}</div>
              ))}
            </div>
          )}
        </section>
      )}

      <section className="card">
        <h2>What happened</h2>
        {events === null ? (
          <Loading label="Loading timeline…" />
        ) : events.length === 0 ? (
          <p className="muted">No events recorded.</p>
        ) : (
          <ol className="timeline">
            {events.map((e) => (
              <li
                key={e.event_id}
                className={allowlistedClass(
                  'timeline__item', 'timeline__item--', EVENT_SEVERITIES, e.severity,
                )}
              >
                <span className="timeline__time">
                  {new Date(e.timestamp).toLocaleTimeString()}
                </span>
                <span className="timeline__msg">{e.message}</span>
                <span className="timeline__meta muted">
                  {humanAgent(e.agent_id)} · {e.type}
                </span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  )
}
