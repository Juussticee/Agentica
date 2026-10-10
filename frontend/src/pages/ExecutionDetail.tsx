import { useEffect, useState } from 'react'
import type { Execution, ExecutionEvent } from '../api/types'
import { getExecution, getEvents } from '../api/client'
import { Loading, ErrorCard, StatusBadge } from '../components/ui'

type ApiError = Error & { why?: string; recovery?: string }

const AGENT_NAMES: Record<string, string> = {
  supervisor: 'Understanding the request',
  research: 'Web Research',
  books: 'Internal Knowledge',
  erp: 'Business Data',
  feasibility: 'Feasibility Review',
  qa: 'Result Check',
}

function humanAgent(id?: string | null): string {
  if (!id) return 'System'
  return AGENT_NAMES[id] ?? id
}

export default function ExecutionDetail({ id }: { id: string }) {
  const [exec, setExec] = useState<Execution | null>(null)
  const [events, setEvents] = useState<ExecutionEvent[] | null>(null)
  const [loadErr, setLoadErr] = useState<ApiError | null>(null)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const e = await getExecution(id)
        const ev = await getEvents(id)
        if (!alive) return
        setExec(e)
        setEvents(ev.events)
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
        if (e.status === 'running' || e.status === 'queued') {
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
        </span>
      </div>

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
              <li key={e.event_id} className={`timeline__item timeline__item--${e.severity}`}>
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
