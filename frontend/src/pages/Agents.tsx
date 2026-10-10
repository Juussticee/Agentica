import { useEffect, useState } from 'react'
import { getAgentActivity, type AgentActivity } from '../api/dashboard'
import { Loading, ErrorCard } from '../components/ui'

const AGENT_NAMES: Record<string, string> = {
  supervisor: 'Understanding the request',
  research: 'Web Research',
  books: 'Internal Knowledge',
  erp: 'Business Data',
  feasibility: 'Feasibility Review',
  qa: 'Result Check',
}

const STATUS_LABEL: Record<AgentActivity['status'], string> = {
  idle: 'Idle',
  working: 'Working',
  completed: 'Completed (last run)',
  failed: 'Failed (last run)',
}

export default function Agents() {
  const [data, setData] = useState<Awaited<ReturnType<typeof getAgentActivity>> | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const d = await getAgentActivity()
        if (alive) { setData(d); setError(null) }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      }
    }
    load()
    const t = setInterval(load, 2000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  const working = (data?.agents ?? []).filter((a) => a.status === 'working')
  const failed = (data?.agents ?? []).filter((a) => a.status === 'failed')

  return (
    <div>
      <h1>Agents</h1>
      <p className="muted">
        {data?.note ?? 'Activity reflects recorded execution events only.'}
        {' '}An agent is marked working only while an execution has recorded
        it running — never just because it exists.
      </p>
      {error && <ErrorCard message={error} recovery="Start the backend, then refresh." />}
      {working.length > 0 && (
        <section className="card card--warn">
          <h2>Working now ({working.length})</h2>
          {working.map((a) => <AgentRow key={a.node} a={a} />)}
        </section>
      )}
      {failed.length > 0 && (
        <section className="card">
          <h2>Failed ({failed.length})</h2>
          {failed.map((a) => <AgentRow key={a.node} a={a} />)}
        </section>
      )}
      {data && working.length === 0 && failed.length === 0 && (
        <section className="card">
          <h2>No agents active</h2>
          <p className="muted">Nothing is running right now. Start an execution to see live activity.</p>
        </section>
      )}
      {data && (
        <section className="card">
          <h2>All agents ({data.agents.length})</h2>
          {data.agents.map((a) => <AgentRow key={a.node} a={a} />)}
        </section>
      )}
      {!data && !error && <Loading label="Reading events…" />}
    </div>
  )
}

function AgentRow({ a }: { a: AgentActivity }) {
  const name = AGENT_NAMES[a.node] ?? a.node
  return (
    <article className={`agent-row agent-row--${a.status}`}>
      <div className="agent-row__main">
        <strong>{name}</strong>
        <span className="agent-row__tech muted">node: {a.node}</span>
        <div className="agent-row__role muted">{a.role}</div>
      </div>
      <div className="agent-row__side">
        <span className={`badge badge--${a.status === 'failed' ? 'failed'
          : a.status === 'working' ? 'running'
            : a.status === 'completed' ? 'completed' : 'queued'}`}>
          {STATUS_LABEL[a.status]}
        </span>
        {a.execution_id ? (
          <a href={`#/executions/${encodeURIComponent(a.execution_id)}`}>
            #{a.execution_id}
          </a>
        ) : (
          <span className="muted">—</span>
        )}
      </div>
      {a.task && <p className="agent-row__task muted">Task: {a.task}</p>}
      {a.last_event && (
        <p className="agent-row__event">Last event: {a.last_event}
          {a.last_event_at && <> · {new Date(a.last_event_at).toLocaleTimeString()}</>}
        </p>
      )}
      {a.last_error && (
        <p className="agent-row__err">Failure category: {a.last_error}.
          Open the execution for next actions.</p>
      )}
    </article>
  )
}
