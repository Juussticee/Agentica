import { useEffect, useState } from 'react'
import { createExecution, listExecutions } from '../api/client'
import { getDashboardSummary, type DashboardSummary } from '../api/catalog'
import { useHealth } from '../api/health'
import { Loading, Empty, ErrorCard, StatusBadge, Button } from '../components/ui'
import { AiReadinessPanel } from '../components/AiReadiness'
import { allowlistedClass } from '../lib/classSuffix'

const SEVERITIES = ['info', 'warning', 'error'] as const

export default function Dashboard({ onNavigate }: { onNavigate: (to: string) => void }) {
  const { health, error: healthError } = useHealth()
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [summaryError, setSummaryError] = useState<string | null>(null)
  const [execs, setExecs] = useState<Awaited<ReturnType<typeof listExecutions>>['executions'] | null>(null)
  const [busy, setBusy] = useState(false)
  const [runError, setRunError] = useState<{ message: string; why?: string } | null>(null)
  const [lastId, setLastId] = useState<string | null>(null)

  async function refresh() {
    try {
      const [summaryData, list] = await Promise.all([
        getDashboardSummary(),
        listExecutions().catch(() => null),
      ])
      setSummary(summaryData)
      if (list) setExecs(list.executions)
      setSummaryError(null)
    } catch (e) {
      setSummaryError(e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => { refresh() }, [])

  // Poll only while something is actually running.
  useEffect(() => {
    const active = (summary?.executions.active ?? 0) > 0
    if (!active) return
    const t = setInterval(refresh, 2000)
    return () => clearInterval(t)
  }, [summary?.executions.active])

  async function quickRun() {
    const question = 'What does chapter 3 say about propolis?'
    setBusy(true)
    setRunError(null)
    try {
      const created = await createExecution(question)
      setLastId(created.id)
      await refresh()
      onNavigate(`executions/${encodeURIComponent(created.id)}`)
    } catch (e) {
      const err = e as Error & { why?: string }
      setRunError({ message: err.message, why: err.why })
    } finally {
      setBusy(false)
    }
  }

  const recent = summary?.recent_executions ?? (execs ?? []).slice(0, 5).map((x) => ({
    id: x.id, question: x.question, status: x.status, route: x.route ?? null,
    created_at: x.created_at, completed_at: x.completed_at ?? null, has_plan: false,
  }))

  return (
    <div>
      <h1>Dashboard</h1>
      {healthError && (
        <ErrorCard
          message="The Agentica backend is not reachable."
          why={healthError}
          recovery="Make sure the API is running (python -m app.api.server), then retry."
        />
      )}
      {summaryError && <ErrorCard message={summaryError} recovery="Refresh the page; the backend may still be starting." />}

      {!summary && !summaryError && <Loading label="Reading the operational summary…" />}

      {summary && (
        <>
          <div className="tiles">
            <button type="button" className="tile" onClick={() => onNavigate('executions')}>
              <span className="tile__value">{summary.executions.total}</span>
              <span className="tile__label">Executions</span>
              <span className="tile__hint">{summary.executions.active} active · {summary.executions.last_24h} in 24h</span>
            </button>
            <button type="button" className="tile" onClick={() => onNavigate('approvals')}>
              <span className="tile__value">{summary.approvals.pending}</span>
              <span className="tile__label">Pending approvals</span>
              <span className="tile__hint">
                {summary.executions.waiting_approval} run(s) paused
              </span>
            </button>
            <button type="button" className="tile" onClick={() => onNavigate('agents')}>
              <span className="tile__value">{summary.agents.available}/{summary.agents.registered}</span>
              <span className="tile__label">Agents available</span>
              <span className="tile__hint">
                {summary.agents.disabled} disabled · {summary.agents.unavailable} unavailable
              </span>
            </button>
            <button type="button" className="tile" onClick={() => onNavigate('skills')}>
              <span className="tile__value">{summary.skills.installed}/{summary.skills.total}</span>
              <span className="tile__label">Skills on disk</span>
              <span className="tile__hint">Documented procedures (not executable)</span>
            </button>
          </div>

          <div className="grid">
            <section className="card">
              <h2>System health</h2>
              <ul className="health-list">
                <li>Agent system: <strong>{summary.health.backend === 'ok' ? 'Working' : summary.health.backend}</strong></li>
                <li>Storage: <strong>{summary.health.storage === 'ok' ? 'Connected' : summary.health.storage}</strong></li>
                <li>Execution infrastructure: <strong>local runner (single process)</strong></li>
                <li>Backend version: <strong>{health?.version ?? '-'}</strong></li>
              </ul>
            </section>
            <AiReadinessPanel ai={health?.ai ?? null} />
          </div>

          <section className="card">
            <h2>Needs attention ({summary.attention.length})</h2>
            {summary.attention.length === 0 ? (
              <p className="muted">Nothing failed and nothing is waiting for a decision.</p>
            ) : (
              <ul className="exec-mini-list">
                {summary.attention.map((item) => (
                  <li key={item.id}>
                    <a href={`#/executions/${encodeURIComponent(item.id)}`}>#{item.id}</a>
                    {' '}<StatusBadge status={item.status as never} /> — {item.question}
                    <div className="muted">
                      {item.human_message ?? (item.status === 'waiting_for_approval'
                        ? 'Paused until you decide in Approval Center.'
                        : 'No structured error was recorded.')}
                      {item.category && <> · category: {item.category}</>}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="card">
            <h2>Recent executions</h2>
            {recent.length === 0 ? (
              <Empty>No executions yet. Start one from Quick start or New Execution.</Empty>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr><th>ID</th><th>Request</th><th>Route</th><th>Status</th><th>Created</th></tr>
                  </thead>
                  <tbody>
                    {recent.map((x) => (
                      <tr key={x.id}>
                        <td><a href={`#/executions/${encodeURIComponent(x.id)}`}>#{x.id}</a></td>
                        <td className="truncate">{x.question}</td>
                        <td>{x.route ?? '—'}</td>
                        <td><StatusBadge status={x.status as never} /></td>
                        <td>{new Date(x.created_at).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="card">
            <h2>Recent activity</h2>
            <p className="muted">Recorded events across the most recent executions.</p>
            {summary.activity.length === 0 ? (
              <Empty>No events recorded yet.</Empty>
            ) : (
              <ol className="timeline">
                {summary.activity.map((item, index) => (
                  <li
                    key={`${item.execution_id}-${item.at}-${index}`}
                    className={allowlistedClass('timeline__item', 'timeline__item--', SEVERITIES, item.severity)}
                  >
                    <span className="timeline__time">{new Date(item.at).toLocaleTimeString()}</span>
                    <span className="timeline__msg">{item.message}</span>
                    <span className="timeline__meta muted">
                      <a href={`#/executions/${encodeURIComponent(item.execution_id)}`}>
                        #{item.execution_id}
                      </a>
                      {item.agent_id ? ` · ${item.agent_id}` : ''} · {item.type}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </section>

          <section className="card">
            <h2>Quick start</h2>
            <p className="muted">Runs a sample books question against the real workflow.</p>
            <Button onClick={quickRun} disabled={busy}>
              {busy ? 'Starting execution…' : 'Run sample execution'}
            </Button>
            {runError && <ErrorCard message={runError.message} why={runError.why} onDismiss={() => setRunError(null)} />}
            {lastId && !runError && <p className="muted">Last started: #{lastId}</p>}
          </section>
        </>
      )}
    </div>
  )
}
