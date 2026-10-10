import { useEffect, useState } from 'react'
import type { Execution } from '../api/types'
import { listExecutions, createExecution } from '../api/client'
import { useHealth } from '../api/health'
import { Loading, Empty, ErrorCard, StatusBadge, Button } from '../components/ui'
import { AiReadinessPanel } from '../components/AiReadiness'

export default function Dashboard({ onNavigate }: { onNavigate: (to: string) => void }) {
  const { health, error: healthError } = useHealth()
  const [execs, setExecs] = useState<Execution[] | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [runError, setRunError] = useState<{ message: string; why?: string } | null>(null)
  const [lastId, setLastId] = useState<string | null>(null)

  async function refresh() {
    try {
      const r = await listExecutions()
      setExecs(r.executions)
      setListError(null)
    } catch (e) {
      setListError(e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => {
    refresh()
  }, [])

  // Poll while anything is running (real progress only; no fake states).
  useEffect(() => {
    const anyActive = (execs ?? []).some((x) => x.status === 'running' || x.status === 'queued')
    if (!anyActive) return
    const t = setInterval(refresh, 1000)
    return () => clearInterval(t)
  }, [execs])

  async function quickRun() {
    const question = "What does chapter 3 say about propolis?"
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

  const failed = (execs ?? []).filter((x) => x.status === 'failed')
  const recent = (execs ?? []).slice(0, 5)

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

      <div className="grid">
        <section className="card">
          <h2>System health</h2>
          {health ? (
            <ul className="health-list">
              <li>Agent system: <strong>{health.backend === 'ok' ? 'Working' : health.backend}</strong></li>
              <li>Backend version: <strong>{health.version ?? '-'}</strong></li>
            </ul>
          ) : (
            !healthError && <Loading label="Checking system health…" />
          )}
        </section>

        <AiReadinessPanel ai={health?.ai ?? null} />
      </div>

      <section className="card">
        <h2>Quick start</h2>
        <p className="muted">Runs a sample books question against the real workflow.</p>
        <Button onClick={quickRun} disabled={busy}>
          {busy ? 'Starting execution…' : 'Run sample execution'}
        </Button>
        {runError && <ErrorCard message={runError.message} why={runError.why} onDismiss={() => setRunError(null)} />}
        {lastId && !runError && (
          <p className="muted">Last started: #{lastId}</p>
        )}
      </section>

      {failed.length > 0 && (
        <section className="card card--warn">
          <h2>Needs attention ({failed.length})</h2>
          <ul className="exec-mini-list">
            {failed.map((x) => (
              <li key={x.id}>
                <a href={`#/executions/${encodeURIComponent(x.id)}`}>#{x.id}</a> — {x.question}
                <div className="muted">{x.error?.human_message}</div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="card">
        <h2>Recent executions</h2>
        {listError ? (
          <ErrorCard message={listError} />
        ) : execs === null ? (
          <Loading />
        ) : recent.length === 0 ? (
          <Empty>No executions yet. Start one from Quick start or New Execution.</Empty>
        ) : (
          <table className="table">
            <thead>
              <tr><th>ID</th><th>Request</th><th>Status</th><th>Created</th></tr>
            </thead>
            <tbody>
              {recent.map((x) => (
                <tr key={x.id}>
                  <td><a href={`#/executions/${encodeURIComponent(x.id)}`}>#{x.id}</a></td>
                  <td className="truncate">{x.question}</td>
                  <td><StatusBadge status={x.status} /></td>
                  <td>{new Date(x.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
