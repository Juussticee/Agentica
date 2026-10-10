import { useEffect, useState } from 'react'
import type { Execution } from '../api/types'
import type { ExecutionSequence } from '../api/dashboard'
import { Loading, ErrorCard, StatusBadge } from '../components/ui'
import { StageRow } from './TasksSequence'

export default function Executions() {
  const [execs, setExecs] = useState<Execution[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<ExecutionSequence | null>(null)

  const openDetail = async (id: string) => {
    setSelected(null)
    const res = await fetch(`/api/v1/executions/${encodeURIComponent(id)}/sequence`)
    if (res.ok) setSelected(await res.json())
    else {
      setError(`Could not load execution ${id} (${res.status}).`)
      setTimeout(() => setError(null), 3000)
    }
  }

  useEffect(() => {
    fetch('/api/v1/executions')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((d) => setExecs(d.executions))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  return (
    <div>
      <h1>Executions</h1>
      {error && <ErrorCard message={error} />}
      {execs === null && !error && <Loading />}
      {execs?.length === 0 && <p className="muted">No executions yet.</p>}
      {execs && execs.length > 0 && (
        <div className="exec-list">
          {execs.map((x) => (
            <div key={x.id} className="card exec-card" onClick={() => openDetail(x.id)}>
              <div className="exec-card__head">
                <strong>#{x.id}</strong>
                <StatusBadge status={x.status} />
              </div>
              <p>{x.question}</p>
              <span className="muted">{new Date(x.created_at).toLocaleString()}</span>
              {x.answer && x.status === 'completed' && (
                <details className="svc-config"><summary>Show answer</summary>
                  <pre>{x.answer}</pre></details>
              )}
              {x.error && x.status === 'failed' && (
                <p className="svc-reason">{x.error.human_message}</p>
              )}
            </div>
          ))}
        </div>
      )}
      {selected && (
        <section className="card">
          <header className="seq-head"><h2>#{selected.execution_id} — sequence</h2></header>
          <ol className="stage-list">
            {selected.stages.map((s) => <StageRow key={s.event_id} s={s} />)}
          </ol>
        </section>
      )}
    </div>
  )
}
