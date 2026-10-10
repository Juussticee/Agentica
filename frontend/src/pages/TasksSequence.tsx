import { useState } from 'react'
import { getSequence, type SequenceStage } from '../api/dashboard'
import { Button, Empty, ErrorCard, StatusBadge } from '../components/ui'

export function StageRow({ s }: { s: SequenceStage }) {
  return (
    <li className={`stage stage--${s.status}`}>
      <div className="stage__head">
        <StatusBadge status={s.status === 'running' ? 'running'
          : s.status === 'failed' ? 'failed' : 'completed'} />
        <span className="stage__msg">{s.message}</span>
      </div>
      <code className="stage__node">{s.node ?? s.stage}</code>
      <span className="muted stage__at">{new Date(s.at).toLocaleTimeString()}</span>
      {s.qa_flags && s.qa_flags.length > 0 && (
        <div className="qa-flags">{s.qa_flags.map((f, i) => <div key={i} className="qa-flag">{f}</div>)}</div>
      )}
      {s.next_action && <p className="svc-next"><strong>Next:</strong> {s.next_action}</p>}
    </li>
  )
}

export default function TasksSequence() {
  const [id, setId] = useState('')
  const [seq, setSeq] = useState<Awaited<ReturnType<typeof getSequence>> | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function load(e: React.FormEvent) {
    e.preventDefault()
    if (!id.trim()) { setError('Enter an execution ID (for example from the Executions page).'); return }
    setBusy(true)
    setError(null)
    setSeq(null)
    try {
      setSeq(await getSequence(id.trim()))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <h1>Tasks &amp; sequence</h1>
      <p className="muted">
        This build runs each request as a sequential graph route — there is no
        separate queue service. Here you can follow one execution from request
        → routing → agents → QA → outcome, using its real recorded events.
        Planner-based dependency graphs arrive with MASTER_PLAN Phase 7/8.
      </p>

      <form onSubmit={load} className="card form form-row">
        <input
          value={id}
          onChange={(e) => setId(e.target.value)}
          placeholder="Execution ID (e.g. e315f9fb9268)"
          disabled={busy}
          aria-label="Execution ID"
        />
        <Button type="submit" disabled={busy}>{busy ? 'Loading…' : 'Show sequence'}</Button>
      </form>
      {error && <ErrorCard message={error} />}

      {seq && (
        <section className="card">
          <header className="seq-head">
            <h2>#{seq.execution_id}</h2>
            <StatusBadge status={seq.status === 'completed' ? 'completed'
              : seq.status === 'failed' ? 'failed' : 'running'} />
            {seq.route && <span className="badge badge--running">route: {seq.route}</span>}
          </header>
          <p className="muted">{seq.question}</p>
          <ol className="stage-list">
            {seq.stages.map((s) => <StageRow key={s.event_id} s={s} />)}
          </ol>
          <p className="muted">{seq.model}</p>
        </section>
      )}
      {!seq && !busy && !error && (
        <section className="card">
          <Empty>No execution selected.</Empty>
        </section>
      )}
    </div>
  )
}
