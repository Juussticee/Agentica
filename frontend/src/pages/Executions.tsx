import { useCallback, useEffect, useState } from 'react'
import type { Execution, ExecutionStatus } from '../api/types'
import { executionQuery, type ExecutionFilters } from '../api/catalog'
import { Empty, ErrorCard, Loading, StatusBadge } from '../components/ui'

const STATUSES: { value: '' | ExecutionStatus; label: string }[] = [
  { value: '', label: 'All statuses' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'waiting_for_approval', label: 'Waiting for approval' },
  { value: 'running', label: 'Running' },
  { value: 'queued', label: 'Queued' },
  { value: 'cancelled', label: 'Cancelled' },
]

type Range = '' | '24h' | '7d' | '30d'

const RANGES: { value: Range; label: string }[] = [
  { value: '', label: 'Any date' },
  { value: '24h', label: 'Last 24 hours' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
]

function sinceFor(range: Range): string | undefined {
  if (!range) return undefined
  const hours = range === '24h' ? 24 : range === '7d' ? 24 * 7 : 24 * 30
  return new Date(Date.now() - hours * 3600_000).toISOString()
}

export default function Executions() {
  const [status, setStatus] = useState<'' | ExecutionStatus>('')
  const [range, setRange] = useState<Range>('')
  const [query, setQuery] = useState('')
  const [execs, setExecs] = useState<Execution[] | null>(null)
  const [total, setTotal] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    const filters: ExecutionFilters = {
      status: status || undefined,
      q: query.trim() || undefined,
      since: sinceFor(range),
      limit: 200,
    }
    try {
      const res = await fetch(`/api/v1/executions${executionQuery(filters)}`)
      if (!res.ok) {
        let message = `Executions failed (${res.status})`
        try {
          const body = await res.json()
          message = body?.detail?.human_message ?? message
        } catch { /* not JSON */ }
        throw new Error(message)
      }
      const body = await res.json()
      setExecs(body.executions)
      setTotal(body.total ?? body.count)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [status, range, query])

  // Debounce the text box only; the filters apply immediately.
  useEffect(() => {
    const t = setTimeout(load, query ? 250 : 0)
    return () => clearTimeout(t)
  }, [load, query])

  const filtering = Boolean(status || range || query.trim())

  return (
    <div>
      <h1>Executions</h1>
      <p className="muted">
        {execs === null
          ? 'Loading history…'
          : `${execs.length} of ${total} recorded execution(s)${filtering ? ' match these filters' : ''}.`}
      </p>

      <div className="toolbar toolbar--wrap">
        <label className="field">
          <span>Status</span>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as '' | ExecutionStatus)}
          >
            {STATUSES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Created</span>
          <select value={range} onChange={(e) => setRange(e.target.value as Range)}>
            {RANGES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
          </select>
        </label>
        <label className="field field--grow">
          <span>Search request text</span>
          <input
            type="search"
            value={query}
            placeholder="Filter by what was asked…"
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        {filtering && (
          <button
            type="button"
            className="btn btn--ghost"
            onClick={() => { setStatus(''); setRange(''); setQuery('') }}
          >
            Clear filters
          </button>
        )}
      </div>

      {error && <ErrorCard message={error} recovery="Refresh the page and retry." />}
      {execs === null && !error && <Loading />}
      {execs && execs.length === 0 && (
        <Empty>
          {filtering
            ? 'No execution matches these filters.'
            : 'No executions yet. Start one from New Execution.'}
        </Empty>
      )}

      {execs && execs.length > 0 && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>ID</th><th>Request</th><th>Route</th><th>Status</th>
                <th>Created</th><th>Finished</th>
              </tr>
            </thead>
            <tbody>
              {execs.map((x) => (
                <tr key={x.id} className="row-click" onClick={() => {
                  window.location.hash = `#/executions/${encodeURIComponent(x.id)}`
                }}>
                  <td><a href={`#/executions/${encodeURIComponent(x.id)}`}>#{x.id}</a></td>
                  <td className="truncate">{x.question}</td>
                  <td>{x.route ?? '—'}</td>
                  <td><StatusBadge status={x.status} /></td>
                  <td>{new Date(x.created_at).toLocaleString()}</td>
                  <td>{x.completed_at ? new Date(x.completed_at).toLocaleString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
