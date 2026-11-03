import { useEffect, useState } from 'react'
import { getMemory, type MemorySection, type MemorySummary } from '../api/catalog'
import { Empty, ErrorCard, Loading, StatusBadge } from '../components/ui'

const STATUS_LABEL: Record<MemorySection['status'], string> = {
  available: 'Available',
  empty: 'Empty',
  not_implemented: 'Not implemented',
}

function SectionHead({ section }: { section: MemorySection }) {
  return (
    <header className="memory-section__head">
      <h2>{section.title}</h2>
      <span className={`badge badge--${section.status}`}>{STATUS_LABEL[section.status]}</span>
    </header>
  )
}

// A count is either a number or a {status: number} breakdown (the backend's
// by_status map). Both are rendered as text; nothing is summed or guessed.
function formatCount(value: number | Record<string, number>): string {
  if (typeof value === 'number') return String(value)
  const parts = Object.entries(value).map(([status, count]) => `${status}: ${count}`)
  return parts.length > 0 ? parts.join(' · ') : '—'
}

function Provenance({
  text,
  status,
}: {
  text: string | null
  status: MemorySection['status']
}) {
  if (!text) {
    return (
      <p className="muted memory-provenance">
        {status === 'not_implemented'
          ? 'No source — this store does not exist, so nothing here can go stale.'
          : 'No source to show yet because nothing has been recorded here. When a record is stored, its origin is shown.'}
      </p>
    )
  }
  return <p className="muted memory-provenance">Source: {text}</p>
}

export default function Memory() {
  const [data, setData] = useState<MemorySummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    getMemory()
      .then((summary) => { if (alive) { setData(summary); setError(null) } })
      .catch((e) => { if (alive) setError(e instanceof Error ? e.message : String(e)) })
    return () => { alive = false }
  }, [])

  if (error) {
    return (
      <div>
        <h1>Memory</h1>
        <ErrorCard message={error} recovery="Check that the backend is running, then refresh." />
      </div>
    )
  }
  if (!data) return <div><h1>Memory</h1><Loading label="Reading memory stores…" /></div>

  return (
    <div>
      <h1>Memory</h1>
      <p className="muted memory__intro">{data.note}</p>

      {data.sections.map((section) => (
        <section className="card memory-section" key={section.id}>
          <SectionHead section={section} />
          <p>{section.summary}</p>

          {section.status === 'not_implemented' && (
            <p className="muted">
              This is shown so the gap is explicit: no interface is provided
              for a store that does not exist.
            </p>
          )}

          {section.counts && (
            <dl className="kv">
              {Object.entries(section.counts).map(([key, value]) => (
                <div key={key} className="kv__row">
                  <dt>{key}</dt>
                  <dd>{formatCount(value)}</dd>
                </div>
              ))}
            </dl>
          )}

          {section.id === 'executions' && section.items.length > 0 && (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr><th>Execution</th><th>Request</th><th>Status</th><th>Created</th></tr>
                </thead>
                <tbody>
                  {section.items.map((item) => (
                    <tr key={item.id}>
                      <td>
                        <a href={`#/executions/${encodeURIComponent(item.id ?? '')}`}>
                          #{item.id}
                        </a>
                      </td>
                      <td className="truncate">{item.question}</td>
                      <td>
                        <StatusBadge status={(item.status ?? 'queued') as never} />
                      </td>
                      <td>{item.created_at ? new Date(item.created_at).toLocaleString() : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {section.id === 'knowledge' && (
            <>
              <dl className="kv">
                <dt>Books directory</dt><dd className="mono-small">{section.books_dir ?? '—'}</dd>
                <dt>Index manifest</dt>
                <dd>
                  {section.index_path ?? '—'}{' '}
                  {section.index_present === null || section.index_present === undefined
                    ? '(not checked)'
                    : section.index_present ? '(present)' : '(missing — ingest has not run)'}
                </dd>
              </dl>
              {section.items.length === 0 ? (
                <Empty>No documents in the books corpus.</Empty>
              ) : (
                <div className="table-wrap">
                  <table className="table">
                    <thead><tr><th>Document</th><th>Path</th><th>Size</th><th>Modified</th></tr></thead>
                    <tbody>
                      {section.items.map((item) => (
                        <tr key={item.path}>
                          <td>{item.name}</td>
                          <td className="mono-small">{item.path}</td>
                          <td>{item.size_bytes ? `${Math.round(item.size_bytes / 1024)} KB` : '—'}</td>
                          <td>{item.modified_at ? new Date(item.modified_at).toLocaleString() : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {section.company_items && section.company_items.length > 0 && (
                <p className="muted">
                  Company documents on disk (not indexed here):{' '}
                  {section.company_items.map((item) => item.name).join(', ')}
                </p>
              )}
            </>
          )}

          <Provenance text={section.provenance} status={section.status} />
        </section>
      ))}

      <section className="card">
        <h2>Database totals</h2>
        <dl className="kv">
          <dt>Executions</dt><dd>{data.counts.executions.total}</dd>
          <dt>Approval records</dt><dd>{data.counts.approvals.total}</dd>
          <dt>Events</dt><dd>{data.counts.events}</dd>
          <dt>Oldest execution</dt>
          <dd>{data.counts.oldest ? new Date(data.counts.oldest).toLocaleString() : '—'}</dd>
          <dt>Newest execution</dt>
          <dd>{data.counts.newest ? new Date(data.counts.newest).toLocaleString() : '—'}</dd>
        </dl>
        <p className="muted">
          Counts are aggregate reads of the same local database the Executions
          page lists — nothing is estimated.
        </p>
      </section>
    </div>
  )
}
