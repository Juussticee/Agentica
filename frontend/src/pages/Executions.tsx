import { useEffect, useState } from 'react'
import type { Execution } from '../api/types'
import { listExecutions } from '../api/client'
import { Loading, Empty, ErrorCard, StatusBadge } from '../components/ui'

export default function Executions({
  openDetail,
}: {
  onNavigate: (to: string) => void
  openDetail: (id: string) => void
}) {
  const [execs, setExecs] = useState<Execution[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const r = await listExecutions()
        if (alive) setExecs(r.executions)
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      }
    }
    load()
    return () => {
      alive = false
    }
  }, [])

  return (
    <div>
      <h1>Executions</h1>
      {error ? (
        <ErrorCard message={error} recovery="Make sure the backend is running, then refresh." />
      ) : execs === null ? (
        <Loading />
      ) : execs.length === 0 ? (
        <Empty>No executions yet.</Empty>
      ) : (
        <table className="table">
          <thead>
            <tr><th>ID</th><th>Request</th><th>Status</th><th>Created</th></tr>
          </thead>
          <tbody>
            {execs.map((x) => (
              <tr key={x.id} className="row-click" onClick={() => openDetail(x.id)}>
                <td><a href={`#/executions/${encodeURIComponent(x.id)}`} onClick={(e) => e.stopPropagation()}>#{x.id}</a></td>
                <td className="truncate">{x.question}</td>
                <td><StatusBadge status={x.status} /></td>
                <td>{new Date(x.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
