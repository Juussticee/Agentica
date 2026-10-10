import { useEffect, useState } from 'react'
import { getServices, type ServiceState, type ServiceStatus } from '../api/dashboard'
import { Loading, ErrorCard, Empty } from '../components/ui'

const STATE_LABEL: Record<ServiceState, string> = {
  running: 'Running',
  stopped: 'Stopped',
  unreachable: 'Unreachable',
  degraded: 'Degraded',
  misconfigured: 'Misconfigured',
  not_configured: 'Not configured',
  unknown: 'Unknown',
}

export function ServiceBadge({ state }: { state: ServiceState }) {
  return (
    <span className={`badge svc-badge svc-badge--${state}`}>
      {STATE_LABEL[state] ?? state}
    </span>
  )
}

export function ServiceCard({ svc }: { svc: ServiceStatus }) {
  const problem = svc.state !== 'running'
  return (
    <article className={`svc-card ${problem ? 'svc-card--problem' : ''}`}>
      <header className="svc-card__head">
        <ServiceBadge state={svc.state} />
        <div>
          <h3>{svc.name}</h3>
          <p className="muted">{svc.purpose}</p>
        </div>
      </header>
      <dl className="svc-meta">
        <div><dt>Required</dt><dd>{svc.required ? 'Yes' : 'Optional'}</dd></div>
        <div><dt>Latency</dt><dd>{svc.latency_ms != null ? `${svc.latency_ms} ms` : '—'}</dd></div>
        <div><dt>Checked</dt><dd>{svc.last_checked ? new Date(svc.last_checked).toLocaleTimeString() : '—'}</dd></div>
      </dl>
      {svc.reason && <p className="svc-reason">{svc.reason}</p>}
      {svc.next_action && (
        <p className="svc-next">
          <strong>Next:</strong> {svc.next_action}
        </p>
      )}
      {svc.config && Object.keys(svc.config).length > 0 && (
        <details className="svc-config">
          <summary>Configuration (technical)</summary>
          <pre>{JSON.stringify(svc.config, null, 2)}</pre>
        </details>
      )}
    </article>
  )
}

export default function Services() {
  const [snap, setSnap] = useState<Awaited<ReturnType<typeof getServices>> | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const s = await getServices()
        if (alive) { setSnap(s); setError(null) }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      }
    }
    load()
    const t = setInterval(load, 15000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  return (
    <div>
      <h1>Services</h1>
      <p className="muted">
        Real availability checks for everything Agentica depends on. Checks
        are lightweight probes — functional AI verification lives in
        Settings → Test connection.
      </p>
      {error && <ErrorCard message={error} recovery="Start the backend, then refresh." />}
      {snap?.error && <ErrorCard message={snap.error.human_message} why={snap.error.why} recovery={snap.error.recovery} />}
      {!snap && !error && <Loading label="Checking services…" />}
      {snap && snap.services.length === 0 && !snap.error && (
        <Empty>No services configured.</Empty>
      )}
      <div className="svc-grid">
        {snap?.services.map((s) => <ServiceCard key={s.id} svc={s} />)}
      </div>
      {snap?.services.some(s => s.required && s.state !== 'running') && (
        <ErrorCard
          message="A required service is not working. Executions that need it will fail until it is fixed."
          recovery="Fix the marked problem above, then check the Dashboard."
        />
      )}
    </div>
  )
}
