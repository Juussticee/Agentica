import type { AiReadiness as AiReadinessType } from '../api/types'

export function AiReadinessPanel({ ai }: { ai: AiReadinessType | null }) {
  if (!ai) {
    return (
      <section className="card">
        <h2>AI connection</h2>
        <p className="muted">Status unavailable.</p>
      </section>
    )
  }
  const ok = ai.state === 'ready'
  return (
    <section className={`card ai-panel ${ok ? 'ai-panel--ok' : 'ai-panel--warn'}`}>
      <h2>AI connection</h2>
      <p>
        <strong>{ok ? 'Configured' : 'Not configured'}</strong> — {ai.what_will_be_tested}
      </p>
      <p className="muted">{ai.state_hint}</p>
      {/* Hash routing: /settings alone would reload the page and return to
          the dashboard, since the router reads only the hash. */}
      <a href="#/settings" className="btn btn--ghost">Open Settings</a>
    </section>
  )
}
