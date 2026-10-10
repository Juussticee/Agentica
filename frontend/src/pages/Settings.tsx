import { useHealth } from '../api/health'
import { AiReadinessPanel } from '../components/AiReadiness'

export default function Settings() {
  const { health, error } = useHealth(15000)

  return (
    <div>
      <h1>Settings</h1>
      <p className="muted">
        AI provider configuration (Gemini / Ollama, connection testing, saved
        profiles) is under reconstruction — the previous AI Settings screen is
        being restored as part of the current recovery work.
      </p>

      <section className="card">
        <h2>Current configuration (read-only)</h2>
        {error ? (
          <p className="muted">Backend unavailable: {error}</p>
        ) : health?.ai ? (
          <>
            <ul className="health-list">
              <li>Provider: <strong>{health.ai.provider}</strong></li>
              <li>Readiness: <strong>{health.ai.state}</strong></li>
              <li>Tested when connecting: <strong>{health.ai.what_will_be_tested}</strong></li>
            </ul>
            <AiReadinessPanel ai={health.ai} />
          </>
        ) : (
          <p className="muted">Loading…</p>
        )}
      </section>

      <section className="card card--warn">
        <h2>Planned, not implemented yet</h2>
        <ul className="health-list muted">
          <li>Provider switching between Gemini and Ollama from the UI</li>
          <li>Bounded connection test (auth → model availability → tiny generation)</li>
          <li>Saved settings with session override (Settings → AI connection)</li>
        </ul>
      </section>
    </div>
  )
}
