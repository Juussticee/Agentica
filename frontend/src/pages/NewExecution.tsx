import { useState } from 'react'
import { createExecution } from '../api/client'
import { Button, ErrorCard } from '../components/ui'

export default function NewExecution({ onNavigate }: { onNavigate: (to: string) => void }) {
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<{ message: string; why?: string; recovery?: string } | null>(null)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (question.trim().length < 3) {
      setError({ message: 'Please describe the task (at least 3 characters).' })
      return
    }
    setBusy(true)
    setError(null)
    try {
      const created = await createExecution(question.trim())
      onNavigate(`executions/${encodeURIComponent(created.id)}`)
    } catch (err) {
      const e2 = err as Error & { why?: string; recovery?: string }
      setError({ message: e2.message, why: e2.why, recovery: e2.recovery })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <h1>New Execution</h1>
      <p className="muted">
        Describe the task in your own words. Agentica will route it to the right
        capability (web research, internal books, business data, or a feasibility
        study) and show live progress.
      </p>
      <form onSubmit={submit} className="card form">
        <label htmlFor="question">Task</label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. Find what my beekeeping guide says about winter feeding."
          rows={4}
          disabled={busy}
        />
        <div className="form-row">
          <Button type="submit" disabled={busy || question.trim().length < 3}>
            {busy ? 'Starting…' : 'Start execution'}
          </Button>
          <Button type="button" variant="ghost" onClick={() => setQuestion('')} disabled={busy}>
            Clear
          </Button>
        </div>
        {error && (
          <ErrorCard message={error.message} why={error.why} recovery={error.recovery} onDismiss={() => setError(null)} />
        )}
      </form>
    </div>
  )
}
