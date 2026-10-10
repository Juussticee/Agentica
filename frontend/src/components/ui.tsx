import type { ExecutionStatus } from '../api/types'

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return <div className="state loading" role="status">{label}</div>
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="state empty">{children}</div>
}

export function ErrorCard({
  message,
  why,
  recovery,
  technical,
  onDismiss,
}: {
  message: string
  why?: string
  recovery?: string
  technical?: string
  onDismiss?: () => void
}) {
  return (
    <div className="error-card" role="alert">
      <div className="error-card__msg">{message}</div>
      {why && <div className="error-card__why"><strong>Why:</strong> {why}</div>}
      {recovery && (
        <div className="error-card__recovery"><strong>You can:</strong> {recovery}</div>
      )}
      {technical && (
        <details className="error-card__details">
          <summary>View technical details</summary>
          <pre>{technical}</pre>
        </details>
      )}
      {onDismiss && (
        <button type="button" className="btn btn--ghost" onClick={onDismiss}>
          Dismiss
        </button>
      )}
    </div>
  )
}

const STATUS_LABELS: Record<ExecutionStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export function StatusBadge({ status }: { status: ExecutionStatus }) {
  return <span className={`badge badge--${status}`}>{STATUS_LABELS[status] ?? status}</span>
}

export function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'ghost' }) {
  const { variant = 'primary', className = '', ...rest } = props
  return <button {...rest} className={`btn btn--${variant} ${className}`} />
}
