import { useCallback, useEffect, useState } from 'react'
import type { ApprovalRequest, ApprovalResumeResponse } from '../api/types'
import { decideApproval, getApproval, getExecution, listApprovals, resumeApproval } from '../api/client'
import { Button, ErrorCard, Loading, StatusBadge } from '../components/ui'

type UiError = Error & { why?: string; recovery?: string }
type ApprovalOutcome = { approval: ApprovalRequest; result?: ApprovalResumeResponse }

function errorShape(error: unknown): UiError {
  return error instanceof Error ? error as UiError : new Error(String(error))
}

function expired(approval: ApprovalRequest): boolean {
  const expiry = Date.parse(approval.expires_at)
  return !Number.isFinite(expiry) || expiry <= Date.now()
}

const OPERATION_TITLES: Record<string, string> = {
  research: 'External web research',
  workspace_patch: 'Apply a workspace patch',
}

const AGENT_TITLES: Record<string, string> = {
  research: 'Web Research',
  coding: 'Workspace Coding',
}

const OPERATION_REASONS: Record<string, string> = {
  research:
    'This request did not explicitly authorize external research, so it was ' +
    'held before any search or page fetch.',
  workspace_patch:
    'The patch was prepared and validated offline but NOT applied: writing ' +
    'to the workspace requires this decision, bound to the exact files and ' +
    'their pre-change hashes.',
}

export default function Approvals() {
  const [items, setItems] = useState<ApprovalRequest[] | null>(null)
  const [error, setError] = useState<UiError | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [outcomes, setOutcomes] = useState<Record<string, ApprovalOutcome>>({})
  const [notice, setNotice] = useState<string | null>(null)
  const [resultBusyId, setResultBusyId] = useState<string | null>(null)
  const [selectedStatus, setSelectedStatus] = useState<'pending' | 'approved' | 'rejected' | 'expired'>('pending')

  const refresh = useCallback(async () => {
    try {
      const queue = await listApprovals(selectedStatus)
      setItems(queue.approvals)
      setError(null)
      return queue.approvals
    } catch (err) {
      setError(errorShape(err))
      return null
    }
  }, [selectedStatus])

  useEffect(() => {
    let alive = true
    const load = async () => {
      const queue = await listApprovals(selectedStatus).catch((err: unknown) => {
        if (alive) setError(errorShape(err))
        return null
      })
      if (alive && queue) {
        setItems(queue.approvals)
        setError(null)
      }
    }
    void load()
    const timer = window.setInterval(() => { void load() }, 5000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [selectedStatus])

  const refreshOne = useCallback(async (approvalId: string) => {
    try {
      const detail = await getApproval(approvalId)
      const queue = await listApprovals(selectedStatus)
      setItems(queue.approvals)
      setError(null)
      return detail.approval
    } catch (err) {
      setError(errorShape(err))
      return null
    }
  }, [selectedStatus])

  async function decide(approval: ApprovalRequest, decision: 'approve' | 'reject') {
    if (busyId) return
    setBusyId(approval.approval_id)
    setError(null)
    setNotice(null)
    try {
      const response = await decideApproval(approval.approval_id, decision)
      setOutcomes((current) => ({
        ...current,
        [approval.approval_id]: { approval: response.approval },
      }))
      setNotice(decision === 'approve'
        ? 'Decision saved. Resolving this request…'
        : 'Rejection saved. Closing this request…')
      // The API separates decision from execution; explicitly request its
      // documented resume endpoint so the result shown reflects real work.
      const result = await resumeApproval(approval.approval_id)
      const latest = await refreshOne(approval.approval_id)
      setOutcomes((current) => ({
        ...current,
        [approval.approval_id]: {
          approval: latest ?? response.approval,
          result,
        },
      }))
      setNotice(result.outcome === 'executed' || result.outcome === 'recovered'
        ? result.execution?.status === 'failed'
          ? 'The approved operation ran, but the execution still has unfinished planned tasks — its status is shown below.'
          : 'The approved operation finished. The execution result is shown below.'
        : result.outcome === 'rejected'
          // Name the operation kind: a research approval is about external
          // calls, a workspace patch is about file writes. Both say plainly
          // that nothing ran.
          ? approval.operation === 'workspace_patch'
            ? 'Request rejected. No files were modified.'
            : 'Request rejected. No external research was performed.'
          : `Request was not executed: ${result.reason ?? 'the approval could not be used'}.`)
    } catch (err) {
      const failure = errorShape(err)
      const status = (failure as UiError & { status?: number }).status
      const latest = await refreshOne(approval.approval_id)
      if (latest) {
        setOutcomes((current) => ({
          ...current,
          [approval.approval_id]: { approval: latest },
        }))
      }
      if (status === 410) {
        setNotice(
          approval.operation === 'workspace_patch'
            ? 'This request expired before it could be decided. No files were modified.'
            : 'This request expired before it could be decided. No research was executed.',
        )
      } else if (status === 409 && latest && latest.status !== 'pending') {
        setNotice(`This request was already ${latest.status}. No earlier decision was overwritten.`)
      }
      setError(failure)
    } finally {
      setBusyId(null)
      await refresh()
    }
  }

  const pending = items ?? []

  async function showRecordedExecution(approval: ApprovalRequest) {
    if (!approval.execution_id || resultBusyId) return
    setResultBusyId(approval.approval_id)
    setError(null)
    try {
      const execution = await getExecution(approval.execution_id)
      setOutcomes((current) => ({
        ...current,
        [approval.approval_id]: {
          approval,
          result: {
            outcome: approval.status === 'approved' ? 'executed' : 'rejected',
            approval_id: approval.approval_id,
            answer: execution.answer ?? undefined,
            execution,
          },
        },
      }))
    } catch (err) {
      setError(errorShape(err))
    } finally {
      setResultBusyId(null)
    }
  }

  return (
    <div>
      <div className="approval-page-head">
        <div>
          <h1>Approval Center</h1>
          <p className="muted">
            Review one-time requests for external web research or for applying a
            workspace patch. Approval does not grant future permissions; the bound
            execution status is shown separately.
          </p>
        </div>
        <Button type="button" variant="ghost" onClick={() => { void refresh() }} disabled={busyId !== null}>
          Refresh queue
        </Button>
      </div>

      <div className="form-row approval-filters" role="group" aria-label="Approval status">
        {(['pending', 'approved', 'rejected', 'expired'] as const).map((status) => (
          <Button
            key={status}
            type="button"
            variant={selectedStatus === status ? 'primary' : 'ghost'}
            aria-pressed={selectedStatus === status}
            onClick={() => { setItems(null); setSelectedStatus(status) }}
            disabled={busyId !== null}
          >
            {status === 'pending' ? 'Pending' : status[0].toUpperCase() + status.slice(1)}
          </Button>
        ))}
      </div>

      <section className="card card--warn" aria-label="Local security notice">
        <strong>Local-only control</strong>
        <p className="muted">
          Approval actions are available only through Agentica's local Control Center.
          This app has no user authentication and must not be exposed to a network.
        </p>
      </section>

      {error && <ErrorCard message={error.message} why={error.why} recovery={error.recovery ?? 'Refresh the queue and verify the local API is available.'} />}
      {notice && <p className="approval-notice" role="status">{notice}</p>}
      {items === null && !error && <Loading label="Loading pending approvals…" />}
      {items !== null && pending.length === 0 && (
        <section className="card">
          <h2>{selectedStatus === 'pending' ? 'No pending approvals' : `No ${selectedStatus} approvals`}</h2>
          <p className="muted">{selectedStatus === 'pending'
            ? 'Requests waiting for a decision will appear here.'
            : 'Approval decisions are recorded separately from whether the execution completed.'}</p>
        </section>
      )}

      {pending.map((approval) => {
        const isBusy = busyId === approval.approval_id
        const isExpired = expired(approval)
        return (
          <article className="card approval-card" key={approval.approval_id}>
            <div className="approval-card__head">
              <div>
                <h2>{OPERATION_TITLES[approval.operation] ?? approval.operation}</h2>
                <p className="muted">
                  Request {approval.approval_id} ·{' '}
                  {approval.execution_id ? (
                    <a href={`#/executions/${encodeURIComponent(approval.execution_id)}`}>
                      Execution #{approval.execution_id}
                    </a>
                  ) : 'not linked to a run'}
                  {' · '}Requested by{' '}
                  {approval.requesting_agent
                    ? (AGENT_TITLES[approval.requesting_agent] ?? approval.requesting_agent)
                    : 'unmapped operation'}
                </p>
              </div>
              <span className={`badge ${isExpired ? 'badge--failed' : 'badge--queued'}`}>
                {isExpired ? 'Expired' : 'Pending decision'}
              </span>
            </div>
            <p><strong>Proposed action:</strong> {approval.description}</p>
            <p className="muted"><strong>Why:</strong> {approval.payload?.books_refusal
              ? 'Local documents did not provide a sufficiently grounded answer; external sources are needed for the proposed fallback.'
              : OPERATION_REASONS[approval.operation]
                ?? 'This operation needs an explicit decision before it runs.'}</p>
            {approval.payload?.question && (
              <details className="approval-question">
                <summary>{approval.operation === 'workspace_patch'
                  ? 'Request this patch answers'
                  : 'Research question'}</summary>
                <p>{approval.payload.question}</p>
              </details>
            )}
            {approval.operation === 'workspace_patch'
              && (approval.payload?.files?.length ?? 0) > 0 && (
              <details className="approval-question">
                <summary>Files in the reviewed patch</summary>
                <ul className="plain-list">
                  {approval.payload?.files?.map((file) => (
                    <li key={file.path}><code>{file.path}</code></li>
                  ))}
                </ul>
                <p className="muted">
                  Applying re-checks each file against the hash recorded when
                  the patch was prepared; a changed file means nothing is
                  written.
                </p>
              </details>
            )}
            <p className="muted">Expires {new Date(approval.expires_at).toLocaleString()}</p>
            {selectedStatus !== 'pending' ? (
              <>
                <p><strong>Decision:</strong> {approval.decision ?? approval.status}
                  {approval.decided_at ? ` · ${new Date(approval.decided_at).toLocaleString()}` : ''}
                </p>
                {approval.status === 'approved' && !approval.executed_at && (
                  <p className="muted">Approval is saved; the external action has not yet been recorded as executed.</p>
                )}
                <Button type="button" variant="ghost" onClick={() => void showRecordedExecution(approval)} disabled={resultBusyId !== null}>
                  {resultBusyId === approval.approval_id ? 'Loading result…' : 'Load execution result'}
                </Button>
              </>
            ) : isExpired ? (
              <p className="approval-expired" role="status">This request has expired. Nothing was executed; submit the request again for a fresh approval.</p>
            ) : (
              <div className="form-row">
                <Button type="button" onClick={() => void decide(approval, 'approve')} disabled={busyId !== null}>
                  {isBusy ? 'Processing…' : 'Approve and run'}
                </Button>
                <Button type="button" variant="ghost" onClick={() => void decide(approval, 'reject')} disabled={busyId !== null}>
                  Reject
                </Button>
              </div>
            )}
          </article>
        )
      })}

      {Object.values(outcomes).map(({ approval, result }) => (
        <section className="card approval-result" key={`result-${approval.approval_id}`}>
          <div className="approval-card__head">
            <div>
              <h2>Decision recorded</h2>
              <p className="muted">Request {approval.approval_id} · {approval.description}</p>
            </div>
            <span className={`badge ${approval.status === 'approved' ? 'badge--completed' : approval.status === 'rejected' ? 'badge--cancelled' : 'badge--failed'}`}>
              {approval.status === 'approved' ? 'Approved' : approval.status === 'rejected' ? 'Rejected' : approval.status}
            </span>
          </div>
          {!result ? approval.status === 'expired' ? (
            <p className="approval-expired">Expired — no external operation ran.</p>
          ) : approval.status === 'approved' ? (
            <p className="muted">Approval is saved; the external action has not yet been recorded as executed. Load the execution result separately.</p>
          ) : approval.status === 'rejected' ? (
            <p className="muted">Decision rejected. The external operation did not run; load the execution to see its final status and local result.</p>
          ) : (
            <Loading label="Waiting for execution result…" />
          ) : (
            <>
              <p><strong>Execution outcome:</strong> {result.outcome}</p>
              {result.execution && <p><strong>Execution status:</strong> <StatusBadge status={result.execution.status} /></p>}
              {result.execution?.error && (
                <div className="approval-execution-error" role="status">
                  <strong>{result.execution.error.human_message}</strong>
                  {result.execution.error.why && <p>{result.execution.error.why}</p>}
                  {result.execution.error.recovery && <p>{result.execution.error.recovery}</p>}
                </div>
              )}
              {result.answer && <div className="answer">{result.answer}</div>}
              {result.reason && <p className="svc-reason">{result.reason}</p>}
              {result.execution?.id && <a href={`#/executions/${encodeURIComponent(result.execution.id)}`}>Open execution #{result.execution.id}</a>}
            </>
          )}
        </section>
      ))}
    </div>
  )
}
