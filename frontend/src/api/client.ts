import type {
  ApprovalDecisionResponse,
  ApprovalQueue,
  ApprovalResumeResponse,
  ApprovalRequest,
  Execution,
  ExecutionEvent,
  Health,
} from './types'

const BASE = '/api/v1'

async function handleError(res: Response): Promise<never> {
  let detail: unknown = null
  try {
    detail = await res.json()
  } catch {
    /* body not JSON */
  }
  const d = detail as { detail?: { human_message?: string; why?: string; recovery?: string } }
  const err = d?.detail
  const error = new Error(
    err?.human_message ?? `Request failed (${res.status})`,
  ) as Error & { why?: string; recovery?: string; status?: number }
  error.why = err?.why
  error.recovery = err?.recovery
  error.status = res.status
  throw error
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) await handleError(res)
  return res.json()
}

export async function listExecutions(): Promise<{ executions: Execution[]; count: number }> {
  const res = await fetch(`${BASE}/executions`)
  if (!res.ok) await handleError(res)
  return res.json()
}

export async function getExecution(id: string): Promise<Execution> {
  const res = await fetch(`${BASE}/executions/${encodeURIComponent(id)}`)
  if (!res.ok) await handleError(res)
  return res.json()
}

export async function getEvents(id: string, after = 0): Promise<{
  execution_id: string
  after: number
  events: ExecutionEvent[]
  count: number
}> {
  const res = await fetch(`${BASE}/executions/${encodeURIComponent(id)}/events?after=${after}`)
  if (!res.ok) await handleError(res)
  return res.json()
}

export async function listApprovals(status = 'pending'): Promise<ApprovalQueue> {
  const res = await fetch(`${BASE}/approvals?status=${encodeURIComponent(status)}`)
  if (!res.ok) await handleError(res)
  return res.json()
}

export async function getApproval(approvalId: string): Promise<{ approval: ApprovalRequest }> {
  const res = await fetch(`${BASE}/approvals/${encodeURIComponent(approvalId)}`)
  if (!res.ok) await handleError(res)
  return res.json()
}

export async function decideApproval(
  approvalId: string,
  decision: 'approve' | 'reject',
): Promise<ApprovalDecisionResponse> {
  const res = await fetch(`${BASE}/approvals/${encodeURIComponent(approvalId)}/${decision}`, {
    method: 'POST',
  })
  if (!res.ok) await handleError(res)
  return res.json()
}

export async function resumeApproval(approvalId: string): Promise<ApprovalResumeResponse> {
  const res = await fetch(`${BASE}/approvals/${encodeURIComponent(approvalId)}/resume`, {
    method: 'POST',
  })
  if (!res.ok) await handleError(res)
  return res.json()
}

export async function createExecution(question: string): Promise<Execution> {
  const res = await fetch(`${BASE}/executions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) await handleError(res)
  return res.json()
}
