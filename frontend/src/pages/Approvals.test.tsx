import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Approvals from './Approvals'

const pending = {
  approval_id: 'approve123456',
  execution_id: 'exec12345678',
  operation: 'research',
  description: 'Web search (up to 8 results) for: latest AI news',
  status: 'pending',
  decision: null,
  created_at: '2026-10-28T10:00:00+00:00',
  expires_at: '2099-10-28T10:00:00+00:00',
  decided_at: null,
  executed_at: null,
  payload: { question: 'Search the latest AI news', route: 'research' },
}

let requestedDecision: 'approve' | 'reject' | null = null

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

beforeEach(() => {
  requestedDecision = null
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.includes('/approvals?status=pending')) return json({
      approvals: requestedDecision ? [] : [pending],
      count: requestedDecision ? 0 : 1,
      status: 'pending',
    })
    if (url.includes('/approvals?status=approved')) return json({
      approvals: [{ ...pending, status: 'approved', decision: 'approved', executed_at: null }],
      count: 1,
      status: 'approved',
    })
    if (url.endsWith('/approve')) {
      requestedDecision = 'approve'
      return json({
        approval: { ...pending, status: 'approved', decision: 'approved' },
        decision: { approval_id: pending.approval_id, outcome: 'approved', decided_at: new Date().toISOString(), already_decided: false },
        resume: `/api/v1/approvals/${pending.approval_id}/resume`,
      })
    }
    if (url.endsWith('/reject')) {
      requestedDecision = 'reject'
      return json({
        approval: { ...pending, status: 'rejected', decision: 'rejected' },
        decision: { approval_id: pending.approval_id, outcome: 'rejected', decided_at: new Date().toISOString(), already_decided: false },
        resume: `/api/v1/approvals/${pending.approval_id}/resume`,
      })
    }
    if (url.endsWith('/resume')) return json({
      outcome: requestedDecision === 'approve' ? 'executed' : 'rejected',
      approval_id: pending.approval_id,
      answer: requestedDecision === 'approve' ? 'Researched answer with sources.' : undefined,
      execution: { id: pending.execution_id, question: pending.payload.question,
        status: requestedDecision === 'approve' ? 'completed' : 'failed',
        error: requestedDecision === 'reject' ? {
          category: 'approval_rejected',
          human_message: 'The execution stopped because research approval was rejected.',
          why: 'The proposed external research was not authorized.',
          recovery: 'Retry with a new research request if you want to reconsider.',
          recoverable: true,
          severity: 'warning',
        } : null,
        created_at: new Date().toISOString(),
        answer: requestedDecision === 'approve' ? 'Researched answer with sources.' : 'Local answer only' },
    })
    if (url.endsWith(`/approvals/${pending.approval_id}`)) return json({
      approval: { ...pending, status: requestedDecision ?? 'pending', decision: requestedDecision },
    })
    return json({ detail: { human_message: 'Unexpected request' } }, 500)
  }))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Approval Center', () => {
  it('shows proposed action and reason, then approves and shows real execution result', async () => {
    const fetchMock = vi.mocked(fetch)
    render(<Approvals />)
    expect(await screen.findByText(/Web search \(up to 8 results\)/)).toBeInTheDocument()
    expect(screen.getByText(/did not explicitly authorize external research/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Approve and run' }))
    expect(await screen.findByText('Researched answer with sources.')).toBeInTheDocument()
    expect(screen.getByText('Execution status:')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open execution #exec12345678' })).toHaveAttribute(
      'href', '#/executions/exec12345678',
    )
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/approvals/approve123456/approve', { method: 'POST' })
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/approvals/approve123456/resume', { method: 'POST' })
  })

  it('rejects then reports that external research did not run', async () => {
    render(<Approvals />)
    await screen.findByRole('button', { name: 'Reject' })
    fireEvent.click(screen.getByRole('button', { name: 'Reject' }))
    expect(await screen.findByText('Request rejected. No external research was performed.')).toBeInTheDocument()
    expect(screen.getByText('Execution outcome:').parentElement).toHaveTextContent('rejected')
    expect(screen.getByText('Rejected', { exact: true })).toBeInTheDocument()
    expect(screen.getByText('Execution status:').parentElement).toHaveTextContent('Failed')
    expect(screen.getByText(/execution stopped because research approval was rejected/i)).toBeInTheDocument()
  })

  it('does not duplicate requests while a decision is in progress', async () => {
    let resolveDecision!: (response: Response) => void
    const fetchMock = vi.mocked(fetch)
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/approvals?status=pending')) return json({ approvals: requestedDecision ? [] : [pending], count: requestedDecision ? 0 : 1, status: 'pending' })
      if (url.endsWith('/approve')) return await new Promise<Response>((resolve) => { resolveDecision = resolve })
      if (url.endsWith('/resume')) return json({ outcome: 'executed', answer: 'done' })
      if (url.endsWith(`/approvals/${pending.approval_id}`)) return json({
        approval: { ...pending, status: 'approved', decision: 'approved' },
      })
      return json({})
    })
    render(<Approvals />)
    await screen.findByRole('button', { name: 'Approve and run' })
    fireEvent.click(screen.getByRole('button', { name: 'Approve and run' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Processing…' })).toBeDisabled())
    fireEvent.click(screen.getByRole('button', { name: 'Reject' }))
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith('/approve'))).toHaveLength(1)
    resolveDecision(json({
      approval: { ...pending, status: 'approved', decision: 'approved' },
      decision: { approval_id: pending.approval_id, outcome: 'approved', decided_at: null, already_decided: false },
      resume: `/api/v1/approvals/${pending.approval_id}/resume`,
    }))
    await screen.findByText('done')
  })

  it('shows decided requests in their history filter and keeps execution status separate', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/approvals?status=approved')) return json({
        approvals: [{ ...pending, status: 'approved', decision: 'approved', executed_at: null }],
        count: 1,
        status: 'approved',
      })
      if (url.includes(`/approvals/${pending.approval_id}`)) return json({
        approval: { ...pending, status: 'approved', decision: 'approved' },
      })
      if (url.includes('/approvals?status=pending')) return json({ approvals: [], count: 0, status: 'pending' })
      if (url.includes('/executions/')) return json({
        id: pending.execution_id, question: pending.payload.question,
        status: 'failed', answer: 'The planned operation failed.',
        created_at: new Date().toISOString(),
      })
      return json({})
    })
    render(<Approvals />)
    fireEvent.click(await screen.findByRole('button', { name: 'Approved' }))
    expect(await screen.findByText(/Approval is saved; the external action has not yet been recorded as executed/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Load execution result' }))
    expect(await screen.findByText('Execution status:')).toBeInTheDocument()
    expect(screen.getByText('Failed', { exact: true })).toBeInTheDocument()
  })

  it('shows persisted approvals with execution still waiting as incomplete', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/approvals?status=approved')) return json({
        approvals: [{ ...pending, status: 'approved', decision: 'approved', executed_at: null }],
        count: 1,
        status: 'approved',
      })
      if (url.includes('/approvals?status=pending')) return json({ approvals: [], count: 0, status: 'pending' })
      if (url.includes('/executions/')) return json({
        id: pending.execution_id, question: pending.payload.question,
        status: 'waiting_for_approval', answer: 'Waiting for approval',
        created_at: new Date().toISOString(),
      })
      return json({})
    })
    render(<Approvals />)
    fireEvent.click(await screen.findByRole('button', { name: 'Approved' }))
    await screen.findByText(/Approval is saved; the external action has not yet been recorded as executed/)
    fireEvent.click(screen.getByRole('button', { name: 'Load execution result' }))
    expect(await screen.findByText('Execution status:')).toBeInTheDocument()
    expect(screen.getByText('Execution status:').parentElement).toHaveTextContent('Waiting for approval')
  })

  it('shows expired requests as read-only', async () => {
    const past = { ...pending, expires_at: '2000-01-01T00:00:00Z' }
    vi.mocked(fetch).mockImplementation(async () => json({ approvals: [past], count: 1, status: 'pending' }))
    render(<Approvals />)
    expect(await screen.findByText('This request has expired. Nothing was executed; submit the request again for a fresh approval.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Approve and run' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Reject' })).not.toBeInTheDocument()
  })

  it('shows queue fetch errors honestly', async () => {
    vi.mocked(fetch).mockImplementation(async () => json({ detail: {
      human_message: 'Approval queue is unavailable.',
      why: 'Storage could not be read.', recovery: 'Retry later.',
    } }, 503))
    render(<Approvals />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Approval queue is unavailable.')
    expect(screen.getByRole('alert')).toHaveTextContent('Storage could not be read.')
  })
})
