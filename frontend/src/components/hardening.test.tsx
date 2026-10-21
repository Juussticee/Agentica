// F2 regression tests: API-derived className values (status / severity /
// state) must map through explicit allowlists — valid values keep their
// exact existing styling, unknown values render neutral base classes with
// no class injection. Plus: HTML-shaped API text must stay inert text.
import '@testing-library/jest-dom/vitest'
import { cleanup, render, waitFor } from '@testing-library/react'
import { screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { StatusBadge } from './ui'
import PlanSection from './PlanSection'
import { StageRow } from '../pages/TasksSequence'
import { ServiceBadge } from '../pages/Services'
import { AgentRow } from '../pages/Agents'
import { ReadinessBadge, FailureGuidance } from '../pages/Settings'
import ExecutionDetail from '../pages/ExecutionDetail'
import type { ExecutionStatus, PlanTask } from '../api/types'
import type { AgentActivity, SequenceStage, ServiceState } from '../api/dashboard'
import type { AiTestResult } from '../api/aiSettings'

afterEach(cleanup)

// Adversarial value shaped for class-attribute breakout plus an inert-payload
// probe: it must appear only as text, never as elements or class content.
const EVIL = '"><img src=x onerror=window.__pwned=1>'

describe('StatusBadge allowlist (F2)', () => {
  const valid = ['queued', 'running', 'completed', 'failed', 'cancelled'] as const
  it.each(valid)('keeps the existing badge--%s styling', (s) => {
    const { container } = render(<StatusBadge status={s} />)
    expect(container.querySelector('span')).toHaveClass('badge', `badge--${s}`)
  })

  it('renders a neutral badge with the raw label for unknown values', () => {
    const { container } = render(<StatusBadge status={EVIL as ExecutionStatus} />)
    const span = container.querySelector('span')!
    expect(span.className).toBe('badge')
    expect(span).toHaveTextContent(EVIL)
    expect(container.querySelector('img')).toBeNull()
  })
})

describe('PlanSection task-status allowlist (F2)', () => {
  const task = (status: string): PlanTask => ({
    task_id: 't1', capability: 'research', depends_on: [], status,
    result: null, result_truncated: false, error: null, block_reason: null,
  })
  const planWith = (status: string) => ({
    goal: null, source: 'llm', fallback_reasons: [], tasks: [task(status)],
  })

  const valid = ['pending', 'running', 'completed', 'failed', 'blocked', 'unsupported'] as const
  it.each(valid)('keeps plan-task--%s styling for valid statuses', (s) => {
    const { container } = render(<PlanSection plan={planWith(s)} />)
    expect(container.querySelector('li')).toHaveClass('plan-task', `plan-task--${s}`)
    expect(container.querySelector('.plan-task__status'))
      .toHaveClass('plan-task__status', `plan-task__status--${s}`)
  })

  it('renders neutral classes and inert text for unknown task statuses', () => {
    const { container } = render(<PlanSection plan={planWith(EVIL)} />)
    const li = container.querySelector('li')!
    expect(li.className).toBe('plan-task')
    expect(li.querySelector('.plan-task__status')!.className).toBe('plan-task__status')
    expect(li).toHaveTextContent(EVIL)
    expect(container.querySelector('img, svg')).toBeNull()
    expect((window as unknown as Record<string, unknown>).__pwned).toBeUndefined()
  })
})

describe('StageRow stage-status allowlist (F2)', () => {
  const stage = (status: string): SequenceStage => ({
    stage: 'supervisor', node: 'supervisor', status: status as SequenceStage['status'],
    at: new Date().toISOString(), event_id: 1, message: 'step message',
  })

  it.each(['running', 'completed', 'failed'] as const)(
    'keeps stage--%s styling for valid statuses', (s) => {
      const { container } = render(<StageRow s={stage(s)} />)
      expect(container.querySelector('li')).toHaveClass('stage', `stage--${s}`)
    })

  it('renders a neutral stage class for unknown statuses', () => {
    const { container } = render(<StageRow s={stage(EVIL)} />)
    expect(container.querySelector('li')!.className).toBe('stage')
    expect(container.querySelector('img')).toBeNull()
  })
})

describe('ServiceBadge state allowlist (F2)', () => {
  const valid = ['running', 'stopped', 'unreachable', 'degraded', 'misconfigured', 'not_configured', 'unknown'] as const
  it.each(valid)('keeps svc-badge--%s styling for valid states', (s) => {
    const { container } = render(<ServiceBadge state={s} />)
    expect(container.querySelector('span')).toHaveClass('badge', 'svc-badge', `svc-badge--${s}`)
  })

  it('renders a neutral badge for unknown states', () => {
    const { container } = render(<ServiceBadge state={EVIL as ServiceState} />)
    const span = container.querySelector('span')!
    expect(span.className).toBe('badge svc-badge')
    expect(span).toHaveTextContent(EVIL)
    expect(container.querySelector('img')).toBeNull()
  })
})

describe('AgentRow status allowlist (F2)', () => {
  const agent = (status: string): AgentActivity => ({
    node: 'books', role: 'Internal knowledge', status: status as AgentActivity['status'],
    execution_id: null, task: null, last_event_at: null, last_event: null, last_error: null,
  })

  it.each(['idle', 'working', 'completed', 'failed'] as const)(
    'keeps agent-row--%s styling for valid statuses', (s) => {
      const { container } = render(<AgentRow a={agent(s)} />)
      expect(container.querySelector('article')).toHaveClass('agent-row', `agent-row--${s}`)
    })

  it('renders a neutral row and the raw label for unknown statuses', () => {
    const { container } = render(<AgentRow a={agent(EVIL)} />)
    expect(container.querySelector('article')!.className).toBe('agent-row')
    expect(container.textContent).toContain(EVIL)
    expect(container.querySelector('img')).toBeNull()
  })
})

describe('Settings readiness states and guidance (F2 + F3)', () => {
  const states = ['ready', 'partial', 'model_missing', 'error', 'not_configured'] as const
  it.each(states)('keeps ai-state--%s styling for valid states', (s) => {
    const { container } = render(<ReadinessBadge state={s} />)
    expect(container.querySelector('div')).toHaveClass('ai-state', `ai-state--${s}`)
  })

  it('renders a neutral panel with safe copy for unknown states', () => {
    const { container } = render(
      <ReadinessBadge state={EVIL as AiTestResult['state']} />,
    )
    const div = container.querySelector('div')!
    expect(div.className).toBe('ai-state')
    expect(screen.getByText('Unknown state')).toBeInTheDocument()
    expect(container.querySelector('img')).toBeNull()
  })

  it('prototype-like category keys get the neutral fallback guidance', () => {
    for (const category of ['__proto__', 'constructor', 'toString']) {
      cleanup()
      const { container } = render(
        <FailureGuidance result={{ state: 'error', phases: {}, category }} />,
      )
      expect(container).toHaveTextContent(
        'The check did not finish successfully. See the step results.',
      )
      // The old bug rendered Object.prototype-derived junk; never that.
      expect(container.textContent).not.toContain('function Object')
    }
  })

  it('documented categories keep their existing guidance', () => {
    render(<FailureGuidance result={{ state: 'error', phases: {}, category: 'timeout' }} />)
    expect(screen.getByText('The AI service did not respond in time.')).toBeInTheDocument()
  })
})

describe('ExecutionDetail: inert HTML and severity allowlist (F2)', () => {
  const HTML_IMG = '<img src=x onerror="window.__pwned=1">'
  const HTML_Q = '<b>bold</b><script>window.__pwned=1</script>'
  const HTML_SVG = '<svg onload=alert(1)>'
  let eventSeverity = 'catastrophe'

  beforeEach(() => {
    eventSeverity = 'catastrophe'
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/events')) {
        return new Response(JSON.stringify({
          execution_id: 'x1', after: 0, count: 1,
          events: [{
            event_id: 1, execution_id: 'x1', timestamp: new Date().toISOString(),
            type: 'agent_message', message: HTML_IMG, severity: eventSeverity, payload: {},
          }],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      }
      return new Response(JSON.stringify({
        id: 'x1', question: HTML_Q, status: 'completed', route: null,
        answer: `${HTML_IMG} plain answer`, qa_flags: [HTML_SVG],
        created_at: new Date().toISOString(), error: null, plan: null,
      }), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders HTML-shaped payloads as literal text (no elements, no side effects)', async () => {
    const { container } = render(<ExecutionDetail id="x1" />)
    await waitFor(() => expect(container.textContent).toContain('plain answer'))
    expect(container.querySelector('img, svg, script, b, iframe')).toBeNull()
    expect(container.textContent).toContain(HTML_IMG)
    expect(container.textContent).toContain(HTML_Q)
    expect(container.textContent).toContain(HTML_SVG)
    expect((window as unknown as Record<string, unknown>).__pwned).toBeUndefined()
  })

  it('uses the neutral timeline class for unknown severities', async () => {
    const { container } = render(<ExecutionDetail id="x1" />)
    await waitFor(() => expect(container.querySelectorAll('.timeline__item')).toHaveLength(1))
    expect(container.querySelector('.timeline__item')!.className).toBe('timeline__item')
  })

  it('keeps the existing severity styling for valid values', async () => {
    eventSeverity = 'warning'
    const { container } = render(<ExecutionDetail id="x1" />)
    await waitFor(() => expect(container.querySelector('.timeline__item'))
      .toHaveClass('timeline__item', 'timeline__item--warning'))
  })
})
