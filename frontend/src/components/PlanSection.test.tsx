// GUI-12 tests: planner visualization rendering (deterministic, mock data
// only — no network, no LLM, no live services). Mirrors the exact plan
// projection the backend serializer emits.
import '@testing-library/jest-dom/vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import PlanSection, { PlanSourceNote } from './PlanSection'
import type { ExecutionPlan } from '../api/types'

afterEach(cleanup)

const task = (over: Partial<ExecutionPlan['tasks'][number]> = {}) => ({
  task_id: 't1',
  capability: 'research',
  capability_label: 'Web research',
  objective: 'Gather web research',
  depends_on: [],
  required_inputs: ['question'],
  expected_output: 'Web research summary',
  done_when: 'Summary available',
  status: 'pending',
  result: null,
  result_truncated: false,
  error: null,
  block_reason: null,
  ...over,
})

const plan = (over: Partial<ExecutionPlan> = {}): ExecutionPlan => ({
  goal: 'Evaluate feasibility of an urban beekeeping venture',
  source: 'llm',
  fallback_reasons: [],
  tasks: [
    task(),
    task({
      task_id: 't2', capability: 'books',
      capability_label: 'Internal knowledge (library)',
      objective: 'Check the library', depends_on: ['t1'],
    }),
    task({
      task_id: 't3', capability: 'synthesis',
      capability_label: 'Combine results into the answer',
      objective: 'Assess feasibility', depends_on: ['t1', 't2'],
    }),
  ],
  ...over,
})

describe('PlanSection rendering', () => {
  it('renders multi-task plans in plan order with objectives', () => {
    render(<PlanSection plan={plan()} />)
    const items = screen.getAllByRole('listitem')
    expect(items).toHaveLength(3)
    expect(items[0]).toHaveTextContent('t1')
    expect(items[1]).toHaveTextContent('t2')
    expect(items[2]).toHaveTextContent('t3')
    expect(screen.getAllByText('Gather web research').length).toBeGreaterThan(0)
  })

  it('shows task statuses using the backend statuses only', () => {
    render(<PlanSection plan={plan()} />)
    const items = screen.getAllByRole('listitem')
    expect(within(items[0]).getByText('pending')).toBeInTheDocument()
    const running = plan({
      tasks: [task({ status: 'running' })],
    })
    cleanup()
    render(<PlanSection plan={running} />)
    expect(screen.getByText('running')).toBeInTheDocument()
  })

  it('renders dependencies only where the plan has them (no fabricated edges)', () => {
    render(<PlanSection plan={plan()} />)
    const items = screen.getAllByRole('listitem')
    // t1 has no deps: no "after ..." text anywhere in its row.
    expect(items[0].querySelector('.plan-task__deps')).toBeNull()
    // t2 depends on t1; t3 on t1 + t2 — exactly as recorded.
    expect(items[1].querySelector('.plan-task__deps')).toHaveTextContent(
      'after t1',
    )
    expect(items[2].querySelector('.plan-task__deps')).toHaveTextContent(
      'after t1, t2',
    )
  })

  it('completed tasks show the real result summary; failed/blocked show reasons', () => {
    const mixed = plan({
      tasks: [
        task({ status: 'completed', result: 'Found 14 sources' }),
        task({
          task_id: 't2', status: 'failed', result: null,
          error: 'RuntimeError: provider down',
        }),
        task({
          task_id: 't3', status: 'blocked', result: null,
          block_reason: 'a dependency did not complete successfully',
        }),
      ],
    })
    render(<PlanSection plan={mixed} />)
    expect(screen.getByText(/Found 14 sources/)).toBeInTheDocument()
    expect(screen.getByText(/RuntimeError: provider down/)).toBeInTheDocument()
    expect(screen.getByText(/a dependency did not complete successfully/))
      .toBeInTheDocument()
  })

  it('result truncation is surfaced honestly', () => {
    render(
      <PlanSection plan={plan({
        tasks: [task({ status: 'completed', result: 'x', result_truncated: true })],
      })} />,
    )
    expect(screen.getByText(/truncated/)).toBeInTheDocument()
  })

  it('flags the built-in fallback plan and its recorded reasons', () => {
    render(
      <PlanSection plan={plan({
        source: 'fallback',
        fallback_reasons: ['planner LLM returned no parsable plan'],
      })} />,
    )
    const note = screen.getByTestId('plan-source-note')
    expect(note).toHaveTextContent('Built-in fallback plan')
    expect(note).toHaveTextContent('planner LLM returned no parsable plan')
  })

  it('missing, empty, or malformed plan data renders nothing at all', () => {
    const { container: c1 } = render(<PlanSection plan={null} />)
    expect(c1).toBeEmptyDOMElement()
    cleanup()
    const { container: c2 } = render(<PlanSection plan={undefined} />)
    expect(c2).toBeEmptyDOMElement()
    cleanup()
    const { container: c3 } = render(
      <PlanSection plan={plan({ tasks: [] })} />,
    )
    expect(c3).toBeEmptyDOMElement()
  })

  it('renders as its own card, visually distinct from the event log', () => {
    render(<PlanSection plan={plan()} />)
    const card = screen.getByLabelText('Execution plan')
    expect(card).toBeInTheDocument()
    expect(card.querySelector('h2')).toHaveTextContent(
      'Plan (what the agent intended)',
    )
  })

  it('uses responsive layout classes on the task rows', () => {
    render(<PlanSection plan={plan()} />)
    // flex-wrap heads let long capability text wrap at mobile widths; the
    // matching @media (max-width: 640px) block exists in index.css.
    const row = screen.getAllByRole('listitem')[0]
    expect(row.querySelector('.plan-task__head')).toBeInTheDocument()
  })
})

describe('PlanSourceNote', () => {
  it('says nothing about planner origin for llm plans', () => {
    const { container } = render(<PlanSourceNote plan={plan({ source: 'llm' })} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('labels default (route-derived) plans without fabricating an LLM call', () => {
    render(<PlanSourceNote plan={plan({ source: 'default', fallback_reasons: [] })} />)
    expect(screen.getByTestId('plan-source-note')).toHaveTextContent(
      'Built-in plan (derived from the request route; no planner call)',
    )
  })
})
