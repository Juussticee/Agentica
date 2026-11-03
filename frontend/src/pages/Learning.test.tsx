import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Learning from './Learning'

const BASE = {
  checked_at: '2026-10-30T10:00:00+00:00',
  counts: {},
  total: 0,
  active: [],
  candidates: [],
  rejected: [],
  disabled: [],
  superseded: [],
  repeated_patterns: [],
  skills: [],
  awaiting_approval: [],
  note: 'Recording a lesson does not prove behaviour changed.',
}

const candidate = {
  id: 'lesson-1',
  status: 'candidate',
  category: 'preference',
  capability: 'research',
  scope: 'research comparisons',
  feedback_type: 'quality_feedback',
  impact: 'quality_improvement',
  temporary: false,
  restricts: false,
  calibration: {
    feedback_type: 'quality_feedback',
    impact: 'quality_improvement',
    restricts: false,
    scope_detail: { kind: 'standing', capability: 'research' },
    overgeneralization: [],
    review_reasons: [],
  },
  problem: 'The comparison had no benchmarks or licensing.',
  root_cause: 'The research instruction never asked for them.',
  corrective_behavior: 'Include benchmarks, VRAM figures and licensing.',
  validation: 'Every model in the answer states a benchmark figure.',
  confidence: 0.7,
  origin: 'user_feedback',
  evidence: [
    {
      kind: 'user_correction',
      kind_of_feedback: 'insufficient',
      missing: ['benchmarks', 'licensing'],
      text: "isn't enough, I need benchmarks and licensing",
      at: '2026-10-30T09:59:00+00:00',
    },
  ],
  source_execution_id: 'exec-1',
  source_text: "isn't enough, I need benchmarks and licensing",
  hits: 0,
  successes: 0,
  failures: 0,
  created_at: '2026-10-30T09:59:30+00:00',
  updated_at: '2026-10-30T09:59:30+00:00',
  activated_at: null,
  disabled_at: null,
  superseded_by: null,
}

const skillVersion = {
  version_id: 'sv-2',
  skill_id: 'research_compare',
  version: 2,
  status: 'candidate',
  instructions: 'Compare with benchmarks.',
  validation: 'cited',
  source_lesson_ids: ['lesson-1', 'lesson-2'],
  evaluation: {
    passed: false,
    checks: [{ check: 'cites_sources', ok: false, detail: 'no source lines' }],
  },
  passed: false,
  created_at: '2026-10-30T09:59:40+00:00',
  activated_at: null,
  retired_at: null,
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const EMPTY_EVALUATION = {
  checked_at: '2026-10-30T10:00:00+00:00',
  available: false,
  latest: null,
  history: [],
  evaluation_dir: 'storage/learning_eval',
  skipped: [],
  note: 'Evaluation is observational.',
  run_command: 'env\\Scripts\\python.exe scripts\\eval_learning.py --print',
  reason: 'No benchmark has been run yet.',
}

const REPORT = {
  benchmark_version: 'learning_eval_v1',
  harness_version: 1,
  mode: 'offline_deterministic',
  live_model: false,
  provider: { description: 'scripted benchmark double' },
  started_at: '2026-10-30T10:00:00+00:00',
  finished_at: '2026-10-30T10:00:04+00:00',
  task_count: 2,
  summary: {
    tasks: 2,
    baseline: {
      requirements: 3, satisfied: 1, completeness: 1 / 3, qa_findings: 0,
      succeeded: 2, failed: 0, empty_answers: 0, timeouts: 0,
      applied_lessons: [], retrieved_lessons: [],
    },
    learning: {
      requirements: 3, satisfied: 3, completeness: 1, qa_findings: 0,
      succeeded: 2, failed: 0, empty_answers: 0, timeouts: 0,
      applied_lessons: ['fx-compare-depth'], retrieved_lessons: ['fx-compare-depth'],
    },
    delta: {
      completeness_pp: 66.67, requirements_satisfied: 2, qa_findings: 0,
      succeeded: 0, empty_answers: 0, regressions: 1,
      improved_tasks: 1, regressed_tasks: 1, unchanged_tasks: 0,
    },
    regressions: [
      { task_id: 'gen-brevity-regression', kind: 'requirement_lost', requirement_id: 'names-tradeoff' },
    ],
    by_capability: [],
    lessons: [
      { lesson_id: 'fx-compare-depth', applied_in: ['gen-model-compare'], improved: 1, regressed: 0, unchanged: 0 },
      { lesson_id: 'fx-too-brief', applied_in: ['gen-brevity-regression'], improved: 0, regressed: 1, unchanged: 0, retrieved_not_applied_in: [] },
    ],
    metrics: [
      {
        metric: 'unnecessary_restriction_rate', total: 2,
        baseline_satisfied: 2, learning_satisfied: 1,
        baseline_rate: 0, learning_rate: 0.5,
        tasks: ['drift-over-broad-restriction'], gate_accuracy: null,
      },
      {
        metric: 'initiative_retention', total: 4,
        baseline_satisfied: 4, learning_satisfied: 4,
        baseline_rate: 0, learning_rate: 0,
        tasks: ['drift-autonomy-retained'], gate_accuracy: null,
      },
    ],
    calibration_accuracy: { feedback_type: 1, scope: 1, learnable: 1, overgeneralization: 1 },
  },
  calibration: {
    count: 11,
    accuracy: { feedback_type: 1, impact: 1, scope: 1, learnable: 1, temporary: 1, overgeneralization: 1 },
    mismatches: [],
  },
  tasks: [
    {
      task_id: 'gen-model-compare', capability: 'general', request: 'Compare two models',
      baseline_execution_id: 'b1', learning_execution_id: 'l1',
      lesson_fixtures: ['fx-compare-depth'],
      baseline_completeness: 0, learning_completeness: 1, delta_pp: 100,
      outcome: 'improved', applied_lessons: ['fx-compare-depth'],
      retrieved_not_applied: [], regressions: [],
    },
    {
      task_id: 'gen-brevity-regression', capability: 'general', request: 'Explain trade-off',
      baseline_execution_id: 'b2', learning_execution_id: 'l2',
      lesson_fixtures: ['fx-too-brief'],
      baseline_completeness: 1, learning_completeness: 0, delta_pp: -100,
      outcome: 'regression', applied_lessons: ['fx-too-brief'],
      retrieved_not_applied: [], regressions: [
        { kind: 'requirement_lost', requirement_id: 'names-tradeoff', detail: 'lost' },
      ],
    },
  ],
  notes: ['Mode is offline_deterministic: the provider is a scripted double.'],
}

let summary: Record<string, unknown>
let evaluation: Record<string, unknown>
let reviewed: string | null
let rolledBack: string | null

beforeEach(() => {
  summary = { ...BASE }
  evaluation = { ...EMPTY_EVALUATION }
  reviewed = null
  rolledBack = null
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('/learning/evaluation')) {
        return json(evaluation)
      }
      if (url.includes('/learning/lessons/') && url.endsWith('/review')) {
        const body = JSON.parse(String(init?.body ?? '{}'))
        const action = body.action as string
        reviewed = action === 'correct' ? `correct:${body.corrective_behavior}` : action
        const updated = {
          ...candidate,
          corrective_behavior: body.corrective_behavior || candidate.corrective_behavior,
          status:
            action === 'correct'
              ? candidate.status
              : action === 'activate'
                ? 'active'
                : action,
        }
        summary = {
          ...summary,
          counts: { [updated.status]: 1 },
          total: 1,
          active: updated.status === 'active' ? [updated] : [],
          candidates: updated.status === 'candidate' ? [updated] : [],
          rejected: updated.status === 'rejected' ? [updated] : [],
        }
        return json({ lesson: updated })
      }
      if (url.includes('/learning/skills/') && url.endsWith('/rollback')) {
        rolledBack = 'research_compare'
        return json({ rolled_back: true, version: { ...skillVersion, version: 1, status: 'active' } })
      }
      return json(summary)
    }),
  )
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Learning page', () => {
  it('says nothing was learned instead of showing fake lessons', async () => {
    render(<Learning />)
    expect(await screen.findByText(/Nothing has been learned yet/i)).toBeInTheDocument()
    expect(screen.getByText(/does not prove behaviour changed/i)).toBeInTheDocument()
  })

  it('shows a candidate lesson with its evidence and origin execution', async () => {
    summary = { ...BASE, counts: { candidate: 1 }, total: 1, candidates: [candidate] }
    render(<Learning />)
    expect(
      await screen.findByRole('heading', { name: candidate.corrective_behavior }),
    ).toBeInTheDocument()
    expect(screen.getByText(/Awaiting review \(1\)/i)).toBeInTheDocument()
    expect(screen.getByText(/benchmarks, licensing/i)).toBeInTheDocument()
    const link = screen.getByRole('link', { name: 'exec-1' })
    expect(link).toHaveAttribute('href', '#/executions/exec-1')
  })

  it('activates a candidate through the review endpoint', async () => {
    summary = { ...BASE, counts: { candidate: 1 }, total: 1, candidates: [candidate] }
    render(<Learning />)
    fireEvent.click(await screen.findByRole('button', { name: 'Activate' }))
    await waitFor(() => expect(reviewed).toBe('activate'))
    expect(await screen.findByText(/lesson-1 is now active/i)).toBeInTheDocument()
  })

  it('corrects a lesson without changing its status', async () => {
    summary = { ...BASE, counts: { candidate: 1 }, total: 1, candidates: [candidate] }
    render(<Learning />)
    const box = await screen.findByLabelText('Corrected behaviour for lesson-1')
    fireEvent.change(box, { target: { value: 'Check licensing before recommending.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save correction' }))
    await waitFor(() =>
      expect(reviewed).toBe('correct:Check licensing before recommending.'),
    )
    expect(await screen.findByText(/now teaches/i)).toBeInTheDocument()
  })

  it('reports that no benchmark has run instead of inventing numbers', async () => {
    render(<Learning />)
    expect(
      await screen.findByText(/No benchmark has been run yet/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/scripts\\eval_learning\.py --print/),
    ).toBeInTheDocument()
  })

  it('renders the measured baseline vs learning comparison', async () => {
    evaluation = { ...EMPTY_EVALUATION, available: true, latest: REPORT, history: [] }
    const { container } = render(<Learning />)
    await screen.findByText('+66.7 pp')
    const meta = container.querySelector('.eval__meta')?.textContent ?? ''
    expect(meta).toContain('learning_eval_v1')
    expect(meta).toContain('2 tasks × 2')
    expect(meta).toContain('offline_deterministic')
    expect(meta).toContain('scripted provider')
    expect(screen.getByText('33.3%')).toBeInTheDocument()
    expect(screen.getByText('+66.7 pp')).toBeInTheDocument()
    expect(screen.getByText('requirement_lost')).toBeInTheDocument()
    const text = container.textContent ?? ''
    expect(text).toContain('names-tradeoff')
    expect(text).toContain('fx-compare-depth')
    expect(text).toContain('fx-too-brief')
  })

  it('shows a report read failure without hiding the lessons', async () => {
    evaluation = { ...EMPTY_EVALUATION }
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/learning/evaluation')) {
          return json({ detail: { human_message: 'report directory unreadable' } }, 500)
        }
        return json({ ...BASE })
      }),
    )
    render(<Learning />)
    expect(await screen.findByText('report directory unreadable')).toBeInTheDocument()
    expect(screen.getByText(/Nothing has been learned yet/i)).toBeInTheDocument()
  })

  it('shows the calibration of a lesson instead of a bare rule', async () => {
    summary = {
      ...BASE,
      counts: { candidate: 1 },
      total: 1,
      candidates: [
        {
          ...candidate,
          feedback_type: 'task_constraint',
          impact: 'constraint',
          restricts: true,
          calibration: {
            feedback_type: 'task_constraint',
            impact: 'constraint',
            restricts: true,
            review_reasons: ['introduces a restriction; a restriction needs review before it applies'],
            overgeneralization: [
              { kind: 'restriction_broadened', detail: 'no object', blocking: true },
            ],
          },
        },
      ],
    }
    render(<Learning />)
    await screen.findByRole('heading', { name: candidate.corrective_behavior })
    expect(screen.getByText('task constraint')).toBeInTheDocument()
    expect(screen.getByText('restriction')).toBeInTheDocument()
    expect(screen.getByText(/Held for review/i)).toBeInTheDocument()
    expect(screen.getByText(/Wider than the correction/i)).toBeInTheDocument()
    expect(screen.getByText(/restriction broadened/i)).toBeInTheDocument()
  })

  it('reports behavioural drift metrics separately from completeness', async () => {
    evaluation = { ...EMPTY_EVALUATION, available: true, latest: REPORT, history: [] }
    const { container } = render(<Learning />)
    await screen.findByText('Behaviour metrics')
    expect(screen.getByText('unnecessary_restriction_rate')).toBeInTheDocument()
    expect(screen.getByText('initiative_retention')).toBeInTheDocument()
    expect(screen.getByText('50.0%')).toBeInTheDocument()
    expect(container.querySelector('.metric-worse')).not.toBeNull()
    expect(container.textContent).toContain('Classification gate: 11 fixture(s)')
  })

  it('shows the failed evaluation and can roll a skill back', async () => {
    summary = { ...BASE, skills: [skillVersion] }
    render(<Learning />)
    expect(await screen.findByText('research_compare v2')).toBeInTheDocument()
    expect(screen.getByText('cites_sources')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Roll back this skill/i }))
    await waitFor(() => expect(rolledBack).toBe('research_compare'))
    expect(await screen.findByText(/restored to version 1/i)).toBeInTheDocument()
  })
})
