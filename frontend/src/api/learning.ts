// Typed access to the persisted learning records. Everything here comes from
// GET /api/v1/learning (app/api/control_center.learning_summary) — the browser
// derives nothing beyond formatting.

export type LessonStatus =
  | 'active'
  | 'candidate'
  | 'rejected'
  | 'disabled'
  | 'superseded'
  | 'validated'

export interface LessonEvidence {
  kind?: string
  kind_of_feedback?: string
  missing?: string[]
  keywords?: string[]
  text?: string
  at?: string
  action?: string
  reason?: string | null
}

export interface LessonCalibration {
  feedback_type?: string
  impact?: string
  restricts?: boolean
  learnable?: boolean
  temporary?: boolean
  scope?: string
  scope_detail?: {
    kind?: string
    capability?: string
    conditions?: string[]
    narrow_object?: boolean
  }
  overgeneralization?: { kind: string; detail: string; blocking?: boolean }[]
  review_reasons?: string[]
  escalate?: string | null
  conflict_with?: string[]
}

export interface Lesson {
  id: string
  status: LessonStatus | string
  category: string
  capability: string
  scope: string
  // Calibration (2026-10-30). null means the row predates the classification;
  // the page says "not classified" instead of guessing.
  feedback_type?: string | null
  impact?: string | null
  temporary?: boolean
  restricts?: boolean
  calibration?: LessonCalibration
  problem: string
  root_cause: string | null
  corrective_behavior: string
  validation: string | null
  confidence: number
  origin: string
  evidence: LessonEvidence[]
  source_execution_id: string | null
  source_text: string | null
  hits: number
  successes: number
  failures: number
  created_at: string
  updated_at: string
  activated_at: string | null
  disabled_at: string | null
  superseded_by: string | null
}

export interface SkillCheck {
  check: string
  ok: boolean
  detail: string
}

export interface SkillEvaluation {
  evaluated_at?: string
  passed?: boolean
  checks?: SkillCheck[]
}

export interface SkillVersion {
  version_id: string
  skill_id: string
  version: number
  status: string
  instructions: string | null
  validation: string | null
  source_lesson_ids: string[]
  evaluation: SkillEvaluation | null
  passed: boolean | null
  created_at: string
  activated_at: string | null
  retired_at: string | null
}

export interface RepeatedPattern {
  capability: string
  category: string
  lessons: number
  occurrences: number
  example: string | null
}

export interface LearningSummary {
  checked_at: string
  counts: Record<string, number>
  total: number
  active: Lesson[]
  candidates: Lesson[]
  rejected: Lesson[]
  disabled: Lesson[]
  superseded: Lesson[]
  repeated_patterns: RepeatedPattern[]
  skills: SkillVersion[]
  awaiting_approval: Lesson[]
  note: string
}

export type LessonAction = 'activate' | 'disable' | 'reject' | 'correct'

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json()
    const detail = body?.detail
    if (typeof detail === 'string') return detail
    if (detail?.human_message) return detail.human_message
  } catch {
    /* not JSON */
  }
  return `Request failed (${res.status})`
}

export async function getLearning(): Promise<LearningSummary> {
  const res = await fetch('/api/v1/learning')
  if (!res.ok) throw new Error(await errorMessage(res))
  const data = (await res.json()) as LearningSummary
  if (!data || !Array.isArray(data.active)) {
    throw new Error('The backend returned an incomplete learning summary.')
  }
  return data
}

export async function reviewLesson(
  lessonId: string,
  action: LessonAction,
  correctiveBehavior?: string,
): Promise<Lesson> {
  const res = await fetch(
    `/api/v1/learning/lessons/${encodeURIComponent(lessonId)}/review`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action,
        corrective_behavior: correctiveBehavior ?? null,
      }),
    },
  )
  if (!res.ok) throw new Error(await errorMessage(res))
  const body = (await res.json()) as { lesson: Lesson }
  return body.lesson
}

export async function rollbackSkill(
  skillId: string,
): Promise<{ rolled_back: boolean; reason?: string; version?: SkillVersion }> {
  const res = await fetch(
    `/api/v1/learning/skills/${encodeURIComponent(skillId)}/rollback`,
    { method: 'POST' },
  )
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

// --- Impact evaluation (2026-10-30) ---------------------------------------- //
// Every value below is read from the benchmark report files by
// GET /api/v1/learning/evaluation; nothing is recomputed in the browser.

export interface EvaluationRegression {
  task_id?: string
  kind: string
  detail?: string
  requirement_id?: string
  new_flags?: string[]
}

export interface EvaluationLessonStat {
  lesson_id: string
  applied_in: string[]
  improved: number
  regressed: number
  unchanged: number
  retrieved_not_applied_in?: string[]
}

export interface EvaluationTotals {
  requirements: number
  satisfied: number
  completeness: number
  qa_findings: number
  succeeded: number
  failed: number
  empty_answers: number
  timeouts: number
  applied_lessons: string[]
  retrieved_lessons: string[]
}

export interface EvaluationTaskRow {
  task_id: string
  capability: string
  request: string
  baseline_execution_id: string
  learning_execution_id: string
  lesson_fixtures: string[]
  baseline_completeness: number
  learning_completeness: number
  delta_pp: number
  outcome: 'improved' | 'regression' | 'mixed' | 'unchanged' | string
  applied_lessons: string[]
  restricted_lessons?: string[]
  retrieved_not_applied: string[]
  regressions: EvaluationRegression[]
}

// A behavioural metric is reported on its own: a task that 'improved' while
// restricting unrelated work is a failure, so these are never summed.
export interface EvaluationMetric {
  metric: string
  total: number
  baseline_satisfied: number
  learning_satisfied: number
  baseline_rate: number
  learning_rate: number
  tasks: string[]
  gate_accuracy?: number | null
}

export interface EvaluationCalibration {
  count: number | null
  accuracy: Record<string, number | null>
  mismatches: { id: string; field: string; expected: unknown; actual: unknown }[]
}

export interface EvaluationReport {
  benchmark_version: string
  harness_version: number
  mode: string
  live_model: boolean
  provider: Record<string, unknown>
  started_at: string
  finished_at: string
  task_count: number
  summary: {
    tasks: number
    baseline: EvaluationTotals
    learning: EvaluationTotals
    delta: {
      completeness_pp: number
      requirements_satisfied: number
      qa_findings: number
      succeeded: number
      empty_answers: number
      regressions: number
      improved_tasks: number
      regressed_tasks: number
      unchanged_tasks: number
    }
    regressions: EvaluationRegression[]
    by_capability: {
      capability: string
      tasks: number
      completeness_baseline: number
      completeness_learning: number
      improved: number
      regressions: number
      unchanged: number
    }[]
    lessons: EvaluationLessonStat[]
    metrics?: EvaluationMetric[]
    calibration_accuracy?: Record<string, number | null>
  }
  calibration?: EvaluationCalibration | null
  tasks: EvaluationTaskRow[]
  notes: string[]
}

export interface EvaluationHistoryRow {
  file: string
  run_at: string
  modified_at: string
  benchmark_version: string
  mode: string
  live_model: boolean
  task_count: number
  completeness_baseline: number
  completeness_learning: number
  delta_pp: number
  qa_findings_baseline: number
  qa_findings_learning: number
  improved_tasks: number
  regressed_tasks: number
  regressions: number
}

export interface LearningEvaluation {
  checked_at: string
  available: boolean
  latest: EvaluationReport | null
  history: EvaluationHistoryRow[]
  evaluation_dir: string
  skipped: { file: string; reason: string }[]
  note: string
  run_command: string
  reason?: string
}

export async function getEvaluation(): Promise<LearningEvaluation> {
  const res = await fetch('/api/v1/learning/evaluation')
  if (!res.ok) throw new Error(await errorMessage(res))
  const data = (await res.json()) as LearningEvaluation
  if (!data || typeof data.available !== 'boolean') {
    throw new Error('The backend returned an incomplete evaluation summary.')
  }
  return data
}
