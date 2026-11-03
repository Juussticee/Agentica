import { useCallback, useEffect, useState } from 'react'
import {
  getEvaluation,
  getLearning,
  reviewLesson,
  rollbackSkill,
  type LearningEvaluation,
  type LearningSummary,
  type Lesson,
  type LessonAction,
} from '../api/learning'
import { Button, Empty, ErrorCard, Loading } from '../components/ui'

// Statuses the backend actually stores; anything else renders as written.
const STATUS_CLASS: Record<string, string> = {
  active: 'available',
  candidate: 'partial',
  rejected: 'unavailable',
  disabled: 'unavailable',
  superseded: 'unknown',
}

function statusBadge(status: string) {
  return `badge badge--${STATUS_CLASS[status] ?? 'unknown'}`
}

const OUTCOME_CLASS: Record<string, string> = {
  improved: 'available',
  regression: 'unavailable',
  mixed: 'partial',
  unchanged: 'empty',
}

function pct(value: number | null | undefined): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
}

function pp(value: number | null | undefined): string {
  if (typeof value !== 'number') return '—'
  return `${value > 0 ? '+' : ''}${value.toFixed(1)} pp`
}

function EvaluationSection({ evaluation }: { evaluation: LearningEvaluation }) {
  const report = evaluation.latest
  return (
    <section>
      <h2>Impact evaluation</h2>
      <p className="muted">
        A benchmark runs every task twice through the real execution pipeline —
        once with no lesson set (<strong>baseline</strong>) and once with the
        applicable active lessons (<strong>learning enabled</strong>) — and
        scores both from the stored answers. {evaluation.note}
      </p>

      {!evaluation.available || !report ? (
        <Empty>
          {evaluation.reason ??
            'No benchmark report is available yet. Run the harness to measure the lesson impact.'}
          <div className="mono-small eval__command">{evaluation.run_command}</div>
        </Empty>
      ) : (
        <>
          <p className="muted eval__meta">
            {report.benchmark_version} · {report.task_count} tasks × 2 ·{' '}
            {report.mode} · {when(report.finished_at)}
            {report.live_model && (
              <>
                {' '}
                <span className="badge badge--partial">live model</span>
              </>
            )}
            {!report.live_model && (
              <>
                {' '}
                <span className="badge badge--empty">scripted provider</span>
              </>
            )}
          </p>

          <div className="card">
            <table className="table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Baseline</th>
                  <th>Learning enabled</th>
                  <th>Difference</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Requirement completeness</td>
                  <td>{pct(report.summary.baseline.completeness)}</td>
                  <td>{pct(report.summary.learning.completeness)}</td>
                  <td>{pp(report.summary.delta.completeness_pp)}</td>
                </tr>
                <tr>
                  <td>Requirements satisfied</td>
                  <td>
                    {report.summary.baseline.satisfied}/
                    {report.summary.baseline.requirements}
                  </td>
                  <td>
                    {report.summary.learning.satisfied}/
                    {report.summary.learning.requirements}
                  </td>
                  <td>{report.summary.delta.requirements_satisfied > 0 ? '+' : ''}
                    {report.summary.delta.requirements_satisfied}</td>
                </tr>
                <tr>
                  <td>QA findings</td>
                  <td>{report.summary.baseline.qa_findings}</td>
                  <td>{report.summary.learning.qa_findings}</td>
                  <td>{report.summary.delta.qa_findings > 0 ? '+' : ''}
                    {report.summary.delta.qa_findings}</td>
                </tr>
                <tr>
                  <td>Successful executions</td>
                  <td>{report.summary.baseline.succeeded}</td>
                  <td>{report.summary.learning.succeeded}</td>
                  <td>{report.summary.delta.succeeded > 0 ? '+' : ''}
                    {report.summary.delta.succeeded}</td>
                </tr>
                <tr>
                  <td>Empty answers</td>
                  <td>{report.summary.baseline.empty_answers}</td>
                  <td>{report.summary.learning.empty_answers}</td>
                  <td>{report.summary.delta.empty_answers > 0 ? '+' : ''}
                    {report.summary.delta.empty_answers}</td>
                </tr>
                <tr>
                  <td>Regressions</td>
                  <td>—</td>
                  <td>{report.summary.delta.regressions}</td>
                  <td>{report.summary.delta.regressions > 0 ? '+' : ''}
                    {report.summary.delta.regressions}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="muted eval__meta">
            improved {report.summary.delta.improved_tasks} · regressed{' '}
            {report.summary.delta.regressed_tasks} · unchanged{' '}
            {report.summary.delta.unchanged_tasks}
          </p>

          {(report.summary.metrics?.length ?? 0) > 0 && (
            <>
              <h3>Behaviour metrics</h3>
              <p className="muted">
                A rate is the share of that metric&apos;s requirements that were
                not met, so lower is better. These are reported separately: a
                task that improved while restricting unrelated work is a
                failure, not a win.
              </p>
              <div className="card eval__table">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Baseline unmet</th>
                      <th>Learning unmet</th>
                      <th>Gate accuracy</th>
                      <th>Requirements</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(report.summary.metrics ?? []).map((metric) => (
                      <tr key={metric.metric}>
                        <td className="mono-small">{metric.metric}</td>
                        <td>{pct(metric.baseline_rate)}</td>
                        <td
                          className={
                            metric.learning_rate > metric.baseline_rate
                              ? 'metric-worse'
                              : undefined
                          }
                        >
                          {pct(metric.learning_rate)}
                        </td>
                        <td>
                          {typeof metric.gate_accuracy === 'number'
                            ? pct(metric.gate_accuracy)
                            : '—'}
                        </td>
                        <td>{metric.total}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {report.calibration && report.calibration.count ? (
            <p className="muted eval__meta">
              Classification gate: {report.calibration.count} fixture(s),
              feedback type{' '}
              {pct(report.calibration.accuracy.feedback_type ?? 0)}, scope{' '}
              {pct(report.calibration.accuracy.scope ?? 0)}, learnable{' '}
              {pct(report.calibration.accuracy.learnable ?? 0)},{' '}
              overgeneralization{' '}
              {report.calibration.accuracy.overgeneralization == null
                ? 'not measured'
                : pct(report.calibration.accuracy.overgeneralization)}
              .
            </p>
          ) : null}

          <div className="card eval__table">
            <table className="table">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Capability</th>
                  <th>Baseline</th>
                  <th>Learning</th>
                  <th>Difference</th>
                  <th>Applied lessons</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {report.tasks.map((task) => (
                  <tr key={task.task_id}>
                    <td>
                      <a
                        href={`#/executions/${encodeURIComponent(task.learning_execution_id)}`}
                      >
                        {task.task_id}
                      </a>
                    </td>
                    <td>{task.capability}</td>
                    <td>{pct(task.baseline_completeness)}</td>
                    <td>{pct(task.learning_completeness)}</td>
                    <td>{pp(task.delta_pp)}</td>
                    <td className="mono-small">
                      {task.applied_lessons.length
                        ? task.applied_lessons.join(', ')
                        : 'none'}
                      {(task.restricted_lessons?.length ?? 0) > 0 && (
                        <>
                          {' '}
                          <span className="badge badge--partial">restriction</span>
                        </>
                      )}
                    </td>
                    <td>
                      <span
                        className={`badge badge--${OUTCOME_CLASS[task.outcome] ?? 'unknown'}`}
                      >
                        {task.outcome}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {report.summary.regressions.length > 0 && (
            <>
              <h3>Regressions</h3>
              <ul className="eval__regressions">
                {report.summary.regressions.map((regression, index) => (
                  <li key={`${regression.task_id}-${index}`}>
                    <span className="badge badge--unavailable">
                      {regression.kind}
                    </span>{' '}
                    <strong>{regression.task_id}</strong> —{' '}
                    {regression.requirement_id ?? regression.detail}
                  </li>
                ))}
              </ul>
            </>
          )}

          <h3>Per lesson</h3>
          {report.summary.lessons.length === 0 ? (
            <Empty>No lesson was applied to any benchmark task.</Empty>
          ) : (
            <div className="card eval__table">
              <table className="table">
                <thead>
                  <tr>
                    <th>Lesson</th>
                    <th>Applied in</th>
                    <th>Improved</th>
                    <th>Regressed</th>
                    <th>Retrieved, not applied</th>
                  </tr>
                </thead>
                <tbody>
                  {report.summary.lessons.map((lesson) => (
                    <tr key={lesson.lesson_id}>
                      <td className="mono-small">{lesson.lesson_id}</td>
                      <td>{lesson.applied_in.join(', ') || '—'}</td>
                      <td>{lesson.improved}</td>
                      <td>{lesson.regressed}</td>
                      <td>{(lesson.retrieved_not_applied_in ?? []).join(', ') || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {evaluation.history.length > 1 && (
            <>
              <h3>Earlier runs</h3>
              <div className="card eval__table">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Run</th>
                      <th>Completeness</th>
                      <th>Difference</th>
                      <th>Regressed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluation.history.slice(1).map((row) => (
                      <tr key={row.file}>
                        <td>{when(row.run_at)}</td>
                        <td>
                          {pct(row.completeness_baseline)} →{' '}
                          {pct(row.completeness_learning)}
                        </td>
                        <td>{pp(row.delta_pp)}</td>
                        <td>{row.regressed_tasks}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {report.notes.length > 0 && (
            <details className="eval__notes">
              <summary>How these numbers were produced</summary>
              <ul>
                {report.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </section>
  )
}

function when(value: string | null): string {
  if (!value) return '—'
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? new Date(parsed).toLocaleString() : value
}

function LessonCard({
  lesson,
  busy,
  onAction,
  onCorrect,
}: {
  lesson: Lesson
  busy: boolean
  onAction: (id: string, action: LessonAction) => void
  onCorrect: (id: string, behaviour: string) => void
}) {
  const [revision, setRevision] = useState(lesson.corrective_behavior)
  const reviews = lesson.evidence.filter((e) => e.kind === 'review')
  const corrections = lesson.evidence.filter((e) => e.kind !== 'review')
  return (
    <article className="card lesson-card">
      <header className="lesson-card__head">
        <div>
          <h3>{lesson.corrective_behavior}</h3>
          <span className="muted mono-small">
            {lesson.id} · {lesson.capability} · {lesson.category} ·{' '}
            scope {lesson.scope}
          </span>
          <div className="lesson-card__badges">
            <span className="badge badge--empty">
              {lesson.feedback_type?.replace(/_/g, ' ') ??
                'not classified'}
            </span>
            {lesson.impact && (
              <span className="badge badge--empty">
                {lesson.impact.replace(/_/g, ' ')}
              </span>
            )}
            {lesson.temporary && (
              <span className="badge badge--partial">this task only</span>
            )}
            {lesson.restricts && (
              <span className="badge badge--partial">restriction</span>
            )}
          </div>
        </div>
        <span className={statusBadge(lesson.status)}>{lesson.status}</span>
      </header>

      <dl className="kv">
        <dt>Problem</dt><dd>{lesson.problem}</dd>
        <dt>Cause</dt><dd>{lesson.root_cause || 'unknown'}</dd>
        <dt>Check</dt><dd>{lesson.validation || '—'}</dd>
        <dt>Origin</dt>
        <dd>
          {lesson.origin}
          {lesson.source_execution_id && (
            <>
              {' · execution '}
              <a href={`#/executions/${encodeURIComponent(lesson.source_execution_id)}`}>
                {lesson.source_execution_id}
              </a>
            </>
          )}
        </dd>
        <dt>Classification</dt>
        <dd>
          {lesson.feedback_type?.replace(/_/g, ' ') ?? 'not classified'}
          {lesson.impact ? ` · ${lesson.impact.replace(/_/g, ' ')}` : ''}
          {lesson.temporary ? ' · applies to one task only' : ''}
          {lesson.restricts ? ' · restricts behaviour' : ''}
          {(lesson.calibration?.review_reasons?.length ?? 0) > 0 && (
            <div className="muted">
              Held for review: {lesson.calibration?.review_reasons?.join('; ')}
            </div>
          )}
          {(lesson.calibration?.overgeneralization?.length ?? 0) > 0 && (
            <div className="muted">
              Wider than the correction:{' '}
              {lesson.calibration?.overgeneralization
                ?.map((f) => f.kind.replace(/_/g, ' '))
                .join(', ')}
            </div>
          )}
          {(lesson.calibration?.conflict_with?.length ?? 0) > 0 && (
            <div className="muted">
              Conflicts with {lesson.calibration?.conflict_with?.join(', ')} —
              both kept, this one waits for review
            </div>
          )}
        </dd>
        <dt>Confidence</dt><dd>{lesson.confidence}</dd>
        <dt>Used</dt>
        <dd>
          {lesson.hits} time(s) · {lesson.successes} with a clean run ·{' '}
          {lesson.failures} with QA issues
        </dd>
        <dt>Learned</dt><dd>{when(lesson.created_at)}</dd>
        {lesson.superseded_by && (
          <>
            <dt>Superseded by</dt>
            <dd className="mono-small">{lesson.superseded_by}</dd>
          </>
        )}
      </dl>

      {corrections.length > 0 && (
        <details className="lesson-card__evidence">
          <summary>Original evidence ({corrections.length})</summary>
          <ul>
            {corrections.map((entry, index) => (
              <li key={index}>
                <span className="muted">
                  {entry.kind_of_feedback ?? entry.kind} · {when(entry.at ?? null)}
                </span>
                <div>“{entry.text}”</div>
                {entry.missing && entry.missing.length > 0 && (
                  <div className="muted">Named: {entry.missing.join(', ')}</div>
                )}
                {entry.keywords && entry.keywords.length > 0 && (
                  <div className="muted mono-small">
                    keywords: {entry.keywords.join(', ')}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}

      {reviews.length > 0 && (
        <p className="muted">
          Reviewed {reviews.length} time(s); last:{' '}
          {reviews[reviews.length - 1].action} on{' '}
          {when(reviews[reviews.length - 1].at ?? null)}
        </p>
      )}

      <div className="row-actions">
        {lesson.status !== 'active' && (
          <Button disabled={busy} onClick={() => onAction(lesson.id, 'activate')}>
            Activate
          </Button>
        )}
        {lesson.status === 'active' && (
          <Button
            variant="ghost"
            disabled={busy}
            onClick={() => onAction(lesson.id, 'disable')}
          >
            Disable
          </Button>
        )}
        {lesson.status !== 'rejected' && (
          <Button
            variant="ghost"
            disabled={busy}
            onClick={() => onAction(lesson.id, 'reject')}
          >
            Reject
          </Button>
        )}
      </div>

      <details className="lesson-card__evidence">
        <summary>Correct this behaviour</summary>
        <p className="muted">
          Rewrites what future tasks are told. The stored status does not change
          and the original correction stays in the evidence above, so the edit
          is auditable.
        </p>
        <textarea
          className="lesson-card__revision"
          rows={3}
          value={revision}
          disabled={busy}
          onChange={(event) => setRevision(event.target.value)}
          aria-label={`Corrected behaviour for ${lesson.id}`}
        />
        <Button
          variant="ghost"
          disabled={busy || !revision.trim() || revision === lesson.corrective_behavior}
          onClick={() => onCorrect(lesson.id, revision.trim())}
        >
          Save correction
        </Button>
      </details>
    </article>
  )
}

export default function Learning() {
  const [data, setData] = useState<LearningSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [evaluation, setEvaluation] = useState<LearningEvaluation | null>(null)
  const [evalError, setEvalError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setData(await getLearning())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
    // The benchmark report is a separate read: a missing or malformed report
    // must never stop the lessons themselves from rendering.
    try {
      setEvaluation(await getEvaluation())
      setEvalError(null)
    } catch (e) {
      setEvalError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const act = useCallback(
    async (lessonId: string, action: LessonAction) => {
      setBusyId(lessonId)
      setNotice(null)
      try {
        const lesson = await reviewLesson(lessonId, action)
        setNotice(`${lesson.id} is now ${lesson.status}.`)
        await refresh()
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusyId(null)
      }
    },
    [refresh],
  )

  const correct = useCallback(
    async (lessonId: string, behavior: string) => {
      setBusyId(lessonId)
      setNotice(null)
      try {
        const lesson = await reviewLesson(lessonId, 'correct', behavior)
        setNotice(`${lesson.id} now teaches: ${lesson.corrective_behavior}`)
        await refresh()
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusyId(null)
      }
    },
    [refresh],
  )

  const rollback = useCallback(
    async (skillId: string) => {
      setBusyId(skillId)
      setNotice(null)
      try {
        const outcome = await rollbackSkill(skillId)
        setNotice(
          outcome.rolled_back
            ? `${skillId} restored to version ${outcome.version?.version}.`
            : `No earlier version of ${skillId} to restore: ${outcome.reason ?? ''}`,
        )
        await refresh()
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusyId(null)
      }
    },
    [refresh],
  )

  if (error && !data) {
    return (
      <div>
        <h1>Learning</h1>
        <ErrorCard
          message={error}
          recovery="Check that the backend is running, then refresh."
        />
      </div>
    )
  }
  if (!data) {
    return (
      <div>
        <h1>Learning</h1>
        <Loading label="Reading learned lessons…" />
      </div>
    )
  }

  const activeSkills = data.skills.filter((v) => v.status === 'active')

  return (
    <div>
      <h1>Learning</h1>
      <p className="muted">
        Lessons are learned from your own corrections and from failed runs.
        Only <strong>active</strong> lessons whose capability matches a task are
        applied to it. {data.note}
      </p>

      {error && <ErrorCard message={error} onDismiss={() => setError(null)} />}
      {notice && <div className="state empty">{notice}</div>}

      <div className="summary-row">
        <div className="summary-tile">
          <span className="summary-tile__value">{data.counts.active ?? 0}</span>
          <span className="summary-tile__label">Active lessons</span>
        </div>
        <div className="summary-tile">
          <span className="summary-tile__value">{data.candidates.length}</span>
          <span className="summary-tile__label">Awaiting review</span>
        </div>
        <div className="summary-tile">
          <span className="summary-tile__value">
            {data.disabled.length + data.rejected.length}
          </span>
          <span className="summary-tile__label">Disabled or rejected</span>
        </div>
        <div className="summary-tile">
          <span className="summary-tile__value">{data.skills.length}</span>
          <span className="summary-tile__label">Skill versions</span>
        </div>
      </div>

      {data.total === 0 && (
        <Empty>
          Nothing has been learned yet. Correct a result in conversation — for
          example “this isn’t enough, I need benchmarks and licensing” — and the
          lesson appears here with the wording that produced it.
        </Empty>
      )}

      {data.candidates.length > 0 && (
        <section>
          <h2>Awaiting review ({data.candidates.length})</h2>
          <p className="muted">
            These were recorded but not applied: a permission or security change
            stays a proposal, a lesson learned from a failure needs evidence, and
            an unscoped factual correction would apply too broadly.
          </p>
          {data.candidates.map((lesson) => (
            <LessonCard
              key={lesson.id}
              lesson={lesson}
              busy={busyId === lesson.id}
              onAction={act}
              onCorrect={correct}
            />
          ))}
        </section>
      )}

      {data.active.length > 0 && (
        <section>
          <h2>Active ({data.active.length})</h2>
          {data.active.map((lesson) => (
            <LessonCard
              key={lesson.id}
              lesson={lesson}
              busy={busyId === lesson.id}
              onAction={act}
              onCorrect={correct}
            />
          ))}
        </section>
      )}

      {data.repeated_patterns.length > 0 && (
        <section>
          <h2>Repeated patterns</h2>
          <p className="muted">
            The same capability corrected in the same way more than once. This is
            the signal that the behaviour, not just the memory, needs fixing.
          </p>
          <div className="card">
            <table className="table">
              <thead>
                <tr>
                  <th>Capability</th>
                  <th>Category</th>
                  <th>Lessons</th>
                  <th>Use count</th>
                  <th>Example behaviour</th>
                </tr>
              </thead>
              <tbody>
                {data.repeated_patterns.map((pattern) => (
                  <tr key={`${pattern.capability}-${pattern.category}`}>
                    <td>{pattern.capability}</td>
                    <td>{pattern.category}</td>
                    <td>{pattern.lessons}</td>
                    <td>{pattern.occurrences}</td>
                    <td>{pattern.example ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section>
        <h2>Skills</h2>
        <p className="muted">
          A learned skill is a procedure built from active lessons for one
          capability. A candidate is activated only when its deterministic
          evaluation passes; the previous version is kept, so a rollback is
          exact.
        </p>
        {data.skills.length === 0 ? (
          <Empty>
            No learned skill version yet. One is proposed once a capability has
            at least two active lessons.
          </Empty>
        ) : (
          data.skills.map((version) => (
            <article key={version.version_id} className="card lesson-card">
              <header className="lesson-card__head">
                <div>
                  <h3>
                    {version.skill_id} v{version.version}
                  </h3>
                  <span className="muted mono-small">
                    {version.version_id} · from {version.source_lesson_ids.length}{' '}
                    lesson(s) · {when(version.created_at)}
                  </span>
                </div>
                <span className={statusBadge(version.status)}>
                  {version.status}
                </span>
              </header>
              {version.evaluation && (
                <ul className="check-list">
                  {(version.evaluation.checks ?? []).map((check) => (
                    <li key={check.check}>
                      <span className={check.ok ? 'ok' : 'fail'}>
                        {check.ok ? '✔' : '✘'}
                      </span>{' '}
                      <span className="mono-small">{check.check}</span> —{' '}
                      <span className="muted">{check.detail}</span>
                    </li>
                  ))}
                </ul>
              )}
              <details>
                <summary>Procedure text</summary>
                <pre className="mono-small">{version.instructions}</pre>
              </details>
              <div className="row-actions">
                <Button
                  variant="ghost"
                  disabled={busyId === version.skill_id}
                  onClick={() => rollback(version.skill_id)}
                >
                  Roll back this skill
                </Button>
                {activeSkills.some((v) => v.skill_id === version.skill_id) && (
                  <span className="muted">
                    active:{' '}
                    {
                      activeSkills.find((v) => v.skill_id === version.skill_id)
                        ?.version
                    }
                  </span>
                )}
              </div>
            </article>
          ))
        )}
      </section>

      {evalError && (
        <ErrorCard
          message={evalError}
          recovery="The benchmark report could not be read. Lessons are unaffected."
          onDismiss={() => setEvalError(null)}
        />
      )}
      {evaluation && <EvaluationSection evaluation={evaluation} />}

      {(data.disabled.length > 0 ||
        data.rejected.length > 0 ||
        data.superseded.length > 0) && (
        <section>
          <h2>Not applied</h2>
          <div className="card">
            <table className="table">
              <thead>
                <tr>
                  <th>Lesson</th>
                  <th>Status</th>
                  <th>Capability</th>
                  <th>Behaviour</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {[...data.disabled, ...data.rejected, ...data.superseded].map(
                  (lesson) => (
                    <tr key={lesson.id}>
                      <td className="mono-small">{lesson.id}</td>
                      <td>{lesson.status}</td>
                      <td>{lesson.capability}</td>
                      <td>{lesson.corrective_behavior}</td>
                      <td>
                        <Button
                          variant="ghost"
                          disabled={busyId === lesson.id}
                          onClick={() => act(lesson.id, 'activate')}
                        >
                          Activate
                        </Button>
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
