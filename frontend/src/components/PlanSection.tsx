import type { ExecutionPlan, PlanTask } from '../api/types'

const STATUS_ICONS: Record<string, string> = {
  pending: '○',
  running: '●',
  completed: '✓',
  failed: '✗',
  blocked: '⊘',
  unsupported: '⚠',
}

function PlanTaskRow({ task }: { task: PlanTask }) {
  const label =
    task.capability_label ??
    (task.capability ? `Capability: ${task.capability}` : task.capability)
  const deps = (task.depends_on ?? []).filter(Boolean)
  return (
    <li className={`plan-task plan-task--${task.status}`}>
      <div className="plan-task__head">
        <span className="plan-task__icon" aria-hidden="true">
          {STATUS_ICONS[task.status] ?? '○'}
        </span>
        <span className={`plan-task__status plan-task__status--${task.status}`}>
          {task.status}
        </span>
        <code className="plan-task__id">{task.task_id}</code>
        <span className="plan-task__cap">{label}</span>
      </div>
      {task.objective && <div className="plan-task__objective">{task.objective}</div>}
      {deps.length > 0 && (
        <div className="plan-task__deps">
          after {deps.join(', ')}
        </div>
      )}
      {task.expected_output && !task.result && (
        <div className="plan-task__expected muted">
          Expected: {task.expected_output}
        </div>
      )}
      {task.status === 'completed' && task.result && (
        <div className="plan-task__result">
          {task.result_truncated ? '…' : ''}
          {task.result}
          {task.result_truncated && <span className="muted"> (truncated)</span>}
        </div>
      )}
      {(task.status === 'failed' || task.status === 'blocked' ||
        task.status === 'unsupported') && (
        <div className="plan-task__error">
          {task.block_reason ??
            task.error ??
            'This task did not run to completion.'}
        </div>
      )}
    </li>
  )
}

export function PlanSourceNote({ plan }: { plan: ExecutionPlan }) {
  if (plan.source === 'fallback') {
    return (
      <div className="plan-source-note" data-testid="plan-source-note">
        Built-in fallback plan (no planner call was used).{' '}
        {plan.fallback_reasons.length > 0 && (
          <ul className="plan-source-note__reasons">
            {plan.fallback_reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        )}
      </div>
    )
  }
  if (plan.source === 'default') {
    return (
      <div className="plan-source-note" data-testid="plan-source-note">
        Built-in plan (derived from the request route; no planner call).
      </div>
    )
  }
  return null
}

export default function PlanSection({ plan }: { plan: ExecutionPlan | null | undefined }) {
  // Legacy / simple executions have no plan: render nothing at all so the
  // detail page stays byte-identical for them.
  if (!plan || !Array.isArray(plan.tasks) || plan.tasks.length === 0) {
    return null
  }
  return (
    <section className="card plan" aria-label="Execution plan">
      <h2>Plan (what the agent intended)</h2>
      {plan.goal && <p className="plan__goal muted">{plan.goal}</p>}
      <PlanSourceNote plan={plan} />
      <ol className="plan-task-list">
        {plan.tasks.map((t, i) => (
          <PlanTaskRow key={t.task_id ?? i} task={t} />
        ))}
      </ol>
      <p className="muted plan__hint">
        Statuses and results come from the executed workflow data — the
        timeline below shows what actually happened.
      </p>
    </section>
  )
}
