export type ExecutionStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'waiting_for_approval'

export interface ExecutionError {
  category: string
  human_message: string
  why: string
  recovery: string
  recoverable: boolean
  step_id?: string | null
  agent_id?: string | null
  severity: string
  details?: Record<string, unknown>
}

// GUI-12: projection of the real persisted plan (backend serializer in
// app/execution/store.py). Task statuses are the dispatcher's canonical
// statuses: pending | running | completed | failed | blocked | unsupported.
export type TaskStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'blocked'
  | 'unsupported'

export interface PlanTask {
  task_id: string
  capability: string
  capability_label?: string | null
  objective?: string | null
  depends_on: string[]
  required_inputs?: string[]
  expected_output?: string | null
  done_when?: string | null
  status: TaskStatus | string
  result?: string | null
  result_truncated?: boolean
  error?: string | null
  block_reason?: string | null
}

export type ApprovalStatus = 'pending' | 'approved' | 'rejected' | 'expired'

export interface ApprovalRequest {
  approval_id: string
  execution_id: string | null
  operation: string
  /** Which registered agent asked for it (added 2026-10-30; null when the
   *  persisted operation has no owner mapping). */
  requesting_agent?: string | null
  description: string
  status: ApprovalStatus | string
  decision: 'approved' | 'rejected' | null
  created_at: string
  expires_at: string
  decided_at: string | null
  executed_at: string | null
  payload: {
    question?: string
    route?: string
    local_answer?: string
    books_refusal?: Record<string, unknown> | null
    /** workspace_patch only: the reviewed files and their content hashes. */
    workspace?: string
    files?: { path: string; before_sha256: string; after_sha256: string }[]
  } | null
}

export interface ApprovalQueue {
  approvals: ApprovalRequest[]
  count: number
  status: string
}

export interface ApprovalDecisionResponse {
  approval: ApprovalRequest
  decision: {
    approval_id: string
    outcome: 'approved' | 'rejected'
    decided_at: string | null
    already_decided: boolean
  }
  resume: string
}

export interface ApprovalResumeResponse {
  // "recovered" is the documented outcome when a previous attempt already
  // performed the operation and the resume only finished the remaining work;
  // nothing is re-executed in that case (app/execution/runner.py).
  outcome: 'executed' | 'rejected' | 'denied' | 'recovered'
  approval_id?: string
  answer?: string
  reason?: string
  finalized?: boolean
  execution?: Execution
}

export interface ExecutionPlan {
  goal?: string | null
  source?: string | null
  fallback_reasons: string[]
  tasks: PlanTask[]
}

export interface Execution {
  id: string
  question: string
  status: ExecutionStatus
  route?: string | null
  answer?: string | null
  qa_flags?: string[] | null
  error?: ExecutionError | null
  created_at: string
  started_at?: string | null
  completed_at?: string | null
  result?: string | null
  /** Real persisted plan; null for simple routes, legacy rows, or corrupt blobs. */
  plan?: ExecutionPlan | null
}

export interface ExecutionEvent {
  event_id: number
  execution_id: string
  timestamp: string
  type: string
  message: string
  step_id?: string | null
  agent_id?: string | null
  severity: 'info' | 'warning' | 'error'
  payload: Record<string, unknown>
}

export interface AiReadiness {
  provider: string
  state: 'ready' | 'not_configured'
  what_will_be_tested: string
  state_hint: string
}

export interface Health {
  status: string
  backend: string
  ai: AiReadiness | null
  version?: string
  storage_error?: { human_message: string; why: string; recovery: string }
}
