export type ExecutionStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

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
