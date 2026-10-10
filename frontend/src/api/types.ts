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
