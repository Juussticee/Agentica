import type { Execution } from './types'

export type ServiceState =
  | 'running' | 'stopped' | 'unreachable' | 'degraded'
  | 'misconfigured' | 'not_configured' | 'unknown'

export interface ServiceStatus {
  id: string
  name: string
  purpose: string
  required: boolean
  state: ServiceState
  reason: string | null
  latency_ms: number | null
  last_checked: string | null
  next_action: string | null
  config: Record<string, unknown>
}

export interface ServicesSnapshot {
  checked_at: string | null
  services: ServiceStatus[]
  ai_provider?: { provider: string; state: string; settings_url: string; note: string }
  error?: { human_message: string; why: string; recovery: string }
}

export interface AgentActivity {
  node: string
  role: string
  status: 'idle' | 'working' | 'completed' | 'failed'
  execution_id: string | null
  task: string | null
  last_event_at: string | null
  last_event: string | null
  last_error: string | null
}

export interface AgentActivityResponse {
  agents: AgentActivity[]
  active_count: number
  failed_count: number
  note: string
}

export interface SequenceStage {
  stage: string
  node: string | null
  status: 'running' | 'completed' | 'failed'
  at: string
  event_id: number
  message: string
  route?: string
  qa_flags?: string[]
  failure_category?: string
  next_action?: string
  error?: unknown
}

export interface ExecutionSequence {
  execution_id: string
  status: string
  route: string | null
  question: string
  stages: SequenceStage[]
  model: string
}

export async function getServices(): Promise<ServicesSnapshot> {
  const res = await fetch('/api/v1/services')
  if (!res.ok) throw new Error(`Service checks failed (${res.status})`)
  return res.json()
}

export async function getAgentActivity(): Promise<AgentActivityResponse> {
  const res = await fetch('/api/v1/agents/activity')
  if (!res.ok) throw new Error(`Agent activity failed (${res.status})`)
  return res.json()
}

export async function getSequence(id: string): Promise<ExecutionSequence> {
  const res = await fetch(`/api/v1/executions/${encodeURIComponent(id)}/sequence`)
  if (!res.ok) {
    let msg = `Sequence failed (${res.status})`
    try { const d = await res.json(); msg = d?.detail?.human_message ?? msg } catch { /* */ }
    throw new Error(msg)
  }
  return res.json()
}

export type { Execution }
