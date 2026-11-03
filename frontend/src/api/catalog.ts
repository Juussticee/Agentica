// Typed access to the read-only control-center projections. Every field here
// comes from the backend (app/api/control_center.py); nothing is derived in
// the browser beyond formatting.

export type Availability = 'available' | 'unavailable' | 'disabled'

export interface AgentPermission {
  network: boolean | string
  filesystem: boolean | string
  writes: boolean | string
  approval_required: boolean
}

export interface AgentActivityItem {
  execution_id: string
  question: string
  execution_status: string
  at: string | null
  task_id: string | null
  task_status: string | null
}

export interface AgentEntry {
  id: string
  title: string
  kind: string
  responsibility: string
  availability: Availability
  availability_reason: string
  tools: string[]
  permissions: AgentPermission
  model: string
  routes: string[]
  planner_capability: string | null
  node: string | null
  reads: string[]
  writes: string[]
  rejects: string
  tests: string
  implemented: boolean
  recent_activity: AgentActivityItem[]
}

export interface OrchestrationEntry {
  id: string
  title: string
  does: string
  availability: string
  availability_reason: string
  tests: string
}

export interface ProviderFacts {
  provider: string | null
  state: string | null
  model: string | null
  note: string | null
}

export interface PlannedCapability {
  id: string
  title: string
  reason: string
}

export interface AgentCatalog {
  checked_at: string
  provider: ProviderFacts
  agents: AgentEntry[]
  orchestration: OrchestrationEntry[]
  counts: { registered: number; available: number; disabled: number; unavailable: number }
  planned: PlannedCapability[]
  note: string
}

export interface SkillEntry {
  id: string
  title: string
  purpose: string
  path: string
  installed: boolean
  size_bytes: number | null
  modified_at: string | null
  kind: string
  executable: boolean
  owner_agent: string | null
  required_tools: string[]
  permissions: string
}

export interface SkillsCatalog {
  checked_at: string
  skills: SkillEntry[]
  count: number
  installed: number
  can_install: boolean
  note: string
}

export interface MemoryItem {
  id?: string
  name?: string
  path?: string
  question?: string
  status?: string
  created_at?: string
  completed_at?: string | null
  size_bytes?: number
  modified_at?: string
}

export interface MemorySection {
  id: string
  title: string
  status: 'available' | 'empty' | 'not_implemented'
  summary: string
  items: MemoryItem[]
  provenance: string | null
  counts?: Record<string, number | Record<string, number>>
  index_present?: boolean | null
  index_path?: string
  books_dir?: string
  company_items?: { name: string; path: string }[]
}

export interface MemorySummary {
  checked_at: string
  sections: MemorySection[]
  counts: {
    executions: { total: number; by_status: Record<string, number> }
    approvals: { total: number; by_status: Record<string, number> }
    events: number
    oldest: string | null
    newest: string | null
  }
  note: string
}

export interface ActivityItem {
  at: string
  type: string
  message: string
  severity: 'info' | 'warning' | 'error'
  agent_id: string | null
  execution_id: string
}

export interface RecentExecution {
  id: string
  question: string
  status: string
  route: string | null
  created_at: string
  completed_at: string | null
  has_plan: boolean
}

export interface AttentionItem {
  id: string
  question: string
  status: string
  category: string | null
  human_message: string | null
  created_at: string
}

export interface DashboardSummary {
  checked_at: string
  provider: ProviderFacts
  agents: {
    registered: number
    available: number
    disabled: number
    unavailable: number
    orchestration: number
  }
  skills: { installed: number; total: number }
  executions: {
    total: number
    by_status: Record<string, number>
    active: number
    waiting_approval: number
    failed: number
    last_24h: number
  }
  approvals: { pending: number }
  attention: AttentionItem[]
  recent_executions: RecentExecution[]
  activity: ActivityItem[]
  health: {
    backend: string
    ai: { state: string | null; provider: string | null; model: string | null }
    storage: string
  }
  note: string
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const body = await res.json()
      message = body?.detail?.human_message ?? message
    } catch {
      /* body not JSON */
    }
    throw new Error(message)
  }
  return res.json() as Promise<T>
}

export function getAgentCatalog(): Promise<AgentCatalog> {
  return getJson<AgentCatalog>('/api/v1/agents')
}

export function getSkills(): Promise<SkillsCatalog> {
  return getJson<SkillsCatalog>('/api/v1/skills')
}

export function getMemory(): Promise<MemorySummary> {
  return getJson<MemorySummary>('/api/v1/memory')
}

export function getDashboardSummary(): Promise<DashboardSummary> {
  // The overview reads several counts from one payload. A backend that is
  // older than this page (or a proxy answering with an empty object) must
  // surface as an honest error card, never as a render crash.
  return getJson<DashboardSummary>('/api/v1/dashboard').then((summary) => {
    if (!summary || !summary.executions || !summary.agents || !summary.approvals) {
      throw new Error(
        'The backend returned an incomplete dashboard summary. It may be an older build than this page.',
      )
    }
    return summary
  })
}

export interface ExecutionFilters {
  status?: string
  q?: string
  since?: string
  until?: string
  limit?: number
}

export function executionQuery(filters: ExecutionFilters): string {
  const params = new URLSearchParams()
  if (filters.status) params.set('status', filters.status)
  if (filters.q) params.set('q', filters.q)
  if (filters.since) params.set('since', filters.since)
  if (filters.until) params.set('until', filters.until)
  if (filters.limit) params.set('limit', String(filters.limit))
  const query = params.toString()
  return query ? `?${query}` : ''
}
