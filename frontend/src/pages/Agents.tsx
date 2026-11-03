import { useEffect, useMemo, useState } from 'react'
import {
  getAgentCatalog,
  type AgentCatalog,
  type AgentEntry,
  type Availability,
} from '../api/catalog'
import { getAgentActivity, type AgentActivity } from '../api/dashboard'
import { Empty, ErrorCard, Loading } from '../components/ui'
import { allowlistedClass } from '../lib/classSuffix'

// Live-activity statuses recorded by the backend (app/api/dashboard.py); any
// other value renders the neutral base class with its raw label.
const ACTIVITY_LABEL: Record<string, string> = {
  idle: 'Idle',
  working: 'Working',
  completed: 'Completed',
  failed: 'Failed',
}

// Activity statuses reuse the existing execution badge palette; the class is
// still chosen through an allowlist, never interpolated from API data.
const ACTIVITY_BADGE: Record<string, string> = {
  idle: 'queued',
  working: 'running',
  completed: 'completed',
  failed: 'failed',
}

/** One agent's last recorded state. Exported for the allowlist tests. */
export function AgentRow({ a }: { a: AgentActivity }) {
  const name = ACTIVITY_LABEL[a.status] ?? a.status
  return (
    <article className={allowlistedClass('agent-row', 'agent-row--', Object.keys(ACTIVITY_LABEL), a.status)}>
      <div className="agent-row__main">
        <strong>{a.node}</strong>
        <span className="agent-row__tech muted">node: {a.node}</span>
        <div className="agent-row__role muted">{a.role}</div>
      </div>
      <div className="agent-row__side">
        <span
          className={allowlistedClass(
            'badge', 'badge--',
            Object.keys(ACTIVITY_LABEL).map((key) => ACTIVITY_BADGE[key]),
            ACTIVITY_BADGE[a.status],
          )}
        >
          {name}
        </span>
        {a.execution_id ? (
          <a href={`#/executions/${encodeURIComponent(a.execution_id)}`}>
            #{a.execution_id}
          </a>
        ) : (
          <span className="muted">—</span>
        )}
      </div>
      {a.task && <p className="agent-row__task muted">Task: {a.task}</p>}
      {a.last_event && (
        <p className="agent-row__event">
          Last event: {a.last_event}
          {a.last_event_at && <> · {new Date(a.last_event_at).toLocaleTimeString()}</>}
        </p>
      )}
      {a.last_error && <p className="agent-row__error">{a.last_error}</p>}
    </article>
  )
}

const AVAILABILITY_LABEL: Record<Availability, string> = {
  available: 'Available',
  unavailable: 'Unavailable',
  disabled: 'Disabled',
}

const AVAILABILITY_HELP: Record<Availability, string> = {
  available: 'Could run now, given the current configuration.',
  unavailable: 'Configured but cannot run as things stand.',
  disabled: 'Intentionally switched off; nothing is missing.',
}

type Filter = 'all' | Availability

function permissionText(value: boolean | string): string {
  if (value === true) return 'yes'
  if (value === false) return 'no'
  return String(value)
}

function AvailabilityBadge({ value }: { value: Availability }) {
  return (
    <span className={`badge badge--${value}`} title={AVAILABILITY_HELP[value]}>
      {AVAILABILITY_LABEL[value]}
    </span>
  )
}

function AgentCard({
  agent,
  onInspect,
  selected,
}: {
  agent: AgentEntry
  onInspect: () => void
  selected: boolean
}) {
  const recent = agent.recent_activity[0]
  return (
    <article className={`agent-card${selected ? ' agent-card--selected' : ''}`}>
      <header className="agent-card__head">
        <div>
          <h3>{agent.title}</h3>
          <code className="agent-card__id">{agent.id}</code>
        </div>
        <AvailabilityBadge value={agent.availability} />
      </header>
      <p className="agent-card__resp">{agent.responsibility}</p>
      <p className="muted agent-card__reason">{agent.availability_reason}</p>
      <dl className="agent-card__facts">
        <div><dt>Tools</dt><dd>{agent.tools.length}</dd></div>
        <div><dt>Planner</dt><dd>{agent.planner_capability ?? 'direct route'}</dd></div>
        <div>
          <dt>Approval</dt>
          <dd>{agent.permissions.approval_required ? 'required' : 'not needed'}</dd>
        </div>
      </dl>
      <p className="muted agent-card__recent">
        {recent ? (
          <>
            Last run: <a href={`#/executions/${encodeURIComponent(recent.execution_id)}`}>
              #{recent.execution_id}
            </a>{' '}
            · {recent.question}
          </>
        ) : (
          'No recorded execution yet.'
        )}
      </p>
      <button type="button" className="btn btn--ghost" onClick={onInspect}>
        {selected ? 'Hide details' : 'Inspect agent'}
      </button>
    </article>
  )
}

function AgentDetail({ agent, onClose }: { agent: AgentEntry; onClose: () => void }) {
  const perms = agent.permissions
  return (
    <section className="card agent-detail" aria-label={`${agent.title} details`}>
      <header className="agent-detail__head">
        <div>
          <h2>{agent.title}</h2>
          <code className="muted">node: {agent.node ?? 'plan capability only'}</code>
        </div>
        <div className="agent-detail__head-actions">
          <AvailabilityBadge value={agent.availability} />
          <button type="button" className="btn btn--ghost" onClick={onClose}>Close</button>
        </div>
      </header>

      <div className="agent-detail__grid">
        <div>
          <h3>Responsibility</h3>
          <p>{agent.responsibility}</p>
          <p className="muted">{agent.availability_reason}</p>

          <h3>Supported work</h3>
          <ul className="plain-list">
            <li>Direct route: {agent.routes.length ? agent.routes.join(', ') : 'none (plan step only)'}</li>
            <li>Planner capability: {agent.planner_capability ?? 'not offered to the planner'}</li>
            <li>Model use: {agent.model}</li>
          </ul>

          <h3>Refuses</h3>
          <p className="muted">{agent.rejects}</p>
        </div>

        <div>
          <h3>Tools</h3>
          {agent.tools.length === 0 ? (
            <p className="muted">No tools — this agent answers or decides only.</p>
          ) : (
            <ul className="plain-list">
              {agent.tools.map((tool) => <li key={tool}>{tool}</li>)}
            </ul>
          )}

          <h3>Permissions</h3>
          <dl className="kv">
            <dt>Network</dt><dd>{permissionText(perms.network)}</dd>
            <dt>Filesystem</dt><dd>{permissionText(perms.filesystem)}</dd>
            <dt>Writes</dt><dd>{permissionText(perms.writes)}</dd>
            <dt>Approval required</dt><dd>{perms.approval_required ? 'yes' : 'no'}</dd>
          </dl>

          <h3>State</h3>
          <dl className="kv">
            <dt>Reads</dt><dd>{agent.reads.join(', ') || '—'}</dd>
            <dt>Writes</dt><dd>{agent.writes.join(', ') || '—'}</dd>
            <dt>Tests</dt><dd className="mono-small">{agent.tests}</dd>
          </dl>
        </div>
      </div>

      <h3>Recent activity</h3>
      {agent.recent_activity.length === 0 ? (
        <p className="muted">This agent has not been part of a recorded execution yet.</p>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Execution</th><th>Request</th><th>Run status</th>
                <th>Task</th><th>Task status</th>
              </tr>
            </thead>
            <tbody>
              {agent.recent_activity.map((item) => (
                <tr key={`${item.execution_id}-${item.task_id ?? 'direct'}`}>
                  <td>
                    <a href={`#/executions/${encodeURIComponent(item.execution_id)}`}>
                      #{item.execution_id}
                    </a>
                  </td>
                  <td className="truncate">{item.question}</td>
                  <td>{item.execution_status}</td>
                  <td>{item.task_id ?? '—'}</td>
                  <td>{item.task_status ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default function Agents({ agentId }: { agentId?: string }) {
  const [catalog, setCatalog] = useState<AgentCatalog | null>(null)
  const [activity, setActivity] = useState<AgentActivity[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string | null>(agentId ?? null)

  useEffect(() => {
    setSelected(agentId ?? null)
  }, [agentId])

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const [cat, act] = await Promise.all([
          getAgentCatalog(),
          getAgentActivity().catch(() => null),
        ])
        if (!alive) return
        setCatalog(cat)
        setActivity(act?.agents ?? null)
        setError(null)
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e))
      }
    }
    load()
    return () => { alive = false }
  }, [])

  const agents = catalog?.agents ?? []
  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return agents.filter((agent) => {
      if (filter !== 'all' && agent.availability !== filter) return false
      if (!needle) return true
      return (
        agent.id.toLowerCase().includes(needle) ||
        agent.title.toLowerCase().includes(needle) ||
        agent.responsibility.toLowerCase().includes(needle) ||
        agent.tools.some((tool) => tool.toLowerCase().includes(needle))
      )
    })
  }, [agents, filter, query])

  const selectedAgent = agents.find((agent) => agent.id === selected) ?? null

  function inspect(id: string) {
    const next = selected === id ? null : id
    setSelected(next)
    window.location.hash = next ? `#/agents/${encodeURIComponent(next)}` : '#/agents'
  }

  if (error) {
    return (
      <div>
        <h1>Agents</h1>
        <ErrorCard
          message={error}
          recovery="Check that the backend is running, then refresh."
        />
      </div>
    )
  }
  if (!catalog) return <div><h1>Agents</h1><Loading label="Reading the agent registry…" /></div>

  return (
    <div>
      <h1>Agents</h1>
      <p className="muted agents__intro">
        {catalog.counts.registered} registered specialist(s):{' '}
        {catalog.counts.available} available · {catalog.counts.unavailable} unavailable ·{' '}
        {catalog.counts.disabled} disabled. {catalog.note}
      </p>

      <div className="toolbar">
        <div className="segmented" role="group" aria-label="Filter by availability">
          {(['all', 'available', 'unavailable', 'disabled'] as Filter[]).map((value) => (
            <button
              key={value}
              type="button"
              className={filter === value ? 'segmented__item active' : 'segmented__item'}
              aria-pressed={filter === value}
              onClick={() => setFilter(value)}
            >
              {value === 'all' ? 'All' : AVAILABILITY_LABEL[value]}
            </button>
          ))}
        </div>
        <input
          type="search"
          className="toolbar__search"
          placeholder="Search agents, tools…"
          aria-label="Search agents"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {shown.length === 0 ? (
        <Empty>No agent matches this filter.</Empty>
      ) : (
        <div className="agent-grid">
          {shown.map((agent) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              selected={selected === agent.id}
              onInspect={() => inspect(agent.id)}
            />
          ))}
        </div>
      )}

      {selectedAgent && (
        <AgentDetail agent={selectedAgent} onClose={() => inspect(selectedAgent.id)} />
      )}

      <section className="card">
        <h2>Orchestration ({catalog.orchestration.length})</h2>
        <p className="muted">
          These decide and verify; they never produce the work themselves.
        </p>
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Stage</th><th>Responsibility</th></tr></thead>
            <tbody>
              {catalog.orchestration.map((entry) => (
                <tr key={entry.id}>
                  <td><strong>{entry.title}</strong><div className="muted mono-small">{entry.id}</div></td>
                  <td>{entry.does}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Live activity</h2>
        <p className="muted">
          From recorded execution events only. An agent shows as working only
          while a run says so.
        </p>
        {activity === null ? (
          <p className="muted">Live activity is unavailable right now.</p>
        ) : activity.length === 0 ? (
          <Empty>No agent has recorded activity yet.</Empty>
        ) : (
          <div className="agent-rows">
            {activity.map((row) => <AgentRow key={row.node} a={row} />)}
          </div>
        )}
      </section>

      <section className="card">
        <h2>Planned, not implemented ({catalog.planned.length})</h2>
        <p className="muted">
          Roles Agentica does not claim. They are listed here instead of being
          registered as agents with a prompt and no capability.
        </p>
        <ul className="plain-list">
          {catalog.planned.map((item) => (
            <li key={item.id}>
              <strong>{item.title}</strong> — {item.reason}
            </li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h2>Provider</h2>
        <dl className="kv">
          <dt>Provider</dt><dd>{catalog.provider.provider ?? 'unknown'}</dd>
          <dt>State</dt><dd>{catalog.provider.state ?? 'unknown'}</dd>
          <dt>Model</dt><dd>{catalog.provider.model ?? '—'}</dd>
        </dl>
        <p className="muted">{catalog.provider.note}</p>
      </section>
    </div>
  )
}
