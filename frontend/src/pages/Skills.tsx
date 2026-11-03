import { useEffect, useState } from 'react'
import { getSkills, type SkillsCatalog } from '../api/catalog'
import { getLearning, type SkillVersion } from '../api/learning'
import { Empty, ErrorCard, Loading } from '../components/ui'

function formatBytes(bytes: number | null): string {
  if (bytes === null) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function Skills() {
  const [data, setData] = useState<SkillsCatalog | null>(null)
  const [learned, setLearned] = useState<SkillVersion[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    getSkills()
      .then((catalog) => { if (alive) { setData(catalog); setError(null) } })
      .catch((e) => { if (alive) setError(e instanceof Error ? e.message : String(e)) })
    // Learned versions come from the learning summary; a backend that cannot
    // answer still leaves the documented list usable (empty, never invented).
    getLearning()
      .then((summary) => { if (alive) setLearned(summary.skills) })
      .catch(() => { if (alive) setLearned([]) })
    return () => { alive = false }
  }, [])

  if (error) {
    return (
      <div>
        <h1>Skills</h1>
        <ErrorCard message={error} recovery="Check that the backend is running, then refresh." />
      </div>
    )
  }
  if (!data) return <div><h1>Skills</h1><Loading label="Reading skill documents…" /></div>

  return (
    <div>
      <h1>Skills</h1>
      <p className="muted skills__intro">
        A <strong>skill</strong> is a reusable procedure kept as a document. It
        is not an <strong>agent</strong> (who does the work) and not a{' '}
        <strong>tool</strong> (what an agent may call). {data.note}
      </p>

      <div className="summary-row">
        <div className="summary-tile">
          <span className="summary-tile__value">{data.count}</span>
          <span className="summary-tile__label">Documented procedures</span>
        </div>
        <div className="summary-tile">
          <span className="summary-tile__value">{data.installed}</span>
          <span className="summary-tile__label">Present on disk</span>
        </div>
        <div className="summary-tile">
          <span className="summary-tile__value">{data.can_install ? 'yes' : 'no'}</span>
          <span className="summary-tile__label">Installable from the UI</span>
        </div>
      </div>

      {data.skills.length === 0 ? (
        <Empty>No skill documents are registered.</Empty>
      ) : (
        <div className="skill-list">
          {data.skills.map((skill) => (
            <article key={skill.id} className="card skill-card">
              <header className="skill-card__head">
                <div>
                  <h2>{skill.title}</h2>
                  <code className="muted mono-small">{skill.path}</code>
                </div>
                <span className={`badge badge--${skill.installed ? 'available' : 'unavailable'}`}>
                  {skill.installed ? 'Present' : 'Missing file'}
                </span>
              </header>
              <p>{skill.purpose}</p>
              <dl className="kv">
                <dt>Kind</dt><dd>{skill.kind}</dd>
                <dt>Executable</dt><dd>{skill.executable ? 'yes' : 'no — documentation only'}</dd>
                <dt>Owner agent</dt><dd>{skill.owner_agent ?? 'none (shared documentation)'}</dd>
                <dt>Required tools</dt>
                <dd>{skill.required_tools.length ? skill.required_tools.join(', ') : 'none'}</dd>
                <dt>Permissions</dt><dd>{skill.permissions}</dd>
                <dt>Last modified</dt>
                <dd>
                  {skill.modified_at
                    ? new Date(skill.modified_at).toLocaleString()
                    : '—'}
                  {skill.size_bytes !== null && <> · {formatBytes(skill.size_bytes)}</>}
                </dd>
              </dl>
              <p className="muted">
                Execution history: skills are not executed, so they have no run
                history of their own. Use the Agents page for who ran what.
              </p>
            </article>
          ))}
        </div>
      )}

      <h2>Learned versions</h2>
      <p className="muted">
        Procedures derived from your corrections. An active version is applied to
        matching tasks; every earlier version is kept for rollback. Manage them
        on the Learning page.
      </p>
      {learned.length === 0 ? (
        <Empty>No learned skill version yet.</Empty>
      ) : (
        <div className="skill-list">
          {learned.map((version) => (
            <article key={version.version_id} className="card skill-card">
              <header className="skill-card__head">
                <div>
                  <h2>
                    {version.skill_id} v{version.version}
                  </h2>
                  <code className="muted mono-small">
                    {version.source_lesson_ids.length} source lesson(s)
                  </code>
                </div>
                <span className={`badge badge--${version.status === 'active' ? 'available' : 'unknown'}`}>
                  {version.status}
                </span>
              </header>
              <dl className="kv">
                <dt>Kind</dt><dd>learned procedure version</dd>
                <dt>Evaluation</dt>
                <dd>
                  {version.passed === null
                    ? 'not evaluated'
                    : version.passed
                      ? 'passed the deterministic gate'
                      : 'failed the deterministic gate'}
                </dd>
                <dt>Created</dt>
                <dd>{version.created_at ? new Date(version.created_at).toLocaleString() : '—'}</dd>
              </dl>
            </article>
          ))}
        </div>
      )}
    </div>
  )
}
