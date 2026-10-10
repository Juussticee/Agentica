import { useEffect, useState } from 'react'
import Dashboard from './pages/Dashboard'
import NewExecution from './pages/NewExecution'
import Executions from './pages/Executions'
import ExecutionDetail from './pages/ExecutionDetail'
import Settings from './pages/Settings'

type View =
  | { name: 'dashboard' }
  | { name: 'new' }
  | { name: 'executions' }
  | { name: 'detail'; id: string }
  | { name: 'settings' }

// Hash routing keeps the multi-view shell testable without a router dependency.
function viewFromHash(): View {
  const h = window.location.hash.replace(/^#\/?/, '')
  if (h.startsWith('executions/')) {
    return { name: 'detail', id: decodeURIComponent(h.slice('executions/'.length)) }
  }
  if (h === 'executions') return { name: 'executions' }
  if (h === 'new') return { name: 'new' }
  if (h === 'settings') return { name: 'settings' }
  return { name: 'dashboard' }
}

export default function App() {
  const [view, setView] = useState<View>(() => viewFromHash())

  useEffect(() => {
    const onHash = () => setView(viewFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  function navigate(to: string) {
    window.location.hash = `/${to}`
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar__brand">Agentica Control Center</div>
        <nav>
          <a className={view.name === 'dashboard' ? 'active' : ''} onClick={() => navigate('')} href="#/">Dashboard</a>
          <a className={view.name === 'new' ? 'active' : ''} onClick={() => navigate('new')} href="#/new">New Execution</a>
          <a className={view.name === 'executions' || view.name === 'detail' ? 'active' : ''} onClick={() => navigate('executions')} href="#/executions">Executions</a>
          <a className={view.name === 'settings' ? 'active' : ''} onClick={() => navigate('settings')} href="#/settings">Settings</a>
        </nav>

        <div className="sidebar__planned">
          <div className="sidebar__planned-title">Planned (not implemented yet)</div>
          <span title="Planned: real approval workflow">Approval Center</span>
          <span title="Planned: knowledge management">Knowledge</span>
          <span title="Planned: chat entry surface">Chat</span>
          <span title="Planned: reports and artifacts">Reports</span>
        </div>
      </aside>

      <main className="main">
        {view.name === 'dashboard' && <Dashboard onNavigate={navigate} />}
        {view.name === 'new' && <NewExecution onNavigate={navigate} />}
        {view.name === 'executions' && (
          <Executions
            onNavigate={navigate}
            openDetail={(id) => navigate(`executions/${encodeURIComponent(id)}`)}
          />
        )}
        {view.name === 'detail' && <ExecutionDetail id={view.id} />}
        {view.name === 'settings' && <Settings />}
      </main>
    </div>
  )
}
