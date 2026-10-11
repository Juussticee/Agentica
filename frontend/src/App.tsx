import { useEffect, useRef, useState } from 'react'
import Dashboard from './pages/Dashboard'
import NewExecution from './pages/NewExecution'
import Executions from './pages/Executions'
import ExecutionDetail from './pages/ExecutionDetail'
import Services from './pages/Services'
import Agents from './pages/Agents'
import TasksSequence from './pages/TasksSequence'
import Settings from './pages/Settings'

type View =
  | { name: 'dashboard' }
  | { name: 'new' }
  | { name: 'executions' }
  | { name: 'detail'; id: string }
  | { name: 'services' }
  | { name: 'agents' }
  | { name: 'tasks' }
  | { name: 'settings' }

// Hash routing keeps the multi-view shell testable without a router dependency.
function viewFromHash(): View {
  const h = window.location.hash.replace(/^#\/?/, '')
  if (h.startsWith('executions/')) {
    return { name: 'detail', id: decodeURIComponent(h.slice('executions/'.length)) }
  }
  if (h === 'executions') return { name: 'executions' }
  if (h === 'new') return { name: 'new' }
  if (h === 'services') return { name: 'services' }
  if (h === 'agents') return { name: 'agents' }
  if (h === 'tasks') return { name: 'tasks' }
  if (h === 'settings') return { name: 'settings' }
  return { name: 'dashboard' }
}

export default function App() {
  const [view, setView] = useState<View>(() => viewFromHash())
  const [navOpen, setNavOpen] = useState(false)
  const menuButtonRef = useRef<HTMLButtonElement>(null)
  const drawerRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const onHash = () => {
      setView(viewFromHash())
      // The mobile drawer must never stay open over the new page.
      setNavOpen(false)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // Escape closes the mobile drawer and returns focus to the menu button.
  useEffect(() => {
    if (!navOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setNavOpen(false)
        menuButtonRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navOpen])

  function navigate(to: string) {
    window.location.hash = `/${to}`
  }

  const navContent = (
    <>
      <div className="sidebar__group">Operate</div>
      <nav>
        <a className={view.name === 'dashboard' ? 'active' : ''} onClick={() => navigate('')} href="#/">Overview</a>
        <a className={view.name === 'new' ? 'active' : ''} onClick={() => navigate('new')} href="#/new">New Execution</a>
        <a className={view.name === 'executions' || view.name === 'detail' ? 'active' : ''} onClick={() => navigate('executions')} href="#/executions">Executions</a>
        <a className={view.name === 'tasks' ? 'active' : ''} onClick={() => navigate('tasks')} href="#/tasks">Tasks &amp; Sequence</a>
      </nav>
      <div className="sidebar__group">Monitor</div>
      <nav>
        <a className={view.name === 'services' ? 'active' : ''} onClick={() => navigate('services')} href="#/services">Services</a>
        <a className={view.name === 'agents' ? 'active' : ''} onClick={() => navigate('agents')} href="#/agents">Agents</a>
      </nav>
      <div className="sidebar__group">Configure</div>
      <nav>
        <a className={view.name === 'settings' ? 'active' : ''} onClick={() => navigate('settings')} href="#/settings">Settings</a>
      </nav>

      <div className="sidebar__planned">
        <div className="sidebar__planned-title">Planned (not implemented yet)</div>
        <span title="Planned: real approval workflow">Approval Center</span>
        <span title="Planned: knowledge management">Knowledge</span>
        <span title="Planned: chat entry surface">Chat</span>
        <span title="Planned: reports and artifacts">Reports</span>
      </div>
    </>
  )

  return (
    <div className={`app${navOpen ? ' app--nav-open' : ''}`}>
      {/* Compact mobile-only top bar: brand + menu toggle. Hidden on desktop. */}
      <header className="topbar">
        <button
          type="button"
          className="topbar__menu"
          ref={menuButtonRef}
          aria-label={navOpen ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={navOpen}
          aria-controls="sidebar-nav"
          onClick={() => setNavOpen((o) => !o)}
        >
          {navOpen ? '✕' : '☰'}
        </button>
        <span className="topbar__brand">Agentica Control Center</span>
      </header>

      {/* Backdrop: only rendered while the drawer is open; tap or focus exits
          it to close (display handled by CSS so the mobile block enables it). */}
      {navOpen && <div className="nav-backdrop" onMouseDown={() => setNavOpen(false)} />}

      <aside className="sidebar" id="sidebar-nav" ref={drawerRef}>
        <div className="sidebar__brand">Agentica Control Center</div>
        {navContent}
      </aside>

      <main className="main">
        {view.name === 'dashboard' && <Dashboard onNavigate={navigate} />}
        {view.name === 'new' && <NewExecution onNavigate={navigate} />}
        {view.name === 'executions' && <Executions />}
        {view.name === 'detail' && <ExecutionDetail id={view.id} />}
        {view.name === 'services' && <Services />}
        {view.name === 'agents' && <Agents />}
        {view.name === 'tasks' && <TasksSequence />}
        {view.name === 'settings' && <Settings />}
      </main>
    </div>
  )
}
