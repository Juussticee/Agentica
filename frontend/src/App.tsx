import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

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

  // Minimal inline stroke icons so the collapsible rail stays dependency-free.
  function Icon({ children }: { children: ReactNode }) {
    return (
      <svg
        className="nav-icon"
        width="18" height="18" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2" strokeLinecap="round"
        strokeLinejoin="round" aria-hidden="true" focusable="false"
      >
        {children}
      </svg>
    )
  }
  const ICON_HOME = <><path d="M3 10.5 12 3l9 7.5" /><path d="M5 9.5V21h14V9.5" /></>
  const ICON_NEW = <><circle cx="12" cy="12" r="9" /><path d="M12 8v8M8 12h8" /></>
  const ICON_HISTORY = <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3.5 2.5" /></>
  const ICON_TASKS = <><path d="M4 6h16M4 12h16M4 18h10" /></>
  const ICON_PULSE = <><path d="M3 12h4l2.5-7 5 14 2.5-7H21" /></>
  const ICON_BOT = <><rect x="4" y="8" width="16" height="12" rx="2" /><path d="M12 4v4M9 14h.01M15 14h.01" /></>
  const ICON_GEAR = <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.03 1.56V21a2 2 0 1 1-4 0v-.09A1.7 1.7 0 0 0 9 19.35a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 4.65 15 1.7 1.7 0 0 0 3.08 14H3a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 4.65 9a1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.7 1.7 0 0 0 9 4.65h.08A1.7 1.7 0 0 0 10.11 3.09V3a2 2 0 1 1 4 0v.09c0 .68.4 1.3 1.03 1.56.6.26 1.35.13 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.7 1.7 0 0 0 19.35 9c.26.62.88 1.03 1.56 1.03H21a2 2 0 1 1 0 4h-.09c-.68 0-1.3.4-1.51.97Z" /></>

  type NavItem = { to: string; key: string; label: string; icon: ReactNode; active: boolean }
  const operateItems: NavItem[] = [
    { to: '', key: '', label: 'Overview', icon: ICON_HOME, active: view.name === 'dashboard' },
    { to: 'new', key: 'new', label: 'New Execution', icon: ICON_NEW, active: view.name === 'new' },
    { to: 'executions', key: 'executions', label: 'Executions', icon: ICON_HISTORY, active: view.name === 'executions' || view.name === 'detail' },
    { to: 'tasks', key: 'tasks', label: 'Tasks & Sequence', icon: ICON_TASKS, active: view.name === 'tasks' },
  ]
  const monitorItems: NavItem[] = [
    { to: 'services', key: 'services', label: 'Services', icon: ICON_PULSE, active: view.name === 'services' },
    { to: 'agents', key: 'agents', label: 'Agents', icon: ICON_BOT, active: view.name === 'agents' },
  ]
  const configureItems: NavItem[] = [
    { to: 'settings', key: 'settings', label: 'Settings', icon: ICON_GEAR, active: view.name === 'settings' },
  ]

  function NavGroup({ title, items }: { title: string; items: NavItem[] }) {
    return (
      <>
        <div className="sidebar__group">{title}</div>
        <nav>
          {items.map((item) => (
            <a
              key={item.to}
              className={item.active ? 'active' : ''}
              onClick={() => navigate(item.to)}
              href={item.to ? `#/${item.to}` : '#/'}
              title={item.label}
              aria-label={item.label}
              aria-current={item.active ? 'page' : undefined}
            >
              <Icon>{item.icon}</Icon>
              <span className="nav-label">{item.label}</span>
            </a>
          ))}
        </nav>
      </>
    )
  }

  const navContent = (
    <>
      <NavGroup title="Operate" items={operateItems} />
      <NavGroup title="Monitor" items={monitorItems} />
      <NavGroup title="Configure" items={configureItems} />

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
        <div className="main-inner">
          {view.name === 'dashboard' && <Dashboard onNavigate={navigate} />}
          {view.name === 'new' && <NewExecution onNavigate={navigate} />}
          {view.name === 'executions' && <Executions />}
          {view.name === 'detail' && <ExecutionDetail id={view.id} />}
          {view.name === 'services' && <Services />}
          {view.name === 'agents' && <Agents />}
          {view.name === 'tasks' && <TasksSequence />}
          {view.name === 'settings' && <Settings />}
        </div>
      </main>
    </div>
  )
}
