// Hash routing keeps the multi-view shell testable without a router
// dependency. The hash is attacker-influencable (any link or script can set
// it), so parsing must be total: never throw, always return a renderable
// view. A malformed hash used to throw URIError from decodeURIComponent and
// white-screen the app on load or on hashchange.
export type View =
  | { name: 'dashboard' }
  | { name: 'new' }
  | { name: 'executions' }
  | { name: 'detail'; id: string }
  | { name: 'services' }
  | { name: 'agents' }
  | { name: 'tasks' }
  | { name: 'settings' }

export function viewFromHash(hash: string = window.location.hash): View {
  const h = hash.replace(/^#\/?/, '')
  if (h.startsWith('executions/')) {
    let id: string
    try {
      id = decodeURIComponent(h.slice('executions/'.length))
    } catch {
      // Malformed percent-encoding (e.g. #/executions/%E0%A4%A): fall back
      // to the dashboard instead of crashing the whole app.
      return { name: 'dashboard' }
    }
    // An execution id is a non-empty hex string; "#/executions/" with no id
    // is not a valid route.
    if (!id) return { name: 'dashboard' }
    return { name: 'detail', id }
  }
  if (h === 'executions') return { name: 'executions' }
  if (h === 'new') return { name: 'new' }
  if (h === 'services') return { name: 'services' }
  if (h === 'agents') return { name: 'agents' }
  if (h === 'tasks') return { name: 'tasks' }
  if (h === 'settings') return { name: 'settings' }
  return { name: 'dashboard' }
}
