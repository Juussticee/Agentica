// F1 regression tests, render level: a malformed hash must never leave the
// app unusable — neither on initial load nor from a later hashchange — and
// the top-level ErrorBoundary must recover from unexpected render crashes.
import '@testing-library/jest-dom/vitest'
import { useState } from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import ErrorBoundary from './components/ErrorBoundary'

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.includes('/events')) {
      return jsonResponse({ execution_id: 'abc/123', after: 0, events: [], count: 0 })
    }
    if (url.includes('/health')) {
      return jsonResponse({ status: 'ok', backend: 'ok', ai: null, version: 'test' })
    }
    if (url.includes('/ai/settings')) {
      return jsonResponse({
        llm_provider: 'ollama', gemini_api_key_masked: null,
        gemini_model: null, main_model: null, ollama_base_url: null, source: {},
      })
    }
    if (url.includes('/executions/abc')) {
      return jsonResponse({
        id: 'abc/123', question: 'q?', status: 'completed', route: null,
        answer: null, qa_flags: null, error: null, created_at: new Date().toISOString(),
        plan: null,
      })
    }
    if (url.includes('/executions')) {
      return jsonResponse({ executions: [], count: 0 })
    }
    return jsonResponse({})
  }))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  window.location.hash = '#/'
})

describe('App with malformed hashes (F1)', () => {
  it('renders the dashboard for a malformed initial hash instead of crashing', async () => {
    window.location.hash = '#/executions/%E0%A4%A'
    render(<App />)
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Dashboard' }),
    ).toBeInTheDocument()
  })

  it('falls back on a malformed hashchange and stays fully navigable', async () => {
    window.location.hash = '#/'
    render(<App />)
    await screen.findByRole('heading', { level: 1, name: 'Dashboard' })

    window.location.hash = '#/executions/%zz'
    fireEvent(window, new HashChangeEvent('hashchange'))
    expect(screen.getByRole('heading', { level: 1, name: 'Dashboard' })).toBeInTheDocument()

    // The app is still usable afterwards: a valid hash still navigates.
    window.location.hash = '#/settings'
    fireEvent(window, new HashChangeEvent('hashchange'))
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Settings' }),
    ).toBeInTheDocument()
  })

  it('still decodes valid encoded execution ids into the detail view', async () => {
    window.location.hash = '#/executions/abc%2F123'
    render(<App />)
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Execution #abc/123' }),
    ).toBeInTheDocument()
  })
})

describe('ErrorBoundary (top-level recovery)', () => {
  it('catches a render crash, shows recovery UI, and remounts cleanly afterwards', () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    // The crash is triggered by an update (not the initial mount) so the
    // boundary catches it deterministically under React 19's concurrent
    // initial render. Once the boundary has engaged, a fresh remount of the
    // child tree (crash condition gone) must render normally again.
    function Flaky() {
      const [crash, setCrash] = useState(false)
      if (crash) throw new Error('simulated render crash')
      return (
        <>
          <div data-testid="ok-child">fine for now</div>
          <button type="button" onClick={() => setCrash(true)}>trigger crash</button>
        </>
      )
    }

    window.location.hash = '#/settings'
    render(
      <ErrorBoundary>
        <Flaky />
      </ErrorBoundary>,
    )
    expect(screen.getByTestId('ok-child')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'trigger crash' }))
    expect(screen.queryByTestId('ok-child')).toBeNull()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Something went wrong displaying this page.',
    )
    expect(screen.getByText('simulated render crash')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Back to overview' }))
    expect(screen.getByTestId('ok-child')).toBeInTheDocument()
    expect(window.location.hash).toBe('#/')
    errSpy.mockRestore()
  })
})
