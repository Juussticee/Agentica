// F1 regression: hash parsing must be total. The previous implementation
// called decodeURIComponent unguarded, so a hash like #/executions/%E0%A4%A
// threw URIError — during useState initialization (white screen on load) or
// inside the hashchange handler (broken navigation).
import { describe, expect, it } from 'vitest'
import { viewFromHash } from './routes'

describe('viewFromHash (malformed-hash hardening)', () => {
  it('routes every known view', () => {
    expect(viewFromHash('#/')).toEqual({ name: 'dashboard' })
    expect(viewFromHash('')).toEqual({ name: 'dashboard' })
    expect(viewFromHash('#/new')).toEqual({ name: 'new' })
    expect(viewFromHash('#/executions')).toEqual({ name: 'executions' })
    expect(viewFromHash('#/services')).toEqual({ name: 'services' })
    expect(viewFromHash('#/agents')).toEqual({ name: 'agents' })
    expect(viewFromHash('#/tasks')).toEqual({ name: 'tasks' })
    expect(viewFromHash('#/settings')).toEqual({ name: 'settings' })
    expect(viewFromHash('#/executions/abc123')).toEqual({ name: 'detail', id: 'abc123' })
  })

  it('decodes valid percent-encoding in execution ids', () => {
    expect(viewFromHash('#/executions/ab%2Fcd')).toEqual({ name: 'detail', id: 'ab/cd' })
  })

  it('falls back to the dashboard for malformed percent-encoding (never throws)', () => {
    const malformed = [
      '#/executions/%E0%A4%A',
      '#/executions/%zz',
      '#/executions/%',
      '#/executions/100%',
    ]
    for (const hash of malformed) {
      expect(() => viewFromHash(hash)).not.toThrow()
      expect(viewFromHash(hash)).toEqual({ name: 'dashboard' })
    }
  })

  it('falls back to the dashboard for an empty or unknown route', () => {
    expect(viewFromHash('#/executions/')).toEqual({ name: 'dashboard' })
    expect(viewFromHash('#/no-such-page')).toEqual({ name: 'dashboard' })
  })
})
