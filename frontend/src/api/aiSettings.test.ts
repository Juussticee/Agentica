// F3 regression: the failure category arrives from the API, so the guidance
// lookup must not resolve inherited properties. CATEGORY_GUIDANCE is a plain
// object literal, so CATEGORY_GUIDANCE['__proto__'] used to return
// Object.prototype (truthy → nonsense guidance rendered).
import { describe, expect, it } from 'vitest'
import { CATEGORY_GUIDANCE, categoryGuidance } from './aiSettings'

describe('categoryGuidance (prototype-safe lookup)', () => {
  it('returns the exact guidance for every documented category', () => {
    for (const key of Object.keys(CATEGORY_GUIDANCE)) {
      expect(categoryGuidance(key)).toBe(CATEGORY_GUIDANCE[key])
    }
  })

  it('preserves the default behaviour: a missing category means not_configured', () => {
    expect(categoryGuidance(undefined)).toBe(CATEGORY_GUIDANCE.not_configured)
    expect(categoryGuidance(null)).toBe(CATEGORY_GUIDANCE.not_configured)
  })

  it('returns undefined for prototype-like and unknown keys', () => {
    for (const key of ['__proto__', 'constructor', 'toString', 'hasOwnProperty', 'unknown_category']) {
      expect(categoryGuidance(key)).toBeUndefined()
    }
    expect(categoryGuidance('__proto__')).not.toBe(Object.prototype)
  })
})
