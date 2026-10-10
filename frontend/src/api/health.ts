import { useEffect, useState } from 'react'
import { getHealth } from './client'
import type { Health } from './types'

export function useHealth(pollMs = 5000) {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    async function poll() {
      try {
        const h = await getHealth()
        if (!alive) return
        setHealth(h)
        setError(null)
      } catch (e) {
        if (!alive) return
        setError(e instanceof Error ? e.message : String(e))
      }
    }
    poll()
    const t = setInterval(poll, pollMs)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [pollMs])

  return { health, error }
}
