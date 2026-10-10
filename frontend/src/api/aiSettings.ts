import type { AiReadiness } from './types'

export interface AiSettings {
  llm_provider: string | null
  gemini_api_key_masked: string | null
  gemini_model: string | null
  main_model: string | null
  ollama_base_url: string | null
  source: Record<string, string>
}

export interface AiTestResult {
  state: 'ready' | 'partial' | 'model_missing' | 'error' | 'not_configured'
  phases: Record<string, { ok: boolean; [k: string]: unknown }>
  category?: string | null
  hint?: string
  note?: string
  verified_at?: string
  available_models?: string[]
}

export async function getAiSettings(): Promise<AiSettings> {
  const res = await fetch('/api/v1/ai/settings')
  if (!res.ok) throw new Error(`Loading settings failed (${res.status})`)
  return res.json()
}

export async function putAiSettings(patch: Record<string, string>): Promise<AiSettings> {
  const res = await fetch('/api/v1/ai/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!res.ok) {
    let msg = `Saving failed (${res.status})`
    try {
      const d = await res.json()
      msg = d?.detail?.human_message ?? msg
    } catch { /* ignore */ }
    throw new Error(msg)
  }
  return res.json()
}

export async function deleteAiSettings(): Promise<AiSettings | { cleared: boolean }> {
  const res = await fetch('/api/v1/ai/settings', { method: 'DELETE' })
  if (!res.ok) throw new Error(`Resetting failed (${res.status})`)
  return res.json()
}

export async function testAi(payload: Record<string, unknown>): Promise<AiTestResult> {
  const res = await fetch('/api/v1/ai/test', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(`The connection test could not run (${res.status})`)
  return res.json()
}

export type { AiReadiness }

// Human copy for each readiness state (UX language rules).
export const STATE_COPY: Record<AiTestResult['state'], { label: string; detail: string }> = {
  ready: {
    label: 'Working',
    detail: 'Endpoint reached, credential accepted, model present, and a small generation succeeded.',
  },
  partial: {
    label: 'Reachable (generation not verified)',
    detail: 'The service answered and the model exists, but no generation was run or attempted.',
  },
  model_missing: {
    label: 'Model not available',
    detail: 'The service is running, but the configured model is not installed on it.',
  },
  error: {
    label: 'Connection problem',
    detail: 'The AI service could not be used. See the step-by-step results below.',
  },
  not_configured: {
    label: 'Not configured',
    detail: 'No endpoint or key is set yet. Fill in the fields and test the connection.',
  },
}

// Category → WHAT/WHY/WHAT NOW guidance (mirrors app/execution/errors.py).
export const CATEGORY_GUIDANCE: Record<string, { what: string; why: string; whatNow: string }> = {
  auth_rejected: {
    what: 'The AI provider rejected the key.',
    why: 'The configured key was not accepted.',
    whatNow: 'Check the API key in Settings → AI connection, then test again.',
  },
  connection_refused: {
    what: 'The AI service could not be reached.',
    why: 'Nothing is listening at that address.',
    whatNow: 'Check that the AI service (or Ollama) is running and the address is correct.',
  },
  dns_unreachable: {
    what: 'The AI service address could not be found.',
    why: 'The host name could not be resolved from this machine.',
    whatNow: 'Check the address and network configuration.',
  },
  timeout: {
    what: 'The AI service did not respond in time.',
    why: 'The request waited too long for an answer.',
    whatNow: 'Try again; if it keeps failing, check the service or choose another provider.',
  },
  wrong_service: {
    what: 'That address is not an AI service.',
    why: 'Something else answered at the configured address.',
    whatNow: 'Verify the endpoint URL.',
  },
  model_missing: {
    what: 'The configured model is not available.',
    why: 'The service is running but that model is not installed.',
    whatNow: 'Choose one of the available models shown below, or install it on the provider machine.',
  },
  bad_response: {
    what: 'The AI service returned an unexpected answer.',
    why: 'TheEndpoint responded in an unexpected format.',
    whatNow: 'Retry. If it keeps happening, report the provider response format issue.',
  },
  malformed_endpoint: {
    what: 'The service address is malformed.',
    why: 'The configured URL could not be used as an endpoint.',
    whatNow: 'Correct the address (e.g. http://host:port).',
  },
  not_configured: {
    what: 'AI is not configured yet.',
    why: 'No key or endpoint is set.',
    whatNow: 'Fill in the provider fields below and test the connection.',
  },
}
