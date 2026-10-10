import { useCallback, useEffect, useState } from 'react'
import {
  CATEGORY_GUIDANCE,
  STATE_COPY,
  deleteAiSettings,
  getAiSettings,
  putAiSettings,
  testAi,
  type AiSettings,
  type AiTestResult,
} from '../api/aiSettings'
import { Button, ErrorCard } from '../components/ui'

function ReadinessBadge({ state }: { state: AiTestResult['state'] }) {
  const copy = STATE_COPY[state]
  return (
    <div className={`ai-state ai-state--${state}`}>
      <strong>{copy.label}</strong>
      <p className="muted">{copy.detail}</p>
    </div>
  )
}

function PhaseList({ phases }: { phases: AiTestResult['phases'] }) {
  const order = ['configured', 'endpoint_parsed', 'reachable', 'authorized',
    'model_present', 'generation_ok', 'verified']
  const entries = order.filter((k) => phases[k]).map((k) => [k, phases[k]] as const)
  return (
    <ul className="phase-list">
      {entries.map(([key, p]) => (
        <li key={key}>
          <span>{p.ok ? '✓' : '✗'}</span> {key.replace(/_/g, ' ')}
        </li>
      ))}
    </ul>
  )
}

function FailureGuidance({ result }: { result: AiTestResult }) {
  if (result.state === 'ready') return null
  const cat = result.category ?? 'not_configured'
  const g = CATEGORY_GUIDANCE[cat]
  if (!g) return (
    <p className="muted">The check did not finish successfully. See the step results.</p>
  )
  return (
    <div className="fail-guidance">
      <p><strong>{g.what}</strong></p>
      <p>Why: {g.why}</p>
      <p>You can: {g.whatNow}</p>
      {result.hint && <p className="muted">{result.hint}</p>}
      {result.available_models && result.available_models.length > 0 && (
        <p className="muted">Available on the service: {result.available_models.join(', ')}</p>
      )}
      {(result.state === 'model_missing' || result.state === 'partial') &&
        result.available_models && result.available_models.length > 0 && (
          <details><summary>View technical details</summary>
            <pre>{JSON.stringify(result.phases, null, 2)}</pre>
          </details>
        )}
      {(result.state === 'error') && (
        <details><summary>View technical details</summary>
          <pre>{JSON.stringify(result.phases, null, 2)}</pre>
        </details>
      )}
    </div>
  )
}

export default function Settings() {
  const [settings, setSettings] = useState<AiSettings | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [provider, setProvider] = useState<'gemini' | 'ollama'>('ollama')
  const [geminiKey, setGeminiKey] = useState('')
  const [ollamaUrl, setOllamaUrl] = useState('')
  const [ollamaModel, setOllamaModel] = useState('')
  const [geminiModel, setGeminiModel] = useState('')
  const [busy, setBusy] = useState<'none' | 'save' | 'test' | 'reset'>('none')
  const [testResult, setTestResult] = useState<AiTestResult | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const s = await getAiSettings()
      setSettings(s)
      setLoadError(null)
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  function currentProviderForm(s: AiSettings | null) {
    if (!s) return
    if (s.llm_provider === 'gemini' || s.llm_provider === 'ollama') setProvider(s.llm_provider)
    setOllamaUrl(s.ollama_base_url ?? '')
    setOllamaModel(s.main_model ?? '')
    setGeminiModel(s.gemini_model ?? '')
  }

  useEffect(() => {
    currentProviderForm(settings)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings])

  async function doSave() {
    setBusy('save')
    setActionError(null)
    setSaved(null)
    try {
      const patch: Record<string, string> = { llm_provider: provider }
      if (provider === 'gemini') {
        if (geminiKey) patch.gemini_api_key = geminiKey
        if (geminiModel) patch.gemini_model = geminiModel
      } else {
        if (ollamaUrl) patch.ollama_base_url = ollamaUrl
        if (ollamaModel) patch.main_model = ollamaModel
      }
      const s = await putAiSettings(patch)
      setSettings(s)
      setGeminiKey('')
      setSaved('Settings saved for this server.')
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('none')
    }
  }

  async function doTest() {
    setBusy('test')
    setActionError(null)
    setTestResult(null)
    try {
      const payload: Record<string, unknown> = { llm_provider: provider }
      if (provider === 'gemini') {
        if (geminiKey) payload.gemini_api_key = geminiKey
        if (geminiModel) payload.gemini_model = geminiModel
      } else {
        if (ollamaUrl) payload.ollama_base_url = ollamaUrl
        if (ollamaModel) payload.main_model = ollamaModel
      }
      const r = await testAi(payload)
      setTestResult(r)
      await load()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('none')
    }
  }

  async function doReset() {
    setBusy('reset')
    setActionError(null)
    setSaved(null)
    try {
      await deleteAiSettings()
      setTestResult(null)
      await load()
      setSaved('Saved settings removed. Environment defaults apply again.')
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('none')
    }
  }

  return (
    <div>
      <h1>Settings</h1>
      <p className="muted">
        Configure which AI provider Agentica uses. A connection test checks the
        endpoint, the credential, the model, and a small generation — each step
        separately, so problems are identifiable.
      </p>

      {loadError && (
        <ErrorCard
          message="The backend could not be reached, so settings could not be loaded."
          why={loadError}
          recovery="Start the API backend, then refresh this page."
        />
      )}

      {settings && (
        <section className="card">
          <h2>Current configuration</h2>
          <ul className="health-list">
            <li>Provider: <strong>{settings.llm_provider}</strong> ({settings.source?.llm_provider ?? 'unknown'})</li>
            <li>Gemini key: <strong>{settings.gemini_api_key_masked ?? 'not set'}</strong></li>
            <li>Gemini model: <strong>{settings.gemini_model ?? 'not set'}</strong></li>
            <li>Ollama address: <strong>{settings.ollama_base_url ?? 'not set'}</strong></li>
            <li>Ollama model: <strong>{settings.main_model ?? 'not set'}</strong></li>
          </ul>
        </section>
      )}

      <section className="card form">
        <h2>AI connection</h2>
        <fieldset className="provider-choice">
          <label>
            <input type="radio" name="provider" checked={provider === 'ollama'}
              onChange={() => setProvider('ollama')} disabled={busy !== 'none'} />
            Ollama (self-hosted)
          </label>
          <label>
            <input type="radio" name="provider" checked={provider === 'gemini'}
              onChange={() => setProvider('gemini')} disabled={busy !== 'none'} />
            Gemini API
          </label>
        </fieldset>

        {provider === 'ollama' ? (
          <>
            <label htmlFor="ollama-url">Service address</label>
            <input id="ollama-url" value={ollamaUrl} onChange={(e) => setOllamaUrl(e.target.value)}
              placeholder="http://localhost:11434" disabled={busy !== 'none'} />
            <label htmlFor="ollama-model">Model</label>
            <input id="ollama-model" value={ollamaModel} onChange={(e) => setOllamaModel(e.target.value)}
              placeholder="qwen3:0.6b" disabled={busy !== 'none'} />
          </>
        ) : (
          <>
            <label htmlFor="gemini-key">API key</label>
            <input id="gemini-key" type="password" value={geminiKey} onChange={(e) => setGeminiKey(e.target.value)}
              placeholder="paste key (shown masked after saving)" disabled={busy !== 'none'} autoComplete="off" />
            <label htmlFor="gemini-model">Model</label>
            <input id="gemini-model" value={geminiModel} onChange={(e) => setGeminiModel(e.target.value)}
              placeholder="gemini-3-flash-preview" disabled={busy !== 'none'} />
          </>
        )}

        <div className="form-row">
          <Button type="button" onClick={doTest} disabled={busy !== 'none'}>
            {busy === 'test' ? 'Testing…' : 'Test connection'}
          </Button>
          <Button type="button" variant="ghost" onClick={doSave} disabled={busy !== 'none'}>
            {busy === 'save' ? 'Saving…' : 'Save for this server'}
          </Button>
          <Button type="button" variant="ghost" onClick={doReset} disabled={busy !== 'none'}>
            {busy === 'reset' ? 'Resetting…' : 'Reset to defaults'}
          </Button>
        </div>
        {saved && <div className="saved-note">{saved}</div>}
        {actionError && (
          <ErrorCard message={actionError} onDismiss={() => setActionError(null)} />
        )}
      </section>

      {testResult && (
        <section className="card">
          <h2>Connection test result</h2>
          <ReadinessBadge state={testResult.state} />
          <PhaseList phases={testResult.phases} />
          {testResult.note && <p className="muted">{testResult.note}</p>}
          {testResult.verified_at && (
            <p className="muted">Verified at {new Date(testResult.verified_at).toLocaleString()}</p>
          )}
          <FailureGuidance result={testResult} />
        </section>
      )}

      <section className="card card--warn">
        <h2>What this page does not do yet</h2>
        <ul className="health-list muted">
          <li>Per-execution approval-mode switching (planned, GUI-10)</li>
          <li>Notification preferences (planned, GUI-17)</li>
          <li>User roles and permissions (planned, GUI-18)</li>
        </ul>
      </section>
    </div>
  )
}
