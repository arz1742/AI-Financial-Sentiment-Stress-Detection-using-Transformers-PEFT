import { useState } from 'react'
import axios from 'axios'

const EXAMPLE_TEXTS = [
  "The market is crashing hard. My portfolio is down 40% and I can't stop watching the red numbers.",
  "Fed just announced a rate pause! Markets are surging. This bull run is just getting started! 🚀",
  "Earnings came in flat. Nothing much to say. Market seems to be holding steady for now.",
  "I've lost everything in crypto. Years of savings gone. I don't know what to do anymore.",
  "Tech stocks are making a huge comeback. AI sector is absolutely on fire right now. Very bullish.",
]

// Maps label → CSS class + short label
const LABEL_STYLE = {
  Bullish: { cls: 'badge-bullish', desc: 'Positive market sentiment' },
  Bearish: { cls: 'badge-bearish', desc: 'Negative financial stress detected' },
  Neutral: { cls: 'badge-neutral', desc: 'No strong directional signal' },
}

export default function LiveAnalyzer({ onResult }) {
  const [text, setText]       = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)

  const analyze = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const { data } = await axios.post('/api/predict', { text })
      setResult(data)
      onResult(data)  // Share with other panels via App state
    } catch (e) {
      setError(e.response?.data?.detail || 'API call failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) analyze()
  }

  const loadExample = () => {
    const txt = EXAMPLE_TEXTS[Math.floor(Math.random() * EXAMPLE_TEXTS.length)]
    setText(txt)
    setResult(null)
  }

  return (
    <div>
      {/* ── Input Card ──────────────────────────── */}
      <div className="card card-glow" style={{ marginBottom: 20 }}>
        <div className="card-title">Financial Text Analyzer</div>
        <div className="card-subtitle">
          Enter any tweet, news headline, or financial comment · Ctrl+Enter to analyze
        </div>

        <textarea
          className="textarea"
          placeholder="e.g. 'The Fed just hiked rates again — markets are bleeding. Selling everything before this gets worse.'"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={5}
        />

        <div style={{ display: 'flex', gap: 10, marginTop: 14, alignItems: 'center' }}>
          <button
            className="btn btn-primary"
            onClick={analyze}
            disabled={loading || !text.trim()}
          >
            {loading ? <><span className="spinner" /> Analyzing</> : <>Analyze</>}
          </button>
          <button className="btn btn-secondary" onClick={loadExample}>
            Load Example
          </button>
          {text && (
            <button className="btn btn-secondary" onClick={() => { setText(''); setResult(null) }}>
              Clear
            </button>
          )}
          <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            {text.length} chars
          </span>
        </div>

        {error && (
          <div style={{
            marginTop: 14, padding: '10px 14px', background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.25)', borderRadius: 'var(--radius-sm)',
            color: '#f87171', fontSize: 13,
          }}>
            Error: {error}
          </div>
        )}
      </div>

      {/* ── Results ─────────────────────────────── */}
      {result && (
        <div className="fade-up">
          {/* Ensemble Decision Banner */}
          <EnsembleBanner result={result} />

          {/* 3-column row: model results */}
          <div className="grid-3" style={{ marginTop: 16 }}>
            <ModelCard
              name="RoBERTa + LoRA"
              subtitle="Champion model · 88.58% acc"
              data={result.roberta}
              labels={result.labels}
              color="var(--green)"
            />
            <ModelCard
              name="FinBERT + LoRA"
              subtitle="Domain-specific · 83.07% acc"
              data={result.finbert}
              labels={result.labels}
              color="var(--blue-bright)"
            />
            <ModelCard
              name="VADER"
              subtitle={`Lexicon baseline · compound: ${result.vader?.compound}`}
              data={result.vader}
              labels={result.labels}
              color="var(--text-muted)"
            />
          </div>

          {/* Stress + info row */}
          <div className="grid-2" style={{ marginTop: 16 }}>
            <StressCard result={result} />
            <InfoCard result={result} />
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          padding: '60px 24px', color: 'var(--text-muted)', gap: 12,
        }}>
          <div style={{ fontSize: 36, color: 'var(--text-muted)', marginBottom: 4 }}>—</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-secondary)' }}>
            Enter text above and click Analyze
          </div>
          <div style={{ fontSize: 12 }}>Results from all 3 models will appear here</div>
        </div>
      )}
    </div>
  )
}

/* ── Sub-components ───────────────────────────────────────────── */

function EnsembleBanner({ result }) {
  const { ensemble, stress } = result
  const meta = LABEL_STYLE[ensemble.label] || {}
  const confPct = Math.round(ensemble.confidence * 100)

  return (
    <div className="card" style={{
      border: '1px solid',
      borderColor: ensemble.label === 'Bullish'
        ? 'rgba(16,185,129,0.3)' : ensemble.label === 'Bearish'
        ? 'rgba(239,68,68,0.3)' : 'rgba(148,163,184,0.2)',
      background: ensemble.label === 'Bullish'
        ? 'rgba(16,185,129,0.05)' : ensemble.label === 'Bearish'
        ? 'rgba(239,68,68,0.05)' : 'var(--bg-card)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              Ensemble Decision
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 4 }}>
              <span className={`badge ${meta.cls}`} style={{ fontSize: 15, padding: '5px 16px' }}>
                {ensemble.label}
              </span>
              <span className={`badge badge-${stress?.level?.toLowerCase()}`} style={{ fontSize: 12 }}>
                {stress?.level} Stress
              </span>
              {result.demo_mode && null}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 5 }}>{meta.desc}</div>
          </div>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Confidence</div>
          <div className="stat-number" style={{ fontSize: 38 }}>{confPct}<span style={{ fontSize: 18 }}>%</span></div>
          {confPct < 60 && (
            <div style={{ fontSize: 11, color: 'var(--gold)', marginTop: 4 }}>Low confidence</div>
          )}
        </div>
      </div>

      {/* Probability bar for ensemble */}
      <div style={{ marginTop: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
          {result.labels.map((lbl, i) => (
            <span key={lbl}>{lbl}: {Math.round(ensemble.probs[i] * 100)}%</span>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 3, height: 8, borderRadius: 99, overflow: 'hidden' }}>
          {result.labels.map((lbl, i) => (
            <div key={lbl} style={{
              width: `${ensemble.probs[i] * 100}%`,
              background: lbl === 'Bullish' ? 'var(--green)' : lbl === 'Bearish' ? 'var(--red)' : 'var(--text-muted)',
              transition: 'width 0.8s var(--ease-out)',
            }} />
          ))}
        </div>
      </div>
    </div>
  )
}

function ModelCard({ name, subtitle, data, labels, color }) {
  if (!data) return null
  const labelIdx = ['Bearish', 'Neutral', 'Bullish'].indexOf(data.label)
  const confPct  = data.confidence ? Math.round(data.confidence * 100) : '--'

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
        <span className="card-title" style={{ fontSize: 13 }}>{name}</span>
      </div>
      <div className="card-subtitle">{subtitle}</div>

      {/* Label badge */}
      <div style={{ marginBottom: 16 }}>
        <span className={`badge badge-${data.label?.toLowerCase()}`}>{data.label}</span>
      </div>

      {/* Per-class probability bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {(['Bearish', 'Neutral', 'Bullish']).map((lbl, i) => {
          const prob = data.probs?.[i] ?? 0
          return (
            <div key={lbl}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                <span style={{ color: 'var(--text-muted)' }}>{lbl}</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {Math.round(prob * 100)}%
                </span>
              </div>
              <div className="prog-bar-bg">
                <div
                  className={`prog-bar-fill prog-${lbl.toLowerCase()}`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }}>
        <span>Confidence</span>
        <span style={{ fontFamily: 'var(--font-mono)', color, fontWeight: 700 }}>{confPct}%</span>
      </div>
    </div>
  )
}

function StressCard({ result }) {
  const { stress } = result
  const stressColor = stress.level === 'High' ? 'var(--red)' : stress.level === 'Moderate' ? 'var(--gold)' : 'var(--green)'

  return (
    <div className="card">
      <div className="card-title">Financial Stress Level</div>
      <div className="card-subtitle">Derived from ensemble sentiment and confidence weighting</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 20, marginTop: 8 }}>
        <div style={{
          width: 80, height: 80, borderRadius: '50%', display: 'flex', alignItems: 'center',
          justifyContent: 'center', flexDirection: 'column',
          border: `3px solid ${stressColor}`, boxShadow: `0 0 20px ${stressColor}40`,
        }}>
          <div style={{ fontSize: 22, fontWeight: 800, color: stressColor, lineHeight: 1, fontFamily: 'var(--font-mono)' }}>
            {stress.score}
          </div>
          <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase' }}>score</div>
        </div>
        <div>
          <span className={`badge badge-${stress.level.toLowerCase()}`} style={{ fontSize: 14, padding: '6px 16px' }}>
            {stress.level} Stress
          </span>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, maxWidth: 200 }}>
            Based on ensemble sentiment prediction + confidence weighting.
          </div>
        </div>
      </div>
    </div>
  )
}

function InfoCard({ result }) {
  const rows = [
    { label: 'Ensemble method',   value: 'Weighted average (50/35/15)' },
    { label: 'RoBERTa weight',    value: '50%' },
    { label: 'FinBERT weight',    value: '35%' },
    { label: 'VADER weight',      value: '15%' },
    { label: 'VADER compound',    value: result.vader?.compound ?? '--' },
    { label: 'Mode',              value: result.demo_mode ? 'Demo simulation' : 'Real model inference' },
  ]

  return (
    <div className="card">
      <div className="card-title">Inference Details</div>
      <div className="card-subtitle">How this prediction was made</div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} style={{ borderBottom: '1px solid var(--border)' }}>
              <td style={{ padding: '7px 0', fontSize: 12, color: 'var(--text-muted)', paddingRight: 12 }}>{r.label}</td>
              <td style={{ padding: '7px 0', fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', textAlign: 'right' }}>{r.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
