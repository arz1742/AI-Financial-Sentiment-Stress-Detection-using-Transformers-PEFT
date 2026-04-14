import { RadialBarChart, RadialBar, ResponsiveContainer, PolarAngleAxis } from 'recharts'

const DEMO_RESULT = {
  ensemble: { label: 'Bearish', confidence: 0.72, probs: [0.72, 0.18, 0.10] },
  roberta:  { label: 'Bearish', confidence: 0.78, probs: [0.78, 0.15, 0.07] },
  finbert:  { label: 'Bearish', confidence: 0.65, probs: [0.65, 0.22, 0.13] },
  vader:    { label: 'Neutral', probs: [0.28, 0.52, 0.20] },
  stress:   { level: 'High', score: 72 },
  demo_mode: true,
}

function ConfidenceGauge({ value, label, color, size = 140 }) {
  const displayVal = Math.round(value * 100)
  const gaugeData = [{ value: displayVal, fill: color }]
  const isLow = displayVal < 60

  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ position: 'relative', width: size, height: size, margin: '0 auto' }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%" cy="50%"
            innerRadius="65%" outerRadius="90%"
            barSize={10}
            data={[{ value: 100, fill: 'rgba(255,255,255,0.04)' }, ...gaugeData]}
            startAngle={220} endAngle={-40}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
            <RadialBar background={false} dataKey="value" cornerRadius={8} angleAxisId={0} />
          </RadialBarChart>
        </ResponsiveContainer>
        {/* Center value */}
        <div style={{
          position: 'absolute', inset: 0, display: 'flex',
          flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{ fontSize: 24, fontWeight: 800, color, fontFamily: 'var(--font-mono)', lineHeight: 1 }}>
            {displayVal}%
          </div>
          {isLow && <div style={{ fontSize: 9, color: 'var(--gold)', marginTop: 3 }}>LOW</div>}
        </div>
      </div>
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginTop: 8 }}>{label}</div>
    </div>
  )
}

export default function ConfidencePanel({ result }) {
  const data = result || DEMO_RESULT
  const { ensemble, roberta, finbert, vader, stress } = data
  const isDemo = !result

  const ensConf  = ensemble?.confidence ?? 0
  const robConf  = roberta?.confidence  ?? 0
  const finConf  = finbert?.confidence  ?? 0
  const isUncertain = ensConf < 0.60

  // Decision quality label
  const getQuality = (conf) => {
    if (conf >= 0.80) return { label: 'High Confidence',     color: 'var(--green)', icon: null }
    if (conf >= 0.60) return { label: 'Moderate Confidence', color: 'var(--gold)',  icon: null }
    return { label: 'Low Confidence', color: 'var(--red)', icon: null }
  }

  const quality = getQuality(ensConf)

  return (
    <div>


      {/* Uncertainty banner */}
      {isUncertain && (
        <div style={{
          padding: '12px 18px', marginBottom: 16,
          background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--red)', flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, color: 'var(--red)', fontSize: 14 }}>Uncertain Prediction</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
              Ensemble confidence is below 60% threshold. Consider this prediction unreliable. The models may be disagreeing.
            </div>
          </div>
        </div>
      )}

      {/* Gauge row */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title" style={{ marginBottom: 20 }}>Confidence Gauges</div>
        <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 24 }}>
          <ConfidenceGauge
            value={ensConf} label="Ensemble" size={160}
            color={ensConf >= 0.8 ? 'var(--green)' : ensConf >= 0.6 ? 'var(--gold)' : 'var(--red)'}
          />
          <ConfidenceGauge value={robConf} label="RoBERTa+LoRA" color="var(--green)" />
          <ConfidenceGauge value={finConf} label="FinBERT+LoRA"  color="var(--blue-bright)" />
        </div>
      </div>

      <div className="grid-2">
        {/* Decision quality card */}
        <div className="card">
          <div className="card-title">Decision Quality</div>
          <div className="card-subtitle">Overall prediction reliability assessment</div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: 16, marginTop: 16,
            padding: '18px', background: 'rgba(255,255,255,0.02)',
            borderRadius: 'var(--radius-md)', border: '1px solid var(--border)',
          }}>
            <div style={{ width: 12, height: 12, borderRadius: '50%', background: quality.color }} />
            <div>
              <div style={{ fontWeight: 700, color: quality.color, fontSize: 16 }}>{quality.label}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                Ensemble confidence: {Math.round(ensConf * 100)}%
              </div>
            </div>
          </div>

          {/* Threshold guide */}
          <div style={{ marginTop: 18 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Confidence thresholds
            </div>
            {[
              { range: '≥ 80%', label: 'High — strong signal, reliable decision', color: 'var(--green)' },
              { range: '60-80%', label: 'Moderate — usable, minor uncertainty',  color: 'var(--gold)' },
              { range: '< 60%', label: 'Low — uncertain, do not trade on this',   color: 'var(--red)' },
            ].map(t => (
              <div key={t.range} style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: t.color, flexShrink: 0 }} />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: t.color, width: 60 }}>{t.range}</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{t.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Model agreement */}
        <div className="card">
          <div className="card-title">Model Agreement</div>
          <div className="card-subtitle">Comparison of individual model predictions</div>

          <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { name: 'RoBERTa+LoRA', label: roberta?.label ?? 'N/A', weight: '50%', color: '#10b981' },
              { name: 'FinBERT+LoRA', label: finbert?.label ?? 'N/A', weight: '35%', color: '#3b82f6' },
              { name: 'VADER',        label: vader?.label   ?? 'N/A', weight: '15%', color: '#94a3b8' },
            ].map(m => {
              const agrees = m.label === ensemble?.label
              return (
                <div key={m.name} className="model-row">
                  <div>
                    <div className="model-name">{m.name}</div>
                    <div className="model-type">weight: {m.weight}</div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`badge badge-${m.label.toLowerCase()}`} style={{ fontSize: 11 }}>{m.label}</span>
                    <span style={{ fontSize: 13, color: agrees ? 'var(--green)' : 'var(--red)' }}>{agrees ? 'Agrees' : 'Differs'}</span>
                  </div>
                </div>
              )
            })}

            <div style={{
              padding: '10px 14px', background: 'rgba(59,130,246,0.06)',
              border: '1px solid rgba(59,130,246,0.15)', borderRadius: 'var(--radius-md)',
              fontSize: 12, color: 'var(--text-muted)', marginTop: 4,
            }}>
              Ensemble label: <strong className={`badge badge-${ensemble?.label?.toLowerCase()}`} style={{ fontSize: 11 }}>{ensemble?.label}</strong>
              {' '}with {Math.round(ensConf * 100)}% confidence after
              weighted averaging (50% RoBERTa + 35% FinBERT + 15% VADER).
            </div>
          </div>
        </div>
      </div>

      {/* Stress gauge */}
      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title">Financial Stress Meter</div>
        <div className="card-subtitle">Interpreted from ensemble label and confidence</div>
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
            <span>Low Stress</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: stress?.level === 'High' ? 'var(--red)' : stress?.level === 'Moderate' ? 'var(--gold)' : 'var(--green)' }}>
              {stress?.level} ({stress?.score}/100)
            </span>
            <span>High Stress</span>
          </div>
          <div style={{ height: 14, background: 'rgba(255,255,255,0.05)', borderRadius: 99, position: 'relative', overflow: 'hidden' }}>
            <div style={{
              position: 'absolute', inset: 0,
              background: 'linear-gradient(90deg, var(--green), var(--gold), var(--red))',
              borderRadius: 99,
            }} />
            <div style={{
              position: 'absolute', top: 0, bottom: 0,
              left: `${(stress?.score ?? 0)}%`, width: 4,
              background: 'white', borderRadius: 2,
              boxShadow: '0 0 8px rgba(255,255,255,0.8)',
              transform: 'translateX(-2px)',
              transition: 'left 0.8s var(--ease-out)',
            }} />
          </div>
        </div>
      </div>
    </div>
  )
}
