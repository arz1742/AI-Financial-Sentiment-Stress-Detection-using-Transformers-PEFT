// ExplainabilityPanel — SHAP-style word importance visualization

const DEMO_TEXT = "The Fed just hiked rates again — markets are bleeding. Selling everything before this gets worse."
const DEMO_TOKENS = [
  { word: "The",        score:  0.001 },
  { word: "Fed",        score:  0.08  },
  { word: "just",       score:  0.01  },
  { word: "hiked",      score: -0.32  },
  { word: "rates",      score: -0.28  },
  { word: "again",      score: -0.12  },
  { word: "—",          score:  0.0   },
  { word: "markets",    score:  0.04  },
  { word: "are",        score:  0.0   },
  { word: "bleeding",   score: -0.75  },
  { word: "Selling",    score: -0.65  },
  { word: "everything", score: -0.22  },
  { word: "before",     score: -0.05  },
  { word: "this",       score:  0.0   },
  { word: "gets",       score: -0.04  },
  { word: "worse",      score: -0.82  },
]

// Convert a score (-1..1) to a background color
function scoreToColor(score) {
  const intensity = Math.min(Math.abs(score), 1)
  if (score > 0.05) {
    const g = Math.round(100 + intensity * 80)
    return `rgba(16, ${g}, 81, ${intensity * 0.5 + 0.1})`
  } else if (score < -0.05) {
    const r = Math.round(150 + intensity * 80)
    return `rgba(${r}, 44, 44, ${intensity * 0.5 + 0.1})`
  }
  return 'rgba(148, 163, 184, 0.05)'
}

function scoreToTextColor(score) {
  if (score > 0.15) return '#34d399'
  if (score < -0.15) return '#f87171'
  return 'var(--text-primary)'
}

function WordToken({ word, score }) {
  const bg   = scoreToColor(score)
  const color = scoreToTextColor(score)
  return (
    <span
      className="word-token"
      style={{ background: bg, color, border: `1px solid ${bg}` }}
      title={`Score: ${score.toFixed(3)}`}
    >
      {word}
    </span>
  )
}

export default function ExplainabilityPanel({ result }) {
  const tokens = result?.token_importance ?? DEMO_TOKENS
  const text   = result?.text ?? DEMO_TEXT
  const label  = result?.ensemble?.label ?? 'Bearish'
  const isDemo = !result

  // Sort tokens by absolute importance for the bar chart
  const sorted = [...tokens].sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, 12)

  return (
    <div>


      {/* Token highlight visualization */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Word Importance (SHAP-style)</div>
        <div className="card-subtitle">
          Words highlighted in <span style={{ color: 'var(--green)' }}>green</span> support <b>Bullish</b>,{' '}
          <span style={{ color: 'var(--red)' }}>red</span> supports <b>Bearish</b> sentiment
        </div>

        <div style={{
          background: 'rgba(0,0,0,0.2)', borderRadius: 'var(--radius-md)',
          padding: '16px', lineHeight: 2, marginBottom: 16,
        }}>
          {tokens.map((t, i) => (
            <WordToken key={i} word={t.word} score={t.score} />
          ))}
        </div>

        <div style={{ display: 'flex', gap: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
            <div style={{ width: 24, height: 10, background: 'rgba(52,211,153,0.4)', borderRadius: 3 }} />
            <span>Positive influence (Bullish)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
            <div style={{ width: 24, height: 10, background: 'rgba(248,113,113,0.4)', borderRadius: 3 }} />
            <span>Negative influence (Bearish)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
            <div style={{ width: 24, height: 10, background: 'rgba(148,163,184,0.1)', borderRadius: 3 }} />
            <span>Neutral / no influence</span>
          </div>
        </div>
      </div>

      <div className="grid-2">
        {/* Top influential words */}
        <div className="card">
          <div className="card-title">Top Influential Words</div>
          <div className="card-subtitle">Ranked by absolute importance score</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
            {sorted.map((t, i) => {
              const pct = Math.abs(t.score) * 100
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{
                    fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', width: 18, textAlign: 'right'
                  }}>{i + 1}</span>
                  <span style={{ fontWeight: 600, fontSize: 13, width: 100, color: scoreToTextColor(t.score), fontFamily: 'var(--font-mono)' }}>
                    {t.word}
                  </span>
                  <div style={{ flex: 1 }} className="prog-bar-bg">
                    <div style={{
                      height: '100%', width: `${pct}%`,
                      background: t.score > 0 ? 'var(--green)' : 'var(--red)',
                      borderRadius: 99, transition: 'width 0.6s',
                    }} />
                  </div>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)', width: 50, textAlign: 'right' }}>
                    {t.score > 0 ? '+' : ''}{t.score.toFixed(3)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Explanation summary */}
        <div className="card">
          <div className="card-title">Why this prediction?</div>
          <div className="card-subtitle">Model reasoning explanation</div>

          <div style={{ borderLeft: '3px solid var(--blue-bright)', paddingLeft: 14, marginTop: 12 }}>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
              The ensemble model classified this text as{' '}
              <strong className={`badge badge-${label.toLowerCase()}`} style={{ fontSize: 12 }}>{label}</strong>{' '}
              primarily because of the following signal words detected in the input.
            </div>
          </div>

          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Key signals detected
            </div>
            {sorted.slice(0, 5).map((t, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '7px 10px', marginBottom: 4,
                background: 'rgba(255,255,255,0.02)', borderRadius: 6, border: '1px solid var(--border)',
              }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: scoreToTextColor(t.score) }}>
                  "{t.word}"
                </span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  {t.score > 0.1 ? 'Bullish signal' : t.score < -0.1 ? 'Bearish signal' : 'Weak signal'}
                </span>
              </div>
            ))}
          </div>

          <div style={{
            marginTop: 16, padding: '12px 14px',
            background: 'rgba(59,130,246,0.06)', border: '1px solid rgba(59,130,246,0.15)',
            borderRadius: 'var(--radius-md)', fontSize: 12, color: 'var(--text-muted)',
          }}>
            <strong style={{ color: 'var(--blue-glow)' }}>Note:</strong> Feature attribution is computed
            using a SHAP-inspired token importance method based on lexical analysis and model attention patterns.
          </div>
        </div>
      </div>
    </div>
  )
}
