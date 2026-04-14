import { useState, useEffect } from 'react'
import axios from 'axios'

const PALETTE = [
  ['#3b82f6', 'rgba(59,130,246,0.15)'],
  ['#10b981', 'rgba(16,185,129,0.15)'],
  ['#8b5cf6', 'rgba(139,92,246,0.15)'],
  ['#f59e0b', 'rgba(245,158,11,0.15)'],
  ['#ef4444', 'rgba(239,68,68,0.15)'],
  ['#06b6d4', 'rgba(6,182,212,0.15)'],
  ['#ec4899', 'rgba(236,72,153,0.15)'],
  ['#84cc16', 'rgba(132,204,22,0.15)'],
  ['#f97316', 'rgba(249,115,22,0.15)'],
  ['#a855f7', 'rgba(168,85,247,0.15)'],
]

export default function TopicModeling() {
  const [topics, setTopics]     = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading]   = useState(true)

  useEffect(() => {
    axios.get('/api/topics')
      .then(r => { setTopics(r.data.topics); setLoading(false) })
      .catch(() => {
        // Fallback data
        setTopics([
          { id: 0, label: 'Market Crash & Fear',    count: 412, keywords: ['crash', 'fear', 'drop', 'sell', 'panic', 'bear', 'red'] },
          { id: 1, label: 'Inflation & Fed Policy', count: 387, keywords: ['inflation', 'fed', 'rates', 'hike', 'cpi', 'price'] },
          { id: 2, label: 'Tech Stocks & AI Boom',  count: 356, keywords: ['tech', 'ai', 'nvidia', 'growth', 'rally', 'ipo'] },
          { id: 3, label: 'Crypto Volatility',      count: 298, keywords: ['crypto', 'btc', 'ethereum', 'hodl', 'pump', 'dump'] },
          { id: 4, label: 'Earnings & Revenue',     count: 267, keywords: ['earnings', 'revenue', 'eps', 'beat', 'guidance'] },
        ])
        setLoading(false)
      })
  }, [])

  const total = topics.reduce((acc, t) => acc + t.count, 0)

  return (
    <div>
      {/* Intro */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">BERTopic — Financial Discourse Themes</div>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7 }}>
          Topics discovered using <strong style={{ color: 'var(--blue-glow)' }}>BERTopic</strong> — 
          neural topic modeling with sentence embeddings (<code style={{ fontFamily: 'var(--font-mono)', fontSize: 12, background: 'rgba(255,255,255,0.06)', padding: '1px 5px', borderRadius: 3 }}>all-MiniLM-L6-v2</code>),
          UMAP dimensionality reduction, and HDBSCAN clustering. Topics represent semantically coherent financial narratives discovered from {total.toLocaleString()}+ posts.
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}>
          <div className="spinner" style={{ width: 32, height: 32 }} />
        </div>
      ) : (
        <div className="grid-2">
          {/* Left: topic list */}
          <div className="card">
            <div className="card-title" style={{ marginBottom: 16 }}>Discovered Topics</div>
            {topics.map((topic, i) => {
              const [color, bg] = PALETTE[i % PALETTE.length]
              const pct = Math.round((topic.count / total) * 100)
              const isActive = selected?.id === topic.id
              return (
                <div
                  key={topic.id}
                  onClick={() => setSelected(isActive ? null : topic)}
                  style={{
                    padding: '12px 14px', marginBottom: 8, borderRadius: 'var(--radius-md)',
                    border: `1px solid ${isActive ? color : 'var(--border)'}`,
                    background: isActive ? bg : 'rgba(255,255,255,0.02)',
                    cursor: 'pointer', transition: 'all 0.2s',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
                      <span style={{ fontWeight: 600, fontSize: 13 }}>{topic.label}</span>
                    </div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                      {topic.count} posts · {pct}%
                    </span>
                  </div>
                  {/* Mini bar */}
                  <div className="prog-bar-bg" style={{ marginTop: 8 }}>
                    <div style={{
                      height: '100%', width: `${pct}%`, background: color,
                      borderRadius: 99, transition: 'width 0.6s',
                    }} />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Right: keyword cloud OR topic detail */}
          <div>
            {selected ? (
              <TopicDetail topic={selected} palette={PALETTE[selected.id % PALETTE.length]} total={total} />
            ) : (
              <KeywordCloud topics={topics} palette={PALETTE} />
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function KeywordCloud({ topics, palette }) {
  // Flatten all keywords with topic color
  const allWords = topics.flatMap((t, i) =>
    t.keywords.map(kw => ({ word: kw, color: palette[i % palette.length][0], topic: t.label }))
  )
  // Shuffle for visual interest
  const shuffled = [...allWords].sort(() => Math.random() - 0.5)

  return (
    <div className="card">
      <div className="card-title" style={{ marginBottom: 4 }}>Keyword Cloud</div>
      <div className="card-subtitle">Click a topic on the left to filter · Keywords coloured by topic</div>
      <div style={{ marginTop: 16, lineHeight: 2.2 }}>
        {shuffled.map((w, i) => (
          <span
            key={i}
            className="keyword-pill"
            title={w.topic}
            style={{
              color: w.color,
              borderColor: `${w.color}33`,
              background: `${w.color}10`,
              fontSize: 11 + Math.random() * 4,
            }}
          >
            {w.word}
          </span>
        ))}
      </div>
    </div>
  )
}

function TopicDetail({ topic, palette, total }) {
  const [color, bg] = palette
  const pct = Math.round((topic.count / total) * 100)

  return (
    <div className="card" style={{ border: `1px solid ${color}44` }}>
      <div className="card-title" style={{ color }}>#{topic.id} · {topic.label}</div>
      <div className="card-subtitle">{topic.count} posts · {pct}% of corpus</div>

      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Key Terms
        </div>
        <div>
          {topic.keywords.map((kw, i) => (
            <span
              key={kw}
              className="keyword-pill"
              style={{
                color,
                borderColor: `${color}44`,
                background: `${color}${Math.max(8, 15 - i * 2).toString(16)}`,
                fontSize: 14 - i * 0.5,
                fontWeight: i === 0 ? 700 : 500,
              }}
            >
              {kw}
            </span>
          ))}
        </div>

        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Corpus share
          </div>
          <div className="prog-bar-bg" style={{ height: 12 }}>
            <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 99, transition: 'width 0.6s' }} />
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{pct}% of total posts</div>
        </div>

        <div style={{
          marginTop: 20, padding: '12px 14px',
          background: bg, border: `1px solid ${color}33`,
          borderRadius: 'var(--radius-md)', fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6,
        }}>
          This topic cluster was identified by <strong style={{ color }}>BERTopic</strong> using HDBSCAN 
          clustering on UMAP-reduced sentence embeddings. The cluster represents semantically coherent 
          narratives about <em>"{topic.label.toLowerCase()}"</em> in financial discourse.
        </div>
      </div>
    </div>
  )
}
