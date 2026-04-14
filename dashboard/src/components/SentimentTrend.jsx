import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart,
} from 'recharts'

// Generate realistic-looking financial sentiment time series
function generateTimeSeriesData(days = 30) {
  const events = [
    { day: 5,  shift: -0.3, label: 'Rate hike announced' },
    { day: 10, shift:  0.2, label: 'Strong earnings season' },
    { day: 15, shift: -0.4, label: 'Banking sector fear' },
    { day: 20, shift:  0.3, label: 'Fed pauses hikes' },
    { day: 25, shift: -0.15,label: 'Inflation surprise' },
  ]

  const data = []
  let bullish = 0.35, bearish = 0.35, neutral = 0.30
  const startDate = new Date()
  startDate.setDate(startDate.getDate() - days)

  for (let i = 0; i < days; i++) {
    const d = new Date(startDate)
    d.setDate(d.getDate() + i)
    const label = `${d.getMonth() + 1}/${d.getDate()}`

    // Apply event shifts
    const ev = events.find(e => e.day === i)
    if (ev) {
      if (ev.shift < 0) { bearish += Math.abs(ev.shift); bullish -= Math.abs(ev.shift) * 0.5 }
      else              { bullish += ev.shift; bearish -= ev.shift * 0.5 }
    }

    // Add noise
    bullish  = Math.max(0.05, Math.min(0.85, bullish  + (Math.random() - 0.5) * 0.04))
    bearish  = Math.max(0.05, Math.min(0.85, bearish  + (Math.random() - 0.5) * 0.04))
    neutral  = Math.max(0.05, Math.min(0.60, 1 - bullish - bearish))

    const total = bullish + bearish + neutral
    data.push({
      date: label,
      Bullish: Math.round((bullish / total) * 100),
      Bearish: Math.round((bearish / total) * 100),
      Neutral: Math.round((neutral / total) * 100),
      stressScore: Math.round((bearish / total) * 100),
      event: ev?.label ?? null,
    })
  }
  return data
}

const EVENTS = [
  { date: 'Day 5',  label: 'Rate Hike', type: 'negative' },
  { date: 'Day 10', label: 'Earnings',  type: 'positive' },
  { date: 'Day 15', label: 'SVB Fears', type: 'negative' },
  { date: 'Day 20', label: 'Fed Pause', type: 'positive' },
  { date: 'Day 25', label: 'Inflation', type: 'negative' },
]

const CustomDot = ({ cx, cy, payload }) => {
  if (!payload.event) return null
  return <circle cx={cx} cy={cy} r={5} fill="var(--gold)" stroke="var(--bg-base)" strokeWidth={2} />
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null
  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '10px 14px', minWidth: 160,
    }}>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6, color: 'var(--text-primary)' }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ fontSize: 12, color: p.color, fontFamily: 'var(--font-mono)', display: 'flex', justifyContent: 'space-between', gap: 12 }}>
          <span>{p.name}</span>
          <span>{p.value}%</span>
        </div>
      ))}
      {payload[0]?.payload?.event && (
        <div style={{ marginTop: 8, padding: '4px 8px', background: 'rgba(245,158,11,0.1)', borderRadius: 4, fontSize: 11, color: 'var(--gold)' }}>
          {payload[0].payload.event}
        </div>
      )}
    </div>
  )
}

export default function SentimentTrend() {
  const [data] = useState(() => generateTimeSeriesData(30))
  const [mode, setMode] = useState('lines')    // 'lines' | 'area' | 'stress'

  const latest  = data[data.length - 1]
  const prevDay = data[data.length - 2]
  const bullishDelta = latest.Bullish - prevDay.Bullish
  const bearishDelta = latest.Bearish - prevDay.Bearish

  return (
    <div>
      {/* Summary stats */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <StatCard label="Current Bullish" value={`${latest.Bullish}%`} delta={bullishDelta} color="var(--green)" />
        <StatCard label="Current Bearish" value={`${latest.Bearish}%`} delta={bearishDelta} color="var(--red)" inverse />
        <StatCard label="Stress Score"    value={`${latest.stressScore}`} color="var(--gold)" />
      </div>

      {/* Chart mode toggle */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
          <div>
            <div className="card-title">30-Day Sentiment Trend</div>
            <div className="card-subtitle">Financial discourse sentiment trajectory (simulated from training data distribution)</div>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {['lines', 'area', 'stress'].map(m => (
              <button
                key={m} className={`btn ${mode === m ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '5px 12px', fontSize: 12 }}
                onClick={() => setMode(m)}
              >
                {m === 'lines' ? 'Lines' : m === 'area' ? 'Area' : 'Stress'}
              </button>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          {mode === 'stress' ? (
            <AreaChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <defs>
                <linearGradient id="stressGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} interval={4} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={50} stroke="rgba(239,68,68,0.4)" strokeDasharray="6 3" label={{ value: 'High Stress', fill: 'var(--red)', fontSize: 11 }} />
              <Area type="monotone" dataKey="stressScore" stroke="#ef4444" fill="url(#stressGrad)" strokeWidth={2} name="Stress Score" dot={<CustomDot />} />
            </AreaChart>
          ) : mode === 'area' ? (
            <AreaChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <defs>
                <linearGradient id="bullGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="bearGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} interval={4} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ paddingTop: 12, fontSize: 12 }} />
              <Area type="monotone" dataKey="Bullish" stroke="#10b981" fill="url(#bullGrad)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="Bearish" stroke="#ef4444" fill="url(#bearGrad)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="Neutral" stroke="#94a3b8" fill="transparent"   strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </AreaChart>
          ) : (
            <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} interval={4} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ paddingTop: 12, fontSize: 12 }} />
              <Line type="monotone" dataKey="Bullish" stroke="#10b981" strokeWidth={2.5} dot={<CustomDot />} activeDot={{ r: 5 }} />
              <Line type="monotone" dataKey="Bearish" stroke="#ef4444" strokeWidth={2.5} dot={<CustomDot />} activeDot={{ r: 5 }} />
              <Line type="monotone" dataKey="Neutral" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Market events timeline */}
      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title">Market Events Timeline</div>
        <div className="card-subtitle">Key events influencing sentiment shifts</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8 }}>
          {EVENTS.map(ev => (
            <div key={ev.date} style={{
              padding: '8px 14px', borderRadius: 'var(--radius-md)',
              background: ev.type === 'positive' ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)',
              border: `1px solid ${ev.type === 'positive' ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.25)'}`,
            }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{ev.date}</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: ev.type === 'positive' ? 'var(--green)' : 'var(--red)', marginTop: 2 }}>
                {ev.type === 'positive' ? '▲' : '▼'} {ev.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, delta, color, inverse = false }) {
  const up = inverse ? delta < 0 : delta > 0
  return (
    <div className="card">
      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>{label}</div>
      <div className="stat-number" style={{ 'WebkitTextFillColor': color, color }}>{value}</div>
      {delta !== undefined && (
        <div style={{ fontSize: 12, color: up ? 'var(--green)' : 'var(--red)', marginTop: 6 }}>
          {up ? '▲' : '▼'} {Math.abs(delta).toFixed(1)}% vs yesterday
        </div>
      )}
    </div>
  )
}
