import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell,
} from 'recharts'

const MODELS = [
  { key: 'roberta', name: 'RoBERTa+LoRA', color: '#10b981' },
  { key: 'finbert', name: 'FinBERT+LoRA', color: '#3b82f6' },
  { key: 'vader',   name: 'VADER',        color: '#94a3b8' },
]

const LABELS = ['Bearish', 'Neutral', 'Bullish']
const LABEL_COLORS = { Bearish: '#ef4444', Neutral: '#94a3b8', Bullish: '#10b981' }

// Demo data used when no live prediction is available
const DEMO_DATA = {
  roberta: { label: 'Bearish', probs: [0.72, 0.18, 0.10] },
  finbert: { label: 'Bearish', probs: [0.63, 0.25, 0.12] },
  vader:   { label: 'Neutral', probs: [0.28, 0.52, 0.20] },
}

export default function ModelComparison({ result }) {
  const [metricsData, setMetricsData] = useState(null)

  useEffect(() => {
    axios.get('/api/metrics').then(r => setMetricsData(r.data)).catch(() => {})
  }, [])

  // Build chart data: one entry per label, each model is a bar
  const activeData = result || { roberta: DEMO_DATA.roberta, finbert: DEMO_DATA.finbert, vader: DEMO_DATA.vader }

  const chartData = LABELS.map((label, i) => ({
    label,
    'RoBERTa+LoRA': Math.round((activeData.roberta?.probs?.[i] ?? 0) * 100),
    'FinBERT+LoRA':  Math.round((activeData.finbert?.probs?.[i] ?? 0) * 100),
    'VADER':         Math.round((activeData.vader?.probs?.[i] ?? 0) * 100),
  }))

  // Performance metrics comparison data
  const perfData = metricsData
    ? metricsData.models.map(m => ({
        name:  m.name,
        'F1 Macro':    Math.round(m.f1_macro * 100),
        'Accuracy':    Math.round(m.accuracy),
        'ROC-AUC':     Math.round(m.roc_auc * 100),
      }))
    : []

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload) return null
    return (
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '10px 14px',
      }}>
        <div style={{ fontWeight: 700, marginBottom: 6, color: 'var(--text-primary)' }}>{label}</div>
        {payload.map(p => (
          <div key={p.name} style={{ fontSize: 13, color: p.color, fontFamily: 'var(--font-mono)' }}>
            {p.name}: {p.value}%
          </div>
        ))}
      </div>
    )
  }

  return (
    <div>


      <div className="grid-2">
        {/* Probability Comparison Chart */}
        <div className="card">
          <div className="card-title">Probability Distribution</div>
          <div className="card-subtitle">Per-class probabilities for each model</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="label" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ paddingTop: 12, fontSize: 12 }} />
              {MODELS.map(m => (
                <Bar key={m.key} dataKey={m.name} fill={m.color} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model Performance Metrics */}
        <div className="card">
          <div className="card-title">Model Performance Metrics</div>
          <div className="card-subtitle">Evaluated on the financial test set</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perfData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ paddingTop: 12, fontSize: 12 }} />
              <Bar dataKey="Accuracy"  fill="#3b82f6" radius={[4,4,0,0]} />
              <Bar dataKey="F1 Macro"  fill="#10b981" radius={[4,4,0,0]} />
              <Bar dataKey="ROC-AUC"   fill="#8b5cf6" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model detail cards */}
      <div className="grid-3" style={{ marginTop: 16 }}>
        {MODELS.map(m => {
          const d = activeData[m.key]
          if (!d) return null
          return (
            <div className="card" key={m.key} style={{ borderColor: `${m.color}22` }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
                <div style={{ width: 10, height: 10, background: m.color, borderRadius: '50%' }} />
                <span style={{ fontWeight: 700, fontSize: 13 }}>{m.name}</span>
                <span className={`badge badge-${d.label?.toLowerCase()}`} style={{ marginLeft: 'auto', fontSize: 11 }}>{d.label}</span>
              </div>
              {LABELS.map((lbl, i) => (
                <div key={lbl} style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                    <span style={{ color: LABEL_COLORS[lbl] }}>{lbl}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                      {Math.round((d.probs?.[i] ?? 0) * 100)}%
                    </span>
                  </div>
                  <div className="prog-bar-bg">
                    <div
                      className={`prog-bar-fill prog-${lbl.toLowerCase()}`}
                      style={{ width: `${(d.probs?.[i] ?? 0) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}
