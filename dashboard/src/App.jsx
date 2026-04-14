import { useState } from 'react'
import Header from './components/Header.jsx'
import LiveAnalyzer from './components/LiveAnalyzer.jsx'
import ModelComparison from './components/ModelComparison.jsx'
import SentimentTrend from './components/SentimentTrend.jsx'
import ExplainabilityPanel from './components/ExplainabilityPanel.jsx'
import TopicModeling from './components/TopicModeling.jsx'
import ConfidencePanel from './components/ConfidencePanel.jsx'
import LiveSimulator from './components/LiveSimulator.jsx'

// Navigation items for the sidebar
const NAV_ITEMS = [
  { id: 'analyzer',       label: 'Live Analyzer' },
  { id: 'comparison',     label: 'Model Comparison' },
  { id: 'trend',          label: 'Sentiment Trend' },
  { id: 'explainability', label: 'Explainability' },
  { id: 'topics',         label: 'Topic Modeling' },
  { id: 'confidence',     label: 'Confidence Panel' },
  { id: 'simulator',      label: 'Live Simulator' },
]

export default function App() {
  const [activePanel, setActivePanel] = useState('analyzer')
  // Shared prediction result — set by LiveAnalyzer, consumed by other panels
  const [predictionResult, setPredictionResult] = useState(null)

  // Render the currently selected panel
  const renderPanel = () => {
    switch (activePanel) {
      case 'analyzer':
        return <LiveAnalyzer onResult={setPredictionResult} />
      case 'comparison':
        return <ModelComparison result={predictionResult} />
      case 'trend':
        return <SentimentTrend />
      case 'explainability':
        return <ExplainabilityPanel result={predictionResult} />
      case 'topics':
        return <TopicModeling />
      case 'confidence':
        return <ConfidencePanel result={predictionResult} />
      case 'simulator':
        return <LiveSimulator onResult={setPredictionResult} />
      default:
        return <LiveAnalyzer onResult={setPredictionResult} />
    }
  }

  return (
    <div className="app-layout">
      {/* ── Sidebar ─────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-mark">FinSentAI</div>
          <div className="logo-sub">Sentiment Analysis System</div>
        </div>

        <span className="nav-section-label">Analysis</span>

        {NAV_ITEMS.map((item) => (
          <div
            key={item.id}
            className={`nav-item ${activePanel === item.id ? 'active' : ''}`}
            onClick={() => setActivePanel(item.id)}
          >
            <span>{item.label}</span>
          </div>
        ))}

        <span className="nav-section-label" style={{ marginTop: 'auto' }}>Models</span>
        <div style={{ padding: '0 20px 8px' }}>
          <ModelBadge name="RoBERTa + LoRA" acc="88.6%" color="var(--green)" />
          <ModelBadge name="FinBERT + LoRA" acc="83.1%" color="var(--blue-bright)" />
          <ModelBadge name="VADER"           acc="49.8%" color="var(--text-muted)" />
        </div>
      </aside>

      {/* ── Main content ─────────────────────────── */}
      <main className="main-content">
        <Header activePanel={activePanel} predictionResult={predictionResult} />
        <div className="fade-up" key={activePanel}>
          {renderPanel()}
        </div>
      </main>
    </div>
  )
}

// Small model status badge shown in sidebar
function ModelBadge({ name, acc, color }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '6px 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 500 }}>{name}</span>
      <span style={{ fontSize: 11, color, fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{acc}</span>
    </div>
  )
}
