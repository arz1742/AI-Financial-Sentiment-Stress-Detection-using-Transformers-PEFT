const PANEL_META = {
  analyzer:       { title: 'Live Text Analyzer',        desc: 'Enter financial text and get real-time AI sentiment analysis' },
  comparison:     { title: 'Model Comparison',          desc: 'Compare predictions across RoBERTa, FinBERT, and VADER' },
  trend:          { title: 'Sentiment Trend Dashboard', desc: 'Historical sentiment trajectory over time' },
  explainability: { title: 'Explainability Panel',      desc: 'Understand why the model made its prediction' },
  topics:         { title: 'Topic Modeling',            desc: 'BERTopic-discovered financial themes in discourse' },
  confidence:     { title: 'Confidence & Decision',     desc: 'Ensemble confidence scores and uncertainty flags' },
  simulator:      { title: 'Real-Time Simulator',       desc: 'Continuous live analysis of financial tweet feed' },
}

export default function Header({ activePanel }) {
  const meta = PANEL_META[activePanel] || {}
  const now  = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })

  return (
    <div className="top-bar">
      <div>
        <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
          {meta.title}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 1 }}>{meta.desc}</div>
      </div>

      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-muted)' }}>
        {now}
      </span>
    </div>
  )
}
