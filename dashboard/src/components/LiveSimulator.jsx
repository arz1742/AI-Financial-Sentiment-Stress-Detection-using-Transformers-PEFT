import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const TWEET_DELAY_MS = 3500  // time per tweet in simulation

const LABEL_MAP = {
  Bullish: { color: 'var(--green)',       bg: 'rgba(16,185,129,0.08)' },
  Bearish: { color: 'var(--red)',         bg: 'rgba(239,68,68,0.08)' },
  Neutral: { color: 'var(--text-muted)', bg: 'rgba(148,163,184,0.05)' },
}

export default function LiveSimulator({ onResult }) {
  const [tweets, setTweets]       = useState([])
  const [running, setRunning]     = useState(false)
  const [history, setHistory]     = useState([])
  const [current, setCurrent]     = useState(null)    // current tweet being analyzed
  const [analyzing, setAnalyzing] = useState(false)
  const [idx, setIdx]             = useState(0)
  const timerRef                  = useRef(null)
  const historyRef                = useRef(null)

  // Load tweet list on mount
  useEffect(() => {
    axios.get('/api/demo-tweets')
      .then(r => setTweets(r.data.tweets))
      .catch(() => {
        setTweets([
          "The market is crashing hard. My portfolio is down 40%. Selling everything.",
          "Fed just paused! Markets surging. Bull market is back baby!",
          "Earnings came in flat. Nothing much to say. Market seems stable.",
          "Inflation data hot again. This recession is looking real. Very scared.",
          "Tech stocks are making a comeback. AI boom just getting started. Very bullish.",
          "Banks are failing left and right. 2008 all over again. Terrified.",
          "Warren Buffett buying more stocks. Long term bull, simple as that.",
          "Oil prices spiking. Stagflation risk very real. Dark days for consumers.",
          "Crypto down 60%. Can't sleep. What do I do? Lost everything.",
          "Strong GDP growth. Economy resilient. Bullish on US equities.",
        ])
      })
  }, [])

  // Auto-scroll history
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight
    }
  }, [history])

  const analyzeNext = async (tweetList, currentIdx) => {
    if (!tweetList.length) return
    const tweet = tweetList[currentIdx % tweetList.length]
    setCurrent(tweet)
    setAnalyzing(true)

    try {
      const { data } = await axios.post('/api/predict', { text: tweet })
      setHistory(h => [{
        id: Date.now(),
        text: tweet,
        result: data,
        time: new Date().toLocaleTimeString('en-US', { hour12: false }),
      }, ...h].slice(0, 20))
      onResult(data)
    } catch (_) {
      // ignore API errors in simulator
    } finally {
      setAnalyzing(false)
    }
  }

  const startSimulation = () => {
    if (!tweets.length) return
    setRunning(true)
    setHistory([])
    let currentIdx = 0

    const tick = async () => {
      await analyzeNext(tweets, currentIdx)
      currentIdx++
      setIdx(currentIdx)
    }

    tick()
    timerRef.current = setInterval(tick, TWEET_DELAY_MS)
  }

  const stopSimulation = () => {
    setRunning(false)
    clearInterval(timerRef.current)
    timerRef.current = null
    setCurrent(null)
    setAnalyzing(false)
  }

  useEffect(() => () => clearInterval(timerRef.current), [])

  // Summary stats from history
  const stats = history.reduce((acc, h) => {
    const lbl = h.result?.ensemble?.label
    if (lbl) acc[lbl] = (acc[lbl] || 0) + 1
    return acc
  }, {})

  return (
    <div>
      {/* Control panel */}
      <div className="card card-glow" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <div className="card-title">Real-Time Feed Simulator</div>
            <div className="card-subtitle">
              Continuously analyzes financial tweets through all 3 AI models
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>

            {!running ? (
              <button className="btn btn-primary" onClick={startSimulation} disabled={!tweets.length}>
                Start Simulation
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stopSimulation}>
                Stop
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {running && (
          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
              Tweet {(idx % tweets.length) + 1} of {tweets.length} · Auto-cycling
            </div>
            <div className="prog-bar-bg" style={{ height: 4 }}>
              <div style={{
                height: '100%',
                width: `${((idx % tweets.length) / tweets.length) * 100}%`,
                background: 'linear-gradient(90deg, var(--blue-bright), var(--purple))',
                borderRadius: 99,
                transition: 'width 0.4s',
              }} />
            </div>
          </div>
        )}

        {/* Current tweet being analyzed */}
        {current && (
          <div style={{
            marginTop: 14, padding: '12px 16px',
            background: 'rgba(59,130,246,0.06)', border: '1px solid rgba(59,130,246,0.2)',
            borderRadius: 'var(--radius-md)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              {analyzing
                ? <><span className="spinner" style={{ width: 12, height: 12 }} /><span style={{ fontSize: 11, color: 'var(--blue-glow)' }}>Analyzing...</span></>
                : <><span style={{ fontSize: 11, color: 'var(--green)' }}>✓ Done</span></>
              }
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', fontStyle: 'italic' }}>
              "{current}"
            </div>
          </div>
        )}
      </div>

      {/* Stats row */}
      {history.length > 0 && (
        <div className="grid-3" style={{ marginBottom: 16 }}>
          {['Bullish', 'Bearish', 'Neutral'].map(lbl => {
            const m = LABEL_MAP[lbl]
            const count = stats[lbl] || 0
            const pct = history.length ? Math.round((count / history.length) * 100) : 0
            return (
              <div className="card" key={lbl} style={{ borderColor: `${m.color}22` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 13, fontWeight: 600 }}>{lbl}</span>
                  <span className={`badge badge-${lbl.toLowerCase()}`} style={{ fontSize: 11 }}>{pct}%</span>
                </div>
                <div className="stat-number" style={{ color: m.color, fontSize: 30, marginTop: 6 }}>{count}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>of {history.length} analyzed</div>
                <div className="prog-bar-bg" style={{ marginTop: 8 }}>
                  <div style={{
                    height: '100%', width: `${pct}%`,
                    background: m.color, borderRadius: 99, transition: 'width 0.5s',
                  }} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Tweet history feed */}
      <div className="card">
        <div className="card-title">Analysis History</div>
        <div className="card-subtitle">{history.length === 0 ? 'Start the simulation to see results here' : `${history.length} tweets analyzed`}</div>
        <div
          ref={historyRef}
          style={{ maxHeight: 400, overflowY: 'auto', marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}
        >
          {history.length === 0 && !running && (
            <div style={{ textAlign: 'center', padding: '40px 24px', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: 36, marginBottom: 12, color: 'var(--border)' }}>&#9632;</div>
              <div>Use the Start button above to begin the live feed</div>
            </div>
          )}
          {history.map((item) => {
            const lbl   = item.result?.ensemble?.label ?? 'Neutral'
            const conf  = item.result?.ensemble?.confidence ?? 0
            const m     = LABEL_MAP[lbl] || LABEL_MAP.Neutral
            return (
              <div
                key={item.id}
                className="fade-up"
                style={{
                  padding: '12px 14px',
                  background: m.bg,
                  border: `1px solid ${m.color}22`,
                  borderRadius: 'var(--radius-md)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`badge badge-${lbl.toLowerCase()}`} style={{ fontSize: 11 }}>{lbl}</span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {Math.round(conf * 100)}% confidence
                    </span>
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{item.time}</span>
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  "{item.text}"
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
