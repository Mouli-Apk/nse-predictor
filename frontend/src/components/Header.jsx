// Header.jsx — Persistent scorecard header
import { useState, useEffect } from 'react'
import { Activity, Cpu, CheckCircle, AlertCircle, RefreshCw, Zap } from 'lucide-react'
import { api } from '../api'

function StatPill({ label, value, color, icon: Icon }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      background: 'rgba(255,255,255,0.12)',
      borderRadius: 8, padding: '7px 14px',
      border: '1px solid rgba(255,255,255,0.15)',
    }}>
      {Icon && <Icon size={14} style={{ color: color || '#fff', opacity: .9 }} />}
      <div>
        <div style={{ fontSize: 9, opacity: .65, textTransform: 'uppercase',
          letterSpacing: 1.2, fontWeight: 600 }}>{label}</div>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 15, fontWeight: 700,
          color: color || '#fff', lineHeight: 1.2,
        }}>{value}</div>
      </div>
    </div>
  )
}

export default function Header({ onRetrain, retraining }) {
  const [scorecard, setScorecard] = useState(null)
  const [health,    setHealth]    = useState(null)
  const [time,      setTime]      = useState(new Date())

  useEffect(() => {
    const tick = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(tick)
  }, [])

  useEffect(() => {
    const load = async () => {
      try {
        const [sc, h] = await Promise.all([api.scorecard(), api.health()])
        setScorecard(sc)
        setHealth(h)
      } catch { /* silent */ }
    }
    load()
    const id = setInterval(load, 30_000)
    return () => clearInterval(id)
  }, [])

  // NSE market hours: 9:15 AM – 3:30 PM IST (UTC+5:30)
  const istHour = (time.getUTCHours() + 5) % 24 + (time.getUTCMinutes() >= 30 ? 0 : 0)
  const istMinute = (time.getUTCMinutes() + 30) % 60
  const istHourAdj = istHour + Math.floor((time.getUTCMinutes() + 30) / 60)
  const marketOpen = (istHourAdj > 9 || (istHourAdj === 9 && istMinute >= 15)) &&
                     (istHourAdj < 15 || (istHourAdj === 15 && istMinute <= 30))

  const confidenceColor =
    (scorecard?.avg_confidence || 0) >= 70 ? '#57D9A3'
    : (scorecard?.avg_confidence || 0) >= 50 ? '#FF8B00'
    : '#FF5630'

  return (
    <header style={{
      background: 'linear-gradient(135deg, #003D99 0%, #0052CC 55%, #0065FF 100%)',
      color: '#fff',
      padding: '0 24px',
      boxShadow: '0 2px 12px rgba(0,0,0,0.2)',
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      {/* ── Top bar ────────────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '12px 0 8px',
        borderBottom: '1px solid rgba(255,255,255,0.12)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, background: 'rgba(255,255,255,0.18)',
            borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Activity size={18} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: '.3px', lineHeight: 1 }}>
              NSE Intraday Predictor
            </div>
            <div style={{ fontSize: 10, opacity: .65, letterSpacing: 1, marginTop: 2 }}>
              XGBoost · T+15 MIN · 25 STOCKS
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* Live market status */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: marketOpen ? 'rgba(87,217,163,0.2)' : 'rgba(255,86,48,0.2)',
            border: `1px solid ${marketOpen ? 'rgba(87,217,163,0.4)' : 'rgba(255,86,48,0.4)'}`,
            borderRadius: 20, padding: '4px 12px', fontSize: 11, fontWeight: 600,
          }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%',
              background: marketOpen ? '#57D9A3' : '#FF5630',
              animation: marketOpen ? 'pulse-dot 1.5s infinite' : 'none',
            }} />
            {marketOpen ? 'MARKET OPEN' : 'MARKET CLOSED'}
          </div>

          {/* Clock */}
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 13, opacity: .8,
            background: 'rgba(0,0,0,0.15)', borderRadius: 6, padding: '4px 10px',
          }}>
            {time.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false })} IST
          </div>

          {/* Retrain */}
          <button
            onClick={onRetrain}
            disabled={retraining}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '6px 14px', borderRadius: 8,
              background: retraining ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.18)',
              border: '1px solid rgba(255,255,255,0.25)',
              color: '#fff', cursor: retraining ? 'not-allowed' : 'pointer',
              fontSize: 12, fontWeight: 600, transition: 'background 0.2s',
            }}
          >
            <RefreshCw size={13} style={{ animation: retraining ? 'spin 1s linear infinite' : 'none' }} />
            {retraining ? 'Retraining…' : 'Retrain All'}
          </button>
        </div>
      </div>

      {/* ── Scorecard row ───────────────────────────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '10px 0 12px', overflowX: 'auto',
      }}>
        <StatPill
          label="Model Confidence"
          value={scorecard ? `${scorecard.avg_confidence}%` : '—'}
          color={confidenceColor}
          icon={Cpu}
        />
        <StatPill
          label="Today's Success Rate"
          value={scorecard ? `${scorecard.success_rate_pct}%` : '—'}
          color={(scorecard?.success_rate_pct || 0) >= 60 ? '#57D9A3' : '#FF8B00'}
          icon={CheckCircle}
        />
        <StatPill
          label="Predictions Made"
          value={scorecard?.total_predictions ?? '—'}
          icon={Zap}
        />
        <StatPill
          label="Models Loaded"
          value={health?.models_loaded ?? '—'}
          icon={Activity}
        />
        <div style={{ marginLeft: 'auto', fontSize: 9, opacity: .5, whiteSpace: 'nowrap' }}>
          Risk: ₹5,000 / Stop: 1.5%
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); }}`}</style>
    </header>
  )
}
