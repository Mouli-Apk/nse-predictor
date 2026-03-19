// AfterMarket.jsx — Post-Close Analysis for Next Morning Pre-Open
// Only active after 3:30 PM IST. Ranks stocks for tomorrow's pre-open bid.

import { useState, useEffect } from 'react'
import { Moon, TrendingUp, Target, ShieldAlert, Package,
         RefreshCw, Clock, Star, BarChart2, ChevronDown, ChevronUp } from 'lucide-react'
import { api } from '../api'

const fmt    = (n, d=2) => n == null ? '—'
  : Number(n).toLocaleString('en-IN', {minimumFractionDigits:d, maximumFractionDigits:d})
const fmtPct = n => n == null ? '—' : `${n>=0?'+':''}${Number(n).toFixed(2)}%`

// ── IST time helpers ──────────────────────────────────────────────────────────
function getISTHour() {
  const now = new Date()
  const ist = new Date(now.getTime() + 5.5*3600*1000)
  return { h: ist.getUTCHours(), m: ist.getUTCMinutes() }
}
function isAfterMarket() {
  const { h, m } = getISTHour()
  return (h > 15) || (h === 15 && m >= 30)
}
function isPreOpen() {
  const { h, m } = getISTHour()
  return (h < 9) || (h === 9 && m < 15)
}
function isAvailable() {
  return isAfterMarket() || isPreOpen()
}

// ── Score bar ─────────────────────────────────────────────────────────────────
function ScoreBar({ label, value, color }) {
  return (
    <div style={{ marginBottom:5 }}>
      <div style={{ display:'flex', justifyContent:'space-between',
        fontSize:9, color:'var(--slate-light)', marginBottom:2,
        fontFamily:'var(--font-mono)', fontWeight:600, letterSpacing:.8 }}>
        <span>{label.toUpperCase()}</span>
        <span style={{ color }}>{Math.round(value)}%</span>
      </div>
      <div style={{ height:4, background:'var(--border)', borderRadius:2, overflow:'hidden' }}>
        <div style={{ height:'100%', width:`${Math.max(0,Math.min(100,value))}%`,
          background:color, borderRadius:2, transition:'width .5s ease' }}/>
      </div>
    </div>
  )
}

// ── Stock pick card ───────────────────────────────────────────────────────────
function PickCard({ stock, rank }) {
  const [expanded, setExpanded] = useState(false)

  const scoreColor = stock.composite_score >= 70 ? 'var(--buy)'
    : stock.composite_score >= 50 ? 'var(--hold)' : 'var(--sell)'

  const SCORE_LABELS = {
    trend:         'Trend alignment',
    rsi:           'RSI positioning',
    volume:        'Volume conviction',
    range_position:'Day range pos.',
    spread:        'Bid-ask spread',
    week52:        '52W positioning',
    model_acc:     'Model accuracy',
  }
  const SCORE_COLORS = {
    trend:         '#0052CC',
    rsi:           '#6554C0',
    volume:        '#00875A',
    range_position:'#FF8B00',
    spread:        '#00B8D9',
    week52:        '#403294',
    model_acc:     '#DE350B',
  }

  return (
    <div style={{
      background:'var(--white)',
      border:`1.5px solid ${stock.actionable ? 'rgba(0,135,90,0.35)' : 'var(--border)'}`,
      borderRadius:12,
      padding:'14px 16px',
      animation:`fade-up .3s ease ${rank*0.05}s both`,
      boxShadow: stock.actionable ? '0 2px 12px rgba(0,135,90,0.1)' : 'var(--shadow-sm)',
    }}>

      {/* Header */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:12 }}>
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          {/* Rank badge */}
          <div style={{
            width:28, height:28, borderRadius:'50%',
            background: rank <= 2 ? 'linear-gradient(135deg,#FFD700,#FFA500)'
              : rank <= 5 ? 'var(--blue-light)' : 'var(--slate-faint)',
            display:'flex', alignItems:'center', justifyContent:'center',
            fontWeight:800, fontSize:12,
            color: rank <= 2 ? '#7A4500' : rank <= 5 ? 'var(--blue)' : 'var(--slate)',
          }}>#{rank}</div>
          <div>
            <div style={{ fontWeight:800, fontSize:15, color:'#172B4D', lineHeight:1 }}>
              {stock.ticker.replace('.NS','')}
            </div>
            <div style={{ fontSize:10, color:'var(--slate-light)', marginTop:2 }}>
              {stock.sector} · {stock.data_interval} data
            </div>
          </div>
        </div>

        {/* Composite score circle */}
        <div style={{ textAlign:'center' }}>
          <div style={{ width:44, height:44, borderRadius:'50%',
            border:`3px solid ${scoreColor}`,
            display:'flex', alignItems:'center', justifyContent:'center',
            flexDirection:'column', background:'#fff' }}>
            <div style={{ fontFamily:'var(--font-mono)', fontWeight:800, fontSize:13,
              color:scoreColor, lineHeight:1 }}>
              {Math.round(stock.composite_score)}
            </div>
            <div style={{ fontSize:7, color:'var(--slate-light)', letterSpacing:.5 }}>SCORE</div>
          </div>
        </div>
      </div>

      {/* Action badge */}
      {stock.actionable && (
        <div style={{
          display:'inline-flex', alignItems:'center', gap:5,
          padding:'4px 12px', borderRadius:20, marginBottom:12,
          background:'var(--buy-bg)', color:'var(--buy)',
          fontSize:11, fontWeight:800, letterSpacing:.5,
        }}>
          <Star size={11} fill="currentColor"/> BUY TOMORROW MORNING
        </div>
      )}

      {/* Price prediction grid */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:7, marginBottom:12 }}>
        <div style={{ background:'var(--slate-faint)', borderRadius:7, padding:'8px 10px' }}>
          <div style={{ fontSize:9, color:'var(--slate-light)', textTransform:'uppercase',
            letterSpacing:1, marginBottom:2, fontWeight:700 }}>Last Close</div>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:13 }}>
            ₹{fmt(stock.current_price)}
          </div>
        </div>
        <div style={{ background:'var(--buy-bg)', borderRadius:7, padding:'8px 10px' }}>
          <div style={{ fontSize:9, color:'var(--buy)', textTransform:'uppercase',
            letterSpacing:1, marginBottom:2, fontWeight:700 }}>Target</div>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:13,
            color:'var(--buy)' }}>₹{fmt(stock.target)}
            <span style={{ fontSize:10, marginLeft:4, opacity:.8 }}>
              ({fmtPct(stock.predicted_change_pct)})
            </span>
          </div>
        </div>
        <div style={{ background:'var(--sell-bg)', borderRadius:7, padding:'8px 10px' }}>
          <div style={{ fontSize:9, color:'var(--sell)', textTransform:'uppercase',
            letterSpacing:1, marginBottom:2, fontWeight:700 }}>Stop Loss</div>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:13,
            color:'var(--sell)' }}>₹{fmt(stock.stop_loss)}</div>
        </div>
      </div>

      {/* Buy range */}
      <div style={{ background:'linear-gradient(135deg,#003D99,#0052CC)',
        borderRadius:8, padding:'10px 14px', marginBottom:12, color:'#fff' }}>
        <div style={{ fontSize:9, opacity:.7, textTransform:'uppercase',
          letterSpacing:1.2, marginBottom:4, fontWeight:700 }}>
          📋 Pre-Open Bid Range
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:800, fontSize:15 }}>
            ₹{fmt(stock.buy_range?.low)} – ₹{fmt(stock.buy_range?.high)}
          </div>
          <div style={{ fontSize:10, opacity:.7, display:'flex', alignItems:'center', gap:4 }}>
            <Package size={11}/> Qty: {stock.risk_qty}
          </div>
        </div>
        <div style={{ fontSize:10, opacity:.6, marginTop:4 }}>
          Risk: ₹5,000 · Stop 1.5% · Model confidence: {stock.confidence}%
        </div>
      </div>

      {/* Today's stats */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:5, marginBottom:12 }}>
        {[
          ['Open',   `₹${fmt(stock.today_stats?.open, 0)}`],
          ['High',   `₹${fmt(stock.today_stats?.high, 0)}`],
          ['Low',    `₹${fmt(stock.today_stats?.low, 0)}`],
          ['Vol×',   `${stock.today_stats?.vol_ratio?.toFixed(1)}x`],
        ].map(([l,v]) => (
          <div key={l} style={{ background:'var(--slate-faint)', borderRadius:5, padding:'5px 7px' }}>
            <div style={{ fontSize:8, color:'var(--slate-light)', letterSpacing:.8,
              textTransform:'uppercase', marginBottom:1 }}>{l}</div>
            <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:11 }}>{v}</div>
          </div>
        ))}
      </div>

      {/* Expand toggle */}
      <button onClick={() => setExpanded(v=>!v)} style={{
        width:'100%', padding:'5px', background:'none',
        border:'1px solid var(--border)', borderRadius:6, cursor:'pointer',
        fontSize:10, color:'var(--slate-light)',
        display:'flex', alignItems:'center', justifyContent:'center', gap:4,
      }}>
        {expanded ? <ChevronUp size={11}/> : <ChevronDown size={11}/>}
        {expanded ? 'Hide' : 'Score Breakdown'}
      </button>

      {expanded && (
        <div style={{ marginTop:12, padding:'12px', background:'var(--slate-faint)',
          borderRadius:8, animation:'fade-up .2s ease' }}>
          {Object.entries(stock.scores || {}).map(([key, val]) => (
            <ScoreBar key={key}
              label={SCORE_LABELS[key] || key}
              value={val}
              color={SCORE_COLORS[key] || 'var(--blue)'}/>
          ))}

          {/* Pre-session data */}
          {(stock.pre_session?.bid || stock.pre_session?.week52_high) && (
            <div style={{ marginTop:10, paddingTop:10,
              borderTop:'1px solid var(--border)', display:'grid',
              gridTemplateColumns:'1fr 1fr 1fr', gap:5 }}>
              {[
                ['Bid',      stock.pre_session?.bid     ? `₹${fmt(stock.pre_session.bid)}`  : '—'],
                ['Ask',      stock.pre_session?.ask     ? `₹${fmt(stock.pre_session.ask)}`  : '—'],
                ['52W High', stock.pre_session?.week52_high ? `₹${fmt(stock.pre_session.week52_high,0)}` : '—'],
                ['52W Low',  stock.pre_session?.week52_low  ? `₹${fmt(stock.pre_session.week52_low,0)}`  : '—'],
                ['P/E',      stock.pre_session?.pe_ratio ? stock.pre_session.pe_ratio.toFixed(1) : '—'],
                ['RSI 14',   `${stock.today_stats?.rsi}`],
              ].map(([l,v]) => (
                <div key={l} style={{ background:'#fff', borderRadius:4, padding:'5px 7px' }}>
                  <div style={{ fontSize:8, color:'var(--slate-light)',
                    textTransform:'uppercase', letterSpacing:.8, marginBottom:1 }}>{l}</div>
                  <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:11 }}>{v}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════
export default function AfterMarket() {
  const [data,     setData]     = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [showAll,  setShowAll]  = useState(false)
  const [clock,    setClock]    = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  const available = isAvailable()

  const runAnalysis = async () => {
    setLoading(true); setError(null)
    try {
      const res = await fetch(
        (import.meta.env.VITE_API_URL || '/api') + '/after-market?top_n=10'
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setData(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const istStr = clock.toLocaleTimeString('en-IN', {
    timeZone:'Asia/Kolkata', hour:'2-digit', minute:'2-digit', second:'2-digit',
    hour12: false,
  })

  const picks      = data?.top_picks  || []
  const actionable = picks.filter(p => p.actionable)
  const displayed  = showAll ? picks : picks.slice(0, 5)

  return (
    <div style={{ padding:'24px 28px', maxWidth:920 }}>

      {/* Header */}
      <div style={{ display:'flex', justifyContent:'space-between',
        alignItems:'flex-start', marginBottom:20 }}>
        <div>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:6 }}>
            <Moon size={22} style={{ color:'#403294' }}/>
            <h2 style={{ fontSize:20, fontWeight:700, color:'#172B4D' }}>
              After-Market Analysis
            </h2>
            <span style={{
              padding:'3px 10px', borderRadius:10, fontSize:11, fontWeight:700,
              background: available ? 'var(--buy-bg)' : 'var(--hold-bg)',
              color:      available ? 'var(--buy)'    : 'var(--hold)',
            }}>
              {available ? '✓ Available now' : '⏰ After 3:30 PM IST'}
            </span>
          </div>
          <p style={{ color:'var(--slate)', fontSize:13 }}>
            Post-close ML analysis for tomorrow's pre-open session (9:00–9:15 AM IST)
          </p>
        </div>
        <div style={{ fontFamily:'var(--font-mono)', fontSize:13,
          color:'var(--slate)', background:'var(--slate-faint)',
          padding:'6px 12px', borderRadius:8 }}>
          🕐 {istStr} IST
        </div>
      </div>

      {/* Market closed gate */}
      {!available && !data && (
        <div style={{
          background:'linear-gradient(135deg,#1a1a2e,#16213e)',
          borderRadius:16, padding:'40px 32px', textAlign:'center', color:'#fff',
          marginBottom:20,
        }}>
          <Moon size={48} style={{ opacity:.4, marginBottom:16 }}/>
          <h3 style={{ fontSize:18, fontWeight:700, marginBottom:8 }}>
            Market is Open
          </h3>
          <p style={{ opacity:.65, fontSize:13, marginBottom:20 }}>
            After-market analysis becomes available at <strong>3:30 PM IST</strong> when
            the NSE session closes. Come back then for tomorrow's pre-open recommendations.
          </p>
          <div style={{ display:'inline-block', padding:'8px 20px', borderRadius:20,
            background:'rgba(255,255,255,0.1)', fontSize:12, opacity:.7 }}>
            Current time: {istStr} IST
          </div>
        </div>
      )}

      {/* Run button (shown after close) */}
      {(available || data) && (
        <div style={{ marginBottom:20 }}>
          <button onClick={runAnalysis} disabled={loading} style={{
            display:'flex', alignItems:'center', gap:8,
            padding:'10px 22px', borderRadius:10, fontSize:13, fontWeight:700,
            background:'linear-gradient(135deg,#1a1a2e,#403294)',
            color:'#fff', border:'none', cursor:loading?'not-allowed':'pointer',
            opacity:loading?0.7:1,
          }}>
            {loading
              ? <RefreshCw size={15} style={{ animation:'spin 1s linear infinite' }}/>
              : <BarChart2 size={15}/>}
            {loading ? 'Analysing all 25 stocks…' : 'Run After-Market Analysis'}
          </button>
          {!available && data && (
            <p style={{ fontSize:11, color:'var(--slate-light)', marginTop:6 }}>
              Analysis from {data.generated_at} · Market opens {data.market_opens}
            </p>
          )}
        </div>
      )}

      {error && (
        <div style={{ background:'var(--sell-bg)', color:'var(--sell)',
          borderRadius:8, padding:'10px 16px', marginBottom:16, fontSize:13 }}>
          ✗ {error}
        </div>
      )}

      {/* Results */}
      {data && data.status === 'ok' && (
        <>
          {/* Summary bar */}
          <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)',
            gap:10, marginBottom:20 }}>
            {[
              ['Stocks Analysed', data.total_analysed,   'var(--blue)'],
              ['Bullish Setups',  data.actionable_count, 'var(--buy)'],
              ['Generated At',    data.generated_at?.split('  ')[1] || '—', 'var(--slate)'],
              ['Market Opens',    data.market_opens?.split('  ')[1] || '—', '#403294'],
            ].map(([label,value,color]) => (
              <div key={label} style={{ background:'var(--white)',
                borderRadius:10, padding:'12px 14px', boxShadow:'var(--shadow-sm)' }}>
                <div style={{ fontSize:9, color:'var(--slate-light)',
                  textTransform:'uppercase', letterSpacing:1, marginBottom:4 }}>{label}</div>
                <div style={{ fontFamily:'var(--font-mono)', fontSize:16,
                  fontWeight:700, color }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Summary text */}
          <div style={{ background:'linear-gradient(135deg,#001f5c,#003D99)',
            color:'#fff', borderRadius:10, padding:'14px 18px', marginBottom:20,
            fontSize:13, lineHeight:1.6 }}>
            <strong>💡 ML Insight:</strong> {data.summary}
          </div>

          {/* Pick cards */}
          {picks.length === 0 ? (
            <div style={{ textAlign:'center', padding:'40px', color:'var(--slate-light)' }}>
              No actionable setups detected for tomorrow.
            </div>
          ) : (
            <>
              {actionable.length > 0 && (
                <div style={{ fontSize:11, fontWeight:700, textTransform:'uppercase',
                  letterSpacing:1, color:'var(--buy)', marginBottom:12 }}>
                  ✅ Recommended for pre-open bid ({actionable.length})
                </div>
              )}

              <div style={{ display:'grid',
                gridTemplateColumns:'repeat(auto-fill,minmax(280px,1fr))',
                gap:14, marginBottom:16 }}>
                {displayed.map((stock, i) => (
                  <PickCard key={stock.ticker} stock={stock} rank={i+1}/>
                ))}
              </div>

              {picks.length > 5 && (
                <button onClick={() => setShowAll(v=>!v)} style={{
                  width:'100%', padding:'10px', background:'var(--slate-faint)',
                  border:'1px solid var(--border)', borderRadius:8, cursor:'pointer',
                  fontSize:12, fontWeight:600, color:'var(--slate)',
                  display:'flex', alignItems:'center', justifyContent:'center', gap:6,
                }}>
                  {showAll ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
                  {showAll ? 'Show less' : `Show all ${picks.length} stocks`}
                </button>
              )}
            </>
          )}
        </>
      )}

      <style>{`@keyframes spin { to { transform:rotate(360deg); } }`}</style>
    </div>
  )
}
