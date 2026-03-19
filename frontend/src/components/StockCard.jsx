// StockCard.jsx — Multi-horizon predictions + pre-session bids
import { useState, useCallback } from 'react'
import { AreaChart, Area, ResponsiveContainer, Tooltip, YAxis } from 'recharts'
import { TrendingUp, TrendingDown, Minus, RefreshCw, Clock, Package,
         ChevronDown, ChevronUp, Activity } from 'lucide-react'
import { api } from '../api'

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt    = (n, d = 2) => n == null ? '—'
  : Number(n).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d })
const fmtPct = n => n == null ? '—' : `${n >= 0 ? '+' : ''}${Number(n).toFixed(2)}%`
const fmtCr  = n => n == null ? '—'
  : n >= 1e7 ? `₹${(n/1e7).toFixed(1)}Cr` : `₹${(n/1e5).toFixed(1)}L`

const SIG = {
  BUY:  { label:'▲ BUY',  bg:'var(--buy-bg)',  color:'var(--buy)',  Icon:TrendingUp  },
  SELL: { label:'▼ SELL', bg:'var(--sell-bg)', color:'var(--sell)', Icon:TrendingDown },
  HOLD: { label:'◆ HOLD', bg:'var(--hold-bg)', color:'var(--hold)', Icon:Minus        },
}
const SECTOR_COLORS = {
  'Defense':'#0052CC','Industrial':'#403294','Green Energy':'#00875A',
  'Finance':'#0065FF','Fintech':'#6554C0','Consumer':'#FF5630','Healthcare':'#00B8D9',
}
const H_TABS = [
  { key:'t15', label:'T+15m' },
  { key:'t1h', label:'T+1h'  },
  { key:'t3h', label:'T+3h'  },
]

// ── Sparkline ─────────────────────────────────────────────────────────────────
function Sparkline({ data, signal }) {
  if (!data?.length) return (
    <div style={{ height:48, display:'flex', alignItems:'center',
      justifyContent:'center', color:'var(--slate-light)', fontSize:11 }}>
      No chart data
    </div>
  )
  const cd = data.map((v, i) => ({ i, v }))
  const color = signal === 'BUY' ? '#00875A' : signal === 'SELL' ? '#DE350B' : '#0052CC'
  return (
    <ResponsiveContainer width="100%" height={48}>
      <AreaChart data={cd} margin={{ top:2, right:0, bottom:0, left:0 }}>
        <defs>
          <linearGradient id={`g-${signal}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.18}/>
            <stop offset="95%" stopColor={color} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <YAxis domain={['auto','auto']} hide/>
        <Tooltip content={({ active, payload }) =>
          active && payload?.[0] ? (
            <div style={{ background:'#172B4D', color:'#fff', padding:'4px 8px',
              borderRadius:4, fontSize:11, fontFamily:'var(--font-mono)' }}>
              ₹{fmt(payload[0].value)}
            </div>
          ) : null
        }/>
        <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.5}
          fill={`url(#g-${signal})`} dot={false} isAnimationActive={false}/>
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ── Confidence bar ────────────────────────────────────────────────────────────
function ConfBar({ value }) {
  const pct   = Math.min(100, Math.max(0, value || 0))
  const color = pct >= 70 ? 'var(--buy)' : pct >= 50 ? 'var(--hold)' : 'var(--sell)'
  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', fontSize:9,
        color:'var(--slate-light)', marginBottom:3, fontFamily:'var(--font-mono)',
        fontWeight:600, letterSpacing:1 }}>
        <span>CONFIDENCE</span><span style={{ color }}>{pct}%</span>
      </div>
      <div style={{ height:4, background:'var(--border)', borderRadius:2, overflow:'hidden' }}>
        <div style={{ height:'100%', width:`${pct}%`, background:color,
          borderRadius:2, transition:'width .6s ease' }}/>
      </div>
    </div>
  )
}

// ── Pre-session bids panel ────────────────────────────────────────────────────
function PreSession({ data }) {
  if (!data) return null
  const { bid, ask, pre_price, pre_change_pct, volume, avg_volume, week52_high, week52_low, pe_ratio } = data
  const hasAny = bid || ask || pre_price || week52_high
  if (!hasAny) return null

  const spread    = bid && ask ? fmt(ask - bid) : null
  const vol_ratio = volume && avg_volume ? (volume / avg_volume * 100).toFixed(0) : null
  const w52_pct   = week52_high && week52_low
    ? ((week52_high - week52_low) / week52_low * 100).toFixed(1) : null

  return (
    <div style={{ marginBottom:10, borderRadius:6, overflow:'hidden',
      border:'1px solid var(--border)' }}>
      <div style={{ background:'#001f5c', color:'#fff', padding:'5px 9px',
        fontSize:9, fontWeight:700, letterSpacing:1.2, textTransform:'uppercase' }}>
        Pre-Session & Market Depth
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr',
        gap:0, background:'#fff' }}>
        {[
          ['BID',  bid  ? `₹${fmt(bid)}`  : '—', bid  ? '#00875A' : '#aaa'],
          ['ASK',  ask  ? `₹${fmt(ask)}`  : '—', ask  ? '#DE350B' : '#aaa'],
          ['SPREAD', spread ? `₹${spread}` : '—', '#42526E'],
          ['PRE-MKT', pre_price ? `₹${fmt(pre_price)}` : '—',
            pre_change_pct >= 0 ? '#00875A' : '#DE350B'],
          ['PRE %', pre_change_pct != null ? fmtPct(pre_change_pct) : '—',
            pre_change_pct >= 0 ? '#00875A' : '#DE350B'],
          ['P/E', pe_ratio ? pe_ratio.toFixed(1) : '—', '#42526E'],
          ['52W HIGH', week52_high ? `₹${fmt(week52_high, 0)}` : '—', '#42526E'],
          ['52W LOW',  week52_low  ? `₹${fmt(week52_low,  0)}` : '—', '#42526E'],
          ['VOL%', vol_ratio ? `${vol_ratio}%` : '—',
            vol_ratio > 150 ? '#DE350B' : vol_ratio > 80 ? '#00875A' : '#42526E'],
        ].map(([label, value, color]) => (
          <div key={label} style={{ padding:'6px 8px',
            borderTop:'1px solid var(--border)', borderRight:'1px solid var(--border)' }}>
            <div style={{ fontSize:8, color:'var(--slate-light)', letterSpacing:1,
              textTransform:'uppercase', marginBottom:2 }}>{label}</div>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:11,
              fontWeight:700, color }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Horizon tab content ───────────────────────────────────────────────────────
function HorizonPanel({ pred, currentPrice }) {
  if (!pred) return (
    <div style={{ padding:'12px', textAlign:'center', color:'var(--slate-light)', fontSize:12 }}>
      Prediction not available
    </div>
  )
  const sig       = SIG[pred.signal] || SIG.HOLD
  const SigIcon   = sig.Icon
  const priceDiff = pred.predicted_price - currentPrice

  return (
    <div style={{ padding:'10px 13px 12px' }}>
      {/* Signal badge */}
      <div style={{ display:'inline-flex', alignItems:'center', gap:5,
        padding:'4px 11px', borderRadius:20, background:sig.bg, color:sig.color,
        fontSize:11, fontWeight:800, letterSpacing:.5, marginBottom:9 }}>
        <SigIcon size={11}/>{sig.label}
      </div>

      {/* Price comparison */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:7, marginBottom:9 }}>
        <div style={{ background:'var(--slate-faint)', borderRadius:6, padding:'7px 9px' }}>
          <div style={{ fontSize:9, color:'var(--slate-light)', fontWeight:700,
            textTransform:'uppercase', letterSpacing:1, marginBottom:2 }}>Current</div>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:14,
            color:'#172B4D' }}>₹{fmt(currentPrice)}</div>
        </div>
        <div style={{ background:sig.bg, borderRadius:6, padding:'7px 9px' }}>
          <div style={{ fontSize:9, fontWeight:700, textTransform:'uppercase',
            letterSpacing:1, marginBottom:2, color:sig.color }}>{pred.label}</div>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:700,
            fontSize:14, color:sig.color }}>₹{fmt(pred.predicted_price)}</div>
        </div>
      </div>

      {/* Change row */}
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:9,
        fontSize:11, alignItems:'center' }}>
        <span style={{ fontFamily:'var(--font-mono)', fontWeight:700,
          color:pred.change_pct >= 0 ? 'var(--buy)' : 'var(--sell)' }}>
          {fmtPct(pred.change_pct)}&nbsp;
          <span style={{ opacity:.7 }}>
            ({priceDiff >= 0 ? '+' : ''}₹{fmt(Math.abs(priceDiff))})
          </span>
        </span>
        <span style={{ display:'flex', alignItems:'center', gap:3,
          color:'var(--slate)', fontSize:10 }}>
          <Clock size={10}/> Exit {pred.exit_time}
        </span>
      </div>

      {/* Confidence */}
      <ConfBar value={pred.confidence}/>

      {/* MAPE */}
      <div style={{ marginTop:7, display:'flex', justifyContent:'space-between',
        fontSize:9, color:'var(--slate-light)', fontFamily:'var(--font-mono)' }}>
        <span>VAL MAPE: <span style={{
          color: pred.val_mape <= 3 ? 'var(--buy)' : pred.val_mape <= 5 ? 'var(--hold)' : 'var(--sell)',
          fontWeight:700 }}>{pred.val_mape}%</span></span>
        {pred.sentiment !== undefined && pred.sentiment !== 0 && (
          <span>SENTIMENT: <span style={{
            color: pred.sentiment > 0 ? 'var(--buy)' : 'var(--sell)',
            fontWeight:700 }}>{pred.sentiment > 0 ? '+' : ''}{pred.sentiment}</span></span>
        )}
      </div>
    </div>
  )
}

// ── Skeleton ──────────────────────────────────────────────────────────────────
function Skeleton() {
  return (
    <div style={{ background:'var(--white)', borderRadius:12,
      padding:16, boxShadow:'var(--shadow-sm)', animation:'fade-up .4s ease both' }}>
      {[80,120,52,40,60].map((w,i) => (
        <div key={i} style={{ height:i===2?52:14, width:`${w}%`, borderRadius:4, marginBottom:i<4?10:0,
          background:'linear-gradient(90deg,#f0f2f5 25%,#e8eaed 50%,#f0f2f5 75%)',
          backgroundSize:'400px 100%', animation:'shimmer 1.4s infinite' }}/>
      ))}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN CARD
// ═══════════════════════════════════════════════════════════════════════════════
export default function StockCard({ ticker, initialData }) {
  const [data,     setData]     = useState(initialData || null)
  const [loading,  setLoading]  = useState(!initialData)
  const [error,    setError]    = useState(null)
  const [activeH,  setActiveH]  = useState('t15')
  const [expanded, setExpanded] = useState(false)

  const fetchData = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const res = await api.predict(ticker)
      setData(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [ticker])

  if (loading && !data) return <Skeleton/>
  if (error) return (
    <div style={{ background:'var(--white)', borderRadius:12, padding:16,
      boxShadow:'var(--shadow-sm)', border:'1px solid var(--sell-bg)' }}>
      <div style={{ fontWeight:700, fontSize:13 }}>{ticker.replace('.NS','')}</div>
      <div style={{ color:'var(--sell)', fontSize:12, marginTop:4 }}>{error}</div>
      <button onClick={fetchData} style={{ marginTop:8, padding:'4px 10px', fontSize:11,
        background:'var(--blue-light)', color:'var(--blue)', border:'none',
        borderRadius:4, cursor:'pointer' }}>Retry</button>
    </div>
  )
  if (!data) return null

  const predictions = data.predictions || {}
  const pred        = predictions[activeH]
  const topSig      = predictions['t15']?.signal || 'HOLD'
  const sc          = SECTOR_COLORS[data.sector] || 'var(--blue)'
  const sigCfg      = SIG[topSig] || SIG.HOLD

  return (
    <div style={{ background:'var(--white)', borderRadius:12,
      boxShadow:'var(--shadow-sm)', overflow:'hidden',
      animation:'fade-up .4s ease both',
      transition:'box-shadow .18s, transform .18s' }}
      onMouseEnter={e => { e.currentTarget.style.boxShadow='var(--shadow-md)'; e.currentTarget.style.transform='translateY(-2px)' }}
      onMouseLeave={e => { e.currentTarget.style.boxShadow='var(--shadow-sm)'; e.currentTarget.style.transform='translateY(0)' }}>

      {/* Sector stripe */}
      <div style={{ height:3, background:sc }}/>

      <div style={{ padding:'12px 13px 0' }}>

        {/* Header */}
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:8 }}>
          <div>
            <div style={{ fontWeight:800, fontSize:13, letterSpacing:.5, color:'#172B4D' }}>
              {ticker.replace('.NS','')}
            </div>
            <div style={{ fontSize:9, color:sc, fontWeight:700, textTransform:'uppercase',
              letterSpacing:1, marginTop:1 }}>{data.sector}</div>
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:6 }}>
            {/* Risk qty */}
            <span style={{ fontSize:10, color:'var(--slate)',
              display:'flex', alignItems:'center', gap:3 }}>
              <Package size={10}/>{data.risk_qty ?? '—'}
            </span>
            <button onClick={fetchData} disabled={loading}
              style={{ background:'none', border:'none', cursor:loading?'not-allowed':'pointer',
                padding:3, color:'var(--slate-light)' }}>
              <RefreshCw size={12} style={{ animation:loading?'spin 1s linear infinite':'none' }}/>
            </button>
          </div>
        </div>

        {/* Current price + data interval badge */}
        <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:10 }}>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:800, fontSize:18,
            color:'#172B4D' }}>₹{fmt(data.current_price)}</div>
          {data.data_interval && (
            <span style={{ fontSize:9, padding:'2px 7px', borderRadius:10, fontWeight:700,
              background:'var(--blue-light)', color:'var(--blue)', letterSpacing:.8 }}>
              {data.data_interval} data
            </span>
          )}
          <span style={{ fontSize:9, padding:'2px 7px', borderRadius:10, fontWeight:700,
            background:sigCfg.bg, color:sigCfg.color }}>
            {sigCfg.label}
          </span>
        </div>

        {/* Sparkline */}
        <Sparkline data={data.sparkline} signal={topSig}/>
        <div style={{ fontSize:9, color:'var(--slate-light)', textAlign:'right',
          fontFamily:'var(--font-mono)', marginBottom:10 }}>
          Last 2hrs · Entry {data.entry_time}
        </div>

        {/* Pre-session bids */}
        <PreSession data={data.pre_session}/>

        {/* Horizon tabs */}
        <div style={{ display:'flex', gap:0, marginBottom:0,
          borderBottom:'2px solid var(--border)' }}>
          {H_TABS.map(({ key, label }) => {
            const hp  = predictions[key]
            const sig = hp?.signal || 'HOLD'
            const dotColor = sig==='BUY'?'var(--buy)':sig==='SELL'?'var(--sell)':'var(--hold)'
            return (
              <button key={key} onClick={() => setActiveH(key)} style={{
                flex:1, padding:'6px 0', fontSize:11, fontWeight:700,
                background:'none', border:'none', cursor:'pointer',
                color: key===activeH ? 'var(--blue)' : 'var(--slate-light)',
                borderBottom: key===activeH ? '2px solid var(--blue)' : '2px solid transparent',
                marginBottom:-2, display:'flex', alignItems:'center',
                justifyContent:'center', gap:4, transition:'color .15s',
              }}>
                {hp && (
                  <div style={{ width:6, height:6, borderRadius:'50%', background:dotColor }}/>
                )}
                {label}
              </button>
            )
          })}
        </div>
      </div>

      {/* Active horizon panel */}
      <HorizonPanel pred={pred} currentPrice={data.current_price}/>

      {/* Expand toggle */}
      <div style={{ padding:'0 13px 12px' }}>
        <button onClick={() => setExpanded(v => !v)} style={{
          width:'100%', padding:'5px 0', background:'none',
          border:'1px solid var(--border)', borderRadius:5, cursor:'pointer',
          fontSize:10, color:'var(--slate-light)',
          display:'flex', alignItems:'center', justifyContent:'center', gap:4 }}>
          {expanded ? <ChevronUp size={11}/> : <ChevronDown size={11}/>}
          {expanded ? 'Hide' : 'All Horizons'}
        </button>

        {expanded && (
          <div style={{ marginTop:10, animation:'fade-up .2s ease' }}>
            {/* All 3 horizons summary table */}
            <div style={{ border:'1px solid var(--border)', borderRadius:8, overflow:'hidden',
              marginBottom:10 }}>
              <div style={{ background:'var(--slate-faint)', padding:'6px 10px',
                fontSize:9, fontWeight:700, color:'var(--slate)',
                textTransform:'uppercase', letterSpacing:1 }}>
                All Horizon Summary
              </div>
              <table style={{ width:'100%', borderCollapse:'collapse', fontSize:11 }}>
                <thead>
                  <tr style={{ background:'var(--slate-faint)' }}>
                    {['Horizon','Pred ₹','Chg %','Signal','Conf','MAPE'].map(h => (
                      <th key={h} style={{ padding:'5px 7px', textAlign:'left',
                        fontSize:9, color:'var(--slate)', fontWeight:700,
                        textTransform:'uppercase', letterSpacing:.5 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {H_TABS.map(({ key, label }) => {
                    const p = predictions[key]
                    if (!p) return null
                    const s = SIG[p.signal] || SIG.HOLD
                    return (
                      <tr key={key} style={{ borderTop:'1px solid var(--border)',
                        background: key===activeH ? 'var(--blue-light)' : 'transparent',
                        cursor:'pointer' }}
                        onClick={() => setActiveH(key)}>
                        <td style={{ padding:'6px 7px', fontWeight:700,
                          fontFamily:'var(--font-mono)', color:'var(--blue)', fontSize:10 }}>
                          {label}
                        </td>
                        <td style={{ padding:'6px 7px', fontFamily:'var(--font-mono)' }}>
                          ₹{fmt(p.predicted_price)}
                        </td>
                        <td style={{ padding:'6px 7px', fontFamily:'var(--font-mono)',
                          fontWeight:700,
                          color: p.change_pct >= 0 ? 'var(--buy)' : 'var(--sell)' }}>
                          {fmtPct(p.change_pct)}
                        </td>
                        <td style={{ padding:'6px 7px' }}>
                          <span style={{ padding:'2px 7px', borderRadius:10,
                            background:s.bg, color:s.color, fontSize:10, fontWeight:700 }}>
                            {p.signal}
                          </span>
                        </td>
                        <td style={{ padding:'6px 7px', fontFamily:'var(--font-mono)',
                          color: p.confidence >= 70 ? 'var(--buy)' : 'var(--slate)' }}>
                          {p.confidence}%
                        </td>
                        <td style={{ padding:'6px 7px', fontFamily:'var(--font-mono)',
                          color: p.val_mape <= 3 ? 'var(--buy)' : 'var(--sell)' }}>
                          {p.val_mape}%
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Model info */}
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:5, fontSize:10 }}>
              {[
                ['Data Interval', data.data_interval || '—'],
                ['Risk ₹',        '₹5,000'],
                ['Stop Loss',     '1.5%'],
                ['Last Trained',  data.trained_at
                  ? new Date(data.trained_at+'Z').toLocaleTimeString('en-IN')
                  : '—'],
              ].map(([label, value]) => (
                <div key={label} style={{ background:'var(--slate-faint)',
                  borderRadius:4, padding:'6px 8px' }}>
                  <div style={{ fontSize:9, color:'var(--slate-light)',
                    textTransform:'uppercase', letterSpacing:1, marginBottom:2 }}>{label}</div>
                  <div style={{ fontFamily:'var(--font-mono)', fontWeight:700,
                    color:'#172B4D' }}>{value}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
