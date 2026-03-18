// StockCard.jsx — Interactive prediction card for a single NSE stock
import { useState, useEffect, useCallback } from 'react'
import {
  AreaChart, Area, ResponsiveContainer, Tooltip, YAxis,
} from 'recharts'
import {
  TrendingUp, TrendingDown, Minus, RefreshCw,
  Clock, Package, Activity, ChevronDown, ChevronUp,
} from 'lucide-react'
import { api } from '../api'

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt = (n, d = 2) =>
  n == null ? '—' : Number(n).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d })

const fmtPct = (n) =>
  n == null ? '—' : `${n >= 0 ? '+' : ''}${Number(n).toFixed(2)}%`

const SIGNAL_CONFIG = {
  BUY:  { label: '▲ BUY NOW',  bg: 'var(--buy-bg)',  color: 'var(--buy)',  Icon: TrendingUp  },
  SELL: { label: '▼ SELL NOW', bg: 'var(--sell-bg)', color: 'var(--sell)', Icon: TrendingDown },
  HOLD: { label: '◆ HOLD',     bg: 'var(--hold-bg)', color: 'var(--hold)', Icon: Minus        },
}

const SECTOR_COLORS = {
  'Defense':     '#0052CC',
  'Industrial':  '#403294',
  'Green Energy':'#00875A',
  'Finance':     '#0065FF',
  'Fintech':     '#6554C0',
  'Consumer':    '#FF5630',
  'Healthcare':  '#00B8D9',
}

// ── Sparkline ─────────────────────────────────────────────────────────────────
function Sparkline({ data, signal }) {
  if (!data || data.length === 0) return (
    <div style={{ height: 52, display: 'flex', alignItems: 'center',
      justifyContent: 'center', color: 'var(--slate-light)', fontSize: 11 }}>
      No chart data
    </div>
  )

  const chartData = data.map((v, i) => ({ i, v }))
  const color = signal === 'BUY' ? '#00875A' : signal === 'SELL' ? '#DE350B' : '#0052CC'

  return (
    <ResponsiveContainer width="100%" height={52}>
      <AreaChart data={chartData} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={`grad-${signal}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.18} />
            <stop offset="95%" stopColor={color} stopOpacity={0}    />
          </linearGradient>
        </defs>
        <YAxis domain={['auto', 'auto']} hide />
        <Tooltip
          content={({ active, payload }) =>
            active && payload?.[0] ? (
              <div style={{
                background: '#172B4D', color: '#fff', padding: '4px 8px',
                borderRadius: 4, fontSize: 11, fontFamily: 'var(--font-mono)',
              }}>
                ₹{fmt(payload[0].value)}
              </div>
            ) : null
          }
        />
        <Area
          type="monotone" dataKey="v"
          stroke={color} strokeWidth={1.5}
          fill={`url(#grad-${signal})`}
          dot={false} isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ── Confidence Bar ────────────────────────────────────────────────────────────
function ConfidenceBar({ value }) {
  const pct = Math.min(100, Math.max(0, value || 0))
  const color = pct >= 70 ? 'var(--buy)' : pct >= 50 ? 'var(--hold)' : 'var(--sell)'
  return (
    <div>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: 10, color: 'var(--slate-light)', marginBottom: 3,
        fontFamily: 'var(--font-mono)',
      }}>
        <span>CONFIDENCE</span>
        <span style={{ color }}>{pct}%</span>
      </div>
      <div style={{
        height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color,
          borderRadius: 2, transition: 'width 0.6s ease',
        }} />
      </div>
    </div>
  )
}

// ── Skeleton loader ───────────────────────────────────────────────────────────
function SkeletonCard() {
  return (
    <div style={{
      background: 'var(--white)', borderRadius: 'var(--radius-lg)',
      padding: 16, boxShadow: 'var(--shadow-sm)',
      animation: 'fade-up 0.4s ease both',
    }}>
      {[80, 120, 52, 40, 60].map((w, i) => (
        <div key={i} style={{
          height: i === 2 ? 52 : 14,
          width: `${w}%`, borderRadius: 4,
          background: 'linear-gradient(90deg,#f0f2f5 25%,#e8eaed 50%,#f0f2f5 75%)',
          backgroundSize: '400px 100%',
          animation: 'shimmer 1.4s infinite',
          marginBottom: i < 4 ? 10 : 0,
        }} />
      ))}
    </div>
  )
}

// ── Main Card ─────────────────────────────────────────────────────────────────
export default function StockCard({ ticker, initialData, refreshInterval = 60_000 }) {
  const [data,     setData]     = useState(initialData || null)
  const [loading,  setLoading]  = useState(!initialData)
  const [error,    setError]    = useState(null)
  const [expanded, setExpanded] = useState(false)
  const [lastRefresh, setLastRefresh] = useState(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.predict(ticker)
      setData(res)
      setLastRefresh(new Date())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [ticker])

  useEffect(() => {
    if (!initialData) fetchData()
  }, [fetchData, initialData])

  useEffect(() => {
    if (!initialData) return         // if bulk-loaded, don't auto-poll per card
    const id = setInterval(fetchData, refreshInterval)
    return () => clearInterval(id)
  }, [fetchData, refreshInterval, initialData])

  if (loading && !data) return <SkeletonCard />

  if (error) return (
    <div style={{
      background: 'var(--white)', borderRadius: 'var(--radius-lg)', padding: 16,
      boxShadow: 'var(--shadow-sm)', animation: 'fade-up 0.4s ease both',
      border: '1px solid var(--sell-bg)',
    }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{ticker}</div>
      <div style={{ color: 'var(--sell)', fontSize: 12 }}>{error}</div>
      <button onClick={fetchData} style={{
        marginTop: 8, padding: '4px 10px', fontSize: 11,
        background: 'var(--blue-light)', color: 'var(--blue)',
        border: 'none', borderRadius: 4, cursor: 'pointer',
      }}>Retry</button>
    </div>
  )

  if (!data) return null

  const sig    = SIGNAL_CONFIG[data.signal] || SIGNAL_CONFIG.HOLD
  const SigIcon = sig.Icon
  const priceDiff = data.predicted_price && data.current_price
    ? data.predicted_price - data.current_price : null
  const sectorColor = SECTOR_COLORS[data.sector] || 'var(--blue)'

  return (
    <div style={{
      background: 'var(--white)',
      borderRadius: 'var(--radius-lg)',
      boxShadow: 'var(--shadow-sm)',
      overflow: 'hidden',
      animation: 'fade-up 0.4s ease both',
      transition: 'box-shadow var(--transition), transform var(--transition)',
      cursor: 'pointer',
    }}
    onMouseEnter={e => {
      e.currentTarget.style.boxShadow = 'var(--shadow-md)'
      e.currentTarget.style.transform = 'translateY(-2px)'
    }}
    onMouseLeave={e => {
      e.currentTarget.style.boxShadow = 'var(--shadow-sm)'
      e.currentTarget.style.transform = 'translateY(0)'
    }}>

      {/* ── Top bar: sector stripe ─────────────────────────────────────────── */}
      <div style={{ height: 3, background: sectorColor }} />

      <div style={{ padding: '12px 14px 14px' }}>

        {/* ── Header row ───────────────────────────────────────────────────── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 13, letterSpacing: '.5px', color: '#172B4D' }}>
              {ticker.replace('.NS', '')}
            </div>
            <div style={{
              fontSize: 10, color: sectorColor, fontWeight: 600,
              textTransform: 'uppercase', letterSpacing: 1,
            }}>
              {data.sector}
            </div>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); fetchData() }}
            disabled={loading}
            title="Refresh prediction"
            style={{
              background: 'none', border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
              padding: 4, color: 'var(--slate-light)', opacity: loading ? 0.5 : 1,
            }}
          >
            <RefreshCw size={13} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          </button>
        </div>

        {/* ── Signal badge ─────────────────────────────────────────────────── */}
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 5,
          padding: '4px 10px', borderRadius: 20,
          background: sig.bg, color: sig.color,
          fontSize: 11, fontWeight: 700, letterSpacing: '.5px',
          marginBottom: 10,
        }}>
          <SigIcon size={11} />
          {sig.label}
        </div>

        {/* ── Price comparison ──────────────────────────────────────────────── */}
        <div style={{
          display: 'grid', gridTemplateColumns: '1fr 1fr',
          gap: 8, marginBottom: 10,
        }}>
          <div style={{
            background: 'var(--slate-faint)', borderRadius: 'var(--radius-sm)', padding: '8px 10px',
          }}>
            <div style={{ fontSize: 9, color: 'var(--slate-light)', fontWeight: 600,
              textTransform: 'uppercase', letterSpacing: 1, marginBottom: 2 }}>Current</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: 15, color: '#172B4D' }}>
              ₹{fmt(data.current_price)}
            </div>
          </div>
          <div style={{
            background: data.signal === 'BUY' ? 'var(--buy-bg)'
              : data.signal === 'SELL' ? 'var(--sell-bg)' : 'var(--hold-bg)',
            borderRadius: 'var(--radius-sm)', padding: '8px 10px',
          }}>
            <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase',
              letterSpacing: 1, marginBottom: 2, color: sig.color }}>T+15 Pred.</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: 15, color: sig.color }}>
              ₹{fmt(data.predicted_price)}
            </div>
          </div>
        </div>

        {/* ── Change & Qty row ──────────────────────────────────────────────── */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 10, fontSize: 11,
        }}>
          <span style={{
            fontFamily: 'var(--font-mono)', fontWeight: 600,
            color: data.change_pct >= 0 ? 'var(--buy)' : 'var(--sell)',
          }}>
            {fmtPct(data.change_pct)} &nbsp;
            {priceDiff != null && (
              <span style={{ opacity: .7 }}>
                ({priceDiff >= 0 ? '+' : ''}₹{fmt(Math.abs(priceDiff))})
              </span>
            )}
          </span>
          <span style={{
            display: 'flex', alignItems: 'center', gap: 4,
            color: 'var(--slate)', fontSize: 11,
          }}>
            <Package size={11} />
            Qty: <strong style={{ fontFamily: 'var(--font-mono)' }}>{data.risk_qty ?? '—'}</strong>
          </span>
        </div>

        {/* ── Sparkline ──────────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 10 }}>
          <Sparkline data={data.sparkline} signal={data.signal} />
          <div style={{ fontSize: 9, color: 'var(--slate-light)', textAlign: 'right',
            marginTop: 2, fontFamily: 'var(--font-mono)' }}>
            Last 2 hrs (1m candles)
          </div>
        </div>

        {/* ── Confidence bar ────────────────────────────────────────────────── */}
        <ConfidenceBar value={data.confidence} />

        {/* ── Entry / Exit ──────────────────────────────────────────────────── */}
        <div style={{
          display: 'flex', justifyContent: 'space-between',
          marginTop: 10, fontSize: 10, color: 'var(--slate-light)',
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            <Clock size={10} /> Entry {data.entry_time}
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            Exit {data.exit_time} <Clock size={10} />
          </span>
        </div>

        {/* ── Expandable details ────────────────────────────────────────────── */}
        <button
          onClick={() => setExpanded(v => !v)}
          style={{
            width: '100%', marginTop: 10, padding: '5px 0',
            background: 'none', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)', cursor: 'pointer',
            fontSize: 10, color: 'var(--slate-light)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
          }}
        >
          {expanded ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          {expanded ? 'Hide' : 'Details'}
        </button>

        {expanded && (
          <div style={{
            marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border)',
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6,
            fontSize: 10, animation: 'fade-up 0.2s ease',
          }}>
            {[
              ['Val MAPE',  `${data.val_mape}%`],
              ['Sentiment', data.sentiment != null
                ? (data.sentiment > 0 ? `+${data.sentiment}` : `${data.sentiment}`)
                : '—'],
              ['Risk ₹',    `₹5,000`],
              ['Stop Loss', '1.5%'],
            ].map(([label, value]) => (
              <div key={label} style={{
                background: 'var(--slate-faint)', borderRadius: 4, padding: '6px 8px',
              }}>
                <div style={{ color: 'var(--slate-light)', marginBottom: 2, letterSpacing: 1 }}>
                  {label.toUpperCase()}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#172B4D' }}>
                  {value}
                </div>
              </div>
            ))}
            <div style={{
              gridColumn: '1/-1', background: 'var(--slate-faint)',
              borderRadius: 4, padding: '6px 8px',
            }}>
              <div style={{ color: 'var(--slate-light)', marginBottom: 2, letterSpacing: 1 }}>
                LAST RETRAINED
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#172B4D' }}>
                {data.trained_at
                  ? new Date(data.trained_at + 'Z').toLocaleTimeString('en-IN')
                  : '—'}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
