// App.jsx — NSE Intraday Predictor · Main Dashboard
import { useState, useEffect, useCallback } from 'react'
import {
  LayoutGrid, BarChart2, Settings, RefreshCw, Lock,
  TrendingUp, TrendingDown, Minus, Search, ChevronDown,
} from 'lucide-react'
import Header    from './components/Header.jsx'
import StockCard from './components/StockCard.jsx'
import BacktestTab from './components/BacktestTab.jsx'
import LockTracker from './components/LockTracker.jsx'
import { api } from './api'

// ── Sector filter config ──────────────────────────────────────────────────────
const SECTORS = ['All', 'Defense', 'Industrial', 'Green Energy', 'Finance', 'Fintech', 'Consumer', 'Healthcare']
const SIGNALS = ['All', 'BUY', 'SELL', 'HOLD']
const SORT_OPTIONS = [
  { value: 'confidence', label: 'Confidence ↓' },
  { value: 'change_asc', label: 'Change ↑'     },
  { value: 'change_desc',label: 'Change ↓'     },
  { value: 'price_asc',  label: 'Price ↑'      },
  { value: 'alpha',      label: 'A → Z'         },
]

// ── Tab definitions ───────────────────────────────────────────────────────────
const TABS = [
  { id: 'dashboard', label: 'Dashboard',   Icon: LayoutGrid },
  { id: 'backtest',  label: 'Backtester',  Icon: BarChart2  },
  { id: 'locktrack', label: 'Lock & Track',Icon: Lock        },
]

// ── Summary strip ─────────────────────────────────────────────────────────────
function SummaryStrip({ predictions }) {
  if (!predictions?.length) return null
  const buys  = predictions.filter(p => p.signal === 'BUY').length
  const sells = predictions.filter(p => p.signal === 'SELL').length
  const holds = predictions.filter(p => p.signal === 'HOLD').length
  const avgConf = predictions.reduce((a, p) => a + (p.confidence || 0), 0) / predictions.length

  return (
    <div style={{
      display: 'flex', gap: 10, padding: '12px 28px', flexWrap: 'wrap',
      background: 'var(--white)', borderBottom: '1px solid var(--border)',
    }}>
      {[
        { label: 'Buy Signals',  count: buys,  color: 'var(--buy)',  Icon: TrendingUp  },
        { label: 'Sell Signals', count: sells, color: 'var(--sell)', Icon: TrendingDown },
        { label: 'Hold',         count: holds, color: 'var(--hold)', Icon: Minus        },
      ].map(({ label, count, color, Icon }) => (
        <div key={label} style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '5px 12px', borderRadius: 20,
          border: `1px solid ${color}22`,
          background: `${color}11`,
        }}>
          <Icon size={12} style={{ color }} />
          <span style={{ fontSize: 12, fontWeight: 600, color }}>
            {count} {label}
          </span>
        </div>
      ))}
      <div style={{
        marginLeft: 'auto', fontSize: 11, color: 'var(--slate)',
        display: 'flex', alignItems: 'center', gap: 4,
      }}>
        Avg Confidence:
        <strong style={{
          fontFamily: 'var(--font-mono)',
          color: avgConf >= 70 ? 'var(--buy)' : avgConf >= 50 ? 'var(--hold)' : 'var(--sell)',
        }}>
          {avgConf.toFixed(1)}%
        </strong>
      </div>
    </div>
  )
}

// ── Filter / sort toolbar ─────────────────────────────────────────────────────
function Toolbar({ sector, setSector, signal, setSignal, sort, setSort, search, setSearch, onRefresh, loading }) {
  return (
    <div style={{
      display: 'flex', gap: 10, padding: '12px 28px', alignItems: 'center',
      background: 'var(--white)', borderBottom: '1px solid var(--border)',
      flexWrap: 'wrap',
    }}>
      {/* Search */}
      <div style={{ position: 'relative', flex: '0 0 auto' }}>
        <Search size={13} style={{
          position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)',
          color: 'var(--slate-light)',
        }} />
        <input
          type="text" placeholder="Search…" value={search}
          onChange={e => setSearch(e.target.value)}
          style={{
            padding: '7px 12px 7px 28px', borderRadius: 8,
            border: '1px solid var(--border)', fontSize: 12,
            background: 'var(--slate-faint)', width: 140,
            outline: 'none', fontFamily: 'var(--font-body)',
          }}
        />
      </div>

      {/* Sector pills */}
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
        {SECTORS.map(s => (
          <button key={s} onClick={() => setSector(s)} style={{
            padding: '5px 12px', borderRadius: 20, fontSize: 11, fontWeight: 600,
            cursor: 'pointer', transition: 'all 0.15s',
            background: sector === s ? 'var(--blue)' : 'var(--slate-faint)',
            color:      sector === s ? '#fff'        : 'var(--slate)',
            border: sector === s ? '1px solid var(--blue)' : '1px solid var(--border)',
          }}>{s}</button>
        ))}
      </div>

      {/* Signal filter */}
      <div style={{ display: 'flex', gap: 5 }}>
        {SIGNALS.map(s => (
          <button key={s} onClick={() => setSignal(s)} style={{
            padding: '5px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600,
            cursor: 'pointer',
            background: signal === s
              ? s === 'BUY' ? 'var(--buy)' : s === 'SELL' ? 'var(--sell)' : s === 'HOLD' ? 'var(--hold)' : 'var(--blue)'
              : 'var(--slate-faint)',
            color: signal === s ? '#fff' : 'var(--slate)',
            border: '1px solid var(--border)',
          }}>{s}</button>
        ))}
      </div>

      {/* Sort */}
      <div style={{ position: 'relative', marginLeft: 'auto' }}>
        <select
          value={sort}
          onChange={e => setSort(e.target.value)}
          style={{
            padding: '7px 30px 7px 12px', borderRadius: 8, fontSize: 12,
            border: '1px solid var(--border)', background: 'var(--white)',
            color: '#172B4D', cursor: 'pointer', appearance: 'none',
          }}
        >
          {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <ChevronDown size={12} style={{
          position: 'absolute', right: 9, top: '50%', transform: 'translateY(-50%)',
          pointerEvents: 'none', color: 'var(--slate-light)',
        }} />
      </div>

      {/* Refresh */}
      <button
        onClick={onRefresh}
        disabled={loading}
        title="Refresh all predictions"
        style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '7px 12px', borderRadius: 8, fontSize: 12,
          background: 'var(--blue-light)', color: 'var(--blue)',
          border: '1px solid var(--blue)33',
          cursor: loading ? 'not-allowed' : 'pointer',
          fontWeight: 600, opacity: loading ? 0.6 : 1,
        }}
      >
        <RefreshCw size={12} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
        Refresh
      </button>

      <style>{`@keyframes spin { to { transform: rotate(360deg); }}`}</style>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab,         setTab]         = useState('dashboard')
  const [predictions, setPredictions] = useState([])
  const [loading,     setLoading]     = useState(false)
  const [retraining,  setRetraining]  = useState(false)
  const [lastFetch,   setLastFetch]   = useState(null)
  const [error,       setError]       = useState(null)

  // Filters
  const [sector,  setSector]  = useState('All')
  const [signal,  setSignal]  = useState('All')
  const [sort,    setSort]    = useState('confidence')
  const [search,  setSearch]  = useState('')

  // ── Fetch all predictions ──────────────────────────────────────────────────
  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.predictAll()
      setPredictions(res.predictions || [])
      setLastFetch(new Date())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const id = setInterval(fetchAll, 5 * 60 * 1000)
    return () => clearInterval(id)
  }, [fetchAll])

  // ── Retrain all ───────────────────────────────────────────────────────────
  const handleRetrain = async () => {
    setRetraining(true)
    try {
      await api.retrain(null)
      // Wait 10 s then re-fetch
      setTimeout(fetchAll, 10_000)
    } catch (e) {
      setError(e.message)
    } finally {
      setTimeout(() => setRetraining(false), 10_000)
    }
  }

  // ── Filter + sort ─────────────────────────────────────────────────────────
  const displayed = predictions
    .filter(p => {
      if (sector !== 'All' && p.sector !== sector) return false
      if (signal !== 'All' && p.signal !== signal) return false
      if (search && !p.ticker.toLowerCase().includes(search.toLowerCase())) return false
      return true
    })
    .sort((a, b) => {
      switch (sort) {
        case 'confidence':  return (b.confidence || 0) - (a.confidence || 0)
        case 'change_asc':  return (a.change_pct || 0) - (b.change_pct || 0)
        case 'change_desc': return (b.change_pct || 0) - (a.change_pct || 0)
        case 'price_asc':   return (a.current_price || 0) - (b.current_price || 0)
        case 'alpha':       return a.ticker.localeCompare(b.ticker)
        default:            return 0
      }
    })

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* ── Sticky header ───────────────────────────────────────────────────── */}
      <Header onRetrain={handleRetrain} retraining={retraining} />

      {/* ── Tab nav ─────────────────────────────────────────────────────────── */}
      <div style={{
        background: 'var(--white)',
        borderBottom: '2px solid var(--border)',
        display: 'flex', gap: 0, padding: '0 28px',
      }}>
        {TABS.map(({ id, label, Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '13px 18px', fontSize: 13, fontWeight: 600,
              background: 'none', border: 'none', cursor: 'pointer',
              borderBottom: `2px solid ${tab === id ? 'var(--blue)' : 'transparent'}`,
              color: tab === id ? 'var(--blue)' : 'var(--slate-light)',
              marginBottom: -2, transition: 'color 0.15s',
            }}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab content ─────────────────────────────────────────────────────── */}
      {tab === 'backtest' ? (
        <BacktestTab />
      ) : tab === 'locktrack' ? (
        <LockTracker />
      ) : (
        <>
          {/* Summary strip */}
          {predictions.length > 0 && <SummaryStrip predictions={predictions} />}

          {/* Toolbar */}
          <Toolbar
            sector={sector}   setSector={setSector}
            signal={signal}   setSignal={setSignal}
            sort={sort}       setSort={setSort}
            search={search}   setSearch={setSearch}
            onRefresh={fetchAll} loading={loading}
          />

          {/* Error */}
          {error && (
            <div style={{
              margin: '16px 28px 0',
              background: 'var(--sell-bg)', color: 'var(--sell)',
              borderRadius: 8, padding: '10px 16px', fontSize: 13,
            }}>
              ✗ {error} — Is the backend running at <code>localhost:8000</code>?
            </div>
          )}

          {/* Last fetch timestamp */}
          {lastFetch && (
            <div style={{
              padding: '8px 28px 0', fontSize: 11, color: 'var(--slate-light)',
              fontFamily: 'var(--font-mono)',
            }}>
              Last updated: {lastFetch.toLocaleTimeString('en-IN')}
              &nbsp;·&nbsp; Showing {displayed.length} of {predictions.length} stocks
            </div>
          )}

          {/* ── Stock grid ───────────────────────────────────────────────── */}
          <main style={{
            flex: 1,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
            gap: 16,
            padding: '16px 28px 32px',
          }}>
            {displayed.length === 0 && !loading ? (
              <div style={{
                gridColumn: '1/-1', textAlign: 'center',
                padding: '60px 20px', color: 'var(--slate-light)',
              }}>
                {predictions.length === 0
                  ? 'Loading predictions… (backend may still be training models)'
                  : 'No stocks match the current filters'}
              </div>
            ) : (
              displayed.map((pred, i) => (
                <div key={pred.ticker} style={{ animationDelay: `${i * 0.03}s` }}>
                  <StockCard
                    ticker={pred.ticker}
                    initialData={pred}
                    refreshInterval={5 * 60_000}
                  />
                </div>
              ))
            )}
          </main>
        </>
      )}
    </div>
  )
}
