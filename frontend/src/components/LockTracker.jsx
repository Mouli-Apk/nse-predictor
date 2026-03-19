// LockTracker.jsx
// Lock any prediction at a specific time, then compare actual vs predicted.
// Uses localStorage so locks persist across browser sessions.
// NOT an artifact — this is a real deployed app, localStorage is safe here.

import { useState, useEffect, useCallback } from 'react'
import { Lock, Unlock, CheckCircle, Clock, Trash2, TrendingUp,
         TrendingDown, Minus, RefreshCw, AlertCircle } from 'lucide-react'
import { api } from '../api'

// ── Storage helpers ────────────────────────────────────────────────────────────
const STORAGE_KEY = 'nse_prediction_locks'

function loadLocks() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
  } catch { return [] }
}
function saveLocks(locks) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(locks))
}

// ── Helpers ────────────────────────────────────────────────────────────────────
const fmt    = (n, d = 2) => n == null ? '—'
  : Number(n).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d })
const fmtPct = n => n == null ? '—' : `${n >= 0 ? '+' : ''}${Number(n).toFixed(2)}%`

function istNow() {
  const now = new Date()
  // IST = UTC + 5:30
  const ist = new Date(now.getTime() + (5.5 * 3600 * 1000))
  return ist
}
function istString(date) {
  return new Date(date).toLocaleString('en-IN', {
    timeZone: 'Asia/Kolkata',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    day: '2-digit', month: 'short',
  })
}
function minutesUntil(isoExitTime) {
  const exit = new Date(isoExitTime)
  const now  = new Date()
  return Math.round((exit - now) / 60000)
}

// ── Status badge ──────────────────────────────────────────────────────────────
function StatusBadge({ lock }) {
  const mins = minutesUntil(lock.exit_iso)

  if (lock.actual_price != null) {
    const err  = Math.abs(lock.actual_price - lock.predicted_price) / lock.actual_price * 100
    const pass = err <= 3
    return (
      <span style={{
        display:'inline-flex', alignItems:'center', gap:4,
        padding:'3px 9px', borderRadius:12, fontSize:11, fontWeight:700,
        background: pass ? 'var(--buy-bg)' : 'var(--sell-bg)',
        color:      pass ? 'var(--buy)'    : 'var(--sell)',
      }}>
        {pass ? <CheckCircle size={11}/> : <AlertCircle size={11}/>}
        {pass ? 'PASSED' : 'FAILED'} ({err.toFixed(2)}%)
      </span>
    )
  }

  if (mins <= 0) {
    return (
      <span style={{ display:'inline-flex', alignItems:'center', gap:4,
        padding:'3px 9px', borderRadius:12, fontSize:11, fontWeight:700,
        background:'var(--blue-light)', color:'var(--blue)' }}>
        <CheckCircle size={11}/> Ready to check
      </span>
    )
  }

  return (
    <span style={{ display:'inline-flex', alignItems:'center', gap:4,
      padding:'3px 9px', borderRadius:12, fontSize:11, fontWeight:700,
      background:'var(--hold-bg)', color:'var(--hold)' }}>
      <Clock size={11}/> {mins}m remaining
    </span>
  )
}

// ── Individual lock row ────────────────────────────────────────────────────────
function LockRow({ lock, onCheck, onDelete, checking }) {
  const mins       = minutesUntil(lock.exit_iso)
  const canCheck   = mins <= 0 && lock.actual_price == null
  const isChecked  = lock.actual_price != null
  const diff       = isChecked ? lock.actual_price - lock.predicted_price : null
  const priceMape  = isChecked
    ? Math.abs(lock.actual_price - lock.predicted_price) / lock.actual_price * 100
    : null

  const sigColor = lock.signal === 'BUY' ? 'var(--buy)'
    : lock.signal === 'SELL' ? 'var(--sell)' : 'var(--hold)'
  const sigBg = lock.signal === 'BUY' ? 'var(--buy-bg)'
    : lock.signal === 'SELL' ? 'var(--sell-bg)' : 'var(--hold-bg)'

  return (
    <div style={{
      background: 'var(--white)',
      border: `1px solid ${isChecked
        ? (priceMape <= 3 ? 'rgba(0,135,90,0.3)' : 'rgba(222,53,11,0.3)')
        : 'var(--border)'}`,
      borderRadius: 10,
      padding: '12px 14px',
      animation: 'fade-up .3s ease both',
    }}>
      {/* Header row */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:10 }}>
        <div>
          <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:4 }}>
            <span style={{ fontWeight:800, fontSize:14, color:'#172B4D' }}>
              {lock.ticker.replace('.NS','')}
            </span>
            <span style={{ fontSize:10, padding:'2px 8px', borderRadius:10, fontWeight:700,
              background: sigBg, color: sigColor }}>{lock.signal}</span>
            <span style={{ fontSize:10, padding:'2px 8px', borderRadius:10, fontWeight:700,
              background:'var(--blue-light)', color:'var(--blue)' }}>
              {lock.horizon_label}
            </span>
          </div>
          <div style={{ fontSize:10, color:'var(--slate-light)', fontFamily:'var(--font-mono)' }}>
            🔒 Locked at {istString(lock.locked_at)} IST
          </div>
          <div style={{ fontSize:10, color:'var(--slate-light)', fontFamily:'var(--font-mono)' }}>
            ⏰ Exit by {istString(lock.exit_iso)} IST
          </div>
        </div>
        <div style={{ display:'flex', gap:6, alignItems:'center' }}>
          <StatusBadge lock={lock}/>
          {canCheck && (
            <button onClick={() => onCheck(lock.id)} disabled={checking}
              style={{
                display:'flex', alignItems:'center', gap:4,
                padding:'5px 12px', borderRadius:8, fontSize:11, fontWeight:700,
                background:'var(--blue)', color:'#fff', border:'none', cursor:'pointer',
              }}>
              {checking ? <RefreshCw size={11} style={{ animation:'spin 1s linear infinite' }}/> : <CheckCircle size={11}/>}
              Check
            </button>
          )}
          <button onClick={() => onDelete(lock.id)} style={{
            background:'none', border:'none', cursor:'pointer',
            color:'var(--slate-light)', padding:4,
          }}>
            <Trash2 size={13}/>
          </button>
        </div>
      </div>

      {/* Price comparison grid */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(110px,1fr))', gap:7 }}>
        <div style={{ background:'var(--slate-faint)', borderRadius:7, padding:'8px 10px' }}>
          <div style={{ fontSize:9, color:'var(--slate-light)', textTransform:'uppercase',
            letterSpacing:1, marginBottom:3, fontWeight:700 }}>Price at Lock</div>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:13 }}>
            ₹{fmt(lock.price_at_lock)}
          </div>
        </div>
        <div style={{ background: sigBg, borderRadius:7, padding:'8px 10px' }}>
          <div style={{ fontSize:9, textTransform:'uppercase', letterSpacing:1,
            marginBottom:3, fontWeight:700, color:sigColor }}>Predicted ({lock.horizon_label})</div>
          <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:13, color:sigColor }}>
            ₹{fmt(lock.predicted_price)}
            <span style={{ fontSize:10, marginLeft:4, opacity:.75 }}>
              ({fmtPct(lock.change_pct)})
            </span>
          </div>
        </div>
        {isChecked && (
          <div style={{
            background: priceMape <= 3 ? 'var(--buy-bg)' : 'var(--sell-bg)',
            borderRadius:7, padding:'8px 10px',
          }}>
            <div style={{ fontSize:9, textTransform:'uppercase', letterSpacing:1,
              marginBottom:3, fontWeight:700,
              color: priceMape <= 3 ? 'var(--buy)' : 'var(--sell)' }}>
              Actual Price
            </div>
            <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:13,
              color: priceMape <= 3 ? 'var(--buy)' : 'var(--sell)' }}>
              ₹{fmt(lock.actual_price)}
              <span style={{ fontSize:10, marginLeft:4, opacity:.75 }}>
                ({diff >= 0 ? '+' : ''}₹{fmt(Math.abs(diff))})
              </span>
            </div>
          </div>
        )}
        {isChecked && (
          <div style={{
            background: priceMape <= 3 ? 'var(--buy-bg)' : 'var(--sell-bg)',
            borderRadius:7, padding:'8px 10px',
          }}>
            <div style={{ fontSize:9, textTransform:'uppercase', letterSpacing:1,
              marginBottom:3, fontWeight:700,
              color: priceMape <= 3 ? 'var(--buy)' : 'var(--sell)' }}>
              Accuracy
            </div>
            <div style={{ fontFamily:'var(--font-mono)', fontWeight:700, fontSize:16,
              color: priceMape <= 3 ? 'var(--buy)' : 'var(--sell)' }}>
              {priceMape <= 3 ? '✓ ' : '✗ '}{priceMape.toFixed(2)}% err
            </div>
          </div>
        )}
      </div>

      {/* Confidence & MAPE at lock time */}
      <div style={{ marginTop:8, display:'flex', gap:12, fontSize:10,
        color:'var(--slate-light)', fontFamily:'var(--font-mono)' }}>
        <span>Confidence at lock: <strong style={{ color:'var(--slate)' }}>{lock.confidence}%</strong></span>
        <span>Val MAPE: <strong style={{ color: lock.val_mape <= 3 ? 'var(--buy)' : 'var(--sell)' }}>
          {lock.val_mape}%</strong></span>
      </div>
    </div>
  )
}

// ── Summary stats ─────────────────────────────────────────────────────────────
function SummaryBar({ locks }) {
  const checked = locks.filter(l => l.actual_price != null)
  if (checked.length === 0) return null

  const passed  = checked.filter(l => {
    const err = Math.abs(l.actual_price - l.predicted_price) / l.actual_price * 100
    return err <= 3
  }).length
  const avgErr  = checked.reduce((s, l) => {
    return s + Math.abs(l.actual_price - l.predicted_price) / l.actual_price * 100
  }, 0) / checked.length

  return (
    <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:8, marginBottom:16 }}>
      {[
        ['Total Locked',     locks.length,                      'var(--blue)'],
        ['Checked',          checked.length,                    'var(--slate)'],
        ['Passed (<3%)',     passed,                            'var(--buy)'],
        ['Avg Price Error',  `${avgErr.toFixed(2)}%`,           avgErr <= 3 ? 'var(--buy)' : 'var(--sell)'],
      ].map(([label, value, color]) => (
        <div key={label} style={{ background:'var(--white)', borderRadius:8, padding:'10px 12px',
          boxShadow:'var(--shadow-sm)' }}>
          <div style={{ fontSize:9, color:'var(--slate-light)', textTransform:'uppercase',
            letterSpacing:1, marginBottom:3 }}>{label}</div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:18,
            fontWeight:700, color }}>{value}</div>
        </div>
      ))}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════
export default function LockTracker() {
  const [locks,    setLocks]    = useState(() => loadLocks())
  const [checking, setChecking] = useState(null)  // lock id being checked

  // Sync to localStorage whenever locks change
  useEffect(() => { saveLocks(locks) }, [locks])

  const deleteLock = useCallback((id) => {
    setLocks(prev => prev.filter(l => l.id !== id))
  }, [])

  const checkLock = useCallback(async (id) => {
    const lock = locks.find(l => l.id === id)
    if (!lock) return
    setChecking(id)
    try {
      const result = await api.predict(lock.ticker)
      const actual = result.current_price
      if (actual) {
        setLocks(prev => prev.map(l => l.id === id
          ? { ...l, actual_price: actual, checked_at: new Date().toISOString() }
          : l
        ))
      }
    } catch (e) {
      console.error('Check failed:', e)
    } finally {
      setChecking(null)
    }
  }, [locks])

  const clearAll = () => {
    if (window.confirm('Clear all locked predictions?')) {
      setLocks([])
    }
  }

  const clearChecked = () => {
    setLocks(prev => prev.filter(l => l.actual_price == null))
  }

  const pending  = locks.filter(l => l.actual_price == null)
  const resolved = locks.filter(l => l.actual_price != null)

  if (locks.length === 0) return (
    <div style={{ padding:'24px 28px' }}>
      <div style={{ textAlign:'center', padding:'60px 20px', color:'var(--slate-light)' }}>
        <Lock size={42} style={{ opacity:.25, marginBottom:14 }}/>
        <p style={{ fontSize:15, fontWeight:600, marginBottom:8 }}>No predictions locked yet</p>
        <p style={{ fontSize:13 }}>
          On any stock card, click a horizon tab (T+15m, T+1h, T+3h) then click
          <strong> 🔒 Lock</strong> to save the prediction. Come back after the window
          expires to check how accurate the model was.
        </p>
      </div>
    </div>
  )

  return (
    <div style={{ padding:'24px 28px', maxWidth:900 }}>
      {/* Title + actions */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:20 }}>
        <div>
          <h2 style={{ fontSize:20, fontWeight:700, color:'#172B4D' }}>
            Lock & Track
          </h2>
          <p style={{ color:'var(--slate)', fontSize:13, marginTop:4 }}>
            Predictions locked in this session — check accuracy after the horizon expires
          </p>
        </div>
        <div style={{ display:'flex', gap:8 }}>
          {resolved.length > 0 && (
            <button onClick={clearChecked} style={{
              padding:'7px 14px', borderRadius:8, fontSize:12, fontWeight:600,
              background:'var(--slate-faint)', color:'var(--slate)',
              border:'1px solid var(--border)', cursor:'pointer',
            }}>
              Clear Checked ({resolved.length})
            </button>
          )}
          <button onClick={clearAll} style={{
            padding:'7px 14px', borderRadius:8, fontSize:12, fontWeight:600,
            background:'var(--sell-bg)', color:'var(--sell)',
            border:'1px solid rgba(222,53,11,.2)', cursor:'pointer',
          }}>
            Clear All
          </button>
        </div>
      </div>

      {/* Summary stats */}
      <SummaryBar locks={locks}/>

      {/* Pending locks */}
      {pending.length > 0 && (
        <div style={{ marginBottom:20 }}>
          <div style={{ fontSize:11, fontWeight:700, textTransform:'uppercase',
            letterSpacing:1, color:'var(--slate)', marginBottom:10 }}>
            ⏰ Waiting ({pending.length})
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
            {pending.map(lock => (
              <LockRow key={lock.id} lock={lock}
                onCheck={checkLock} onDelete={deleteLock}
                checking={checking === lock.id}/>
            ))}
          </div>
        </div>
      )}

      {/* Resolved locks */}
      {resolved.length > 0 && (
        <div>
          <div style={{ fontSize:11, fontWeight:700, textTransform:'uppercase',
            letterSpacing:1, color:'var(--slate)', marginBottom:10 }}>
            ✅ Checked ({resolved.length})
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
            {resolved.slice().reverse().map(lock => (
              <LockRow key={lock.id} lock={lock}
                onCheck={checkLock} onDelete={deleteLock}
                checking={checking === lock.id}/>
            ))}
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform:rotate(360deg); } }`}</style>
    </div>
  )
}

// ── Exported helper — called from StockCard when user clicks Lock ─────────────
export function lockPrediction({ ticker, sector, signal, horizon_key, horizon_label,
  predicted_price, price_at_lock, change_pct, confidence, val_mape, horizon_mins }) {

  const locked_at = new Date().toISOString()
  const exit_iso  = new Date(Date.now() + horizon_mins * 60 * 1000).toISOString()

  const newLock = {
    id:              `${ticker}-${horizon_key}-${Date.now()}`,
    ticker, sector, signal, horizon_key, horizon_label,
    predicted_price, price_at_lock, change_pct, confidence, val_mape,
    horizon_mins, locked_at, exit_iso,
    actual_price: null,
    checked_at:   null,
  }

  const existing = loadLocks()
  saveLocks([...existing, newLock])
  return newLock
}
