// BacktestTab.jsx — Yesterday's walk-forward backtest results
import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts'
import { CheckCircle, XCircle, Play, Loader } from 'lucide-react'
import { api } from '../api'

const WATCHLIST = [
  'MAZDOCK.NS','BEL.NS','BHEL.NS','BHARATFORG.NS','SOLARINDS.NS','COCHINSHIP.NS',
  'WAAREEENER.NS','TATAPOWER.NS','SUZLON.NS','IREDA.NS','NHPC.NS','JSWENERGY.NS','KPIGREEN.NS',
  'MUTHOOTFIN.NS','HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','PFC.NS','RECLTD.NS','ABCAPITAL.NS','PBFINTECH.NS',
  'TRENT.NS','DIXON.NS','MAXHEALTH.NS','IHCL.NS',
]

function PassBadge({ passed }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 8px', borderRadius: 20, fontSize: 11, fontWeight: 700,
      background: passed ? 'var(--buy-bg)'  : 'var(--sell-bg)',
      color:      passed ? 'var(--buy)'     : 'var(--sell)',
    }}>
      {passed ? <CheckCircle size={11} /> : <XCircle size={11} />}
      {passed ? 'PASS' : 'FAIL'}
    </span>
  )
}

function MapeBar({ data }) {
  const threshold = 3.0
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 40, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
        <XAxis
          dataKey="ticker"
          tick={{ fontSize: 9, fontFamily: 'var(--font-mono)' }}
          angle={-55} textAnchor="end" interval={0}
        />
        <YAxis
          tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
          tickFormatter={v => `${v}%`}
          width={36}
        />
        <Tooltip
          formatter={(v) => [`${v.toFixed(2)}%`, 'MAPE']}
          contentStyle={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}
        />
        <ReferenceLine
          y={threshold} stroke="var(--sell)"
          strokeDasharray="6 3" strokeWidth={2}
          label={{ value: '3% target', position: 'insideTopRight', fontSize: 10, fill: 'var(--sell)' }}
        />
        <Bar dataKey="mape" radius={[3, 3, 0, 0]} maxBarSize={28}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.mape <= threshold ? 'var(--buy)' : 'var(--sell)'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function BacktestTab() {
  const [results,  setResults]  = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [selected, setSelected] = useState(null)
  const [error,    setError]    = useState(null)

  const runAll = async () => {
    setLoading(true); setError(null); setSelected(null)
    try {
      const r = await api.backtestAll()
      setResults(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const runOne = async (ticker) => {
    setLoading(true); setError(null)
    try {
      const r = await api.backtest(ticker)
      // merge into results
      setResults(prev => {
        if (!prev) return { results: [r], total: 1, passed: r.passed ? 1 : 0, pass_rate_pct: r.passed ? 100 : 0 }
        const updated = prev.results.map(x => x.ticker === ticker ? r : x)
        const passed  = updated.filter(x => x.passed).length
        return { ...prev, results: updated, passed, pass_rate_pct: (passed / updated.length * 100).toFixed(1) }
      })
      setSelected(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Support both old flat mape and new overall_mape from multi-horizon response
  const getMape = r => r.overall_mape ?? r.mape ?? 999
  const chartData = results?.results
    ?.filter(r => r.status === 'ok')
    .map(r => ({ ticker: r.ticker.replace('.NS', ''), mape: getMape(r) }))
    .sort((a, b) => a.mape - b.mape) || []

  return (
    <div style={{ padding: '24px 28px', maxWidth: 1200 }}>

      {/* ── Page title ────────────────────────────────────────────────────── */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, color: '#172B4D' }}>
          Permanent Backtester
        </h2>
        <p style={{ color: 'var(--slate)', fontSize: 13, marginTop: 4 }}>
          Walk-forward validation on yesterday's 1-minute data.
          Model must achieve &lt;3% MAPE to gate today's trading.
        </p>
      </div>

      {/* ── Run controls ──────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 24, flexWrap: 'wrap' }}>
        <button
          onClick={runAll}
          disabled={loading}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '9px 20px', borderRadius: 8, fontSize: 13, fontWeight: 600,
            background: 'var(--blue)', color: '#fff', border: 'none',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? <Loader size={14} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={14} />}
          {loading ? 'Running…' : 'Run Backtest (All 25)'}
        </button>

        <select
          onChange={e => e.target.value && runOne(e.target.value)}
          disabled={loading}
          defaultValue=""
          style={{
            padding: '9px 14px', borderRadius: 8, fontSize: 13,
            border: '1px solid var(--border)', background: 'var(--white)',
            color: '#172B4D', cursor: 'pointer', minWidth: 180,
          }}
        >
          <option value="">Run single stock…</option>
          {WATCHLIST.map(t => <option key={t} value={t}>{t.replace('.NS', '')}</option>)}
        </select>
      </div>

      {error && (
        <div style={{
          background: 'var(--sell-bg)', color: 'var(--sell)',
          borderRadius: 8, padding: '10px 16px', marginBottom: 20,
          fontSize: 13,
        }}>
          ✗ {error}
        </div>
      )}

      {/* ── Summary cards ─────────────────────────────────────────────────── */}
      {results && (
        <>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px,1fr))',
            gap: 12, marginBottom: 24,
          }}>
            {[
              { label: 'Total Tested',  value: results.total,         color: 'var(--blue)' },
              { label: 'Passed (<3%)',  value: results.passed,        color: 'var(--buy)' },
              { label: 'Failed',        value: results.total - results.passed, color: 'var(--sell)' },
              { label: 'Pass Rate',     value: `${results.pass_rate_pct}%`, color:
                results.pass_rate_pct >= 80 ? 'var(--buy)' : 'var(--hold)' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                background: 'var(--white)', borderRadius: 10,
                padding: '14px 16px', boxShadow: 'var(--shadow-sm)',
              }}>
                <div style={{ fontSize: 10, color: 'var(--slate-light)',
                  textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                  {label}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 22,
                  fontWeight: 700, color }}>
                  {value}
                </div>
              </div>
            ))}
          </div>

          {/* ── MAPE Bar Chart ─────────────────────────────────────────────── */}
          {chartData.length > 0 && (
            <div style={{
              background: 'var(--white)', borderRadius: 10,
              padding: 20, boxShadow: 'var(--shadow-sm)', marginBottom: 20,
            }}>
              <h3 style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, color: '#172B4D' }}>
                MAPE by Stock (↑ red = failed threshold)
              </h3>
              <MapeBar data={chartData} />
            </div>
          )}

          {/* ── Per-stock table ────────────────────────────────────────────── */}
          <div style={{
            background: 'var(--white)', borderRadius: 10,
            boxShadow: 'var(--shadow-sm)', overflow: 'hidden',
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: 'var(--slate-faint)' }}>
                  {['Ticker', 'Date', 'MAPE %', 'Status', 'Steps', ''].map(h => (
                    <th key={h} style={{
                      padding: '10px 14px', textAlign: 'left', fontWeight: 700,
                      fontSize: 10, textTransform: 'uppercase', letterSpacing: 1,
                      color: 'var(--slate)',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.results.map((r, i) => (
                  <tr key={r.ticker} style={{
                    borderTop: '1px solid var(--border)',
                    background: selected?.ticker === r.ticker ? 'var(--blue-light)' : (i % 2 ? 'var(--slate-faint)' : 'var(--white)'),
                    cursor: r.status === 'ok' ? 'pointer' : 'default',
                    transition: 'background 0.15s',
                  }}
                    onClick={() => r.status === 'ok' && setSelected(selected?.ticker === r.ticker ? null : r)}
                  >
                    <td style={{ padding: '9px 14px', fontWeight: 700,
                      fontFamily: 'var(--font-mono)' }}>
                      {r.ticker.replace('.NS', '')}
                    </td>
                    <td style={{ padding: '9px 14px', color: 'var(--slate)',
                      fontFamily: 'var(--font-mono)' }}>
                      {r.date || '—'}
                    </td>
                    <td style={{ padding: '9px 14px', fontFamily: 'var(--font-mono)',
                      color: getMape(r) <= 3 ? 'var(--buy)' : 'var(--sell)', fontWeight: 600 }}>
                      {r.status === 'ok'
                        ? <>
                            <span>{getMape(r).toFixed(2)}%</span>
                            {r.overall_dir_acc != null && (
                              <span style={{ fontSize:10, color:'var(--slate-light)', marginLeft:6 }}>
                                dir:{r.overall_dir_acc.toFixed(0)}%
                              </span>
                            )}
                          </>
                        : r.message || '—'}
                    </td>
                    <td style={{ padding: '9px 14px' }}>
                      {r.status === 'ok' ? <PassBadge passed={r.passed} /> : (
                        <span style={{ color: 'var(--slate-light)', fontSize: 11 }}>
                          {r.status}
                        </span>
                      )}
                    </td>
                    <td style={{ padding: '9px 14px', color: 'var(--slate)',
                      fontFamily: 'var(--font-mono)' }}>
                      {r.steps?.length ?? '—'}
                    </td>
                    <td style={{ padding: '9px 14px', color: 'var(--blue)', fontSize: 11 }}>
                      {r.status === 'ok' ? (selected?.ticker === r.ticker ? '▲ Hide' : '▼ Steps') : ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Step detail panel ─────────────────────────────────────────── */}
          {selected?.steps && (
            <div style={{
              marginTop: 16, background: 'var(--white)', borderRadius: 10,
              boxShadow: 'var(--shadow-sm)', overflow: 'hidden',
              animation: 'fade-up 0.2s ease',
            }}>
              <div style={{
                padding: '12px 16px', background: 'var(--blue)',
                color: '#fff', fontSize: 13, fontWeight: 700,
                display: 'flex', justifyContent: 'space-between',
              }}>
                <span>{selected.ticker} — Step Details</span>
                <PassBadge passed={selected.passed} />
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: 'var(--slate-faint)' }}>
                      {['Time', 'Actual ₹', 'Predicted ₹', 'True ₹', 'Price Err%', 'Dir'].map(h => (
                        <th key={h} style={{
                          padding: '8px 12px', textAlign: 'left', fontWeight: 600,
                          fontSize: 10, textTransform: 'uppercase', color: 'var(--slate)',
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selected.steps.map((s, i) => (
                      <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                        <td style={{ padding: '7px 12px', fontFamily: 'var(--font-mono)',
                          color: 'var(--slate)' }}>
                          {new Date(s.time).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
                        </td>
                        {[s.actual_price, s.predicted_price, s.true_future].map((v, j) => (
                          <td key={j} style={{ padding: '7px 12px',
                            fontFamily: 'var(--font-mono)', fontWeight: 500 }}>
                            ₹{v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                          </td>
                        ))}
                        <td style={{
                          padding: '7px 12px', fontFamily: 'var(--font-mono)',
                          fontWeight: 700,
                          color: (s.price_error_pct ?? s.error_pct ?? 0) <= 3 ? 'var(--buy)' : 'var(--sell)',
                        }}>
                          {((s.price_error_pct ?? s.error_pct) || 0).toFixed(2)}%
                        </td>
                        <td style={{ padding:'7px 12px', fontSize:11, fontWeight:700,
                          color: s.dir_correct ? 'var(--buy)' : s.dir_correct === false ? 'var(--sell)' : 'var(--slate-light)',
                        }}>
                          {s.dir_correct != null ? (s.dir_correct ? '✓' : '✗') : '—'}
                          {s.pred_direction ? ` ${s.pred_direction}` : ''}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {!results && !loading && (
        <div style={{
          textAlign: 'center', padding: '60px 20px',
          color: 'var(--slate-light)',
        }}>
          <Play size={40} style={{ opacity: .3, marginBottom: 12 }} />
          <p style={{ fontSize: 14 }}>
            Click <strong>Run Backtest</strong> to validate model accuracy against yesterday's data
            before starting the trading session.
          </p>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); }}`}</style>
    </div>
  )
}
