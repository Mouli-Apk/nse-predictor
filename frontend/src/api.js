// api.js — centralised API calls with error handling

const BASE = import.meta.env.VITE_API_URL || '/api'

async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health:      ()         => request('/health'),
  watchlist:   ()         => request('/watchlist'),
  predictAll:  ()         => request('/predict-all'),
  predict:     (ticker)   => request(`/predict/${ticker}`),
  backtest:    (ticker)   => request(`/backtest/${ticker}`),
  backtestAll: ()         => request('/backtest-all'),
  scorecard:   ()         => request('/scorecard'),
  retrain:     (tickers)  => request('/retrain', {
    method: 'POST',
    body: JSON.stringify({ tickers: tickers || null }),
  }),
}
