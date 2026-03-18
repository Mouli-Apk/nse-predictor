# NSE Intraday Predictor 📈

> **XGBoost-powered T+15 minute price predictions for 25 high-growth NSE stocks.**
> Self-retraining on 7-day rolling 1-minute data every session. Built for daily professional use.

---

## Architecture

```
nse-predictor/
├── backend/
│   ├── main.py          ← FastAPI server (endpoints: predict, backtest, retrain)
│   ├── ml_logic.py      ← XGBoost pipeline + feature engineering + VADER sentiment
│   ├── config.py        ← Watchlist, model params, risk settings (edit here!)
│   ├── requirements.txt
│   └── pyproject.toml   ← Cloudflare Workers / hatch build config
└── frontend/
    ├── src/
    │   ├── App.jsx               ← Main dashboard with filtering/sorting
    │   ├── api.js                ← API service layer
    │   ├── components/
    │   │   ├── StockCard.jsx     ← Interactive prediction card + sparkline
    │   │   ├── Header.jsx        ← Live scorecard header
    │   │   └── BacktestTab.jsx   ← Yesterday's walk-forward backtest
    │   └── index.css             ← Design system tokens
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Quick Start — CodeSandbox

### 1. Open in CodeSandbox

1. Push this repo to GitHub (instructions below).
2. Go to [codesandbox.io](https://codesandbox.io) → **Import from GitHub** → paste your repo URL.
3. CodeSandbox will auto-detect the Vite frontend.

> **Note:** The ML backend requires Python 3.11+ with XGBoost/yfinance, which runs best in a local terminal or Railway/Render. CodeSandbox is ideal for frontend-only preview against a deployed backend.

### 2. Local Development (Recommended)

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/nse-predictor.git
cd nse-predictor

# ── Backend ──────────────────────────────────────────────
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# ── Frontend (new terminal) ───────────────────────────────
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The backend auto-retrains all 25 models on startup (~3–8 min first run).

---

## GitHub Setup

```bash
git init
git add .
git commit -m "feat: NSE Intraday Predictor v1.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/nse-predictor.git
git push -u origin main
```

---

## Cloudflare Deployment

### Frontend → Cloudflare Pages

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com) → **Pages** → **Create a project**.
2. Connect your GitHub repo.
3. Configure build:
   - **Framework preset:** Vite
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
   - **Root directory:** `frontend`
4. Add environment variable:
   - `VITE_API_URL` = `https://your-backend-url.railway.app`  (or Worker URL)
5. Click **Save and Deploy**.

Every `git push` to `main` triggers a new Cloudflare Pages deployment automatically.

### Backend → Railway (Recommended for ML)

> Cloudflare Workers' Python runtime (Pyodide) does not support native extensions like XGBoost.
> Use **Railway** or **Render** for the FastAPI/ML backend and proxy via a Cloudflare Worker.

```bash
# Install Railway CLI
npm i -g @railway/cli
railway login
railway init
railway up --service backend
```

Set environment variables in Railway:
- `PORT=8000`
- `FRONTEND_URL=https://your-app.pages.dev`

### Backend → Cloudflare Worker (Lightweight / No ML)

If you want a pure Cloudflare Worker for the API layer (forwarding to an external ML service):

```bash
cd backend
npm install -g wrangler
wrangler login
wrangler deploy
```

> The `pyproject.toml` is pre-configured with `compatibility_flags = ["python_workers"]`.
> This mode works for simple CRUD/proxy endpoints but **not** for XGBoost training (use Railway).

---

## API Reference

| Method | Endpoint            | Description                                 |
|--------|---------------------|---------------------------------------------|
| GET    | `/health`           | Server status, models loaded                |
| GET    | `/watchlist`        | All 25 tickers with sector mapping          |
| GET    | `/predict/{ticker}` | Single stock T+15 prediction                |
| GET    | `/predict-all`      | All 25 stocks + session scorecard           |
| POST   | `/retrain`          | Retrain models (body: `{"tickers": [...]}`) |
| GET    | `/backtest/{ticker}`| Yesterday's walk-forward backtest           |
| GET    | `/backtest-all`     | Backtest all 25 stocks                      |
| GET    | `/scorecard`        | Session accuracy metrics                    |

---

## Configuration (`backend/config.py`)

| Setting               | Default    | Description                          |
|-----------------------|------------|--------------------------------------|
| `WATCHLIST`           | 25 stocks  | Edit to add/remove stocks            |
| `PREDICTION_HORIZON`  | 15 min     | T+N forecast window                  |
| `TRAINING_PERIOD_DAYS`| 7          | Rolling training window              |
| `RISK_CAPITAL_INR`    | ₹5,000     | Capital per trade for qty calculation|
| `RISK_PCT`            | 1.5%       | Stop-loss percentage                 |
| `MAX_ERROR_THRESHOLD` | 3.0%       | Backtest pass gate                   |

---

## ML Pipeline

```
yfinance (7d × 1m OHLCV)
         ↓
Feature Engineering:
  RSI(14) · EMA(9,21,50) · EMA ratios
  MACD(12,26,9) · Bollinger Bands(20,2)
  Candle patterns · Volume ratio
  Rate-of-change(1,5,15) · Lagged returns
  Rolling stats · Time-of-day features
         ↓
Sentiment: VADER on yfinance news headlines
         ↓
XGBoost Regressor (n_estimators=400, depth=6)
  → Time-series CV (4 folds, no leakage)
  → Target: % return over next 15 candles
         ↓
Inference: predict % return → convert to price
  → Signal: BUY (>0.4%) / SELL (<-0.4%) / HOLD
  → Sentiment overlay (reverses extreme signals)
  → Risk Qty = ₹5000 / (price × 1.5%)
```

---

## Risk Disclaimer

> This tool is for **educational and research purposes only**.
> Past model performance does not guarantee future results.
> Always apply your own judgment and risk management.
> The author is not responsible for any trading losses.

---

## Tech Stack

| Layer     | Technology                                |
|-----------|-------------------------------------------|
| Frontend  | React 18 · Vite · Recharts · Lucide React |
| Backend   | FastAPI · Uvicorn · Pydantic v2           |
| ML        | XGBoost · scikit-learn · NumPy · pandas   |
| Data      | yfinance (NSE 1-minute OHLCV)             |
| Sentiment | VADER Sentiment                           |
| Hosting   | Cloudflare Pages + Railway                |
| CI/CD     | GitHub → Cloudflare Pages (auto-deploy)   |
| Fonts     | IBM Plex Mono · Sora                      |
