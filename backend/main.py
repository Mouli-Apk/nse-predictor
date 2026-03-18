"""
main.py — NSE Intraday Predictor · FastAPI Backend
Endpoints:
  GET  /health
  GET  /watchlist
  GET  /predict/{ticker}
  GET  /predict-all
  POST /retrain          (body: {"tickers": [...]} or empty for all)
  GET  /backtest/{ticker}
  GET  /backtest-all
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import ml_logic

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("main")

# ── Session accuracy tracker ──────────────────────────────────────────────────
#   Tracks predictions made this session: {ticker: [(predicted_pct, actual_pct)]}
_session_stats: dict[str, list[dict]] = {}
_session_start: datetime = datetime.utcnow()


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP — retrain all models at launch
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Start server immediately so Render detects the open port ──────────────
    # Training runs in the background AFTER the port is bound.
    logger.info("🚀  NSE Predictor starting — port binding first, training in background.")
    asyncio.create_task(_background_train_all())
    yield
    logger.info("🛑  Shutting down.")


async def _background_train_all() -> None:
    """Train all models in background after server has started."""
    # Small delay to ensure server is fully up before heavy CPU work
    await asyncio.sleep(3)
    logger.info("📊  Background training started for %d tickers…", len(config.WATCHLIST))
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, ml_logic.train_all, config.WATCHLIST)
        ok  = sum(1 for r in results if r.get("status") == "ok")
        err = len(results) - ok
        logger.info("✅  Background training complete: %d/%d models trained. %d skipped.",
                    ok, len(config.WATCHLIST), err)
    except Exception as exc:
        logger.error("❌  Background training error: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# APP FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NSE Intraday Predictor",
    version="1.0.0",
    description="XGBoost-powered T+15 min price prediction for 25 high-growth NSE stocks.",
    lifespan=lifespan,
)

# Allow Cloudflare Pages frontend + local dev
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://*.pages.dev",
    os.environ.get("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in production using ALLOWED_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RetrainRequest(BaseModel):
    tickers: list[str] | None = None


class PredictionResponse(BaseModel):
    ticker:          str
    sector:          str
    status:          str
    current_price:   float | None
    predicted_price: float | None
    change_pct:      float | None
    signal:          str
    confidence:      float
    val_mape:        float | None = None
    sentiment:       float | None = None
    risk_qty:        int | None   = None
    entry_time:      str | None   = None
    exit_time:       str | None   = None
    sparkline:       list[float]  = []
    trained_at:      str | None   = None
    message:         str | None   = None


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict:
    return {
        "status":         "ok",
        "timestamp":      datetime.utcnow().isoformat(),
        "models_loaded":  len(ml_logic._MODEL_REGISTRY),
        "watchlist_size": len(config.WATCHLIST),
        "session_start":  _session_start.isoformat(),
    }


@app.get("/watchlist")
def get_watchlist() -> dict:
    return {
        "tickers":  config.WATCHLIST,
        "sectors":  config.SECTOR_MAP,
        "count":    len(config.WATCHLIST),
    }


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict_ticker(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in config.WATCHLIST:
        raise HTTPException(status_code=404, detail=f"{ticker} not in watchlist")
    try:
        result = ml_logic.predict(ticker)
        _log_prediction(result)
        return result
    except Exception as exc:
        logger.exception("Prediction failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict-all")
def predict_all() -> dict:
    results = []
    for ticker in config.WATCHLIST:
        try:
            r = ml_logic.predict(ticker)
            _log_prediction(r)
            results.append(r)
        except Exception as exc:
            results.append({"ticker": ticker, "status": "error", "message": str(exc)})
    return {
        "predictions":  results,
        "count":        len(results),
        "timestamp":    datetime.utcnow().isoformat(),
        "scorecard":    _build_scorecard(),
    }


@app.post("/retrain")
async def retrain(body: RetrainRequest, background_tasks: BackgroundTasks) -> dict:
    tickers = body.tickers or config.WATCHLIST
    invalid = [t for t in tickers if t not in config.WATCHLIST]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown tickers: {invalid}")

    background_tasks.add_task(_retrain_background, tickers)
    return {
        "status":   "retraining_started",
        "tickers":  tickers,
        "message":  f"Retraining {len(tickers)} models in background.",
    }


@app.get("/backtest/{ticker}")
def backtest_ticker(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in config.WATCHLIST:
        raise HTTPException(status_code=404, detail=f"{ticker} not in watchlist")
    try:
        return ml_logic.backtest_yesterday(ticker)
    except Exception as exc:
        logger.exception("Backtest failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/backtest-all")
def backtest_all() -> dict:
    results = []
    passed  = 0
    for ticker in config.WATCHLIST:
        try:
            r = ml_logic.backtest_yesterday(ticker)
            results.append(r)
            if r.get("passed"):
                passed += 1
        except Exception as exc:
            results.append({"ticker": ticker, "status": "error", "message": str(exc)})
    return {
        "results":          results,
        "total":            len(results),
        "passed":           passed,
        "pass_rate_pct":    round(passed / max(len(results), 1) * 100, 1),
        "threshold_pct":    config.MAX_ERROR_THRESHOLD_PCT,
        "timestamp":        datetime.utcnow().isoformat(),
    }


@app.get("/scorecard")
def scorecard() -> dict:
    return _build_scorecard()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _log_prediction(result: dict) -> None:
    ticker = result.get("ticker", "")
    if ticker and result.get("status") == "ok":
        _session_stats.setdefault(ticker, []).append({
            "time":          datetime.utcnow().isoformat(),
            "change_pct":    result.get("change_pct"),
            "confidence":    result.get("confidence"),
            "signal":        result.get("signal"),
        })


def _build_scorecard() -> dict[str, Any]:
    total_preds = sum(len(v) for v in _session_stats.values())
    # Confidence-weighted average
    confidences = [
        p["confidence"]
        for preds in _session_stats.values()
        for p in preds
        if p.get("confidence") is not None
    ]
    avg_confidence = round(sum(confidences) / max(len(confidences), 1), 1)

    # "Success" = confidence > 60 (proxy for model certainty)
    successes      = sum(1 for c in confidences if c >= 60)
    success_rate   = round(successes / max(len(confidences), 1) * 100, 1)

    return {
        "session_start":    _session_start.isoformat(),
        "total_predictions":total_preds,
        "avg_confidence":   avg_confidence,
        "success_rate_pct": success_rate,
        "tickers_tracked":  len(_session_stats),
    }


def _retrain_background(tickers: list[str]) -> None:
    logger.info("Background retrain started for %d tickers", len(tickers))
    results = ml_logic.train_all(tickers)
    ok  = sum(1 for r in results if r.get("status") == "ok")
    logger.info("Background retrain complete: %d/%d succeeded", ok, len(tickers))
