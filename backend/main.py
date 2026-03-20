"""
main.py — NSE Intraday Predictor · FastAPI Backend

Key design decisions:
- Port binds IMMEDIATELY on startup (Render requires this within ~60s)
- Model training runs in a background THREAD (not asyncio task) so it
  never blocks the event loop or delays port binding
- Every endpoint is wrapped in try/except so one bad ticker never crashes
  the whole server
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import ml_logic

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("main")

_session_stats: dict[str, list[dict]] = {}
_session_start: datetime = datetime.utcnow()
_training_complete: bool = False


def _train_in_background() -> None:
    """Runs in a daemon thread so uvicorn can bind the port immediately."""
    global _training_complete
    logger.info("Background training started for %d tickers", len(config.WATCHLIST))
    try:
        results = ml_logic.train_all(config.WATCHLIST)
        ok  = sum(1 for r in results if r.get("status") == "ok")
        err = len(results) - ok
        logger.info("Training complete: %d/%d succeeded, %d skipped.", ok, len(results), err)
    except Exception as exc:
        logger.error("Training thread error: %s", exc)
    finally:
        _training_complete = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NSE Predictor starting - binding port first, training in background.")
    t = threading.Thread(target=_train_in_background, daemon=True)
    t.start()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="NSE Intraday Predictor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrainRequest(BaseModel):
    tickers: list[str] | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status":            "ok",
        "timestamp":         datetime.utcnow().isoformat(),
        "models_loaded":     len(ml_logic._MODEL_REGISTRY),
        "watchlist_size":    len(config.WATCHLIST),
        "training_complete": _training_complete,
        "session_start":     _session_start.isoformat(),
    }


@app.get("/watchlist")
def get_watchlist() -> dict:
    return {
        "tickers": config.WATCHLIST,
        "sectors": config.SECTOR_MAP,
        "count":   len(config.WATCHLIST),
    }


@app.get("/predict/{ticker}")
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
            results.append({
                "ticker":     ticker,
                "sector":     config.SECTOR_MAP.get(ticker, ""),
                "status":     "error",
                "message":    str(exc),
                "signal":     "HOLD",
                "confidence": 0,
                "sparkline":  [],
            })
    return {
        "predictions": results,
        "count":       len(results),
        "timestamp":   datetime.utcnow().isoformat(),
        "scorecard":   _build_scorecard(),
    }


@app.post("/retrain")
def retrain(body: RetrainRequest, background_tasks: BackgroundTasks) -> dict:
    tickers = body.tickers or config.WATCHLIST
    invalid = [t for t in tickers if t not in config.WATCHLIST]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown tickers: {invalid}")
    background_tasks.add_task(_retrain_background, tickers)
    return {"status": "retraining_started", "tickers": tickers}


@app.get("/backtest/{ticker}")
def backtest_ticker(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in config.WATCHLIST:
        raise HTTPException(status_code=404, detail=f"{ticker} not in watchlist")
    try:
        return ml_logic.backtest_yesterday(ticker)
    except Exception as exc:
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
        "results":       results,
        "total":         len(results),
        "passed":        passed,
        "pass_rate_pct": round(passed / max(len(results), 1) * 100, 1),
        "threshold_pct": config.MAX_ERROR_THRESHOLD_PCT,
        "timestamp":     datetime.utcnow().isoformat(),
    }


@app.get("/scorecard")
def scorecard() -> dict:
    return _build_scorecard()


@app.get("/after-market")
def after_market(top_n: int = 7) -> dict:
    """
    Post-close analysis — top stocks for tomorrow's pre-open session.
    Best called after 3:30 PM IST or before 9:15 AM IST.
    """
    return ml_logic.after_market_analysis(top_n=top_n)


@app.get("/cache-status")
def cache_status() -> dict:
    """How many predictions are cached and their ages."""
    now = datetime.utcnow()
    entries = []
    for ticker, c in ml_logic._PREDICT_CACHE.items():
        age = (now - c["cached_at"]).total_seconds()
        entries.append({"ticker": ticker, "age_seconds": round(age, 1)})
    return {
        "cached_predictions": len(entries),
        "ttl_seconds":        ml_logic._CACHE_TTL_SECONDS,
        "entries":            entries,
    }


@app.delete("/cache")
def clear_cache() -> dict:
    """Clear all prediction caches — forces fresh OHLCV fetch on next predict."""
    count = len(ml_logic._PREDICT_CACHE)
    ml_logic._PREDICT_CACHE.clear()
    for ticker in ml_logic._MODEL_REGISTRY:
        ml_logic._MODEL_REGISTRY[ticker].pop("last_fetched", None)
    return {"cleared": count, "message": "Cache cleared — next predict-all will re-fetch live data"}


@app.get("/market-status")
def market_status() -> dict:
    """Returns whether NSE market is currently open/closed in IST."""
    from datetime import timezone
    now_utc = datetime.utcnow()
    ist_offset = timedelta(hours=5, minutes=30)
    now_ist = now_utc + ist_offset
    h, m = now_ist.hour, now_ist.minute
    is_open = ((h > 9) or (h == 9 and m >= 15)) and ((h < 15) or (h == 15 and m <= 30))
    return {
        "is_open":    is_open,
        "ist_time":   now_ist.strftime("%H:%M IST"),
        "ist_date":   now_ist.strftime("%d %b %Y"),
        "opens_at":   "09:15 IST",
        "closes_at":  "15:30 IST",
        "after_close": not is_open and h >= 15,
        "pre_open":    not is_open and h < 9,
    }


def _log_prediction(result: dict) -> None:
    ticker = result.get("ticker", "")
    if ticker and result.get("status") == "ok":
        # New multi-horizon format: read from predictions.t15 (primary horizon)
        preds  = result.get("predictions", {})
        t15    = preds.get("t15", {})
        # Fallback to top-level for backward compatibility
        conf   = t15.get("confidence") or result.get("confidence")
        signal = t15.get("signal")     or result.get("signal", "HOLD")
        chg    = t15.get("change_pct") or result.get("change_pct")
        _session_stats.setdefault(ticker, []).append({
            "time":       datetime.utcnow().isoformat(),
            "change_pct": chg,
            "confidence": conf,
            "signal":     signal,
        })


def _build_scorecard() -> dict[str, Any]:
    total_preds = sum(len(v) for v in _session_stats.values())
    confidences = [
        p["confidence"]
        for preds in _session_stats.values()
        for p in preds
        if p.get("confidence") is not None
    ]
    avg_confidence = round(sum(confidences) / max(len(confidences), 1), 1)
    successes      = sum(1 for c in confidences if c >= 60)
    success_rate   = round(successes / max(len(confidences), 1) * 100, 1)
    return {
        "session_start":     _session_start.isoformat(),
        "total_predictions": total_preds,
        "avg_confidence":    avg_confidence,
        "success_rate_pct":  success_rate,
        "tickers_tracked":   len(_session_stats),
        "training_complete": _training_complete,
        "models_loaded":     len(ml_logic._MODEL_REGISTRY),
    }


def _retrain_background(tickers: list[str]) -> None:
    logger.info("Background retrain started for %d tickers", len(tickers))
    try:
        # Clear prediction cache so fresh data is fetched after retrain
        for t in tickers:
            ml_logic._PREDICT_CACHE.pop(t, None)
        results = ml_logic.train_all(tickers)
        ok = sum(1 for r in results if r.get("status") == "ok")
        logger.info("Background retrain complete: %d/%d succeeded", ok, len(tickers))
    except Exception as exc:
        logger.error("Background retrain failed: %s", exc)
