"""
ml_logic.py — NSE Intraday Predictor
XGBoost training pipeline with technical indicators + VADER sentiment.
Uses Ticker.history() which is resilient to Yahoo Finance geo-blocking.
"""

from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("ml_logic")

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed — predictions will be stubs.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed — sentiment fixed at 0.")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

import config

# In-memory model registry
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


# =============================================================================
# 1. DATA ACQUISITION
# =============================================================================

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, keep OHLCV, strip timezone."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.dropna(inplace=True)
    return df


def fetch_ohlcv(ticker: str, days: int = config.TRAINING_PERIOD_DAYS) -> pd.DataFrame:
    """
    Fetch OHLCV using Ticker.history() with period= parameter.
    More resilient to Yahoo Finance geo-blocking than yf.download().

    Priority: 1m (7d) -> 5m (60d) -> 1d (6mo, always works from any IP)
    """
    tkr = yf.Ticker(ticker)

    for interval, period in [("1m", "7d"), ("5m", "60d")]:
        try:
            df = tkr.history(period=period, interval=interval, auto_adjust=True)
            if not df.empty and len(df) >= 50:
                logger.info("  fetched %s  interval=%s  rows=%d", ticker, interval, len(df))
                return _clean_df(df)
        except Exception as exc:
            logger.warning("  %s  interval=%s  error: %s", ticker, interval, exc)
        time.sleep(1.0)

    # Daily fallback — works from any server IP worldwide
    try:
        df = tkr.history(period="6mo", interval="1d", auto_adjust=True)
        if not df.empty:
            logger.info("  fetched %s  interval=1d (fallback)  rows=%d", ticker, len(df))
            return _clean_df(df)
    except Exception as exc:
        logger.warning("  %s  daily fallback error: %s", ticker, exc)

    raise ValueError(f"All fetch strategies failed for {ticker}")


def fetch_recent_prices(ticker: str, bars: int = 120) -> list[float]:
    """Last bars closing prices for sparkline display."""
    try:
        tkr = yf.Ticker(ticker)
        for interval, period in [("1m", "2d"), ("5m", "5d"), ("1d", "6mo")]:
            try:
                df = tkr.history(period=period, interval=interval, auto_adjust=True)
                if not df.empty:
                    return _clean_df(df)["Close"].tail(bars).tolist()
            except Exception:
                continue
    except Exception:
        pass
    return []


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def _rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast   = series.ewm(span=config.MACD_FAST,   adjust=False).mean()
    slow   = series.ewm(span=config.MACD_SLOW,   adjust=False).mean()
    macd   = fast - slow
    signal = macd.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    return macd, signal, macd - signal


def _bollinger(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = series.rolling(config.BB_PERIOD).mean()
    std   = series.rolling(config.BB_PERIOD).std()
    return mid + config.BB_STD * std, mid, mid - config.BB_STD * std


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]
    out   = df.copy()

    out["rsi_14"] = _rsi(close)

    for span in config.EMA_PERIODS:
        out[f"ema_{span}"]       = close.ewm(span=span, adjust=False).mean()
        out[f"ema_ratio_{span}"] = close / out[f"ema_{span}"]

    out["macd"], out["macd_signal"], out["macd_hist"] = _macd(close)

    bb_u, bb_m, bb_l = _bollinger(close)
    out["bb_upper"] = bb_u
    out["bb_mid"]   = bb_m
    out["bb_lower"] = bb_l
    out["bb_width"] = (bb_u - bb_l) / bb_m.replace(0, np.nan)
    out["bb_pct"]   = (close - bb_l) / (bb_u - bb_l).replace(0, np.nan)

    out["candle_body"] = (close - df["Open"]).abs() / close
    out["upper_wick"]  = (high - close.clip(lower=df["Open"])) / close
    out["lower_wick"]  = (close.clip(upper=df["Open"]) - low) / close

    out["vol_ema_9"]  = vol.ewm(span=9, adjust=False).mean()
    out["vol_ratio"]  = vol / out["vol_ema_9"].replace(0, np.nan)

    for lag in [1, 5, 15]:
        out[f"roc_{lag}"] = close.pct_change(lag)

    for lag in [1, 3, 5, 10]:
        out[f"lag_ret_{lag}"] = close.pct_change(lag).shift(lag)

    out["rolling_std_15"]  = close.rolling(15).std()
    out["rolling_high_15"] = high.rolling(15).max()
    out["rolling_low_15"]  = low.rolling(15).min()
    out["range_pct_15"]    = (out["rolling_high_15"] - out["rolling_low_15"]) / close

    idx = out.index
    out["hour"]      = idx.hour
    out["minute"]    = idx.minute
    out["time_frac"] = (idx.hour * 60 + idx.minute) / (6.5 * 60)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"Open", "High", "Low", "Close", "Volume", "target"}
    return [c for c in df.columns if c not in exclude]


# =============================================================================
# 3. SENTIMENT
# =============================================================================

def get_sentiment_score(ticker: str) -> float:
    if not _VADER_AVAILABLE:
        return 0.0
    try:
        news   = yf.Ticker(ticker).news or []
        scores = [
            _vader.polarity_scores(a.get("title", ""))["compound"]
            for a in news[:config.NEWS_MAX_ARTICLES]
            if a.get("title")
        ]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_model(ticker: str) -> dict[str, Any]:
    if not _XGB_AVAILABLE:
        return {"ticker": ticker, "status": "error", "message": "xgboost not installed"}

    logger.info("Training model for %s ...", ticker)
    try:
        df = fetch_ohlcv(ticker)
        df = add_features(df)

        horizon = config.PREDICTION_HORIZON_MINS
        df["target"] = df["Close"].shift(-horizon) / df["Close"] - 1
        df.dropna(inplace=True)

        if len(df) < 100:
            logger.warning("  %s  insufficient data (%d rows)", ticker, len(df))
            return {"ticker": ticker, "status": "error", "message": "Insufficient data"}

        feature_cols = get_feature_columns(df)
        X = df[feature_cols].values
        y = df["target"].values

        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        tscv  = TimeSeriesSplit(n_splits=4)
        mapes = []
        model = None

        for _, (tr_idx, va_idx) in enumerate(tscv.split(X_scaled)):
            m = XGBRegressor(**config.XGB_PARAMS)
            m.fit(
                X_scaled[tr_idx], y[tr_idx],
                eval_set=[(X_scaled[va_idx], y[va_idx])],
                verbose=False,
            )
            preds = m.predict(X_scaled[va_idx])
            mape  = mean_absolute_percentage_error(y[va_idx], preds) * 100
            mapes.append(mape)
            model = m

        val_mape = float(np.mean(mapes))
        _MODEL_REGISTRY[ticker] = {
            "model":      model,
            "scaler":     scaler,
            "features":   feature_cols,
            "trained_at": datetime.utcnow(),
            "val_mape":   val_mape,
            "row_count":  len(df),
        }

        logger.info("  OK %s  val_mape=%.2f%%  rows=%d", ticker, val_mape, len(df))
        return {
            "ticker":     ticker,
            "status":     "ok",
            "val_mape":   round(val_mape, 3),
            "row_count":  len(df),
            "trained_at": _MODEL_REGISTRY[ticker]["trained_at"].isoformat(),
        }

    except Exception as exc:
        logger.warning("  SKIP %s  error: %s", ticker, exc)
        return {"ticker": ticker, "status": "error", "message": str(exc)}


def train_all(tickers: list[str] | None = None) -> list[dict]:
    """Retrain all tickers with a 1s delay between requests."""
    tickers = tickers or config.WATCHLIST
    results = []
    for i, ticker in enumerate(tickers):
        results.append(train_model(ticker))
        if i < len(tickers) - 1:
            time.sleep(1.0)
    return results


# =============================================================================
# 5. INFERENCE
# =============================================================================

def predict(ticker: str) -> dict[str, Any]:
    if ticker not in _MODEL_REGISTRY:
        result = train_model(ticker)
        if result["status"] != "ok":
            return {
                **result,
                "sector":          config.SECTOR_MAP.get(ticker, ""),
                "predicted_price": None,
                "current_price":   None,
                "signal":          "HOLD",
                "confidence":      0,
                "sparkline":       [],
            }

    reg    = _MODEL_REGISTRY[ticker]
    model  = reg["model"]
    scaler = reg["scaler"]
    feats  = reg["features"]

    try:
        df = fetch_ohlcv(ticker, days=3)
        df = add_features(df)
        df.dropna(inplace=True)

        if df.empty:
            raise ValueError("No live data")

        latest_row    = df[feats].iloc[-1:].values
        current_price = float(df["Close"].iloc[-1])
        X_scaled      = scaler.transform(latest_row)
        pred_return   = float(model.predict(X_scaled)[0])
        predicted_price = round(current_price * (1 + pred_return), 2)
        change_pct      = pred_return * 100

        val_mape   = reg.get("val_mape", 5.0)
        confidence = max(0, min(100, round(100 - val_mape * 5, 1)))

        if change_pct > 0.4:
            signal = "BUY"
        elif change_pct < -0.4:
            signal = "SELL"
        else:
            signal = "HOLD"

        sentiment = get_sentiment_score(ticker)
        if sentiment < -0.3 and signal == "BUY":
            signal     = "HOLD"
            confidence = max(0, confidence - 10)
        elif sentiment > 0.3 and signal == "SELL":
            signal     = "HOLD"
            confidence = max(0, confidence - 10)

        risk_qty = max(1, int(config.RISK_CAPITAL_INR / (current_price * config.RISK_PCT)))

        now        = datetime.utcnow()
        entry_time = now.strftime("%H:%M")
        exit_time  = (now + timedelta(minutes=config.PREDICTION_HORIZON_MINS)).strftime("%H:%M")
        sparkline  = fetch_recent_prices(ticker)

        return {
            "ticker":          ticker,
            "sector":          config.SECTOR_MAP.get(ticker, ""),
            "status":          "ok",
            "current_price":   current_price,
            "predicted_price": predicted_price,
            "change_pct":      round(change_pct, 3),
            "signal":          signal,
            "confidence":      confidence,
            "val_mape":        round(val_mape, 2),
            "sentiment":       round(sentiment, 3),
            "risk_qty":        risk_qty,
            "entry_time":      entry_time,
            "exit_time":       exit_time,
            "sparkline":       sparkline,
            "trained_at":      reg["trained_at"].isoformat(),
        }

    except Exception as exc:
        logger.warning("Predict failed for %s: %s", ticker, exc)
        return {
            "ticker":          ticker,
            "sector":          config.SECTOR_MAP.get(ticker, ""),
            "status":          "error",
            "message":         str(exc),
            "current_price":   None,
            "predicted_price": None,
            "signal":          "HOLD",
            "confidence":      0,
            "sparkline":       [],
        }


# =============================================================================
# 6. BACKTESTER
# =============================================================================

def backtest_yesterday(ticker: str) -> dict[str, Any]:
    if ticker not in _MODEL_REGISTRY:
        train_model(ticker)

    reg    = _MODEL_REGISTRY[ticker]
    model  = reg["model"]
    scaler = reg["scaler"]
    feats  = reg["features"]

    try:
        df_full = fetch_ohlcv(ticker, days=config.TRAINING_PERIOD_DAYS + 1)
        df_full = add_features(df_full)
        df_full["target"] = df_full["Close"].shift(-config.PREDICTION_HORIZON_MINS) / df_full["Close"] - 1
        df_full.dropna(inplace=True)

        today     = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        mask      = df_full.index.date == yesterday
        df_yday   = df_full[mask]

        if len(df_yday) < 5:
            last_date = df_full.index.date[-1]
            df_yday   = df_full[df_full.index.date == last_date]

        if df_yday.empty:
            return {"ticker": ticker, "status": "error", "message": "No yesterday data"}

        sample = df_yday.iloc[::max(1, config.PREDICTION_HORIZON_MINS)]
        X      = scaler.transform(sample[feats].values)
        y_true = sample["target"].values
        y_pred = model.predict(X)

        actual_prices    = sample["Close"].values
        predicted_prices = actual_prices * (1 + y_pred)
        true_prices      = actual_prices * (1 + y_true)
        errors           = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8) * 100
        mape             = float(np.mean(errors))
        passed           = mape <= config.MAX_ERROR_THRESHOLD_PCT

        steps = [
            {
                "time":            str(ts),
                "actual_price":    round(float(ap), 2),
                "predicted_price": round(float(pp), 2),
                "true_future":     round(float(tp), 2),
                "error_pct":       round(float(err), 3),
            }
            for ts, ap, pp, tp, err in zip(
                sample.index, actual_prices, predicted_prices, true_prices, errors
            )
        ]

        return {
            "ticker":    ticker,
            "status":    "ok",
            "date":      str(yesterday),
            "mape":      round(mape, 3),
            "passed":    passed,
            "threshold": config.MAX_ERROR_THRESHOLD_PCT,
            "steps":     steps,
            "summary":   f"{'PASSED' if passed else 'FAILED'} - MAPE {mape:.2f}%",
        }

    except Exception as exc:
        return {"ticker": ticker, "status": "error", "message": str(exc)}
