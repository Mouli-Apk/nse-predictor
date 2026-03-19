"""
ml_logic.py — NSE Intraday Predictor v2
Improvements over v1:
  - Multi-horizon predictions: T+15min, T+1hr, T+3hr
  - Pre-session bid/ask data from Yahoo Finance
  - Enhanced feature set: ATR, OBV, VWAP, Stochastic RSI, ADX
  - LightGBM ensemble (with XGBoost fallback)
  - Feature importance pruning — keeps only top features
  - Early stopping in XGBoost to prevent overfitting
  - Price-based MAPE evaluation (not return-based)
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
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("ml_logic")

# Configure yfinance
try:
    yf.set_tz_cache_location("/tmp/yf_tz_cache")
except Exception:
    pass

# ── Model availability checks ──────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
    logger.info("LightGBM available — will use as primary model")
except ImportError:
    _LGBM_AVAILABLE = False
    logger.info("LightGBM not installed — using XGBoost")

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — predictions unavailable")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _VADER_AVAILABLE = False

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel

import config

# ── In-memory model registry ───────────────────────────────────────────────────
# Structure per ticker:
# {
#   "models":    {"t15": model, "t1h": model, "t3h": model},
#   "scaler":    RobustScaler,
#   "features":  list[str],       # pruned feature list
#   "interval":  "5m",
#   "horizons":  {"t15": 3, "t1h": 12, "t3h": 36},  # candle shifts
#   "val_mapes": {"t15": 1.2, "t1h": 1.8, "t3h": 2.4},
#   "trained_at": datetime,
#   "row_count":  int,
# }
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


# =============================================================================
# 1. DATA ACQUISITION
# =============================================================================

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex, keep OHLCV, strip timezone."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.dropna(inplace=True)
    return df


def fetch_ohlcv(ticker: str, days: int = config.TRAINING_PERIOD_DAYS) -> tuple[pd.DataFrame, str]:
    """
    Fetch OHLCV via Ticker.history() — resilient to Yahoo Finance geo-blocking.
    Returns (dataframe, interval_string).
    Priority: 1m → 5m → 1d
    """
    tkr = yf.Ticker(ticker)

    for interval, period in [("1m", "7d"), ("5m", "60d")]:
        try:
            df = tkr.history(period=period, interval=interval,
                             auto_adjust=True, raise_errors=False)
            if not df.empty and len(df) >= 50:
                logger.info("  fetched %s  interval=%s  rows=%d", ticker, interval, len(df))
                return _clean_df(df), interval
        except Exception as exc:
            logger.warning("  %s  interval=%s  error: %s", ticker, interval, exc)
        time.sleep(1.5)

    for period in ["6mo", "1y", "max"]:
        try:
            df = tkr.history(period=period, interval="1d",
                             auto_adjust=True, raise_errors=False)
            if not df.empty:
                logger.info("  fetched %s  interval=1d period=%s  rows=%d", ticker, period, len(df))
                return _clean_df(df), "1d"
        except Exception as exc:
            logger.warning("  %s  daily period=%s error: %s", ticker, period, exc)
        time.sleep(0.5)

    raise ValueError(f"All fetch strategies failed for {ticker}")


def fetch_recent_prices(ticker: str, bars: int = 120) -> list[float]:
    """Last `bars` closing prices for sparkline."""
    try:
        tkr = yf.Ticker(ticker)
        for interval, period in [("1m", "2d"), ("5m", "5d"), ("1d", "6mo")]:
            try:
                df = tkr.history(period=period, interval=interval,
                                 auto_adjust=True, raise_errors=False)
                if not df.empty:
                    return _clean_df(df)["Close"].tail(bars).tolist()
            except Exception:
                continue
    except Exception:
        pass
    return []


def fetch_pre_session(ticker: str) -> dict[str, Any]:
    """
    Fetch pre-session / live bid-ask data from Yahoo Finance.
    Returns bid, ask, pre-market price, volume, and 52-week range.
    """
    result: dict[str, Any] = {
        "bid":           None,
        "ask":           None,
        "pre_price":     None,
        "pre_change_pct":None,
        "volume":        None,
        "avg_volume":    None,
        "week52_high":   None,
        "week52_low":    None,
        "market_cap":    None,
        "pe_ratio":      None,
    }
    try:
        tkr  = yf.Ticker(ticker)
        info = tkr.fast_info

        result["bid"]         = getattr(info, "bid",          None)
        result["ask"]         = getattr(info, "ask",          None)
        result["volume"]      = getattr(info, "last_volume",  None)
        result["week52_high"] = getattr(info, "year_high",    None)
        result["week52_low"]  = getattr(info, "year_low",     None)
        result["market_cap"]  = getattr(info, "market_cap",   None)

        # Pre-market price from regular info (slower but more complete)
        try:
            full_info = tkr.info
            result["pre_price"]   = full_info.get("preMarketPrice")
            result["avg_volume"]  = full_info.get("averageVolume")
            result["pe_ratio"]    = full_info.get("trailingPE")

            last_close = full_info.get("previousClose") or full_info.get("regularMarketPreviousClose")
            if result["pre_price"] and last_close:
                result["pre_change_pct"] = round(
                    (result["pre_price"] - last_close) / last_close * 100, 2
                )
        except Exception:
            pass

    except Exception as exc:
        logger.warning("  pre_session fetch failed for %s: %s", ticker, exc)

    return result


# =============================================================================
# 2. ENHANCED FEATURE ENGINEERING
# =============================================================================

def _rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stoch_rsi(series: pd.Series, period: int = 14, smooth: int = 3) -> pd.Series:
    """Stochastic RSI — momentum within RSI range."""
    rsi  = _rsi(series, period)
    lo   = rsi.rolling(period).min()
    hi   = rsi.rolling(period).max()
    stoch = (rsi - lo) / (hi - lo + 1e-8)
    return stoch.rolling(smooth).mean()


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast   = series.ewm(span=config.MACD_FAST,   adjust=False).mean()
    slow   = series.ewm(span=config.MACD_SLOW,   adjust=False).mean()
    macd   = fast - slow
    signal = macd.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    return macd, signal, macd - signal


def _bollinger(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid  = series.rolling(config.BB_PERIOD).mean()
    std  = series.rolling(config.BB_PERIOD).std()
    return mid + config.BB_STD * std, mid, mid - config.BB_STD * std


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — measures volatility."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — volume momentum indicator."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _vwap_approx(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling VWAP approximation using typical price."""
    typical = (high + low + close) / 3
    return (typical * volume).rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength 0-100."""
    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr      = _atr(high, low, close, period)
    plus_di  = pd.Series(plus_dm,  index=close.index).rolling(period).mean() / atr * 100
    minus_di = pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr * 100
    dx       = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8) * 100
    return dx.rolling(period).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    out    = df.copy()

    # ── RSI & Stochastic RSI ──────────────────────────────────────────────────
    out["rsi_14"]    = _rsi(close, 14)
    out["rsi_7"]     = _rsi(close, 7)
    out["stoch_rsi"] = _stoch_rsi(close)

    # ── EMAs & ratios ─────────────────────────────────────────────────────────
    for span in config.EMA_PERIODS:
        out[f"ema_{span}"]       = close.ewm(span=span, adjust=False).mean()
        out[f"ema_ratio_{span}"] = close / out[f"ema_{span}"]

    # ── EMA crossovers (signal line crosses) ──────────────────────────────────
    out["ema_cross_9_21"]  = out["ema_9"]  - out["ema_21"]
    out["ema_cross_21_50"] = out["ema_21"] - out["ema_50"]

    # ── MACD ─────────────────────────────────────────────────────────────────
    out["macd"], out["macd_signal"], out["macd_hist"] = _macd(close)
    out["macd_cross"] = np.sign(out["macd_hist"])   # +1/-1 crossover signal

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_u, bb_m, bb_l = _bollinger(close)
    out["bb_upper"] = bb_u
    out["bb_mid"]   = bb_m
    out["bb_lower"] = bb_l
    out["bb_width"] = (bb_u - bb_l) / bb_m.replace(0, np.nan)
    out["bb_pct"]   = (close - bb_l) / (bb_u - bb_l + 1e-8)

    # ── ATR — volatility ──────────────────────────────────────────────────────
    out["atr_14"]      = _atr(high, low, close, 14)
    out["atr_ratio"]   = out["atr_14"] / close          # normalised ATR
    out["atr_7"]       = _atr(high, low, close, 7)
    out["atr_expand"]  = out["atr_7"] / out["atr_14"]   # expanding/contracting vol

    # ── OBV — volume momentum ─────────────────────────────────────────────────
    out["obv"]         = _obv(close, volume)
    out["obv_ema"]     = out["obv"].ewm(span=9, adjust=False).mean()
    out["obv_signal"]  = out["obv"] - out["obv_ema"]    # OBV divergence

    # ── VWAP ─────────────────────────────────────────────────────────────────
    out["vwap"]        = _vwap_approx(high, low, close, volume)
    out["vwap_ratio"]  = close / out["vwap"].replace(0, np.nan)

    # ── ADX — trend strength ──────────────────────────────────────────────────
    out["adx"]         = _adx(high, low, close, 14)
    out["adx_strong"]  = (out["adx"] > 25).astype(int)  # 1 if strong trend

    # ── Candle patterns ───────────────────────────────────────────────────────
    out["candle_body"]  = (close - df["Open"]).abs() / close
    out["upper_wick"]   = (high - close.clip(lower=df["Open"])) / close
    out["lower_wick"]   = (close.clip(upper=df["Open"]) - low) / close
    out["candle_dir"]   = np.sign(close - df["Open"])   # +1 bull / -1 bear
    out["doji"]         = (out["candle_body"] < 0.001).astype(int)

    # ── Volume features ───────────────────────────────────────────────────────
    out["vol_ema_9"]    = volume.ewm(span=9,  adjust=False).mean()
    out["vol_ema_20"]   = volume.ewm(span=20, adjust=False).mean()
    out["vol_ratio"]    = volume / out["vol_ema_9"].replace(0, np.nan)
    out["vol_surge"]    = (out["vol_ratio"] > 2).astype(int)  # volume spike flag

    # ── Rate-of-change / momentum ─────────────────────────────────────────────
    for lag in [1, 3, 5, 10, 15]:
        out[f"roc_{lag}"] = close.pct_change(lag)

    # ── Lagged returns ────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        out[f"lag_ret_{lag}"] = close.pct_change(lag).shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for window in [5, 15, 30]:
        out[f"roll_std_{window}"]  = close.rolling(window).std() / close
        out[f"roll_high_{window}"] = high.rolling(window).max() / close
        out[f"roll_low_{window}"]  = low.rolling(window).min() / close

    # ── Regime detection ─────────────────────────────────────────────────────
    # Is price above/below key EMAs? (trend regime flags)
    out["above_ema_9"]  = (close > out["ema_9"]).astype(int)
    out["above_ema_21"] = (close > out["ema_21"]).astype(int)
    out["above_ema_50"] = (close > out["ema_50"]).astype(int)

    # ── Time-of-day features (important for intraday patterns) ────────────────
    idx = out.index
    out["hour"]       = idx.hour
    out["minute"]     = idx.minute
    out["time_frac"]  = (idx.hour * 60 + idx.minute) / (6.5 * 60)
    # NSE session segments: opening (9:15-10:30), midday, closing (14:30-15:30)
    out["is_opening"] = ((idx.hour == 9) | ((idx.hour == 10) & (idx.minute <= 30))).astype(int)
    out["is_closing"] = ((idx.hour == 14) & (idx.minute >= 30) | (idx.hour == 15)).astype(int)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"Open", "High", "Low", "Close", "Volume", "target"}
    return [c for c in df.columns if c not in exclude]


# =============================================================================
# 3. MODEL BUILDING
# =============================================================================

def _build_model(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray,   y_val: np.ndarray):
    """
    Build and train one model. Uses LightGBM if available, else XGBoost.
    Both support early stopping to prevent overfitting.
    """
    if _LGBM_AVAILABLE:
        import lightgbm as lgb
        params = {**config.LGBM_PARAMS}
        params.pop("verbose", None)
        model = lgb.LGBMRegressor(**params, verbose=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        return model

    if _XGB_AVAILABLE:
        params = {k: v for k, v in config.XGB_PARAMS.items()
                  if k != "early_stopping_rounds"}
        model = XGBRegressor(**params, early_stopping_rounds=20)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    raise RuntimeError("Neither LightGBM nor XGBoost is available")


def _price_mape(actual_prices: np.ndarray, pred_returns: np.ndarray,
                true_returns: np.ndarray) -> float:
    """MAPE on prices, not returns — eliminates near-zero division explosion."""
    pred_prices = actual_prices * (1 + pred_returns)
    true_prices = actual_prices * (1 + true_returns)
    safe_true   = np.where(np.abs(true_prices) < 1e-8, 1e-8, true_prices)
    return float(np.mean(np.abs(pred_prices - true_prices) / safe_true) * 100)


# =============================================================================
# 4. SENTIMENT
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
# 5. TRAINING — one model per horizon per ticker
# =============================================================================

def train_model(ticker: str) -> dict[str, Any]:
    if not _XGB_AVAILABLE and not _LGBM_AVAILABLE:
        return {"ticker": ticker, "status": "error", "message": "No ML library available"}

    logger.info("Training model for %s ...", ticker)
    try:
        df, interval = fetch_ohlcv(ticker)
        df = add_features(df)

        # Cap rows to avoid OOM on 512MB free tier
        if len(df) > 2000:
            df = df.tail(2000).copy()

        # Get candle shifts for this interval
        horizon_candles = config.HORIZON_CANDLES.get(interval, config.HORIZON_CANDLES["1m"])

        # Check we have enough rows for the longest horizon
        max_horizon = max(horizon_candles.values())
        if len(df) < max_horizon + 50:
            return {"ticker": ticker, "status": "error",
                    "message": f"Insufficient data ({len(df)} rows)"}

        # Build feature matrix once (shared across all 3 horizon models)
        feature_cols = get_feature_columns(df)

        # Scaler fitted once on all features
        X_all    = df[feature_cols].values
        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X_all)

        # Replace any remaining NaN/Inf after scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Feature pruning via importance ────────────────────────────────────
        # Train a quick model on T+15 to get feature importances,
        # then keep only features with cumulative importance >= 95%
        h_candles_t15 = horizon_candles["t15"]
        y_quick = (df["Close"].shift(-h_candles_t15) / df["Close"] - 1).values
        valid   = ~np.isnan(y_quick)
        X_q, y_q = X_scaled[valid][:-10], y_quick[valid][:-10]
        X_qv, y_qv = X_scaled[valid][-10:], y_quick[valid][-10:]

        if _LGBM_AVAILABLE:
            import lightgbm as lgb
            quick = lgb.LGBMRegressor(n_estimators=50, max_depth=4,
                                      random_state=42, n_jobs=1, verbose=-1)
            quick.fit(X_q, y_q, eval_set=[(X_qv, y_qv)],
                      callbacks=[lgb.early_stopping(10, verbose=False),
                                 lgb.log_evaluation(period=-1)])
            importances = quick.feature_importances_
        else:
            quick = XGBRegressor(n_estimators=50, max_depth=4, random_state=42,
                                 n_jobs=1, tree_method="hist", early_stopping_rounds=10)
            quick.fit(X_q, y_q, eval_set=[(X_qv, y_qv)], verbose=False)
            importances = quick.feature_importances_

        # Sort by importance, keep features until we hit 95% cumulative
        order      = np.argsort(importances)[::-1]
        cumsum     = np.cumsum(importances[order]) / (importances.sum() + 1e-8)
        keep_idx   = order[:np.searchsorted(cumsum, 0.95) + 1]
        pruned_features = [feature_cols[i] for i in sorted(keep_idx)]
        X_pruned = X_scaled[:, sorted(keep_idx)]
        logger.info("  %s  feature pruning: %d → %d features",
                    ticker, len(feature_cols), len(pruned_features))

        # ── Train one model per horizon ────────────────────────────────────────
        tscv      = TimeSeriesSplit(n_splits=2)
        models    = {}
        val_mapes = {}

        for h_key, h_candles in horizon_candles.items():
            y_full = (df["Close"].shift(-h_candles) / df["Close"] - 1).values
            valid  = ~np.isnan(y_full)

            X_h = X_pruned[valid]
            y_h = y_full[valid]
            prices_h = df["Close"].values[valid]

            if len(X_h) < 60:
                logger.warning("  %s  %s: not enough data after shift", ticker, h_key)
                continue

            fold_mapes = []
            best_model = None

            for _, (tr_idx, va_idx) in enumerate(tscv.split(X_h)):
                m = _build_model(X_h[tr_idx], y_h[tr_idx],
                                 X_h[va_idx],  y_h[va_idx])
                preds = m.predict(X_h[va_idx])
                mape  = _price_mape(prices_h[va_idx], preds, y_h[va_idx])
                fold_mapes.append(mape)
                best_model = m

            models[h_key]    = best_model
            val_mapes[h_key] = round(float(np.mean(fold_mapes)), 3)
            logger.info("  %s  %-4s  val_mape=%.2f%%", ticker, h_key, val_mapes[h_key])

        if not models:
            return {"ticker": ticker, "status": "error", "message": "All horizons failed"}

        _MODEL_REGISTRY[ticker] = {
            "models":    models,
            "scaler":    scaler,
            "features":  pruned_features,
            "feature_idx": sorted(keep_idx),
            "interval":  interval,
            "horizons":  horizon_candles,
            "val_mapes": val_mapes,
            "trained_at":datetime.utcnow(),
            "row_count": len(df),
        }

        return {
            "ticker":     ticker,
            "status":     "ok",
            "interval":   interval,
            "val_mapes":  val_mapes,
            "features":   len(pruned_features),
            "row_count":  len(df),
            "trained_at": _MODEL_REGISTRY[ticker]["trained_at"].isoformat(),
        }

    except Exception as exc:
        import traceback
        logger.error("  FAIL %s: %s\n%s", ticker, exc, traceback.format_exc())
        return {"ticker": ticker, "status": "error", "message": str(exc)}


def train_all(tickers: list[str] | None = None) -> list[dict]:
    """Train all tickers with 1s polite delay between requests."""
    tickers = tickers or config.WATCHLIST
    results = []
    for i, ticker in enumerate(tickers):
        results.append(train_model(ticker))
        if i < len(tickers) - 1:
            time.sleep(1.0)
    return results


# =============================================================================
# 6. INFERENCE — multi-horizon predictions + pre-session
# =============================================================================

def _horizon_label(h_key: str, interval: str) -> str:
    """Human-readable prediction window label."""
    mins_map = {"1m": 1, "5m": 5, "1d": 1440}
    candles  = config.HORIZON_CANDLES.get(interval, config.HORIZON_CANDLES["1m"])
    mins     = candles.get(h_key, 15) * mins_map.get(interval, 1)
    if mins >= 60:
        return f"T+{mins//60}h"
    return f"T+{mins}min"


def _signal_for_change(change_pct: float, confidence: float) -> str:
    """BUY/SELL/HOLD based on predicted change and model confidence."""
    threshold = 0.3 if confidence >= 70 else 0.5
    if change_pct > threshold:
        return "BUY"
    if change_pct < -threshold:
        return "SELL"
    return "HOLD"


def predict(ticker: str) -> dict[str, Any]:
    """
    Return multi-horizon predictions + pre-session data for one ticker.
    Trains on-demand if model not in registry.
    """
    if ticker not in _MODEL_REGISTRY:
        result = train_model(ticker)
        if result["status"] != "ok":
            return {
                **result,
                "sector":       config.SECTOR_MAP.get(ticker, ""),
                "predictions":  {},
                "pre_session":  {},
                "current_price":None,
                "sparkline":    [],
            }

    reg      = _MODEL_REGISTRY[ticker]
    models   = reg["models"]
    scaler   = reg["scaler"]
    feats    = reg["features"]
    feat_idx = reg["feature_idx"]
    interval = reg["interval"]
    val_mapes= reg["val_mapes"]

    try:
        df, _ = fetch_ohlcv(ticker, days=3)
        df    = add_features(df)
        df.dropna(inplace=True)

        if df.empty:
            raise ValueError("No live data")

        current_price = float(df["Close"].iloc[-1])
        all_features  = get_feature_columns(df)

        # Build latest feature row using the same pruned feature set
        latest_all  = df[all_features].iloc[-1:].values
        latest_all  = np.nan_to_num(latest_all, nan=0.0, posinf=0.0, neginf=0.0)
        latest_scaled = scaler.transform(latest_all)
        latest_pruned = latest_scaled[:, feat_idx]

        # ── Per-horizon predictions ───────────────────────────────────────────
        horizon_preds: dict[str, Any] = {}
        for h_key, model in models.items():
            if model is None:
                continue
            pred_return   = float(model.predict(latest_pruned)[0])
            predicted_price = round(current_price * (1 + pred_return), 2)
            change_pct      = pred_return * 100

            mape       = val_mapes.get(h_key, 5.0)
            confidence = max(0, min(100, round(100 - mape * 8, 1)))
            signal     = _signal_for_change(change_pct, confidence)

            # Sentiment overlay on T+15 only (most actionable)
            if h_key == "t15":
                sentiment = get_sentiment_score(ticker)
                if sentiment < -0.3 and signal == "BUY":
                    signal     = "HOLD"
                    confidence = max(0, confidence - 10)
                elif sentiment > 0.3 and signal == "SELL":
                    signal     = "HOLD"
                    confidence = max(0, confidence - 10)
            else:
                sentiment = 0.0

            # Entry/exit times
            mins_map     = {"1m": 1, "5m": 5, "1d": 1440}
            candles      = config.HORIZON_CANDLES.get(interval, config.HORIZON_CANDLES["1m"])
            horizon_mins = candles.get(h_key, 15) * mins_map.get(interval, 1)
            now          = datetime.utcnow()
            exit_time    = (now + timedelta(minutes=horizon_mins)).strftime("%H:%M")

            horizon_preds[h_key] = {
                "label":           _horizon_label(h_key, interval),
                "predicted_price": predicted_price,
                "change_pct":      round(change_pct, 3),
                "signal":          signal,
                "confidence":      confidence,
                "val_mape":        mape,
                "exit_time":       exit_time,
                "horizon_mins":    horizon_mins,
                "sentiment":       round(sentiment, 3),
            }

        # ── Risk sizing (based on T+15 signal) ───────────────────────────────
        risk_qty = max(1, int(config.RISK_CAPITAL_INR / (current_price * config.RISK_PCT)))

        # ── Pre-session data ──────────────────────────────────────────────────
        pre_session = fetch_pre_session(ticker)

        return {
            "ticker":        ticker,
            "sector":        config.SECTOR_MAP.get(ticker, ""),
            "status":        "ok",
            "current_price": current_price,
            "predictions":   horizon_preds,
            "pre_session":   pre_session,
            "risk_qty":      risk_qty,
            "entry_time":    datetime.utcnow().strftime("%H:%M"),
            "sparkline":     fetch_recent_prices(ticker),
            "data_interval": interval,
            "trained_at":    reg["trained_at"].isoformat(),
            "val_mapes":     val_mapes,
        }

    except Exception as exc:
        logger.warning("Predict failed for %s: %s", ticker, exc)
        return {
            "ticker":        ticker,
            "sector":        config.SECTOR_MAP.get(ticker, ""),
            "status":        "error",
            "message":       str(exc),
            "current_price": None,
            "predictions":   {},
            "pre_session":   {},
            "sparkline":     [],
        }


# =============================================================================
# 7. BACKTESTER — price-based MAPE per horizon
# =============================================================================

def backtest_yesterday(ticker: str) -> dict[str, Any]:
    if ticker not in _MODEL_REGISTRY:
        train_model(ticker)

    reg      = _MODEL_REGISTRY[ticker]
    models   = reg["models"]
    scaler   = reg["scaler"]
    feats    = reg["features"]
    feat_idx = reg["feature_idx"]
    interval = reg["interval"]
    horizons = reg["horizons"]

    try:
        df_full, _ = fetch_ohlcv(ticker, days=config.TRAINING_PERIOD_DAYS + 1)
        df_full    = add_features(df_full)
        df_full.dropna(inplace=True)

        # Find last trading day (walk back up to 7 days)
        today = datetime.utcnow().date()
        df_yday   = pd.DataFrame()
        backtest_date = today
        for days_back in range(1, 8):
            test_date = today - timedelta(days=days_back)
            mask = df_full.index.date == test_date
            if mask.sum() >= 2:
                df_yday = df_full[mask]
                backtest_date = test_date
                break

        if df_yday.empty:
            backtest_date = df_full.index.date[-1]
            df_yday = df_full[df_full.index.date == backtest_date]

        if df_yday.empty:
            return {"ticker": ticker, "status": "error", "message": "No recent trading data"}

        all_features = get_feature_columns(df_full)
        X_yday       = df_yday[all_features].values
        X_yday       = np.nan_to_num(X_yday, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled     = scaler.transform(X_yday)
        X_pruned     = X_scaled[:, feat_idx]

        results_per_horizon: dict[str, Any] = {}
        overall_mapes: list[float] = []

        for h_key, h_candles in horizons.items():
            model = models.get(h_key)
            if model is None:
                continue

            # Target: actual future price
            y_true_ret = (df_yday["Close"].shift(-h_candles) / df_yday["Close"] - 1).values
            valid = ~np.isnan(y_true_ret)
            if valid.sum() < 2:
                continue

            sample_idx = np.where(valid)[0][::max(1, h_candles)]
            X_s        = X_pruned[sample_idx]
            y_true_s   = y_true_ret[sample_idx]
            prices_s   = df_yday["Close"].values[sample_idx]
            y_pred_s   = model.predict(X_s)

            pred_prices = prices_s * (1 + y_pred_s)
            true_prices = prices_s * (1 + y_true_s)
            safe_true   = np.where(np.abs(true_prices) < 1e-8, 1e-8, true_prices)
            errors      = np.abs(pred_prices - true_prices) / safe_true * 100
            mape        = float(np.mean(errors))

            mins_map     = {"1m": 1, "5m": 5, "1d": 1440}
            horizon_mins = h_candles * mins_map.get(interval, 1)

            steps = [
                {
                    "time":            str(df_yday.index[sample_idx[i]]),
                    "actual_price":    round(float(prices_s[i]), 2),
                    "predicted_price": round(float(pred_prices[i]), 2),
                    "true_future":     round(float(true_prices[i]), 2),
                    "price_error_pct": round(float(errors[i]), 3),
                }
                for i in range(len(sample_idx))
            ]

            passed = mape <= config.MAX_ERROR_THRESHOLD_PCT
            results_per_horizon[h_key] = {
                "label":       _horizon_label(h_key, interval),
                "horizon_mins":horizon_mins,
                "mape":        round(mape, 3),
                "passed":      passed,
                "steps":       steps,
            }
            overall_mapes.append(mape)

        overall_mape   = round(float(np.mean(overall_mapes)), 3) if overall_mapes else 999.0
        overall_passed = overall_mape <= config.MAX_ERROR_THRESHOLD_PCT

        return {
            "ticker":           ticker,
            "status":           "ok",
            "date":             str(backtest_date),
            "data_interval":    interval,
            "overall_mape":     overall_mape,
            "passed":           overall_passed,
            "threshold":        config.MAX_ERROR_THRESHOLD_PCT,
            "horizons":         results_per_horizon,
            "mape":             overall_mape,   # backward compat for frontend
            "summary": (
                f"{'PASSED' if overall_passed else 'FAILED'} — "
                f"avg MAPE {overall_mape:.2f}% across all horizons"
            ),
        }

    except Exception as exc:
        import traceback
        logger.error("Backtest failed for %s: %s\n%s", ticker, exc, traceback.format_exc())
        return {"ticker": ticker, "status": "error", "message": str(exc)}
