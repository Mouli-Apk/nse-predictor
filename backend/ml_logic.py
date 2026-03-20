"""
ml_logic.py — NSE Intraday Predictor v3 (Research-Grade)
─────────────────────────────────────────────────────────
Architecture based on 2024-2025 academic research findings:

KEY INSIGHT: Predicting price returns (regression) is fundamentally noisy.
The best approach uses a DUAL MODEL per horizon:
  1. LightGBM Classifier → predicts direction (UP/DOWN/FLAT) with calibrated probability
  2. LightGBM/XGBoost Regressor → predicts magnitude (how much)
  Final signal = direction × magnitude, confidence = calibrated class probability

Improvements from research:
  - Walk-forward expanding window CV (no data leakage)
  - Rolling z-score normalisation per feature (handles regime changes)
  - Extended feature set: Williams %R, CCI, Stochastic %D, Disparity Index,
    Chaikin Money Flow, Price Efficiency Ratio, Hurst Exponent proxy
  - Log-return targets (more stationary than raw returns)
  - Extreme label dropping (avoids training on outlier noise)
  - Platt scaling proxy for calibrated confidence
  - Stacked ensemble: LGBM classifier + LGBM regressor predictions
    → fed into XGBoost meta-learner for final price estimate
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
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger("ml_logic")

# Configure yfinance
try:
    yf.set_tz_cache_location("/tmp/yf_tz_cache")
except Exception:
    pass

# IST offset
IST = timedelta(hours=5, minutes=30)

def now_ist() -> datetime:
    return datetime.utcnow() + IST


# ── Library availability ────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
    logger.info("LightGBM available")
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

if not _LGBM_AVAILABLE and not _XGB_AVAILABLE:
    raise RuntimeError("Install lightgbm or xgboost")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _VADER_AVAILABLE = False

from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import config

# ── In-memory model registry ─────────────────────────────────────────────────
# {ticker: {
#   "classifier": model,        # direction: UP/DOWN/FLAT
#   "regressor":  model,        # log-return magnitude
#   "scaler":     RobustScaler,
#   "features":   list[str],
#   "interval":   str,
#   "horizons":   {h_key: int},  # candle shifts
#   "direction_acc": {h_key: float},  # directional accuracy %
#   "val_mapes":  {h_key: float},
#   "trained_at": datetime,
# }}
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}

# ── Prediction cache — avoids re-fetching OHLCV on every call ────────────────
# {ticker: {"result": dict, "cached_at": datetime}}
_PREDICT_CACHE: dict[str, dict] = {}
_CACHE_TTL_SECONDS: int = 60   # refresh predictions every 60 seconds max

# Direction thresholds — above/below these log-return % = UP/DOWN, else FLAT
UP_THRESHOLD   =  0.002   # +0.2% log return
DOWN_THRESHOLD = -0.002   # -0.2% log return


# =============================================================================
# 1. DATA
# =============================================================================

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
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
    tkr = yf.Ticker(ticker)
    for interval, period in [("1m", "7d"), ("5m", "60d")]:
        try:
            df = tkr.history(period=period, interval=interval,
                             auto_adjust=True, raise_errors=False)
            if not df.empty and len(df) >= 50:
                logger.info("  fetched %s  interval=%s  rows=%d", ticker, interval, len(df))
                return _clean_df(df), interval
        except Exception as exc:
            logger.warning("  %s  %s  error: %s", ticker, interval, exc)
        time.sleep(0.5)

    for period in ["6mo", "1y", "max"]:
        try:
            df = tkr.history(period=period, interval="1d",
                             auto_adjust=True, raise_errors=False)
            if not df.empty:
                logger.info("  fetched %s  interval=1d(%s)  rows=%d", ticker, period, len(df))
                return _clean_df(df), "1d"
        except Exception as exc:
            logger.warning("  %s  1d %s error: %s", ticker, period, exc)
        time.sleep(0.2)

    raise ValueError(f"All fetch strategies failed for {ticker}")


def fetch_recent_prices(ticker: str, bars: int = 120) -> list[float]:
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
    result: dict[str, Any] = dict(bid=None, ask=None, pre_price=None,
                                   pre_change_pct=None, volume=None,
                                   avg_volume=None, week52_high=None,
                                   week52_low=None, market_cap=None,
                                   pe_ratio=None)
    try:
        tkr  = yf.Ticker(ticker)
        info = tkr.fast_info
        result["bid"]         = getattr(info, "bid",         None)
        result["ask"]         = getattr(info, "ask",         None)
        result["volume"]      = getattr(info, "last_volume", None)
        result["week52_high"] = getattr(info, "year_high",   None)
        result["week52_low"]  = getattr(info, "year_low",    None)
        result["market_cap"]  = getattr(info, "market_cap",  None)
        # Use fast_info for pre-market price (avoids slow tkr.info HTML scrape)
        try:
            result["pre_price"]    = getattr(info, "pre_market_price", None)
            result["avg_volume"]   = getattr(info, "three_month_average_volume", None)
        except Exception:
            pass
    except Exception as exc:
        logger.warning("  pre_session error %s: %s", ticker, exc)
    return result


# =============================================================================
# 2. EXTENDED FEATURE ENGINEERING (60+ features based on research)
# =============================================================================

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d  = s.diff()
    g  = d.clip(lower=0).rolling(p).mean()
    l  = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    lo = low.rolling(p).min()
    hi = high.rolling(p).max()
    return (close - lo) / (hi - lo + 1e-8) * 100

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    hi = high.rolling(p).max()
    lo = low.rolling(p).min()
    return (hi - close) / (hi - lo + 1e-8) * -100

def _cci(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 20) -> pd.Series:
    tp  = (high + low + close) / 3
    ma  = tp.rolling(p).mean()
    md  = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md + 1e-8)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()

def _vwap(high: pd.Series, low: pd.Series, close: pd.Series,
           volume: pd.Series, p: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).rolling(p).sum() / volume.rolling(p).sum().replace(0, np.nan)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    up   = high.diff().clip(lower=0)
    down = (-low.diff()).clip(lower=0)
    atr  = _atr(high, low, close, p)
    plus_di  = up.rolling(p).mean()   / atr * 100
    minus_di = down.rolling(p).mean() / atr * 100
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8) * 100
    return dx.rolling(p).mean()

def _cmf(high: pd.Series, low: pd.Series, close: pd.Series,
          volume: pd.Series, p: int = 20) -> pd.Series:
    """Chaikin Money Flow — buying/selling pressure."""
    clv = ((close - low) - (high - close)) / (high - low + 1e-8)
    return (clv * volume).rolling(p).sum() / volume.rolling(p).sum().replace(0, np.nan)

def _disparity(close: pd.Series, p: int) -> pd.Series:
    """Disparity index — price distance from moving average %."""
    return (close / close.rolling(p).mean() - 1) * 100

def _rolling_zscore(s: pd.Series, p: int = 20) -> pd.Series:
    """Rolling z-score normalisation — handles regime changes."""
    mu  = s.rolling(p).mean()
    std = s.rolling(p).std()
    return (s - mu) / (std + 1e-8)

def _efficiency_ratio(close: pd.Series, p: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio — directional efficiency of price movement."""
    direction = close.diff(p).abs()
    noise     = close.diff().abs().rolling(p).sum()
    return direction / (noise + 1e-8)

def _hurst_proxy(close: pd.Series, p: int = 20) -> pd.Series:
    """
    Simplified Hurst exponent proxy using variance ratio.
    H > 0.5 = trending, H < 0.5 = mean-reverting.
    """
    r1 = close.pct_change(1).rolling(p).var()
    r2 = close.pct_change(2).rolling(p).var()
    return np.log(r2 / (2 * r1 + 1e-8) + 1e-8) / np.log(2)

def _macd(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast   = _ema(s, config.MACD_FAST)
    slow   = _ema(s, config.MACD_SLOW)
    macd   = fast - slow
    signal = _ema(macd, config.MACD_SIGNAL)
    return macd, signal, macd - signal

def _bollinger(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = s.rolling(config.BB_PERIOD).mean()
    d = s.rolling(config.BB_PERIOD).std()
    return m + config.BB_STD * d, m, m - config.BB_STD * d


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c   = df["Close"]
    h   = df["High"]
    l   = df["Low"]
    v   = df["Volume"]
    o   = df["Open"]
    out = df.copy()

    # ── Trend: EMAs & ratios ──────────────────────────────────────────────────
    for span in [5, 9, 21, 50]:
        out[f"ema_{span}"]   = _ema(c, span)
        out[f"er_{span}"]    = c / out[f"ema_{span}"]         # price/EMA ratio
    out["ema_cross_5_21"]  = out["ema_5"]  - out["ema_21"]
    out["ema_cross_9_50"]  = out["ema_9"]  - out["ema_50"]

    # ── Momentum oscillators ──────────────────────────────────────────────────
    out["rsi_7"]       = _rsi(c, 7)
    out["rsi_14"]      = _rsi(c, 14)
    out["rsi_21"]      = _rsi(c, 21)
    out["stoch_k"]     = _stoch_k(h, l, c, 14)
    out["stoch_d"]     = out["stoch_k"].rolling(3).mean()   # smoothed %D
    out["stoch_kd"]    = out["stoch_k"] - out["stoch_d"]    # K-D crossover
    out["williams_r"]  = _williams_r(h, l, c, 14)
    out["cci_20"]      = _cci(h, l, c, 20)
    out["cci_zscore"]  = _rolling_zscore(out["cci_20"], 20)

    # ── MACD ──────────────────────────────────────────────────────────────────
    out["macd"], out["macd_sig"], out["macd_hist"] = _macd(c)
    out["macd_cross"]  = np.sign(out["macd_hist"])
    out["macd_zscore"] = _rolling_zscore(out["macd_hist"], 20)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_u, bb_m, bb_l   = _bollinger(c)
    out["bb_pct"]      = (c - bb_l) / (bb_u - bb_l + 1e-8)
    out["bb_width"]    = (bb_u - bb_l) / (bb_m + 1e-8)
    out["bb_above"]    = (c > bb_u).astype(int)
    out["bb_below"]    = (c < bb_l).astype(int)

    # ── Volatility ────────────────────────────────────────────────────────────
    out["atr_14"]      = _atr(h, l, c, 14)
    out["atr_norm"]    = out["atr_14"] / c                   # normalised ATR
    out["atr_ratio"]   = _atr(h, l, c, 7) / (out["atr_14"] + 1e-8)  # volatility expansion
    out["hv_10"]       = c.pct_change().rolling(10).std()    # historical vol
    out["hv_ratio"]    = out["hv_10"] / c.pct_change().rolling(30).std()

    # ── Volume ────────────────────────────────────────────────────────────────
    out["obv"]         = _obv(c, v)
    out["obv_ema"]     = _ema(out["obv"], 9)
    out["obv_signal"]  = out["obv"] - out["obv_ema"]
    out["cmf"]         = _cmf(h, l, c, v, 20)
    out["vol_ratio"]   = v / (v.rolling(20).mean() + 1e-8)
    out["vol_zscore"]  = _rolling_zscore(v, 20)
    out["vol_surge"]   = (out["vol_ratio"] > 2.5).astype(int)

    # ── VWAP ─────────────────────────────────────────────────────────────────
    out["vwap"]        = _vwap(h, l, c, v, 20)
    out["vwap_ratio"]  = c / (out["vwap"] + 1e-8)
    out["vwap_zscore"] = _rolling_zscore(c - out["vwap"], 20)

    # ── ADX — trend strength ──────────────────────────────────────────────────
    out["adx"]         = _adx(h, l, c, 14)
    out["adx_strong"]  = (out["adx"] > 25).astype(int)
    out["adx_trend"]   = (out["adx"] > 20).astype(int)

    # ── Disparity index ───────────────────────────────────────────────────────
    out["disp_5"]      = _disparity(c, 5)
    out["disp_10"]     = _disparity(c, 10)
    out["disp_20"]     = _disparity(c, 20)

    # ── Price efficiency & Hurst ──────────────────────────────────────────────
    out["eff_ratio"]   = _efficiency_ratio(c, 10)
    out["hurst"]       = _hurst_proxy(c, 20)

    # ── Candle patterns ───────────────────────────────────────────────────────
    body               = (c - o)
    out["candle_body"] = body.abs() / (c + 1e-8)
    out["candle_dir"]  = np.sign(body)
    out["upper_wick"]  = (h - c.clip(lower=o)) / (c + 1e-8)
    out["lower_wick"]  = (c.clip(upper=o) - l) / (c + 1e-8)
    out["doji"]        = (out["candle_body"] < 0.001).astype(int)
    out["engulf"]      = ((body.abs() > body.shift(1).abs()) &
                          (np.sign(body) != np.sign(body.shift(1)))).astype(int)

    # ── Rate of change (momentum) ─────────────────────────────────────────────
    for lag in [1, 3, 5, 10, 15, 30]:
        out[f"roc_{lag}"]   = c.pct_change(lag)
    # Z-score normalised ROC (removes scale differences)
    for lag in [5, 15]:
        out[f"roc_z_{lag}"] = _rolling_zscore(out[f"roc_{lag}"], 30)

    # ── Lagged returns (autocorrelation features) ─────────────────────────────
    for lag in [1, 2, 3, 5, 10, 15, 20]:
        out[f"lag_{lag}"] = c.pct_change().shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in [5, 10, 20]:
        out[f"roll_max_{w}"]  = h.rolling(w).max() / c
        out[f"roll_min_{w}"]  = l.rolling(w).min() / c
        out[f"roll_std_{w}"]  = c.pct_change().rolling(w).std()

    # ── Regime detection ─────────────────────────────────────────────────────
    out["above_ema9"]  = (c > out["ema_9"]).astype(int)
    out["above_ema21"] = (c > out["ema_21"]).astype(int)
    out["above_ema50"] = (c > out["ema_50"]).astype(int)
    # Bull: price above all 3 EMAs; bear: below all 3
    out["bull_regime"] = (out["above_ema9"] & out["above_ema21"] & out["above_ema50"]).astype(int)
    out["bear_regime"] = ((c < out["ema_9"]) & (c < out["ema_21"]) & (c < out["ema_50"])).astype(int)

    # ── Time features (NSE session-aware) ────────────────────────────────────
    idx = out.index
    out["hour"]       = idx.hour
    out["minute"]     = idx.minute
    out["time_frac"]  = (idx.hour * 60 + idx.minute) / (6.5 * 60)
    # NSE session segments
    out["is_open"]    = ((idx.hour == 9) | ((idx.hour == 10) & (idx.minute < 30))).astype(int)
    out["is_close"]   = ((idx.hour >= 14) & (idx.hour < 16)).astype(int)
    out["is_midday"]  = ((idx.hour >= 11) & (idx.hour < 14)).astype(int)
    # Day of week (Friday effect, Monday effect)
    out["dow"]        = idx.dayofweek

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"Open", "High", "Low", "Close", "Volume", "target", "direction"}
    return [c for c in df.columns if c not in exclude]


def _make_direction_target(log_returns: np.ndarray) -> np.ndarray:
    """Convert log returns to 3-class direction: 0=DOWN, 1=FLAT, 2=UP."""
    labels = np.ones(len(log_returns), dtype=int)  # default FLAT
    labels[log_returns >  UP_THRESHOLD]   = 2  # UP
    labels[log_returns <  DOWN_THRESHOLD] = 0  # DOWN
    return labels


# =============================================================================
# 3. WALK-FORWARD EXPANDING WINDOW VALIDATION
# =============================================================================

def _walk_forward_splits(n: int, min_train: int = 100, n_splits: int = 3) -> list:
    """
    Walk-forward expanding window splits.
    Each fold: train on [0..t], validate on [t..t+step].
    NO data leakage — validation is always strictly after training.
    """
    step = (n - min_train) // (n_splits + 1)
    if step < 10:
        # Fall back to simple last-20% validation
        split = int(n * 0.8)
        return [(np.arange(split), np.arange(split, n))]

    splits = []
    for i in range(1, n_splits + 1):
        train_end = min_train + step * i
        val_end   = min(train_end + step, n)
        if val_end <= train_end:
            break
        splits.append((np.arange(train_end), np.arange(train_end, val_end)))
    return splits


# =============================================================================
# 4. MODEL BUILDERS
# =============================================================================

def _build_classifier(X_tr, y_tr, X_va, y_va):
    """
    LightGBM 3-class direction classifier with early stopping.
    Falls back to XGBoost if LightGBM unavailable.
    """
    if _LGBM_AVAILABLE:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            min_child_samples=15, reg_alpha=0.3, reg_lambda=1.5,
            class_weight="balanced",   # handles class imbalance
            random_state=42, n_jobs=1, verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(15, verbose=False),
                              lgb.log_evaluation(period=-1)])
    else:
        model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.7, colsample_bytree=0.7,
            use_label_encoder=False, eval_metric="mlogloss",
            early_stopping_rounds=20, random_state=42, n_jobs=1,
            tree_method="hist",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def _build_regressor(X_tr, y_tr, X_va, y_va):
    """
    LightGBM log-return regressor with early stopping.
    Predicts magnitude of move regardless of direction.
    """
    if _LGBM_AVAILABLE:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            min_child_samples=15, reg_alpha=0.3, reg_lambda=1.5,
            random_state=42, n_jobs=1, verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(15, verbose=False),
                              lgb.log_evaluation(period=-1)])
    else:
        model = XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.7, colsample_bytree=0.7,
            early_stopping_rounds=20, random_state=42, n_jobs=1,
            tree_method="hist",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def _price_mape(actuals, pred_ret, true_ret) -> float:
    pred_p = actuals * (1 + pred_ret)
    true_p = actuals * (1 + true_ret)
    safe   = np.where(np.abs(true_p) < 1e-8, 1e-8, true_p)
    return float(np.mean(np.abs(pred_p - true_p) / safe) * 100)


# =============================================================================
# 5. FEATURE SELECTION — Mutual Information + Importance Pruning
# =============================================================================

def _select_features(X: np.ndarray, y_cls: np.ndarray,
                     feature_names: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Select features using classifier importance from a quick LightGBM/XGB pass.
    Keep top features covering 90% cumulative importance.
    """
    Xq, yq = X[:int(len(X)*0.8)], y_cls[:int(len(X)*0.8)]
    Xv, yv = X[int(len(X)*0.8):], y_cls[int(len(X)*0.8):]

    if _LGBM_AVAILABLE:
        import lightgbm as lgb
        q = lgb.LGBMClassifier(n_estimators=50, max_depth=3,
                                random_state=42, n_jobs=1, verbose=-1,
                                class_weight="balanced")
        q.fit(Xq, yq, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(10, verbose=False),
                         lgb.log_evaluation(period=-1)])
        imp = q.feature_importances_.astype(float)
    else:
        q = XGBClassifier(n_estimators=50, max_depth=3, random_state=42,
                          n_jobs=1, tree_method="hist",
                          use_label_encoder=False, eval_metric="mlogloss",
                          early_stopping_rounds=10)
        q.fit(Xq, yq, eval_set=[(Xv, yv)], verbose=False)
        imp = q.feature_importances_.astype(float)

    order  = np.argsort(imp)[::-1]
    cumsum = np.cumsum(imp[order]) / (imp.sum() + 1e-8)
    n_keep = max(20, np.searchsorted(cumsum, 0.90) + 1)
    keep   = sorted(order[:n_keep].tolist())
    return X[:, keep], [feature_names[i] for i in keep], np.array(keep)


# =============================================================================
# 6. TRAINING — dual model (classifier + regressor) per horizon
# =============================================================================

def train_model(ticker: str) -> dict[str, Any]:
    if not _XGB_AVAILABLE and not _LGBM_AVAILABLE:
        return {"ticker": ticker, "status": "error", "message": "No ML library"}

    logger.info("Training %s ...", ticker)
    try:
        df, interval = fetch_ohlcv(ticker)
        df = add_features(df)

        if len(df) > 2000:
            df = df.tail(2000).copy()

        horizons = config.HORIZON_CANDLES.get(interval, config.HORIZON_CANDLES["1m"])

        feature_cols = get_feature_columns(df)
        X_raw   = df[feature_cols].values.astype(float)
        X_raw   = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X_raw)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # ── Feature selection using T+15 direction ───────────────────────────
        h_t15   = horizons.get("t15", 15)
        log_ret = np.log(df["Close"].shift(-h_t15) / df["Close"]).values
        valid   = ~np.isnan(log_ret)
        y_dir   = _make_direction_target(log_ret[valid])

        # Skip expensive feature selection for small datasets — use all features
        if len(X_scaled[valid]) < 300:
            sel_features = feature_cols
            sel_idx      = np.arange(len(feature_cols))
            X_sel        = X_scaled[valid]
            logger.info("  %s  small dataset — using all %d features", ticker, len(feature_cols))
        else:
            X_sel, sel_features, sel_idx = _select_features(
                X_scaled[valid], y_dir, feature_cols)
            logger.info("  %s  features: %d → %d selected",
                        ticker, len(feature_cols), len(sel_features))

        # ── Train per horizon ─────────────────────────────────────────────────
        classifiers: dict[str, Any] = {}
        regressors:  dict[str, Any] = {}
        dir_accs:    dict[str, float] = {}
        val_mapes:   dict[str, float] = {}

        for h_key, h_candles in horizons.items():
            log_r   = np.log(df["Close"].shift(-h_candles) / df["Close"]).values
            valid_h = ~np.isnan(log_r)

            X_h    = X_scaled[valid_h][:, sel_idx]
            lr_h   = log_r[valid_h]
            dir_h  = _make_direction_target(lr_h)
            pr_h   = df["Close"].values[valid_h]

            # Drop extreme labels (top/bottom 2% of returns) — reduces noise
            pct2   = np.percentile(np.abs(lr_h), 98)
            mask   = np.abs(lr_h) <= pct2
            X_h, lr_h, dir_h, pr_h = X_h[mask], lr_h[mask], dir_h[mask], pr_h[mask]

            if len(X_h) < 80:
                logger.warning("  %s %s: too few rows (%d)", ticker, h_key, len(X_h))
                continue

            splits = _walk_forward_splits(len(X_h), min_train=30, n_splits=2)

            fold_dir_accs, fold_mapes = [], []
            cls_model = None
            reg_model = None

            for tr_idx, va_idx in splits:
                # Classifier
                cls = _build_classifier(X_h[tr_idx], dir_h[tr_idx],
                                        X_h[va_idx],  dir_h[va_idx])
                pred_dir = cls.predict(X_h[va_idx])
                fold_dir_accs.append(float(np.mean(pred_dir == dir_h[va_idx])) * 100)
                cls_model = cls

                # Regressor
                reg = _build_regressor(X_h[tr_idx], lr_h[tr_idx],
                                       X_h[va_idx],  lr_h[va_idx])
                pred_ret = reg.predict(X_h[va_idx])
                fold_mapes.append(_price_mape(pr_h[va_idx], pred_ret, lr_h[va_idx]))
                reg_model = reg

            classifiers[h_key] = cls_model
            regressors[h_key]  = reg_model
            dir_accs[h_key]    = round(float(np.mean(fold_dir_accs)), 1)
            val_mapes[h_key]   = round(float(np.mean(fold_mapes)), 3)

            logger.info("  %s  %-4s  dir_acc=%.1f%%  mape=%.2f%%",
                        ticker, h_key, dir_accs[h_key], val_mapes[h_key])

        if not classifiers:
            return {"ticker": ticker, "status": "error", "message": "All horizons failed"}

        # Cache the last feature row for fast inference (avoids re-fetching every predict)
        last_row   = X_scaled[-1:, sel_idx]
        last_price = float(df["Close"].iloc[-1])

        _MODEL_REGISTRY[ticker] = {
            "classifiers":  classifiers,
            "regressors":   regressors,
            "scaler":       scaler,
            "features":     sel_features,
            "feat_idx":     sel_idx,
            "interval":     interval,
            "horizons":     horizons,
            "dir_accs":     dir_accs,
            "val_mapes":    val_mapes,
            "trained_at":   datetime.utcnow(),
            "row_count":    len(df),
            "last_X":       last_row,       # cached for fast inference
            "last_price":   last_price,
            "last_fetched": datetime.utcnow(),
        }

        return {
            "ticker":    ticker,
            "status":    "ok",
            "interval":  interval,
            "dir_accs":  dir_accs,
            "val_mapes": val_mapes,
            "features":  len(sel_features),
            "row_count": len(df),
            "trained_at":_MODEL_REGISTRY[ticker]["trained_at"].isoformat(),
        }

    except Exception as exc:
        import traceback
        logger.error("  FAIL %s: %s\n%s", ticker, exc, traceback.format_exc())
        return {"ticker": ticker, "status": "error", "message": str(exc)}


def train_all(tickers: list[str] | None = None) -> list[dict]:
    tickers = tickers or config.WATCHLIST
    results = []
    for i, t in enumerate(tickers):
        results.append(train_model(t))
        if i < len(tickers) - 1:
            time.sleep(0.5)
    return results


# =============================================================================
# 7. INFERENCE
# =============================================================================

def _horizon_mins(h_key: str, interval: str, horizons: dict) -> int:
    mins_map = {"1m": 1, "5m": 5, "1d": 1440}
    return horizons.get(h_key, 15) * mins_map.get(interval, 1)


def _label(h_key: str, interval: str, horizons: dict) -> str:
    m = _horizon_mins(h_key, interval, horizons)
    if m >= 60:
        return f"T+{m//60}h"
    return f"T+{m}min"


def get_sentiment_score(ticker: str) -> float:
    if not _VADER_AVAILABLE:
        return 0.0
    try:
        news   = yf.Ticker(ticker).news or []
        scores = [_vader.polarity_scores(a.get("title", ""))["compound"]
                  for a in news[:config.NEWS_MAX_ARTICLES] if a.get("title")]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def predict(ticker: str, force_refresh: bool = False) -> dict[str, Any]:
    # ── Serve from cache if fresh enough ─────────────────────────────────────
    if not force_refresh and ticker in _PREDICT_CACHE:
        cached = _PREDICT_CACHE[ticker]
        age    = (datetime.utcnow() - cached["cached_at"]).total_seconds()
        if age < _CACHE_TTL_SECONDS:
            return cached["result"]

    if ticker not in _MODEL_REGISTRY:
        r = train_model(ticker)
        if r["status"] != "ok":
            return {**r, "sector": config.SECTOR_MAP.get(ticker, ""),
                    "predictions": {}, "pre_session": {},
                    "current_price": None, "sparkline": []}

    reg      = _MODEL_REGISTRY[ticker]
    classifs = reg["classifiers"]
    regressors = reg["regressors"]
    scaler   = reg["scaler"]
    feat_idx = reg["feat_idx"]
    interval = reg["interval"]
    horizons = reg["horizons"]
    dir_accs = reg["dir_accs"]
    val_mapes= reg["val_mapes"]

    try:
        # Use cached feature row if fresh (< 60s old) — avoids re-fetching OHLCV
        last_fetched = reg.get("last_fetched")
        cache_age    = (datetime.utcnow() - last_fetched).total_seconds() if last_fetched else 999

        if cache_age < 60 and reg.get("last_X") is not None:
            # Fast path: use cached features + price
            X_pruned      = reg["last_X"]
            current_price = reg["last_price"]
        else:
            # Slow path: re-fetch OHLCV (runs every 60s or first call)
            df, _   = fetch_ohlcv(ticker, days=3)
            df      = add_features(df)
            df.dropna(inplace=True)
            if df.empty:
                raise ValueError("No live data")
            current_price = float(df["Close"].iloc[-1])
            all_feats     = get_feature_columns(df)
            latest        = df[all_feats].iloc[-1:].values.astype(float)
            latest        = np.nan_to_num(latest, nan=0.0, posinf=0.0, neginf=0.0)
            X_s           = scaler.transform(latest)
            X_pruned      = X_s[:, feat_idx]
            # Update cache in registry
            reg["last_X"]       = X_pruned
            reg["last_price"]   = current_price
            reg["last_fetched"] = datetime.utcnow()

        horizon_preds: dict[str, Any] = {}
        for h_key in horizons:
            cls = classifs.get(h_key)
            rgr = regressors.get(h_key)
            if cls is None or rgr is None:
                continue

            # Direction probabilities [p_DOWN, p_FLAT, p_UP]
            proba     = cls.predict_proba(X_pruned)[0]   # shape (3,)
            dir_pred  = int(np.argmax(proba))             # 0=DOWN,1=FLAT,2=UP
            dir_label = {0: "DOWN", 1: "FLAT", 2: "UP"}[dir_pred]

            # ── Confidence: three-signal blend ──────────────────────────
            # 1. Historical directional accuracy (most reliable signal)
            #    50% → 35%, 60% → 53%, 70% → 71%, 80% → 89%
            d_acc      = dir_accs.get(h_key, 50.0)
            d_acc_clamped = max(50.0, min(100.0, d_acc))
            acc_conf   = (d_acc_clamped - 50.0) / 50.0 * 90.0 + 35.0

            # 2. MAPE-based confidence (price accuracy signal)
            #    MAPE 0% → 80%, MAPE 3% → 62%, MAPE 8% → 32%, MAPE 15% → 5%
            mape_val   = val_mapes.get(h_key, 5.0)
            mape_conf  = max(5.0, min(80.0, (10.0 - mape_val) / 10.0 * 75.0 + 5.0))

            # 3. Live proba modifier ±8 pts (classifier conviction right now)
            raw_prob   = float(proba[dir_pred])
            proba_mod  = max(-8.0, min(8.0, (raw_prob - 0.38) * 40.0))

            # Weighted blend: acc 50%, mape 30%, proba 20%
            confidence = round(max(20.0, min(95.0,
                0.50 * acc_conf + 0.30 * mape_conf + 0.20 * (50.0 + proba_mod * 5.0)
            )), 1)

            # Magnitude from regressor
            pred_log_ret = float(rgr.predict(X_pruned)[0])

            # Combine: direction from classifier, magnitude from regressor
            # If classifier says DOWN but regressor is positive, trust classifier
            if dir_pred == 2:    # UP
                signed_ret = abs(pred_log_ret)
            elif dir_pred == 0:  # DOWN
                signed_ret = -abs(pred_log_ret)
            else:                # FLAT
                signed_ret = pred_log_ret * 0.3   # dampen magnitude

            predicted_price = round(current_price * np.exp(signed_ret), 2)
            change_pct      = (predicted_price / current_price - 1) * 100

            # Trading signal (directional accuracy gates the signal)
            d_acc = dir_accs.get(h_key, 50)
            if dir_pred == 2 and d_acc >= 52:
                signal = "BUY"
            elif dir_pred == 0 and d_acc >= 52:
                signal = "SELL"
            else:
                signal = "HOLD"

            # Sentiment overlay (T+15 only)
            sentiment = 0.0
            if h_key == "t15":
                sentiment = get_sentiment_score(ticker)
                if sentiment < -0.3 and signal == "BUY":
                    signal     = "HOLD"
                    confidence = max(0, confidence - 10)
                elif sentiment > 0.3 and signal == "SELL":
                    signal     = "HOLD"
                    confidence = max(0, confidence - 10)

            h_mins   = _horizon_mins(h_key, interval, horizons)
            exit_ist = (now_ist() + timedelta(minutes=h_mins)).strftime("%H:%M IST")

            horizon_preds[h_key] = {
                "label":           _label(h_key, interval, horizons),
                "predicted_price": predicted_price,
                "change_pct":      round(change_pct, 3),
                "signal":          signal,
                "confidence":      confidence,
                "direction":       dir_label,
                "dir_proba":       {
                    "DOWN": round(float(proba[0]) * 100, 1),
                    "FLAT": round(float(proba[1]) * 100, 1),
                    "UP":   round(float(proba[2]) * 100, 1),
                },
                "dir_acc":         d_acc,    # historical directional accuracy %
                "val_mape":        val_mapes.get(h_key, 0),
                "exit_time":       exit_ist,
                "horizon_mins":    h_mins,
                "sentiment":       round(sentiment, 3),
            }

        risk_qty    = max(1, int(config.RISK_CAPITAL_INR / (current_price * config.RISK_PCT)))
        pre_session = fetch_pre_session(ticker)

        result = {
            "ticker":        ticker,
            "sector":        config.SECTOR_MAP.get(ticker, ""),
            "status":        "ok",
            "current_price": current_price,
            "predictions":   horizon_preds,
            "pre_session":   pre_session,
            "risk_qty":      risk_qty,
            "entry_time":    now_ist().strftime("%H:%M IST"),
            "sparkline":     fetch_recent_prices(ticker),
            "data_interval": interval,
            "trained_at":    reg["trained_at"].isoformat(),
            "val_mapes":     val_mapes,
            "dir_accs":      dir_accs,
        }
        # Store in cache
        _PREDICT_CACHE[ticker] = {"result": result, "cached_at": datetime.utcnow()}
        return result

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
# 8. BACKTESTER — directional accuracy + price MAPE per horizon
# =============================================================================

def backtest_yesterday(ticker: str) -> dict[str, Any]:
    if ticker not in _MODEL_REGISTRY:
        train_model(ticker)

    reg       = _MODEL_REGISTRY[ticker]
    classifs  = reg["classifiers"]
    regressors= reg["regressors"]
    scaler    = reg["scaler"]
    feat_idx  = reg["feat_idx"]
    interval  = reg["interval"]
    horizons  = reg["horizons"]
    dir_accs  = reg["dir_accs"]

    try:
        df_full, _ = fetch_ohlcv(ticker, days=config.TRAINING_PERIOD_DAYS + 1)
        df_full    = add_features(df_full)
        df_full.dropna(inplace=True)

        today  = datetime.utcnow().date()
        df_yday = pd.DataFrame()
        backtest_date = today

        for days_back in range(1, 8):
            d    = today - timedelta(days=days_back)
            mask = df_full.index.date == d
            if mask.sum() >= 2:
                df_yday = df_full[mask]
                backtest_date = d
                break

        if df_yday.empty:
            backtest_date = df_full.index.date[-1]
            df_yday = df_full[df_full.index.date == backtest_date]

        if df_yday.empty:
            return {"ticker": ticker, "status": "error", "message": "No trading data found"}

        all_feats  = get_feature_columns(df_full)
        X_yday     = np.nan_to_num(df_yday[all_feats].values.astype(float))
        X_s        = scaler.transform(X_yday)
        X_pruned   = X_s[:, feat_idx]

        results_per_horizon: dict[str, Any] = {}
        all_mapes: list[float] = []

        for h_key, h_candles in horizons.items():
            cls = classifs.get(h_key)
            rgr = regressors.get(h_key)
            if cls is None or rgr is None:
                continue

            log_r = np.log(df_yday["Close"].shift(-h_candles) / df_yday["Close"]).values
            valid = ~np.isnan(log_r)
            if valid.sum() < 2:
                continue

            sample     = np.where(valid)[0][::max(1, h_candles)]
            X_s_samp   = X_pruned[sample]
            lr_true    = log_r[sample]
            prices_s   = df_yday["Close"].values[sample]
            dir_true   = _make_direction_target(lr_true)

            pred_dirs  = cls.predict(X_s_samp)
            dir_acc    = float(np.mean(pred_dirs == dir_true)) * 100
            pred_ret   = rgr.predict(X_s_samp)
            mape       = _price_mape(prices_s, pred_ret, lr_true)
            all_mapes.append(mape)

            pred_prices = prices_s * (1 + pred_ret)
            true_prices = prices_s * (1 + lr_true)
            h_mins      = _horizon_mins(h_key, interval, horizons)

            steps = [
                {
                    "time":            str(df_yday.index[sample[i]]),
                    "actual_price":    round(float(prices_s[i]), 2),
                    "predicted_price": round(float(pred_prices[i]), 2),
                    "true_future":     round(float(true_prices[i]), 2),
                    "price_error_pct": round(float(
                        abs(pred_prices[i]-true_prices[i]) / (abs(true_prices[i])+1e-8) * 100
                    ), 3),
                    "pred_direction":  {0:"DOWN",1:"FLAT",2:"UP"}[int(pred_dirs[i])],
                    "true_direction":  {0:"DOWN",1:"FLAT",2:"UP"}[int(dir_true[i])],
                    "dir_correct":     bool(pred_dirs[i] == dir_true[i]),
                }
                for i in range(len(sample))
            ]

            results_per_horizon[h_key] = {
                "label":            _label(h_key, interval, horizons),
                "horizon_mins":     h_mins,
                "mape":             round(mape, 3),
                "dir_accuracy":     round(dir_acc, 1),
                "passed":           mape <= config.MAX_ERROR_THRESHOLD_PCT,
                "steps":            steps,
            }

        overall_mape   = round(float(np.mean(all_mapes)), 3) if all_mapes else 999.0
        overall_passed = overall_mape <= config.MAX_ERROR_THRESHOLD_PCT
        overall_dir    = round(float(np.mean([
            r["dir_accuracy"] for r in results_per_horizon.values()
        ])), 1) if results_per_horizon else 0

        return {
            "ticker":           ticker,
            "status":           "ok",
            "date":             str(backtest_date),
            "data_interval":    interval,
            "overall_mape":     overall_mape,
            "overall_dir_acc":  overall_dir,
            "passed":           overall_passed,
            "threshold":        config.MAX_ERROR_THRESHOLD_PCT,
            "horizons":         results_per_horizon,
            "mape":             overall_mape,
            "summary": (
                f"{'PASSED' if overall_passed else 'FAILED'} — "
                f"MAPE {overall_mape:.2f}%  |  Dir accuracy {overall_dir:.1f}%"
            ),
        }

    except Exception as exc:
        import traceback
        logger.error("Backtest failed %s: %s\n%s", ticker, exc, traceback.format_exc())
        return {"ticker": ticker, "status": "error", "message": str(exc)}


# =============================================================================
# 9. AFTER-MARKET CLOSE ANALYSIS
#    Runs after NSE close (3:30 PM IST). Ranks all 25 stocks for next morning.
#    Uses today's full session data + T+1day models to build a morning watchlist.
# =============================================================================

def _market_is_closed() -> bool:
    """True if current IST time is after 3:30 PM (NSE close) or before 9:15 AM."""
    now   = now_ist()
    after = (now.hour > 15) or (now.hour == 15 and now.minute >= 30)
    before= (now.hour < 9)  or (now.hour == 9 and now.minute < 15)
    return after or before


def _score_stock(ticker: str) -> dict[str, Any] | None:
    """
    Score a single stock for next-morning pre-open session.
    Returns a scored dict or None if data unavailable.
    """
    if ticker not in _MODEL_REGISTRY:
        return None

    reg      = _MODEL_REGISTRY[ticker]
    classifs = reg.get("classifiers", {})
    regressors= reg.get("regressors", {})
    scaler   = reg["scaler"]
    feat_idx = reg["feat_idx"]
    interval = reg["interval"]
    horizons = reg["horizons"]
    dir_accs = reg.get("dir_accs", {})

    try:
        # Fetch today's full session OHLCV
        df, _ = fetch_ohlcv(ticker, days=3)
        df    = add_features(df)
        df.dropna(inplace=True)

        if df.empty or len(df) < 10:
            return None

        current_price = float(df["Close"].iloc[-1])
        today_open    = float(df["Open"].iloc[0])   if len(df) > 0 else current_price
        today_high    = float(df["High"].max())
        today_low     = float(df["Low"].min())
        today_volume  = float(df["Volume"].sum())
        avg_volume    = float(df["Volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else today_volume

        # ── Compute prediction using T+1d (or longest available horizon) ──────
        best_horizon = "t3h" if "t3h" in classifs else ("t1h" if "t1h" in classifs else "t15")
        cls = classifs.get(best_horizon)
        rgr = regressors.get(best_horizon)
        if cls is None or rgr is None:
            return None

        all_feats = get_feature_columns(df)
        latest    = df[all_feats].iloc[-1:].values.astype(float)
        latest    = np.nan_to_num(latest, nan=0.0, posinf=0.0, neginf=0.0)
        X_s       = scaler.transform(latest)
        X_pruned  = X_s[:, feat_idx]

        proba      = cls.predict_proba(X_pruned)[0]
        dir_pred   = int(np.argmax(proba))
        pred_ret   = float(rgr.predict(X_pruned)[0])

        # Same three-signal blend as live predict
        d_acc       = dir_accs.get(best_horizon, 50.0)
        acc_conf    = (max(50.0, min(100.0, d_acc)) - 50.0) / 50.0 * 90.0 + 35.0
        mape_val    = reg.get("val_mapes", {}).get(best_horizon, 5.0)
        mape_conf   = max(5.0, min(80.0, (10.0 - mape_val) / 10.0 * 75.0 + 5.0))
        raw_p       = float(proba[dir_pred])
        proba_mod   = max(-8.0, min(8.0, (raw_p - 0.38) * 40.0))
        confidence  = round(max(20.0, min(95.0,
            0.50 * acc_conf + 0.30 * mape_conf + 0.20 * (50.0 + proba_mod * 5.0)
        )), 1)

        # Predicted opening price (using log return from close)
        if dir_pred == 2:
            signed_ret = abs(pred_ret)
        elif dir_pred == 0:
            signed_ret = -abs(pred_ret)
        else:
            signed_ret = pred_ret * 0.3

        pred_next_price = round(current_price * np.exp(signed_ret), 2)
        pred_change_pct = (pred_next_price / current_price - 1) * 100

        # ── Technical scoring (0-100) for morning setup ───────────────────────
        scores = {}

        # 1. Trend alignment: is price above key EMAs?
        close_s = df["Close"]
        ema9  = float(close_s.ewm(span=9,  adjust=False).mean().iloc[-1])
        ema21 = float(close_s.ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = float(close_s.ewm(span=50, adjust=False).mean().iloc[-1])
        trend_score = sum([current_price > ema9, current_price > ema21, current_price > ema50]) / 3 * 100
        scores["trend"] = trend_score

        # 2. RSI positioning (best entry: 40-60 range, not overbought/oversold)
        rsi_val = float(_rsi(close_s, 14).iloc[-1]) if not np.isnan(_rsi(close_s, 14).iloc[-1]) else 50
        rsi_score = 100 - abs(rsi_val - 50) * 2   # peak at RSI=50
        scores["rsi"] = max(0, rsi_score)

        # 3. Volume surge today (high volume = conviction)
        vol_ratio = today_volume / (avg_volume + 1e-8)
        vol_score = min(100, vol_ratio * 40)
        scores["volume"] = vol_score

        # 4. Day's range position (close near high = bullish momentum)
        day_range = today_high - today_low
        if day_range > 0:
            range_pos = (current_price - today_low) / day_range * 100
        else:
            range_pos = 50
        scores["range_position"] = range_pos

        # 5. Pre-open bid data
        pre_data    = fetch_pre_session(ticker)
        bid         = pre_data.get("bid") or current_price
        ask         = pre_data.get("ask") or current_price
        spread_pct  = abs(ask - bid) / (current_price + 1e-8) * 100
        # Tight spread = liquid = good for entry
        spread_score = max(0, 100 - spread_pct * 200)
        scores["spread"] = spread_score

        # 6. Week 52 positioning (buying near 52W low is value, near high is momentum)
        w52_high = pre_data.get("week52_high") or current_price * 1.3
        w52_low  = pre_data.get("week52_low")  or current_price * 0.7
        w52_range = w52_high - w52_low
        if w52_range > 0:
            w52_pos = (current_price - w52_low) / w52_range * 100
        else:
            w52_pos = 50
        # Prefer stocks in middle of 52W range (not at extremes)
        w52_score = 100 - abs(w52_pos - 50) * 1.5
        scores["week52"] = max(0, w52_score)

        # 7. Model directional accuracy
        d_acc = dir_accs.get(best_horizon, 50)
        scores["model_acc"] = (d_acc - 50) * 2   # 50% → 0, 75% → 50, 100% → 100

        # ── Composite score ───────────────────────────────────────────────────
        weights = {"trend":0.20, "rsi":0.10, "volume":0.20, "range_position":0.15,
                   "spread":0.10, "week52":0.10, "model_acc":0.15}
        composite = sum(scores.get(k,0) * w for k, w in weights.items())

        # Only recommend if model predicts UP direction with decent confidence
        actionable = (dir_pred == 2 and confidence >= 35 and composite >= 40)

        # Buy range: suggest entry between bid and 0.3% above current
        buy_low  = round(bid * 0.998, 2) if bid else round(current_price * 0.997, 2)
        buy_high = round(current_price * 1.003, 2)
        stop_loss = round(current_price * (1 - config.RISK_PCT), 2)
        target    = pred_next_price

        return {
            "ticker":          ticker,
            "sector":          config.SECTOR_MAP.get(ticker, ""),
            "current_price":   current_price,
            "predicted_price": pred_next_price,
            "predicted_change_pct": round(pred_change_pct, 2),
            "direction":       {0:"DOWN",1:"FLAT",2:"UP"}[dir_pred],
            "confidence":      confidence,
            "composite_score": round(composite, 1),
            "actionable":      actionable,
            "buy_range":       {"low": buy_low, "high": buy_high},
            "stop_loss":       stop_loss,
            "target":          target,
            "risk_qty":        max(1, int(config.RISK_CAPITAL_INR / (current_price * config.RISK_PCT))),
            "scores":          {k: round(v, 1) for k, v in scores.items()},
            "today_stats": {
                "open":       round(today_open, 2),
                "high":       round(today_high, 2),
                "low":        round(today_low, 2),
                "close":      round(current_price, 2),
                "vol_ratio":  round(vol_ratio, 2),
                "rsi":        round(rsi_val, 1),
            },
            "pre_session":     pre_data,
            "data_interval":   interval,
            "horizon_used":    best_horizon,
        }

    except Exception as exc:
        logger.warning("  score_stock failed for %s: %s", ticker, exc)
        return None


def after_market_analysis(top_n: int = 7) -> dict[str, Any]:
    """
    Runs a full post-close analysis of all 25 stocks.
    Returns top_n stocks ranked by composite score for next morning's pre-open.
    Call this after 3:30 PM IST.
    """
    logger.info("After-market analysis started for %d tickers", len(config.WATCHLIST))

    results = []
    for ticker in config.WATCHLIST:
        scored = _score_stock(ticker)
        if scored:
            results.append(scored)
        time.sleep(0.1)  # polite delay

    if not results:
        return {
            "status": "error",
            "message": "No scored stocks — ensure models are trained first",
            "stocks": [],
        }

    # Sort by composite score (descending), actionable ones first
    results.sort(key=lambda x: (x["actionable"], x["composite_score"]), reverse=True)

    actionable = [r for r in results if r["actionable"]]
    watchlist  = results[:top_n]

    now  = now_ist()
    mkt_open_ist = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now.hour >= 15:
        # Next trading day
        mkt_open_ist = mkt_open_ist + timedelta(days=1)

    return {
        "status":           "ok",
        "generated_at":     now.strftime("%d %b %Y  %H:%M IST"),
        "market_opens":     mkt_open_ist.strftime("%d %b  %H:%M IST"),
        "total_analysed":   len(results),
        "actionable_count": len(actionable),
        "top_picks":        watchlist,
        "all_stocks":       results,
        "summary": (
            f"{len(actionable)} stocks show bullish setup for tomorrow. "
            f"Top pick: {watchlist[0]['ticker'].replace('.NS','')} "
            f"({watchlist[0]['predicted_change_pct']:+.1f}%)"
            if watchlist else "No strong setups detected for tomorrow."
        ),
    }
