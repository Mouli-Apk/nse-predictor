# =============================================================================
# NSE Intraday Predictor — Configuration
# Update WATCHLIST here to change tracked stocks. No other file needs editing.
# =============================================================================

WATCHLIST: list[str] = [
    # ── Defense / Industrial ──────────────────────────────────────────────────
    "BEL.NS",
    "BHEL.NS",
    "BHARATFORG.NS",
    "SOLARINDS.NS",
    "HAL.NS",        # replaced MAZDOCK.NS — Hindustan Aeronautics, highly liquid
    "RELIANCE.NS",   # replaced COCHINSHIP.NS — most liquid NSE stock
    # ── Green Energy ──────────────────────────────────────────────────────────
    "WAAREEENER.NS",
    "TATAPOWER.NS",
    "SUZLON.NS",
    "IREDA.NS",
    "NHPC.NS",
    "JSWENERGY.NS",
    "KPIGREEN.NS",
    # ── Finance / Fintech ─────────────────────────────────────────────────────
    "MUTHOOTFIN.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "PFC.NS",
    "RECLTD.NS",
    "ABCAPITAL.NS",
    "ANGELONE.NS",  # replaced PBFINTECH.NS
    # ── Consumer / Growth ─────────────────────────────────────────────────────
    "TRENT.NS",
    "DIXON.NS",
    "MAXHEALTH.NS",
    "TITAN.NS",     # replaced IHCL.NS
]

SECTOR_MAP: dict[str, str] = {
    "HAL.NS":       "Defense",
    "RELIANCE.NS":  "Industrial",
    "BEL.NS":       "Defense",
    "BHEL.NS":      "Industrial",
    "BHARATFORG.NS":"Industrial",
    "SOLARINDS.NS": "Defense",
    "WAAREEENER.NS":"Green Energy",
    "TATAPOWER.NS": "Green Energy",
    "SUZLON.NS":    "Green Energy",
    "IREDA.NS":     "Green Energy",
    "NHPC.NS":      "Green Energy",
    "JSWENERGY.NS": "Green Energy",
    "KPIGREEN.NS":  "Green Energy",
    "MUTHOOTFIN.NS":"Finance",
    "HDFCBANK.NS":  "Finance",
    "ICICIBANK.NS": "Finance",
    "SBIN.NS":      "Finance",
    "PFC.NS":       "Finance",
    "RECLTD.NS":    "Finance",
    "ABCAPITAL.NS": "Fintech",
    "ANGELONE.NS":  "Fintech",
    "TRENT.NS":     "Consumer",
    "DIXON.NS":     "Consumer",
    "MAXHEALTH.NS": "Healthcare",
    "TITAN.NS":     "Consumer",
}

# ── Model Hyperparameters ──────────────────────────────────────────────────────
# Multi-horizon predictions — one XGBoost model trained per horizon per ticker
PREDICTION_HORIZONS: dict = {
    "t15":  {"label": "T+15 min", "mins": 15},
    "t1h":  {"label": "T+1 Hour", "mins": 60},
    "t3h":  {"label": "T+3 Hours","mins": 180},
}
# Candle shift per interval for each horizon
# {data_interval: {horizon_key: candle_shift}}
HORIZON_CANDLES: dict = {
    "1m": {"t15": 15,  "t1h": 60,  "t3h": 180},
    "5m": {"t15": 3,   "t1h": 12,  "t3h": 36},
    "1d": {"t15": 1,   "t1h": 1,   "t3h": 2},
}
TRAINING_PERIOD_DAYS:    int   = 7           # rolling window
CANDLE_INTERVAL:         str   = "1m"        # preferred interval
MAX_ERROR_THRESHOLD_PCT: float = 3.0         # accuracy gate (%)

# ── Risk Engine ────────────────────────────────────────────────────────────────
RISK_CAPITAL_INR: float = 5_000.0            # max capital per trade
RISK_PCT:         float = 0.015              # 1.5 % stop-loss

# ── Technical Indicator Periods ───────────────────────────────────────────────
RSI_PERIOD:   int = 14
EMA_PERIODS:  list[int] = [9, 21, 50]
MACD_FAST:    int = 12
MACD_SLOW:    int = 26
MACD_SIGNAL:  int = 9
BB_PERIOD:    int = 20
BB_STD:       float = 2.0

# ── XGBoost Defaults ──────────────────────────────────────────────────────────
# XGBoost — memory-efficient settings for 512MB Render free tier
XGB_PARAMS: dict = {
    "n_estimators":    150,
    "max_depth":       4,
    "learning_rate":   0.08,
    "subsample":       0.75,
    "colsample_bytree":0.75,
    "min_child_weight":3,
    "reg_alpha":       0.2,
    "reg_lambda":      1.5,
    "random_state":    42,
    "n_jobs":          1,       # single thread — prevents OOM on free tier
    "tree_method":     "hist",  # memory-efficient
    "early_stopping_rounds": 20,
}

# LightGBM params (used if lgbm installed — much faster, less memory)
LGBM_PARAMS: dict = {
    "n_estimators":    200,
    "max_depth":       5,
    "learning_rate":   0.08,
    "subsample":       0.75,
    "colsample_bytree":0.75,
    "min_child_samples":10,
    "reg_alpha":       0.2,
    "reg_lambda":      1.5,
    "random_state":    42,
    "n_jobs":          1,
    "verbose":        -1,
}

# ── News Sentiment ─────────────────────────────────────────────────────────────
NEWS_QUERY_TEMPLATE: str = "{ticker} NSE stock"
NEWS_MAX_ARTICLES:   int = 5
