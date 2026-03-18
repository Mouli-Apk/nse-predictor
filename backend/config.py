# =============================================================================
# NSE Intraday Predictor — Configuration
# Update WATCHLIST here to change tracked stocks. No other file needs editing.
# =============================================================================

WATCHLIST: list[str] = [
    # ── Defense / Industrial ──────────────────────────────────────────────────
    "MAZDOCK.NS",
    "BEL.NS",
    "BHEL.NS",
    "BHARATFORG.NS",
    "SOLARINDS.NS",
    "COCHINSHIP.NS",
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
    "PBFINTECH.NS",
    # ── Consumer / Growth ─────────────────────────────────────────────────────
    "TRENT.NS",
    "DIXON.NS",
    "MAXHEALTH.NS",
    "IHCL.NS",
]

SECTOR_MAP: dict[str, str] = {
    "MAZDOCK.NS":   "Defense",
    "BEL.NS":       "Defense",
    "BHEL.NS":      "Industrial",
    "BHARATFORG.NS":"Industrial",
    "SOLARINDS.NS": "Defense",
    "COCHINSHIP.NS":"Defense",
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
    "PBFINTECH.NS": "Fintech",
    "TRENT.NS":     "Consumer",
    "DIXON.NS":     "Consumer",
    "MAXHEALTH.NS": "Healthcare",
    "IHCL.NS":      "Consumer",
}

# ── Model Hyperparameters ──────────────────────────────────────────────────────
PREDICTION_HORIZON_MINS: int   = 15          # T+15 min forecast
TRAINING_PERIOD_DAYS:    int   = 7           # rolling window
CANDLE_INTERVAL:         str   = "1m"        # 1-minute OHLCV
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
XGB_PARAMS: dict = {
    "n_estimators":    400,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "random_state":    42,
    "n_jobs":          -1,
}

# ── News Sentiment ─────────────────────────────────────────────────────────────
NEWS_QUERY_TEMPLATE: str = "{ticker} NSE stock"
NEWS_MAX_ARTICLES:   int = 5
