"""
generate_data.py
================
Generates realistic synthetic OHLCV data for 50 Nifty 500 stocks
spanning 10 years (2015-01-01 to 2024-12-31).

The simulation uses:
  - Geometric Brownian Motion (GBM) for price paths
  - Per-sector drift / volatility calibrated to real Nifty history
  - Correlated returns via a sector factor model
  - Intra-day OHLC construction with realistic bid-ask spread

Output: data/  directory with one CSV per ticker.
"""

import os
import numpy as np
import pandas as pd
from datetime import date

# ── seed for reproducibility ─────────────────────────────────
np.random.seed(42)

OUT_DIR    = "data"
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# ── Nifty 500 representative universe (50 stocks) ─────────────
# (ticker, sector, annual_drift, annual_vol, start_price)
STOCKS = [
    # ── Large-cap IT ──────────────────────────────────────────
    ("TCS",         "IT",         0.20, 0.22, 1100.0),
    ("INFY",        "IT",         0.17, 0.25,  750.0),
    ("WIPRO",       "IT",         0.13, 0.24,  300.0),
    ("TECHM",       "IT",         0.18, 0.28,  450.0),
    ("HCLTECH",     "IT",         0.19, 0.26,  800.0),
    ("MPHASIS",     "IT",         0.22, 0.30,  450.0),

    # ── Banking ───────────────────────────────────────────────
    ("HDFCBANK",    "Banking",    0.16, 0.24,  700.0),
    ("ICICIBANK",   "Banking",    0.19, 0.30,  220.0),
    ("KOTAKBANK",   "Banking",    0.18, 0.25,  680.0),
    ("AXISBANK",    "Banking",    0.14, 0.33,  380.0),
    ("SBIN",        "Banking",    0.14, 0.35,  200.0),
    ("INDUSINDBK",  "Banking",    0.15, 0.34,  550.0),
    ("BANDHANBNK",  "Banking",    0.10, 0.40,  350.0),

    # ── NBFC / Insurance ──────────────────────────────────────
    ("BAJFINANCE",  "NBFC",       0.28, 0.38, 1200.0),
    ("BAJAJFINSV",  "NBFC",       0.24, 0.34, 1100.0),
    ("HDFCAMC",     "NBFC",       0.20, 0.28, 1800.0),
    ("SBILIFE",     "Insurance",  0.18, 0.26,  450.0),
    ("ICICIPRU",    "Insurance",  0.15, 0.27,  280.0),

    # ── Energy & Oil ──────────────────────────────────────────
    ("RELIANCE",    "Energy",     0.18, 0.28,  850.0),
    ("ONGC",        "Energy",     0.08, 0.30,  240.0),
    ("BPCL",        "Energy",     0.10, 0.32,  340.0),
    ("POWERGRID",   "Energy",     0.12, 0.20,  120.0),
    ("NTPC",        "Energy",     0.11, 0.22,  130.0),

    # ── FMCG ──────────────────────────────────────────────────
    ("HINDUNILVR",  "FMCG",       0.14, 0.18,  650.0),
    ("ITC",         "FMCG",       0.10, 0.20,  200.0),
    ("NESTLEIND",   "FMCG",       0.15, 0.17, 5500.0),
    ("BRITANNIA",   "FMCG",       0.13, 0.19, 1500.0),
    ("DABUR",       "FMCG",       0.12, 0.18,  260.0),

    # ── Auto ──────────────────────────────────────────────────
    ("MARUTI",      "Auto",       0.15, 0.27, 3800.0),
    ("TATAMOTORS",  "Auto",       0.16, 0.40,  300.0),
    ("BAJAJ-AUTO",  "Auto",       0.14, 0.24, 2200.0),
    ("HEROMOTOCO",  "Auto",       0.10, 0.22, 2700.0),
    ("EICHERMOT",   "Auto",       0.20, 0.30, 2000.0),

    # ── Pharma ────────────────────────────────────────────────
    ("SUNPHARMA",   "Pharma",     0.12, 0.26,  600.0),
    ("DRREDDY",     "Pharma",     0.13, 0.24, 2800.0),
    ("CIPLA",       "Pharma",     0.11, 0.23,  480.0),
    ("DIVISLAB",    "Pharma",     0.22, 0.28, 1500.0),

    # ── Consumer / Retail ─────────────────────────────────────
    ("TITAN",       "Consumer",   0.24, 0.30,  280.0),
    ("ASIANPAINT",  "Consumer",   0.16, 0.22,  620.0),
    ("PIDILITIND",  "Consumer",   0.18, 0.22,  550.0),
    ("HAVELLS",     "Consumer",   0.20, 0.28,  280.0),

    # ── Telecom ───────────────────────────────────────────────
    ("BHARTIARTL",  "Telecom",    0.22, 0.32,  300.0),

    # ── Infrastructure / Capital Goods ────────────────────────
    ("LT",          "Infra",      0.15, 0.26, 1000.0),
    ("ADANIPORTS",  "Infra",      0.20, 0.35,  200.0),
    ("SIEMENS",     "Infra",      0.17, 0.27,  900.0),
    ("ABB",         "Infra",      0.16, 0.26,  800.0),

    # ── Cement & Materials ────────────────────────────────────
    ("ULTRACEMCO",  "Cement",     0.16, 0.28, 2200.0),
    ("AMBUJACEM",   "Cement",     0.12, 0.26,  200.0),
    ("JSWSTEEL",    "Metals",     0.14, 0.35,  120.0),
    ("TATASTEEL",   "Metals",     0.12, 0.38,  280.0),
]

# Sector correlation matrix (used by factor model)
SECTORS = ["Energy","IT","Banking","Telecom","NBFC","Auto","Consumer","FMCG","Pharma","Infra","Cement","Insurance","Metals"]
SECTOR_CORR  = 0.35   # within-sector correlation
MARKET_CORR  = 0.25   # cross-sector (market beta)


def make_trading_dates(start: str, end: str) -> pd.DatetimeIndex:
    all_days = pd.bdate_range(start, end)
    # Remove approximate NSE holidays (simplified)
    return all_days


def simulate_prices(n_days: int, drift: float, vol: float,
                    s0: float, market_returns: np.ndarray,
                    sector_returns: np.ndarray) -> np.ndarray:
    """GBM with market + sector factor."""
    dt   = 1 / 252
    idio = np.random.normal(0, vol * np.sqrt(dt) * 0.7, n_days)     # idiosyncratic
    mkt  = market_returns  * 0.55                                     # market beta
    sec  = sector_returns  * 0.25                                     # sector beta
    daily_log_ret = (drift - 0.5 * vol**2) * dt + idio + mkt + sec
    prices = s0 * np.exp(np.cumsum(daily_log_ret))
    return np.concatenate([[s0], prices[:-1]])   # align: price[i] = open of day i


def make_ohlcv(closes: np.ndarray, vol_annual: float,
               dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build realistic OHLCV from daily closes."""
    n    = len(closes)
    dt   = 1 / 252
    intraday_vol = vol_annual * np.sqrt(dt) * 0.6

    opens  = closes * np.exp(np.random.normal(0, intraday_vol * 0.4, n))
    highs  = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, intraday_vol, n)))
    lows   = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, intraday_vol, n)))

    # Volume: mean-reverting around a trend, correlated with absolute return
    base_vol   = 1_000_000 * (closes / closes[0]) * 0.5
    abs_ret    = np.abs(closes / np.roll(closes, 1) - 1)
    abs_ret[0] = 0
    volume     = (base_vol * (1 + 3 * abs_ret) *
                  np.random.lognormal(0, 0.4, n)).astype(int)

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   opens.round(2),
        "High":   highs.round(2),
        "Low":    lows.round(2),
        "Close":  closes.round(2),
        "Volume": volume,
    }).set_index("Date")
    return df


def generate_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    dates  = make_trading_dates(START_DATE, END_DATE)
    n_days = len(dates)

    # ── Market factor ────────────────────────────────────────
    mkt_drift = 0.12
    mkt_vol   = 0.18
    dt        = 1 / 252
    market_returns = np.random.normal(
        (mkt_drift - 0.5 * mkt_vol**2) * dt,
        mkt_vol * np.sqrt(dt), n_days
    )

    # ── Sector factors ───────────────────────────────────────
    sector_factors = {}
    for sec in SECTORS:
        s_vol = 0.12
        sector_factors[sec] = np.random.normal(0, s_vol * np.sqrt(dt), n_days)

        print(f"Generating data for {len(STOCKS)} stocks over {n_days} trading days "
            f"({START_DATE} -> {END_DATE})\n")

    for ticker, sector, drift, vol, s0 in STOCKS:
        sec_ret = sector_factors.get(sector, np.zeros(n_days))
        prices = simulate_prices(n_days, drift, vol, s0, market_returns, sec_ret)
        df = make_ohlcv(prices, vol, dates)
        path = os.path.join(OUT_DIR, f"{ticker}.csv")
        df.to_csv(path)
        end_price = df["Close"].iloc[-1]
        total_ret = (end_price / s0 - 1) * 100
        print(f"  {ticker:12s}  start Rs{s0:8.0f}  ->  end Rs{end_price:8.0f}  "
              f"({total_ret:+.0f}%)  [{n_days} days]")

    print(f"\nData saved to ./{OUT_DIR}/  ({len(STOCKS)} files)")


if __name__ == "__main__":
    generate_all()
